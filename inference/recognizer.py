"""
人脸识别推理封装
"""

import os
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path

import numpy as np
import cv2
import torch
import torch.nn.functional as F

from models.recognition import DDFD_Rec, build_recognizer


class Recognizer:
    """
    人脸识别器推理封装
    
    支持:
    - PyTorch 模型推理
    - ONNX Runtime 推理
    - TensorRT 推理
    """
    
    def __init__(
        self,
        checkpoint: Optional[str] = None,
        device: Optional[str] = None,
        model_type: str = "ddfd_rec",
        use_onnx: bool = False,
        use_tensorrt: bool = False,
        trt_engine_path: Optional[str] = None,
    ):
        """
        Args:
            checkpoint: 模型权重路径
            device: 计算设备
            model_type: 模型类型
            use_onnx: 是否使用 ONNX Runtime
            use_tensorrt: 是否使用 TensorRT
            trt_engine_path: TensorRT 引擎路径
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 输入配置
        self.input_size = 112
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        
        # 初始化推理引擎
        if use_tensorrt and trt_engine_path:
            self._init_tensorrt(trt_engine_path)
            self.mode = "tensorrt"
        elif use_onnx:
            self._init_onnx(checkpoint)
            self.mode = "onnx"
        else:
            self._init_pytorch(checkpoint, model_type)
            self.mode = "pytorch"
    
    def _init_pytorch(
        self,
        checkpoint: Optional[str],
        model_type: str
    ) -> None:
        """初始化 PyTorch 模型"""
        self.model = build_recognizer(model_type=model_type)
        
        if checkpoint and os.path.exists(checkpoint):
            state_dict = torch.load(checkpoint, map_location=self.device)
            if 'model_state_dict' in state_dict:
                state_dict = state_dict['model_state_dict']
            self.model.load_state_dict(state_dict, strict=False)
        
        self.model.to(self.device)
        self.model.eval()
    
    def _init_onnx(self, onnx_path: str) -> None:
        """初始化 ONNX Runtime"""
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError("onnxruntime is required for ONNX inference")
        
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.session = ort.InferenceSession(onnx_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
    
    def _init_tensorrt(self, engine_path: str) -> None:
        """初始化 TensorRT"""
        try:
            import tensorrt as trt
        except ImportError:
            raise ImportError("tensorrt is required for TensorRT inference")
        
        logger = trt.Logger(trt.Logger.INFO)
        trt.init_libnvinfer_plugins(logger, '')
        runtime = trt.Runtime(logger)
        
        with open(engine_path, 'rb') as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        
        self.engine = engine
        self.context = engine.create_execution_context()
    
    def preprocess(
        self,
        image: np.ndarray,
        bbox: np.ndarray,
        landmarks: Optional[np.ndarray] = None
    ) -> Tuple[torch.Tensor, Dict]:
        """
        人脸对齐和预处理
        
        Args:
            image: (H, W, C) BGR 图像
            bbox: (4,) [x1, y1, x2, y2] 人脸框
            landmarks: (5, 2) 关键点
            
        Returns:
            tensor: 预处理后的张量
            meta: 元数据
        """
        # 如果没有关键点，使用 bbox 中心
        if landmarks is None:
            x1, y1, x2, y2 = bbox
            landmarks = np.array([
                [(x1 + x2) / 2, (y1 + y2 * 0.3)],  # 鼻子
                [x1 + (x2 - x1) * 0.3, y1 + (y2 - y1) * 0.4],  # 左眼
                [x1 + (x2 - x1) * 0.7, y1 + (y2 - y1) * 0.4],  # 右眼
                [x1 + (x2 - x1) * 0.3, y1 + (y2 - y1) * 0.7],  # 左嘴角
                [x1 + (x2 - x1) * 0.7, y1 + (y2 - y1) * 0.7],  # 右嘴角
            ])
        
        # 人脸对齐
        face_aligned = self._align_face(image, landmarks)
        
        # 归一化
        face_norm = face_aligned.astype(np.float32) / 255.0
        face_norm = (face_norm - self.mean) / self.std
        
        # HWC -> CHW
        face_tensor = torch.from_numpy(face_norm.transpose(2, 0, 1))
        face_tensor = face_tensor.unsqueeze(0)  # (1, 3, H, W)
        
        meta = {
            'bbox': bbox,
            'landmarks': landmarks,
        }
        
        return face_tensor, meta
    
    def _align_face(
        self,
        image: np.ndarray,
        landmarks: np.ndarray,
        output_size: int = 112,
        scale_factor: float = 1.0
    ) -> np.ndarray:
        """
        人脸对齐
        
        Args:
            image: 原始图像
            landmarks: (5, 2) 关键点
            output_size: 输出尺寸
            scale_factor: 缩放因子
            
        Returns:
            aligned_face: 对齐后的人脸
        """
        # 标准关键点位置
        dst_landmarks = np.array([
            [56.0, 56.0],      # 鼻子
            [38.0, 42.0],      # 左眼
            [74.0, 42.0],      # 右眼
            [40.0, 78.0],      # 左嘴角
            [72.0, 78.0],      # 右嘴角
        ], dtype=np.float32)
        
        # 使用眼睛和嘴巴计算仿射变换
        # 简化处理：使用全部 5 个点
        src_landmarks = landmarks.astype(np.float32)
        
        # 计算仿射变换矩阵
        M = cv2.estimateAffinePartial2D(src_landmarks, dst_landmarks)[0]
        
        # 应用变换
        aligned = cv2.warpAffine(
            image,
            M,
            (output_size, output_size),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE
        )
        
        return aligned
    
    def postprocess(
        self,
        feature: torch.Tensor
    ) -> np.ndarray:
        """
        后处理
        
        Args:
            feature: (1, D) 特征张量
            
        Returns:
            feature_np: (D,) 归一化特征向量
        """
        # L2 归一化
        feature_norm = F.normalize(feature, p=2, dim=1)
        
        # 转为 numpy
        feature_np = feature_norm.cpu().numpy()[0]
        
        return feature_np
    
    @torch.no_grad()
    def extract(
        self,
        image: np.ndarray,
        bbox: np.ndarray,
        landmarks: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        提取人脸特征
        
        Args:
            image: (H, W, C) BGR 图像
            bbox: (4,) [x1, y1, x2, y2] 人脸框
            landmarks: (5, 2) 关键点 (可选)
            
        Returns:
            feature: (D,) 归一化特征向量
        """
        # 预处理
        tensor, _ = self.preprocess(image, bbox, landmarks)
        tensor = tensor.to(self.device)
        
        # 推理
        if self.mode == "pytorch":
            feature = self.model.get_identity_feature(tensor)
        elif self.mode == "onnx":
            outputs = self.session.run(None, {self.input_name: tensor.cpu().numpy()})
            feature = torch.from_numpy(outputs[0])
        else:
            # TensorRT 推理
            pass
        
        # 后处理
        feature_np = self.postprocess(feature)
        
        return feature_np
    
    def extract_batch(
        self,
        images: List[np.ndarray],
        bboxes: List[np.ndarray],
        landmarks: Optional[List[np.ndarray]] = None
    ) -> List[np.ndarray]:
        """
        批量特征提取
        
        Args:
            images: 图像列表
            bboxes: 人脸框列表
            landmarks: 关键点列表
            
        Returns:
            features: 特征向量列表
        """
        features = []
        for i, (image, bbox) in enumerate(zip(images, bboxes)):
            lm = landmarks[i] if landmarks else None
            feature = self.extract(image, bbox, lm)
            features.append(feature)
        return features
    
    def verify(
        self,
        image1: np.ndarray,
        bbox1: np.ndarray,
        image2: np.ndarray,
        bbox2: np.ndarray,
        landmarks1: Optional[np.ndarray] = None,
        landmarks2: Optional[np.ndarray] = None
    ) -> Tuple[bool, float]:
        """
        人脸验证 (1:1)
        
        Args:
            image1: 图像 1
            bbox1: 人脸框 1
            image2: 图像 2
            bbox2: 人脸框 2
            landmarks1: 关键点 1
            landmarks2: 关键点 2
            
        Returns:
            is_same: 是否同一人
            similarity: 相似度
        """
        # 提取特征
        feat1 = self.extract(image1, bbox1, landmarks1)
        feat2 = self.extract(image2, bbox2, landmarks2)
        
        # 计算相似度
        similarity = np.dot(feat1, feat2)
        
        # 阈值判断
        threshold = 0.6
        is_same = similarity >= threshold
        
        return is_same, float(similarity)


# ============================================
# 人脸对齐工具
# ============================================

class FaceAligner:
    """
    人脸对齐工具
    """
    
    def __init__(
        self,
        output_size: int = 112,
        scale_factor: float = 1.0
    ):
        self.output_size = output_size
        self.scale_factor = scale_factor
        
        # 标准 landmarks
        self.dst_landmarks = np.array([
            [56.0, 56.0],
            [38.0, 42.0],
            [74.0, 42.0],
            [40.0, 78.0],
            [72.0, 78.0],
        ], dtype=np.float32)
    
    def align(
        self,
        image: np.ndarray,
        landmarks: np.ndarray
    ) -> np.ndarray:
        """
        人脸对齐
        
        Args:
            image: 原始图像
            landmarks: (5, 2) 关键点
            
        Returns:
            aligned: 对齐后的人脸
        """
        M = cv2.estimateAffinePartial2D(
            landmarks.astype(np.float32),
            self.dst_landmarks
        )[0]
        
        aligned = cv2.warpAffine(
            image,
            M,
            (self.output_size, self.output_size),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE
        )
        
        return aligned
    
    def align_batch(
        self,
        image: np.ndarray,
        landmarks_list: List[np.ndarray]
    ) -> List[np.ndarray]:
        """批量对齐"""
        aligned_faces = []
        for landmarks in landmarks_list:
            aligned = self.align(image, landmarks)
            aligned_faces.append(aligned)
        return aligned_faces


if __name__ == "__main__":
    # 测试
    print("Testing Recognizer...")
    
    recognizer = Recognizer()
    
    # 创建测试图像和 bbox
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    test_bbox = np.array([100, 100, 200, 250])
    
    # 提取特征
    feature = recognizer.extract(test_image, test_bbox)
    print(f"Feature shape: {feature.shape}")
    print(f"Feature norm: {np.linalg.norm(feature):.4f}")
    
    # 验证
    test_bbox2 = np.array([300, 100, 400, 250])
    is_same, sim = recognizer.verify(test_image, test_bbox, test_image, test_bbox2)
    print(f"Verification: is_same={is_same}, similarity={sim:.4f}")
