"""
人脸检测推理封装

修复版本:
- 集成修复后的 bbox 解码
- 集成修复后的 NMS
- 置信度校准
"""

import os
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path

import numpy as np
import cv2
import torch
import torch.nn.functional as F

from models.detection import DKGA_Det, build_detector
from models.detection.post_process import decode_bbox_fixed, nms_fixed, clip_boxes_to_image


class Detector:
    """
    人脸检测器推理封装
    
    支持:
    - PyTorch 模型推理
    - ONNX Runtime 推理
    - TensorRT 推理
    """
    
    def __init__(
        self,
        checkpoint: Optional[str] = None,
        device: Optional[str] = None,
        model_type: str = "dkga_det",
        score_thresh: float = 0.6,
        nms_thresh: float = 0.45,
        max_faces: int = 300,
        use_onnx: bool = False,
        use_tensorrt: bool = False,
        trt_engine_path: Optional[str] = None,
    ):
        """
        Args:
            checkpoint: 模型权重路径
            device: 计算设备
            model_type: 模型类型
            score_thresh: 置信度阈值
            nms_thresh: NMS 阈值
            max_faces: 最大检测数量
            use_onnx: 是否使用 ONNX Runtime
            use_tensorrt: 是否使用 TensorRT
            trt_engine_path: TensorRT 引擎路径
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.max_faces = max_faces
        
        # 输入配置
        self.input_size = 640
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
        self.model = build_detector(model_name=model_type)
        
        if checkpoint and os.path.exists(checkpoint):
            state_dict = torch.load(checkpoint, map_location=self.device)
            if 'model_state_dict' in state_dict:
                state_dict = state_dict['model_state_dict']
            self.model.load_state_dict(state_dict)
        
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
        
        # 创建 logger 和 runtime
        logger = trt.Logger(trt.Logger.INFO)
        trt.init_libnvinfer_plugins(logger, '')
        runtime = trt.Runtime(logger)
        
        # 加载引擎
        with open(engine_path, 'rb') as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        
        self.engine = engine
        self.context = engine.create_execution_context()
        
        # 获取输入输出绑定
        self.input_binding = engine.get_binding_index('input')
        self.output_bindings = [
            engine.get_binding_index(name)
            for name in ['cls_output', 'reg_output', 'kpt_output']
        ]
    
    def preprocess(self, image: np.ndarray) -> Tuple[torch.Tensor, Dict]:
        """
        图像预处理
        
        Args:
            image: (H, W, C) BGR 图像
            
        Returns:
            tensor: 预处理后的张量
            meta: 元数据
        """
        H, W = image.shape[:2]
        
        # 计算缩放比例
        scale = min(self.input_size / H, self.input_size / W)
        new_H, new_W = int(H * scale), int(W * scale)
        
        # 缩放
        image_resized = cv2.resize(image, (new_W, new_H))
        
        # padding 到目标尺寸
        pad_H = self.input_size - new_H
        pad_W = self.input_size - new_W
        image_padded = np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8)
        image_padded[:new_H, :new_W] = image_resized
        
        # 归一化
        image_norm = image_padded.astype(np.float32) / 255.0
        image_norm = (image_norm - self.mean) / self.std
        
        # HWC -> CHW
        image_tensor = torch.from_numpy(image_norm.transpose(2, 0, 1))
        image_tensor = image_tensor.unsqueeze(0)  # (1, 3, H, W)
        
        meta = {
            'original_size': (H, W),
            'scaled_size': (new_H, new_W),
            'pad_size': (pad_H, pad_W),
            'scale': scale,
        }
        
        return image_tensor, meta
    
    def postprocess(
        self,
        outputs: Dict,
        meta: Dict
    ) -> List[Dict]:
        """
        后处理（修复版本）
        
        修复内容:
        - 使用修复后的 bbox 解码函数
        - 添加坐标裁剪
        - 置信度校准（sigmoid）

        Args:
            outputs: 模型输出
            meta: 元数据

        Returns:
            detections: 检测结果列表
        """
        scale = meta['scale']
        original_size = meta['original_size']

        detections = []

        # 解析输出
        cls_preds = outputs['cls_preds']
        reg_preds = outputs['reg_preds']
        kpt_preds = outputs['kpt_preds']

        all_boxes = []
        all_scores = []

        # 收集所有层级的预测
        for level in range(len(cls_preds)):
            # 应用 sigmoid 激活（修复置信度恒为 1.0 的问题）
            cls_score = cls_preds[level].sigmoid()

            # 获取高置信度预测
            mask = cls_score > self.score_thresh
            if mask.sum() == 0:
                continue

            # 获取坐标
            ys, xs = torch.where(mask[0])

            # 解码 bbox（使用修复后的函数）
            reg_pred = reg_preds[level][0][:, mask[0]]
            stride = 8 * (2 ** level)

            # 构建 anchors
            anchors = torch.stack([
                (xs + 0.5) * stride,
                (ys + 0.5) * stride,
                (xs + 0.5) * stride + stride,
                (ys + 0.5) * stride + stride,
            ], dim=-1)  # (N, 4)

            # 解码偏移量（使用修复后的函数）
            # reg_pred shape: (4, N) -> (N, 4)
            reg_pred_t = reg_pred.t().unsqueeze(0)  # (1, N, 4)
            decoded_boxes = decode_bbox_fixed(
                reg_pred_t, 
                anchors.unsqueeze(0), 
                clip=False  # 稍后统一裁剪
            )[0]  # (N, 4)

            # 还原到原始尺寸
            decoded_boxes = decoded_boxes / scale

            # 转换为 numpy
            boxes_np = decoded_boxes.cpu().numpy()
            scores_np = cls_score[0, mask].cpu().numpy()

            all_boxes.append(boxes_np)
            all_scores.append(scores_np)

        # 合并所有层级的预测
        if len(all_boxes) > 0:
            all_boxes = np.vstack(all_boxes)
            all_scores = np.concatenate(all_scores)

            # 裁剪到图像范围（修复坐标超出图像的问题）
            all_boxes = clip_boxes_to_image(
                torch.from_numpy(all_boxes),
                original_size
            ).numpy()

            # 创建检测结果
            for i in range(len(all_boxes)):
                detections.append({
                    'bbox': all_boxes[i],
                    'score': float(all_scores[i]),  # 已经是 sigmoid 后的值
                    'landmarks': None,  # 简化处理
                })

        # NMS
        if len(detections) > 0:
            detections = self._nms(detections, self.nms_thresh)

        # 限制最大数量
        detections = detections[:self.max_faces]

        return detections
    
    def _nms(
        self,
        detections: List[Dict],
        iou_thresh: float
    ) -> List[Dict]:
        """
        非极大值抑制（使用修复后的实现）
        
        Args:
            detections: 检测结果列表
            iou_thresh: IoU 阈值
        
        Returns:
            过滤后的检测结果
        """
        if len(detections) == 0:
            return []
        
        # 转换为 tensor
        boxes = torch.tensor([d['bbox'] for d in detections], dtype=torch.float32)
        scores = torch.tensor([d['score'] for d in detections], dtype=torch.float32)
        
        # 使用修复后的 NMS
        keep_indices = nms_fixed(boxes, scores, iou_threshold=iou_thresh)
        
        return [detections[i] for i in keep_indices]
    
    @torch.no_grad()
    def detect(
        self,
        image: np.ndarray,
        min_face_size: int = 32,
        max_faces: Optional[int] = None
    ) -> List[Dict]:
        """
        人脸检测

        Args:
            image: (H, W, C) BGR 图像
            min_face_size: 最小人脸尺寸
            max_faces: 最大检测数量

        Returns:
            detections: 检测结果列表
        """
        # 预处理
        tensor, meta = self.preprocess(image)
        tensor = tensor.to(self.device)

        # 推理 - 使用 model 的完整 forward（返回检测结果）
        if self.mode == "pytorch":
            # DKGA_Det 在 eval 模式下返回检测结果列表
            detections = self.model(tensor)
            return detections[0] if len(detections) > 0 else []
        elif self.mode == "onnx":
            outputs = self.session.run(None, {self.input_name: tensor.cpu().numpy()})
            # 解析 ONNX 输出
            outputs = {
                'cls_preds': [torch.from_numpy(o) for o in outputs[:3]],
                'reg_preds': [torch.from_numpy(o) for o in outputs[3:6]],
                'kpt_preds': [torch.from_numpy(o) for o in outputs[6:]],
            }
        else:
            # TensorRT 推理
            pass
        
        # 后处理
        detections = self.postprocess(outputs, meta)
        
        # 过滤小人脸
        detections = [
            d for d in detections
            if min(d['bbox'][2] - d['bbox'][0], d['bbox'][3] - d['bbox'][1]) >= min_face_size
        ]
        
        # 限制数量
        max_faces = max_faces or self.max_faces
        detections = detections[:max_faces]
        
        return detections
    
    def detect_batch(
        self,
        images: List[np.ndarray]
    ) -> List[List[Dict]]:
        """批量检测"""
        results = []
        for image in images:
            result = self.detect(image)
            results.append(result)
        return results


if __name__ == "__main__":
    # 测试
    print("Testing Detector...")
    
    detector = Detector(score_thresh=0.6)
    
    # 创建测试图像
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # 检测
    detections = detector.detect(test_image)
    print(f"Detected {len(detections)} faces")
    
    for i, det in enumerate(detections[:3]):
        print(f"  Face {i}: bbox={det['bbox']}, score={det['score']:.3f}")
