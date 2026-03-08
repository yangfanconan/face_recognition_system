"""
人脸检测推理封装
"""

import os
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path

import numpy as np
import cv2
import torch
import torch.nn.functional as F

from models.detection import DKGA_Det, build_detector


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
        后处理
        
        Args:
            outputs: 模型输出
            meta: 元数据
            
        Returns:
            detections: 检测结果列表
        """
        scale = meta['scale']
        
        detections = []
        
        # 解析输出
        cls_preds = outputs['cls_preds']
        reg_preds = outputs['reg_preds']
        kpt_preds = outputs['kpt_preds']
        
        # 简化处理：假设已经过 NMS
        for level in range(len(cls_preds)):
            cls_score = cls_preds[level].sigmoid()
            
            # 获取高置信度预测
            mask = cls_score > self.score_thresh
            if mask.sum() == 0:
                continue
            
            # 获取坐标
            ys, xs = torch.where(mask[0])
            
            # 解码 bbox
            reg_pred = reg_preds[level][0][:, mask[0]]
            stride = 8 * (2 ** level)
            
            cx = (xs + 0.5) * stride
            cy = (ys + 0.5) * stride
            
            cx_decoded = cx + reg_pred[0] * stride
            cy_decoded = cy + reg_pred[1] * stride
            w_decoded = torch.exp(reg_pred[2]) * stride
            h_decoded = torch.exp(reg_pred[3]) * stride
            
            # 转换为 xyxy
            x1 = cx_decoded - w_decoded / 2
            y1 = cy_decoded - h_decoded / 2
            x2 = cx_decoded + w_decoded / 2
            y2 = cy_decoded + h_decoded / 2
            
            # 还原到原始尺寸
            x1 = (x1 / scale).cpu().numpy()
            y1 = (y1 / scale).cpu().numpy()
            x2 = (x2 / scale).cpu().numpy()
            y2 = (y2 / scale).cpu().numpy()
            
            scores = cls_score[0, mask].cpu().numpy()
            
            for i in range(len(x1)):
                detections.append({
                    'bbox': np.array([x1[i], y1[i], x2[i], y2[i]]),
                    'score': float(scores[i]),
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
        """非极大值抑制"""
        if len(detections) == 0:
            return []
        
        boxes = np.array([d['bbox'] for d in detections])
        scores = np.array([d['score'] for d in detections])
        
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        
        order = scores.argsort()[::-1]
        keep = []
        
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            inter = w * h
            
            iou = inter / (areas[i] + areas[order[1:]] - inter)
            order = order[1:][iou < iou_thresh]
        
        return [detections[i] for i in keep]
    
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
        
        # 推理
        if self.mode == "pytorch":
            outputs = self.model(tensor)
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
