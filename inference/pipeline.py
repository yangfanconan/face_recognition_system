"""
推理流水线模块

完整流程：图像采集 → 人脸检测 → 关键点对齐 → 特征提取 → 质量评估 → 匹配/搜索
"""

import time
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path

import numpy as np
import cv2
import torch

from inference.detector import Detector
from inference.recognizer import Recognizer
from inference.matcher import Matcher, QualityAssessor
from inference.index.hnsw_index import HNSWIndex


# ============================================
# 推理流水线
# ============================================

class FaceRecognitionPipeline:
    """
    人脸识别完整流水线
    """
    
    def __init__(
        self,
        detector_config: Optional[Dict] = None,
        recognizer_config: Optional[Dict] = None,
        matcher_config: Optional[Dict] = None,
        device: Optional[str] = None,
    ):
        """
        Args:
            detector_config: 检测器配置
            recognizer_config: 识别器配置
            matcher_config: 匹配器配置
            device: 计算设备
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 初始化组件
        self.detector = Detector(**(detector_config or {}))
        self.recognizer = Recognizer(**(recognizer_config or {}))
        self.matcher = Matcher(**(matcher_config or {}))
        self.assessor = QualityAssessor()
        
        # 搜索索引
        self.search_index: Optional[HNSWIndex] = None
        
        # 统计信息
        self.stats = {
            'total_processed': 0,
            'total_faces_detected': 0,
            'total_faces_recognized': 0,
        }
    
    def detect(
        self,
        image: np.ndarray,
        min_face_size: int = 32,
        max_faces: int = 50
    ) -> Dict:
        """
        人脸检测
        
        Args:
            image: (H, W, C) 输入图像
            min_face_size: 最小人脸尺寸
            max_faces: 最大检测数量
            
        Returns:
            results: 检测结果
        """
        start_time = time.time()
        
        # 检测
        detections = self.detector.detect(
            image,
            min_face_size=min_face_size,
            max_faces=max_faces
        )
        
        # 质量过滤
        valid_faces = []
        for det in detections:
            # 置信度过滤
            if det.get('score', 0) < 0.6:
                continue
            
            # 尺寸过滤
            bbox = det['bbox']
            face_size = min(bbox[2] - bbox[0], bbox[3] - bbox[1])
            if face_size < min_face_size:
                continue
            
            valid_faces.append(det)
        
        elapsed = time.time() - start_time
        
        return {
            'faces': valid_faces,
            'count': len(valid_faces),
            'inference_time': elapsed,
        }
    
    def extract(
        self,
        image: np.ndarray,
        bbox: Optional[np.ndarray] = None,
        face_index: int = 0
    ) -> Dict:
        """
        特征提取
        
        Args:
            image: (H, W, C) 输入图像
            bbox: (4,) 人脸框，None 则自动检测
            face_index: 使用第几个人脸
            
        Returns:
            result: 特征提取结果
        """
        start_time = time.time()
        
        # 如果没有提供 bbox，先检测
        if bbox is None:
            det_results = self.detect(image, max_faces=face_index + 1)
            if det_results['count'] <= face_index:
                return {
                    'success': False,
                    'error': 'No face detected',
                }
            bbox = det_results['faces'][face_index]['bbox']
        
        # 提取特征
        feature = self.recognizer.extract(image, bbox)
        
        # 质量评估
        quality = self.assessor.assess(feature, image[bbox[1]:bbox[3], bbox[0]:bbox[2]])
        
        elapsed = time.time() - start_time
        
        return {
            'success': True,
            'feature': feature,
            'bbox': bbox,
            'quality': quality,
            'inference_time': elapsed,
        }
    
    def verify(
        self,
        image1: np.ndarray,
        image2: np.ndarray,
        bbox1: Optional[np.ndarray] = None,
        bbox2: Optional[np.ndarray] = None,
        threshold: Optional[float] = None
    ) -> Dict:
        """
        人脸验证 (1:1)
        
        Args:
            image1: 图像 1
            image2: 图像 2
            bbox1: 图像 1 的人脸框
            bbox2: 图像 2 的人脸框
            threshold: 验证阈值
            
        Returns:
            result: 验证结果
        """
        start_time = time.time()
        
        # 提取特征
        result1 = self.extract(image1, bbox1)
        result2 = self.extract(image2, bbox2)
        
        if not result1['success'] or not result2['success']:
            return {
                'success': False,
                'error': 'Failed to extract features',
            }
        
        # 验证
        is_same, similarity = self.matcher.verify(
            result1['feature'],
            result2['feature'],
            threshold=threshold
        )
        
        elapsed = time.time() - start_time
        
        self.stats['total_processed'] += 2
        self.stats['total_faces_recognized'] += 1 if is_same else 0
        
        return {
            'success': True,
            'is_same': is_same,
            'similarity': similarity,
            'quality1': result1['quality'],
            'quality2': result2['quality'],
            'inference_time': elapsed,
        }
    
    def search(
        self,
        image: np.ndarray,
        bbox: Optional[np.ndarray] = None,
        top_k: int = 10,
        threshold: float = 0.6
    ) -> Dict:
        """
        人脸搜索 (1:N)
        
        Args:
            image: 查询图像
            bbox: 人脸框
            top_k: 返回数量
            threshold: 相似度阈值
            
        Returns:
            result: 搜索结果
        """
        if self.search_index is None:
            return {
                'success': False,
                'error': 'Search index not initialized',
            }
        
        start_time = time.time()
        
        # 提取特征
        extract_result = self.extract(image, bbox)
        if not extract_result['success']:
            return extract_result
        
        # 搜索
        results = self.matcher.search(
            extract_result['feature'],
            top_k=top_k,
            threshold=threshold
        )
        
        elapsed = time.time() - start_time
        
        self.stats['total_processed'] += 1
        
        return {
            'success': True,
            'query_quality': extract_result['quality'],
            'results': results,
            'count': len(results),
            'inference_time': elapsed,
        }
    
    def register(
        self,
        image: np.ndarray,
        person_id: Union[str, int],
        bbox: Optional[np.ndarray] = None,
        metadata: Optional[Dict] = None
    ) -> Dict:
        """
        注册人脸到图库
        
        Args:
            image: 注册图像
            person_id: 人员 ID
            bbox: 人脸框
            metadata: 元数据
            
        Returns:
            result: 注册结果
        """
        if self.search_index is None:
            # 初始化索引
            self.search_index = HNSWIndex(dim=512, max_elements=1000000)
            self.matcher.init_search_index(self.search_index)
        
        # 提取特征
        extract_result = self.extract(image, bbox)
        if not extract_result['success']:
            return extract_result
        
        # 质量检查
        if not extract_result['quality'].get('passed', False):
            return {
                'success': False,
                'error': 'Feature quality too low',
                'quality': extract_result['quality'],
            }
        
        # 添加到索引
        self.search_index.add(
            extract_result['feature'].reshape(1, -1),
            np.array([person_id])
        )
        
        return {
            'success': True,
            'person_id': person_id,
            'quality': extract_result['quality'],
        }
    
    def process(
        self,
        image: np.ndarray,
        mode: str = "detect",
        **kwargs
    ) -> Dict:
        """
        统一处理接口
        
        Args:
            image: 输入图像
            mode: 处理模式 ('detect', 'extract', 'verify', 'search', 'register')
            **kwargs: 模式相关参数
            
        Returns:
            result: 处理结果
        """
        if mode == "detect":
            return self.detect(image, **kwargs)
        elif mode == "extract":
            return self.extract(image, **kwargs)
        elif mode == "verify":
            return self.verify(image, **kwargs)
        elif mode == "search":
            return self.search(image, **kwargs)
        elif mode == "register":
            return self.register(image, **kwargs)
        else:
            return {
                'success': False,
                'error': f'Unknown mode: {mode}',
            }
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        return self.stats.copy()
    
    def reset_stats(self) -> None:
        """重置统计信息"""
        self.stats = {
            'total_processed': 0,
            'total_faces_detected': 0,
            'total_faces_recognized': 0,
        }


# ============================================
# 批量处理
# ============================================

class BatchProcessor:
    """
    批量处理器
    """
    
    def __init__(self, pipeline: FaceRecognitionPipeline, batch_size: int = 32):
        self.pipeline = pipeline
        self.batch_size = batch_size
    
    def extract_batch(
        self,
        images: List[np.ndarray],
        bboxes: Optional[List[np.ndarray]] = None
    ) -> List[Dict]:
        """
        批量特征提取
        
        Args:
            images: 图像列表
            bboxes: 人脸框列表
            
        Returns:
            results: 提取结果列表
        """
        results = []
        
        for i in range(0, len(images), self.batch_size):
            batch_images = images[i:i + self.batch_size]
            batch_bboxes = bboxes[i:i + self.batch_size] if bboxes else [None] * len(batch_images)
            
            for img, bbox in zip(batch_images, batch_bboxes):
                result = self.pipeline.extract(img, bbox)
                results.append(result)
        
        return results
    
    def verify_batch(
        self,
        pairs: List[Tuple[np.ndarray, np.ndarray]]
    ) -> List[Dict]:
        """
        批量验证
        
        Args:
            pairs: (image1, image2) 对列表
            
        Returns:
            results: 验证结果列表
        """
        results = []
        
        for img1, img2 in pairs:
            result = self.pipeline.verify(img1, img2)
            results.append(result)
        
        return results


# ============================================
# 工厂函数
# ============================================

def build_pipeline(
    config_path: Optional[str] = None,
    **kwargs
) -> FaceRecognitionPipeline:
    """
    构建推理流水线
    
    Args:
        config_path: 配置文件路径
        **kwargs: 配置参数
        
    Returns:
        流水线实例
    """
    # 从配置文件加载 (可选)
    if config_path:
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        kwargs.update(config.get('pipeline', {}))
    
    return FaceRecognitionPipeline(**kwargs)


if __name__ == "__main__":
    # 测试流水线
    print("Testing Face Recognition Pipeline...")
    
    # 创建流水线
    pipeline = FaceRecognitionPipeline()
    
    # 创建测试图像
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # 测试检测
    print("\n--- Testing Detection ---")
    det_result = pipeline.detect(test_image)
    print(f"Detected {det_result['count']} faces")
    print(f"Inference time: {det_result['inference_time']:.3f}s")
    
    # 测试特征提取
    print("\n--- Testing Feature Extraction ---")
    if det_result['count'] > 0:
        bbox = det_result['faces'][0]['bbox']
        extract_result = pipeline.extract(test_image, bbox=bbox)
        if extract_result['success']:
            print(f"Feature shape: {extract_result['feature'].shape}")
            print(f"Quality: {extract_result['quality']}")
    
    # 测试统计
    print("\n--- Statistics ---")
    print(f"Stats: {pipeline.get_stats()}")
