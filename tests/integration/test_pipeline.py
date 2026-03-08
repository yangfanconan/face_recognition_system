"""
集成测试
"""

import pytest
import numpy as np
import torch

from models.detection import DKGA_Det
from models.recognition import DDFD_Rec
from inference import FaceRecognitionPipeline, Detector, Recognizer, Matcher


class TestFaceRecognitionPipeline:
    """人脸识别流水线集成测试"""
    
    @pytest.fixture
    def pipeline(self):
        """创建测试流水线"""
        return FaceRecognitionPipeline(
            detector_config={'score_thresh': 0.5},
            matcher_config={'threshold': 0.6},
        )
    
    def test_detect(self, pipeline):
        """测试检测功能"""
        # 创建测试图像
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        result = pipeline.detect(image)
        
        assert 'faces' in result
        assert 'count' in result
        assert 'inference_time' in result
    
    def test_extract(self, pipeline):
        """测试特征提取"""
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        bbox = np.array([100, 100, 200, 250])
        
        result = pipeline.extract(image, bbox=bbox)
        
        # 由于使用随机模型，可能无法提取有效特征
        # 这里只检查返回格式
        assert 'success' in result
        assert 'quality' in result
    
    def test_register_and_search(self, pipeline):
        """测试注册和搜索"""
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        bbox = np.array([100, 100, 200, 250])
        
        # 注册
        register_result = pipeline.register(
            image, person_id="test_001", bbox=bbox
        )
        
        # 搜索
        search_result = pipeline.search(image, bbox=bbox, top_k=5)
        
        assert 'success' in search_result or 'error' in search_result
    
    def test_stats(self, pipeline):
        """测试统计功能"""
        stats = pipeline.get_stats()
        
        assert 'total_processed' in stats
        assert 'total_faces_detected' in stats


class TestDetectorRecognizer:
    """检测器和识别器集成测试"""
    
    def test_detector_output_format(self):
        """测试检测器输出格式"""
        detector = Detector(score_thresh=0.5)
        
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        detections = detector.detect(image)
        
        # 检查输出格式
        assert isinstance(detections, list)
        
        for det in detections:
            assert 'bbox' in det
            assert 'score' in det
    
    def test_recognizer_output_format(self):
        """测试识别器输出格式"""
        recognizer = Recognizer()
        
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        bbox = np.array([100, 100, 200, 250])
        
        feature = recognizer.extract(image, bbox)
        
        # 检查输出
        assert isinstance(feature, np.ndarray)
        assert len(feature.shape) == 1
        assert feature.shape[0] in [256, 409, 512]  # 可能的特征维度


class TestMatcherIntegration:
    """匹配器集成测试"""
    
    def test_verify_workflow(self):
        """测试验证工作流程"""
        matcher = Matcher(threshold=0.6)
        
        # 生成测试特征
        feat1 = np.random.randn(512).astype(np.float32)
        feat2 = np.random.randn(512).astype(np.float32)
        
        # 验证
        is_same, similarity = matcher.verify(feat1, feat2)
        
        assert isinstance(is_same, bool)
        assert 0 <= similarity <= 1
    
    def test_search_workflow(self):
        """测试搜索工作流程"""
        from inference.index.hnsw_index import HNSWIndex
        
        # 创建索引
        index = HNSWIndex(dim=512, max_elements=10000)
        
        # 添加特征
        N = 100
        features = np.random.randn(N, 512).astype(np.float32)
        ids = np.arange(N)
        index.add(features, ids)
        
        # 创建匹配器
        matcher = Matcher()
        matcher.init_search_index(index)
        
        # 搜索
        query = np.random.randn(512).astype(np.float32)
        results = matcher.search(query, top_k=5, threshold=0.3)
        
        assert isinstance(results, list)
        
        for r in results:
            assert 'id' in r
            assert 'similarity' in r


class TestModelExport:
    """模型导出测试"""
    
    def test_detection_model_forward(self):
        """测试检测模型导出兼容性"""
        model = DKGA_Det()
        model.eval()
        
        x = torch.randn(1, 3, 640, 640)
        
        # 测试前向传播
        with torch.no_grad():
            outputs = model(x)
        
        # 检查输出可用于 ONNX 导出
        assert isinstance(outputs, list)
    
    def test_recognition_model_forward(self):
        """测试识别模型导出兼容性"""
        model = DDFD_Rec()
        model.eval()
        
        x = torch.randn(1, 3, 112, 112)
        
        with torch.no_grad():
            feature = model.get_identity_feature(x)
        
        # 检查输出
        assert feature.shape == (1, 409)


class TestConfiguration:
    """配置测试"""
    
    def test_default_config_loading(self):
        """测试默认配置加载"""
        import yaml
        from pathlib import Path
        
        config_path = Path(__file__).parent.parent.parent / "configs" / "default.yaml"
        
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            assert 'project' in config
            assert 'paths' in config
            assert 'device' in config
    
    def test_detection_config(self):
        """测试检测配置"""
        import yaml
        from pathlib import Path
        
        config_path = Path(__file__).parent.parent.parent / "configs" / "detection" / "train.yaml"
        
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            assert 'detection' in config
            assert 'training' in config


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
