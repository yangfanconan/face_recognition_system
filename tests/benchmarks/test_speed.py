"""
性能基准测试
"""

import pytest
import time
import numpy as np
import torch

from models.detection import DKGA_Det
from models.recognition import DDFD_Rec
from inference import Detector, Recognizer, FaceRecognitionPipeline
from inference.index.hnsw_index import HNSWIndex


class TestDetectionSpeed:
    """检测速度基准测试"""
    
    @pytest.fixture
    def model(self):
        """加载模型"""
        model = DKGA_Det()
        model.eval()
        return model
    
    def test_inference_speed(self, model):
        """测试推理速度"""
        x = torch.randn(1, 3, 640, 640)
        
        # 预热
        with torch.no_grad():
            for _ in range(10):
                model(x)
        
        # 测试
        iterations = 100
        start = time.time()
        with torch.no_grad():
            for _ in range(iterations):
                model(x)
        end = time.time()
        
        avg_time = (end - start) / iterations * 1000  # ms
        
        print(f"\nDetection inference time: {avg_time:.2f}ms")
        
        # 目标：<10ms
        assert avg_time < 10, f"Detection too slow: {avg_time:.2f}ms"
    
    def test_batch_inference(self, model):
        """测试批量推理"""
        x = torch.randn(4, 3, 640, 640)
        
        with torch.no_grad():
            start = time.time()
            model(x)
            end = time.time()
        
        batch_time = (end - start) * 1000  # ms
        per_image_time = batch_time / 4
        
        print(f"\nBatch detection (4 images): {batch_time:.2f}ms")
        print(f"Per image: {per_image_time:.2f}ms")


class TestRecognitionSpeed:
    """识别速度基准测试"""
    
    @pytest.fixture
    def model(self):
        """加载模型"""
        model = DDFD_Rec()
        model.eval()
        return model
    
    def test_inference_speed(self, model):
        """测试推理速度"""
        x = torch.randn(1, 3, 112, 112)
        
        # 预热
        with torch.no_grad():
            for _ in range(10):
                model.get_identity_feature(x)
        
        # 测试
        iterations = 100
        start = time.time()
        with torch.no_grad():
            for _ in range(iterations):
                model.get_identity_feature(x)
        end = time.time()
        
        avg_time = (end - start) / iterations * 1000  # ms
        
        print(f"\nRecognition inference time: {avg_time:.2f}ms")
        
        # 目标：<10ms
        assert avg_time < 10, f"Recognition too slow: {avg_time:.2f}ms"


class TestSearchSpeed:
    """搜索速度基准测试"""
    
    def test_hnsw_search_speed(self):
        """测试 HNSW 搜索速度"""
        dim = 512
        
        # 创建索引
        index = HNSWIndex(dim=dim, max_elements=1000000)
        
        # 添加大量特征
        N = 100000
        features = np.random.randn(N, dim).astype(np.float32)
        ids = np.arange(N)
        
        print(f"\nAdding {N} features to index...")
        add_start = time.time()
        index.add(features, ids)
        add_time = time.time() - add_start
        print(f"Add time: {add_time:.2f}s")
        
        # 搜索
        query = np.random.randn(1, dim).astype(np.float32)
        
        # 预热
        index.search(query, k=10)
        
        # 测试
        iterations = 100
        start = time.time()
        for _ in range(iterations):
            index.search(query, k=10)
        end = time.time()
        
        avg_time = (end - start) / iterations * 1000  # ms
        
        print(f"\nHNSW search time (100K features): {avg_time:.2f}ms")
        
        # 目标：<10ms
        assert avg_time < 10, f"Search too slow: {avg_time:.2f}ms"
    
    def test_large_scale_search(self):
        """测试大规模搜索"""
        dim = 512
        
        index = HNSWIndex(dim=dim, max_elements=1000000)
        
        # 100 万特征
        N = 1000000
        features = np.random.randn(N, dim).astype(np.float32)
        ids = np.arange(N)
        
        print(f"\nAdding {N} features to index...")
        index.add(features, ids)
        
        # 搜索
        query = np.random.randn(1, dim).astype(np.float32)
        
        iterations = 100
        start = time.time()
        for _ in range(iterations):
            index.search(query, k=10)
        end = time.time()
        
        avg_time = (end - start) / iterations * 1000  # ms
        
        print(f"\nHNSW search time (1M features): {avg_time:.2f}ms")
        
        # 目标：<20ms
        assert avg_time < 20, f"Search too slow: {avg_time:.2f}ms"


class TestPipelineSpeed:
    """流水线速度基准测试"""
    
    def test_end_to_end_speed(self):
        """测试端到端速度"""
        # 创建流水线 (使用随机模型)
        pipeline = FaceRecognitionPipeline(
            detector_config={'score_thresh': 0.6},
        )
        
        # 创建测试图像
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # 预热
        pipeline.detect(image)
        
        # 测试
        iterations = 50
        start = time.time()
        for _ in range(iterations):
            pipeline.detect(image)
        end = time.time()
        
        avg_time = (end - start) / iterations * 1000  # ms
        
        print(f"\nEnd-to-end detection time: {avg_time:.2f}ms")


class TestMemoryUsage:
    """内存使用测试"""
    
    def test_detection_memory(self):
        """测试检测模型内存"""
        model = DKGA_Det()
        model.eval()
        
        # GPU 内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            start_mem = torch.cuda.memory_allocated()
            
            x = torch.randn(1, 3, 640, 640).cuda()
            with torch.no_grad():
                model(x)
            
            end_mem = torch.cuda.memory_allocated()
            used_mem = (end_mem - start_mem) / 1024 / 1024  # MB
            
            print(f"\nDetection GPU memory: {used_mem:.2f}MB")
    
    def test_recognition_memory(self):
        """测试识别模型内存"""
        model = DDFD_Rec()
        model.eval()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            start_mem = torch.cuda.memory_allocated()
            
            x = torch.randn(1, 3, 112, 112).cuda()
            with torch.no_grad():
                model.get_identity_feature(x)
            
            end_mem = torch.cuda.memory_allocated()
            used_mem = (end_mem - start_mem) / 1024 / 1024  # MB
            
            print(f"\nRecognition GPU memory: {used_mem:.2f}MB")


class TestAccuracy:
    """准确率基准测试"""
    
    def test_feature_discriminability(self):
        """测试特征判别性"""
        model = DDFD_Rec()
        model.eval()
        
        # 生成两个不同的随机图像
        x1 = torch.randn(1, 3, 112, 112)
        x2 = torch.randn(1, 3, 112, 112)
        
        with torch.no_grad():
            feat1 = model.get_identity_feature(x1)
            feat2 = model.get_identity_feature(x2)
        
        # 计算相似度
        sim = torch.cosine_similarity(feat1, feat2, dim=1).item()
        
        print(f"\nRandom images similarity: {sim:.4f}")
        
        # 随机图像相似度应该较低
        assert sim < 0.5, f"Random similarity too high: {sim:.4f}"
    
    def test_same_image_feature(self):
        """测试相同图像特征一致性"""
        model = DDFD_Rec()
        model.eval()
        
        x = torch.randn(1, 3, 112, 112)
        
        with torch.no_grad():
            feat1 = model.get_identity_feature(x)
            feat2 = model.get_identity_feature(x)
        
        # 相同图像特征应该完全相同
        sim = torch.cosine_similarity(feat1, feat2, dim=1).item()
        
        print(f"\nSame image similarity: {sim:.4f}")
        
        assert sim > 0.999, f"Same image similarity too low: {sim:.4f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
