"""
特征匹配模块单元测试
"""

import pytest
import numpy as np

from inference.matcher import (
    FaceVerifier, FaceSearcher, Matcher, QualityAssessor,
    cosine_similarity, weighted_cosine_similarity,
)
from inference.index.hnsw_index import HNSWIndex, build_index


class TestCosineSimilarity:
    """测试余弦相似度计算"""
    
    def test_same_vector(self):
        """测试相同向量"""
        feat = np.random.randn(512).astype(np.float32)
        sim = cosine_similarity(feat, feat)
        
        # 相同向量相似度应该接近 1
        assert np.isclose(sim[0, 0], 1.0, atol=1e-5)
    
    def test_orthogonal_vectors(self):
        """测试正交向量"""
        feat1 = np.random.randn(512).astype(np.float32)
        feat2 = np.random.randn(512).astype(np.float32)
        
        sim = cosine_similarity(feat1, feat2)
        
        # 随机向量相似度应该在 -1 到 1 之间
        assert -1 <= sim[0, 0] <= 1
    
    def test_batch(self):
        """测试批量计算"""
        features1 = np.random.randn(10, 512).astype(np.float32)
        features2 = np.random.randn(10, 512).astype(np.float32)
        
        sim = cosine_similarity(features1, features2)
        
        assert sim.shape == (10, 10)


class TestWeightedCosineSimilarity:
    """测试加权余弦相似度"""
    
    def test_forward(self):
        """测试前向传播"""
        # 完整特征 (id + attr)
        feat1 = np.random.randn(512).astype(np.float32)
        feat2 = np.random.randn(512).astype(np.float32)
        
        sim = weighted_cosine_similarity(
            feat1.reshape(1, -1),
            feat2.reshape(1, -1),
            id_weight=0.85,
            attr_weight=0.15,
            id_dim=409
        )
        
        assert -1 <= sim[0, 0] <= 1


class TestFaceVerifier:
    """测试人脸验证器"""
    
    def test_same_person(self):
        """测试同一人"""
        verifier = FaceVerifier(threshold=0.6)
        
        # 生成相似特征
        base_feat = np.random.randn(512).astype(np.float32)
        feat1 = base_feat + np.random.randn(512).astype(np.float32) * 0.1
        feat2 = base_feat + np.random.randn(512).astype(np.float32) * 0.1
        
        is_same, sim = verifier.verify(feat1, feat2)
        
        # 相似特征应该被判定为同一人
        assert sim > 0.5  # 相似度应该较高
    
    def test_different_person(self):
        """测试不同人"""
        verifier = FaceVerifier(threshold=0.6)
        
        # 生成不同特征
        feat1 = np.random.randn(512).astype(np.float32)
        feat2 = np.random.randn(512).astype(np.float32)
        
        is_same, sim = verifier.verify(feat1, feat2)
        
        # 随机特征相似度应该较低
        assert sim < 0.5
    
    def test_batch_verify(self):
        """测试批量验证"""
        verifier = FaceVerifier(threshold=0.6)
        
        features1 = np.random.randn(10, 512).astype(np.float32)
        features2 = np.random.randn(10, 512).astype(np.float32)
        
        is_same, sims = verifier.verify_batch(features1, features2)
        
        assert len(is_same) == 10
        assert len(sims) == 10


class TestHNSWIndex:
    """测试 HNSW 索引"""
    
    @pytest.fixture
    def index(self):
        """创建测试索引"""
        return HNSWIndex(dim=512, max_elements=10000)
    
    def test_add_and_search(self, index):
        """测试添加和搜索"""
        # 添加特征
        N = 100
        features = np.random.randn(N, 512).astype(np.float32)
        ids = np.arange(N)
        index.add(features, ids)
        
        # 检查统计
        stats = index.get_stats()
        assert stats['element_count'] == N
        
        # 搜索
        query = features[0:1]  # 使用第一个特征作为查询
        labels, similarities = index.search(query, k=5)
        
        # 第一个结果应该是查询本身
        assert labels[0, 0] == 0
        assert similarities[0, 0] > 0.99  # 自身相似度应该接近 1
    
    def test_save_and_load(self, index, tmp_path):
        """测试保存和加载"""
        # 添加特征
        N = 100
        features = np.random.randn(N, 512).astype(np.float32)
        ids = np.arange(N)
        index.add(features, ids)
        
        # 保存
        save_path = str(tmp_path / "test_index")
        index.save(save_path)
        
        # 加载
        loaded_index = HNSWIndex.load(save_path)
        
        # 验证
        assert loaded_index.element_count == N
        
        # 搜索结果应该相同
        query = features[0:1]
        labels1, sims1 = index.search(query, k=5)
        labels2, sims2 = loaded_index.search(query, k=5)
        
        assert np.array_equal(labels1, labels2)


class TestQualityAssessor:
    """测试质量评估器"""
    
    def test_assess_feature(self):
        """测试特征质量评估"""
        assessor = QualityAssessor(quality_threshold=0.5)
        
        # 正常特征 (范数接近 1)
        feat = np.random.randn(512).astype(np.float32)
        feat = feat / np.linalg.norm(feat)
        
        quality = assessor.assess(feat)
        
        assert 'norm' in quality
        assert 'overall_quality' in quality
        assert 'passed' in quality
        
        # 归一化特征应该通过
        assert quality['norm'] > 0.9
    
    def test_assess_with_image(self):
        """测试带图像的质量评估"""
        assessor = QualityAssessor()
        
        feat = np.random.randn(512).astype(np.float32)
        feat = feat / np.linalg.norm(feat)
        
        # 创建测试图像
        image = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
        
        quality = assessor.assess(feat, image)
        
        assert 'image_quality' in quality


class TestMatcher:
    """测试统一匹配器"""
    
    def test_verify(self):
        """测试验证功能"""
        matcher = Matcher(threshold=0.6)
        
        feat1 = np.random.randn(512).astype(np.float32)
        feat2 = np.random.randn(512).astype(np.float32)
        
        is_same, sim = matcher.verify(feat1, feat2)
        
        assert isinstance(is_same, bool)
        assert isinstance(sim, float)
    
    def test_quality_assess(self):
        """测试质量评估"""
        matcher = Matcher()
        
        feat = np.random.randn(512).astype(np.float32)
        quality = matcher.assess_quality(feat)
        
        assert 'overall_quality' in quality


class TestBuildIndex:
    """测试索引工厂"""
    
    def test_build_hnsw(self):
        """测试构建 HNSW 索引"""
        index = build_index(
            index_type="hnsw",
            dim=512,
            max_elements=10000
        )
        
        assert isinstance(index, HNSWIndex)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
