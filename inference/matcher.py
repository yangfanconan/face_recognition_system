"""
特征匹配模块

包含:
- 余弦相似度匹配
- 加权特征匹配
- 人脸验证 (1:1)
- 人脸搜索 (1:N)
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union


# ============================================
# 相似度计算
# ============================================

def cosine_similarity(
    feat1: np.ndarray,
    feat2: np.ndarray,
    axis: int = -1
) -> np.ndarray:
    """
    计算余弦相似度
    
    Args:
        feat1: (N, D) 或 (D,) 特征 1
        feat2: (M, D) 或 (D,) 特征 2
        axis: 特征维度
        
    Returns:
        similarity: 相似度分数
    """
    # 归一化
    feat1_norm = feat1 / (np.linalg.norm(feat1, axis=axis, keepdims=True) + 1e-8)
    feat2_norm = feat2 / (np.linalg.norm(feat2, axis=axis, keepdims=True) + 1e-8)
    
    # 点积
    similarity = np.dot(feat1_norm, feat2_norm.T)
    
    return similarity


def weighted_cosine_similarity(
    feat1: np.ndarray,
    feat2: np.ndarray,
    id_weight: float = 0.85,
    attr_weight: float = 0.15,
    id_dim: int = 409
) -> np.ndarray:
    """
    加权余弦相似度
    
    身份特征和属性特征分别计算相似度后加权
    
    Args:
        feat1: (N, D) 完整特征 (id + attr)
        feat2: (M, D) 完整特征
        id_weight: 身份权重
        attr_weight: 属性权重
        id_dim: 身份子空间维度
        
    Returns:
        similarity: 加权相似度
    """
    # 分解特征
    id_feat1 = feat1[:, :id_dim]
    attr_feat1 = feat1[:, id_dim:]
    
    id_feat2 = feat2[:, :id_dim]
    attr_feat2 = feat2[:, id_dim:]
    
    # 分别计算相似度
    id_sim = cosine_similarity(id_feat1, id_feat2)
    attr_sim = cosine_similarity(attr_feat1, attr_feat2)
    
    # 加权融合
    # 属性相似度取反 (属性差异越大，相似度越低)
    final_sim = id_weight * id_sim + attr_weight * (1 - np.abs(attr_sim))
    
    return final_sim


def euclidean_distance(
    feat1: np.ndarray,
    feat2: np.ndarray
) -> np.ndarray:
    """计算欧氏距离"""
    diff = feat1 - feat2
    return np.sqrt(np.sum(diff ** 2, axis=-1))


# ============================================
# 人脸验证 (1:1)
# ============================================

class FaceVerifier:
    """
    人脸验证器 (1:1 比对)
    """
    
    def __init__(
        self,
        threshold: float = 0.6,
        id_weight: float = 0.85,
        attr_weight: float = 0.15,
        id_dim: int = 409,
        use_weighted: bool = True
    ):
        """
        Args:
            threshold: 验证阈值
            id_weight: 身份权重
            attr_weight: 属性权重
            id_dim: 身份子空间维度
            use_weighted: 是否使用加权相似度
        """
        self.threshold = threshold
        self.id_weight = id_weight
        self.attr_weight = attr_weight
        self.id_dim = id_dim
        self.use_weighted = use_weighted
    
    def verify(
        self,
        feat1: np.ndarray,
        feat2: np.ndarray,
        threshold: Optional[float] = None
    ) -> Tuple[bool, float]:
        """
        验证两张人脸是否同一人
        
        Args:
            feat1: (D,) 特征 1
            feat2: (D,) 特征 2
            threshold: 临时阈值
            
        Returns:
            is_same: 是否同一人
            similarity: 相似度
        """
        # 确保 2D
        if feat1.ndim == 1:
            feat1 = feat1.reshape(1, -1)
        if feat2.ndim == 1:
            feat2 = feat2.reshape(1, -1)
        
        # 计算相似度
        if self.use_weighted:
            similarity = weighted_cosine_similarity(
                feat1, feat2,
                id_weight=self.id_weight,
                attr_weight=self.attr_weight,
                id_dim=self.id_dim
            )[0, 0]
        else:
            similarity = cosine_similarity(feat1, feat2)[0, 0]
        
        # 判断
        thresh = threshold if threshold is not None else self.threshold
        is_same = similarity >= thresh
        
        return is_same, float(similarity)
    
    def verify_batch(
        self,
        features1: np.ndarray,
        features2: np.ndarray,
        threshold: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        批量验证
        
        Args:
            features1: (N, D) 特征 1
            features2: (N, D) 特征 2
            
        Returns:
            is_same: (N,) 是否同一人
            similarities: (N,) 相似度
        """
        if self.use_weighted:
            similarities = np.array([
                weighted_cosine_similarity(
                    features1[i:i+1], features2[i:i+1],
                    id_weight=self.id_weight,
                    attr_weight=self.attr_weight,
                    id_dim=self.id_dim
                )[0, 0]
                for i in range(len(features1))
            ])
        else:
            similarities = np.diag(cosine_similarity(features1, features2))
        
        thresh = threshold if threshold is not None else self.threshold
        is_same = similarities >= thresh
        
        return is_same, similarities


# ============================================
# 人脸搜索 (1:N)
# ============================================

class FaceSearcher:
    """
    人脸搜索器 (1:N 检索)
    """
    
    def __init__(
        self,
        index,  # HNSWIndex 或 FaissIndex
        id_weight: float = 0.85,
        attr_weight: float = 0.15,
        id_dim: int = 409,
        use_weighted: bool = True
    ):
        """
        Args:
            index: 特征索引
            id_weight: 身份权重
            attr_weight: 属性权重
            id_dim: 身份子空间维度
            use_weighted: 是否使用加权相似度
        """
        self.index = index
        self.id_weight = id_weight
        self.attr_weight = attr_weight
        self.id_dim = id_dim
        self.use_weighted = use_weighted
    
    def search(
        self,
        query_feat: np.ndarray,
        top_k: int = 10,
        threshold: float = 0.6
    ) -> List[Dict]:
        """
        搜索相似人脸
        
        Args:
            query_feat: (D,) 查询特征
            top_k: 返回数量
            threshold: 相似度阈值
            
        Returns:
            results: 搜索结果列表
        """
        if query_feat.ndim == 1:
            query_feat = query_feat.reshape(1, -1)
        
        # 索引搜索
        labels, similarities = self.index.search(query_feat, k=top_k)
        
        # 过滤低于阈值的
        results = []
        for i in range(len(labels[0])):
            sim = similarities[0, i]
            if sim < threshold:
                continue
            
            results.append({
                'id': int(labels[0, i]),
                'similarity': float(sim),
                'rank': i + 1,
            })
        
        return results
    
    def add_to_gallery(
        self,
        features: np.ndarray,
        ids: np.ndarray
    ) -> None:
        """
        添加特征到图库
        
        Args:
            features: (N, D) 特征
            ids: (N,) ID
        """
        self.index.add(features, ids)


# ============================================
# 质量评估
# ============================================

class QualityAssessor:
    """
    特征质量评估器
    """
    
    def __init__(
        self,
        quality_threshold: float = 0.5
    ):
        self.quality_threshold = quality_threshold
    
    def assess(
        self,
        feature: np.ndarray,
        image: Optional[np.ndarray] = None
    ) -> Dict:
        """
        评估特征质量
        
        Args:
            feature: (D,) 特征向量
            image: (H, W, C) 原始图像 (可选)
            
        Returns:
            quality_info: 质量信息字典
        """
        quality_info = {}
        
        # 特征范数 (应该接近 1)
        norm = np.linalg.norm(feature)
        quality_info['norm'] = float(norm)
        quality_info['norm_quality'] = min(norm, 2.0 - norm) if norm < 2 else 0.0
        
        # 图像质量 (如果提供)
        if image is not None:
            quality_info['image_quality'] = self._assess_image_quality(image)
        
        # 综合质量
        quality_info['overall_quality'] = (
            quality_info.get('norm_quality', 0.5) * 0.5 +
            quality_info.get('image_quality', 0.5) * 0.5
        )
        
        quality_info['passed'] = quality_info['overall_quality'] >= self.quality_threshold
        
        return quality_info
    
    def _assess_image_quality(
        self,
        image: np.ndarray,
        min_size: int = 32
    ) -> float:
        """评估图像质量"""
        H, W = image.shape[:2]
        
        # 尺寸检查
        size_score = min(1.0, min(H, W) / min_size)
        
        # 模糊度检测 (Laplacian 方差)
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image
        
        try:
            import cv2
            blur_score = cv2.Laplacian(gray.astype(np.uint8), cv2.CV_64F).var()
            blur_quality = min(1.0, blur_score / 100)
        except:
            blur_quality = 0.5
        
        return size_score * 0.3 + blur_quality * 0.7


# ============================================
# 匹配器工厂
# ============================================

class Matcher:
    """
    统一匹配器接口
    """
    
    def __init__(
        self,
        threshold: float = 0.6,
        id_weight: float = 0.85,
        attr_weight: float = 0.15,
        id_dim: int = 409,
        **kwargs
    ):
        self.verifier = FaceVerifier(
            threshold=threshold,
            id_weight=id_weight,
            attr_weight=attr_weight,
            id_dim=id_dim
        )
        self.searcher = None  # 需要时初始化
        self.assessor = QualityAssessor()
    
    def verify(
        self,
        feat1: np.ndarray,
        feat2: np.ndarray,
        threshold: Optional[float] = None
    ) -> Tuple[bool, float]:
        """1:1 验证"""
        return self.verifier.verify(feat1, feat2, threshold)
    
    def search(
        self,
        query_feat: np.ndarray,
        top_k: int = 10,
        threshold: float = 0.6
    ) -> List[Dict]:
        """1:N 搜索"""
        if self.searcher is None:
            raise RuntimeError("Search index not initialized")
        return self.searcher.search(query_feat, top_k, threshold)
    
    def init_search_index(self, index) -> None:
        """初始化搜索索引"""
        self.searcher = FaceSearcher(index=index)
    
    def assess_quality(
        self,
        feature: np.ndarray,
        image: Optional[np.ndarray] = None
    ) -> Dict:
        """评估质量"""
        return self.assessor.assess(feature, image)


# ============================================
# PyTorch 版本工具函数
# ============================================

def torch_cosine_similarity(
    feat1: torch.Tensor,
    feat2: torch.Tensor,
    dim: int = -1
) -> torch.Tensor:
    """PyTorch 余弦相似度"""
    return F.cosine_similarity(feat1, feat2, dim=dim)


def torch_weighted_similarity(
    feat1: torch.Tensor,
    feat2: torch.Tensor,
    id_weight: float = 0.85,
    attr_weight: float = 0.15,
    id_dim: int = 409
) -> torch.Tensor:
    """PyTorch 加权相似度"""
    id_feat1 = feat1[..., :id_dim]
    attr_feat1 = feat1[..., id_dim:]
    id_feat2 = feat2[..., :id_dim]
    attr_feat2 = feat2[..., id_dim:]
    
    id_sim = torch_cosine_similarity(id_feat1, id_feat2)
    attr_sim = torch_cosine_similarity(attr_feat1, attr_feat2)
    
    return id_weight * id_sim + attr_weight * (1 - torch.abs(attr_sim))


if __name__ == "__main__":
    # 测试
    print("Testing Face Verifier...")
    
    verifier = FaceVerifier(threshold=0.6)
    
    # 生成随机特征
    feat1 = np.random.randn(512).astype(np.float32)
    feat2 = np.random.randn(512).astype(np.float32)
    
    # 验证
    is_same, sim = verifier.verify(feat1, feat2)
    print(f"Similarity: {sim:.4f}, Is same: {is_same}")
    
    # 批量验证
    features1 = np.random.randn(10, 512).astype(np.float32)
    features2 = np.random.randn(10, 512).astype(np.float32)
    
    is_same_batch, sims_batch = verifier.verify_batch(features1, features2)
    print(f"Batch similarities: {sims_batch}")
    
    # 质量评估
    assessor = QualityAssessor()
    quality = assessor.assess(feat1)
    print(f"Quality: {quality}")
