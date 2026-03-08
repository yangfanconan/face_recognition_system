"""
HNSW 特征索引模块

高效近似最近邻搜索
"""

import os
import pickle
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path

import numpy as np
import torch

try:
    import hnswlib
    HNSW_AVAILABLE = True
except ImportError:
    HNSW_AVAILABLE = False
    print("Warning: hnswlib not installed. Install with: pip install hnswlib")


class HNSWIndex:
    """
    HNSW 特征索引
    
    支持:
    - 增量添加特征
    - 批量搜索
    - 索引保存/加载
    - 多线程搜索
    """
    
    def __init__(
        self,
        dim: int = 512,
        space: str = "cosine",
        max_elements: int = 1000000,
        ef_construction: int = 200,
        M: int = 16,
        num_threads: int = -1
    ):
        """
        Args:
            dim: 特征维度
            space: 距离空间 ('cosine', 'l2', 'ip')
            max_elements: 最大元素数量
            ef_construction: 构建时 ef 参数
            M: 最大连接数
            num_threads: 线程数 (-1 表示自动)
        """
        if not HNSW_AVAILABLE:
            raise ImportError("hnswlib is required for HNSWIndex")
        
        self.dim = dim
        self.space = space
        self.max_elements = max_elements
        self.ef_construction = ef_construction
        self.M = M
        self.num_threads = num_threads
        
        # 初始化索引
        self.index = hnswlib.Index(space=space, dim=dim)
        self.index.init_index(
            max_elements=max_elements,
            ef_construction=ef_construction,
            M=M
        )
        
        # 搜索参数
        self.ef_search = 50
        
        # 元素计数
        self.element_count = 0
        self.id_mapping: Dict[int, int] = {}  # 外部 ID -> 内部 ID
        self.reverse_mapping: Dict[int, int] = {}  # 内部 ID -> 外部 ID
    
    def set_ef(self, ef: int) -> None:
        """设置搜索时的 ef 参数"""
        self.ef_search = ef
        self.index.set_ef(ef)
    
    def add(
        self,
        features: np.ndarray,
        ids: Optional[np.ndarray] = None,
        replace_deleted: bool = False
    ) -> None:
        """
        添加特征到索引
        
        Args:
            features: (N, dim) 特征数组
            ids: (N,) 外部 ID 数组，None 则自动生成
            replace_deleted: 是否替换已删除的元素
        """
        N = features.shape[0]
        
        # 确保特征归一化 (cosine 空间)
        if self.space == "cosine":
            norms = np.linalg.norm(features, axis=1, keepdims=True)
            norms[norms == 0] = 1
            features = features / norms
        
        # 生成 ID
        if ids is None:
            ids = np.arange(self.element_count, self.element_count + N)
        
        # 检查索引容量
        if self.element_count + N > self.max_elements:
            # 需要扩容
            new_max = max(self.max_elements * 2, self.element_count + N)
            self.index.resize_index(new_max)
            self.max_elements = new_max
        
        # 添加
        internal_ids = np.arange(self.element_count, self.element_count + N)
        self.index.add_items(features, internal_ids, num_threads=self.num_threads)
        
        # 更新映射
        for ext_id, int_id in zip(ids, internal_ids):
            self.id_mapping[ext_id] = int_id
            self.reverse_mapping[int_id] = ext_id
        
        self.element_count += N
    
    def search(
        self,
        query_features: np.ndarray,
        k: int = 10,
        filter_ids: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        搜索最近邻
        
        Args:
            query_features: (Q, dim) 查询特征
            k: 返回的最近邻数量
            filter_ids: 可选的 ID 过滤列表
            
        Returns:
            labels: (Q, k) 最近邻 ID
            distances: (Q, k) 距离
        """
        Q = query_features.shape[0]
        
        # 确保特征归一化
        if self.space == "cosine":
            norms = np.linalg.norm(query_features, axis=1, keepdims=True)
            norms[norms == 0] = 1
            query_features = query_features / norms
        
        # 搜索
        labels, distances = self.index.knn_query(
            query_features,
            k=k,
            num_threads=self.num_threads
        )
        
        # 转换 ID
        external_labels = np.zeros_like(labels)
        for i in range(Q):
            for j in range(k):
                int_id = labels[i, j]
                external_labels[i, j] = self.reverse_mapping.get(int_id, int_id)
        
        # 距离转相似度 (cosine 空间)
        if self.space == "cosine":
            similarities = 1 - distances
            return external_labels, similarities
        
        return external_labels, distances
    
    def delete(self, ids: np.ndarray) -> None:
        """
        删除指定 ID 的特征
        
        Args:
            ids: 要删除的外部 ID 列表
        """
        for ext_id in ids:
            if ext_id in self.id_mapping:
                int_id = self.id_mapping[ext_id]
                self.index.mark_deleted(int_id)
                del self.id_mapping[ext_id]
                del self.reverse_mapping[int_id]
    
    def save(self, path: str) -> None:
        """
        保存索引到磁盘
        
        Args:
            path: 保存路径
        """
        # 保存 HNSW 索引
        self.index.save_index(path)
        
        # 保存元数据
        metadata_path = path + ".meta.pkl"
        metadata = {
            'dim': self.dim,
            'space': self.space,
            'max_elements': self.max_elements,
            'ef_construction': self.ef_construction,
            'M': self.M,
            'element_count': self.element_count,
            'id_mapping': self.id_mapping,
            'reverse_mapping': self.reverse_mapping,
        }
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
    
    @classmethod
    def load(cls, path: str, num_threads: int = -1) -> 'HNSWIndex':
        """
        从磁盘加载索引
        
        Args:
            path: 索引路径
            num_threads: 线程数
            
        Returns:
            HNSWIndex 实例
        """
        # 加载元数据
        metadata_path = path + ".meta.pkl"
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        # 创建实例
        index = cls(
            dim=metadata['dim'],
            space=metadata['space'],
            max_elements=metadata['max_elements'],
            ef_construction=metadata['ef_construction'],
            M=metadata['M'],
            num_threads=num_threads
        )
        
        # 加载索引
        index.index.load_index(path)
        index.element_count = metadata['element_count']
        index.id_mapping = metadata['id_mapping']
        index.reverse_mapping = metadata['reverse_mapping']
        
        return index
    
    def get_stats(self) -> Dict:
        """获取索引统计信息"""
        return {
            'dim': self.dim,
            'space': self.space,
            'element_count': self.element_count,
            'max_elements': self.max_elements,
            'ef_construction': self.ef_construction,
            'M': self.M,
            'ef_search': self.ef_search,
        }


# ============================================
# Faiss 索引 (可选后端)
# ============================================

class FaissIndex:
    """
    Faiss 特征索引
    
    支持多种索引类型:
    - IVF: 倒排文件索引
    - PQ: 乘积量化
    - HNSW: HNSW 索引
    """
    
    def __init__(
        self,
        dim: int = 512,
        index_type: str = "IVF4096,PQ64",
        use_gpu: bool = True,
        nprobe: int = 32
    ):
        """
        Args:
            dim: 特征维度
            index_type: 索引类型字符串
            use_gpu: 是否使用 GPU
            nprobe: 搜索时探测的聚类数
        """
        try:
            import faiss
        except ImportError:
            raise ImportError("faiss is required for FaissIndex")
        
        self.dim = dim
        self.index_type = index_type
        self.use_gpu = use_gpu
        self.nprobe = nprobe
        self.faiss = faiss
        
        # 创建索引
        self.index = self._create_index(index_type)
        
        # GPU 资源
        if use_gpu:
            self.res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(self.res, 0, self.index)
        
        self.element_count = 0
        self.id_mapping: Dict[int, int] = {}
    
    def _create_index(self, index_type: str):
        """创建 Faiss 索引"""
        if index_type.startswith("IVF"):
            # IVF 索引
            nlist = int(index_type.split(',')[0].replace('IVF', ''))
            quantizer = self.faiss.IndexFlatL2(self.dim)
            index = self.faiss.IndexIVFFlat(quantizer, self.dim, nlist)
        elif index_type.startswith("HNSW"):
            # HNSW 索引
            index = self.faiss.IndexHNSWFlat(self.dim, 32)
        else:
            # 默认使用 Flat 索引
            index = self.faiss.IndexFlatL2(self.dim)
        
        return index
    
    def add(self, features: np.ndarray, ids: Optional[np.ndarray] = None) -> None:
        """添加特征"""
        N = features.shape[0]
        
        # 训练 IVF 索引 (如果需要)
        if isinstance(self.index, self.faiss.IndexIVFFlat) and not self.index.is_trained:
            self.index.train(features)
        
        # 添加
        self.index.add(features)
        
        # ID 映射
        if ids is not None:
            for i, ext_id in enumerate(ids):
                self.id_mapping[ext_id] = self.element_count + i
        
        self.element_count += N
    
    def search(
        self,
        query_features: np.ndarray,
        k: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        """搜索"""
        # 设置 nprobe
        if isinstance(self.index, self.faiss.IndexIVFFlat):
            self.index.nprobe = self.nprobe
        
        # 搜索
        distances, labels = self.index.search(query_features, k)
        
        # 转换 ID
        external_labels = np.zeros_like(labels)
        for i in range(len(labels)):
            for j in range(k):
                int_id = labels[i, j]
                external_labels[i, j] = self.id_mapping.get(int_id, int_id)
        
        return external_labels, distances


# ============================================
# 索引工厂
# ============================================

def build_index(
    index_type: str = "hnsw",
    **kwargs
) -> Union[HNSWIndex, FaissIndex]:
    """
    构建特征索引
    
    Args:
        index_type: 索引类型 ('hnsw', 'faiss')
        **kwargs: 配置参数
        
    Returns:
        索引实例
    """
    if index_type == "hnsw":
        return HNSWIndex(**kwargs)
    elif index_type == "faiss":
        return FaissIndex(**kwargs)
    else:
        raise ValueError(f"Unknown index type: {index_type}")


if __name__ == "__main__":
    # 测试 HNSW 索引
    print("Testing HNSW Index...")
    
    dim = 512
    index = HNSWIndex(dim=dim, max_elements=10000)
    
    # 添加随机特征
    N = 1000
    features = np.random.randn(N, dim).astype(np.float32)
    ids = np.arange(N)
    index.add(features, ids)
    
    print(f"Added {N} features")
    print(f"Index stats: {index.get_stats()}")
    
    # 搜索
    query = np.random.randn(1, dim).astype(np.float32)
    labels, similarities = index.search(query, k=5)
    
    print(f"\nSearch results:")
    print(f"  Labels: {labels[0]}")
    print(f"  Similarities: {similarities[0]}")
    
    # 保存/加载测试
    test_path = "/tmp/test_hnsw.index"
    index.save(test_path)
    print(f"\nSaved to {test_path}")
    
    loaded_index = HNSWIndex.load(test_path)
    print(f"Loaded index with {loaded_index.element_count} elements")
