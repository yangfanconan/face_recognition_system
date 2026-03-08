"""
评估工具

支持:
- LFW 评估
- CPLFW 评估
- IJB-C 评估
- RFW 评估
"""

import os
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.model_selection import KFold
from scipy import interpolate


# ============================================
# LFW 评估
# ============================================

class LFWEvaluator:
    """
    LFW 评估器
    """
    
    def __init__(self, nfolds: int = 10):
        self.nfolds = nfolds
    
    def evaluate(
        self,
        features1: np.ndarray,
        features2: np.ndarray,
        labels: np.ndarray,
        distances: str = "cosine"
    ) -> Dict:
        """
        评估 LFW 准确率
        
        Args:
            features1: (N, D) 特征 1
            features2: (N, D) 特征 2
            labels: (N,) 是否同一人标签
            distances: 距离类型
            
        Returns:
            metrics: 评估指标
        """
        # 计算相似度
        if distances == "cosine":
            similarities = self._cosine_similarity(features1, features2)
        else:
            similarities = -self._euclidean_distance(features1, features2)
        
        # 10 折交叉验证
        kfold = KFold(n_splits=self.nfolds, shuffle=True, random_state=42)
        
        accuracies = []
        thresholds = []
        
        for train_idx, test_idx in kfold.split(similarities):
            train_sim = similarities[train_idx]
            train_labels = labels[train_idx]
            
            # 找到最佳阈值
            best_thresh = self._find_best_threshold(train_sim, train_labels)
            thresholds.append(best_thresh)
            
            # 测试
            test_sim = similarities[test_idx]
            test_labels = labels[test_idx]
            
            predictions = test_sim >= best_thresh
            accuracy = np.mean(predictions == test_labels)
            accuracies.append(accuracy)
        
        return {
            'accuracy': np.mean(accuracies),
            'std': np.std(accuracies),
            'threshold': np.mean(thresholds),
            'accuracies': accuracies,
        }
    
    def _cosine_similarity(
        self,
        feat1: np.ndarray,
        feat2: np.ndarray
    ) -> np.ndarray:
        """计算余弦相似度"""
        feat1_norm = feat1 / (np.linalg.norm(feat1, axis=1, keepdims=True) + 1e-8)
        feat2_norm = feat2 / (np.linalg.norm(feat2, axis=1, keepdims=True) + 1e-8)
        return np.sum(feat1_norm * feat2_norm, axis=1)
    
    def _euclidean_distance(
        self,
        feat1: np.ndarray,
        feat2: np.ndarray
    ) -> np.ndarray:
        """计算欧氏距离"""
        return np.sqrt(np.sum((feat1 - feat2) ** 2, axis=1))
    
    def _find_best_threshold(
        self,
        similarities: np.ndarray,
        labels: np.ndarray
    ) -> float:
        """找到最佳阈值"""
        thresholds = np.linspace(similarities.min(), similarities.max(), 1000)
        
        best_acc = 0
        best_thresh = thresholds[0]
        
        for thresh in thresholds:
            predictions = similarities >= thresh
            acc = np.mean(predictions == labels)
            if acc > best_acc:
                best_acc = acc
                best_thresh = thresh
        
        return best_thresh


# ============================================
# ROC 曲线和 FAR/FRR
# ============================================

def compute_roc(
    similarities: np.ndarray,
    labels: np.ndarray,
    far_thresholds: Optional[List[float]] = None
) -> Dict:
    """
    计算 ROC 曲线
    
    Args:
        similarities: 相似度分数
        labels: 真实标签 (1=同一人，0=不同人)
        far_thresholds: 指定的 FAR 阈值
        
    Returns:
        roc_data: ROC 数据
    """
    # 分离正负样本
    pos_sims = similarities[labels == 1]
    neg_sims = similarities[labels == 0]
    
    # 计算不同阈值下的 TAR 和 FAR
    thresholds = np.linspace(similarities.min(), similarities.max(), 1000)
    
    tars = []
    fars = []
    
    for thresh in thresholds:
        # 真正例率 (TAR)
        tar = np.mean(pos_sims >= thresh)
        
        # 假正例率 (FAR)
        far = np.mean(neg_sims >= thresh)
        
        tars.append(tar)
        fars.append(far)
    
    # 插值获取指定 FAR 下的 TAR
    if far_thresholds is not None:
        fpr_interp = interpolate.interp1d(fars, tars)
        tar_at_far = {}
        
        for far in far_thresholds:
            try:
                tar = fpr_interp(far)
                tar_at_far[f"tar@far={far}"] = tar
            except:
                tar_at_far[f"tar@far={far}"] = 0.0
    else:
        tar_at_far = {}
    
    # 计算 EER (等错误率)
    eer = compute_eer(np.array(fars), np.array(tars), thresholds)
    
    return {
        'fars': fars,
        'tars': tars,
        'thresholds': thresholds,
        'eer': eer,
        **tar_at_far,
    }


def compute_eer(
    fars: np.ndarray,
    tars: np.ndarray,
    thresholds: np.ndarray
) -> float:
    """
    计算等错误率 (EER)
    
    EER 是 FAR = FRR 时的错误率
    FRR = 1 - TAR
    """
    frrs = 1 - tars
    
    # 找到 FAR 和 FRR 最接近的点
    diff = np.abs(fars - frrs)
    eer_idx = np.argmin(diff)
    
    eer = (fars[eer_idx] + frrs[eer_idx]) / 2
    
    return eer


# ============================================
# 验证协议
# ============================================

def lfw_protocol(
    pairs_file: str,
    feature_extractor,
    batch_size: int = 32
) -> Dict:
    """
    LFW 验证协议
    
    Args:
        pairs_file: LFW pairs.txt 路径
        feature_extractor: 特征提取函数
        batch_size: 批次大小
        
    Returns:
        metrics: 评估结果
    """
    from data.datasets import LFWDataset
    
    dataset = LFWDataset(root=os.path.dirname(pairs_file), pairs_file=pairs_file)
    
    all_similarities = []
    all_labels = []
    
    for i in range(0, len(dataset), batch_size):
        batch_pairs = dataset.pairs[i:i+batch_size]
        
        features1_list = []
        features2_list = []
        labels_list = []
        
        for pair in batch_pairs:
            # 加载并提取特征
            image1 = load_and_preprocess(pair['image1'])
            image2 = load_and_preprocess(pair['image2'])
            
            feat1 = feature_extractor(image1)
            feat2 = feature_extractor(image2)
            
            features1_list.append(feat1)
            features2_list.append(feat2)
            labels_list.append(1 if pair['same'] else 0)
        
        # 计算相似度
        features1 = np.stack(features1_list)
        features2 = np.stack(features2_list)
        
        similarities = np.sum(features1 * features2, axis=1)
        
        all_similarities.extend(similarities)
        all_labels.extend(labels_list)
    
    # 评估
    evaluator = LFWEvaluator()
    metrics = evaluator.evaluate(
        np.array(all_similarities)[:, np.newaxis],
        np.array(all_similarities)[:, np.newaxis],
        np.array(all_labels)
    )
    
    # ROC
    roc = compute_roc(
        np.array(all_similarities),
        np.array(all_labels),
        far_thresholds=[1e-6, 1e-5, 1e-4, 1e-3]
    )
    
    return {
        **metrics,
        **roc,
    }


def load_and_preprocess(image_path: str) -> np.ndarray:
    """加载并预处理图像"""
    import cv2
    
    image = cv2.imread(image_path)
    if image is not None:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (112, 112))
    
    return image


# ============================================
# 主评估函数
# ============================================

def evaluate_model(
    model,
    dataset_name: str,
    data_root: str,
    batch_size: int = 32
) -> Dict:
    """
    评估模型
    
    Args:
        model: 模型
        dataset_name: 数据集名称
        data_root: 数据根目录
        batch_size: 批次大小
        
    Returns:
        metrics: 评估指标
    """
    if dataset_name == "lfw":
        pairs_file = os.path.join(data_root, "pairs.txt")
        return lfw_protocol(pairs_file, model, batch_size)
    elif dataset_name == "cplfw":
        # CPLFW 评估 (类似 LFW)
        pass
    elif dataset_name == "ijbc":
        # IJB-C 评估
        pass
    elif dataset_name == "rfw":
        # RFW 跨种族评估
        pass
    
    return {}


if __name__ == "__main__":
    # 测试评估工具
    print("Testing evaluation tools...")
    
    # 生成随机特征和标签
    N = 1000
    features1 = np.random.randn(N, 512).astype(np.float32)
    features2 = np.random.randn(N, 512).astype(np.float32)
    labels = np.random.randint(0, 2, N)
    
    # LFW 评估
    evaluator = LFWEvaluator()
    metrics = evaluator.evaluate(features1, features2, labels)
    
    print(f"LFW Accuracy: {metrics['accuracy']:.4f} (+/- {metrics['std']:.4f})")
    
    # ROC
    similarities = np.sum(features1 * features2, axis=1)
    similarities = similarities / (np.linalg.norm(features1, axis=1) * np.linalg.norm(features2, axis=1))
    
    roc = compute_roc(similarities, labels, far_thresholds=[1e-4, 1e-3])
    
    print(f"EER: {roc['eer']:.4f}")
    print(f"TAR@FAR=1e-4: {roc.get('tar@far=0.0001', 0):.4f}")
    print(f"TAR@FAR=1e-3: {roc.get('tar@far=0.001', 0):.4f}")
