"""
模型评估脚本

支持:
- LFW 评估
- CPLFW 评估
- IJB-C 评估
- RFW 跨种族评估
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.model_selection import KFold
from scipy import interpolate

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.recognition import build_recognizer
from models.common import load_checkpoint, get_device


# ============================================
# 评估工具函数
# ============================================

def cosine_similarity(feat1: np.ndarray, feat2: np.ndarray) -> np.ndarray:
    """计算余弦相似度"""
    feat1_norm = feat1 / (np.linalg.norm(feat1, axis=1, keepdims=True) + 1e-8)
    feat2_norm = feat2 / (np.linalg.norm(feat2, axis=1, keepdims=True) + 1e-8)
    return np.sum(feat1_norm * feat2_norm, axis=1)


def find_best_threshold(similarities: np.ndarray, labels: np.ndarray) -> float:
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


def compute_accuracy(similarities: np.ndarray, labels: np.ndarray, nfolds: int = 10) -> Dict:
    """
    计算 10 折交叉验证准确率
    
    Args:
        similarities: 相似度分数
        labels: 真实标签 (1=同一人，0=不同人)
        nfolds: 折数
        
    Returns:
        metrics: 评估指标
    """
    kfold = KFold(n_splits=nfolds, shuffle=True, random_state=42)
    
    accuracies = []
    thresholds = []
    
    for train_idx, test_idx in kfold.split(similarities):
        train_sim = similarities[train_idx]
        train_labels = labels[train_idx]
        
        # 找到最佳阈值
        best_thresh = find_best_threshold(train_sim, train_labels)
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


def compute_roc(similarities: np.ndarray, labels: np.ndarray, 
                far_thresholds: Optional[List[float]] = None) -> Dict:
    """
    计算 ROC 曲线
    
    Returns:
        roc_data: ROC 数据
    """
    pos_sims = similarities[labels == 1]
    neg_sims = similarities[labels == 0]
    
    thresholds = np.linspace(similarities.min(), similarities.max(), 1000)
    
    tars = []
    fars = []
    
    for thresh in thresholds:
        tar = np.mean(pos_sims >= thresh) if len(pos_sims) > 0 else 0
        far = np.mean(neg_sims >= thresh) if len(neg_sims) > 0 else 0
        tars.append(tar)
        fars.append(far)
    
    result = {
        'fars': fars,
        'tars': tars,
        'thresholds': thresholds,
    }
    
    # 计算指定 FAR 下的 TAR
    if far_thresholds:
        fpr_interp = interpolate.interp1d(fars, tars)
        for far in far_thresholds:
            try:
                tar = fpr_interp(far)
                result[f'tar@far={far}'] = tar
            except:
                result[f'tar@far={far}'] = 0.0
    
    # 计算 EER
    fars_arr = np.array(fars)
    tars_arr = np.array(tars)
    frrs = 1 - tars_arr
    diff = np.abs(fars_arr - frrs)
    eer_idx = np.argmin(diff)
    result['eer'] = (fars_arr[eer_idx] + frrs[eer_idx]) / 2
    
    return result


# ============================================
# LFW 评估
# ============================================

class LFWEvaluator:
    """LFW 评估器"""
    
    def __init__(self, root: str, model, device: torch.device, batch_size: int = 32):
        self.root = root
        self.model = model
        self.device = device
        self.batch_size = batch_size
        
        self.pairs_file = os.path.join(root, 'pairs.txt')
        self.pairs = self._load_pairs()
    
    def _load_pairs(self) -> List[Dict]:
        """加载 pairs.txt"""
        pairs = []
        
        with open(self.pairs_file, 'r') as f:
            lines = f.readlines()
        
        # 跳过第一行 (N)
        for line in lines[1:]:
            parts = line.strip().split()
            if len(parts) == 3:
                # 同一人
                name, img1, img2 = parts
                pairs.append({
                    'same': True,
                    'image1': os.path.join(self.root, name, f"{name}_{img1}.jpg"),
                    'image2': os.path.join(self.root, name, f"{name}_{img2}.jpg"),
                })
            else:
                # 不同人
                name1, img1, name2, img2 = parts
                pairs.append({
                    'same': False,
                    'image1': os.path.join(self.root, name1, f"{name1}_{img1}.jpg"),
                    'image2': os.path.join(self.root, name2, f"{name2}_{img2}.jpg"),
                })
        
        return pairs
    
    def extract_feature(self, image_path: str) -> Optional[np.ndarray]:
        """提取单张图像特征"""
        import cv2
        
        image = cv2.imread(image_path)
        if image is None:
            return None
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (112, 112))
        
        # 归一化
        image = image.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = (image - mean) / std
        
        # 转为 tensor
        image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        
        # 提取特征
        with torch.no_grad():
            feature = self.model.get_identity_feature(image_tensor)
        
        return feature.cpu().numpy()[0]
    
    def evaluate(self) -> Dict:
        """执行 LFW 评估"""
        logging.info(f"Evaluating on LFW with {len(self.pairs)} pairs...")
        
        all_similarities = []
        all_labels = []
        
        for i in range(0, len(self.pairs), self.batch_size):
            batch_pairs = self.pairs[i:i+self.batch_size]
            
            features1 = []
            features2 = []
            labels = []
            
            for pair in batch_pairs:
                feat1 = self.extract_feature(pair['image1'])
                feat2 = self.extract_feature(pair['image2'])
                
                if feat1 is not None and feat2 is not None:
                    features1.append(feat1)
                    features2.append(feat2)
                    labels.append(1 if pair['same'] else 0)
            
            if len(features1) > 0:
                features1 = np.stack(features1)
                features2 = np.stack(features2)
                
                similarities = cosine_similarity(features1, features2)
                all_similarities.extend(similarities)
                all_labels.extend(labels)
        
        all_similarities = np.array(all_similarities)
        all_labels = np.array(all_labels)
        
        # 计算准确率
        acc_metrics = compute_accuracy(all_similarities, all_labels)
        
        # 计算 ROC
        roc_metrics = compute_roc(
            all_similarities, all_labels,
            far_thresholds=[1e-6, 1e-5, 1e-4, 1e-3]
        )
        
        return {
            **acc_metrics,
            **roc_metrics,
            'num_pairs': len(all_labels),
        }


# ============================================
# CPLFW 评估
# ============================================

class CPLFWEvaluator:
    """CPLFW (跨姿态 LFW) 评估器"""
    
    def __init__(self, root: str, model, device: torch.device, batch_size: int = 32):
        self.root = root
        self.model = model
        self.device = device
        self.batch_size = batch_size
        
        self.pairs_file = os.path.join(root, 'cplfw_pair.txt')
        self.pairs = self._load_pairs()
    
    def _load_pairs(self) -> List[Dict]:
        """加载 CPLFW pairs"""
        pairs = []
        
        if not os.path.exists(self.pairs_file):
            logging.warning(f"CPLFW pairs file not found: {self.pairs_file}")
            return []
        
        with open(self.pairs_file, 'r') as f:
            lines = f.readlines()
        
        for line in lines[1:]:
            parts = line.strip().split(',')
            if len(parts) >= 4:
                pairs.append({
                    'same': parts[0] == parts[2],
                    'image1': os.path.join(self.root, parts[0], parts[1]),
                    'image2': os.path.join(self.root, parts[2], parts[3]),
                })
        
        return pairs
    
    def extract_feature(self, image_path: str) -> Optional[np.ndarray]:
        """提取特征 (同 LFW)"""
        import cv2
        
        image = cv2.imread(image_path)
        if image is None:
            return None
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (112, 112))
        image = image.astype(np.float32) / 255.0
        
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = (image - mean) / std
        
        image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            feature = self.model.get_identity_feature(image_tensor)
        
        return feature.cpu().numpy()[0]
    
    def evaluate(self) -> Dict:
        """执行 CPLFW 评估"""
        if len(self.pairs) == 0:
            return {'accuracy': 0, 'error': 'No pairs found'}
        
        logging.info(f"Evaluating on CPLFW with {len(self.pairs)} pairs...")
        
        all_similarities = []
        all_labels = []
        
        for pair in self.pairs:
            feat1 = self.extract_feature(pair['image1'])
            feat2 = self.extract_feature(pair['image2'])
            
            if feat1 is not None and feat2 is not None:
                sim = cosine_similarity(feat1.reshape(1, -1), feat2.reshape(1, -1))[0]
                all_similarities.append(sim)
                all_labels.append(1 if pair['same'] else 0)
        
        all_similarities = np.array(all_similarities)
        all_labels = np.array(all_labels)
        
        acc_metrics = compute_accuracy(all_similarities, all_labels)
        roc_metrics = compute_roc(all_similarities, all_labels)
        
        return {
            **acc_metrics,
            **roc_metrics,
            'num_pairs': len(all_labels),
        }


# ============================================
# 主评估函数
# ============================================

def evaluate_model(
    checkpoint_path: str,
    dataset_name: str,
    data_root: str,
    batch_size: int = 32,
    device: Optional[str] = None
) -> Dict:
    """
    评估模型
    
    Args:
        checkpoint_path: 模型权重路径
        dataset_name: 数据集名称 (lfw/cplfw/ijbc)
        data_root: 数据根目录
        batch_size: 批次大小
        device: 计算设备
        
    Returns:
        metrics: 评估结果
    """
    # 加载模型
    logging.info(f"Loading model from: {checkpoint_path}")
    
    model = build_recognizer(model_type="ddfd_rec")
    
    if os.path.exists(checkpoint_path):
        device = get_device(device)
        state_dict = torch.load(checkpoint_path, map_location=device)
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        model.load_state_dict(state_dict, strict=False)
        logging.info("Model loaded successfully")
    else:
        logging.error(f"Checkpoint not found: {checkpoint_path}")
        return {'error': 'Checkpoint not found'}
    
    model.to(device)
    model.eval()
    
    # 选择评估器
    if dataset_name == 'lfw':
        evaluator = LFWEvaluator(data_root, model, device, batch_size)
    elif dataset_name == 'cplfw':
        evaluator = CPLFWEvaluator(data_root, model, device, batch_size)
    else:
        logging.error(f"Unknown dataset: {dataset_name}")
        return {'error': f'Unknown dataset: {dataset_name}'}
    
    # 执行评估
    metrics = evaluator.evaluate()
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate face recognition model")
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Model checkpoint path"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=['lfw', 'cplfw', 'ijbc', 'rfw'],
        default='lfw',
        help="Evaluation dataset"
    )
    parser.add_argument(
        "--data-root",
        type=str,
        required=True,
        help="Dataset root directory"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (cuda/cpu)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for results"
    )
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # 执行评估
    metrics = evaluate_model(
        checkpoint_path=args.checkpoint,
        dataset_name=args.dataset,
        data_root=args.data_root,
        batch_size=args.batch_size,
        device=args.device
    )
    
    # 打印结果
    print("\n" + "=" * 60)
    print(f"评估结果 - {args.dataset.upper()}")
    print("=" * 60)
    
    if 'error' in metrics:
        print(f"错误：{metrics['error']}")
    else:
        print(f"准确率：{metrics['accuracy']*100:.2f}% (+/- {metrics['std']*100:.2f}%)")
        print(f"最佳阈值：{metrics['threshold']:.4f}")
        print(f"EER: {metrics['eer']*100:.2f}%")
        
        for far in [1e-6, 1e-5, 1e-4, 1e-3]:
            key = f'tar@far={far}'
            if key in metrics:
                print(f"TAR@FAR={far}: {metrics[key]*100:.2f}%")
    
    print("=" * 60)
    
    # 保存结果
    if args.output:
        import json
        with open(args.output, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)
        logging.info(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
