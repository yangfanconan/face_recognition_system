#!/usr/bin/env python3
"""
NIST 标准指标计算模块

实现以下指标计算：
- FRTE: 1:1 验证 (FNMR@FMR), 1:N 识别 (Top-N 准确率)
- FIVE: 视频流处理指标
- FMD: 变形人脸检测指标
- 通用指标：AP, mAP, ROC, DET 曲线等
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field
from collections import defaultdict

import numpy as np
from scipy import interpolate
from scipy.optimize import brentq
from sklearn.metrics import (
    roc_curve, 
    roc_auc_score, 
    precision_recall_curve, 
    average_precision_score,
    confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger

# 配置日志
logger.remove()
logger.add(sys.stdout, level="INFO", format="<green>{time:HH:mm:ss}</green> | <level>{message}</level>")


@dataclass
class MetricResult:
    """指标结果"""
    name: str
    value: float
    confidence_interval: Optional[Tuple[float, float]] = None
    additional_info: Dict = field(default_factory=dict)


class NISTMetrics:
    """
    NIST FRTE 标准指标计算器
    
    参考：NIST FRTE Evaluation Protocol
    https://pages.nist.gov/frte/
    """
    
    def __init__(self):
        self.similarities = []
        self.labels = []
        
    def add_scores(self, similarities: List[float], labels: List[int]):
        """
        添加比对分数
        
        Args:
            similarities: 相似度分数列表
            labels: 标签列表 (1=正样本对，0=负样本对)
        """
        self.similarities.extend(similarities)
        self.labels.extend(labels)
        
    def compute_fnmr_at_fmr(
        self, 
        fmr_target: float = 1e-4
    ) -> Tuple[float, float]:
        """
        计算指定 FMR 下的 FNMR
        
        FRTE 核心指标：False Non-Match Rate @ False Match Rate
        
        Args:
            fmr_target: 目标 FMR (如 1e-4, 1e-6)
            
        Returns:
            (threshold, fnmr)
        """
        similarities = np.array(self.similarities)
        labels = np.array(self.labels)
        
        # 正负样本
        pos_scores = similarities[labels == 1]  # 同类对
        neg_scores = similarities[labels == 0]  # 异类对
        
        if len(pos_scores) == 0 or len(neg_scores) == 0:
            logger.warning("正样本或负样本为空")
            return None, None
            
        # 计算不同阈值下的 FMR 和 FNMR
        thresholds = np.linspace(similarities.min(), similarities.max(), 10000)
        
        fnmr_values = []
        fmr_values = []
        
        for thresh in thresholds:
            # FMR: 负样本被错误接受的比例
            fmr = np.mean(neg_scores >= thresh)
            
            # FNMR: 正样本被错误拒绝的比例
            fnmr = np.mean(pos_scores < thresh)
            
            fmr_values.append(fmr)
            fnmr_values.append(fnmr)
            
        fmr_values = np.array(fmr_values)
        fnmr_values = np.array(fnmr_values)
        
        # 找到最接近目标 FMR 的阈值
        idx = np.argmin(np.abs(fmr_values - fmr_target))
        threshold = thresholds[idx]
        fnmr = fnmr_values[idx]
        actual_fmr = fmr_values[idx]
        
        logger.info(f"FMR={fmr_target:.2e}: Threshold={threshold:.4f}, FNMR={fnmr:.4f}, Actual FMR={actual_fmr:.2e}")
        
        return float(threshold), float(fnmr)
    
    def compute_roc(
        self, 
        fmr_range: Tuple[float, float] = (1e-6, 1e-1)
    ) -> Dict:
        """
        计算 ROC 曲线数据
        
        Returns:
            {
                'fmr': List[float],
                'fnmr': List[float],
                'thresholds': List[float],
                'auc': float
            }
        """
        similarities = np.array(self.similarities)
        labels = np.array(self.labels)
        
        # 使用 sklearn 计算 ROC
        # 注意：sklearn 的 FPR = FMR, TPR = 1 - FNMR
        fpr, tpr, thresholds = roc_curve(labels, similarities)
        fnmr = 1 - tpr
        
        # 计算 AUC
        auc = roc_auc_score(labels, similarities)
        
        # 筛选指定 FMR 范围
        mask = (fpr >= fmr_range[0]) & (fpr <= fmr_range[1])
        
        return {
            'fmr': fpr[mask].tolist(),
            'fnmr': fnmr[mask].tolist(),
            'thresholds': thresholds[mask].tolist(),
            'auc': float(auc),
            'full_fmr': fpr.tolist(),
            'full_fnmr': fnmr.tolist(),
            'full_thresholds': thresholds.tolist()
        }
    
    def compute_det(self) -> Dict:
        """
        计算 DET 曲线 (Detection Error Tradeoff)
        
        Returns:
            {
                'fmr': List[float],
                'fnmr': List[float],
                'thresholds': List[float],
                'eer': float,  # 等错误率
                'eer_threshold': float
            }
        """
        similarities = np.array(self.similarities)
        labels = np.array(self.labels)
        
        pos_scores = similarities[labels == 1]
        neg_scores = similarities[labels == 0]
        
        thresholds = np.linspace(similarities.min(), similarities.max(), 10000)
        
        fmr_values = []
        fnmr_values = []
        
        for thresh in thresholds:
            fmr = np.mean(neg_scores >= thresh)
            fnmr = np.mean(pos_scores < thresh)
            fmr_values.append(fmr)
            fnmr_values.append(fnmr)
            
        fmr_values = np.array(fmr_values)
        fnmr_values = np.array(fnmr_values)
        
        # 计算 EER (FMR = FNMR 的点)
        diff = np.abs(fmr_values - fnmr_values)
        eer_idx = np.argmin(diff)
        eer = (fmr_values[eer_idx] + fnmr_values[eer_idx]) / 2
        eer_threshold = thresholds[eer_idx]
        
        return {
            'fmr': fmr_values.tolist(),
            'fnmr': fnmr_values.tolist(),
            'thresholds': thresholds.tolist(),
            'eer': float(eer),
            'eer_threshold': float(eer_threshold)
        }
    
    def compute_all_metrics(self, fmr_targets: List[float] = None) -> Dict:
        """
        计算所有 FRTE 指标
        
        Args:
            fmr_targets: 目标 FMR 列表 [1e-4, 1e-6]
            
        Returns:
            包含所有指标的字典
        """
        if fmr_targets is None:
            fmr_targets = [1e-4, 1e-6]
            
        results = {
            'metric_type': 'FRTE',
            'total_pairs': len(self.labels),
            'positive_pairs': int(np.sum(self.labels)),
            'negative_pairs': int(np.sum(np.array(self.labels) == 0)),
        }
        
        # FNM R@FMR
        fnmr_results = {}
        for fmr in fmr_targets:
            _, fnmr = self.compute_fnmr_at_fmr(fmr)
            fnmr_results[f"FNM R@FMR={fmr:.0e}"] = fnmr
        results.update(fnmr_results)
        
        # ROC 和 AUC
        roc_data = self.compute_roc()
        results['auc'] = roc_data['auc']
        results['roc'] = roc_data
        
        # DET 和 EER
        det_data = self.compute_det()
        results['eer'] = det_data['eer']
        results['eer_threshold'] = det_data['eer_threshold']
        results['det'] = det_data
        
        return results


class DetectionMetrics:
    """人脸检测指标计算器"""
    
    def __init__(self, iou_threshold: float = 0.5):
        self.iou_threshold = iou_threshold
        self.predictions = []  # [(bbox, score, img_id), ...]
        self.ground_truths = defaultdict(list)  # {img_id: [bbox, ...]}
        
    def add_prediction(self, bbox: List[float], score: float, img_id: str):
        """添加预测结果"""
        self.predictions.append({
            'bbox': bbox,
            'score': score,
            'img_id': img_id
        })
        
    def add_ground_truth(self, bbox: List[float], img_id: str):
        """添加真实标注"""
        self.ground_truths[img_id].append(bbox)
        
    def compute_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        """计算 IoU"""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        
        bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        
        union_area = bbox1_area + bbox2_area - inter_area
        
        if union_area == 0:
            return 0
            
        return inter_area / union_area
    
    def compute_ap(self) -> float:
        """
        计算平均精度 (Average Precision)
        
        参考 Pascal VOC 评估方法
        """
        # 按置信度排序
        self.predictions.sort(key=lambda x: x['score'], reverse=True)
        
        # 统计每个预测的 TP/FP
        n_gt = sum(len(bboxes) for bboxes in self.ground_truths.values())
        gt_matched = defaultdict([False] * len)
        
        tp = np.zeros(len(self.predictions))
        fp = np.zeros(len(self.predictions))
        
        for i, pred in enumerate(self.predictions):
            img_id = pred['img_id']
            pred_bbox = pred['bbox']
            
            if img_id not in self.ground_truths:
                fp[i] = 1
                continue
                
            # 找到最佳匹配
            best_iou = 0
            best_j = -1
            
            for j, gt_bbox in enumerate(self.ground_truths[img_id]):
                if gt_matched[img_id][j]:
                    continue
                    
                iou = self.compute_iou(pred_bbox, gt_bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_j = j
                    
            if best_iou >= self.iou_threshold and best_j >= 0:
                tp[i] = 1
                gt_matched[img_id][best_j] = True
            else:
                fp[i] = 1
                
        # 计算精度 - 召回率曲线
        cum_tp = np.cumsum(tp)
        cum_fp = np.cumsum(fp)
        
        precision = cum_tp / (cum_tp + cum_fp + 1e-10)
        recall = cum_tp / (n_gt + 1e-10)
        
        # 计算 AP (11 点插值法)
        ap = 0.0
        recall_thresholds = np.linspace(0, 1, 11)
        
        for rt in recall_thresholds:
            p = precision[recall >= rt]
            if len(p) > 0:
                ap += np.max(p) / 11.0
                
        return float(ap)
    
    def compute_map(self, iou_thresholds: List[float] = None) -> Dict:
        """
        计算 mAP (mean Average Precision)
        
        参考 COCO 评估方法
        """
        if iou_thresholds is None:
            iou_thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
            
        ap_values = []
        
        for iou_thresh in iou_thresholds:
            self.iou_threshold = iou_thresh
            ap = self.compute_ap()
            ap_values.append(ap)
            
        return {
            'mAP': float(np.mean(ap_values)),
            'AP_50': ap_values[0],
            'AP_75': ap_values[5] if len(ap_values) > 5 else None,
            'AP_95': ap_values[-1],
            'ap_by_iou': dict(zip(iou_thresholds, ap_values))
        }
    
    def compute_recall_precision(self, threshold: float = 0.5) -> Dict:
        """
        计算召回率和精确率
        """
        n_gt = sum(len(bboxes) for bboxes in self.ground_truths.values())
        
        # 过滤低置信度预测
        valid_preds = [p for p in self.predictions if p['score'] >= threshold]
        
        tp = 0
        fp = 0
        gt_matched = defaultdict(lambda: [False] * 1000)
        
        for pred in valid_preds:
            img_id = pred['img_id']
            pred_bbox = pred['bbox']
            
            if img_id not in self.ground_truths:
                fp += 1
                continue
                
            matched = False
            for j, gt_bbox in enumerate(self.ground_truths[img_id]):
                if gt_matched[img_id][j]:
                    continue
                    
                if self.compute_iou(pred_bbox, gt_bbox) >= self.iou_threshold:
                    tp += 1
                    gt_matched[img_id][j] = True
                    matched = True
                    break
                    
            if not matched:
                fp += 1
                
        fn = n_gt - tp
        
        precision = tp / (tp + fp + 1e-10)
        recall = tp / (n_gt + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)
        
        return {
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'n_gt': n_gt
        }


class RecognitionMetrics:
    """人脸识别指标计算器"""
    
    def __init__(self):
        self.nist_metrics = NISTMetrics()
        
    def add_verification_scores(self, similarities: List[float], labels: List[int]):
        """添加验证分数"""
        self.nist_metrics.add_scores(similarities, labels)
        
    def compute_verification_metrics(self) -> Dict:
        """计算验证指标"""
        return self.nist_metrics.compute_all_metrics()
    
    def compute_identification_metrics(
        self,
        gallery_features: np.ndarray,
        gallery_labels: List[str],
        probe_features: np.ndarray,
        probe_labels: List[str],
        top_k: List[int] = [1, 5, 10]
    ) -> Dict:
        """
        计算 1:N 识别指标
        
        Args:
            gallery_features: 库特征 (N, D)
            gallery_labels: 库标签
            probe_features: 查询特征 (M, D)
            probe_labels: 查询标签
            top_k: Top-K 准确率
            
        Returns:
            {
                'top1_accuracy': float,
                'top5_accuracy': float,
                'rank': List[float]
            }
        """
        # 归一化特征
        gallery_features = gallery_features / (np.linalg.norm(gallery_features, axis=1, keepdims=True) + 1e-10)
        probe_features = probe_features / (np.linalg.norm(probe_features, axis=1, keepdims=True) + 1e-10)
        
        # 计算相似度矩阵
        similarity_matrix = probe_features @ gallery_features.T
        
        # 计算 Rank-K 准确率
        n_probes = len(probe_labels)
        correct_at_k = defaultdict(int)
        
        for i in range(n_probes):
            probe_label = probe_labels[i]
            scores = similarity_matrix[i]
            
            # 按相似度排序
            sorted_indices = np.argsort(scores)[::-1]
            
            # 检查前 K 个是否包含正确标签
            for k in top_k:
                top_k_labels = [gallery_labels[j] for j in sorted_indices[:k]]
                if probe_label in top_k_labels:
                    correct_at_k[k] += 1
                    
        results = {
            'n_probes': n_probes,
            'n_gallery': len(gallery_labels),
        }
        
        for k in top_k:
            results[f'top{k}_accuracy'] = correct_at_k[k] / n_probes
            
        return results
    
    def plot_roc_curve(
        self, 
        save_path: str = "roc_curve.png",
        title: str = "ROC Curve"
    ):
        """绘制 ROC 曲线"""
        roc_data = self.nist_metrics.compute_roc()
        
        plt.figure(figsize=(10, 8))
        
        # 绘制 ROC
        plt.plot(
            roc_data['full_fmr'], 
            roc_data['full_fnmr'],
            label=f"AUC = {roc_data['auc']:.4f}",
            linewidth=2
        )
        
        # 对数坐标
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('False Match Rate (FMR)')
        plt.ylabel('False Non-Match Rate (FNMR)')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"ROC 曲线已保存：{save_path}")
        
    def plot_det_curve(
        self,
        save_path: str = "det_curve.png",
        title: str = "DET Curve"
    ):
        """绘制 DET 曲线"""
        det_data = self.nist_metrics.compute_det()
        
        plt.figure(figsize=(10, 8))
        
        plt.plot(
            det_data['fmr'],
            det_data['fnmr'],
            label=f"EER = {det_data['eer']:.4f}",
            linewidth=2
        )
        
        # 标记 EER 点
        plt.scatter(
            [det_data['eer']], 
            [det_data['eer']],
            color='red',
            s=100,
            zorder=5,
            label=f"EER Threshold = {det_data['eer_threshold']:.4f}"
        )
        
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('False Match Rate (FMR)')
        plt.ylabel('False Non-Match Rate (FNMR)')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"DET 曲线已保存：{save_path}")


class VideoMetrics:
    """
    FIVE (Face-in-Video-Evaluation) 指标计算器
    """
    
    def __init__(self):
        self.frame_results = []
        
    def add_frame_result(
        self,
        frame_id: int,
        detected_faces: int,
        tracked_faces: int,
        recognized_faces: int,
        processing_time: float
    ):
        """添加帧结果"""
        self.frame_results.append({
            'frame_id': frame_id,
            'detected_faces': detected_faces,
            'tracked_faces': tracked_faces,
            'recognized_faces': recognized_faces,
            'processing_time': processing_time
        })
        
    def compute_metrics(self) -> Dict:
        """计算视频流指标"""
        if not self.frame_results:
            return {}
            
        n_frames = len(self.frame_results)
        total_time = sum(r['processing_time'] for r in self.frame_results)
        
        # FPS
        fps = n_frames / total_time if total_time > 0 else 0
        
        # 平均检测/跟踪/识别人脸数
        avg_detected = np.mean([r['detected_faces'] for r in self.frame_results])
        avg_tracked = np.mean([r['tracked_faces'] for r in self.frame_results])
        avg_recognized = np.mean([r['recognized_faces'] for r in self.frame_results])
        
        # 跟踪率
        track_rate = avg_tracked / (avg_detected + 1e-10)
        
        # 识别率
        recognize_rate = avg_recognized / (avg_detected + 1e-10)
        
        return {
            'total_frames': n_frames,
            'total_time_seconds': total_time,
            'fps': fps,
            'avg_detected_faces': avg_detected,
            'avg_tracked_faces': avg_tracked,
            'avg_recognized_faces': avg_recognized,
            'track_rate': track_rate,
            'recognize_rate': recognize_rate
        }


class MorphingDetectionMetrics:
    """
    FMD (Facial Morphing Detection) 指标计算器
    """
    
    def __init__(self):
        self.scores = []
        self.labels = []  # 1=real, 0=morphed
        
    def add_scores(self, scores: List[float], labels: List[int]):
        """
        添加检测分数
        
        Args:
            scores: 真实人脸概率
            labels: 1=真实，0=变形
        """
        self.scores.extend(scores)
        self.labels.extend(labels)
        
    def compute_metrics(self) -> Dict:
        """计算变形人脸检测指标"""
        scores = np.array(self.scores)
        labels = np.array(self.labels)
        
        # AP (Average Precision)
        ap = average_precision_score(labels, scores)
        
        # AUC
        auc = roc_auc_score(labels, scores)
        
        # 最佳阈值
        precision, recall, thresholds = precision_recall_curve(labels, scores)
        f1_scores = 2 * precision * recall / (precision + recall + 1e-10)
        best_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
        
        # 在最佳阈值下的指标
        predictions = (scores >= best_threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
        
        fpr = fp / (fp + tn + 1e-10)  # 误检率
        tpr = tp / (tp + fn + 1e-10)  # 检测率
        
        return {
            'ap': float(ap),
            'auc': float(auc),
            'best_threshold': float(best_threshold),
            'fpr': float(fpr),  # False Positive Rate
            'tpr': float(tpr),  # True Positive Rate
            'tp': int(tp),
            'fp': int(fp),
            'tn': int(tn),
            'fn': int(fn)
        }


def main():
    """测试指标计算模块"""
    import argparse
    
    parser = argparse.ArgumentParser(description="NIST 指标计算测试")
    parser.add_argument("--test", action="store_true", help="运行测试")
    args = parser.parse_args()
    
    if args.test:
        # 测试 FRTE 指标
        print("\n=== 测试 FRTE 指标 ===")
        nist = NISTMetrics()
        
        # 生成模拟数据
        np.random.seed(42)
        pos_scores = np.random.normal(0.8, 0.1, 1000)  # 正样本对
        neg_scores = np.random.normal(0.3, 0.15, 1000)  # 负样本对
        
        nist.add_scores(
            pos_scores.tolist() + neg_scores.tolist(),
            [1] * 1000 + [0] * 1000
        )
        
        results = nist.compute_all_metrics()
        print(f"AUC: {results['auc']:.4f}")
        print(f"EER: {results['eer']:.4f}")
        print(f"FNMR@FMR=1e-4: {results.get('FNMR@FMR=1e-4', 'N/A')}")
        print(f"FNMR@FMR=1e-6: {results.get('FNMR@FMR=1e-6', 'N/A')}")
        
        # 测试检测指标
        print("\n=== 测试检测指标 ===")
        det = DetectionMetrics()
        
        # 添加模拟数据
        det.add_ground_truth([10, 10, 100, 100], "img1")
        det.add_ground_truth([200, 200, 300, 300], "img1")
        
        det.add_prediction([12, 12, 102, 102], 0.95, "img1")
        det.add_prediction([205, 205, 305, 305], 0.9, "img1")
        det.add_prediction([400, 400, 500, 500], 0.8, "img1")  # FP
        
        ap = det.compute_ap()
        print(f"AP@IoU=0.5: {ap:.4f}")
        
        map_results = det.compute_map()
        print(f"mAP: {map_results['mAP']:.4f}")
        
        print("\n测试完成!")


if __name__ == "__main__":
    main()
