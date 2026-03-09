#!/usr/bin/env python3
"""
人脸识别模型 - 端到端全面测试

测试自研的 DKGA-Det 检测模型和 DDFD-Rec 识别模型
使用 LFW 数据集进行 1:1 验证测试
"""

import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Dict
from collections import defaultdict

import numpy as np
import cv2
import torch
from tqdm import tqdm
from loguru import logger
from sklearn.metrics import roc_curve, auc, confusion_matrix

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 配置日志
logger.remove()
logger.add(sys.stdout, level="INFO", format="<green>{time:HH:mm:ss}</green> | <level>{message}</level>")

def load_models(device: str = 'cuda'):
    """加载检测器和识别器"""
    from inference.detector import Detector
    from inference.recognizer import Recognizer
    
    logger.info(f"加载模型到设备：{device}")
    
    # 加载检测器
    detector_checkpoint = "checkpoints/detection/best.pth"
    logger.info(f"加载检测模型：{detector_checkpoint}")
    detector = Detector(score_thresh=0.5, device=device)
    if Path(detector_checkpoint).exists():
        state_dict = torch.load(detector_checkpoint, map_location=device, weights_only=True)
        logger.info("检测模型权重加载成功")
    
    # 加载识别器
    recognizer_checkpoint = "checkpoints/recognition/best.pth"
    logger.info(f"加载识别模型：{recognizer_checkpoint}")
    recognizer = Recognizer(device=device)
    if Path(recognizer_checkpoint).exists():
        state_dict = torch.load(recognizer_checkpoint, map_location=device, weights_only=True)
        logger.info("识别模型权重加载成功")
    
    return detector, recognizer

def extract_lfw_samples(lfw_dir: str, max_people: int = 200):
    """从 LFW 提取测试样本"""
    lfw_path = Path(lfw_dir)
    
    if not lfw_path.exists():
        logger.error(f"LFW 目录不存在：{lfw_dir}")
        return None, None
    
    # 扫描所有人脸目录
    person_dirs = [d for d in lfw_path.iterdir() if d.is_dir()]
    logger.info(f"找到 {len(person_dirs)} 个人")
    
    if max_people:
        person_dirs = person_dirs[:max_people]
    
    # 收集每个人的图像
    person_images = defaultdict(list)
    for person_dir in tqdm(person_dirs, desc="扫描图像"):
        images = list(person_dir.glob("*.jpg"))
        if len(images) >= 2:
            person_images[person_dir.name] = images[:5]
    
    logger.info(f"{len(person_images)} 个人有 2 张以上图像")
    
    # 生成同人对
    same_person_pairs = []
    for name, images in person_images.items():
        if len(images) >= 2:
            for i in range(len(images)):
                for j in range(i + 1, len(images)):
                    same_person_pairs.append((str(images[i]), str(images[j])))
    
    # 生成异人对
    different_person_pairs = []
    names = list(person_images.keys())
    np.random.seed(42)
    
    n_pairs = min(len(same_person_pairs) * 3, 3000)
    for _ in range(n_pairs):
        i, j = np.random.choice(len(names), 2, replace=False)
        if len(person_images[names[i]]) >= 1 and len(person_images[names[j]]) >= 1:
            img1 = np.random.choice(person_images[names[i]])
            img2 = np.random.choice(person_images[names[j]])
            different_person_pairs.append((str(img1), str(img2)))
    
    logger.info(f"生成 {len(same_person_pairs)} 个同人对，{len(different_person_pairs)} 个异人对")
    
    return same_person_pairs, different_person_pairs

def run_lfw_test(detector, recognizer, same_pairs: List[Tuple], diff_pairs: List[Tuple]) -> Dict:
    """运行 LFW 测试"""
    logger.info("=" * 60)
    logger.info("开始 LFW 1:1 验证测试（检测 + 识别端到端）")
    logger.info("=" * 60)
    
    similarities = []
    labels = []
    
    all_pairs = [(p, 1) for p in same_pairs] + [(p, 0) for p in diff_pairs]
    np.random.shuffle(all_pairs)
    
    for (img1_path, img2_path), label in tqdm(all_pairs, desc="LFW 端到端测试"):
        try:
            img1 = cv2.imread(img1_path)
            img2 = cv2.imread(img2_path)
            
            if img1 is None or img2 is None:
                continue
            
            # 检测人脸
            det1 = detector(img1)
            det2 = detector(img2)
            
            # 检查是否检测到人脸
            if len(det1.get('boxes', [])) == 0 or len(det2.get('boxes', [])) == 0:
                continue
            
            # 使用置信度最高的人脸
            max_idx1 = np.argmax(det1['scores'])
            max_idx2 = np.argmax(det2['scores'])
            
            bbox1 = det1['boxes'][max_idx1]
            bbox2 = det2['boxes'][max_idx2]
            
            landmarks1 = det1.get('landmarks', [None])[max_idx1]
            landmarks2 = det2.get('landmarks', [None])[max_idx2]
            
            # 提取特征
            feat1 = recognizer.extract(img1, bbox1, landmarks1)
            feat2 = recognizer.extract(img2, bbox2, landmarks2)
            
            if feat1 is None or feat2 is None:
                continue
            
            # 计算余弦相似度
            feat1_norm = feat1 / (np.linalg.norm(feat1) + 1e-10)
            feat2_norm = feat2 / (np.linalg.norm(feat2) + 1e-10)
            sim = float(np.dot(feat1_norm, feat2_norm))
            
            similarities.append(sim)
            labels.append(label)
            
        except Exception as e:
            continue
    
    if len(similarities) < 10:
        logger.error(f"有效样本太少：{len(similarities)}")
        return {'error': 'not enough samples', 'n_samples': len(similarities)}
    
    logger.info(f"有效样本：{len(similarities)} (同人：{sum(labels)}, 异人：{len(labels)-sum(labels)})")
    
    # 计算指标
    similarities = np.array(similarities)
    labels = np.array(labels)
    
    # ROC 曲线
    fpr, tpr, thresholds = roc_curve(labels, similarities)
    fnmr = 1 - tpr
    roc_auc = auc(fpr, tpr)
    
    # EER
    diff = np.abs(fpr - fnmr)
    eer_idx = np.argmin(diff)
    eer = (fpr[eer_idx] + fnmr[eer_idx]) / 2
    
    # FNMR@FMR
    def fnmr_at_fmr(target_fmr):
        idx = np.argmin(np.abs(fpr - target_fmr))
        return float(fnmr[idx])
    
    fnmr_1e4 = fnmr_at_fmr(1e-4)
    fnmr_1e6 = fnmr_at_fmr(1e-6)
    
    # 最佳阈值
    youden_index = tpr - fpr
    optimal_idx = np.argmax(youden_index)
    optimal_threshold = float(thresholds[optimal_idx])
    
    # 准确率
    predictions = (similarities >= optimal_threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    
    results = {
        'total_pairs': len(similarities),
        'same_person_pairs': int(np.sum(labels)),
        'different_person_pairs': int(np.sum(labels == 0)),
        'detection_rate': len(similarities) / len(all_pairs),
        'auc': float(roc_auc),
        'eer': float(eer),
        'eer_threshold': float(thresholds[eer_idx]),
        'optimal_threshold': optimal_threshold,
        'fnmr_at_fmr_1e-4': float(fnmr_1e4),
        'fnmr_at_fmr_1e-6': float(fnmr_1e6),
        'accuracy': float(accuracy),
        'tp': int(tp),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'precision': float(tp / (tp + fp + 1e-10)),
        'recall': float(tp / (tp + fn + 1e-10))
    }
    
    logger.info("=" * 40)
    logger.info("LFW 测试结果")
    logger.info("=" * 40)
    logger.info(f"有效样本数：{results['total_pairs']}")
    logger.info(f"检测率：{results['detection_rate']:.2%}")
    logger.info(f"AUC: {results['auc']:.4f}")
    logger.info(f"EER: {results['eer']:.4f}")
    logger.info(f"最佳阈值：{results['optimal_threshold']:.4f}")
    logger.info(f"准确率：{results['accuracy']:.4f}")
    logger.info(f"FNMR@FMR=1e-4: {results['fnmr_at_fmr_1e-4']:.4f}")
    logger.info(f"FNMR@FMR=1e-6: {results['fnmr_at_fmr_1e-6']:.4f}")
    
    return results

def main():
    """主函数"""
    logger.info("=" * 80)
    logger.info("自研人脸识别模型 - 端到端全面测试")
    logger.info("=" * 80)
    
    # 配置
    lfw_dir = "datasets/lfw"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    logger.info(f"设备：{device}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # 检查文件
    if not Path(lfw_dir).exists():
        logger.error(f"LFW 目录不存在：{lfw_dir}")
        return
    
    # 加载模型
    detector, recognizer = load_models(device)
    
    # 提取样本
    logger.info("从 LFW 提取测试样本...")
    same_pairs, diff_pairs = extract_lfw_samples(lfw_dir, max_people=200)
    
    if same_pairs is None or len(same_pairs) == 0:
        logger.error("无法生成测试对")
        return
    
    # 运行测试
    start_time = time.time()
    results = run_lfw_test(detector, recognizer, same_pairs, diff_pairs)
    elapsed_time = time.time() - start_time
    
    results['test_time_seconds'] = elapsed_time
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"tests/benchmarks/results/lfw_full_test_{timestamp}.json"
    
    Path("tests/benchmarks/results").mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"结果已保存：{output_file}")
    
    # 打印最终摘要
    print("\n" + "=" * 80)
    print("测试结果摘要")
    print("=" * 80)
    
    if 'error' in results:
        print(f"错误：{results['error']}")
    else:
        print(f"测试样本数：{results.get('total_pairs', 'N/A')}")
        print(f"检测率：{results.get('detection_rate', 'N/A'):.2%}")
        auc_val = results.get('auc')
        print(f"AUC: {auc_val:.4f}" if auc_val is not None else "AUC: N/A")
        eer_val = results.get('eer')
        print(f"EER: {eer_val:.4f}" if eer_val is not None else "EER: N/A")
        acc_val = results.get('accuracy')
        print(f"准确率：{acc_val:.4f}" if acc_val is not None else "准确率：N/A")
        opt_thresh = results.get('optimal_threshold')
        print(f"最佳阈值：{opt_thresh:.4f}" if opt_thresh is not None else "最佳阈值：N/A")
        if results.get('fnmr_at_fmr_1e-4') is not None:
            print(f"FNMR@FMR=1e-4: {results['fnmr_at_fmr_1e-4']:.4f}")
        if results.get('fnmr_at_fmr_1e-6') is not None:
            print(f"FNMR@FMR=1e-6: {results['fnmr_at_fmr_1e-6']:.4f}")
    
    print(f"测试时间：{elapsed_time:.1f} 秒")
    print("=" * 80)

if __name__ == "__main__":
    main()
