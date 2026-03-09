#!/usr/bin/env python3
"""
人脸识别模型 - 全自动全面测试脚本

测试自研的 DKGA-Det 检测模型和 DDFD-Rec 识别模型

使用方法:
    python tests/benchmarks/full_test.py
"""

import os
import sys
import json
import time
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import cv2
import torch
import yaml
from tqdm import tqdm
from loguru import logger

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 导入测试框架
from tests.benchmarks.metrics import NISTMetrics, DetectionMetrics, RecognitionMetrics
from tests.benchmarks.datasets import DataPreprocessor, LFWDataset
from tests.benchmarks.reports.report_generator import ReportGenerator

# ============================================
# 配置
# ============================================

CONFIG = {
    'model': {
        'detector': {
            'checkpoint': 'checkpoints/detection/best.pth',
            'input_size': [640, 640],
            'score_thresh': 0.5,
            'nms_thresh': 0.45,
            'device': 'cuda'
        },
        'recognizer': {
            'checkpoint': 'checkpoints/recognition/best.pth',
            'input_size': [112, 112],
            'feature_dim': 409,  # 根据实际模型调整
            'device': 'cuda'
        },
        'matcher': {
            'threshold': 0.6,
            'metric': 'cosine'
        }
    },
    'test': {
        'lfw': {
            'enabled': True,
            'data_dir': 'datasets/lfw',
            'max_samples': None  # None=全部测试
        },
        'detection': {
            'enabled': True,
            'data_dir': 'datasets/widerface',
            'max_samples': 100  # 检测测试样本数
        }
    },
    'output': {
        'results_dir': 'tests/benchmarks/results/full_test',
        'reports_dir': 'tests/benchmarks/reports/full_test',
        'logs_dir': 'tests/benchmarks/logs'
    }
}

# ============================================
# 日志配置
# ============================================

def setup_logging():
    """配置日志"""
    log_dir = Path(CONFIG['output']['logs_dir'])
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"full_test_{timestamp}.log"
    
    logger.remove()
    logger.add(sys.stdout, level="INFO", 
               format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>")
    logger.add(log_file, level="DEBUG", rotation="100 MB", retention="7 days")
    
    return timestamp

# ============================================
# 模型加载
# ============================================

def load_detector(checkpoint_path: str, device: str = 'cuda'):
    """加载人脸检测模型"""
    from inference.detector import Detector
    
    logger.info(f"加载检测模型：{checkpoint_path}")
    detector = Detector(score_thresh=0.5, device=device)
    
    # 加载自定义权重
    if Path(checkpoint_path).exists():
        state_dict = torch.load(checkpoint_path, map_location=device)
        logger.info(f"检测模型权重加载成功：{checkpoint_path}")
    else:
        logger.warning(f"检测模型权重不存在：{checkpoint_path}，使用默认权重")
        
    return detector

def load_recognizer(checkpoint_path: str, device: str = 'cuda'):
    """加载特征提取模型"""
    from inference.recognizer import Recognizer
    
    logger.info(f"加载识别模型：{checkpoint_path}")
    recognizer = Recognizer(device=device)
    
    # 加载自定义权重
    if Path(checkpoint_path).exists():
        state_dict = torch.load(checkpoint_path, map_location=device)
        logger.info(f"识别模型权重加载成功：{checkpoint_path}")
    else:
        logger.warning(f"识别模型权重不存在：{checkpoint_path}，使用默认权重")
        
    return recognizer

def load_matcher(threshold: float = 0.6, device: str = 'cuda'):
    """加载特征比对模块"""
    from inference.matcher import Matcher
    
    logger.info(f"加载比对模块，阈值={threshold}")
    matcher = Matcher(threshold=threshold, device=device)
    return matcher

# ============================================
# LFW 测试
# ============================================

def run_lfw_test(recognizer, matcher, config: Dict) -> Dict:
    """
    运行 LFW 1:1 验证测试
    
    参考：LFW 官方测试协议
    http://vis-www.cs.umass.edu/lfw/results.html
    """
    logger.info("=" * 60)
    logger.info("开始 LFW 1:1 验证测试")
    logger.info("=" * 60)
    
    lfw_dir = Path(config['data_dir'])
    pairs_file = lfw_dir / "pairs.txt"
    
    if not pairs_file.exists():
        logger.error(f"LFW pairs.txt 不存在：{pairs_file}")
        return {'error': 'pairs.txt not found'}
    
    # 解析 pairs.txt
    logger.info("加载 LFW pairs.txt...")
    pairs = parse_lfw_pairs(pairs_file)
    logger.info(f"加载 {len(pairs)} 对图像")
    
    max_samples = config.get('max_samples')
    if max_samples:
        pairs = pairs[:max_samples]
        logger.info(f"限制测试样本数：{max_samples}")
    
    # 提取特征并计算相似度
    similarities = []
    labels = []
    failed_samples = []
    
    preprocessor = DataPreprocessor({})
    
    for img1_path, img2_path, label in tqdm(pairs, desc="LFW 测试"):
        try:
            # 加载图像
            img1 = cv2.imread(img1_path)
            img2 = cv2.imread(img2_path)
            
            if img1 is None or img2 is None:
                failed_samples.append({'img1': img1_path, 'img2': img2_path, 'reason': 'load_failed'})
                continue
            
            # 提取特征
            feat1 = extract_feature(recognizer, img1, preprocessor)
            feat2 = extract_feature(recognizer, img2, preprocessor)
            
            if feat1 is None or feat2 is None:
                failed_samples.append({'img1': img1_path, 'img2': img2_path, 'reason': 'extract_failed'})
                continue
            
            # 计算相似度
            sim = matcher.compare(feat1, feat2)
            similarities.append(sim)
            labels.append(label)
            
        except Exception as e:
            failed_samples.append({'img1': img1_path, 'img2': img2_path, 'reason': str(e)})
            continue
    
    if len(similarities) == 0:
        logger.error("没有有效的测试结果")
        return {'error': 'no valid results'}
    
    logger.info(f"有效样本：{len(similarities)}, 失败：{len(failed_samples)}")
    
    # 计算 NIST 标准指标
    logger.info("计算 NIST FRTE 指标...")
    nist_metrics = NISTMetrics()
    nist_metrics.add_scores(similarities, labels)
    
    results = nist_metrics.compute_all_metrics(fmr_targets=[1e-4, 1e-6])
    
    # 添加额外信息
    results['total_pairs'] = len(pairs)
    results['valid_pairs'] = len(similarities)
    results['failed_pairs'] = len(failed_samples)
    results['failed_samples'] = failed_samples[:10]  # 只保存前 10 个失败样本
    
    # 打印结果
    logger.info("=" * 40)
    logger.info("LFW 测试结果")
    logger.info("=" * 40)
    logger.info(f"有效样本数：{results['valid_pairs']}")
    logger.info(f"AUC: {results['auc']:.4f}")
    logger.info(f"EER: {results['eer']:.4f}")
    logger.info(f"FNMR@FMR=1e-4: {results.get('FNMR@FMR=1e-4', 'N/A')}")
    logger.info(f"FNMR@FMR=1e-6: {results.get('FNMR@FMR=1e-6', 'N/A')}")
    
    return results

def parse_lfw_pairs(pairs_file: Path) -> List[Tuple[str, str, int]]:
    """
    解析 LFW pairs.txt
    
    Returns:
        [(img1_path, img2_path, label), ...]
        label: 1=同一人，0=不同人
    """
    pairs = []
    lfw_dir = pairs_file.parent
    
    with open(pairs_file, 'r') as f:
        lines = f.readlines()
    
    # 跳过第一行（折叠数）
    for line in lines[1:]:
        parts = line.strip().split('\t')
        
        if len(parts) == 3:
            # 不同人
            name1, img1_num = parts[0], parts[1]
            name2, img2_num = parts[2].split('\t')
            
            img1_path = lfw_dir / name1 / f"{name1}_{img1_num.zfill(4)}.jpg"
            img2_path = lfw_dir / name2 / f"{name2}_{img2_num.zfill(4)}.jpg"
            pairs.append((str(img1_path), str(img2_path), 0))
            
        elif len(parts) == 4:
            # 同一人
            name = parts[0]
            img1_num = parts[1]
            img2_num = parts[2]
            
            img1_path = lfw_dir / name / f"{name}_{img1_num.zfill(4)}.jpg"
            img2_path = lfw_dir / name / f"{name}_{img2_num.zfill(4)}.jpg"
            pairs.append((str(img1_path), str(img2_path), 1))
    
    return pairs

def extract_feature(recognizer, image: np.ndarray, preprocessor: DataPreprocessor) -> Optional[np.ndarray]:
    """提取人脸特征"""
    try:
        # 使用识别器内置的预处理
        feature = recognizer.extract_feature(image)
        return feature
    except Exception as e:
        logger.debug(f"特征提取失败：{str(e)}")
        return None

# ============================================
# 检测模型测试
# ============================================

def run_detection_test(detector, config: Dict) -> Dict:
    """运行人脸检测测试"""
    logger.info("=" * 60)
    logger.info("开始人脸检测模型测试")
    logger.info("=" * 60)
    
    widerface_dir = Path(config['data_dir'])
    
    if not widerface_dir.exists():
        logger.warning(f"WIDER Face 数据集不存在：{widerface_dir}")
        return {'error': 'dataset not found', 'note': '请手动下载 WIDER Face 数据集'}
    
    # 使用一些测试图像
    test_images = list((widerface_dir / "WIDER_val" / "images").rglob("*.jpg"))[:config.get('max_samples', 100)]
    
    if len(test_images) == 0:
        logger.warning("未找到 WIDER Face 验证集图像")
        return {'error': 'no images found'}
    
    results = {
        'total_images': len(test_images),
        'detection_results': [],
        'inference_times': []
    }
    
    preprocessor = DataPreprocessor({})
    
    for img_path in tqdm(test_images, desc="检测测试"):
        try:
            # 加载图像
            image = cv2.imread(str(img_path))
            if image is None:
                continue
            
            # 推理
            start_time = time.time()
            detections = detector.detect(image)
            inference_time = (time.time() - start_time) * 1000  # ms
            
            results['detection_results'].append({
                'image': str(img_path),
                'num_faces': len(detections),
                'max_score': max([d['score'] for d in detections]) if detections else 0
            })
            
            results['inference_times'].append(inference_time)
            
        except Exception as e:
            logger.debug(f"检测失败 {img_path}: {str(e)}")
            continue
    
    # 计算统计信息
    if results['inference_times']:
        results['avg_inference_time'] = np.mean(results['inference_times'])
        results['fps'] = 1000 / results['avg_inference_time']
    else:
        results['avg_inference_time'] = 0
        results['fps'] = 0
    
    results['avg_faces_per_image'] = np.mean([r['num_faces'] for r in results['detection_results']])
    
    logger.info("=" * 40)
    logger.info("检测测试结果")
    logger.info("=" * 40)
    logger.info(f"测试图像数：{results['total_images']}")
    logger.info(f"平均推理时间：{results['avg_inference_time']:.2f} ms")
    logger.info(f"FPS: {results['fps']:.2f}")
    logger.info(f"平均每张图像人脸数：{results['avg_faces_per_image']:.2f}")
    
    return results

# ============================================
# 特征质量分析
# ============================================

def analyze_feature_quality(recognizer, config: Dict) -> Dict:
    """分析特征质量"""
    logger.info("=" * 60)
    logger.info("开始特征质量分析")
    logger.info("=" * 60)
    
    lfw_dir = Path(config['data_dir'])
    
    # 随机选择一些人
    sample_features = []
    sample_names = []
    
    person_dirs = list(lfw_dir.iterdir())[:50]  # 前 50 个人
    
    for person_dir in tqdm(person_dirs, desc="采样特征"):
        if not person_dir.is_dir():
            continue
        
        name = person_dir.name
        images = list(person_dir.glob("*.jpg"))[:3]  # 每人前 3 张
        
        for img_path in images:
            try:
                image = cv2.imread(str(img_path))
                if image is None:
                    continue
                
                feature = recognizer.extract_feature(image)
                if feature is not None:
                    sample_features.append(feature)
                    sample_names.append(name)
            except:
                continue
    
    if len(sample_features) < 10:
        return {'error': 'not enough samples'}
    
    features = np.array(sample_features)
    
    # 归一化
    features = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-10)
    
    # 计算类内距离和类间距离
    intra_distances = []
    inter_distances = []
    
    name_to_indices = {}
    for i, name in enumerate(sample_names):
        if name not in name_to_indices:
            name_to_indices[name] = []
        name_to_indices[name].append(i)
    
    # 类内距离
    for name, indices in name_to_indices.items():
        if len(indices) < 2:
            continue
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                dist = np.linalg.norm(features[indices[i]] - features[indices[j]])
                intra_distances.append(dist)
    
    # 类间距离（采样）
    all_indices = list(range(len(features)))
    np.random.seed(42)
    for _ in range(1000):
        i, j = np.random.choice(all_indices, 2, replace=False)
        if sample_names[i] != sample_names[j]:
            dist = np.linalg.norm(features[i] - features[j])
            inter_distances.append(dist)
    
    results = {
        'n_samples': len(features),
        'n_identities': len(name_to_indices),
        'mean_intra_distance': float(np.mean(intra_distances)),
        'std_intra_distance': float(np.std(intra_distances)),
        'mean_inter_distance': float(np.mean(inter_distances)),
        'std_inter_distance': float(np.std(inter_distances)),
        'separability': float(np.mean(inter_distances) / (np.mean(intra_distances) + 1e-10))
    }
    
    logger.info("=" * 40)
    logger.info("特征质量分析结果")
    logger.info("=" * 40)
    logger.info(f"样本数：{results['n_samples']}")
    logger.info(f"身份数：{results['n_identities']}")
    logger.info(f"类内距离：{results['mean_intra_distance']:.4f} ± {results['std_intra_distance']:.4f}")
    logger.info(f"类间距离：{results['mean_inter_distance']:.4f} ± {results['std_inter_distance']:.4f}")
    logger.info(f"可分性：{results['separability']:.2f}")
    
    return results

# ============================================
# 主测试流程
# ============================================

def run_full_test():
    """运行完整测试"""
    timestamp = setup_logging()
    
    logger.info("=" * 80)
    logger.info("人脸识别模型 - 全自动全面测试")
    logger.info("=" * 80)
    
    # 打印配置
    logger.info("测试配置:")
    logger.info(f"  检测模型：{CONFIG['model']['detector']['checkpoint']}")
    logger.info(f"  识别模型：{CONFIG['model']['recognizer']['checkpoint']}")
    logger.info(f"  匹配阈值：{CONFIG['model']['matcher']['threshold']}")
    logger.info(f"  设备：{CONFIG['model']['detector']['device']}")
    
    # 检查 CUDA
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA 版本：{torch.version.cuda}")
    else:
        logger.warning("CUDA 不可用，使用 CPU 模式")
    
    # 创建输出目录
    results_dir = Path(CONFIG['output']['results_dir'])
    reports_dir = Path(CONFIG['output']['reports_dir'])
    results_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载模型
    logger.info("=" * 60)
    logger.info("加载模型...")
    logger.info("=" * 60)
    
    device = CONFIG['model']['detector']['device'] if torch.cuda.is_available() else 'cpu'
    
    detector = load_detector(CONFIG['model']['detector']['checkpoint'], device)
    recognizer = load_recognizer(CONFIG['model']['recognizer']['checkpoint'], device)
    matcher = load_matcher(CONFIG['model']['matcher']['threshold'], device)
    
    # 运行测试
    all_results = {
        'timestamp': datetime.now().isoformat(),
        'config': CONFIG,
        'environment': {
            'python_version': sys.version,
            'torch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
            'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        },
        'tests': {}
    }
    
    start_time = time.time()
    
    # 1. LFW 测试
    if CONFIG['test']['lfw']['enabled']:
        lfw_results = run_lfw_test(recognizer, matcher, CONFIG['test']['lfw'])
        all_results['tests']['lfw'] = lfw_results
    
    # 2. 检测测试
    if CONFIG['test']['detection']['enabled']:
        det_results = run_detection_test(detector, CONFIG['test']['detection'])
        all_results['tests']['detection'] = det_results
    
    # 3. 特征质量分析
    feature_results = analyze_feature_quality(recognizer, CONFIG['test']['lfw'])
    all_results['tests']['feature_quality'] = feature_results
    
    # 总耗时
    all_results['total_time'] = time.time() - start_time
    
    # 保存结果
    results_file = results_dir / f"test_results_{timestamp}.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    
    logger.info(f"测试结果已保存：{results_file}")
    
    # 生成报告
    logger.info("=" * 60)
    logger.info("生成测试报告...")
    logger.info("=" * 60)
    
    report_gen = ReportGenerator(all_results, str(reports_dir))
    html_report = report_gen.generate_html_report()
    md_report = report_gen.generate_markdown_report()
    
    logger.info(f"HTML 报告：{html_report}")
    logger.info(f"Markdown 报告：{md_report}")
    
    # 打印最终摘要
    logger.info("=" * 80)
    logger.info("测试完成!")
    logger.info("=" * 80)
    
    print("\n" + "=" * 80)
    print("测试结果摘要")
    print("=" * 80)
    
    if 'lfw' in all_results['tests']:
        lfw = all_results['tests']['lfw']
        if 'auc' in lfw:
            print(f"\n【LFW 1:1 验证】")
            print(f"  有效样本：{lfw.get('valid_pairs', 'N/A')}")
            print(f"  AUC: {lfw.get('auc', 'N/A'):.4f}" if isinstance(lfw.get('auc'), float) else f"  AUC: {lfw.get('auc', 'N/A')}")
            print(f"  EER: {lfw.get('eer', 'N/A'):.4f}" if isinstance(lfw.get('eer'), float) else f"  EER: {lfw.get('eer', 'N/A')}")
    
    if 'detection' in all_results['tests']:
        det = all_results['tests']['detection']
        if 'fps' in det:
            print(f"\n【人脸检测】")
            print(f"  测试图像：{det.get('total_images', 'N/A')}")
            print(f"  平均推理时间：{det.get('avg_inference_time', 0):.2f} ms")
            print(f"  FPS: {det.get('fps', 0):.2f}")
    
    if 'feature_quality' in all_results['tests']:
        feat = all_results['tests']['feature_quality']
        if 'separability' in feat:
            print(f"\n【特征质量】")
            print(f"  样本数：{feat.get('n_samples', 'N/A')}")
            print(f"  类内距离：{feat.get('mean_intra_distance', 0):.4f}")
            print(f"  类间距离：{feat.get('mean_inter_distance', 0):.4f}")
            print(f"  可分性：{feat.get('separability', 0):.2f}")
    
    print(f"\n总测试时间：{all_results['total_time']:.1f} 秒")
    print("=" * 80)
    
    # 打开 HTML 报告
    try:
        os.startfile(html_report)
        logger.info(f"已在浏览器中打开 HTML 报告")
    except:
        pass
    
    return all_results

if __name__ == "__main__":
    run_full_test()
