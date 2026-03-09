#!/usr/bin/env python3
"""
人脸识别端到端自动化测试框架 - 主测试入口

使用方法:
    python run_test.py --config configs/default_config.yaml
    python run_test.py --test detection --dataset widerface
    python run_test.py --test recognition --dataset lfw
"""

import os
import sys
import json
import time
import argparse
import platform
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import traceback

import yaml
import numpy as np
import torch
from loguru import logger
from tqdm import tqdm

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 导入本地模块
from tests.benchmarks.datasets import DatasetDownloader, DataPreprocessor, LFWDataset
from tests.benchmarks.metrics import (
    NISTMetrics, 
    DetectionMetrics, 
    RecognitionMetrics,
    VideoMetrics,
    MorphingDetectionMetrics
)


# ============================================
# 配置日志
# ============================================

def setup_logging(log_file: str):
    """配置日志"""
    logger.remove()
    
    # 控制台输出
    logger.add(
        sys.stdout, 
        level="INFO",
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>"
    )
    
    # 文件输出
    logger.add(
        log_file,
        level="DEBUG",
        rotation="100 MB",
        retention="30 days",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}"
    )


# ============================================
# 模型接口定义
# ============================================

class ModelInterface:
    """
    模型接口基类
    
    用户需要继承此类并实现自己的模型加载和推理方法
    """
    
    def __init__(self, config: Dict, device: str = "cuda"):
        self.config = config
        self.device = device
        self.model = None
        self.is_loaded = False
        
    def load_model(self, checkpoint_path: str):
        """加载模型权重"""
        raise NotImplementedError
        
    def preprocess(self, image: np.ndarray) -> Any:
        """预处理输入"""
        raise NotImplementedError
        
    def inference(self, inputs: Any) -> Any:
        """模型推理"""
        raise NotImplementedError
        
    def postprocess(self, outputs: Any) -> Any:
        """后处理输出"""
        raise NotImplementedError
        
    def __call__(self, image: np.ndarray) -> Any:
        """完整的推理流程"""
        inputs = self.preprocess(image)
        outputs = self.inference(inputs)
        return self.postprocess(outputs)


class DetectorInterface(ModelInterface):
    """人脸检测模型接口"""
    
    def __init__(self, config: Dict, device: str = "cuda"):
        super().__init__(config, device)
        self.score_thresh = config.get('score_thresh', 0.5)
        self.nms_thresh = config.get('nms_thresh', 0.45)
        
    def detect(self, image: np.ndarray) -> List[Dict]:
        """
        人脸检测
        
        Returns:
            [
                {
                    'bbox': [x1, y1, x2, y2],
                    'score': float,
                    'landmarks': [[x1,y1], [x2,y2], ...]  # 5 个关键点
                },
                ...
            ]
        """
        results = self(image)
        return results


class RecognizerInterface(ModelInterface):
    """特征提取模型接口"""
    
    def __init__(self, config: Dict, device: str = "cuda"):
        super().__init__(config, device)
        self.feature_dim = config.get('feature_dim', 512)
        
    def extract_feature(self, face_image: np.ndarray) -> np.ndarray:
        """
        提取人脸特征
        
        Returns:
            特征向量 (D,)
        """
        feature = self(face_image)
        return feature


class MatcherInterface:
    """特征比对接口"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.threshold = config.get('threshold', 0.6)
        
    def compare(self, feature1: np.ndarray, feature2: np.ndarray) -> float:
        """
        比较两个特征
        
        Returns:
            相似度分数
        """
        # 余弦相似度
        feature1 = feature1 / (np.linalg.norm(feature1) + 1e-10)
        feature2 = feature2 / (np.linalg.norm(feature2) + 1e-10)
        return float(np.dot(feature1, feature2))
        
    def verify(self, feature1: np.ndarray, feature2: np.ndarray) -> Tuple[bool, float]:
        """
        验证两个人脸是否同一人
        
        Returns:
            (is_same, similarity)
        """
        similarity = self.compare(feature1, feature2)
        return similarity >= self.threshold, similarity


# ============================================
# 测试执行器
# ============================================

class TestExecutor:
    """测试执行器"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.results = {}
        self.start_time = None
        
        # 初始化日志
        log_dir = Path("tests/benchmarks/logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        setup_logging(str(log_dir / f"test_{timestamp}.log"))
        
        logger.info("=" * 60)
        logger.info("人脸识别端到端自动化测试框架")
        logger.info("=" * 60)
        
        # 打印环境信息
        self._print_environment_info()
        
    def _print_environment_info(self):
        """打印环境信息"""
        logger.info("环境信息:")
        logger.info(f"  Python: {platform.python_version()}")
        logger.info(f"  PyTorch: {torch.__version__}")
        logger.info(f"  CUDA: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"  GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"  CUDA Version: {torch.version.cuda}")
            
    def run_detection_test(self, detector: DetectorInterface, dataset_config: Dict) -> Dict:
        """运行人脸检测测试"""
        logger.info("=" * 40)
        logger.info("开始人脸检测模块测试")
        logger.info("=" * 40)
        
        results = {
            'module': 'detection',
            'dataset': dataset_config.get('name', 'unknown'),
            'metrics': {}
        }
        
        # 这里应该加载数据集并执行测试
        # 由于实际数据集需要下载，这里提供测试框架示例
        
        logger.info("检测测试完成")
        return results
        
    def run_recognition_test(self, recognizer: RecognizerInterface, matcher: MatcherInterface, dataset_config: Dict) -> Dict:
        """运行人脸识别测试"""
        logger.info("=" * 40)
        logger.info("开始人脸识别模块测试")
        logger.info("=" * 40)
        
        results = {
            'module': 'recognition',
            'dataset': dataset_config.get('name', 'unknown'),
            'metrics': {}
        }
        
        # 使用 LFW 数据集测试
        if dataset_config.get('name') == 'lfw':
            lfw = LFWDataset(dataset_config.get('data_dir', 'datasets/lfw'))
            
            try:
                pairs = lfw.load_pairs()
                logger.info(f"加载 LFW pairs: {len(pairs)} 对")
                
                # 计算相似度
                similarities = []
                labels = []
                
                for img1_path, img2_path, label in tqdm(pairs, desc="Extracting features"):
                    # 加载图像
                    img1 = cv2.imread(img1_path)
                    img2 = cv2.imread(img2_path)
                    
                    if img1 is None or img2 is None:
                        continue
                        
                    # 提取特征
                    feat1 = recognizer.extract_feature(img1)
                    feat2 = recognizer.extract_feature(img2)
                    
                    # 计算相似度
                    sim = matcher.compare(feat1, feat2)
                    similarities.append(sim)
                    labels.append(label)
                    
                # 计算指标
                nist_metrics = NISTMetrics()
                nist_metrics.add_scores(similarities, labels)
                results['metrics'] = nist_metrics.compute_all_metrics()
                
                logger.info(f"AUC: {results['metrics']['auc']:.4f}")
                logger.info(f"EER: {results['metrics']['eer']:.4f}")
                
            except Exception as e:
                logger.error(f"LFW 测试失败：{str(e)}")
                results['error'] = str(e)
                
        return results
        
    def run_all_tests(self, detector, recognizer, matcher) -> Dict:
        """运行所有测试"""
        self.start_time = time.time()
        
        all_results = {
            'timestamp': datetime.now().isoformat(),
            'config': self.config,
            'tests': {}
        }
        
        # 检测测试
        if self.config.get('model', {}).get('detector', {}).get('enabled'):
            det_config = self.config.get('datasets', {}).get('academic', {}).get('wider_face', {})
            all_results['tests']['detection'] = self.run_detection_test(detector, det_config)
            
        # 识别测试
        if self.config.get('model', {}).get('recognizer', {}).get('enabled'):
            rec_config = self.config.get('datasets', {}).get('academic', {}).get('lfw', {})
            all_results['tests']['recognition'] = self.run_recognition_test(recognizer, matcher, rec_config)
            
        # 计算总耗时
        all_results['total_time'] = time.time() - self.start_time
        
        self.results = all_results
        return all_results


# ============================================
# 实际模型实现（适配自研模型）
# ============================================

class CustomDetector(DetectorInterface):
    """
    自研人脸检测模型实现
    
    请根据实际模型接口修改此类的实现
    """
    
    def load_model(self, checkpoint_path: str):
        """加载检测模型"""
        logger.info(f"加载检测模型：{checkpoint_path}")
        
        # TODO: 替换为实际的模型加载代码
        # 示例：
        # from models.detection import DKGA_Det
        # self.model = DKGA_Det()
        # self.model.load_state_dict(torch.load(checkpoint_path))
        # self.model.to(self.device)
        # self.model.eval()
        
        self.is_loaded = True
        
    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """预处理"""
        preprocessor = DataPreprocessor({})
        tensor = preprocessor.preprocess_for_detection(image, self.config.get('input_size', [640, 640]))
        return torch.from_numpy(tensor).unsqueeze(0).to(self.device)
        
    def inference(self, inputs: torch.Tensor) -> Dict:
        """推理"""
        with torch.no_grad():
            outputs = self.model(inputs)
        return outputs
        
    def postprocess(self, outputs: Dict) -> List[Dict]:
        """后处理"""
        # TODO: 实现实际的后处理逻辑（NMS 等）
        return []


class CustomRecognizer(RecognizerInterface):
    """
    自研特征提取模型实现
    
    请根据实际模型接口修改此类的实现
    """
    
    def load_model(self, checkpoint_path: str):
        """加载识别模型"""
        logger.info(f"加载识别模型：{checkpoint_path}")
        
        # TODO: 替换为实际的模型加载代码
        # from models.recognition import DDFD_Rec
        # self.model = DDFD_Rec()
        # self.model.load_state_dict(torch.load(checkpoint_path))
        # self.model.to(self.device)
        # self.model.eval()
        
        self.is_loaded = True
        
    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """预处理"""
        preprocessor = DataPreprocessor({})
        tensor = preprocessor.preprocess_for_recognition(image, input_size=self.config.get('input_size', [112, 112]))
        return torch.from_numpy(tensor).unsqueeze(0).to(self.device)
        
    def inference(self, inputs: torch.Tensor) -> torch.Tensor:
        """推理"""
        with torch.no_grad():
            feature = self.model(inputs)
        return feature.cpu().numpy().squeeze()
        
    def postprocess(self, outputs: np.ndarray) -> np.ndarray:
        """后处理"""
        # L2 归一化
        return outputs / (np.linalg.norm(outputs) + 1e-10)


# ============================================
# 主函数
# ============================================

def load_config(config_path: str) -> Dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description="人脸识别端到端自动化测试")
    parser.add_argument("--config", type=str, default="tests/benchmarks/configs/default_config.yaml",
                        help="配置文件路径")
    parser.add_argument("--test", type=str, choices=["detection", "recognition", "all"], default="all",
                        help="测试类型")
    parser.add_argument("--dataset", type=str, default="lfw",
                        help="测试数据集")
    parser.add_argument("--detector_ckpt", type=str, default="checkpoints/detection/best.pth",
                        help="检测模型权重路径")
    parser.add_argument("--recognizer_ckpt", type=str, default="checkpoints/recognition/best.pth",
                        help="识别模型权重路径")
    
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 创建测试执行器
    executor = TestExecutor(config)
    
    # 初始化模型
    device = config.get('model', {}).get('detector', {}).get('device', 'cuda')
    
    detector_cfg = config.get('model', {}).get('detector', {})
    recognizer_cfg = config.get('model', {}).get('recognizer', {})
    matcher_cfg = config.get('model', {}).get('matcher', {})
    
    detector = CustomDetector(detector_cfg, device)
    recognizer = CustomRecognizer(recognizer_cfg, device)
    matcher = MatcherInterface(matcher_cfg)
    
    # 加载模型权重
    detector.load_model(args.detector_ckpt)
    recognizer.load_model(args.recognizer_ckpt)
    
    # 运行测试
    if args.test == "all":
        results = executor.run_all_tests(detector, recognizer, matcher)
    elif args.test == "detection":
        det_config = config.get('datasets', {}).get('academic', {}).get('wider_face', {})
        results = executor.run_detection_test(detector, det_config)
    elif args.test == "recognition":
        rec_config = config.get('datasets', {}).get('academic', {}).get('lfw', {})
        results = executor.run_recognition_test(recognizer, matcher, rec_config)
        
    # 保存结果
    results_dir = Path("tests/benchmarks/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = results_dir / f"test_results_{timestamp}.json"
    
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
        
    logger.info(f"测试结果已保存：{results_path}")
    
    # 生成报告
    logger.info("测试完成!")
    print("\n" + "=" * 60)
    print("测试结果摘要")
    print("=" * 60)
    
    if isinstance(results, dict) and 'tests' in results:
        for test_name, test_result in results['tests'].items():
            print(f"\n{test_name.upper()}:")
            if 'metrics' in test_result:
                for metric, value in test_result['metrics'].items():
                    if isinstance(value, float):
                        print(f"  {metric}: {value:.4f}")
                    else:
                        print(f"  {metric}: {value}")
                        
    print("\n" + "=" * 60)


if __name__ == "__main__":
    import cv2
    main()
