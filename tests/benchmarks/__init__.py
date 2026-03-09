"""
人脸识别端到端自动化测试框架
================================

本框架提供完整的权威测试方案，包括：
- NIST FRTE 标准指标计算
- 学术基准数据集测试
- 自动化测试执行
- 测试报告生成

使用方法:
    from tests.benchmarks import TestExecutor, NISTMetrics
    
    # 运行测试
    executor = TestExecutor(config)
    results = executor.run_all_tests(detector, recognizer, matcher)
    
    # 计算指标
    metrics = NISTMetrics()
    metrics.add_scores(similarities, labels)
    results = metrics.compute_all_metrics()
"""

from .metrics import (
    NISTMetrics,
    DetectionMetrics,
    RecognitionMetrics,
    VideoMetrics,
    MorphingDetectionMetrics
)

from .datasets import (
    DatasetDownloader,
    DataPreprocessor,
    LFWDataset
)

from .reports.report_generator import ReportGenerator

__version__ = "1.0.0"

__all__ = [
    # Metrics
    "NISTMetrics",
    "DetectionMetrics",
    "RecognitionMetrics",
    "VideoMetrics",
    "MorphingDetectionMetrics",
    
    # Datasets
    "DatasetDownloader",
    "DataPreprocessor",
    "LFWDataset",
    
    # Reports
    "ReportGenerator",
]
