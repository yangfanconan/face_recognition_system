"""
测试配置文件
"""

import pytest
import torch
import numpy as np


# 全局 fixture
@pytest.fixture(scope="session")
def device():
    """获取测试设备"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@pytest.fixture(scope="session")
def test_image():
    """创建测试图像"""
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)


@pytest.fixture(scope="session")
def test_bbox():
    """创建测试人脸框"""
    return np.array([100, 100, 200, 250])


@pytest.fixture(scope="session")
def test_landmarks():
    """创建测试关键点"""
    return np.array([
        [150, 140],  # 鼻子
        [120, 130],  # 左眼
        [180, 130],  # 右眼
        [125, 180],  # 左嘴角
        [175, 180],  # 右嘴角
    ], dtype=np.float32)


@pytest.fixture
def detection_model():
    """创建检测模型"""
    from models.detection import DKGA_Det
    model = DKGA_Det()
    model.eval()
    return model


@pytest.fixture
def recognition_model():
    """创建识别模型"""
    from models.recognition import DDFD_Rec
    model = DDFD_Rec()
    model.eval()
    return model


#  pytest 配置钩子
def pytest_configure(config):
    """pytest 配置"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests as requiring GPU"
    )


# 命令行选项
def pytest_addoption(parser):
    """添加命令行选项"""
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="Run slow tests"
    )
    parser.addoption(
        "--run-gpu",
        action="store_true",
        default=False,
        help="Run GPU tests"
    )


# 测试收集
def pytest_collection_modifyitems(config, items):
    """修改测试项"""
    if not config.getoption("--run-slow"):
        skip_slow = pytest.mark.skip(reason="need --run-slow option to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)
    
    if not config.getoption("--run-gpu"):
        skip_gpu = pytest.mark.skip(reason="need --run-gpu option to run")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)
