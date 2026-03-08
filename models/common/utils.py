"""
DDFD-FaceRec - 通用工具函数
Dual-Domain Feature Decoupling Face Recognition System
"""

import os
import random
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist


# ============================================
# 随机种子设置
# ============================================

def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """
    设置随机种子以保证复现性
    
    Args:
        seed: 随机种子
        deterministic: 是否启用确定性模式
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ============================================
# 设备配置
# ============================================

def get_device(gpu_id: Optional[Union[int, List[int]]] = None) -> torch.device:
    """
    获取计算设备
    
    Args:
        gpu_id: GPU ID，可以是整数或列表。None 则自动选择
        
    Returns:
        torch.device 对象
    """
    if gpu_id is None:
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    
    if isinstance(gpu_id, list):
        if len(gpu_id) > 0 and torch.cuda.is_available():
            return torch.device(f"cuda:{gpu_id[0]}")
        return torch.device("cpu")
    
    if torch.cuda.is_available() and gpu_id >= 0:
        return torch.device(f"cuda:{gpu_id}")
    return torch.device("cpu")


# ============================================
# 模型工具
# ============================================

def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    统计模型参数量
    
    Args:
        model: PyTorch 模型
        
    Returns:
        包含总参数量和可训练参数字典
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "total": total_params,
        "trainable": trainable_params,
        "non_trainable": total_params - trainable_params,
        "total_mb": total_params * 4 / 1024 / 1024,  # FP32
    }


def get_flops(model: nn.Module, input_size: Tuple[int, ...]) -> float:
    """
    估算模型 FLOPs (需要 torchprofile 或 thop)
    
    Args:
        model: PyTorch 模型
        input_size: 输入尺寸 (B, C, H, W)
        
    Returns:
        GFLOPs
    """
    try:
        from thop import profile
        input_tensor = torch.randn(1, *input_size)
        macs, _ = profile(model, inputs=(input_tensor,), verbose=False)
        return macs * 2 / 1e9  # MACs to FLOPs
    except ImportError:
        print("Warning: thop not installed. Install with: pip install thop")
        return 0.0


def freeze_layer(layer: nn.Module) -> None:
    """冻结网络层"""
    for param in layer.parameters():
        param.requires_grad = False


def unfreeze_layer(layer: nn.Module) -> None:
    """解冻网络层"""
    for param in layer.parameters():
        param.requires_grad = True


def freeze_bn(model: nn.Module) -> None:
    """冻结 BatchNorm 层"""
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            m.eval()
            freeze_layer(m)


# ============================================
# 分布式训练工具
# ============================================

def is_dist_available_and_initialized() -> bool:
    """检查分布式环境是否可用"""
    if not dist.is_available():
        return False
    return dist.is_initialized()


def get_rank() -> int:
    """获取当前进程 rank"""
    if not is_dist_available_and_initialized():
        return 0
    return dist.get_rank()


def get_world_size() -> int:
    """获取世界大小 (进程总数)"""
    if not is_dist_available_and_initialized():
        return 1
    return dist.get_world_size()


def is_main_process() -> bool:
    """判断是否为主进程"""
    return get_rank() == 0


def synchronize() -> None:
    """进程同步"""
    if not is_dist_available_and_initialized():
        return
    dist.barrier()


# ============================================
# 检查点工具
# ============================================

def save_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    epoch: int = 0,
    metrics: Optional[Dict] = None,
    is_best: bool = False,
    keep_last: int = 3
) -> None:
    """
    保存模型检查点
    
    Args:
        checkpoint_path: 保存路径
        model: 模型
        optimizer: 优化器
        scheduler: 学习率调度器
        epoch: 当前 epoch
        metrics: 评估指标
        is_best: 是否为最佳模型
        keep_last: 保留最近 N 个检查点
    """
    checkpoint_dir = os.path.dirname(checkpoint_path)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 构建检查点字典
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "metrics": metrics or {},
    }
    
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    
    # 保存当前检查点
    torch.save(checkpoint, checkpoint_path)
    
    # 保存最佳模型
    if is_best:
        best_path = os.path.join(checkpoint_dir, "best.pth")
        torch.save(checkpoint, best_path)
    
    # 清理旧检查点
    if keep_last > 0:
        cleanup_old_checkpoints(checkpoint_dir, keep_last)


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    strict: bool = True,
    device: Optional[torch.device] = None
) -> Dict:
    """
    加载模型检查点
    
    Args:
        checkpoint_path: 检查点路径
        model: 模型
        optimizer: 优化器
        scheduler: 学习率调度器
        strict: 是否严格匹配
        device: 计算设备
        
    Returns:
        检查点信息字典
    """
    if device is None:
        device = get_device()
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 加载模型权重
    model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
    
    # 加载优化器状态
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    # 加载调度器状态
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    
    return {
        "epoch": checkpoint.get("epoch", 0),
        "metrics": checkpoint.get("metrics", {}),
    }


def cleanup_old_checkpoints(checkpoint_dir: str, keep_last: int) -> None:
    """
    清理旧的检查点文件
    
    Args:
        checkpoint_dir: 检查点目录
        keep_last: 保留最近 N 个
    """
    checkpoints = list(Path(checkpoint_dir).glob("epoch_*.pth"))
    checkpoints.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    for old_checkpoint in checkpoints[keep_last:]:
        if old_checkpoint.name != "best.pth":
            old_checkpoint.unlink()


# ============================================
# 日志工具
# ============================================

class AverageMeter:
    """计算并存储平均值和当前值"""
    
    def __init__(self, name: str = ""):
        self.name = name
        self.reset()
    
    def reset(self) -> None:
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def __str__(self) -> str:
        return f"{self.name}: {self.avg:.4f}"


class ProgressMeter:
    """进度显示"""
    
    def __init__(self, num_batches: int, meters: List[AverageMeter], prefix: str = ""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
    
    def display(self, batch: int) -> None:
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))
    
    def _get_batch_fmtstr(self, num_batches: int) -> str:
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


# ============================================
# 图像工具
# ============================================

def normalize_image(
    image: np.ndarray,
    mean: List[float] = [0.485, 0.456, 0.406],
    std: List[float] = [0.229, 0.224, 0.225]
) -> np.ndarray:
    """
    图像归一化
    
    Args:
        image: HWC 格式图像
        mean: 均值
        std: 标准差
        
    Returns:
        归一化后的图像
    """
    image = image.astype(np.float32) / 255.0
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    return (image - mean) / std


def denormalize_image(
    image: np.ndarray,
    mean: List[float] = [0.485, 0.456, 0.406],
    std: List[float] = [0.229, 0.224, 0.225]
) -> np.ndarray:
    """
    图像反归一化
    
    Args:
        image: 归一化后的图像
        mean: 均值
        std: 标准差
        
    Returns:
        原始图像
    """
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    image = image * std + mean
    return np.clip(image * 255, 0, 255).astype(np.uint8)


# ============================================
# 框工具
# ============================================

def xyxy_to_xywh(bbox: np.ndarray) -> np.ndarray:
    """[x1, y1, x2, y2] 转 [x, y, w, h]"""
    return np.array([
        bbox[0],
        bbox[1],
        bbox[2] - bbox[0],
        bbox[3] - bbox[1]
    ])


def xywh_to_xyxy(bbox: np.ndarray) -> np.ndarray:
    """[x, y, w, h] 转 [x1, y1, x2, y2]"""
    return np.array([
        bbox[0],
        bbox[1],
        bbox[0] + bbox[2],
        bbox[1] + bbox[3]
    ])


def clip_bbox(bbox: np.ndarray, height: int, width: int) -> np.ndarray:
    """裁剪边界框到图像范围内"""
    bbox[0] = max(0, bbox[0])
    bbox[1] = max(0, bbox[1])
    bbox[2] = min(width, bbox[2])
    bbox[3] = min(height, bbox[3])
    return bbox


def bbox_area(bbox: np.ndarray) -> float:
    """计算边界框面积"""
    return max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1])


def bbox_iou(bbox1: np.ndarray, bbox2: np.ndarray) -> float:
    """计算两个边界框的 IoU"""
    inter_x1 = max(bbox1[0], bbox2[0])
    inter_y1 = max(bbox1[1], bbox2[1])
    inter_x2 = min(bbox1[2], bbox2[2])
    inter_y2 = min(bbox1[3], bbox2[3])
    
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    
    bbox1_area = bbox_area(bbox1)
    bbox2_area = bbox_area(bbox2)
    
    union_area = bbox1_area + bbox2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0


# ============================================
# 配置工具
# ============================================

def load_yaml_config(config_path: str) -> Dict:
    """
    加载 YAML 配置文件
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        配置字典
    """
    import yaml
    
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    return config


def merge_configs(base_config: Dict, override_config: Dict) -> Dict:
    """
    合并配置字典
    
    Args:
        base_config: 基础配置
        override_config: 覆盖配置
        
    Returns:
        合并后的配置
    """
    result = base_config.copy()
    
    for key, value in override_config.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    
    return result


# ============================================
# 其他工具
# ============================================

def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Tensor 转 NumPy"""
    if tensor.requires_grad:
        return tensor.detach().cpu().numpy()
    return tensor.cpu().numpy()


def to_tensor(array: np.ndarray, device: Optional[torch.device] = None) -> torch.Tensor:
    """NumPy 转 Tensor"""
    tensor = torch.from_numpy(array)
    if device is not None:
        tensor = tensor.to(device)
    return tensor


def format_time(seconds: float) -> str:
    """格式化时间为可读字符串"""
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.1f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}h {minutes}m {secs:.1f}s"


def format_number(num: int) -> str:
    """格式化数字 (如 1M, 1K)"""
    if num >= 1e9:
        return f"{num / 1e9:.1f}B"
    elif num >= 1e6:
        return f"{num / 1e6:.1f}M"
    elif num >= 1e3:
        return f"{num / 1e3:.1f}K"
    return str(num)
