# 人脸识别系统端到端优化方案

**基于 LFW 测试结果**: AUC=0.5773, EER=0.4385, 准确率=66.62%  
**优化目标**: LFW AUC≥0.95, EER≤0.05, 准确率≥95%  
**文档版本**: v1.0  
**日期**: 2026 年 3 月 9 日

---

## 一、问题根因深度分析

### 1.1 识别模型问题分析

#### 测试结果回顾

| 指标 | 结果 | 行业标准 | 差距 |
|-----|------|---------|------|
| AUC | 0.5773 | > 0.95 | -0.37 |
| EER | 0.4385 | < 0.05 | +0.39 |
| 准确率 | 66.62% | > 95% | -28% |
| 最佳阈值 | 0.9883 | 0.4-0.6 | +0.4 |

#### 四维根因分析

```
┌─────────────────────────────────────────────────────────────────┐
│                    识别模型性能差的根因分析                       │
├──────────────┬──────────────────────────────────────────────────┤
│   分析维度    │                   根因分析                       │
├──────────────┼──────────────────────────────────────────────────┤
│ 1. 损失函数   │ • 使用基础 CrossEntropy，缺乏判别性              │
│              │ • 缺少角度边界约束（margin）                      │
│              │ • 特征空间未做归一化约束                          │
├──────────────┼──────────────────────────────────────────────────┤
│ 2. 网络结构   │ • 512 维特征维度可能不足                          │
│              │ • 缺少注意力机制增强判别特征                      │
│              │ • 频域分支可能未充分发挥作用                      │
├──────────────┼──────────────────────────────────────────────────┤
│ 3. 训练策略   │ • 仅 5 个 epoch，远未收敛                         │
│              │ • 学习率调度可能不合理                            │
│              │ • 缺少验证集监控，可能过拟合                      │
├──────────────┼──────────────────────────────────────────────────┤
│ 4. 数据分布   │ • LFW 仅 13k 样本，数据量不足                    │
│              │ • 缺少多样性（姿态/光照/年龄）                    │
│              │ • 类间样本不均衡（TP=142 vs FP=519）             │
└──────────────┴──────────────────────────────────────────────────┘
```

### 1.2 检测模型问题分析

#### 异常现象

```python
# 检测输出示例
boxes: tensor([[-1.55, -1353.13, 1.33, 1242.93], ...])  # 坐标异常
scores: tensor([1.0, 1.0, ...])  # 置信度全为 1.0
```

#### 技术根因

| 问题 | 可能原因 | 验证方法 |
|-----|---------|---------|
| **坐标负数/超大值** | • 解码时未正确应用锚点偏移<br>• 坐标归一化因子错误<br>• 输出层缺少激活约束 | 检查 decode 函数 |
| **置信度恒为 1.0** | • Sigmoid 激活函数缺失<br>• 输出未通过 softmax<br>• 分数计算逻辑错误 | 检查 score 计算 |
| **NMS 失效** | • IoU 计算错误<br>• 阈值设置不合理<br>• 未正确执行 NMS | 检查 nms 实现 |

### 1.3 问题 - 根因 - 解决方案对应表

| 问题编号 | 问题描述 | 根因 | 解决方案 | 优先级 |
|---------|---------|------|---------|--------|
| P1 | 识别 AUC 仅 0.58 | 损失函数无判别性 | 引入 ArcFace Loss | 🔴 高 |
| P2 | 识别 EER 高达 44% | 训练不充分（5 epochs） | 继续训练至 50+ epochs | 🔴 高 |
| P3 | 最佳阈值 0.99 | 特征未归一化 | 添加 L2 归一化层 | 🔴 高 |
| P4 | 检测坐标异常 | 坐标解码错误 | 修复 decode_bbox 函数 | 🔴 高 |
| P5 | 检测置信度恒 1.0 | 缺少 sigmoid 激活 | 添加 sigmoid 激活 | 🔴 高 |
| P6 | 检测 NMS 失效 | IoU 计算/阈值问题 | 重写 NMS 实现 | 🔴 高 |
| P7 | 训练数据不足 | 仅 LFW 13k 样本 | 引入 CASIA-WebFace | 🟡 中 |
| P8 | 类间不均衡 | TP/FP 比例失调 | 困难样本挖掘 | 🟡 中 |

---

## 二、识别模型核心优化方案

### 2.1 训练策略优化

#### 超参数调整方案

```yaml
# 优化后的训练配置
training:
  epochs: 50                    # 从 5 增加到 50
  batch_size: 64                # 从 16 增加到 64（需要更多显存）
  
  optimizer:
    type: "SGD"
    lr: 0.1                     # 初始学习率
    momentum: 0.9
    weight_decay: 5e-4
  
  lr_scheduler:
    type: "CosineAnnealingLR"   # 从 StepLR 改为余弦退火
    T_max: 50
    eta_min: 1e-6
  
  warmup:
    enabled: true
    epochs: 5
    initial_lr: 0.01
  
  early_stopping:
    enabled: true
    patience: 10
    min_delta: 0.001
```

#### 增量训练代码

```python
# tools/train_recognition_v2.py (增量训练版本)

import torch
import torch.nn as nn
from pathlib import Path

def load_checkpoint_with_resume(model, optimizer, scheduler, resume_path):
    """
    从检查点恢复训练（增量训练）
    
    Args:
        model: 模型
        optimizer: 优化器
        scheduler: 学习率调度器
        resume_path: 检查点路径
    
    Returns:
        start_epoch: 起始 epoch
        best_acc: 最佳准确率
    """
    if not Path(resume_path).exists():
        print(f"检查点不存在：{resume_path}，从头开始训练")
        return 0, 0.0
    
    checkpoint = torch.load(resume_path, map_location='cpu')
    
    # 加载模型权重
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    
    # 加载优化器状态
    if 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # 加载学习率调度器
    if 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    # 获取训练进度
    start_epoch = checkpoint.get('epoch', 0) + 1
    best_acc = checkpoint.get('best_acc', 0.0)
    
    print(f"从 epoch {start_epoch-1} 恢复训练，最佳准确率：{best_acc:.4f}")
    
    return start_epoch, best_acc


# 使用示例
def main():
    # ... 初始化模型、优化器、调度器 ...
    
    resume_path = "checkpoints/recognition/best.pth"
    start_epoch, best_acc = load_checkpoint_with_resume(
        model, optimizer, scheduler, resume_path
    )
    
    # 从恢复的 epoch 继续训练
    for epoch in range(start_epoch, 50):
        train_one_epoch(model, optimizer, scheduler, epoch)
        val_acc = validate(model)
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_acc': best_acc,
            }, "checkpoints/recognition/best_v2.pth")
```

#### 早停策略与过拟合监控

```python
# tools/utils/early_stopping.py

class EarlyStopping:
    """
    早停策略实现
    
    当验证指标在连续多个 epoch 内未改善时停止训练
    """
    
    def __init__(self, patience=10, min_delta=0.001, mode='max'):
        """
        Args:
            patience: 容忍的未改善 epoch 数
            min_delta: 最小改善阈值
            mode: 'max'（准确率）或 'min'（损失）
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value = None
        self.should_stop = False
    
    def __call__(self, value):
        if self.best_value is None:
            self.best_value = value
            return False
        
        if self.mode == 'max':
            improved = value > self.best_value + self.min_delta
        else:
            improved = value < self.best_value - self.min_delta
        
        if improved:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                print(f"早停触发：连续{self.patience}个 epoch 未改善")
        
        return self.should_stop


# 过拟合监控
class OverfitMonitor:
    """过拟合监控"""
    
    def __init__(self):
        self.train_losses = []
        self.val_losses = []
    
    def update(self, train_loss, val_loss):
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        
        # 检测过拟合
        if len(self.train_losses) > 5:
            recent_train_gap = self.train_losses[-1] - self.train_losses[-5]
            recent_val_gap = self.val_losses[-1] - self.val_losses[-5]
            
            # 训练损失下降但验证损失上升 = 过拟合
            if recent_train_gap < -0.01 and recent_val_gap > 0.01:
                print("⚠️  检测到过拟合迹象！")
                return True
        
        return False
```

### 2.2 损失函数优化（ArcFace 实现）

#### ArcFace Loss 完整实现

```python
# models/recognition/losses.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ArcFaceLoss(nn.Module):
    """
    ArcFace Loss 实现
    
    Additive Angular Margin Loss for Deep Face Recognition
    
    参考: https://arxiv.org/abs/1801.07698
    
    Args:
        in_features: 输入特征维度 (如 512)
        out_features: 类别数 (如身份数)
        margin: 角度边界 (默认 0.5)
        scale: 缩放因子 (默认 30)
    """
    
    def __init__(self, in_features=512, out_features=10000, margin=0.5, scale=30):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.margin = margin
        self.scale = scale
        
        # 权重矩阵 W (out_features x in_features)
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        
        # 预计算 cos(m) 和 sin(m)
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin
    
    def forward(self, features, labels):
        """
        Args:
            features: 归一化特征 (batch_size, in_features)
            labels: 身份标签 (batch_size,)
        
        Returns:
            loss: ArcFace loss 值
        """
        # L2 归一化特征和权重
        features_norm = F.normalize(features, p=2, dim=1)
        weight_norm = F.normalize(self.weight, p=2, dim=1)
        
        # 计算余弦相似度 cos(θ) = X · W
        cosine = F.linear(features_norm, weight_norm)
        
        # 计算正弦 sin(θ)
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        
        # 计算 cos(θ + m) = cos(θ)cos(m) - sin(θ)sin(m)
        phi = cosine * self.cos_m - sine * self.sin_m
        
        # 确保 φ 在正确范围内（避免 θ+m > π）
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        
        # 创建 one-hot 标签
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1)
        
        # 只在真实类别上应用 margin
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        
        # 应用缩放因子
        output *= self.scale
        
        # 计算交叉熵损失
        loss = F.cross_entropy(output, labels)
        
        return loss


class CosFaceLoss(nn.Module):
    """
    CosFace Loss 实现
    
    Large Margin Cosine Loss for Face Recognition
    
    参考: https://arxiv.org/abs/1801.09414
    
    Args:
        in_features: 输入特征维度
        out_features: 类别数
        margin: 余弦边界 (默认 0.4)
        scale: 缩放因子 (默认 30)
    """
    
    def __init__(self, in_features=512, out_features=10000, margin=0.4, scale=30):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.margin = margin
        self.scale = scale
        
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
    
    def forward(self, features, labels):
        # L2 归一化
        features_norm = F.normalize(features, p=2, dim=1)
        weight_norm = F.normalize(self.weight, p=2, dim=1)
        
        # 余弦相似度
        cosine = F.linear(features_norm, weight_norm)
        
        # 应用余弦边界
        phi = cosine - self.margin
        
        # one-hot 标签
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1)
        
        # 只在真实类别上应用 margin
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        
        # 缩放并计算损失
        output *= self.scale
        loss = F.cross_entropy(output, labels)
        
        return loss


class CombinedLoss(nn.Module):
    """
    组合损失：ArcFace + CrossEntropy
    
    用于平滑过渡到新损失函数
    """
    
    def __init__(self, in_features, out_features, 
                 arcface_margin=0.5, arcface_scale=30,
                 ce_weight=0.1):
        super().__init__()
        self.arcface = ArcFaceLoss(in_features, out_features, 
                                   arcface_margin, arcface_scale)
        self.ce_weight = ce_weight
    
    def forward(self, features, logits, labels):
        """
        Args:
            features: 特征向量 (归一化后)
            logits: 原始 logits（用于 CE loss）
            labels: 标签
        """
        arcface_loss = self.arcface(features, labels)
        ce_loss = F.cross_entropy(logits, labels)
        
        return arcface_loss + self.ce_weight * ce_loss
```

#### 损失函数参数调优指南

```python
# 参数调优范围建议

# ArcFace margin 调优
margin_search_range = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55]
# 建议：从 0.4 开始，根据验证集调整
# - margin 太小 → 判别性不足
# - margin 太大 → 难以收敛

# scale 调优
scale_search_range = [20, 25, 30, 35, 40]
# 建议：固定为 30 或 32
# - scale 太小 → 梯度消失
# - scale 太大 → 梯度爆炸

# 推荐配置
OPTIMAL_CONFIG = {
    'arcface': {
        'margin': 0.5,      # 角度边界（弧度）
        'scale': 30,        # 特征缩放
    },
    'cosface': {
        'margin': 0.4,      # 余弦边界
        'scale': 30,
    }
}
```

### 2.3 数据扩充方案

#### CASIA-WebFace 数据集下载与融合

```python
# tools/data/prepare_casia_webface.py

"""
CASIA-WebFace 数据集准备脚本

数据集信息:
- 10,575 个身份
- 494,414 张图像
- 下载地址：https://www.cbsr.ia.ac.cn/english/CASIA-WebFace-Database.html

使用方法:
    python prepare_casia_webface.py --download-dir ./datasets/casia_webface
"""

import os
import shutil
from pathlib import Path
from tqdm import tqdm
import cv2
import numpy as np


def download_casia_webface(output_dir, username, password):
    """
    下载 CASIA-WebFace 数据集
    
    需要先在官网注册：https://www.cbsr.ia.ac.cn/english/CASIA-WebFace-Database.html
    """
    print("CASIA-WebFace 需要手动下载")
    print("1. 访问：https://www.cbsr.ia.ac.cn/english/CASIA-WebFace-Database.html")
    print("2. 注册账号并申请下载权限")
    print("3. 下载 WebFace.zip 到指定目录")
    print("4. 运行 extract_casia_webface() 解压")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 检查是否已下载
    zip_file = output_path / "WebFace.zip"
    if not zip_file.exists():
        print(f"请将 WebFace.zip 放置到：{zip_file}")
        return False
    
    return True


def extract_casia_webface(zip_path, output_dir):
    """解压 CASIA-WebFace"""
    import zipfile
    
    zip_path = Path(zip_path)
    output_path = Path(output_dir)
    
    print(f"解压 CASIA-WebFace: {zip_path}")
    
    with zipfile.ZipFile(str(zip_path), 'r') as zip_ref:
        zip_ref.extractall(str(output_path))
    
    print("解压完成")


def convert_casia_to_lfw_format(casia_dir, output_dir):
    """
    将 CASIA-WebFace 转换为 LFW 格式
    
    CASIA 格式: CASIA-WebFace/0000045/000.jpg
    LFW 格式：lfw/Person_Name/Person_Name_0001.jpg
    """
    casia_path = Path(casia_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 扫描所有身份目录
    identity_dirs = [d for d in casia_path.iterdir() if d.is_dir()]
    
    print(f"处理 {len(identity_dirs)} 个身份...")
    
    for identity_dir in tqdm(identity_dirs):
        identity_id = identity_dir.name
        identity_name = f"CASIA_{identity_id}"  # 添加前缀避免命名冲突
        
        output_identity_dir = output_path / identity_name
        output_identity_dir.mkdir(parents=True, exist_ok=True)
        
        # 处理该身份的所有图像
        images = list(identity_dir.glob("*.jpg"))
        
        for idx, img_path in enumerate(images):
            # 重命名为 LFW 格式
            new_name = f"{identity_name}_{str(idx+1).zfill(4)}.jpg"
            new_path = output_identity_dir / new_name
            
            shutil.copy(str(img_path), str(new_path))
    
    print(f"转换完成：{output_path}")


def merge_lfw_and_casia(lfw_dir, casia_dir, output_dir):
    """
    合并 LFW 和 CASIA-WebFace 数据集
    
    Args:
        lfw_dir: LFW 数据集目录
        casia_dir: CASIA-WebFace 目录（已转换格式）
        output_dir: 合并输出目录
    """
    lfw_path = Path(lfw_dir)
    casia_path = Path(casia_dir)
    output_path = Path(output_dir)
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 复制 LFW
    print("复制 LFW 数据集...")
    for person_dir in lfw_path.iterdir():
        if person_dir.is_dir():
            shutil.copytree(str(person_dir), str(output_path / person_dir.name))
    
    # 复制 CASIA
    print("复制 CASIA-WebFace 数据集...")
    for person_dir in casia_path.iterdir():
        if person_dir.is_dir():
            shutil.copytree(str(person_dir), str(output_path / person_dir.name))
    
    # 统计
    total_identities = len(list(output_path.iterdir()))
    total_images = sum(len(list(d.glob("*.jpg"))) for d in output_path.iterdir() if d.is_dir())
    
    print(f"合并完成:")
    print(f"  总身份数：{total_identities}")
    print(f"  总图像数：{total_images}")


# 一键运行函数
def prepare_training_data():
    """一键准备训练数据"""
    
    # 1. CASIA-WebFace 下载（需手动）
    casia_dir = "datasets/casia_webface"
    download_casia_webface(casia_dir, "your_username", "your_password")
    
    # 2. 解压
    extract_casia_webface(f"{casia_dir}/WebFace.zip", casia_dir)
    
    # 3. 格式转换
    casia_lfw_format = "datasets/casia_webface_lfw"
    convert_casia_to_lfw_format(f"{casia_dir}/CASIA-WebFace", casia_lfw_format)
    
    # 4. 合并数据集
    lfw_dir = "datasets/lfw"
    merged_dir = "datasets/lfw_merged"
    merge_lfw_and_casia(lfw_dir, casia_lfw_format, merged_dir)
    
    print("\n数据准备完成！")
    print(f"训练数据目录：{merged_dir}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="准备 CASIA-WebFace 训练数据")
    parser.add_argument("--mode", choices=["download", "extract", "convert", "merge", "all"], 
                        default="all", help="运行模式")
    parser.add_argument("--casia-dir", default="datasets/casia_webface", help="CASIA 目录")
    parser.add_argument("--lfw-dir", default="datasets/lfw", help="LFW 目录")
    parser.add_argument("--output-dir", default="datasets/lfw_merged", help="输出目录")
    
    args = parser.parse_args()
    
    if args.mode == "all":
        prepare_training_data()
```

#### 数据增强策略

```python
# data/transforms/face_augmentation.py

"""
人脸识别专用数据增强

针对人脸识别场景设计：
- 光照变化增强
- 姿态变化增强
- 遮挡增强
- 颜色扰动
"""

import cv2
import numpy as np
import random
from albumentations import (
    Compose, Normalize, Resize,
    HorizontalFlip, ShiftScaleRotate,
    GaussNoise, IAAAdditiveGaussianNoise,
    RandomBrightnessContrast, HueSaturationValue,
    OneOf, CoarseDropout
)


def get_face_augmentation(train=True, img_size=112):
    """
    获取人脸识别数据增强配置
    
    Args:
        train: 是否为训练模式
        img_size: 输出图像尺寸
    
    Returns:
        albumentations 变换
    """
    if train:
        return Compose([
            # 基础变换
            Resize(img_size, img_size),
            HorizontalFlip(p=0.5),  # 水平翻转
            
            # 几何变换
            ShiftScaleRotate(
                shift_limit=0.05,  # 平移 5%
                scale_limit=0.1,   # 缩放 10%
                rotate_limit=15,   # 旋转 15 度
                p=0.5
            ),
            
            # 光照增强
            OneOf([
                RandomBrightnessContrast(
                    brightness_limit=0.3,
                    contrast_limit=0.3,
                    p=0.5
                ),
                HueSaturationValue(
                    hue_shift_limit=10,
                    sat_shift_limit=20,
                    val_shift_limit=30,
                    p=0.5
                ),
            ], p=0.5),
            
            # 噪声增强
            OneOf([
                GaussNoise(var_limit=(10.0, 50.0), p=0.5),
                IAAAdditiveGaussianNoise(scale=(10.0, 50.0), p=0.5),
            ], p=0.3),
            
            # 随机遮挡（模拟口罩/墨镜）
            CoarseDropout(
                max_holes=3,
                max_height=20,
                max_width=20,
                min_holes=1,
                min_height=10,
                min_width=10,
                p=0.3
            ),
            
            # 归一化
            Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
    else:
        # 验证/测试模式：仅基础变换
        return Compose([
            Resize(img_size, img_size),
            Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])


# 自定义增强：多姿态合成
def synthesize_multi_pose(image, angle_range=(-30, 30)):
    """
    合成多姿态人脸（通过仿射变换）
    
    Args:
        image: 输入人脸图像
        angle_range: 旋转角度范围
    
    Returns:
        旋转后的人脸图像
    """
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    
    angle = random.uniform(*angle_range)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    rotated = cv2.warpAffine(
        image, M, (w, h),
        borderMode=cv2.BORDER_REFLECT_101
    )
    
    return rotated


# 自定义增强：光照变化模拟
def simulate_lighting_variation(image, light_direction='random'):
    """
    模拟光照变化
    
    Args:
        image: 输入图像
        light_direction: 光照方向 ('left', 'right', 'top', 'bottom', 'random')
    
    Returns:
        光照变化后的图像
    """
    if light_direction == 'random':
        light_direction = random.choice(['left', 'right', 'top', 'bottom'])
    
    # 创建光照梯度
    h, w = image.shape[:2]
    
    if light_direction == 'left':
        gradient = np.linspace(0.5, 1.0, w).reshape(1, -1)
    elif light_direction == 'right':
        gradient = np.linspace(1.0, 0.5, w).reshape(1, -1)
    elif light_direction == 'top':
        gradient = np.linspace(0.5, 1.0, h).reshape(-1, 1)
    else:  # bottom
        gradient = np.linspace(1.0, 0.5, h).reshape(-1, 1)
    
    # 应用光照
    gradient_3ch = np.repeat(gradient, 3, axis=-1 if len(gradient.shape) == 2 else 0)
    if len(gradient_3ch.shape) == 2:
        gradient_3ch = np.tile(gradient_3ch, (h if light_direction in ['left', 'right'] else 1, 1))
    
    augmented = (image.astype(np.float32) * gradient_3ch).clip(0, 255).astype(np.uint8)
    
    return augmented
```

### 2.4 网络结构优化

#### 添加注意力模块

```python
# models/recognition/ddfd_rec_v2.py

"""
DDFD-Rec 识别模型 v2 版本

优化点:
1. 添加 SE 注意力机制
2. 增加特征维度（512 → 1024）
3. 添加 L2 归一化输出层
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation 注意力模块
    
    参考: https://arxiv.org/abs/1709.01507
    """
    
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        
        # Squeeze: 全局平均池化
        y = self.avg_pool(x).view(b, c)
        
        # Excitation: 全连接层
        y = self.fc(y).view(b, c, 1, 1)
        
        # Scale: 逐通道加权
        return x * y.expand_as(x)


class DDFD_Rec_v2(nn.Module):
    """
    DDFD-Rec v2 版本
    
    在原版基础上添加:
    1. SE 注意力机制
    2. 更大的特征维度
    3. L2 归一化输出
    """
    
    def __init__(self, feature_dim=1024, use_se=True):
        super().__init__()
        self.use_se = use_se
        
        # ... 保留原有 backbone ...
        # 假设原有 backbone 输出 512 维特征
        
        # 新增：特征投影层（512 → 1024）
        self.feature_proj = nn.Sequential(
            nn.Linear(512, 1024, bias=False),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )
        
        # 新增：SE 注意力（可选）
        if use_se:
            self.se_attention = nn.Sequential(
                nn.Linear(1024, 64, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(64, 1024, bias=False),
                nn.Sigmoid()
            )
        
        # 输出层：L2 归一化
        self.output_layer = nn.Sequential(
            nn.BatchNorm1d(1024),
            nn.Dropout(0.5),
        )
        
        self.feature_dim = feature_dim
    
    def forward(self, x):
        # ... 原有 backbone 前向传播 ...
        # 假设得到 512 维特征
        features_512 = self.backbone(x)
        
        # 特征投影
        features = self.feature_proj(features_512)
        
        # SE 注意力（可选）
        if self.use_se:
            se_weights = self.se_attention(features)
            features = features * se_weights
        
        # L2 归一化
        features = F.normalize(features, p=2, dim=1)
        
        return features


# 模型构建函数
def build_recognizer_v2(pretrained_path=None, feature_dim=1024, use_se=True):
    """
    构建 v2 版本识别器
    
    Args:
        pretrained_path: 预训练权重路径
        feature_dim: 特征维度
        use_se: 是否使用 SE 注意力
    
    Returns:
        model: 识别模型
    """
    model = DDFD_Rec_v2(feature_dim=feature_dim, use_se=use_se)
    
    if pretrained_path and Path(pretrained_path).exists():
        # 加载预训练权重（需要处理维度不匹配）
        checkpoint = torch.load(pretrained_path, map_location='cpu')
        
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # 过滤不匹配的层
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in state_dict.items() 
                          if k in model_dict and v.shape == model_dict[k].shape}
        
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        
        print(f"加载预训练权重：{pretrained_path}")
        print(f"  成功加载 {len(pretrained_dict)} 层")
    
    return model
```

### 2.5 特征匹配优化

#### 动态阈值调整

```python
# inference/matcher_v2.py

"""
特征匹配器 v2 版本

优化点:
1. 动态阈值调整
2. 温度缩放校准
3. 多阈值策略
"""

import numpy as np
from sklearn.metrics import roc_curve


class Matcher_v2:
    """
    特征匹配器 v2
    
    支持动态阈值调整和温度缩放
    """
    
    def __init__(self, threshold=0.5, metric='cosine', 
                 use_temperature_scaling=False, temperature=1.0):
        """
        Args:
            threshold: 匹配阈值
            metric: 相似度度量 ('cosine', 'euclidean')
            use_temperature_scaling: 是否使用温度缩放
            temperature: 温度参数
        """
        self.threshold = threshold
        self.metric = metric
        self.use_temperature_scaling = use_temperature_scaling
        self.temperature = temperature
    
    def compare(self, feature1, feature2):
        """
        计算两个特征的相似度
        
        Args:
            feature1: 特征向量 1 (D,)
            feature2: 特征向量 2 (D,)
        
        Returns:
            similarity: 相似度分数
        """
        # L2 归一化
        f1 = feature1 / (np.linalg.norm(feature1) + 1e-10)
        f2 = feature2 / (np.linalg.norm(feature2) + 1e-10)
        
        # 余弦相似度
        similarity = np.dot(f1, f2)
        
        # 温度缩放（校准分数分布）
        if self.use_temperature_scaling:
            similarity = self._temperature_scale(similarity)
        
        return float(similarity)
    
    def _temperature_scale(self, similarity):
        """温度缩放校准"""
        # 将 [-1, 1] 映射到 [0, 1]
        scaled = (similarity + 1) / 2
        
        # 应用温度
        scaled = np.power(scaled, 1 / self.temperature)
        
        # 映射回 [-1, 1]
        return scaled * 2 - 1
    
    def calibrate_threshold(self, similarities, labels, target_fmr=1e-4):
        """
        根据验证集校准阈值
        
        Args:
            similarities: 相似度分数列表
            labels: 标签列表 (1=同人，0=异人)
            target_fmr: 目标误识率
        
        Returns:
            optimal_threshold: 最优阈值
        """
        fpr, tpr, thresholds = roc_curve(labels, similarities)
        fnmr = 1 - tpr
        
        # 找到最接近目标 FMR 的阈值
        idx = np.argmin(np.abs(fpr - target_fmr))
        optimal_threshold = thresholds[idx]
        
        print(f"校准阈值 (FMR={target_fmr}): {optimal_threshold:.4f}")
        
        return optimal_threshold
    
    def verify(self, feature1, feature2, adaptive_threshold=False):
        """
        人脸验证
        
        Args:
            feature1: 特征 1
            feature2: 特征 2
            adaptive_threshold: 是否使用自适应阈值
        
        Returns:
            is_same: 是否同一人
            similarity: 相似度
        """
        similarity = self.compare(feature1, feature2)
        
        if adaptive_threshold:
            # 根据相似度分布动态调整阈值
            if similarity > 0.8:
                threshold = 0.6  # 高置信度
            elif similarity > 0.5:
                threshold = 0.5  # 中等置信度
            else:
                threshold = 0.4  # 低置信度
        else:
            threshold = self.threshold
        
        return similarity >= threshold, similarity
```

---

## 三、检测模型修复方案

### 3.1 后处理修复代码

#### 修复坐标解码逻辑

```python
# models/detection/heads.py (修复版本)

"""
检测头修复版本

修复问题:
1. 坐标解码错误（负数/超大值）
2. 置信度计算错误（恒为 1.0）
3. NMS 实现问题
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DetectionHead(nn.Module):
    """
    修复后的检测头
    """
    
    def __init__(self, num_classes=1, num_anchors=9):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        
        # 分类头（添加 sigmoid 激活）
        self.cls_head = nn.Sequential(
            nn.Conv2d(256, num_anchors * num_classes, 1),
            nn.Sigmoid()  # 添加 sigmoid 确保输出在 [0, 1]
        )
        
        # 回归头（bbox 偏移量）
        self.reg_head = nn.Sequential(
            nn.Conv2d(256, num_anchors * 4, 1),
        )
        
        # 关键点头
        self.kpt_head = nn.Sequential(
            nn.Conv2d(256, num_anchors * 10, 1),  # 5 个关键点 × 2 坐标
        )
    
    def forward(self, x):
        """
        Returns:
            cls_scores: 分类分数 (B, num_anchors, H, W)
            bbox_offsets: bbox 偏移量 (B, num_anchors*4, H, W)
            keypoints: 关键点 (B, num_anchors*10, H, W)
        """
        cls_scores = self.cls_head(x)
        bbox_offsets = self.reg_head(x)
        keypoints = self.kpt_head(x)
        
        return cls_scores, bbox_offsets, keypoints


def decode_bbox(bbox_offsets, anchors, clip=True):
    """
    解码 bbox 坐标（修复版本）
    
    Args:
        bbox_offsets: bbox 偏移量 (B, N, 4)
        anchors: 锚框 (N, 4)
        clip: 是否裁剪到 [0, 1]
    
    Returns:
        boxes: 解码后的 bbox (B, N, 4)
    """
    # 确保输入形状正确
    if bbox_offsets.dim() == 4:
        B, _, H, W = bbox_offsets.shape
        bbox_offsets = bbox_offsets.permute(0, 2, 3, 1).reshape(B, -1, 4)
    
    if anchors.dim() == 4:
        anchors = anchors.reshape(-1, 4)
    
    # 计算锚框中心点和宽高
    anchor_centers = (anchors[:, :2] + anchors[:, 2:]) / 2
    anchor_sizes = anchors[:, 2:] - anchors[:, :2]
    
    # 解码偏移量（标准 FPN 解码公式）
    dx = bbox_offsets[:, :, 0]
    dy = bbox_offsets[:, :, 1]
    dw = bbox_offsets[:, :, 2]
    dh = bbox_offsets[:, :, 3]
    
    # 应用指数变换确保宽高为正
    pred_centers_x = dx * anchor_sizes[None, :, 0] + anchor_centers[None, :, 0]
    pred_centers_y = dy * anchor_sizes[None, :, 1] + anchor_centers[None, :, 1]
    pred_widths = torch.exp(dw) * anchor_sizes[None, :, 0]
    pred_heights = torch.exp(dh) * anchor_sizes[None, :, 1]
    
    # 转换为 [x1, y1, x2, y2] 格式
    boxes = torch.stack([
        pred_centers_x - pred_widths / 2,
        pred_centers_y - pred_heights / 2,
        pred_centers_x + pred_widths / 2,
        pred_centers_y + pred_heights / 2,
    ], dim=-1)
    
    # 裁剪到合理范围（避免负数和超大值）
    if clip:
        boxes = boxes.clamp(min=0, max=640)  # 假设最大尺寸 640
    
    return boxes


def nms(boxes, scores, iou_threshold=0.45):
    """
    非极大值抑制（修复版本）
    
    Args:
        boxes: bbox 框 (N, 4)
        scores: 置信度 (N,)
        iou_threshold: IoU 阈值
    
    Returns:
        keep_indices: 保留的索引
    """
    if len(boxes) == 0:
        return []
    
    # 按置信度排序
    _, order = scores.sort(descending=True)
    
    keep = []
    while order.numel() > 0:
        if order.numel() == 1:
            keep.append(order.item())
            break
        
        # 选择置信度最高的框
        i = order[0].item()
        keep.append(i)
        
        # 计算与其他框的 IoU
        remaining = order[1:]
        
        # IoU 计算
        xx1 = torch.max(boxes[i, 0], boxes[remaining, 0])
        yy1 = torch.max(boxes[i, 1], boxes[remaining, 1])
        xx2 = torch.min(boxes[i, 2], boxes[remaining, 2])
        yy2 = torch.min(boxes[i, 3], boxes[remaining, 3])
        
        inter = (xx2 - xx1).clamp(min=0) * (yy2 - yy1).clamp(min=0)
        
        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_r = (boxes[remaining, 2] - boxes[remaining, 0]) * \
                 (boxes[remaining, 3] - boxes[remaining, 1])
        
        union = area_i + area_r - inter
        iou = inter / union.clamp(min=1e-10)
        
        # 保留 IoU 低于阈值的框
        mask = iou <= iou_threshold
        order = order[mask]
        order = order[1:]  # 移除已选择的框
    
    return torch.tensor(keep, dtype=torch.long)
```

---

## 四、端到端验证方案

### 4.1 验证流程

```bash
# 优化后模型验证流程

# 1. 单模块验证
# 1.1 验证识别模型
python tests/benchmarks/lfw_recognition_only_test.py \
  --checkpoint checkpoints/recognition/best_v2.pth

# 1.2 验证检测模型
python tests/benchmarks/detection_test.py \
  --checkpoint checkpoints/detection/best_v2.pth \
  --dataset widerface

# 2. 端到端验证
python tests/benchmarks/lfw_end_to_end_test.py \
  --detector_ckpt checkpoints/detection/best_v2.pth \
  --recognizer_ckpt checkpoints/recognition/best_v2.pth

# 3. 生成对比报告
python tests/benchmarks/compare_results.py \
  --before tests/benchmarks/results/lfw_recognition_test_20260309.json \
  --after tests/benchmarks/results/lfw_recognition_test_optimized.json
```

### 4.2 优化效果对比模板

| 指标 | 优化前 | 优化后（预期） | 提升 |
|-----|--------|--------------|------|
| **识别 AUC** | 0.5773 | 0.95+ | +65% |
| **识别 EER** | 0.4385 | <0.05 | -89% |
| **识别准确率** | 66.62% | >95% | +43% |
| **最佳阈值** | 0.9883 | 0.4-0.6 | 回归正常 |
| **检测 mAP** | N/A | >80% | - |
| **检测 FPS** | N/A | >30 | - |

---

## 五、一键运行命令

```bash
# ============================================
# 人脸识别系统优化训练一键命令
# ============================================

# 1. 准备数据
python tools/data/prepare_casia_webface.py --mode all

# 2. 训练识别模型（50 epochs，ArcFace Loss）
python tools/train_recognition_v2.py \
  --data-dir datasets/lfw_merged \
  --epochs 50 \
  --batch-size 64 \
  --lr 0.1 \
  --loss arcface \
  --margin 0.5 \
  --scale 30 \
  --resume checkpoints/recognition/best.pth \
  --save-dir checkpoints/recognition_v2

# 3. 训练检测模型（修复后处理）
python tools/train_detection_v2.py \
  --data-dir datasets/widerface \
  --epochs 50 \
  --batch-size 16 \
  --resume checkpoints/detection/best.pth \
  --save-dir checkpoints/detection_v2

# 4. 验证识别模型
python tests/benchmarks/lfw_recognition_only_test.py \
  --checkpoint checkpoints/recognition_v2/best.pth

# 5. 端到端测试
python tests/benchmarks/lfw_end_to_end_test.py \
  --detector_ckpt checkpoints/detection_v2/best.pth \
  --recognizer_ckpt checkpoints/recognition_v2/best.pth
```

---

## 六、工程化落地建议

### 6.1 硬件资源配置

| 配置 | 最低要求 | 推荐配置 |
|-----|---------|---------|
| **GPU** | RTX 3060 (12GB) | RTX 4090 (24GB) |
| **显存** | 12GB | 24GB |
| **内存** | 32GB | 64GB |
| **存储** | 100GB SSD | 500GB NVMe |

### 6.2 训练耗时预估

| 模型 | 数据集 | 50 epochs | 100 epochs |
|-----|--------|----------|-----------|
| 识别 | LFW (13k) | ~8 小时 | ~16 小时 |
| 识别 | LFW+WebFace (500k) | ~3 天 | ~6 天 |
| 检测 | WIDER Face (12k) | ~10 小时 | ~20 小时 |

### 6.3 部署建议

```bash
# ONNX 转换
python tools/export_onnx.py \
  --model recognizer \
  --checkpoint checkpoints/recognition_v2/best.pth \
  --output checkpoints/recognition_v2/model.onnx

# TensorRT 量化（FP16）
python tools/export_tensorrt.py \
  --onnx checkpoints/recognition_v2/model.onnx \
  --output checkpoints/recognition_v2/model.trt \
  --precision fp16
```

---

*文档版本*: v1.0  
*最后更新*: 2026 年 3 月 9 日
