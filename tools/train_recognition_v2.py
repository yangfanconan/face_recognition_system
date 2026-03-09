#!/usr/bin/env python3
"""
人脸识别模型训练脚本 v2（增量训练版本）

支持:
- 从检查点恢复训练（增量训练）
- ArcFace / CosFace 损失函数
- 余弦退火学习率调度
- 早停策略
- TensorBoard 日志

使用方法:
    python train_recognition_v2.py \
      --data-dir datasets/lfw \
      --epochs 50 \
      --batch-size 64 \
      --loss arcface \
      --resume checkpoints/recognition/best.pth
"""

import os
import sys
import argparse
import time
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.datasets.loader import LFWDataset
from models.recognition import DDFD_Rec
from models.recognition.losses import ArcFaceLoss, CosFaceLoss
from torch.utils.tensorboard import SummaryWriter


# ============================================
# 工具类
# ============================================

class EarlyStopping:
    """早停策略"""
    
    def __init__(self, patience=10, min_delta=0.001, mode='max'):
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
                print(f"🛑 早停触发：连续{self.patience}个 epoch 未改善")
        
        return self.should_stop


class AverageMeter:
    """指标平均器"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# ============================================
# 训练函数
# ============================================

def load_checkpoint(model, optimizer, scheduler, resume_path, device):
    """从检查点恢复训练"""
    if not Path(resume_path).exists():
        print(f"⚠️  检查点不存在：{resume_path}，从头开始训练")
        return 0, 0.0
    
    print(f"📥 加载检查点：{resume_path}")
    checkpoint = torch.load(resume_path, map_location=device)
    
    # 加载模型权重
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    
    # 加载优化器状态
    if 'optimizer_state_dict' in checkpoint and optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # 加载学习率调度器
    if 'scheduler_state_dict' in checkpoint and scheduler:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    # 获取训练进度
    start_epoch = checkpoint.get('epoch', 0) + 1
    best_acc = checkpoint.get('best_acc', 0.0)
    
    print(f"✅ 从 epoch {start_epoch-1} 恢复训练，最佳准确率：{best_acc:.4f}")
    
    return start_epoch, best_acc


def save_checkpoint(epoch, model, optimizer, scheduler, best_acc, is_best, save_dir):
    """保存检查点"""
    save_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_acc': best_acc,
    }
    
    # 保存最新
    torch.save(checkpoint, save_dir / 'latest.pth')
    
    # 保存最佳
    if is_best:
        torch.save(checkpoint, save_dir / 'best.pth')
        print(f"💾 保存最佳模型：{save_dir / 'best.pth'}")


def train_one_epoch(model, loader, criterion, optimizer, scaler, 
                    epoch, device, writer, log_interval=10):
    """训练一个 epoch"""
    model.train()
    
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    
    pbar = tqdm(loader, desc=f"Epoch {epoch+1} [Train]")
    
    for batch_idx, (images, labels) in enumerate(pbar):
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        with autocast():
            # 前向传播
            features = model(images)
            
            # 计算损失
            if isinstance(criterion, (ArcFaceLoss, CosFaceLoss)):
                # ArcFace/CosFace 需要原始特征（未归一化）
                loss = criterion(features, labels)
            else:
                # 普通 CrossEntropy
                loss = criterion(features, labels)
        
        # 反向传播
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # 计算准确率
        if isinstance(criterion, (ArcFaceLoss, CosFaceLoss)):
            # 对于 ArcFace，使用 logits 计算准确率
            with torch.no_grad():
                logits = model.fc(features) if hasattr(model, 'fc') else features
                preds = logits.argmax(dim=1)
                correct = (preds == labels).sum().item()
                acc = correct / labels.size(0)
        else:
            with torch.no_grad():
                preds = features.argmax(dim=1)
                correct = (preds == labels).sum().item()
                acc = correct / labels.size(0)
        
        loss_meter.update(loss.item(), images.size(0))
        acc_meter.update(acc, images.size(0))
        
        # 更新进度条
        pbar.set_postfix({
            'loss': f'{loss_meter.avg:.4f}',
            'acc': f'{acc_meter.avg:.4f}'
        })
        
        # TensorBoard 日志
        if writer and batch_idx % log_interval == 0:
            global_step = epoch * len(loader) + batch_idx
            writer.add_scalar('train/loss', loss_meter.avg, global_step)
            writer.add_scalar('train/acc', acc_meter.avg, global_step)
    
    return loss_meter.avg, acc_meter.avg


@torch.no_grad()
def validate(model, loader, criterion, device):
    """验证"""
    model.eval()
    
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    
    pbar = tqdm(loader, desc="Validating")
    
    all_features = []
    all_labels = []
    
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        # 前向传播
        features = model(images)
        
        # 计算损失
        if isinstance(criterion, (ArcFaceLoss, CosFaceLoss)):
            loss = criterion(features, labels)
        else:
            loss = criterion(features, labels)
        
        # 计算准确率
        with torch.no_grad():
            if isinstance(criterion, (ArcFaceLoss, CosFaceLoss)):
                logits = model.fc(features) if hasattr(model, 'fc') else features
                preds = logits.argmax(dim=1)
            else:
                preds = features.argmax(dim=1)
            
            correct = (preds == labels).sum().item()
            acc = correct / labels.size(0)
        
        loss_meter.update(loss.item(), images.size(0))
        acc_meter.update(acc, images.size(0))
        
        # 收集特征用于后续分析
        all_features.append(features.cpu())
        all_labels.append(labels.cpu())
        
        pbar.set_postfix({
            'loss': f'{loss_meter.avg:.4f}',
            'acc': f'{acc_meter.avg:.4f}'
        })
    
    return loss_meter.avg, acc_meter.avg, torch.cat(all_features), torch.cat(all_labels)


def main():
    parser = argparse.ArgumentParser(description="人脸识别模型训练 v2")
    
    # 数据参数
    parser.add_argument('--data-dir', type=str, default='datasets/lfw',
                        help='数据集目录')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='数据加载线程数')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=50,
                        help='训练轮次')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='批次大小')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='初始学习率')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD 动量')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        help='权重衰减')
    
    # 损失函数
    parser.add_argument('--loss', type=str, default='arcface',
                        choices=['arcface', 'cosface', 'crossentropy'],
                        help='损失函数类型')
    parser.add_argument('--margin', type=float, default=0.5,
                        help='ArcFace/CosFace margin')
    parser.add_argument('--scale', type=float, default=30,
                        help='ArcFace/CosFace scale')
    
    # 恢复训练
    parser.add_argument('--resume', type=str, default=None,
                        help='检查点路径（增量训练）')
    parser.add_argument('--save-dir', type=str, default='checkpoints/recognition_v2',
                        help='保存目录')
    
    # 其他
    parser.add_argument('--device', type=str, default='cuda',
                        help='计算设备')
    parser.add_argument('--log-interval', type=int, default=10,
                        help='日志打印间隔')
    parser.add_argument('--tb-dir', type=str, default='runs/recognition_v2',
                        help='TensorBoard 日志目录')
    
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  使用设备：{device}")
    
    if torch.cuda.is_available():
        print(f"📊 GPU: {torch.cuda.get_device_name(0)}")
    
    # 创建保存目录
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # TensorBoard
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(log_dir=f"{args.tb_dir}_{timestamp}")
    
    # ============================================
    # 数据加载
    # ============================================
    print("📁 加载数据集...")
    
    # 简化：使用 ImageFolder 格式
    # 实际使用需要替换为完整的数据加载代码
    from torchvision import datasets, transforms
    
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    
    # 加载数据（假设数据集是 ImageFolder 格式）
    train_dataset = datasets.ImageFolder(args.data_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(args.data_dir, transform=val_transform)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    print(f"✅ 训练集：{len(train_dataset)} 样本")
    print(f"✅ 验证集：{len(val_dataset)} 样本")
    
    # ============================================
    # 模型、损失、优化器
    # ============================================
    print("🔧 初始化模型...")
    
    # 获取类别数
    num_classes = len(train_dataset.classes)
    print(f"📊 类别数：{num_classes}")
    
    # 创建模型
    model = DDFD_Rec(feature_dim=512, num_classes=num_classes)
    model = model.to(device)
    
    # 创建损失函数
    if args.loss == 'arcface':
        criterion = ArcFaceLoss(
            in_features=512,
            out_features=num_classes,
            margin=args.margin,
            scale=args.scale
        ).to(device)
        print(f"✅ 使用 ArcFace Loss (margin={args.margin}, scale={args.scale})")
    elif args.loss == 'cosface':
        criterion = CosFaceLoss(
            in_features=512,
            out_features=num_classes,
            margin=args.margin,
            scale=args.scale
        ).to(device)
        print(f"✅ 使用 CosFace Loss (margin={args.margin}, scale={args.scale})")
    else:
        criterion = nn.CrossEntropyLoss().to(device)
        print("✅ 使用 CrossEntropy Loss")
    
    # 创建优化器
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    
    # 学习率调度器（余弦退火）
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-6
    )
    
    # 混合精度训练
    scaler = GradScaler()
    
    # ============================================
    # 恢复训练（增量训练）
    # ============================================
    start_epoch = 0
    best_acc = 0.0
    
    if args.resume:
        start_epoch, best_acc = load_checkpoint(
            model, optimizer, scheduler, args.resume, device
        )
    
    # 早停策略
    early_stopping = EarlyStopping(patience=10, mode='max')
    
    # ============================================
    # 训练循环
    # ============================================
    print("\n" + "="*60)
    print("🚀 开始训练")
    print("="*60)
    
    for epoch in range(start_epoch, args.epochs):
        print(f"\n📍 Epoch {epoch+1}/{args.epochs}")
        
        # 训练
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler,
            epoch, device, writer, args.log_interval
        )
        
        # 验证
        val_loss, val_acc, _, _ = validate(model, val_loader, criterion, device)
        
        # 更新学习率
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        # 打印摘要
        print(f"\n📊 训练结果:")
        print(f"  训练损失：{train_loss:.4f}, 训练准确率：{train_acc:.4f}")
        print(f"  验证损失：{val_loss:.4f}, 验证准确率：{val_acc:.4f}")
        print(f"  学习率：{current_lr:.6f}")
        
        # TensorBoard
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/acc', val_acc, epoch)
        writer.add_scalar('lr', current_lr, epoch)
        
        # 保存最佳模型
        is_best = val_acc > best_acc
        if is_best:
            best_acc = val_acc
        
        save_checkpoint(
            epoch, model, optimizer, scheduler, best_acc,
            is_best, save_dir
        )
        
        # 早停检查
        if early_stopping(val_acc):
            print("🛑 早停触发，结束训练")
            break
    
    # 完成
    writer.close()
    
    print("\n" + "="*60)
    print("✅ 训练完成!")
    print(f"🏆 最佳验证准确率：{best_acc:.4f}")
    print(f"📁 模型保存至：{save_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
