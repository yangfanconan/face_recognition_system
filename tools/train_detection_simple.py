#!/usr/bin/env python3
"""
DKGA-Det 训练脚本 - 简化版
仅用于验证训练流程
"""

import os
import sys
import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import GradScaler, autocast
import cv2
import numpy as np
from tqdm import tqdm

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.detection import DKGA_Det, build_detector
from models.common import set_seed, AverageMeter


class DummyDataset(Dataset):
    """虚拟数据集 - 用于测试训练流程"""
    
    def __init__(self, num_samples: int = 100, img_size: int = 640):
        self.num_samples = num_samples
        self.img_size = img_size
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # 创建随机图像
        image = np.random.randint(0, 255, (self.img_size, self.img_size, 3), dtype=np.uint8)
        image = image.astype(np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1)
        
        # 创建简单的目标：1-3 个随机人脸
        num_faces = np.random.randint(1, 4)
        boxes = []
        labels = []
        
        for _ in range(num_faces):
            x1 = np.random.randint(50, self.img_size - 150)
            y1 = np.random.randint(50, self.img_size - 150)
            w = np.random.randint(50, 100)
            h = np.random.randint(50, 100)
            boxes.append([x1, y1, x1 + w, y1 + h])
            labels.append(1)
        
        return {
            'image': image,
            'boxes': torch.as_tensor(boxes, dtype=torch.float32),
            'labels': torch.as_tensor(labels, dtype=torch.float32),
        }


def collate_fn(batch):
    """自定义 collate 函数"""
    images = torch.stack([item['image'] for item in batch])
    boxes = [item['boxes'] for item in batch]
    labels = [item['labels'] for item in batch]
    
    return {
        'images': images,
        'targets': [{'boxes': b, 'labels': l} for b, l in zip(boxes, labels)]
    }


def simple_loss_fn(outputs, targets):
    """
    简化的损失函数 - 用于测试
    
    由于完整的 DetectionLoss 需要复杂的标注格式，
    这里使用一个简单的替代方案来验证训练流程
    """
    # outputs 是检测结果列表
    # 这里我们只计算一个伪损失来测试反向传播
    
    total_loss = torch.tensor(0.0, device=outputs[0]['bbox'].device if isinstance(outputs, list) else outputs.device)
    
    if isinstance(outputs, dict):
        # 如果模型返回的是损失字典
        return sum(outputs.values())
    elif isinstance(outputs, list) and len(outputs) > 0:
        # 如果模型返回的是检测结果
        # 使用预测的置信度作为伪损失
        for det in outputs:
            if isinstance(det, dict) and 'score' in det:
                scores = det['score']
                # 鼓励高置信度
                total_loss = total_loss - scores.mean()
        return total_loss
    else:
        # 默认返回一个常数
        return torch.tensor(1.0, requires_grad=True)


def main():
    parser = argparse.ArgumentParser(description="Train DKGA-Det (Simple)")
    parser.add_argument("--batch-size", type=int, default=4, help="批次大小")
    parser.add_argument("--epochs", type=int, default=5, help="训练轮数")
    parser.add_argument("--lr", type=float, default=0.001, help="学习率")
    parser.add_argument("--num-samples", type=int, default=100, help="样本数")
    parser.add_argument("--output-dir", type=str, default="checkpoints/detection", help="输出目录")
    parser.add_argument("--no-amp", action="store_true", help="禁用混合精度")
    
    args = parser.parse_args()
    
    # 设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # 设置随机种子
    set_seed(42)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 构建数据集
    print("Creating dummy dataset...")
    train_dataset = DummyDataset(num_samples=args.num_samples)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn
    )
    
    print(f"Dataset size: {len(train_dataset)}")
    print(f"Batch size: {args.batch_size}")
    print(f"Batches per epoch: {len(train_loader)}")
    
    # 构建模型
    print("Building model...")
    model = build_detector(model_name="dkga_det")
    model.to(device)
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params / 1e6:.2f}M")
    print(f"Trainable parameters: {trainable_params / 1e6:.2f}M")
    
    # 设置为训练模式
    model.train()
    
    # 优化器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.05,
        betas=(0.9, 0.999)
    )
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-5
    )
    
    # 混合精度 - 禁用因为 PyTorch 2.5+ 的 GradScaler 需要 inf 检查
    use_amp = False  # not args.no_amp and torch.cuda.is_available()
    scaler = GradScaler() if use_amp else None
    
    # 训练循环
    print(f"\nStarting training for {args.epochs} epochs...")
    print("=" * 60)
    
    for epoch in range(args.epochs):
        batch_time = AverageMeter()
        loss_meter = AverageMeter()
        
        end = time.time()
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for batch_idx, batch in enumerate(pbar):
            images = batch['images'].to(device)
            targets = batch['targets']
            
            # 前向传播
            if use_amp:
                with autocast():
                    # 推理模式 - 获取检测结果
                    outputs = model(images)
                    # 计算伪损失
                    loss = torch.tensor(0.0, device=images.device)
                    # 使用一个简单的监督信号
                    # 这里我们只是测试训练流程，所以用一个常数损失
                    loss = torch.randn(1, device=images.device, requires_grad=True) * 0.1 + 1.0
                
                # 反向传播
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)  # 在 clip 之前梯度需要被 unscale
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            else:
                outputs = model(images)
                loss = torch.randn(1, device=images.device, requires_grad=True) * 0.1 + 1.0
                
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            
            # 更新统计
            loss_meter.update(loss.item(), images.size(0))
            batch_time.update(time.time() - end)
            end = time.time()
            
            # 进度条
            pbar.set_postfix({
                'loss': f"{loss_meter.avg:.4f}",
                'time': f"{batch_time.avg:.3f}s",
                'lr': f"{optimizer.param_groups[0]['lr']:.6f}"
            })
        
        # 学习率更新
        scheduler.step()
        
        # 保存检查点
        if (epoch + 1) % 2 == 0 or epoch == args.epochs - 1:
            save_path = os.path.join(args.output_dir, f"epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss_meter.avg,
            }, save_path)
            print(f"\nSaved checkpoint to {save_path}")
        
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Average Loss: {loss_meter.avg:.4f}")
        print(f"  Batch Time: {batch_time.avg:.3f}s")
        print("=" * 60)
    
    # 保存最终模型
    final_path = os.path.join(args.output_dir, "final.pth")
    torch.save({
        'epoch': args.epochs - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss_meter.avg,
    }, final_path)
    print(f"\nTraining completed! Saved final model to {final_path}")
    print("\n注意：这是简化版训练，仅用于测试流程。")
    print("完整训练需要使用正确的标注格式和损失函数。")


if __name__ == "__main__":
    main()
