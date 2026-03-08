#!/usr/bin/env python3
"""
DKGA-Det 完整训练脚本

使用:
- WIDER Face 数据集
- Focal Loss + CIoU Loss + Wing Loss
- Mosaic + MixUp 数据增强
- AMP 混合精度训练

用法:
    # 单卡训练
    python tools/train_detection_complete.py --data-dir datasets/widerface --epochs 100
    
    # 测试模式 (少量样本)
    python tools/train_detection_complete.py --max-samples 500 --epochs 10
"""

import os
import sys
import argparse
import time
import json
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.detection import DKGA_Det, build_detector
from models.detection.complete_loss import DetectionLoss
from data.datasets.widerface_loader import WiderFaceDataset, collate_fn
from models.common import set_seed, AverageMeter


def parse_args():
    parser = argparse.ArgumentParser(description="Train DKGA-Det on WIDER Face")
    
    # 数据参数
    parser.add_argument("--data-dir", type=str, default="datasets/widerface",
                        help="数据集路径")
    parser.add_argument("--output-dir", type=str, default="checkpoints/detection",
                        help="输出目录")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="最大样本数 (None=全部)")
    
    # 训练参数
    parser.add_argument("--epochs", type=int, default=100,
                        help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="批次大小")
    parser.add_argument("--lr", type=float, default=0.0001,
                        help="初始学习率")
    parser.add_argument("--weight-decay", type=float, default=0.05,
                        help="权重衰减")
    parser.add_argument("--warmup-epochs", type=int, default=5,
                        help="预热轮数")
    
    # 模型参数
    parser.add_argument("--img-size", type=int, default=640,
                        help="输入图像尺寸")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="数据加载线程数")
    
    # 增强参数
    parser.add_argument("--no-mosaic", action="store_true",
                        help="禁用 Mosaic 增强")
    parser.add_argument("--no-mixup", action="store_true",
                        help="禁用 MixUp 增强")
    parser.add_argument("--mosaic-prob", type=float, default=0.5,
                        help="Mosaic 概率")
    parser.add_argument("--mixup-prob", type=float, default=0.2,
                        help="MixUp 概率")
    
    # 恢复训练
    parser.add_argument("--resume", type=str, default=None,
                        help="检查点路径")
    parser.add_argument("--pretrained", type=str, default=None,
                        help="预训练权重路径")
    
    # 其他
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    parser.add_argument("--print-freq", type=int, default=10,
                        help="打印频率")
    parser.add_argument("--save-freq", type=int, default=5,
                        help="保存频率")
    parser.add_argument("--no-amp", action="store_true",
                        help="禁用混合精度")
    
    return parser.parse_args()


class Trainer:
    """训练器"""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 创建输出目录
        os.makedirs(args.output_dir, exist_ok=True)
        self.log_dir = os.path.join(args.output_dir, "logs")
        os.makedirs(self.log_dir, exist_ok=True)
        
        # 设置随机种子
        set_seed(args.seed)
        
        # 构建模型
        print("Building model...")
        self.model = build_detector(model_name="dkga_det")
        self.model.to(self.device)
        
        # 加载预训练
        if args.pretrained and os.path.exists(args.pretrained):
            print(f"Loading pretrained weights from {args.pretrained}")
            state_dict = torch.load(args.pretrained, map_location=self.device)
            if 'model_state_dict' in state_dict:
                state_dict = state_dict['model_state_dict']
            self.model.load_state_dict(state_dict, strict=False)
        
        # 计算参数量
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params / 1e6:.2f}M")
        print(f"Trainable parameters: {trainable_params / 1e6:.2f}M")
        
        # 构建数据加载器
        print(f"\nLoading dataset from {args.data_dir}...")
        self.train_dataset = WiderFaceDataset(
            root_dir=args.data_dir,
            split="train",
            img_size=args.img_size,
            max_samples=args.max_samples,
            use_mosaic=not args.no_mosaic,
            mosaic_prob=args.mosaic_prob,
            use_mixup=not args.no_mixup,
            mixup_prob=args.mixup_prob,
        )
        
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )
        
        print(f"Dataset size: {len(self.train_dataset)}")
        print(f"Batch size: {args.batch_size}")
        print(f"Batches per epoch: {len(self.train_loader)}")
        
        # 损失函数
        self.criterion = DetectionLoss(
            num_classes=1,
            cls_weight=1.0,
            reg_weight=2.0,
            kpt_weight=1.5,
            strides=[8, 16, 32],
        )
        
        # 优化器
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(0.9, 0.999),
        )
        
        # 学习率调度器
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=args.epochs,
            eta_min=args.lr * 0.01,
        )
        
        # 混合精度
        self.use_amp = not args.no_amp and torch.cuda.is_available()
        self.scaler = GradScaler() if self.use_amp else None
        
        # 训练状态
        self.start_epoch = 0
        self.best_loss = float('inf')
        self.history = {
            'train_loss': [],
            'lr': [],
        }
        
        # 恢复训练
        if args.resume and os.path.exists(args.resume):
            self._load_checkpoint(args.resume)
    
    def _load_checkpoint(self, path: str):
        """加载检查点"""
        print(f"Loading checkpoint from {path}")
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.start_epoch = checkpoint.get('epoch', 0) + 1
        self.best_loss = checkpoint.get('best_loss', float('inf'))
        self.history = checkpoint.get('history', {'train_loss': [], 'lr': []})
        
        print(f"Resumed from epoch {self.start_epoch}, best loss: {self.best_loss:.4f}")
    
    def _save_checkpoint(self, epoch: int, is_best: bool = False):
        """保存检查点"""
        save_path = os.path.join(self.args.output_dir, f"epoch_{epoch+1}.pth")
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_loss': self.best_loss,
            'history': self.history,
            'args': vars(self.args),
        }, save_path)
        
        print(f"Saved checkpoint to {save_path}")
        
        if is_best:
            best_path = os.path.join(self.args.output_dir, "best.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'best_loss': self.best_loss,
                'history': self.history,
                'args': vars(self.args),
            }, best_path)
            print(f"Saved best model to {best_path}")
        
        # 保存历史
        history_path = os.path.join(self.log_dir, "training_history.json")
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def train_epoch(self, epoch: int):
        """训练一个 epoch"""
        self.model.train()
        
        batch_time = AverageMeter()
        data_time = AverageMeter()
        loss_meter = AverageMeter()
        loss_cls_meter = AverageMeter()
        loss_reg_meter = AverageMeter()
        loss_kpt_meter = AverageMeter()
        
        end = time.time()
        
        pbar = enumerate(self.train_loader)
        if self.args.print_freq > 0:
            pbar = enumerate(self.train_loader)
        
        for batch_idx, batch in pbar:
            # 测量数据加载时间
            data_time.update(time.time() - end)
            
            images = batch['images'].to(self.device)
            targets = batch['targets']
            
            # 前向传播 - 使用模型 training 模式直接获取损失
            if self.use_amp:
                with autocast():
                    # 模型在 training 模式下接收 targets 会返回损失字典
                    losses = self.model(images, targets=targets)
                    loss = losses['total_loss']
                
                # 反向传播
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # 模型在 training 模式下接收 targets 会返回损失字典
                losses = self.model(images, targets=targets)
                loss = losses['total_loss']
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
            
            self.optimizer.zero_grad()
            
            # 更新统计
            loss_meter.update(loss.item(), images.size(0))
            if 'loss_cls' in losses:
                loss_cls_meter.update(losses['loss_cls'].item(), images.size(0))
            if 'loss_reg' in losses:
                loss_reg_meter.update(losses['loss_reg'].item(), images.size(0))
            if 'loss_kpt' in losses:
                loss_kpt_meter.update(losses['loss_kpt'].item(), images.size(0))
            
            batch_time.update(time.time() - end)
            end = time.time()
            
            # 打印进度
            if self.args.print_freq > 0 and (batch_idx + 1) % self.args.print_freq == 0:
                lr = self.optimizer.param_groups[0]['lr']
                print(f"Epoch [{epoch+1}/{self.args.epochs}] "
                      f"Batch [{batch_idx+1}/{len(self.train_loader)}] "
                      f"Loss {loss_meter.avg:.4f} "
                      f"Time {batch_time.avg:.3f}s "
                      f"LR {lr:.6f}")
        
        # 更新学习率
        self.scheduler.step()
        
        return {
            'loss': loss_meter.avg,
            'loss_cls': loss_cls_meter.avg,
            'loss_reg': loss_reg_meter.avg,
            'loss_kpt': loss_kpt_meter.avg,
            'batch_time': batch_time.avg,
            'data_time': data_time.avg,
        }
    
    def train(self):
        """开始训练"""
        print(f"\n{'='*60}")
        print(f"Starting training for {self.args.epochs} epochs...")
        print(f"Device: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"{'='*60}\n")
        
        for epoch in range(self.start_epoch, self.args.epochs):
            print(f"\nEpoch {epoch+1}/{self.args.epochs}")
            print("-" * 40)
            
            # 训练
            metrics = self.train_epoch(epoch)
            
            # 更新历史
            self.history['train_loss'].append(metrics['loss'])
            self.history['lr'].append(self.optimizer.param_groups[0]['lr'])
            
            # 打印总结
            print(f"\n{'='*60}")
            print(f"Epoch {epoch+1} Summary:")
            print(f"  Train Loss: {metrics['loss']:.4f}")
            print(f"  Loss CLS: {metrics['loss_cls']:.4f}")
            print(f"  Loss REG: {metrics['loss_reg']:.4f}")
            print(f"  Loss KPT: {metrics['loss_kpt']:.4f}")
            print(f"  Batch Time: {metrics['batch_time']:.3f}s")
            print(f"  Data Time: {metrics['data_time']:.3f}s")
            print(f"  Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
            print(f"{'='*60}")
            
            # 保存检查点
            is_best = metrics['loss'] < self.best_loss
            self.best_loss = min(self.best_loss, metrics['loss'])
            
            save_freq = self.args.save_freq
            if (epoch + 1) % save_freq == 0 or epoch == self.args.epochs - 1 or is_best:
                self._save_checkpoint(epoch, is_best)
        
        # 保存最终模型
        final_path = os.path.join(self.args.output_dir, "final.pth")
        torch.save({
            'epoch': self.args.epochs - 1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_loss': self.best_loss,
            'history': self.history,
            'args': vars(self.args),
        }, final_path)
        
        print(f"\n{'='*60}")
        print(f"Training completed!")
        print(f"Best loss: {self.best_loss:.4f}")
        print(f"Final model: {final_path}")
        print(f"{'='*60}")


def main():
    args = parse_args()
    
    # 设置 args_epochs 用于 Trainer 内部使用
    args_epochs = args.epochs
    
    # 创建训练器
    trainer = Trainer(args)
    
    # 开始训练
    trainer.train()


if __name__ == "__main__":
    main()
