#!/usr/bin/env python3
"""
DKGA-Det 人脸检测模型训练脚本

用法:
    # 单卡训练
    python tools/train_detection.py --config configs/detection/train.yaml
    
    # 多卡训练 (DDP)
    python -m torch.distributed.launch --nproc_per_node=8 \
        tools/train_detection.py --config configs/detection/train.yaml --ddp
"""

import os
import sys
import argparse
import time
from pathlib import Path
from typing import Dict, Optional, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.detection import DKGA_Det, build_detector, DetectionLoss
from models.common import (
    set_seed, get_device, AverageMeter, ProgressMeter,
    save_checkpoint, load_checkpoint, is_main_process, synchronize,
    load_yaml_config, merge_configs,
)


class DetectionTrainer:
    """检测模型训练器"""
    
    def __init__(
        self,
        config: Dict,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        optimizer: Optional[optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        device: Optional[torch.device] = None,
        output_dir: str = "checkpoints/detection",
        resume: Optional[str] = None,
    ):
        self.config = config
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device or get_device(config.get('device', {}).get('gpu_id'))
        self.output_dir = output_dir
        self.resume = resume
        
        # 训练配置
        self.epochs = config.get('training', {}).get('epochs', 100)
        self.grad_clip = config.get('training', {}).get('grad_clip', 5.0)
        self.use_amp = config.get('training', {}).get('use_amp', True)
        self.print_freq = config.get('training', {}).get('logging', {}).get('print_freq', 10)
        
        # 优化器
        if optimizer is None:
            opt_config = config.get('training', {}).get('optimizer', {})
            if opt_config.get('name', 'AdamW') == 'AdamW':
                self.optimizer = optim.AdamW(
                    model.parameters(),
                    lr=opt_config.get('lr', 0.001),
                    weight_decay=opt_config.get('weight_decay', 0.05),
                    betas=tuple(opt_config.get('betas', [0.9, 0.999]))
                )
            else:
                self.optimizer = optim.SGD(
                    model.parameters(),
                    lr=opt_config.get('lr', 0.1),
                    momentum=opt_config.get('momentum', 0.9),
                    weight_decay=opt_config.get('weight_decay', 0.0005)
                )
        else:
            self.optimizer = optimizer
        
        # 学习率调度器
        if scheduler is None:
            sched_config = config.get('training', {}).get('scheduler', {})
            if sched_config.get('name', 'CosineAnnealingLR') == 'CosineAnnealingLR':
                self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    T_max=sched_config.get('T_max', 100),
                    eta_min=sched_config.get('eta_min', 1e-5)
                )
            else:
                self.scheduler = optim.lr_scheduler.MultiStepLR(
                    self.optimizer,
                    milestones=sched_config.get('milestones', [30, 60, 90]),
                    gamma=sched_config.get('gamma', 0.1)
                )
        else:
            self.scheduler = scheduler
        
        # 损失函数
        loss_config = config.get('detection', {}).get('loss', {})
        self.criterion = DetectionLoss(
            cls_weight=loss_config.get('cls_weight', 1.0),
            reg_weight=loss_config.get('reg_weight', 2.0),
            kpt_weight=loss_config.get('kpt_weight', 1.5),
            focal_gamma=loss_config.get('focal_gamma', 2.0),
            focal_alpha=loss_config.get('focal_alpha', 0.25),
        )
        
        # 混合精度
        self.scaler = GradScaler() if self.use_amp else None
        
        # 日志
        self.start_epoch = 0
        self.best_metric = 0.0
        self._load_resume()
    
    def _load_resume(self) -> None:
        """加载断点续训"""
        if self.resume and os.path.isfile(self.resume):
            checkpoint = load_checkpoint(
                self.resume, self.model, self.optimizer, self.scheduler,
                device=self.device
            )
            self.start_epoch = checkpoint.get('epoch', 0) + 1
            self.best_metric = checkpoint.get('metrics', {}).get('map_0.5', 0.0)
            print(f"Resumed from epoch {self.start_epoch}")
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """训练一个 epoch"""
        self.model.train()
        
        batch_time = AverageMeter('Time')
        data_time = AverageMeter('Data')
        losses = AverageMeter('Loss')
        loss_cls = AverageMeter('Loss_Cls')
        loss_reg = AverageMeter('Loss_Reg')
        loss_kpt = AverageMeter('Loss_Kpt')
        
        progress = ProgressMeter(
            len(self.train_loader),
            [batch_time, data_time, losses, loss_cls, loss_reg, loss_kpt],
            prefix=f"Epoch: [{epoch}]"
        )
        
        end = time.time()
        
        for batch_idx, batch in enumerate(self.train_loader):
            # 数据加载时间
            data_time.update(time.time() - end)
            
            # 准备数据
            images = batch['images'].to(self.device, non_blocking=True)
            targets = {k: v.to(self.device, non_blocking=True) for k, v in batch['targets'].items()}
            
            # 前向传播
            with autocast(enabled=self.use_amp):
                outputs = self.model(images, targets)
                loss_dict = self.criterion(outputs, targets)
                loss = loss_dict['loss_total']
            
            # 反向传播
            self.optimizer.zero_grad()
            
            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()
            
            # 更新统计
            batch_time.update(time.time() - end)
            end = time.time()
            
            losses.update(loss.item(), images.size(0))
            loss_cls.update(loss_dict.get('loss_cls', 0).item(), images.size(0))
            loss_reg.update(loss_dict.get('loss_reg', 0).item(), images.size(0))
            loss_kpt.update(loss_dict.get('loss_kpt', 0).item(), images.size(0))
            
            # 打印进度
            if batch_idx % self.print_freq == 0 and is_main_process():
                progress.display(batch_idx)
        
        return {
            'loss': losses.avg,
            'loss_cls': loss_cls.avg,
            'loss_reg': loss_reg.avg,
            'loss_kpt': loss_kpt.avg,
        }
    
    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """验证评估"""
        self.model.eval()
        
        # TODO: 实现 WiderFace 评估逻辑
        # 这里返回 placeholder
        metrics = {
            'map_0.5': 0.0,
            'map': 0.0,
        }
        
        return metrics
    
    def train(self) -> None:
        """完整训练流程"""
        print(f"Starting training for {self.epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Output directory: {self.output_dir}")
        
        for epoch in range(self.start_epoch, self.epochs):
            # 训练
            train_metrics = self.train_epoch(epoch)
            
            # 更新学习率
            self.scheduler.step()
            
            # 验证
            if self.val_loader is not None and (epoch + 1) % self.config.get('training', {}).get('logging', {}).get('eval_freq', 1) == 0:
                val_metrics = self.evaluate()
            else:
                val_metrics = {}
            
            # 保存检查点
            if is_main_process():
                is_best = val_metrics.get('map_0.5', 0) > self.best_metric
                self.best_metric = max(val_metrics.get('map_0.5', 0), self.best_metric)
                
                save_checkpoint(
                    os.path.join(self.output_dir, f"epoch_{epoch}.pth"),
                    self.model,
                    self.optimizer,
                    self.scheduler,
                    epoch=epoch,
                    metrics={**train_metrics, **val_metrics},
                    is_best=is_best,
                    keep_last=self.config.get('training', {}).get('checkpoint', {}).get('keep_last', 3)
                )
            
            synchronize()
        
        print("Training completed!")


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Train DKGA-Det Face Detector")
    
    parser.add_argument(
        "--config",
        type=str,
        default="configs/detection/train.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="datasets",
        help="Dataset directory"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="checkpoints/detection",
        help="Output directory for checkpoints"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--ddp",
        action="store_true",
        help="Use DistributedDataParallel"
    )
    parser.add_argument(
        "--local-rank",
        type=int,
        default=-1,
        help="Local rank for DDP"
    )
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    # 加载配置
    config = load_yaml_config(args.config)
    
    # 设置随机种子
    set_seed(config.get('seed', 42), config.get('deterministic', True))
    
    # DDP 初始化
    if args.ddp:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        device = torch.device(f"cuda:{args.local_rank}")
    else:
        device = get_device()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 构建模型
    model = build_detector(
        model_name=config.get('detection', {}).get('model', {}).get('name', 'dkga_det'),
        **config.get('detection', {}).get('model', {}).get('kwargs', {})
    )
    model.to(device)
    
    # DDP 包装
    if args.ddp:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            find_unused_parameters=config.get('ddp', {}).get('find_unused_parameters', False)
        )
    
    # TODO: 构建数据加载器
    # train_loader = build_dataloader(...)
    # val_loader = build_dataloader(...)
    train_loader = None
    val_loader = None
    
    # 创建训练器
    trainer = DetectionTrainer(
        config=config,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        output_dir=args.output_dir,
        resume=args.resume,
    )
    
    # 开始训练
    trainer.train()


if __name__ == "__main__":
    main()
