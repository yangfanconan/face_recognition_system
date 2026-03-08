"""
微调脚本

支持场景:
- 口罩人脸
- 低照度人脸
- 小目标人脸
- 跨年龄人脸
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.recognition import DDFD_Rec, build_recognizer, RecognitionLoss
from models.common import (
    set_seed, load_checkpoint, freeze_layer, unfreeze_layer,
    load_yaml_config, get_device,
)


# ============================================
# 微调场景配置
# ============================================

SCENARIO_CONFIGS = {
    'mask': {
        'description': '口罩人脸微调',
        'data_ratio': 0.4,  # 口罩数据占比
        'loss_weights': {
            'id_loss': 1.0,
            'attr_loss': 0.5,
        },
        'lr_multiplier': 0.5,
        'frozen_layers': ['spatial_branch.stem', 'spatial_branch.layer1'],
        'special_aug': ['RandomRectangleMask', 'RandomErasing'],
        'epochs': 30,
        'batch_size': 32,
    },
    'low_light': {
        'description': '低照度人脸微调',
        'data_ratio': 0.3,
        'loss_weights': {
            'freq_branch': 1.5,
            'spatial_branch': 0.8,
        },
        'lr_multiplier': 0.3,
        'frozen_layers': ['spatial_branch.stem'],
        'special_aug': ['RandomDCTMask', 'LowLightEnhance'],
        'epochs': 40,
        'batch_size': 32,
    },
    'tiny_face': {
        'description': '小目标人脸微调',
        'data_ratio': 0.5,
        'loss_weights': {
            'small_face_loss': 2.0,
        },
        'lr_multiplier': 0.5,
        'frozen_layers': [],
        'special_aug': ['RandomCrop(scale=(0.3, 0.6))'],
        'epochs': 30,
        'batch_size': 64,
    },
    'cross_age': {
        'description': '跨年龄人脸微调',
        'data_ratio': 0.3,
        'loss_weights': {
            'id_loss': 1.0,
            'age_invariant_loss': 1.2,
        },
        'lr_multiplier': 0.3,
        'frozen_layers': ['spatial_branch.stem', 'spatial_branch.layer1', 'spatial_branch.layer2'],
        'special_aug': ['AgeSimulation'],
        'epochs': 50,
        'batch_size': 32,
    },
}


class FineTuner:
    """微调器"""
    
    def __init__(
        self,
        scenario: str,
        model: nn.Module,
        checkpoint_path: Optional[str] = None,
        device: Optional[torch.device] = None,
        output_dir: str = "checkpoints/finetuned",
    ):
        self.scenario = scenario
        self.model = model
        self.device = device or get_device()
        self.output_dir = output_dir
        
        # 获取场景配置
        if scenario not in SCENARIO_CONFIGS:
            raise ValueError(f"Unknown scenario: {scenario}. Available: {list(SCENARIO_CONFIGS.keys())}")
        
        self.config = SCENARIO_CONFIGS[scenario]
        
        # 加载预训练权重
        if checkpoint_path:
            self._load_checkpoint(checkpoint_path)
        
        # 冻结层
        self._freeze_layers()
        
        # 配置损失函数
        self._configure_loss()
        
        # 配置优化器
        self._configure_optimizer()
        
        os.makedirs(output_dir, exist_ok=True)
    
    def _load_checkpoint(self, path: str) -> None:
        """加载检查点"""
        logging.info(f"Loading checkpoint: {path}")
        load_checkpoint(path, self.model, device=self.device)
    
    def _freeze_layers(self) -> None:
        """冻结指定层"""
        frozen = self.config.get('frozen_layers', [])
        
        for layer_name in frozen:
            layer = self._get_layer(layer_name)
            if layer is not None:
                freeze_layer(layer)
                logging.info(f"Frozen layer: {layer_name}")
    
    def _get_layer(self, name: str) -> Optional[nn.Module]:
        """获取模型层"""
        parts = name.split('.')
        layer = self.model
        for part in parts:
            if hasattr(layer, part):
                layer = getattr(layer, part)
            else:
                return None
        return layer
    
    def _configure_loss(self) -> None:
        """配置损失函数"""
        loss_weights = self.config.get('loss_weights', {})
        
        self.criterion = RecognitionLoss(
            num_classes=100000,
            ortho_weight=loss_weights.get('ortho', 0.1),
            attr_weight=loss_weights.get('attr_loss', 0.5),
        )
    
    def _configure_optimizer(self) -> None:
        """配置优化器"""
        lr_multiplier = self.config.get('lr_multiplier', 1.0)
        base_lr = 0.01 * lr_multiplier
        
        # 只对未冻结的参数进行优化
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        
        self.optimizer = optim.SGD(
            trainable_params,
            lr=base_lr,
            momentum=0.9,
            weight_decay=0.0005
        )
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.get('epochs', 30)
        )
    
    def train_epoch(
        self,
        epoch: int,
        train_loader: DataLoader
    ) -> Dict[str, float]:
        """训练一个 epoch"""
        self.model.train()
        
        total_loss = 0
        num_batches = 0
        
        for batch in train_loader:
            images = batch['images'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # 前向传播
            outputs = self.model(images, labels=labels)
            
            # 计算损失
            loss_dict = self.criterion(outputs, labels)
            loss = loss_dict['loss_total']
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return {
            'loss': total_loss / num_batches,
        }
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None
    ) -> None:
        """完整训练流程"""
        epochs = self.config.get('epochs', 30)
        
        logging.info(f"Starting fine-tuning for scenario: {self.scenario}")
        logging.info(f"Epochs: {epochs}")
        logging.info(f"Frozen layers: {self.config.get('frozen_layers', [])}")
        
        best_accuracy = 0.0
        
        for epoch in range(epochs):
            # 训练
            train_metrics = self.train_epoch(epoch, train_loader)
            
            # 更新学习率
            self.scheduler.step()
            
            # 验证
            if val_loader is not None:
                val_metrics = self.evaluate(val_loader)
                
                if val_metrics.get('accuracy', 0) > best_accuracy:
                    best_accuracy = val_metrics['accuracy']
                    self.save_checkpoint(f"best_{self.scenario}.pth")
            
            # 保存检查点
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(f"epoch_{epoch}.pth")
            
            logging.info(
                f"Epoch {epoch+1}/{epochs} - Loss: {train_metrics['loss']:.4f}"
            )
        
        logging.info(f"Fine-tuning completed. Best accuracy: {best_accuracy:.4f}")
    
    @torch.no_grad()
    def evaluate(self, val_loader: DataLoader) -> Dict[str, float]:
        """验证评估"""
        self.model.eval()
        
        correct = 0
        total = 0
        
        for batch in val_loader:
            images = batch['images'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            outputs = self.model(images)
            id_feat = outputs['id_features']
            
            # 简化评估：使用余弦相似度
            # 实际应使用 LFW 等标准协议
            
        return {'accuracy': correct / total if total > 0 else 0}
    
    def save_checkpoint(self, name: str) -> str:
        """保存检查点"""
        path = os.path.join(self.output_dir, name)
        
        torch.save({
            'scenario': self.scenario,
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
        }, path)
        
        logging.info(f"Saved checkpoint: {path}")
        return path


def parse_args():
    """解析参数"""
    parser = argparse.ArgumentParser(description="Fine-tune face recognition model")
    
    parser.add_argument(
        "--scenario",
        type=str,
        required=True,
        choices=list(SCENARIO_CONFIGS.keys()),
        help="Fine-tuning scenario"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Pre-trained checkpoint path"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Dataset directory"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="checkpoints/finetuned",
        help="Output directory"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (cuda/cpu)"
    )
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # 设置随机种子
    set_seed(42)
    
    # 构建设备
    device = get_device(args.device)
    
    # 构建模型
    model = build_recognizer(model_type="ddfd_rec")
    model.to(device)
    
    # 创建微调器
    finetuner = FineTuner(
        scenario=args.scenario,
        model=model,
        checkpoint_path=args.checkpoint,
        device=device,
        output_dir=args.output_dir,
    )
    
    # TODO: 构建数据加载器
    # train_loader, val_loader = build_dataloaders(...)
    train_loader = None
    val_loader = None
    
    # 开始微调
    if train_loader is not None:
        finetuner.train(train_loader, val_loader)
    else:
        logging.warning("No data loader provided. Skipping training.")
        logging.info("Model loaded and ready for fine-tuning.")


if __name__ == "__main__":
    main()
