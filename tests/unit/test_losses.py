"""
损失函数单元测试

测试内容:
- ArcFace Loss 前向传播
- CosFace Loss 前向传播
- 损失值范围验证
- 梯度计算验证
"""

import unittest
import torch
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.recognition.losses import ArcFaceLoss, CosFaceLoss, AMSoftmaxLoss


class TestArcFaceLoss(unittest.TestCase):
    """ArcFace Loss 测试"""
    
    def setUp(self):
        """测试前准备"""
        self.in_features = 512
        self.out_features = 100
        self.batch_size = 16
        self.margin = 0.5
        self.scale = 30
        
        # 创建损失函数
        self.criterion = ArcFaceLoss(
            in_features=self.in_features,
            out_features=self.out_features,
            margin=self.margin,
            scale=self.scale
        )
        
        # 创建测试数据
        self.features = torch.randn(self.batch_size, self.in_features)
        self.labels = torch.randint(0, self.out_features, (self.batch_size,))
    
    def test_forward(self):
        """测试前向传播"""
        loss = self.criterion(self.features, self.labels)
        
        # 检查损失值
        self.assertIsInstance(loss, torch.Tensor)
        self.assertTrue(loss.item() > 0, "损失值应该大于 0")
        self.assertTrue(torch.isfinite(loss), "损失值应该是有限的")
    
    def test_gradient(self):
        """测试梯度计算"""
        self.features.requires_grad_(True)
        
        loss = self.criterion(self.features, self.labels)
        loss.backward()
        
        # 检查梯度
        self.assertIsNotNone(self.features.grad)
        self.assertTrue(torch.isfinite(self.features.grad).all(), "梯度应该是有限的")
    
    def test_normalized_features(self):
        """测试归一化特征"""
        # 归一化特征
        features_norm = torch.nn.functional.normalize(self.features, p=2, dim=1)
        loss = self.criterion(features_norm, self.labels)
        
        self.assertTrue(loss.item() > 0)
    
    def test_different_batch_sizes(self):
        """测试不同批次大小"""
        for batch_size in [1, 4, 16, 64]:
            features = torch.randn(batch_size, self.in_features)
            labels = torch.randint(0, self.out_features, (batch_size,))
            
            loss = self.criterion(features, labels)
            self.assertTrue(torch.isfinite(loss), f"Batch size {batch_size}: 损失值应该是有限的")


class TestCosFaceLoss(unittest.TestCase):
    """CosFace Loss 测试"""
    
    def setUp(self):
        """测试前准备"""
        self.in_features = 512
        self.out_features = 100
        self.batch_size = 16
        self.margin = 0.4
        self.scale = 30
        
        self.criterion = CosFaceLoss(
            in_features=self.in_features,
            out_features=self.out_features,
            margin=self.margin,
            scale=self.scale
        )
        
        self.features = torch.randn(self.batch_size, self.in_features)
        self.labels = torch.randint(0, self.out_features, (self.batch_size,))
    
    def test_forward(self):
        """测试前向传播"""
        loss = self.criterion(self.features, self.labels)
        
        self.assertIsInstance(loss, torch.Tensor)
        self.assertTrue(loss.item() > 0)
        self.assertTrue(torch.isfinite(loss))
    
    def test_gradient(self):
        """测试梯度计算"""
        self.features.requires_grad_(True)
        
        loss = self.criterion(self.features, self.labels)
        loss.backward()
        
        self.assertIsNotNone(self.features.grad)
        self.assertTrue(torch.isfinite(self.features.grad).all())


class TestAMSoftmaxLoss(unittest.TestCase):
    """AM-Softmax Loss 测试"""
    
    def setUp(self):
        """测试前准备"""
        self.in_features = 512
        self.out_features = 100
        self.batch_size = 16
        self.margin = 0.3
        self.scale = 30
        
        self.criterion = AMSoftmaxLoss(
            in_features=self.in_features,
            out_features=self.out_features,
            margin=self.margin,
            scale=self.scale
        )
        
        self.features = torch.randn(self.batch_size, self.in_features)
        self.labels = torch.randint(0, self.out_features, (self.batch_size,))
    
    def test_forward(self):
        """测试前向传播"""
        loss = self.criterion(self.features, self.labels)
        
        self.assertIsInstance(loss, torch.Tensor)
        self.assertTrue(loss.item() > 0)
        self.assertTrue(torch.isfinite(loss))


class TestLossComparison(unittest.TestCase):
    """损失函数对比测试"""
    
    def setUp(self):
        """测试前准备"""
        self.in_features = 512
        self.out_features = 100
        self.batch_size = 16
        
        self.arcface = ArcFaceLoss(self.in_features, self.out_features, margin=0.5, scale=30)
        self.cosface = CosFaceLoss(self.in_features, self.out_features, margin=0.4, scale=30)
        self.amsoftmax = AMSoftmaxLoss(self.in_features, self.out_features, margin=0.3, scale=30)
        
        self.features = torch.randn(self.batch_size, self.in_features)
        self.labels = torch.randint(0, self.out_features, (self.batch_size,))
    
    def test_all_losses(self):
        """测试所有损失函数"""
        losses = {}
        
        # ArcFace
        losses['arcface'] = self.arcface(self.features, self.labels)
        
        # CosFace
        losses['cosface'] = self.cosface(self.features, self.labels)
        
        # AM-Softmax
        losses['amsoftmax'] = self.amsoftmax(self.features, self.labels)
        
        # 所有损失都应该是有限的正数
        for name, loss in losses.items():
            self.assertTrue(loss.item() > 0, f"{name}: 损失值应该大于 0")
            self.assertTrue(torch.isfinite(loss), f"{name}: 损失值应该是有限的")


if __name__ == "__main__":
    # 运行测试
    unittest.main(verbosity=2)
