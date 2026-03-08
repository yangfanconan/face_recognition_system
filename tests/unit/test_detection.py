"""
检测模型单元测试
"""

import pytest
import torch
import numpy as np

from models.detection import (
    DKGA_Det, build_detector,
    CSPDarknet, BiFPNLite, DecoupledHead,
    FocalLoss, CIoULoss, WingLoss,
)


class TestCSPDarknet:
    """测试 CSPDarknet Backbone"""
    
    def test_forward(self):
        """测试前向传播"""
        model = CSPDarknet(
            depths=[3, 6, 6, 3],
            channels=[64, 128, 256, 512, 1024]
        )
        model.eval()
        
        x = torch.randn(2, 3, 640, 640)
        
        with torch.no_grad():
            features = model(x)
        
        # 检查输出
        assert len(features) == 3  # P3, P4, P5
        assert features[0].shape == (2, 256, 80, 80)  # P3
        assert features[1].shape == (2, 512, 40, 40)  # P4
        assert features[2].shape == (2, 1024, 20, 20)  # P5
    
    def test_parameter_count(self):
        """测试参数量"""
        model = CSPDarknet()
        total_params = sum(p.numel() for p in model.parameters())
        
        # CSPDarknet 参数量应该在 20M-40M 之间
        assert 20e6 < total_params < 40e6


class TestBiFPNLite:
    """测试 BiFPN-Lite Neck"""
    
    def test_forward(self):
        """测试前向传播"""
        model = BiFPNLite(
            in_channels=[256, 512, 1024],
            out_channels=256,
            use_p2=True
        )
        model.eval()
        
        features = [
            torch.randn(2, 256, 80, 80),  # P3
            torch.randn(2, 512, 40, 40),  # P4
            torch.randn(2, 1024, 20, 20),  # P5
        ]
        
        with torch.no_grad():
            outputs = model(features)
        
        # 检查输出 (P2, P3, P4, P5)
        assert len(outputs) == 4
        assert all(o.shape[1] == 256 for o in outputs)


class TestDecoupledHead:
    """测试解耦检测头"""
    
    def test_forward(self):
        """测试前向传播"""
        model = DecoupledHead(
            in_channels=256,
            num_classes=1,
            num_keypoints=5
        )
        model.eval()
        
        features = [
            torch.randn(2, 256, 80, 80),
            torch.randn(2, 256, 40, 40),
            torch.randn(2, 256, 20, 20),
        ]
        
        with torch.no_grad():
            outputs = model(features)
        
        # 检查输出
        assert 'cls_preds' in outputs
        assert 'reg_preds' in outputs
        assert 'kpt_preds' in outputs
        
        assert len(outputs['cls_preds']) == 3
        assert outputs['cls_preds'][0].shape[1] == 1  # num_classes


class TestDKGADet:
    """测试完整检测模型"""
    
    def test_forward(self):
        """测试前向传播"""
        model = DKGA_Det()
        model.eval()
        
        x = torch.randn(2, 3, 640, 640)
        
        with torch.no_grad():
            outputs = model(x)
        
        # 推理模式应该返回检测结果
        assert isinstance(outputs, list)
        assert len(outputs) == 2  # batch_size
    
    def test_parameter_count(self):
        """测试参数量"""
        model = DKGA_Det()
        total_params = sum(p.numel() for p in model.parameters())
        
        # DKGA-Det 参数量应该在 5M-15M 之间
        assert 5e6 < total_params < 15e6
        print(f"Total parameters: {total_params / 1e6:.2f}M")


class TestFocalLoss:
    """测试 Focal Loss"""
    
    def test_forward(self):
        """测试前向传播"""
        loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
        
        cls_pred = torch.randn(32, 1, 80, 80)
        cls_target = torch.rand(32, 1, 80, 80)
        
        loss = loss_fn(cls_pred, cls_target)
        
        assert loss > 0
        assert loss.dim() == 0  # 标量


class TestCIoULoss:
    """测试 CIoU Loss"""
    
    def test_forward(self):
        """测试前向传播"""
        loss_fn = CIoULoss()
        
        # [cx, cy, w, h] 格式
        pred = torch.randn(32, 4) * 50 + 100
        target = torch.randn(32, 4) * 50 + 100
        
        loss = loss_fn(pred, target)
        
        assert loss > 0
        assert loss.dim() == 0


class TestWingLoss:
    """测试 Wing Loss"""
    
    def test_forward(self):
        """测试前向传播"""
        loss_fn = WingLoss(w=10.0, epsilon=2.0)
        
        kpt_pred = torch.randn(32, 10) * 50
        kpt_target = torch.randn(32, 10) * 50
        
        loss = loss_fn(kpt_pred, kpt_target)
        
        assert loss > 0
        assert loss.dim() == 0


class TestDetectorIntegration:
    """检测模型集成测试"""
    
    def test_end_to_end(self):
        """端到端测试"""
        model = build_detector(model_name="dkga_det")
        model.eval()
        
        # 创建测试图像
        x = torch.randn(1, 3, 640, 640)
        
        with torch.no_grad():
            outputs = model(x)
        
        # 检查输出格式
        assert len(outputs) == 1
        
        det = outputs[0]
        assert 'boxes' in det
        assert 'scores' in det
        assert 'keypoints' in det


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
