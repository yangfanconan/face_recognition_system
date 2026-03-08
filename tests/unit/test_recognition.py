"""
识别模型单元测试
"""

import pytest
import torch
import numpy as np

from models.recognition import (
    DDFD_Rec, DDFD_Rec_Tiny, build_recognizer,
    SpatialBranch, FrequencyBranch,
    FrequencyGatedFusion, TransformerEncoder,
    IdentityDisentangledHead,
    AdaArcLoss, OrthogonalLoss, RecognitionLoss,
)


class TestSpatialBranch:
    """测试空域分支"""
    
    def test_forward(self):
        """测试前向传播"""
        model = SpatialBranch(
            block=torch.nn.Module,  # BasicBlock
            layers=[2, 2, 2, 2],
            channels=[64, 128, 256, 512]
        )
        model.eval()
        
        x = torch.randn(2, 3, 112, 112)
        
        with torch.no_grad():
            features = model(x)
        
        # 检查输出
        assert len(features) == 4
        assert features[-1].shape[1] == 512  # 最后阶段通道数


class TestFrequencyBranch:
    """测试频域分支"""
    
    def test_forward(self):
        """测试前向传播"""
        model = FrequencyBranch(
            input_size=112,
            channels=[64, 128, 256],
            num_blocks=[2, 2, 2],
            use_dct=True
        )
        model.eval()
        
        x = torch.randn(2, 3, 112, 112)
        
        with torch.no_grad():
            features = model(x)
        
        # 检查输出
        assert len(features) == 3
        assert features[-1].shape[1] == 256


class TestFrequencyGatedFusion:
    """测试频域门控融合"""
    
    def test_forward(self):
        """测试前向传播"""
        model = FrequencyGatedFusion(channels=256)
        model.eval()
        
        x_spatial = torch.randn(2, 256, 14, 14)
        x_freq = torch.randn(2, 256, 14, 14)
        
        with torch.no_grad():
            x_fused = model(x_spatial, x_freq)
        
        # 检查输出形状
        assert x_fused.shape == x_spatial.shape


class TestTransformerEncoder:
    """测试 Transformer 编码器"""
    
    def test_forward(self):
        """测试前向传播"""
        model = TransformerEncoder(
            dim=256,
            depth=4,
            num_heads=8,
            mlp_ratio=4.0
        )
        model.eval()
        
        # 空间特征 (B, C, H, W)
        x = torch.randn(2, 256, 14, 14)
        
        with torch.no_grad():
            out = model(x)
        
        # 输出应该是全局特征 (B, dim)
        assert out.shape == (2, 256)


class TestIdentityDisentangledHead:
    """测试身份解耦头"""
    
    def test_forward(self):
        """测试前向传播"""
        model = IdentityDisentangledHead(
            in_channels=256,
            id_dim=409,
            attr_dim=103
        )
        model.eval()
        
        x = torch.randn(2, 256, 7, 7)
        
        with torch.no_grad():
            features, id_feat, attr_feat = model(x, return_separate=True)
        
        # 检查输出
        assert features.shape == (2, 512)  # 409 + 103
        assert id_feat.shape == (2, 409)
        assert attr_feat.shape == (2, 103)
        
        # 检查归一化
        id_norm = torch.norm(id_feat, p=2, dim=1)
        assert torch.allclose(id_norm, torch.ones_like(id_norm), atol=1e-5)


class TestDDFDRec:
    """测试完整识别模型"""
    
    def test_forward(self):
        """测试前向传播"""
        model = DDFD_Rec(
            spatial_kwargs={'model_type': 'resnet18'},
            transformer_kwargs={'dim': 256, 'depth': 2, 'num_heads': 4},
        )
        model.eval()
        
        x = torch.randn(2, 3, 112, 112)
        
        with torch.no_grad():
            features, id_feat, attr_feat = model.extract_features(x)
        
        # 检查输出
        assert features.shape[0] == 2
        assert id_feat.shape == (2, 409)
        assert attr_feat.shape == (2, 103)
    
    def test_get_identity_feature(self):
        """测试获取身份特征"""
        model = DDFD_Rec()
        model.eval()
        
        x = torch.randn(2, 3, 112, 112)
        
        with torch.no_grad():
            id_feat = model.get_identity_feature(x)
        
        assert id_feat.shape == (2, 409)
    
    def test_parameter_count(self):
        """测试参数量"""
        model = DDFD_Rec()
        total_params = sum(p.numel() for p in model.parameters())
        
        # DDFD-Rec 参数量应该在 10M-30M 之间
        assert 10e6 < total_params < 30e6
        print(f"Total parameters: {total_params / 1e6:.2f}M")


class TestDDFDRecTiny:
    """测试轻量级识别模型"""
    
    def test_forward(self):
        """测试前向传播"""
        model = DDFD_Rec_Tiny()
        model.eval()
        
        x = torch.randn(2, 3, 112, 112)
        
        with torch.no_grad():
            feat = model(x)
        
        assert feat.shape[0] == 2
    
    def test_parameter_count(self):
        """测试参数量"""
        model = DDFD_Rec_Tiny()
        total_params = sum(p.numel() for p in model.parameters())
        
        # Tiny 版本应该 < 5M
        assert total_params < 5e6
        print(f"Tiny parameters: {total_params / 1e6:.2f}M")


class TestAdaArcLoss:
    """测试 AdaArc Loss"""
    
    def test_forward(self):
        """测试前向传播"""
        loss_fn = AdaArcLoss(
            num_classes=1000,
            embedding_size=409,
            scale=32.0,
            m_base=0.5
        )
        
        features = torch.randn(32, 409)
        labels = torch.randint(0, 1000, (32,))
        
        loss = loss_fn(features, labels)
        
        assert loss > 0
        assert loss.dim() == 0


class TestOrthogonalLoss:
    """测试正交约束损失"""
    
    def test_forward(self):
        """测试前向传播"""
        loss_fn = OrthogonalLoss()
        
        id_feat = torch.randn(32, 409)
        attr_feat = torch.randn(32, 103)
        
        loss = loss_fn(id_feat, attr_feat)
        
        assert loss >= 0
        assert loss.dim() == 0


class TestRecognitionLoss:
    """测试组合损失"""
    
    def test_forward(self):
        """测试前向传播"""
        loss_fn = RecognitionLoss(
            num_classes=1000,
            embedding_size=512,
            ortho_weight=0.1,
            attr_weight=0.5
        )
        
        outputs = {
            'id_features': torch.randn(32, 409),
            'attr_features': torch.randn(32, 103),
            'arcface_logits': torch.randn(32, 1000),
            'attr_logits': [torch.randn(32, 10) for _ in range(5)],
        }
        labels = torch.randint(0, 1000, (32,))
        
        losses = loss_fn(outputs, labels)
        
        assert 'loss_total' in losses
        assert losses['loss_total'] > 0


class TestRecognizerIntegration:
    """识别模型集成测试"""
    
    def test_end_to_end(self):
        """端到端测试"""
        model = build_recognizer(model_type="ddfd_rec")
        model.eval()
        
        x = torch.randn(1, 3, 112, 112)
        
        with torch.no_grad():
            id_feat = model.get_identity_feature(x)
        
        # 检查输出
        assert id_feat.shape == (1, 409)
        
        # 检查归一化
        norm = torch.norm(id_feat, p=2, dim=1)
        assert torch.allclose(norm, torch.ones_like(norm), atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
