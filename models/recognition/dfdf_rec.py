"""
DDFD-Rec 识别模型 - 主模型

完整模型 = 空域分支 + 频域分支 + FGA 融合 + Transformer + 身份解耦头
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List

from models.recognition.spatial_branch import build_spatial_branch, SpatialBranch
from models.recognition.frequency_branch import build_frequency_branch, FrequencyBranch
from models.recognition.fusion import FrequencyGatedFusion, build_transformer_encoder, TransformerEncoder
from models.recognition.head import RecognitionHead, IdentityDisentangledHead


# ============================================
# DDFD-Rec 主模型
# ============================================

class DDFD_Rec(nn.Module):
    """
    DDFD-Rec 人脸识别模型
    
    Dual-Domain Feature Decoupling Recognition
    
    特性:
    - 空域 + 频域双分支特征提取
    - FGA 频域门控注意力融合
    - Transformer 全局建模
    - 身份 - 属性解耦头
    """
    
    def __init__(
        self,
        # 空域分支配置
        spatial_kwargs: Optional[Dict] = None,
        
        # 频域分支配置
        frequency_kwargs: Optional[Dict] = None,
        
        # 融合配置
        fusion_kwargs: Optional[Dict] = None,
        
        # Transformer 配置
        transformer_kwargs: Optional[Dict] = None,
        
        # 识别头配置
        head_kwargs: Optional[Dict] = None,
        
        # 模型配置
        input_size: int = 112,
        embedding_size: int = 512,
        id_dim: int = 409,
        attr_dim: int = 103,
        dropout: float = 0.4,
    ):
        super().__init__()
        
        self.input_size = input_size
        self.embedding_size = embedding_size
        self.id_dim = id_dim
        self.attr_dim = attr_dim
        
        # 空域分支
        if spatial_kwargs is None:
            spatial_kwargs = {
                'model_type': 'resnet18',
                'channels': [64, 128, 256, 512],
                'use_se': False,
            }
        self.spatial_branch = build_spatial_branch(**spatial_kwargs)
        
        # 频域分支
        if frequency_kwargs is None:
            frequency_kwargs = {
                'channels': [64, 128, 256],
                'num_blocks': [2, 2, 2],
                'use_dct': True,
                'use_se': False,
            }
        self.frequency_branch = build_frequency_branch(**frequency_kwargs)
        
        # 融合模块
        if fusion_kwargs is None:
            fusion_kwargs = {
                'channels': 256,
                'reduction': 4,
            }
        self.fusion = FrequencyGatedFusion(**fusion_kwargs)
        
        # Transformer 编码器
        if transformer_kwargs is None:
            transformer_kwargs = {
                'dim': 256,
                'depth': 4,
                'num_heads': 8,
                'mlp_ratio': 4.0,
                'drop_path_rate': 0.1,
            }
        self.transformer = build_transformer_encoder(**transformer_kwargs)
        
        # 识别头
        if head_kwargs is None:
            head_kwargs = {}
        
        # 计算 head 输入通道
        head_in_channels = transformer_kwargs.get('dim', 256)
        
        self.head = IdentityDisentangledHead(
            in_channels=head_in_channels,
            id_dim=id_dim,
            attr_dim=attr_dim,
            dropout=dropout
        )
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self) -> None:
        """初始化模型权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def extract_features(
        self,
        x: torch.Tensor,
        return_intermediate: bool = False
    ) -> Tuple[torch.Tensor, ...]:
        """
        提取特征
        
        Args:
            x: (B, 3, H, W) 输入图像
            return_intermediate: 是否返回中间特征
            
        Returns:
            features: (B, embedding_size) 最终特征
            id_feat: (B, id_dim) 身份特征
            attr_feat: (B, attr_dim) 属性特征
        """
        B = x.shape[0]
        
        # 空域特征提取
        spatial_feats = self.spatial_branch(x)  # (feat1, feat2, feat3, feat4)
        spatial_feat = spatial_feats[-1]  # 使用最高级特征
        
        # 频域特征提取
        frequency_feats = self.frequency_branch(x)  # (f1, f2, f3)
        frequency_feat = frequency_feats[-1]
        
        # 调整尺寸以匹配
        if spatial_feat.shape[2:] != frequency_feat.shape[2:]:
            frequency_feat = F.interpolate(
                frequency_feat,
                size=spatial_feat.shape[2:],
                mode='bilinear',
                align_corners=False
            )
        
        # FGA 融合
        fused_feat = self.fusion(spatial_feat, frequency_feat)
        
        # Transformer 全局建模
        global_feat = self.transformer(fused_feat)  # (B, dim)
        
        # 重塑为空间特征 (用于 head)
        spatial_size = int(math.sqrt(global_feat.shape[0] // B)) if global_feat.dim() == 2 else 7
        if global_feat.dim() == 2:
            # 如果是 (B, dim)，需要重塑
            # 这里我们使用一个投影
            global_feat_spatial = global_feat.view(B, -1, 1, 1).expand(-1, -1, 7, 7)
        else:
            global_feat_spatial = global_feat
        
        # 身份解耦
        features, id_feat, attr_feat = self.head(global_feat_spatial, return_separate=True)
        
        if return_intermediate:
            return features, id_feat, attr_feat, spatial_feats, frequency_feats
        return features, id_feat, attr_feat
    
    def forward(
        self,
        x: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: (B, 3, H, W) 输入图像
            labels: (B,) 身份标签 (训练时需要)
            
        Returns:
            outputs: 输出字典
        """
        # 提取特征
        features, id_feat, attr_feat = self.extract_features(x)
        
        outputs = {
            'features': features,
            'id_features': id_feat,
            'attr_features': attr_feat,
        }
        
        # 如果有标签，计算 ArcFace logits
        if labels is not None:
            # 这里简化处理，实际需要 ArcFace head
            outputs['labels'] = labels
        
        return outputs
    
    def get_identity_feature(self, x: torch.Tensor) -> torch.Tensor:
        """
        仅获取身份特征 (用于推理)
        
        Args:
            x: (B, 3, H, W) 输入图像
            
        Returns:
            id_feat: (B, id_dim) 身份特征
        """
        _, id_feat, _ = self.extract_features(x)
        return id_feat


# ============================================
# DDFD-Rec 轻量版 (用于移动端)
# ============================================

class DDFD_Rec_Tiny(nn.Module):
    """轻量级 DDFD-Rec"""
    
    def __init__(
        self,
        embedding_size: int = 256,
        id_dim: int = 205,
        attr_dim: int = 51,
    ):
        super().__init__()
        
        # 轻量空域分支
        self.spatial_branch = build_spatial_branch(
            model_type='tiny',
            channels=[32, 64, 128, 256]
        )
        
        # 轻量频域分支
        self.frequency_branch = build_frequency_branch(
            channels=[32, 64, 128],
            num_blocks=[1, 1, 1],
            use_dct=True
        )
        
        # 简化融合
        self.fusion = nn.Sequential(
            nn.Conv2d(256 + 128, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        
        # 简化头
        self.head = IdentityDisentangledHead(
            in_channels=256,
            id_dim=id_dim,
            attr_dim=attr_dim,
            dropout=0.3
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 空域特征
        spatial_feats = self.spatial_branch(x)
        spatial_feat = spatial_feats[-1]
        
        # 频域特征
        frequency_feats = self.frequency_branch(x)
        frequency_feat = frequency_feats[-1]
        
        # 上采样匹配
        frequency_feat = F.interpolate(
            frequency_feat,
            size=spatial_feat.shape[2:],
            mode='bilinear',
            align_corners=False
        )
        
        # 融合
        concat = torch.cat([spatial_feat, frequency_feat], dim=1)
        fused = self.fusion(concat)
        
        # GAP
        pooled = F.adaptive_avg_pool2d(fused, 1).view(x.shape[0], -1)
        
        # 身份解耦
        features, id_feat, _ = self.head(
            pooled.unsqueeze(-1).unsqueeze(-1),
            return_separate=True
        )
        
        return id_feat


# ============================================
# 模型工厂
# ============================================

def build_recognizer(
    model_type: str = "ddfd_rec",
    **kwargs
) -> nn.Module:
    """
    构建识别模型
    
    Args:
        model_type: 模型类型
        **kwargs: 配置参数
        
    Returns:
        识别模型
    """
    if model_type == "ddfd_rec":
        return DDFD_Rec(**kwargs)
    elif model_type == "ddfd_rec_tiny":
        return DDFD_Rec_Tiny(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# 导入 math
import math

if __name__ == "__main__":
    # 测试完整模型
    model = DDFD_Rec(
        spatial_kwargs={'model_type': 'resnet18'},
        frequency_kwargs={'channels': [64, 128, 256]},
        transformer_kwargs={'dim': 256, 'depth': 2, 'num_heads': 4},
    )
    model.eval()
    
    x = torch.randn(2, 3, 112, 112)
    
    with torch.no_grad():
        features, id_feat, attr_feat = model.extract_features(x)
    
    print("Input shape:", x.shape)
    print(f"Features shape: {features.shape}")
    print(f"ID feature shape: {id_feat.shape}")
    print(f"Attr feature shape: {attr_feat.shape}")
    
    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params / 1e6:.2f}M")
    print(f"Trainable parameters: {trainable_params / 1e6:.2f}M")
    
    # 测试轻量版
    print("\n--- Testing Tiny Model ---")
    tiny_model = DDFD_Rec_Tiny()
    tiny_model.eval()
    
    with torch.no_grad():
        tiny_feat = tiny_model(x)
    
    print(f"Tiny model feature shape: {tiny_feat.shape}")
    tiny_params = sum(p.numel() for p in tiny_model.parameters())
    print(f"Tiny model parameters: {tiny_params / 1e6:.2f}M")
