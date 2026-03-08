"""
DDFD-Rec 识别模型 - 身份解耦头

将特征分解为身份子空间和属性子空间
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List, Dict


# ============================================
# 身份解耦头
# ============================================

class IdentityDisentangledHead(nn.Module):
    """
    身份解耦头
    
    将特征分解为:
    - 身份子空间 (80%): 用于人脸识别
    - 属性子空间 (20%): 编码姿态、光照、年龄等变化
    """
    
    def __init__(
        self,
        in_channels: int = 256,
        id_dim: int = 409,
        attr_dim: int = 103,
        fc_dim: int = 512,
        dropout: float = 0.4,
        use_bn: bool = False
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.id_dim = id_dim
        self.attr_dim = attr_dim
        self.embedding_size = id_dim + attr_dim
        
        # 全局平均池化
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # 身份分支
        self.id_fc1 = nn.Linear(in_channels, fc_dim)
        self.id_bn1 = nn.BatchNorm1d(fc_dim) if use_bn else nn.Identity()
        self.id_drop1 = nn.Dropout(dropout)
        self.id_fc2 = nn.Linear(fc_dim, id_dim)
        
        # 属性分支
        self.attr_fc1 = nn.Linear(in_channels, fc_dim // 2)
        self.attr_bn1 = nn.BatchNorm1d(fc_dim // 2) if use_bn else nn.Identity()
        self.attr_drop1 = nn.Dropout(dropout / 2)
        self.attr_fc2 = nn.Linear(fc_dim // 2, attr_dim)
        
        # 正交约束投影 (可选)
        self.ortho_proj = nn.Parameter(torch.eye(id_dim, attr_dim) * 0.01)
        
        self._init_weights()
    
    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        return_separate: bool = True
    ) -> Tuple[torch.Tensor, ...]:
        """
        前向传播
        
        Args:
            x: (B, C, H, W) 输入特征
            return_separate: 是否返回分离的特征
            
        Returns:
            features: (B, id_dim + attr_dim) 完整特征
            id_feat: (B, id_dim) 身份特征
            attr_feat: (B, attr_dim) 属性特征
        """
        B = x.shape[0]
        
        # 全局平均池化
        x = self.gap(x).view(B, -1)
        
        # 身份分支
        id_feat = self.id_fc1(x)
        id_feat = self.id_bn1(id_feat)
        id_feat = F.relu(id_feat)
        id_feat = self.id_drop1(id_feat)
        id_feat = self.id_fc2(id_feat)
        id_feat = F.normalize(id_feat, p=2, dim=1)
        
        # 属性分支
        attr_feat = self.attr_fc1(x)
        attr_feat = self.attr_bn1(attr_feat)
        attr_feat = F.relu(attr_feat)
        attr_feat = self.attr_drop1(attr_feat)
        attr_feat = self.attr_fc2(attr_feat)
        attr_feat = F.normalize(attr_feat, p=2, dim=1)
        
        # 拼接
        features = torch.cat([id_feat, attr_feat], dim=1)
        
        if return_separate:
            return features, id_feat, attr_feat
        return features
    
    def get_identity_feature(self, x: torch.Tensor) -> torch.Tensor:
        """仅获取身份特征"""
        _, id_feat, _ = self.forward(x, return_separate=True)
        return id_feat


# ============================================
#  ArcFace 分类头
# ============================================

class ArcFaceHead(nn.Module):
    """
    ArcFace 分类头
    
    用于训练时的分类监督
    """
    
    def __init__(
        self,
        embedding_size: int,
        num_classes: int,
        scale: float = 32.0,
        margin: float = 0.5,
        easy_margin: bool = False
    ):
        super().__init__()
        
        self.embedding_size = embedding_size
        self.num_classes = num_classes
        self.scale = scale
        self.margin = margin
        self.easy_margin = easy_margin
        
        # 权重矩阵
        self.kernel = nn.Parameter(torch.Tensor(num_classes, embedding_size))
        nn.init.xavier_uniform_(self.kernel)
        
        # cos(margin) 和 sin(margin)
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin
    
    def forward(
        self,
        features: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            features: (B, embedding_size) 归一化特征
            labels: (B,) 标签 (训练时需要)
            
        Returns:
            output: (B, num_classes) 分类 logits
        """
        # 归一化
        features_norm = F.normalize(features, p=2, dim=1)
        kernel_norm = F.normalize(self.kernel, p=2, dim=1)
        
        # 计算余弦相似度
        cos_theta = F.linear(features_norm, kernel_norm)
        cos_theta = cos_theta.clamp(-1, 1)
        
        if labels is not None and self.training:
            # 应用 ArcFace margin
            sin_theta = torch.sqrt(1.0 - cos_theta.pow(2))
            cos_theta_m = cos_theta * self.cos_m - sin_theta * self.sin_m
            
            # 处理边界情况
            if self.easy_margin:
                cos_theta_m = torch.where(
                    cos_theta > 0, cos_theta_m, cos_theta
                )
            else:
                cos_theta_m = torch.where(
                    cos_theta > self.th, cos_theta_m, cos_theta - self.mm
                )
            
            # One-hot
            one_hot = F.one_hot(labels, num_classes=self.num_classes).float()
            
            # 应用 margin
            output = cos_theta * (1 - one_hot) + cos_theta_m * one_hot
        else:
            output = cos_theta
        
        # 缩放
        output *= self.scale
        
        return output


# ============================================
# 属性分类头 (辅助监督)
# ============================================

class AttributeHead(nn.Module):
    """
    属性分类头
    
    用于属性预测的辅助监督
    """
    
    def __init__(
        self,
        attr_dim: int,
        num_attributes: int = 5,  # 姿态、光照、年龄、性别、遮挡
        attr_classes: Optional[List[int]] = None
    ):
        super().__init__()
        
        self.attr_dim = attr_dim
        self.num_attributes = num_attributes
        
        if attr_classes is None:
            # 默认每个属性 10 类
            attr_classes = [10] * num_attributes
        
        self.attr_classes = attr_classes
        
        # 为每个属性创建分类头
        self.attribute_classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(attr_dim, 64),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(64, num_cls)
            )
            for num_cls in attr_classes
        ])
    
    def forward(
        self,
        attr_feat: torch.Tensor
    ) -> List[torch.Tensor]:
        """
        Args:
            attr_feat: (B, attr_dim) 属性特征
            
        Returns:
            attr_logits: 各属性的分类 logits 列表
        """
        return [clf(attr_feat) for clf in self.attribute_classifiers]


# ============================================
# 完整识别头
# ============================================

class RecognitionHead(nn.Module):
    """
    完整识别头
    
    包含:
    - 身份解耦
    - ArcFace 分类
    - 属性预测
    """
    
    def __init__(
        self,
        in_channels: int = 256,
        id_dim: int = 409,
        attr_dim: int = 103,
        num_classes: int = 100000,
        num_attributes: int = 5,
        scale: float = 32.0,
        margin: float = 0.5,
        dropout: float = 0.4
    ):
        super().__init__()
        
        # 身份解耦头
        self.id_head = IdentityDisentangledHead(
            in_channels=in_channels,
            id_dim=id_dim,
            attr_dim=attr_dim,
            dropout=dropout
        )
        
        # ArcFace 分类头 (仅用于身份特征)
        self.arcface_head = ArcFaceHead(
            embedding_size=id_dim,
            num_classes=num_classes,
            scale=scale,
            margin=margin
        )
        
        # 属性头
        self.attr_head = AttributeHead(
            attr_dim=attr_dim,
            num_attributes=num_attributes
        )
    
    def forward(
        self,
        x: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: (B, C, H, W) 输入特征
            labels: (B,) 身份标签
            
        Returns:
            outputs: 包含各项输出的字典
        """
        # 身份解耦
        features, id_feat, attr_feat = self.id_head(x, return_separate=True)
        
        # ArcFace 分类
        arcface_logits = self.arcface_head(id_feat, labels)
        
        # 属性预测
        attr_logits = self.attr_head(attr_feat)
        
        return {
            'features': features,
            'id_features': id_feat,
            'attr_features': attr_feat,
            'arcface_logits': arcface_logits,
            'attr_logits': attr_logits,
        }


# ============================================
# 工厂函数
# ============================================

def build_recognition_head(
    head_type: str = "disentangled",
    **kwargs
) -> nn.Module:
    """
    构建识别头
    
    Args:
        head_type: 头类型
        **kwargs: 配置参数
        
    Returns:
        识别头模块
    """
    if head_type == "disentangled":
        return RecognitionHead(**kwargs)
    elif head_type == "identity":
        return IdentityDisentangledHead(**kwargs)
    elif head_type == "arcface":
        return ArcFaceHead(**kwargs)
    else:
        raise ValueError(f"Unknown head type: {head_type}")


# 导入 math
import math

if __name__ == "__main__":
    # 测试
    head = RecognitionHead(
        in_channels=256,
        id_dim=409,
        attr_dim=103,
        num_classes=10000,
        num_attributes=5
    )
    
    x = torch.randn(4, 256, 7, 7)
    labels = torch.randint(0, 10000, (4,))
    
    outputs = head(x, labels)
    
    print("Output shapes:")
    for key, value in outputs.items():
        if isinstance(value, list):
            print(f"  {key}: {[v.shape for v in value]}")
        else:
            print(f"  {key}: {value.shape}")
    
    # 统计参数量
    total_params = sum(p.numel() for p in head.parameters())
    print(f"\nTotal parameters: {total_params / 1e6:.2f}M")
