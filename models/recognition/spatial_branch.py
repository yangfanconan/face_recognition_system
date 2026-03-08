"""
DDFD-Rec 识别模型 - 空域分支网络

提取人脸纹理、边缘等空间结构特征
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple

from models.common import ConvBNAct, SEBlock, make_divisible


# ============================================
# 基础残差块
# ============================================

class BasicBlock(nn.Module):
    """
    基础残差块 (ResNet-style)
    
    Conv -> BN -> ReLU -> Conv -> BN -> (+ shortcut) -> ReLU
    """
    
    expansion: int = 1
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[nn.Module] = None,
        activation: str = "ReLU"
    ):
        super().__init__()
        
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        
        width = int(out_channels * (base_width / 64.0)) * groups
        
        self.conv1 = nn.Conv2d(
            in_channels, width, kernel_size=3, stride=stride,
            padding=dilation, groups=groups, dilation=dilation, bias=False
        )
        self.bn1 = norm_layer(width)
        
        self.conv2 = nn.Conv2d(
            width, out_channels * self.expansion, kernel_size=3,
            stride=1, padding=dilation, groups=groups, dilation=dilation, bias=False
        )
        self.bn2 = norm_layer(out_channels * self.expansion)
        
        self.downsample = downsample
        self.stride = stride
        
        if activation == "ReLU":
            self.act = nn.ReLU(inplace=True)
        elif activation == "SiLU":
            self.act = nn.SiLU(inplace=True)
        else:
            self.act = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.act(out)
        
        return out


class Bottleneck(nn.Module):
    """
    Bottleneck 残差块 (ResNet-style)
    
    Conv 1x1 -> Conv 3x3 -> Conv 1x1
    """
    
    expansion: int = 4
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[nn.Module] = None,
        activation: str = "ReLU"
    ):
        super().__init__()
        
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        
        width = int(out_channels * (base_width / 64.0)) * groups
        
        self.conv1 = nn.Conv2d(in_channels, width, 1, bias=False)
        self.bn1 = norm_layer(width)
        
        self.conv2 = nn.Conv2d(
            width, width, 3, stride, dilation, dilation, groups, bias=False
        )
        self.bn2 = norm_layer(width)
        
        self.conv3 = nn.Conv2d(width, out_channels * self.expansion, 1, bias=False)
        self.bn3 = norm_layer(out_channels * self.expansion)
        
        self.downsample = downsample
        self.stride = stride
        
        if activation == "ReLU":
            self.act = nn.ReLU(inplace=True)
        elif activation == "SiLU":
            self.act = nn.SiLU(inplace=True)
        else:
            self.act = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.act(out)
        
        return out


# ============================================
# 空域分支主干网络
# ============================================

class SpatialBranch(nn.Module):
    """
    空域特征提取分支
    
    基于 ResNet 架构，针对人脸特征优化
    """
    
    def __init__(
        self,
        block: type = BasicBlock,
        layers: List[int] = [2, 2, 2, 2],
        channels: List[int] = [64, 128, 256, 512],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[nn.Module] = None,
        activation: str = "ReLU",
        use_se: bool = False,
        dropout: float = 0.0
    ):
        super().__init__()
        
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        
        self._norm_layer = norm_layer
        self.in_channels = channels[0]
        self.dilation = 1
        self.use_se = use_se
        self.groups = groups
        self.base_width = width_per_group
        
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                f"replace_stride_with_dilation should have 3 values, "
                f"got {replace_stride_with_dilation}"
            )
        
        # Stem (针对 112x112 输入优化)
        self.conv1 = nn.Conv2d(3, self.in_channels, 7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.in_channels)
        
        if activation == "ReLU":
            self.act = nn.ReLU(inplace=True)
        elif activation == "SiLU":
            self.act = nn.SiLU(inplace=True)
        else:
            self.act = nn.ReLU(inplace=True)
        
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Stages
        self.layer1 = self._make_layer(
            block, channels[0], layers[0], stride=1
        )
        self.layer2 = self._make_layer(
            block, channels[1], layers[1], stride=2,
            dilate=replace_stride_with_dilation[0]
        )
        self.layer3 = self._make_layer(
            block, channels[2], layers[2], stride=2,
            dilate=replace_stride_with_dilation[1]
        )
        self.layer4 = self._make_layer(
            block, channels[3], layers[3], stride=2,
            dilate=replace_stride_with_dilation[2]
        )
        
        # SE 注意力 (可选)
        if use_se:
            self.se_layer3 = SEBlock(channels[2] * block.expansion)
            self.se_layer4 = SEBlock(channels[3] * block.expansion)
        else:
            self.se_layer3 = nn.Identity()
            self.se_layer4 = nn.Identity()
        
        # Dropout
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        
        # 初始化
        self._init_weights(zero_init_residual)
    
    def _make_layer(
        self,
        block: type,
        channels: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        
        if dilate:
            self.dilation *= stride
            stride = 1
        
        if stride != 1 or self.in_channels != channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels, channels * block.expansion,
                    1, stride, bias=False
                ),
                norm_layer(channels * block.expansion),
            )
        
        layers = [
            block(
                self.in_channels, channels, stride, downsample,
                groups=self.groups, base_width=self.base_width,
                dilation=previous_dilation, norm_layer=norm_layer
            )
        ]
        
        self.in_channels = channels * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.in_channels, channels,
                    groups=self.groups, base_width=self.base_width,
                    dilation=self.dilation, norm_layer=norm_layer
                )
            )
        
        return nn.Sequential(*layers)
    
    def _init_weights(self, zero_init_residual: bool = False) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.zeros_(m.bn3.weight)
                elif isinstance(m, BasicBlock):
                    nn.init.zeros_(m.bn2.weight)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        前向传播
        
        Args:
            x: (B, 3, H, W) 输入图像
            
        Returns:
            features: (feat1, feat2, feat3, feat4) 多尺度特征
        """
        # Stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.maxpool(x)
        
        # Stage 1
        feat1 = self.layer1(x)  # (B, C1, H/4, W/4)
        
        # Stage 2
        feat2 = self.layer2(feat1)  # (B, C2, H/8, W/8)
        
        # Stage 3 + SE
        feat3 = self.layer3(feat2)  # (B, C3, H/16, W/16)
        feat3 = self.se_layer3(feat3)
        feat3 = self.dropout(feat3)
        
        # Stage 4 + SE
        feat4 = self.layer4(feat3)  # (B, C4, H/32, W/32)
        feat4 = self.se_layer4(feat4)
        feat4 = self.dropout(feat4)
        
        return (feat1, feat2, feat3, feat4)


# ============================================
# 轻量级空域分支 (用于移动端)
# ============================================

class SpatialBranchTiny(nn.Module):
    """轻量级空域分支"""
    
    def __init__(
        self,
        channels: List[int] = [32, 64, 128, 256],
        activation: str = "SiLU"
    ):
        super().__init__()
        
        # Stem
        self.stem = nn.Sequential(
            ConvBNAct(3, channels[0], 3, 2, 1, activation=activation),
            ConvBNAct(channels[0], channels[1], 3, 2, 1, activation=activation),
        )
        
        # Stages
        self.stage1 = nn.Sequential(
            BasicBlock(channels[1], channels[1], stride=1),
            BasicBlock(channels[1], channels[1], stride=1),
        )
        
        self.stage2 = nn.Sequential(
            ConvBNAct(channels[1], channels[2], 3, 2, 1, activation=activation),
            BasicBlock(channels[2], channels[2], stride=1),
        )
        
        self.stage3 = nn.Sequential(
            ConvBNAct(channels[2], channels[3], 3, 2, 1, activation=activation),
            BasicBlock(channels[3], channels[3], stride=1),
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        x = self.stem(x)
        feat1 = self.stage1(x)
        feat2 = self.stage2(feat1)
        feat3 = self.stage3(feat2)
        return (feat1, feat2, feat3)


# ============================================
# 工厂函数
# ============================================

def build_spatial_branch(
    model_type: str = "resnet18",
    **kwargs
) -> nn.Module:
    """
    构建空域分支
    
    Args:
        model_type: 模型类型
        **kwargs: 配置参数
        
    Returns:
        空域分支模块
    """
    # ResNet 配置
    resnet_configs = {
        'resnet18': (BasicBlock, [2, 2, 2, 2]),
        'resnet34': (BasicBlock, [3, 4, 6, 3]),
        'resnet50': (Bottleneck, [3, 4, 6, 3]),
        'resnet101': (Bottleneck, [3, 4, 23, 3]),
    }
    
    if model_type in resnet_configs:
        block, layers = resnet_configs[model_type]
        return SpatialBranch(block=block, layers=layers, **kwargs)
    elif model_type == "tiny":
        return SpatialBranchTiny(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    # 测试
    model = SpatialBranch(block=BasicBlock, layers=[2, 2, 2, 2])
    model.eval()
    
    x = torch.randn(2, 3, 112, 112)
    
    with torch.no_grad():
        features = model(x)
    
    print("Input shape:", x.shape)
    for i, feat in enumerate(features):
        print(f"Feature {i+1} shape: {feat.shape}")
    
    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params / 1e6:.2f}M")
