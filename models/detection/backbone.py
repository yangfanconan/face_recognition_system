"""
DKGA-Det 检测模型 - 主干网络

基于 CSPDarknet 架构，集成 DCNv2 可变形卷积
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional

from models.common import DeformConv2d, make_divisible


# ============================================
# 基础模块
# ============================================

class ConvBNAct(nn.Module):
    """Conv + BN + Activation"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        groups: int = 1,
        activation: str = "SiLU",
        use_dcn: bool = False,
        use_gn: bool = False
    ):
        super().__init__()

        if use_dcn:
            self.conv = DeformConv2d(
                in_channels, out_channels, kernel_size, stride, padding,
                deformable_groups=1, use_mask=True
            )
        else:
            self.conv = nn.Conv2d(
                in_channels, out_channels, kernel_size, stride, padding,
                groups=groups, bias=False
            )

        if use_gn:
            self.bn = nn.GroupNorm(32, out_channels)
        else:
            self.bn = nn.BatchNorm2d(out_channels)

        if activation == "ReLU":
            self.act = nn.ReLU(inplace=True)
        elif activation == "SiLU":
            self.act = nn.SiLU(inplace=True)
        elif activation == "LeakyReLU":
            self.act = nn.LeakyReLU(0.1, inplace=True)
        elif activation == "GELU":
            self.act = nn.GELU()
        else:
            self.act = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class Focus(nn.Module):
    """
    Focus 模块 - 将空间信息转移到通道维度
    
    用于 YOLOv5 的 Stem
    """
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 1):
        super().__init__()
        
        self.conv = ConvBNAct(in_channels * 4, out_channels, kernel_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 隔像素采样
        # x: (B, C, H, W) -> (B, 4C, H/2, W/2)
        patch_top_left = x[..., ::2, ::2]
        patch_top_right = x[..., ::2, 1::2]
        patch_bot_left = x[..., 1::2, ::2]
        patch_bot_right = x[..., 1::2, 1::2]
        
        x = torch.cat([
            patch_top_left,
            patch_top_right,
            patch_bot_left,
            patch_bot_right,
        ], dim=1)
        
        return self.conv(x)


class Bottleneck(nn.Module):
    """标准 Bottleneck 模块"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        shortcut: bool = True,
        expansion: float = 0.5,
        activation: str = "SiLU"
    ):
        super().__init__()
        
        hidden_channels = int(out_channels * expansion)
        
        self.conv1 = ConvBNAct(in_channels, hidden_channels, 1, 1, 0, activation=activation)
        self.conv2 = ConvBNAct(hidden_channels, out_channels, 3, 1, 1, activation=activation)
        
        self.use_shortcut = shortcut and in_channels == out_channels
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv2(self.conv1(x))
        if self.use_shortcut:
            return x + out
        return out


class BottleneckCSP(nn.Module):
    """
    CSP Bottleneck - Cross Stage Partial
    
    用于 CSPDarknet
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_bottlenecks: int = 1,
        expansion: float = 0.5,
        activation: str = "SiLU"
    ):
        super().__init__()
        
        hidden_channels = int(out_channels * expansion)
        
        # CSP 分支 1
        self.conv1 = ConvBNAct(in_channels, hidden_channels, 1, 1, 0, activation=activation)
        
        # CSP 分支 2
        self.conv2 = ConvBNAct(in_channels, hidden_channels, 1, 1, 0, activation=activation)
        
        # Bottlenecks
        self.bottlenecks = nn.Sequential(*[
            Bottleneck(hidden_channels, hidden_channels, shortcut=True, expansion=1.0, activation=activation)
            for _ in range(num_bottlenecks)
        ])
        
        # 融合
        self.conv3 = ConvBNAct(hidden_channels * 2, out_channels, 1, 1, 0, activation=activation)
        
        # BN + SiLU
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 分支 1
        x1 = self.conv1(x)
        x1 = self.bottlenecks(x1)
        
        # 分支 2
        x2 = self.conv2(x)
        
        # 拼接
        x = torch.cat([x1, x2], dim=1)
        
        # 融合
        x = self.conv3(x)
        x = self.bn(x)
        x = self.act(x)
        
        return x


# ============================================
# DCNv2 Bottleneck
# ============================================

class DCNBottleneck(nn.Module):
    """带 DCNv2 的 Bottleneck"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        shortcut: bool = True,
        expansion: float = 0.5,
        activation: str = "SiLU"
    ):
        super().__init__()
        
        hidden_channels = int(out_channels * expansion)
        
        # 第二个卷积使用 DCNv2
        self.conv1 = ConvBNAct(in_channels, hidden_channels, 1, 1, 0, activation=activation)
        self.conv2 = ConvBNAct(hidden_channels, out_channels, 3, 1, 1, activation=activation, use_dcn=True)
        
        self.use_shortcut = shortcut and in_channels == out_channels
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv2(self.conv1(x))
        if self.use_shortcut:
            return x + out
        return out


class DCNCSPBlock(nn.Module):
    """带 DCNv2 的 CSP Block"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_bottlenecks: int = 1,
        expansion: float = 0.5,
        activation: str = "SiLU"
    ):
        super().__init__()
        
        hidden_channels = int(out_channels * expansion)
        
        self.conv1 = ConvBNAct(in_channels, hidden_channels, 1, 1, 0, activation=activation)
        self.conv2 = ConvBNAct(in_channels, hidden_channels, 1, 1, 0, activation=activation)
        
        # 使用 DCN Bottleneck
        self.bottlenecks = nn.Sequential(*[
            DCNBottleneck(hidden_channels, hidden_channels, shortcut=True, expansion=1.0, activation=activation)
            for _ in range(num_bottlenecks)
        ])
        
        self.conv3 = ConvBNAct(hidden_channels * 2, out_channels, 1, 1, 0, activation=activation)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.conv1(x)
        x1 = self.bottlenecks(x1)
        x2 = self.conv2(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.conv3(x)
        x = self.bn(x)
        x = self.act(x)
        return x


# ============================================
# CSPDarknet Backbone
# ============================================

class CSPDarknet(nn.Module):
    """
    CSPDarknet 主干网络
    
    Args:
        depths: 每个阶段的 Bottleneck 数量 [depth1, depth2, depth3, depth4]
        channels: 每个阶段的输出通道数 [ch1, ch2, ch3, ch4, ch5]
        use_dcnv2: 是否使用 DCNv2
        dcnv2_stages: 使用 DCNv2 的阶段索引列表
        activation: 激活函数类型
    """
    
    def __init__(
        self,
        depths: List[int] = [3, 6, 6, 3],
        channels: List[int] = [64, 128, 256, 512, 1024],
        use_dcnv2: bool = True,
        dcnv2_stages: List[int] = [2, 3],
        activation: str = "SiLU",
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.num_stages = len(depths)
        self.channels = channels
        
        # Stem
        self.stem = nn.Sequential(
            ConvBNAct(3, channels[0], 6, 2, 2, activation=activation),
            ConvBNAct(channels[0], channels[1], 3, 2, 1, activation=activation),
        )
        
        # Stages
        self.stages = nn.ModuleList()
        
        for i in range(self.num_stages):
            in_ch = channels[i]
            out_ch = channels[i + 1]
            depth = depths[i]
            
            # 下采样
            downsample = ConvBNAct(in_ch, out_ch, 3, 2, 1, activation=activation)
            
            # CSP Block
            if use_dcnv2 and i in dcnv2_stages:
                stage_block = DCNCSPBlock(out_ch, out_ch, num_bottlenecks=depth, expansion=0.5, activation=activation)
            else:
                stage_block = BottleneckCSP(out_ch, out_ch, num_bottlenecks=depth, expansion=0.5, activation=activation)
            
            self.stages.append(nn.Sequential(downsample, stage_block))
        
        # Dropout
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        前向传播

        Returns:
            features: 多尺度特征图 (P3, P4, P5)
        """
        # Stem
        x = self.stem(x)  # x shape: (B, 128, H/4, W/4)

        # Stages
        features = []
        for i, stage in enumerate(self.stages):
            x = stage(x)
            # 输出 P3, P4, P5 (对应 stage 1, 2, 3)
            # Stage 0 输出 256 通道 (P3)
            # Stage 1 输出 512 通道 (P4)
            # Stage 2 输出 1024 通道 (P5)
            features.append(self.dropout(x))

        return tuple(features)  # (P3, P4, P5)


# ============================================
# 轻量级 Backbone (用于移动端)
# ============================================

class CSPDarknetTiny(nn.Module):
    """轻量级 CSPDarknet"""
    
    def __init__(
        self,
        channels: List[int] = [32, 64, 128, 256, 512],
        activation: str = "SiLU"
    ):
        super().__init__()
        
        # Stem
        self.stem = nn.Sequential(
            ConvBNAct(3, channels[0], 3, 2, 1, activation=activation),
            ConvBNAct(channels[0], channels[1], 3, 2, 1, activation=activation),
        )
        
        # Stages (简化版)
        self.stage1 = nn.Sequential(
            ConvBNAct(channels[1], channels[2], 3, 2, 1, activation=activation),
            BottleneckCSP(channels[2], channels[2], num_bottlenecks=3, activation=activation),
        )
        
        self.stage2 = nn.Sequential(
            ConvBNAct(channels[2], channels[3], 3, 2, 1, activation=activation),
            BottleneckCSP(channels[3], channels[3], num_bottlenecks=3, activation=activation),
        )
        
        self.stage3 = nn.Sequential(
            ConvBNAct(channels[3], channels[4], 3, 2, 1, activation=activation),
            BottleneckCSP(channels[4], channels[4], num_bottlenecks=1, activation=activation),
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        x = self.stem(x)
        p3 = self.stage1(x)
        p4 = self.stage2(p3)
        p5 = self.stage3(p4)
        return (p3, p4, p5)


# ============================================
# 工厂函数
# ============================================

def build_backbone(
    name: str = "cspdarknet",
    **kwargs
) -> nn.Module:
    """
    构建 Backbone
    
    Args:
        name: Backbone 名称
        **kwargs: 配置参数
        
    Returns:
        Backbone 模块
    """
    backbones = {
        'cspdarknet': CSPDarknet,
        'cspdarknet_tiny': CSPDarknetTiny,
    }
    
    if name not in backbones:
        raise ValueError(f"Unknown backbone: {name}. Available: {list(backbones.keys())}")
    
    return backbones[name](**kwargs)


if __name__ == "__main__":
    # 测试
    model = CSPDarknet(depths=[3, 6, 6, 3], channels=[64, 128, 256, 512, 1024])
    x = torch.randn(2, 3, 640, 640)
    features = model(x)
    
    print("Input shape:", x.shape)
    for i, feat in enumerate(features):
        print(f"P{i+3} shape: {feat.shape}")
    
    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params / 1e6:.2f}M")
