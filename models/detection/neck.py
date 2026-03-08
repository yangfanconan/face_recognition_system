"""
DKGA-Det 检测模型 - 颈部特征融合网络

基于 BiFPN-Lite 架构，支持 P2 层小目标检测
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional

from models.detection.backbone import ConvBNAct
from models.common import build_attention


# ============================================
# PANet/FPN 基础模块
# ============================================

class FPNBlock(nn.Module):
    """
    标准 FPN 上采样融合模块
    """
    
    def __init__(self, in_channels: List[int], out_channels: int = 256):
        super().__init__()
        
        self.num_levels = len(in_channels)
        
        # 横向连接 (1x1 卷积)
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, 1, bias=False)
            for in_ch in in_channels
        ])
        
        # 输出卷积
        self.out_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
            for _ in range(self.num_levels)
        ])
        
        self._init_weights()
    
    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    
    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Args:
            features: [P3, P4, P5] 多尺度特征
            
        Returns:
            fused_features: [F3, F4, F5] 融合后的特征
        """
        # 横向连接
        lateral_features = [
            lateral_conv(feat)
            for lateral_conv, feat in zip(self.lateral_convs, features)
        ]
        
        # 自顶向下路径
        fused = []
        for i in range(self.num_levels - 1, -1, -1):
            if i == self.num_levels - 1:
                # 最顶层
                fused.append(lateral_features[i])
            else:
                # 上采样 + 融合
                upsampled = F.interpolate(
                    fused[-1],
                    size=lateral_features[i].shape[2:],
                    mode='nearest'
                )
                fused.append(lateral_features[i] + upsampled)
        
        # 反转顺序 (从 P3 到 P5)
        fused = fused[::-1]
        
        # 输出卷积 (消除混叠效应)
        output = [
            out_conv(feat)
            for out_conv, feat in zip(self.out_convs, fused)
        ]
        
        return output


class PANetBlock(nn.Module):
    """
    PANet 下采样增强模块
    """
    
    def __init__(self, in_channels: int = 256):
        super().__init__()
        
        # 自底向上路径的卷积
        self.downsample_convs = nn.ModuleList([
            ConvBNAct(in_channels, in_channels, 3, 2, 1)
            for _ in range(3)  # 最多支持 4 层
        ])
    
    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Args:
            features: [F3, F4, F5] 自顶向下后的特征
            
        Returns:
            enhanced_features: 增强后的特征
        """
        enhanced = [features[0]]  # P3
        
        for i in range(1, len(features)):
            downsampled = self.downsample_convs[i - 1](enhanced[-1])
            enhanced.append(downsampled + features[i])
        
        return enhanced


# ============================================
# BiFPN (Bidirectional FPN)
# ============================================

class BiFPNBlock(nn.Module):
    """
    BiFPN 双向特征金字塔
    
    Reference: EfficientDet (https://arxiv.org/abs/1911.09070)
    
    注意：第一层接收多尺度输入，输出统一通道数
    后续层接收统一通道数输入，输出统一通道数
    """
    
    def __init__(
        self,
        in_channels: List[int],
        out_channels: int = 256,
        num_levels: int = 4,
        attention_type: str = "se",
        is_first_layer: bool = False
    ):
        super().__init__()
        
        self.num_levels = num_levels
        self.is_first_layer = is_first_layer
        self.out_channels = out_channels
        
        # 横向连接
        # 第一层：使用原始 in_channels
        # 后续层：所有输入都是 out_channels
        if is_first_layer:
            lateral_in_channels = in_channels[:num_levels]
        else:
            lateral_in_channels = [out_channels] * num_levels
        
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, 1, bias=False)
            for in_ch in lateral_in_channels
        ])
        
        # 自顶向下路径
        self.td_convs = nn.ModuleList([
            ConvBNAct(out_channels, out_channels, 3, 1, 1)
            for _ in range(num_levels - 1)
        ])
        
        # 自底向上路径
        self.bd_convs = nn.ModuleList([
            ConvBNAct(out_channels, out_channels, 3, 1, 1)
            for _ in range(num_levels - 1)
        ])
        
        # 下采样
        self.downsample = nn.Conv2d(out_channels, out_channels, 3, 2, 1, bias=False)
        
        # 注意力权重 (可学习)
        self.attention_weights = nn.Parameter(torch.ones(num_levels))
        self.attention_type = attention_type
        
        if attention_type == "se":
            self.attention = build_attention("se", out_channels)
        elif attention_type == "eca":
            self.attention = build_attention("eca", out_channels)
        else:
            self.attention = nn.Identity()
        
        self._init_weights()
    
    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    
    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Args:
            features: [P2, P3, P4, P5] 输入特征
            
        Returns:
            output: [F2, F3, F4, F5] BiFPN 输出
        """
        # 横向连接
        feats = [conv(f) for conv, f in zip(self.lateral_convs, features)]
        
        # 自顶向下
        td_feats = [feats[-1]]  # 最顶层
        for i in range(self.num_levels - 2, -1, -1):
            upsampled = F.interpolate(td_feats[-1], size=feats[i].shape[2:], mode='nearest')
            # 加权融合
            weight = torch.sigmoid(self.attention_weights[i])
            fused = feats[i] + weight * upsampled
            td_feats.append(self.td_convs[self.num_levels - 2 - i](fused))
        
        td_feats = td_feats[::-1]  # 反转
        
        # 自底向上
        bd_feats = [td_feats[0]]
        for i in range(1, self.num_levels):
            if i < len(td_feats):
                downsampled = self.downsample(bd_feats[-1])
                weight = torch.sigmoid(self.attention_weights[i])
                fused = td_feats[i] + weight * downsampled
                bd_feats.append(self.bd_convs[i - 1](fused))
        
        # 应用注意力
        output = [self.attention(f) for f in bd_feats]
        
        return output


# ============================================
# BiFPN-Lite (简化版)
# ============================================

class BiFPNLite(nn.Module):
    """
    BiFPN-Lite 简化版

    减少层数和通道数，平衡精度和速度
    """

    def __init__(
        self,
        in_channels: List[int] = [256, 512, 1024],
        out_channels: int = 256,
        num_layers: int = 2,
        use_p2: bool = True,
        attention: bool = True
    ):
        super().__init__()

        self.use_p2 = use_p2
        self.out_channels = out_channels

        # P2 层上采样
        if use_p2:
            self.p2_conv = nn.Conv2d(in_channels[0], out_channels, 1, bias=False)
            # P2 输出 out_channels, 所以 P2 通道也是 out_channels
            self.bifpn_in_channels = [out_channels] + in_channels  # [256, 256, 512, 1024]
            self.num_levels = len(self.bifpn_in_channels)
        else:
            self.bifpn_in_channels = in_channels
            self.num_levels = len(in_channels)

        # BiFPN 层 - 第一层使用原始通道，后续层使用统一通道
        self.bifpn_layers = nn.ModuleList()
        for i in range(num_layers):
            bifpn = BiFPNBlock(
                in_channels=self.bifpn_in_channels,
                out_channels=out_channels,
                num_levels=self.num_levels,
                attention_type="se" if attention else None,
                is_first_layer=(i == 0)  # 只有第一层使用原始通道
            )
            self.bifpn_layers.append(bifpn)

        # 输出卷积
        self.out_convs = nn.ModuleList([
            ConvBNAct(out_channels, out_channels, 3, 1, 1)
            for _ in range(self.num_levels)
        ])
    
    def forward(self, features: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, ...]:
        """
        Args:
            features: (P3, P4, P5) 来自 backbone
            
        Returns:
            output: (P2/P3, P3/P4, P4/P5, P5) 融合后的特征
        """
        # 处理 P2
        if self.use_p2:
            p3_upsampled = F.interpolate(features[0], scale_factor=2, mode='nearest')
            p2 = self.p2_conv(p3_upsampled)
            feats = [p2] + list(features)
        else:
            feats = list(features)
        
        # 通过 BiFPN 层
        for bifpn in self.bifpn_layers:
            feats = bifpn(feats)
        
        # 输出卷积
        output = tuple(conv(f) for conv, f in zip(self.out_convs, feats))
        
        return output


# ============================================
# 小目标增强金字塔
# ============================================

class SmallFaceFPN(nn.Module):
    """
    针对小目标人脸优化的 FPN
    
    特点:
    - 增加 P2 层 (80x80 @ 640 输入)
    - 高分辨率特征保留
    - 浅层特征增强
    """
    
    def __init__(
        self,
        in_channels: List[int] = [256, 512, 1024],
        out_channels: int = 256,
        num_fpn_layers: int = 3
    ):
        super().__init__()
        
        # P2 层处理
        self.p2_conv = nn.Sequential(
            nn.Conv2d(in_channels[0], out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True)
        )
        
        # P3 层处理
        self.p3_conv = nn.Sequential(
            nn.Conv2d(in_channels[0], out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True)
        )
        
        # P4 层处理
        self.p4_conv = nn.Sequential(
            nn.Conv2d(in_channels[1], out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True)
        )
        
        # P5 层处理
        self.p5_conv = nn.Sequential(
            nn.Conv2d(in_channels[2], out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True)
        )
        
        # 自顶向下路径
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        
        self.p4_top = ConvBNAct(out_channels, out_channels, 3, 1, 1)
        self.p3_top = ConvBNAct(out_channels, out_channels, 3, 1, 1)
        self.p2_top = ConvBNAct(out_channels, out_channels, 3, 1, 1)
        
        # 自底向上路径
        self.p3_down = ConvBNAct(out_channels, out_channels, 3, 2, 1)
        self.p4_down = ConvBNAct(out_channels, out_channels, 3, 2, 1)
        
        # 输出卷积
        self.p2_out = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.p3_out = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.p4_out = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.p5_out = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        
        self._init_weights()
    
    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    
    def forward(self, features: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, ...]:
        """
        Args:
            features: (P3, P4, P5)
            
        Returns:
            (P2, P3, P4, P5)
        """
        p3, p4, p5 = features
        
        # 横向连接
        p5 = self.p5_conv(p5)
        p4 = self.p4_conv(p4)
        p3 = self.p3_conv(p3)
        p2 = self.p2_conv(self.upsample(p3))
        
        # 自顶向下
        p4 = self.p4_top(p4 + self.upsample(p5))
        p3 = self.p3_top(p3 + self.upsample(p4))
        p2 = self.p2_top(p2 + self.upsample(p3))
        
        # 自底向上
        p3 = self.p3_down(p2) + p3
        p4 = self.p4_down(p3) + p4
        
        # 输出卷积
        p2_out = self.p2_out(p2)
        p3_out = self.p3_out(p3)
        p4_out = self.p4_out(p4)
        p5_out = self.p5_out(p5)
        
        return (p2_out, p3_out, p4_out, p5_out)


# ============================================
# 工厂函数
# ============================================

def build_neck(
    name: str = "bifpn_lite",
    **kwargs
) -> nn.Module:
    """
    构建 Neck 模块
    
    Args:
        name: Neck 类型
        **kwargs: 配置参数
        
    Returns:
        Neck 模块
    """
    necks = {
        'fpn': FPNBlock,
        'panet': PANetBlock,
        'bifpn': BiFPNBlock,
        'bifpn_lite': BiFPNLite,
        'small_face_fpn': SmallFaceFPN,
    }
    
    if name not in necks:
        raise ValueError(f"Unknown neck: {name}. Available: {list(necks.keys())}")
    
    return necks[name](**kwargs)


if __name__ == "__main__":
    # 测试
    features = [
        torch.randn(2, 256, 80, 80),   # P3
        torch.randn(2, 512, 40, 40),   # P4
        torch.randn(2, 1024, 20, 20),  # P5
    ]
    
    model = BiFPNLite(in_channels=[256, 512, 1024], out_channels=256, use_p2=True)
    output = model(features)
    
    print("Input shapes:", [f.shape for f in features])
    print("Output shapes:", [o.shape for o in output])
