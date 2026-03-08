"""
注意力机制模块

包含:
- SE (Squeeze-and-Excitation)
- CBAM (Convolutional Block Attention Module)
- ECA (Efficient Channel Attention)
- FGA (Frequency Gated Attention, 本方案特色)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


# ============================================
# SE Attention (Squeeze-and-Excitation)
# ============================================

class SEBlock(nn.Module):
    """
    SE 注意力模块
    
    Reference: https://arxiv.org/abs/1709.01507
    """
    
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        
        # Squeeze
        y = self.avg_pool(x).view(B, C)
        
        # Excitation
        y = self.fc(y).view(B, C, 1, 1)
        
        # Scale
        return x * y


# ============================================
# CBAM (Convolutional Block Attention Module)
# ============================================

class CBAM(nn.Module):
    """
    CBAM 注意力模块
    
    Reference: https://arxiv.org/abs/1807.06521
    """
    
    def __init__(
        self,
        channels: int,
        reduction: int = 16,
        kernel_size: int = 7
    ):
        super().__init__()
        
        # 通道注意力
        self.channel_attention = ChannelAttention(channels, reduction)
        
        # 空间注意力
        self.spatial_attention = SpatialAttention(kernel_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class ChannelAttention(nn.Module):
    """通道注意力"""
    
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        
        # 平均池化和最大池化
        avg_out = self.avg_pool(x).view(B, C)
        max_out = self.max_pool(x).view(B, C)
        
        # 通过 FC 层
        avg_out = self.fc(avg_out).view(B, C, 1, 1)
        max_out = self.fc(max_out).view(B, C, 1, 1)
        
        # 融合
        out = avg_out + max_out
        out = self.sigmoid(out)
        
        return x * out


class SpatialAttention(nn.Module):
    """空间注意力"""
    
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        
        assert kernel_size in (3, 5, 7), "kernel_size must be 3, 5, or 7"
        padding = kernel_size // 2
        
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 沿通道维度拼接平均池化和最大池化
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        out = self.sigmoid(out)
        
        return x * out


# ============================================
# ECA (Efficient Channel Attention)
# ============================================

class ECABlock(nn.Module):
    """
    ECA 注意力模块 (无需降维)
    
    Reference: https://arxiv.org/abs/1910.03151
    """
    
    def __init__(self, channels: int, gamma: int = 2, b: int = 1):
        super().__init__()
        
        # 自适应计算 kernel_size
        t = int(abs((math.log2(channels) + b) / gamma))
        kernel_size = max(t, 3)
        padding = kernel_size // 2
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        
        # 全局平均池化
        y = self.avg_pool(x).view(B, 1, C)
        
        # 一维卷积捕捉跨通道交互
        y = self.conv(y).view(B, C, 1, 1)
        y = self.sigmoid(y)
        
        return x * y


# ============================================
# FGA (Frequency Gated Attention) - 本方案特色
# ============================================

class FrequencyGatedAttention(nn.Module):
    """
    频域门控注意力模块
    
    融合空域和频域特征，通过门控机制动态调整权重
    """
    
    def __init__(
        self,
        channels: int,
        freq_channels: Optional[int] = None,
        reduction: int = 4
    ):
        super().__init__()
        
        freq_channels = freq_channels or channels
        
        # 空域特征处理
        self.spatial_conv = nn.Conv2d(channels, channels, 1, bias=False)
        
        # 频域特征处理
        self.freq_conv = nn.Conv2d(freq_channels, channels, 1, bias=False)
        
        # 门控生成器
        self.gate_generator = nn.Sequential(
            nn.Conv2d(channels * 2, channels // reduction, 1, bias=False),
            nn.BatchNorm2d(channels // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels * 2, 1, bias=False),
            nn.Sigmoid()
        )
        
        # 频域变换 (DCT)
        self.dct_transform = DCTTransform(channels)
        
        # 初始化
        self._init_weights()
    
    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(
        self,
        x_spatial: torch.Tensor,
        x_freq: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x_spatial: (B, C, H, W) 空域特征
            x_freq: (B, C_freq, H, W) 频域特征，None 则从 x_spatial 计算
            
        Returns:
            x_fused: (B, C, H, W) 融合特征
        """
        B, C, H, W = x_spatial.shape
        
        # 空域特征处理
        x_spatial_out = self.spatial_conv(x_spatial)
        
        # 频域特征处理
        if x_freq is None:
            x_freq = self.dct_transform(x_spatial)
        x_freq_out = self.freq_conv(x_freq)
        
        # 生成门控权重
        concat = torch.cat([x_spatial_out, x_freq_out], dim=1)
        gate = self.gate_generator(concat)  # (B, 2C, H, W)
        
        # 分割门控
        gate_spatial = gate[:, :C, :, :]
        gate_freq = gate[:, C:, :, :]
        
        # 归一化门控权重
        gate_sum = gate_spatial + gate_freq + 1e-6
        gate_spatial = gate_spatial / gate_sum
        gate_freq = gate_freq / gate_sum
        
        # 加权融合
        x_fused = gate_spatial * x_spatial_out + gate_freq * x_freq_out
        
        return x_fused


# ============================================
# DCT 变换工具
# ============================================

class DCTTransform(nn.Module):
    """
    离散余弦变换 (DCT) 模块
    
    用于提取频域特征
    """
    
    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        对输入进行 DCT 变换
        
        Args:
            x: (B, C, H, W)
            
        Returns:
            x_dct: (B, C, H, W) 频域系数
        """
        B, C, H, W = x.shape
        
        # 简化的 DCT 实现
        # 实际生产环境应使用优化的 DCT 库
        
        # 对每个通道进行 DCT
        x_dct = self._dct_2d(x)
        
        return x_dct
    
    def _dct_2d(self, x: torch.Tensor) -> torch.Tensor:
        """2D DCT 变换"""
        # 使用 FFT 近似 DCT
        x_fft = torch.fft.fft2(x)
        x_fft = torch.abs(x_fft)
        return x_fft
    
    def _idct_2d(self, x: torch.Tensor) -> torch.Tensor:
        """2D 逆 DCT 变换"""
        # 逆 FFT
        x_ifft = torch.fft.ifft2(x)
        return torch.real(x_ifft)


# ============================================
# 自注意力 (Self-Attention)
# ============================================

class SelfAttention(nn.Module):
    """
    自注意力模块 (Non-local)
    
    Reference: https://arxiv.org/abs/1711.07971
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        attention_type: str = "gaussian"
    ):
        super().__init__()
        
        out_channels = out_channels or in_channels
        self.attention_type = attention_type
        
        self.theta = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.phi = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.g = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        
        self.out_conv = nn.Conv2d(out_channels, in_channels, 1, bias=False)
        self.gamma = nn.Parameter(torch.zeros(1))
        
        if attention_type == "gaussian":
            self.operation = self._gaussian
        elif attention_type == "dot":
            self.operation = self._dot
        elif attention_type == "concat":
            self.operation = self._concat
        else:
            raise ValueError(f"Unknown attention_type: {attention_type}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        
        theta = self.theta(x)  # (B, C, H, W)
        phi = self.phi(x)  # (B, C, H, W)
        g = self.g(x)  # (B, C, H, W)
        
        # 计算注意力
        attention = self.operation(theta, phi, g)
        
        # 重塑回原尺寸
        attention = attention.view(B, C, H, W)
        attention = self.out_conv(attention)
        
        return x + self.gamma * attention
    
    def _gaussian(self, theta: torch.Tensor, phi: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """高斯嵌入"""
        B, C, H, W = theta.shape
        
        theta = theta.view(B, C, -1)  # (B, C, HW)
        phi = phi.view(B, C, -1)  # (B, C, HW)
        g = g.view(B, C, -1)  # (B, C, HW)
        
        # 计算相似度矩阵
        f = torch.matmul(theta.transpose(1, 2), phi)  # (B, HW, HW)
        f = F.softmax(f, dim=-1)
        
        # 应用注意力
        y = torch.matmul(f, g.transpose(1, 2))  # (B, HW, C)
        y = y.transpose(1, 2).contiguous()  # (B, C, HW)
        
        return y.view(B, C, H, W)
    
    def _dot(self, theta: torch.Tensor, phi: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """点积注意力"""
        B, C, H, W = theta.shape
        
        theta = theta.view(B, C, -1)
        phi = phi.view(B, C, -1)
        g = g.view(B, C, -1)
        
        f = torch.matmul(theta.transpose(1, 2), phi)
        f = f / (C ** 0.5)
        f = F.softmax(f, dim=-1)
        
        y = torch.matmul(f, g.transpose(1, 2))
        y = y.transpose(1, 2).contiguous()
        
        return y.view(B, C, H, W)
    
    def _concat(self, theta: torch.Tensor, phi: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """拼接注意力"""
        # 简化实现
        return self._gaussian(theta, phi, g)


# ============================================
# Coordinate Attention
# ============================================

class CoordinateAttention(nn.Module):
    """
    Coordinate Attention
    
    Reference: https://arxiv.org/abs/2103.02907
    """
    
    def __init__(self, channels: int, reduction: int = 32):
        super().__init__()
        
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        
        mid_channels = max(8, channels // reduction)
        
        self.conv1 = nn.Conv2d(channels, mid_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.act = nn.Hardswish(inplace=True)
        
        self.conv_h = nn.Conv2d(mid_channels, channels, 1, bias=False)
        self.conv_w = nn.Conv2d(mid_channels, channels, 1, bias=False)
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        
        # 分别沿 H 和 W 方向池化
        x_h = self.pool_h(x)  # (B, C, H, 1)
        x_w = self.pool_w(x)  # (B, C, 1, W)
        
        # 拼接并降维
        y = torch.cat([x_h, x_w], dim=2)  # (B, C, H+1, W) - 实际是 (B, C, H, 1) 和 (B, C, 1, W)
        y = y.transpose(1, 2).contiguous()  # 调整维度
        y = y.view(B, -1, H + W)  # 简化处理
        y = y.unsqueeze(-1)
        
        # 通过卷积
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        
        # 分割回 h 和 w 方向
        y_h, y_w = torch.split(y, [H, W], dim=2)
        y_h = y_h.view(B, -1, H, 1)
        y_w = y_w.view(B, -1, 1, W)
        
        # 生成注意力权重
        a_h = self.sigmoid(self.conv_h(y_h))
        a_w = self.sigmoid(self.conv_w(y_w))
        
        return x * a_h * a_w


# ============================================
# 注意力模块工厂
# ============================================

def build_attention(attention_type: str, channels: int, **kwargs) -> nn.Module:
    """
    构建注意力模块
    
    Args:
        attention_type: 注意力类型 ('se', 'cbam', 'eca', 'fga', 'coord', 'self')
        channels: 通道数
        **kwargs: 其他参数
        
    Returns:
        注意力模块
    """
    attention_types = {
        'se': SEBlock,
        'cbam': CBAM,
        'eca': ECABlock,
        'fga': FrequencyGatedAttention,
        'coord': CoordinateAttention,
        'self': SelfAttention,
    }
    
    if attention_type not in attention_types:
        raise ValueError(f"Unknown attention_type: {attention_type}. "
                        f"Available types: {list(attention_types.keys())}")
    
    return attention_types[attention_type](channels, **kwargs)


# 导入 math 用于 ECA
import math
