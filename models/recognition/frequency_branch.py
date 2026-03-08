"""
DDFD-Rec 识别模型 - 频域分支网络

通过 DCT 变换提取光照不变性频域特征
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple
import numpy as np

from models.common import ConvBNAct, SEBlock


# ============================================
# DCT 变换工具
# ============================================

class DCTTransform2D(nn.Module):
    """
    2D 离散余弦变换 (DCT-II)
    
    将空间域图像转换到频域
    """
    
    def __init__(self, height: int = 112, width: int = 112):
        super().__init__()
        
        self.height = height
        self.width = width
        
        # 预计算 DCT 基函数
        self.register_buffer('dct_basis_u', self._compute_dct_basis(height))
        self.register_buffer('dct_basis_v', self._compute_dct_basis(width))
    
    def _compute_dct_basis(self, n: int) -> torch.Tensor:
        """计算 1D DCT 基函数"""
        basis = torch.zeros(n, n)
        for k in range(n):
            for i in range(n):
                if k == 0:
                    basis[k, i] = 1.0 / np.sqrt(n)
                else:
                    basis[k, i] = np.sqrt(2.0 / n) * np.cos(
                        np.pi * k * (2 * i + 1) / (2 * n)
                    )
        return basis
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        对输入进行 2D DCT 变换
        
        Args:
            x: (B, C, H, W) 空间域图像
            
        Returns:
            x_dct: (B, C, H, W) 频域系数
        """
        B, C, H, W = x.shape
        
        # 重塑为 (B*C, H, W)
        x = x.view(B * C, H, W)
        
        # 2D DCT = 1D DCT 行 + 1D DCT 列
        # 行变换
        x_dct = torch.matmul(self.dct_basis_u[:H, :H], x)
        # 列变换
        x_dct = torch.matmul(x_dct, self.dct_basis_v[:W, :W].t())
        
        # 恢复形状
        x_dct = x_dct.view(B, C, H, W)
        
        return x_dct
    
    def inverse(self, x_dct: torch.Tensor) -> torch.Tensor:
        """
        逆 DCT 变换 (IDCT)
        
        Args:
            x_dct: (B, C, H, W) 频域系数
            
        Returns:
            x: (B, C, H, W) 空间域图像
        """
        B, C, H, W = x_dct.shape
        
        # 重塑
        x_dct = x_dct.view(B * C, H, W)
        
        # 2D IDCT
        x = torch.matmul(self.dct_basis_u[:H, :H].t(), x_dct)
        x = torch.matmul(x, self.dct_basis_v[:W, :W])
        
        x = x.view(B, C, H, W)
        
        return x


# ============================================
# 频域卷积模块
# ============================================

class FrequencyConvBlock(nn.Module):
    """
    频域卷积分块
    
    在频域进行卷积操作，提取频域特征
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        groups: int = 1,
        use_se: bool = False,
        activation: str = "ReLU"
    ):
        super().__init__()

        # 频域卷积 (使用标准卷积近似)
        # stride 只在 conv1 上使用
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding,
            groups=groups, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        # conv2 不使用 stride
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size, 1, padding,
            groups=groups, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # SE 注意力
        if use_se:
            self.se = SEBlock(out_channels)
        else:
            self.se = nn.Identity()
        
        # 激活函数
        if activation == "ReLU":
            self.act = nn.ReLU(inplace=True)
        elif activation == "SiLU":
            self.act = nn.SiLU(inplace=True)
        elif activation == "LeakyReLU":
            self.act = nn.LeakyReLU(0.1, inplace=True)
        else:
            self.act = nn.ReLU(inplace=True)
        
        # 下采样
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.downsample = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)
        
        out += self.downsample(identity)
        out = self.act(out)
        
        return out


# ============================================
# 频域分支主干网络
# ============================================

class FrequencyBranch(nn.Module):
    """
    频域特征提取分支
    
    流程：
    1. 输入 RGB 图像
    2. DCT 变换到频域
    3. 频域卷积提取特征
    4. 逆 DCT 变换回空间域 (可选)
    """
    
    def __init__(
        self,
        input_size: int = 112,
        channels: List[int] = [64, 128, 256],
        num_blocks: List[int] = [2, 2, 2],
        use_dct: bool = True,
        use_se: bool = False,
        activation: str = "ReLU",
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.input_size = input_size
        self.use_dct = use_dct
        
        # DCT 变换
        if use_dct:
            self.dct = DCTTransform2D(input_size, input_size)
        else:
            self.dct = nn.Identity()
        
        # Stem - 处理频域输入
        self.stem = nn.Sequential(
            nn.Conv2d(3, channels[0], 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True),
        )
        
        # 频域特征提取阶段
        self.stages = nn.ModuleList()
        
        in_ch = channels[0]
        for i, (out_ch, num_b) in enumerate(zip(channels, num_blocks)):
            stage = self._make_stage(
                in_ch, out_ch, num_blocks=num_b,
                stride=2 if i > 0 else 1,
                use_se=use_se,
                activation=activation
            )
            self.stages.append(stage)
            in_ch = out_ch
        
        # Dropout
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        
        # 初始化
        self._init_weights()
    
    def _make_stage(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int,
        stride: int = 1,
        use_se: bool = False,
        activation: str = "ReLU"
    ) -> nn.Sequential:
        layers = []

        # 第一个 block 带下采样
        layers.append(FrequencyConvBlock(
            in_channels, out_channels, stride=stride,
            use_se=use_se, activation=activation
        ))

        # 后续 blocks - in_channels 应该是 out_channels
        for _ in range(1, num_blocks):
            layers.append(FrequencyConvBlock(
                out_channels, out_channels, stride=1,
                use_se=use_se, activation=activation
            ))

        return nn.Sequential(*layers)
    
    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        return_dct: bool = False
    ) -> Tuple[torch.Tensor, ...]:
        """
        前向传播
        
        Args:
            x: (B, 3, H, W) 输入图像
            return_dct: 是否返回 DCT 系数
            
        Returns:
            features: 频域特征列表
            dct_coeffs: (可选) DCT 系数
        """
        # DCT 变换
        if self.use_dct:
            x_dct = self.dct(x)
        else:
            x_dct = x
        
        # Stem
        x = self.stem(x_dct)
        
        # Stages
        features = []
        for stage in self.stages:
            x = stage(x)
            features.append(self.dropout(x))
        
        if return_dct:
            return tuple(features), x_dct
        return tuple(features)


# ============================================
# 频域增强模块
# ============================================

class FrequencyEnhancementModule(nn.Module):
    """
    频域增强模块
    
    通过频域滤波增强特征
    """
    
    def __init__(
        self,
        channels: int,
        freq_range: Tuple[float, float] = (0.1, 0.5),
        use_learnable: bool = True
    ):
        super().__init__()
        
        self.freq_range = freq_range
        self.use_learnable = use_learnable
        
        if use_learnable:
            # 可学习的频带权重
            self.freq_weights = nn.Parameter(
                torch.ones(1, channels, 1, 1)
            )
            self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
        else:
            self.register_buffer('freq_weights', torch.ones(1, channels, 1, 1))
            self.register_buffer('bias', torch.zeros(1, channels, 1, 1))
        
        # 频带滤波器
        self.filter = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        频域增强
        
        Args:
            x: (B, C, H, W) 输入特征
            
        Returns:
            x_enhanced: 增强后的特征
        """
        B, C, H, W = x.shape
        
        # 计算频域表示 (简化版，使用 FFT)
        x_fft = torch.fft.fft2(x)
        x_fft_mag = torch.abs(x_fft)
        x_fft_phase = torch.angle(x_fft)
        
        # 应用频带滤波
        freq_filter = self.filter(x)
        x_fft_mag_filtered = x_fft_mag * freq_filter
        
        # 逆 FFT
        x_fft_filtered = x_fft_mag_filtered * torch.exp(1j * x_fft_phase)
        x_enhanced = torch.real(torch.fft.ifft2(x_fft_filtered))
        
        # 可学习增强
        x_enhanced = x_enhanced * self.freq_weights + self.bias
        
        return x_enhanced


# ============================================
# 低光照增强分支
# ============================================

class LowLightEnhancementBranch(nn.Module):
    """
    低光照图像增强分支
    
    专门处理低照度场景
    """
    
    def __init__(
        self,
        channels: int = 64,
        enhancement_type: str = "retinex"
    ):
        super().__init__()
        
        self.enhancement_type = enhancement_type
        
        if enhancement_type == "retinex":
            # Retinex 理论增强
            self.illumination_est = nn.Sequential(
                nn.Conv2d(3, channels, 3, 1, 1, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels, 3, 3, 1, 1, bias=False),
                nn.Sigmoid()
            )
        elif enhancement_type == "learnable":
            # 可学习增强
            self.enhance = nn.Sequential(
                nn.Conv2d(3, channels, 3, 1, 1, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels, 16, 3, 1, 1, bias=False),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 3, 3, 1, 1, bias=False),
            )
        else:
            self.enhance = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        低光照增强
        
        Args:
            x: (B, 3, H, W) 低光照图像
            
        Returns:
            x_enhanced: 增强后的图像
        """
        if self.enhancement_type == "retinex":
            # Retinex: I = R * L => R = I / L
            illumination = self.illumination_est(x) + 1e-6
            x_enhanced = x / illumination
            x_enhanced = torch.clamp(x_enhanced, 0, 1)
        elif self.enhancement_type == "learnable":
            residual = self.enhance(x)
            x_enhanced = torch.clamp(x + residual, 0, 1)
        else:
            x_enhanced = x
        
        return x_enhanced


# ============================================
# 工厂函数
# ============================================

def build_frequency_branch(
    model_type: str = "standard",
    **kwargs
) -> nn.Module:
    """
    构建频域分支
    
    Args:
        model_type: 模型类型
        **kwargs: 配置参数
        
    Returns:
        频域分支模块
    """
    if model_type == "standard":
        return FrequencyBranch(**kwargs)
    elif model_type == "low_light":
        return nn.Sequential(
            LowLightEnhancementBranch(**kwargs.get('low_light_kwargs', {})),
            FrequencyBranch(**kwargs)
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    # 测试 DCT 变换
    dct = DCTTransform2D(112, 112)
    x = torch.randn(2, 3, 112, 112)
    
    x_dct = dct(x)
    x_recon = dct.inverse(x_dct)
    
    print("Input shape:", x.shape)
    print("DCT shape:", x_dct.shape)
    print("Reconstruction shape:", x_recon.shape)
    print("Reconstruction error:", (x - x_recon).abs().mean().item())
    
    # 测试频域分支
    model = FrequencyBranch(channels=[64, 128, 256], num_blocks=[2, 2, 2])
    model.eval()
    
    with torch.no_grad():
        features = model(x)
    
    print("\nFrequency Branch Features:")
    for i, feat in enumerate(features):
        print(f"  Feature {i+1}: {feat.shape}")
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params / 1e6:.2f}M")
