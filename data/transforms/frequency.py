"""
频域变换模块

包含:
- DCT (离散余弦变换)
- FFT (快速傅里叶变换)
- 频域滤波
- 频域增强
"""

import numpy as np
import cv2
from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================
# DCT 变换 (NumPy 版本)
# ============================================

class DCTTransform:
    """
    离散余弦变换
    
    用于提取频域特征
    """
    
    def __init__(self, shape: Tuple[int, int] = (112, 112)):
        self.shape = shape
        self._create_dct_basis()
    
    def _create_dct_basis(self) -> None:
        """创建 DCT 基函数"""
        H, W = self.shape
        
        # 行基函数
        self.row_basis = np.zeros((H, H))
        for k in range(H):
            for i in range(H):
                if k == 0:
                    self.row_basis[k, i] = 1.0 / np.sqrt(H)
                else:
                    self.row_basis[k, i] = np.sqrt(2.0 / H) * np.cos(
                        np.pi * k * (2 * i + 1) / (2 * H)
                    )
        
        # 列基函数
        self.col_basis = np.zeros((W, W))
        for k in range(W):
            for i in range(W):
                if k == 0:
                    self.col_basis[k, i] = 1.0 / np.sqrt(W)
                else:
                    self.col_basis[k, i] = np.sqrt(2.0 / W) * np.cos(
                        np.pi * k * (2 * i + 1) / (2 * W)
                    )
    
    def forward(self, image: np.ndarray) -> np.ndarray:
        """
        2D DCT 变换
        
        Args:
            image: (H, W) 或 (H, W, C) 输入图像
            
        Returns:
            dct_coeffs: DCT 系数
        """
        if len(image.shape) == 3:
            # 多通道
            coeffs = []
            for c in range(image.shape[2]):
                coeff = self._dct_2d(image[:, :, c])
                coeffs.append(coeff)
            return np.stack(coeffs, axis=-1)
        else:
            return self._dct_2d(image)
    
    def _dct_2d(self, image: np.ndarray) -> np.ndarray:
        """2D DCT"""
        # 行变换
        temp = np.dot(self.row_basis, image)
        # 列变换
        dct_coeffs = np.dot(temp, self.col_basis.T)
        return dct_coeffs
    
    def inverse(self, dct_coeffs: np.ndarray) -> np.ndarray:
        """
        逆 DCT 变换
        
        Args:
            dct_coeffs: DCT 系数
            
        Returns:
            image: 重建图像
        """
        if len(dct_coeffs.shape) == 3:
            images = []
            for c in range(dct_coeffs.shape[2]):
                img = self._idct_2d(dct_coeffs[:, :, c])
                images.append(img)
            return np.stack(images, axis=-1)
        else:
            return self._idct_2d(dct_coeffs)
    
    def _idct_2d(self, dct_coeffs: np.ndarray) -> np.ndarray:
        """2D 逆 DCT"""
        # 行逆变换
        temp = np.dot(self.row_basis.T, dct_coeffs)
        # 列逆变换
        image = np.dot(temp, self.col_basis)
        return image
    
    def get_low_freq_mask(
        self,
        ratio: float = 0.3
    ) -> np.ndarray:
        """
        获取低频掩码
        
        Args:
            ratio: 低频比例
            
        Returns:
            mask: 低频掩码
        """
        H, W = self.shape
        cy, cx = H // 2, W // 2
        
        mask = np.zeros((H, W), dtype=np.float32)
        
        # 低频区域 (中心)
        radius_h = int(H * ratio / 2)
        radius_w = int(W * ratio / 2)
        
        mask[cy-radius_h:cy+radius_h, cx-radius_w:cx+radius_w] = 1.0
        
        return mask
    
    def get_high_freq_mask(
        self,
        ratio: float = 0.3
    ) -> np.ndarray:
        """
        获取高频掩码
        
        Args:
            ratio: 高频比例
            
        Returns:
            mask: 高频掩码
        """
        H, W = self.shape
        
        # 高频 = 1 - 低频
        low_freq_mask = self.get_low_freq_mask(ratio)
        return 1.0 - low_freq_mask


# ============================================
# FFT 变换
# ============================================

class FFTTransform:
    """
    快速傅里叶变换
    
    用于频域分析和滤波
    """
    
    def __init__(self):
        pass
    
    def forward(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        2D FFT 变换
        
        Args:
            image: (H, W) 或 (H, W, C) 输入图像
            
        Returns:
            magnitude: 幅度谱
            phase: 相位谱
        """
        if len(image.shape) == 3:
            magnitudes = []
            phases = []
            for c in range(image.shape[2]):
                mag, phase = self._fft_2d(image[:, :, c])
                magnitudes.append(mag)
                phases.append(phase)
            return np.stack(magnitudes, axis=-1), np.stack(phases, axis=-1)
        else:
            return self._fft_2d(image)
    
    def _fft_2d(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """2D FFT"""
        # FFT
        f = np.fft.fft2(image)
        f_shift = np.fft.fftshift(f)
        
        # 幅度和相位
        magnitude = np.abs(f_shift)
        phase = np.angle(f_shift)
        
        return magnitude, phase
    
    def inverse(
        self,
        magnitude: np.ndarray,
        phase: np.ndarray
    ) -> np.ndarray:
        """
        逆 FFT 变换
        
        Args:
            magnitude: 幅度谱
            phase: 相位谱
            
        Returns:
            image: 重建图像
        """
        # 复数表示
        f_shift = magnitude * np.exp(1j * phase)
        
        # 逆 FFT
        f_ishift = np.fft.ifftshift(f_shift)
        image = np.fft.ifft2(f_ishift)
        
        return np.abs(image)
    
    def get_spectrum(
        self,
        image: np.ndarray,
        log_scale: bool = True
    ) -> np.ndarray:
        """
        获取频谱图
        
        Args:
            image: 输入图像
            log_scale: 是否使用对数缩放
            
        Returns:
            spectrum: 频谱图
        """
        magnitude, _ = self.forward(image)
        
        if log_scale:
            spectrum = np.log(1 + magnitude)
        else:
            spectrum = magnitude
        
        # 归一化到 0-255
        spectrum = (spectrum - spectrum.min()) / (spectrum.max() - spectrum.min()) * 255
        
        return spectrum.astype(np.uint8)


# ============================================
# 频域滤波
# ============================================

class FrequencyFilter:
    """
    频域滤波器
    """
    
    def __init__(self, filter_type: str = 'lowpass'):
        self.filter_type = filter_type
    
    def apply(
        self,
        image: np.ndarray,
        cutoff: float = 0.3,
        order: int = 2
    ) -> np.ndarray:
        """
        应用频域滤波
        
        Args:
            image: 输入图像
            cutoff: 截止频率
            order: 滤波器阶数
            
        Returns:
            filtered: 滤波后的图像
        """
        H, W = image.shape[:2]
        
        # FFT
        fft_transform = FFTTransform()
        magnitude, phase = fft_transform.forward(image.astype(np.float32))
        
        # 创建滤波器
        filter_mask = self._create_filter(H, W, cutoff, order)
        
        # 应用滤波
        if len(magnitude.shape) == 3:
            for c in range(magnitude.shape[2]):
                magnitude[:, :, c] *= filter_mask
        else:
            magnitude *= filter_mask
        
        # 逆 FFT
        filtered = fft_transform.inverse(magnitude, phase)
        
        return np.clip(filtered, 0, 255).astype(np.uint8)
    
    def _create_filter(
        self,
        H: int,
        W: int,
        cutoff: float,
        order: int
    ) -> np.ndarray:
        """创建滤波器"""
        cy, cx = H // 2, W // 2
        
        y, x = np.ogrid[:H, :W]
        distance = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
        max_distance = np.sqrt(cy ** 2 + cx ** 2)
        
        normalized_distance = distance / max_distance
        
        if self.filter_type == 'lowpass':
            # 巴特沃斯低通
            filter_mask = 1.0 / (1.0 + (normalized_distance / cutoff) ** (2 * order))
        elif self.filter_type == 'highpass':
            # 巴特沃斯高通
            filter_mask = 1.0 - 1.0 / (1.0 + (normalized_distance / cutoff) ** (2 * order))
        elif self.filter_type == 'bandpass':
            # 带通
            center = cutoff
            bandwidth = cutoff / 2
            filter_mask = np.exp(-((normalized_distance - center) ** 2) / (2 * bandwidth ** 2))
        else:
            filter_mask = np.ones((H, W))
        
        return filter_mask.astype(np.float32)


# ============================================
# PyTorch 版本 (用于训练)
# ============================================

class DCTLayer(nn.Module):
    """
    PyTorch DCT 层
    
    可嵌入神经网络中
    """
    
    def __init__(self, height: int = 112, width: int = 112):
        super().__init__()
        
        self.height = height
        self.width = width
        
        # 注册 DCT 基函数为 buffer
        self.register_buffer('dct_basis_u', self._compute_dct_basis(height))
        self.register_buffer('dct_basis_v', self._compute_dct_basis(width))
    
    def _compute_dct_basis(self, n: int) -> torch.Tensor:
        """计算 DCT 基函数"""
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
        Args:
            x: (B, C, H, W) 输入
            
        Returns:
            x_dct: (B, C, H, W) DCT 系数
        """
        B, C, H, W = x.shape
        
        # 重塑为 (B*C, H, W)
        x = x.view(B * C, H, W)
        
        # 2D DCT
        x_dct = torch.matmul(self.dct_basis_u[:H, :H], x)
        x_dct = torch.matmul(x_dct, self.dct_basis_v[:W, :W].t())
        
        # 恢复形状
        x_dct = x_dct.view(B, C, H, W)
        
        return x_dct


class IDCTLayer(nn.Module):
    """PyTorch 逆 DCT 层"""
    
    def __init__(self, height: int = 112, width: int = 112):
        super().__init__()
        
        self.height = height
        self.width = width
        
        self.register_buffer('dct_basis_u', self._compute_dct_basis(height))
        self.register_buffer('dct_basis_v', self._compute_dct_basis(width))
    
    def _compute_dct_basis(self, n: int) -> torch.Tensor:
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
    
    def forward(self, x_dct: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x_dct.shape
        
        x_dct = x_dct.view(B * C, H, W)
        
        # 2D IDCT
        x = torch.matmul(self.dct_basis_u[:H, :H].t(), x_dct)
        x = torch.matmul(x, self.dct_basis_v[:W, :W])
        
        x = x.view(B, C, H, W)
        
        return x


# ============================================
# 频域增强 (用于数据增强)
# ============================================

def random_dct_mask(
    image: np.ndarray,
    freq_range: Tuple[float, float] = (0.1, 0.5),
    p: float = 0.2
) -> np.ndarray:
    """
    随机 DCT 掩码增强
    
    Args:
        image: 输入图像
        freq_range: 频率范围
        p: 应用概率
        
    Returns:
        enhanced: 增强后的图像
    """
    if np.random.random() > p:
        return image
    
    dct = DCTTransform(shape=(image.shape[0], image.shape[1]))
    
    # DCT 变换
    coeffs = dct.forward(image.astype(np.float32))
    
    # 创建随机掩码
    H, W = image.shape[:2]
    cy, cx = H // 2, W // 2
    
    freq_ratio = np.random.uniform(*freq_range)
    radius_h = int(H * freq_ratio / 2)
    radius_w = int(W * freq_ratio / 2)
    
    if len(coeffs.shape) == 3:
        for c in range(coeffs.shape[2]):
            coeffs[cy-radius_h:cy+radius_h, cx-radius_w:cx+radius_w, c] *= 0.5
    else:
        coeffs[cy-radius_h:cy+radius_h, cx-radius_w:cx+radius_w] *= 0.5
    
    # 逆 DCT
    enhanced = dct.inverse(coeffs)
    
    return np.clip(enhanced, 0, 255).astype(np.uint8)


if __name__ == "__main__":
    # 测试 DCT
    dct = DCTTransform(shape=(112, 112))
    
    # 创建测试图像
    image = np.random.randint(0, 255, (112, 112), dtype=np.uint8)
    
    # DCT 变换
    coeffs = dct.forward(image)
    
    # 逆 DCT
    reconstructed = dct.inverse(coeffs)
    
    # 计算误差
    error = np.abs(image.astype(np.float32) - reconstructed).mean()
    
    print(f"DCT reconstruction error: {error:.4f}")
    
    # 测试 PyTorch 层
    dct_layer = DCTLayer(112, 112)
    x = torch.randn(2, 3, 112, 112)
    
    with torch.no_grad():
        x_dct = dct_layer(x)
        x_recon = IDCTLayer(112, 112)(x_dct)
    
    print(f"PyTorch DCT reconstruction error: {(x - x_recon).abs().mean().item():.6f}")
