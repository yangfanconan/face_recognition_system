"""
数据增强模块

包含:
- 几何变换
- 光照变换
- 遮挡模拟
- 高级增强 (Mosaic, MixUp)
- 频域增强
"""

import random
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import cv2


# ============================================
# 几何变换
# ============================================

class RandomHorizontalFlip:
    """随机水平翻转"""
    
    def __init__(self, p: float = 0.5):
        self.p = p
    
    def __call__(
        self,
        image: np.ndarray,
        bboxes: Optional[np.ndarray] = None,
        landmarks: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        if random.random() < self.p:
            image = np.fliplr(image).copy()
            
            if bboxes is not None:
                bboxes[:, [0, 2]] = image.shape[1] - bboxes[:, [2, 0]]
            
            if landmarks is not None:
                landmarks[:, 0] = image.shape[1] - landmarks[:, 0]
                # 交换左右眼、左右嘴角
                landmarks = landmarks[[0, 2, 1, 4, 3]]
        
        return image, bboxes, landmarks


class RandomCrop:
    """随机裁剪"""
    
    def __init__(
        self,
        scale: Tuple[float, float] = (0.8, 1.0),
        ratio: Tuple[float, float] = (0.75, 1.33)
    ):
        self.scale = scale
        self.ratio = ratio
    
    def __call__(
        self,
        image: np.ndarray,
        bboxes: Optional[np.ndarray] = None,
        landmarks: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        H, W = image.shape[:2]
        
        # 随机生成裁剪区域
        area = H * W
        for _ in range(10):
            target_area = random.uniform(*self.scale) * area
            aspect_ratio = random.uniform(*self.ratio)
            
            h = int(round(np.sqrt(target_area * aspect_ratio)))
            w = int(round(np.sqrt(target_area / aspect_ratio)))
            
            if w <= W and h <= H:
                i = random.randint(0, H - h)
                j = random.randint(0, W - w)
                
                image = image[i:i+h, j:j+w]
                
                if bboxes is not None:
                    bboxes[:, [0, 2]] = np.clip(bboxes[:, [0, 2]] - j, 0, w)
                    bboxes[:, [1, 3]] = np.clip(bboxes[:, [1, 3]] - i, 0, h)
                
                if landmarks is not None:
                    landmarks[:, 0] = np.clip(landmarks[:, 0] - j, 0, w)
                    landmarks[:, 1] = np.clip(landmarks[:, 1] - i, 0, h)
                
                break
        
        return image, bboxes, landmarks


class RandomRotation:
    """随机旋转"""
    
    def __init__(self, degrees: float = 15, p: float = 0.5):
        self.degrees = degrees
        self.p = p
    
    def __call__(
        self,
        image: np.ndarray,
        bboxes: Optional[np.ndarray] = None,
        landmarks: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        if random.random() < self.p:
            H, W = image.shape[:2]
            angle = random.uniform(-self.degrees, self.degrees)
            
            center = (W / 2, H / 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            
            image = cv2.warpAffine(
                image, M, (W, H),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE
            )
            
            if landmarks is not None:
                landmarks_2d = landmarks.reshape(-1, 2)
                landmarks_2d = cv2.transform(
                    landmarks_2d.reshape(1, -1, 2), M
                ).reshape(-1, 2)
                landmarks = landmarks_2d
        
        return image, bboxes, landmarks


# ============================================
# 光照变换
# ============================================

class ColorJitter:
    """颜色抖动"""
    
    def __init__(
        self,
        brightness: float = 0.3,
        contrast: float = 0.3,
        saturation: float = 0.3,
        hue: float = 0.1,
        p: float = 0.5
    ):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.p = p
    
    def __call__(
        self,
        image: np.ndarray,
        **kwargs
    ) -> Tuple[np.ndarray, ...]:
        if random.random() < self.p:
            image = image.astype(np.float32) / 255.0
            
            # 亮度
            if random.random() < 0.5:
                alpha = random.uniform(1 - self.brightness, 1 + self.brightness)
                image = np.clip(image * alpha, 0, 1)
            
            # 对比度
            if random.random() < 0.5:
                alpha = random.uniform(1 - self.contrast, 1 + self.contrast)
                gray = np.mean(image, axis=2, keepdims=True)
                image = np.clip(image * alpha + gray * (1 - alpha), 0, 1)
            
            # 饱和度
            if random.random() < 0.5 and self.saturation > 0:
                alpha = random.uniform(1 - self.saturation, 1 + self.saturation)
                gray = np.mean(image, axis=2, keepdims=True)
                image = np.clip(image * alpha + gray * (1 - alpha), 0, 1)
            
            # 色调
            if random.random() < 0.5 and self.hue > 0:
                delta = random.uniform(-self.hue, self.hue)
                image[:, :, 0] = np.clip(image[:, :, 0] + delta, 0, 1)
                image[:, :, 1] = np.clip(image[:, :, 1] + delta, 0, 1)
                image[:, :, 2] = np.clip(image[:, :, 2] + delta, 0, 1)
            
            image = (image * 255).astype(np.uint8)
        
        return image, kwargs.get('bboxes'), kwargs.get('landmarks')


class RandomGrayscale:
    """随机灰度化"""
    
    def __init__(self, p: float = 0.1):
        self.p = p
    
    def __call__(self, image: np.ndarray, **kwargs) -> Tuple[np.ndarray, ...]:
        if random.random() < self.p:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            image = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        
        return image, kwargs.get('bboxes'), kwargs.get('landmarks')


class RandomGaussianBlur:
    """随机高斯模糊"""
    
    def __init__(self, kernel_size: int = 3, p: float = 0.1):
        self.kernel_size = kernel_size
        self.p = p
    
    def __call__(self, image: np.ndarray, **kwargs) -> Tuple[np.ndarray, ...]:
        if random.random() < self.p:
            image = cv2.GaussianBlur(image, (self.kernel_size, self.kernel_size), 0)
        
        return image, kwargs.get('bboxes'), kwargs.get('landmarks')


# ============================================
# 遮挡模拟
# ============================================

class RandomErasing:
    """随机擦除"""
    
    def __init__(
        self,
        scale: Tuple[float, float] = (0.02, 0.15),
        ratio: Tuple[float, float] = (0.5, 2.0),
        p: float = 0.3
    ):
        self.scale = scale
        self.ratio = ratio
        self.p = p
    
    def __call__(self, image: np.ndarray, **kwargs) -> Tuple[np.ndarray, ...]:
        if random.random() < self.p:
            H, W = image.shape[:2]
            area = H * W
            
            for _ in range(10):
                target_area = random.uniform(*self.scale) * area
                aspect_ratio = random.uniform(*self.ratio)
                
                h = int(round(np.sqrt(target_area * aspect_ratio)))
                w = int(round(np.sqrt(target_area / aspect_ratio)))
                
                if w <= W and h <= H:
                    i = random.randint(0, H - h)
                    j = random.randint(0, W - w)
                    
                    # 随机颜色或黑色
                    if random.random() < 0.5:
                        image[i:i+h, j:j+w] = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
                    else:
                        image[i:i+h, j:j+w] = 0
                    
                    break
        
        return image, kwargs.get('bboxes'), kwargs.get('landmarks')


class RandomRectangleMask:
    """随机矩形遮挡"""
    
    def __init__(
        self,
        num_masks: int = 1,
        size: Tuple[float, float] = (0.1, 0.3),
        p: float = 0.2
    ):
        self.num_masks = num_masks
        self.size = size
        self.p = p
    
    def __call__(self, image: np.ndarray, **kwargs) -> Tuple[np.ndarray, ...]:
        if random.random() < self.p:
            H, W = image.shape[:2]
            
            for _ in range(self.num_masks):
                mask_h = int(H * random.uniform(*self.size))
                mask_w = int(W * random.uniform(*self.size))
                
                i = random.randint(0, H - mask_h)
                j = random.randint(0, W - mask_w)
                
                # 黑色遮挡
                image[i:i+mask_h, j:j+mask_w] = 0
        
        return image, kwargs.get('bboxes'), kwargs.get('landmarks')


# ============================================
# 高级增强
# ============================================

class Mosaic:
    """Mosaic 数据增强 (4 图拼接)"""
    
    def __init__(self, p: float = 0.15):
        self.p = p
    
    def __call__(
        self,
        image: np.ndarray,
        bboxes: Optional[np.ndarray] = None,
        landmarks: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        if random.random() < self.p:
            # 需要额外 3 张图像 (这里简化处理，使用同一张图像的裁剪)
            H, W = image.shape[:2]
            
            # 创建 4 个裁剪区域
            crops = []
            for _ in range(4):
                i = random.randint(0, H // 2)
                j = random.randint(0, W // 2)
                crop = image[i:i+H//2, j:j+W//2]
                crop = cv2.resize(crop, (W // 2, H // 2))
                crops.append(crop)
            
            # 拼接
            top = np.hstack([crops[0], crops[1]])
            bottom = np.hstack([crops[2], crops[3]])
            image = np.vstack([top, bottom])
        
        return image, bboxes, landmarks


class MixUp:
    """MixUp 数据增强"""
    
    def __init__(self, alpha: float = 0.2, p: float = 0.1):
        self.alpha = alpha
        self.p = p
    
    def __call__(
        self,
        image: np.ndarray,
        image2: Optional[np.ndarray] = None,
        **kwargs
    ) -> Tuple[np.ndarray, ...]:
        if random.random() < self.p and image2 is not None:
            lam = np.random.beta(self.alpha, self.alpha)
            
            image = image.astype(np.float32) * lam + image2.astype(np.float32) * (1 - lam)
            image = np.clip(image, 0, 255).astype(np.uint8)
        
        return image, kwargs.get('bboxes'), kwargs.get('landmarks')


# ============================================
# 频域增强
# ============================================

class RandomDCTMask:
    """随机 DCT 频域掩码"""
    
    def __init__(
        self,
        freq_range: Tuple[float, float] = (0.1, 0.5),
        p: float = 0.2
    ):
        self.freq_range = freq_range
        self.p = p
    
    def __call__(self, image: np.ndarray, **kwargs) -> Tuple[np.ndarray, ...]:
        if random.random() < self.p:
            H, W = image.shape[:2]
            
            # 简化的频域处理 (使用 FFT 近似 DCT)
            image_float = image.astype(np.float32)
            
            # FFT
            f = np.fft.fft2(image_float, axes=(0, 1))
            f_shift = np.fft.fftshift(f)
            
            # 创建频域掩码
            mask = np.ones((H, W), dtype=np.float32)
            cy, cx = H // 2, W // 2
            
            # 随机遮挡某个频带
            freq_ratio = random.uniform(*self.freq_range)
            radius_h = int(H * freq_ratio / 2)
            radius_w = int(W * freq_ratio / 2)
            
            mask[cy-radius_h:cy+radius_h, cx-radius_w:cx+radius_w] = 0
            
            # 应用掩码
            for c in range(3):
                f_shift[:, :, c] *= mask
            
            # 逆 FFT
            f_ishift = np.fft.ifftshift(f_shift)
            image_back = np.fft.ifft2(f_ishift, axes=(0, 1))
            image = np.abs(image_back)
            image = np.clip(image, 0, 255).astype(np.uint8)
        
        return image, kwargs.get('bboxes'), kwargs.get('landmarks')


# ============================================
# 组合增强
# ============================================

class Compose:
    """组合多个增强"""
    
    def __init__(self, transforms: List):
        self.transforms = transforms
    
    def __call__(
        self,
        image: np.ndarray,
        bboxes: Optional[np.ndarray] = None,
        landmarks: Optional[np.ndarray] = None
    ) -> Dict:
        for t in self.transforms:
            if hasattr(t, '__call__'):
                result = t(image, bboxes=bboxes, landmarks=landmarks)
                if isinstance(result, tuple):
                    image, bboxes, landmarks = result
                else:
                    image = result
        
        return {
            'image': image,
            'bboxes': bboxes,
            'landmarks': landmarks,
        }


# ============================================
# 预设增强配置
# ============================================

def get_train_augmentation() -> Compose:
    """获取训练增强配置"""
    return Compose([
        RandomHorizontalFlip(p=0.5),
        RandomCrop(scale=(0.8, 1.0)),
        RandomRotation(degrees=15, p=0.3),
        ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, p=0.5),
        RandomGrayscale(p=0.1),
        RandomGaussianBlur(p=0.1),
        RandomErasing(scale=(0.02, 0.15), p=0.3),
        RandomRectangleMask(p=0.2),
        Mosaic(p=0.15),
        MixUp(alpha=0.2, p=0.1),
        RandomDCTMask(p=0.2),
    ])


def get_val_augmentation() -> Compose:
    """获取验证增强配置 (无随机增强)"""
    return Compose([])


if __name__ == "__main__":
    # 测试增强
    import matplotlib.pyplot as plt
    
    image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    aug = get_train_augmentation()
    result = aug(image)
    
    print("Augmentation test:")
    print(f"  Input shape: {image.shape}")
    print(f"  Output shape: {result['image'].shape}")
    
    # 显示原图和增强后的图像
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(image)
    axes[0].set_title("Original")
    axes[1].imshow(result['image'])
    axes[1].set_title("Augmented")
    plt.tight_layout()
    plt.savefig('/tmp/augmentation_test.png')
    print("  Saved test image to /tmp/augmentation_test.png")
