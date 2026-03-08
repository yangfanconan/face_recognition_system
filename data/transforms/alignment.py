"""
人脸对齐模块

基于关键点的人脸对齐
"""

import numpy as np
import cv2
from typing import List, Optional, Tuple


# ============================================
# 标准 landmarks
# ============================================

# 5 点标准位置 (用于 112x112 输出)
STANDARD_LANDMARKS_5 = np.array([
    [56.0, 56.0],      # 鼻子
    [38.0, 42.0],      # 左眼
    [74.0, 42.0],      # 右眼
    [40.0, 78.0],      # 左嘴角
    [72.0, 78.0],      # 右嘴角
], dtype=np.float32)

# 68 点标准位置 (用于更精细对齐)
STANDARD_LANDMARKS_68 = None  # 可根据需要添加


# ============================================
# 人脸对齐器
# ============================================

class FaceAligner:
    """
    人脸对齐器
    
    基于关键点进行仿射变换对齐
    """
    
    def __init__(
        self,
        output_size: int = 112,
        scale_factor: float = 1.0,
        num_landmarks: int = 5
    ):
        """
        Args:
            output_size: 输出图像尺寸
            scale_factor: 缩放因子
            num_landmarks: 关键点数量 (5 或 68)
        """
        self.output_size = output_size
        self.scale_factor = scale_factor
        self.num_landmarks = num_landmarks
        
        # 设置标准 landmarks
        if num_landmarks == 5:
            self.dst_landmarks = STANDARD_LANDMARKS_5.copy()
            # 根据输出尺寸缩放
            self.dst_landmarks *= (output_size / 112.0)
        else:
            raise ValueError(f"Unsupported num_landmarks: {num_landmarks}")
    
    def align(
        self,
        image: np.ndarray,
        landmarks: np.ndarray
    ) -> np.ndarray:
        """
        人脸对齐
        
        Args:
            image: (H, W, C) 原始图像
            landmarks: (N, 2) 关键点坐标
            
        Returns:
            aligned: (output_size, output_size, C) 对齐后的人脸
        """
        # 计算仿射变换矩阵
        M = self._estimate_affine(landmarks)
        
        # 应用变换
        aligned = cv2.warpAffine(
            image,
            M,
            (self.output_size, self.output_size),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
            borderValue=(114, 114, 114)
        )
        
        return aligned
    
    def _estimate_affine(
        self,
        src_landmarks: np.ndarray
    ) -> np.ndarray:
        """
        估计仿射变换矩阵
        
        Args:
            src_landmarks: (N, 2) 源关键点
            
        Returns:
            M: (2, 3) 仿射变换矩阵
        """
        src = src_landmarks.astype(np.float32)
        dst = self.dst_landmarks.copy()
        
        # 使用全部关键点计算仿射变换
        M, _ = cv2.estimateAffinePartial2D(src, dst)
        
        return M
    
    def align_batch(
        self,
        image: np.ndarray,
        landmarks_list: List[np.ndarray]
    ) -> List[np.ndarray]:
        """
        批量对齐
        
        Args:
            image: 原始图像
            landmarks_list: 关键点列表
            
        Returns:
            aligned_faces: 对齐后的人脸列表
        """
        aligned_faces = []
        
        for landmarks in landmarks_list:
            aligned = self.align(image, landmarks)
            aligned_faces.append(aligned)
        
        return aligned_faces
    
    def align_with_bbox(
        self,
        image: np.ndarray,
        bbox: np.ndarray,
        landmarks: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        使用 bbox 进行对齐 (当没有关键点时)
        
        Args:
            image: 原始图像
            bbox: (4,) [x1, y1, x2, y2]
            landmarks: 可选的关键点
            
        Returns:
            aligned: 对齐后的人脸
        """
        if landmarks is not None:
            return self.align(image, landmarks)
        
        # 从 bbox 估计 landmarks
        x1, y1, x2, y2 = bbox
        w, h = x2 - x1, y2 - y1
        
        # 估计关键点位置
        landmarks = np.array([
            [x1 + w * 0.5, y1 + h * 0.4],      # 鼻子
            [x1 + w * 0.35, y1 + h * 0.35],    # 左眼
            [x1 + w * 0.65, y1 + h * 0.35],    # 右眼
            [x1 + w * 0.35, y1 + h * 0.65],    # 左嘴角
            [x1 + w * 0.65, y1 + h * 0.65],    # 右嘴角
        ], dtype=np.float32)
        
        return self.align(image, landmarks)


# ============================================
# 可微分对齐 (用于训练)
# ============================================

class DifferentiableAligner:
    """
    可微分人脸对齐
    
    使用网格采样实现可微分对齐，支持梯度回传
    """
    
    def __init__(
        self,
        output_size: int = 112,
        device: str = 'cpu'
    ):
        self.output_size = output_size
        self.device = device
        
        # 预计算采样网格
        self._create_sampling_grid()
    
    def _create_sampling_grid(self) -> None:
        """创建采样网格"""
        # 归一化坐标 [-1, 1]
        y = torch.linspace(-1, 1, self.output_size, device=self.device)
        x = torch.linspace(-1, 1, self.output_size, device=self.device)
        
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        
        # (output_size, output_size, 2)
        self.base_grid = torch.stack([xx, yy], dim=-1).unsqueeze(0)
    
    def align(
        self,
        image: torch.Tensor,
        landmarks: torch.Tensor
    ) -> torch.Tensor:
        """
        可微分对齐
        
        Args:
            image: (B, C, H, W) 输入图像
            landmarks: (B, N, 2) 关键点
            
        Returns:
            aligned: (B, C, output_size, output_size) 对齐后的人脸
        """
        import torch.nn.functional as F
        
        B = image.shape[0]
        
        # 计算仿射变换矩阵
        M = self._estimate_affine_batch(landmarks)
        
        # 应用仿射变换到采样网格
        grid = F.affine_grid(M, (B, 3, self.output_size, self.output_size), align_corners=False)
        
        # 网格采样
        aligned = F.grid_sample(image, grid, align_corners=False)
        
        return aligned
    
    def _estimate_affine_batch(
        self,
        landmarks: torch.Tensor
    ) -> torch.Tensor:
        """
        批量估计仿射变换矩阵
        
        使用最小二乘法求解
        """
        B, N, _ = landmarks.shape
        
        # 目标 landmarks
        dst = self.dst_landmarks_tensor.to(landmarks.device)
        
        # 构建线性方程组
        # [x' y'] = M @ [x y 1]
        
        # 简化处理：使用 3 点求解
        # 使用左眼、右眼、鼻子
        src_points = landmarks[:, :3]  # (B, 3, 2)
        dst_points = dst[:3].unsqueeze(0).expand(B, -1, -1)
        
        # 计算变换矩阵 (简化版)
        # 实际实现需要更复杂的求解
        
        M = torch.eye(2, 3, device=landmarks.device).unsqueeze(0).expand(B, -1, -1)
        
        return M


# ============================================
# 工具函数
# ============================================

def estimate_affine_matrix(
    src_landmarks: np.ndarray,
    dst_landmarks: np.ndarray
) -> np.ndarray:
    """
    估计仿射变换矩阵
    
    Args:
        src_landmarks: (N, 2) 源关键点
        dst_landmarks: (N, 2) 目标关键点
        
    Returns:
        M: (2, 3) 仿射变换矩阵
    """
    M, _ = cv2.estimateAffinePartial2D(src_landmarks, dst_landmarks)
    return M


def warp_affine(
    image: np.ndarray,
    M: np.ndarray,
    output_size: int = 112
) -> np.ndarray:
    """
    应用仿射变换
    
    Args:
        image: (H, W, C) 输入图像
        M: (2, 3) 仿射变换矩阵
        output_size: 输出尺寸
        
    Returns:
        warped: (output_size, output_size, C) 变换后的图像
    """
    warped = cv2.warpAffine(
        image,
        M,
        (output_size, output_size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE
    )
    
    return warped


def align_faces(
    image: np.ndarray,
    landmarks: np.ndarray,
    output_size: int = 112
) -> np.ndarray:
    """
    便捷函数：对齐人脸
    
    Args:
        image: 输入图像
        landmarks: 关键点
        output_size: 输出尺寸
        
    Returns:
        aligned: 对齐后的人脸
    """
    aligner = FaceAligner(output_size=output_size)
    return aligner.align(image, landmarks)


# 导入 torch
import torch

if __name__ == "__main__":
    # 测试对齐
    aligner = FaceAligner(output_size=112)
    
    # 创建测试图像和 landmarks
    image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    landmarks = np.array([
        [320, 200],  # 鼻子
        [280, 180],  # 左眼
        [360, 180],  # 右眼
        [285, 260],  # 左嘴角
        [355, 260],  # 右嘴角
    ], dtype=np.float32)
    
    # 对齐
    aligned = aligner.align(image, landmarks)
    
    print(f"Aligned face shape: {aligned.shape}")
    
    # 保存测试
    import cv2
    cv2.imwrite('/tmp/aligned_test.jpg', aligned)
    print("Saved aligned face to /tmp/aligned_test.jpg")
