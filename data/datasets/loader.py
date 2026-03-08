"""
数据集加载器

支持数据集:
- WebFace12M
- VGGFace2
- CASIA-WebFace
- MS-Celeb-1M
- 自定义数据集
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable

import numpy as np
from torch.utils.data import Dataset, DataLoader
import cv2


# ============================================
# 基础人脸数据集
# ============================================

class FaceDataset(Dataset):
    """
    基础人脸数据集
    
    支持:
    - 图像加载
    - 边界框和关键点
    - 数据增强
    """
    
    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        loader: str = "cv2"
    ):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        
        # 数据列表
        self.samples: List[Dict] = []
        self._load_annotations()
    
    def _load_annotations(self) -> None:
        """加载标注文件"""
        # 子类实现
        raise NotImplementedError
    
    def load_image(self, path: str) -> np.ndarray:
        """加载图像"""
        if self.loader == "cv2":
            image = cv2.imread(path)
            if image is not None:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif self.loader == "pillow":
            from PIL import Image
            image = np.array(Image.open(path))
        else:
            raise ValueError(f"Unknown loader: {self.loader}")
        
        return image
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, index: int) -> Dict:
        sample = self.samples[index]
        
        # 加载图像
        image = self.load_image(sample['image_path'])
        
        # 获取标注
        target = {
            'identity': sample.get('identity', -1),
            'bbox': sample.get('bbox', None),
            'landmarks': sample.get('landmarks', None),
        }
        
        # 应用变换
        if self.transform:
            result = self.transform(image=image, bboxes=target['bbox'], landmarks=target['landmarks'])
            image = result['image']
            target['bbox'] = result.get('bboxes')
            target['landmarks'] = result.get('landmarks')
        
        if self.target_transform:
            target = self.target_transform(target)
        
        return {
            'images': image,
            'labels': target['identity'],
            'bbox': target['bbox'],
            'landmarks': target['landmarks'],
            'image_path': sample['image_path'],
        }


# ============================================
# WebFace12M 数据集
# ============================================

class WebFace12M(FaceDataset):
    """
    WebFace12M 数据集
    
    12M 图像，617K ID
    """
    
    def _load_annotations(self) -> None:
        """加载 WebFace12M 标注"""
        anno_path = os.path.join(self.root, "face.json")
        
        if not os.path.exists(anno_path):
            raise FileNotFoundError(f"Annotation file not found: {anno_path}")
        
        with open(anno_path, 'r') as f:
            data = json.load(f)
        
        for identity, faces in data.items():
            for face in faces:
                self.samples.append({
                    'image_path': os.path.join(self.root, face['image_path']),
                    'identity': int(identity),
                    'bbox': face.get('bbox'),
                    'landmarks': face.get('landmarks'),
                })
        
        print(f"Loaded {len(self.samples)} images from WebFace12M")


# ============================================
# VGGFace2 数据集
# ============================================

class VGGFace2(FaceDataset):
    """
    VGGFace2 数据集
    
    3.3M 图像，9K ID
    """
    
    def _load_annotations(self) -> None:
        """加载 VGGFace2 标注"""
        # 训练集
        train_list_path = os.path.join(self.root, "train_list.txt")
        
        if not os.path.exists(train_list_path):
            raise FileNotFoundError(f"Train list not found: {train_list_path}")
        
        # 加载 ID 映射
        id_to_idx = {}
        idx = 0
        
        with open(train_list_path, 'r') as f:
            for line in f:
                parts = line.strip().split('/')
                if len(parts) != 2:
                    continue
                
                identity = parts[0]
                image_name = parts[1]
                
                if identity not in id_to_idx:
                    id_to_idx[identity] = idx
                    idx += 1
                
                self.samples.append({
                    'image_path': os.path.join(self.root, "train", identity, image_name),
                    'identity': id_to_idx[identity],
                })
        
        print(f"Loaded {len(self.samples)} images from VGGFace2 ({len(id_to_idx)} IDs)")


# ============================================
# CASIA-WebFace 数据集
# ============================================

class CASIAWebFace(FaceDataset):
    """
    CASIA-WebFace 数据集
    
    494K 图像，10K ID
    """
    
    def _load_annotations(self) -> None:
        """加载 CASIA-WebFace 标注"""
        id_to_idx = {}
        idx = 0
        
        for identity_dir in sorted(os.listdir(self.root)):
            identity_path = os.path.join(self.root, identity_dir)
            if not os.path.isdir(identity_path):
                continue
            
            if identity_dir not in id_to_idx:
                id_to_idx[identity_dir] = idx
                idx += 1
            
            for image_name in os.listdir(identity_path):
                if not image_name.endswith(('.jpg', '.png', '.jpeg')):
                    continue
                
                self.samples.append({
                    'image_path': os.path.join(identity_path, image_name),
                    'identity': id_to_idx[identity_dir],
                })
        
        print(f"Loaded {len(self.samples)} images from CASIA-WebFace ({len(id_to_idx)} IDs)")


# ============================================
# 自定义数据集
# ============================================

class CustomFaceDataset(FaceDataset):
    """
    自定义人脸数据集
    
    支持目录结构:
    data/
    ├── id_001/
    │   ├── img1.jpg
    │   └── img2.jpg
    ├── id_002/
    │   └── img1.jpg
    └── annotations.json (可选)
    """
    
    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        use_annotations: bool = True,
        **kwargs
    ):
        self.use_annotations = use_annotations
        super().__init__(root, transform, **kwargs)
    
    def _load_annotations(self) -> None:
        """加载自定义标注"""
        anno_path = os.path.join(self.root, "annotations.json")
        
        if self.use_annotations and os.path.exists(anno_path):
            # 从标注文件加载
            with open(anno_path, 'r') as f:
                data = json.load(f)
            
            for sample in data.get('samples', []):
                self.samples.append({
                    'image_path': os.path.join(self.root, sample['image']),
                    'identity': sample.get('identity', -1),
                    'bbox': sample.get('bbox'),
                    'landmarks': sample.get('landmarks'),
                })
        else:
            # 从目录结构加载
            id_to_idx = {}
            idx = 0
            
            for identity_dir in sorted(os.listdir(self.root)):
                identity_path = os.path.join(self.root, identity_dir)
                if not os.path.isdir(identity_path):
                    continue
                
                if identity_dir not in id_to_idx:
                    id_to_idx[identity_dir] = idx
                    idx += 1
                
                for image_name in os.listdir(identity_path):
                    if not image_name.endswith(('.jpg', '.png', '.jpeg')):
                        continue
                    
                    self.samples.append({
                        'image_path': os.path.join(identity_path, image_name),
                        'identity': id_to_idx[identity_dir],
                    })
        
        print(f"Loaded {len(self.samples)} images from custom dataset")


# ============================================
# 数据加载器工厂
# ============================================

def build_dataloader(
    dataset_name: str,
    root: str,
    transform: Optional[Callable] = None,
    batch_size: int = 32,
    num_workers: int = 8,
    shuffle: bool = True,
    **kwargs
) -> DataLoader:
    """
    构建数据加载器
    
    Args:
        dataset_name: 数据集名称
        root: 数据根目录
        transform: 数据变换
        batch_size: 批次大小
        num_workers: 工作线程数
        shuffle: 是否打乱
        
    Returns:
        DataLoader
    """
    datasets = {
        'webface12m': WebFace12M,
        'vggface2': VGGFace2,
        'casia_webface': CASIAWebFace,
        'custom': CustomFaceDataset,
    }
    
    if dataset_name not in datasets:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    dataset = datasets[dataset_name](root=root, transform=transform, **kwargs)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=True,
        drop_last=True,
    )


# ============================================
# 验证数据集
# ============================================

class LFWDataset(Dataset):
    """
    LFW 验证数据集
    """
    
    def __init__(self, root: str, pairs_file: str = "pairs.txt"):
        self.root = root
        self.pairs = []
        self._load_pairs(pairs_file)
    
    def _load_pairs(self, pairs_file: str) -> None:
        """加载 pairs.txt"""
        with open(pairs_file, 'r') as f:
            lines = f.readlines()
        
        # 跳过第一行 (N)
        for line in lines[1:]:
            parts = line.strip().split()
            if len(parts) == 3:
                # 同一人
                name, img1, img2 = parts
                self.pairs.append({
                    'same': True,
                    'image1': os.path.join(self.root, name, f"{name}_{img1}.jpg"),
                    'image2': os.path.join(self.root, name, f"{name}_{img2}.jpg"),
                })
            else:
                # 不同人
                name1, img1, name2, img2 = parts
                self.pairs.append({
                    'same': False,
                    'image1': os.path.join(self.root, name1, f"{name1}_{img1}.jpg"),
                    'image2': os.path.join(self.root, name2, f"{name2}_{img2}.jpg"),
                })
    
    def __len__(self) -> int:
        return len(self.pairs)
    
    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray, bool]:
        pair = self.pairs[index]
        
        image1 = cv2.imread(pair['image1'])
        image2 = cv2.imread(pair['image2'])
        
        if image1 is not None:
            image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
        if image2 is not None:
            image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
        
        return image1, image2, pair['same']


if __name__ == "__main__":
    # 测试数据集
    print("Testing datasets...")
    
    # 自定义数据集
    dataset = CustomFaceDataset(root="/path/to/data")
    print(f"Dataset size: {len(dataset)}")
    
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"Sample keys: {sample.keys()}")
        print(f"Image shape: {sample['images'].shape}")
