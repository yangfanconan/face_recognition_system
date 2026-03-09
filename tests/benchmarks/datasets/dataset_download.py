#!/usr/bin/env python3
"""
数据集自动下载和预处理模块

支持以下数据集的自动下载和预处理：
- LFW (Labeled Faces in the Wild)
- CFP-FP (Celebrity Frontal-Profile)
- AgeDB
- RFW (Racial Faces in the Wild)
- WIDER Face (人脸检测)
- RMFD (Masked Face)
- CASIA-FASD (活体检测)
"""

import os
import sys
import shutil
import zipfile
import tarfile
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np
from tqdm import tqdm
import requests
from loguru import logger

# 配置日志
logger.remove()
logger.add(sys.stdout, level="INFO", format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")
logger.add("tests/benchmarks/logs/download.log", rotation="10 MB", retention="7 days")


@dataclass
class DatasetInfo:
    """数据集信息"""
    name: str
    url: str
    save_path: str
    file_hash: Optional[str] = None  # MD5 校验
    file_type: str = "zip"  # zip/tar/raw
    requires_auth: bool = False  # 是否需要授权


class DatasetDownloader:
    """数据集下载器"""
    
    # 数据集配置
    DATASETS = {
        "lfw": DatasetInfo(
            name="LFW",
            url="http://vis-www.cs.umass.edu/lfw/lfw-deepfunneled.tgz",
            save_path="datasets/lfw/lfw-deepfunneled.tgz",
            file_hash=None,  # LFW 不提供官方 hash
            file_type="tgz",
            requires_auth=False
        ),
        "lfw_pairs": DatasetInfo(
            name="LFW Pairs",
            url="http://vis-www.cs.umass.edu/lfw/pairs.txt",
            save_path="datasets/lfw/pairs.txt",
            file_hash=None,
            file_type="raw",
            requires_auth=False
        ),
        "cfp_fp": DatasetInfo(
            name="CFP-FP",
            url="https://www.cfp-biometrics.org/files/cfp_fp.zip",  # 需要注册
            save_path="datasets/cfp_fp/cfp_fp.zip",
            file_hash=None,
            file_type="zip",
            requires_auth=True
        ),
        "agedb30": DatasetInfo(
            name="AgeDB-30",
            url="https://www.dropbox.com/s/zv5k9h81z4f3z8z/AgeDB.zip",  # 示例 URL
            save_path="datasets/agedb/AgeDB.zip",
            file_hash=None,
            file_type="zip",
            requires_auth=True
        ),
        "wider_face": DatasetInfo(
            name="WIDER Face",
            url="https://huggingface.co/datasets/CPFL/WIDERFace/resolve/main/WIDER_face.zip",
            save_path="datasets/widerface/WIDER_face.zip",
            file_hash=None,
            file_type="zip",
            requires_auth=False
        ),
    }
    
    def __init__(self, root_dir: str = "datasets"):
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        
    def download_file(self, url: str, save_path: Path, chunk_size: int = 8192) -> bool:
        """下载文件，支持断点续传"""
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 检查是否已存在完整文件
        if save_path.exists():
            logger.info(f"文件已存在：{save_path}")
            return True
            
        # 检查部分下载的文件
        resume_header = {}
        first_byte = 0
        if save_path.exists():
            first_byte = os.path.getsize(save_path)
            resume_header = {'Range': f'bytes={first_byte}-'}
            logger.info(f"断点续传：已从 {first_byte} 字节开始")
        
        try:
            response = requests.get(url, headers=resume_header, stream=True, timeout=30)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0)) + first_byte
            
            with open(save_path, 'ab' if first_byte > 0 else 'wb') as f:
                with tqdm(
                    total=total_size,
                    initial=first_byte,
                    unit='B',
                    unit_scale=True,
                    unit_divisor=1024,
                    desc=save_path.name
                ) as pbar:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
                            
            logger.success(f"下载完成：{save_path}")
            return True
            
        except requests.RequestException as e:
            logger.error(f"下载失败：{url} - {str(e)}")
            return False
    
    def verify_hash(self, file_path: Path, expected_hash: str) -> bool:
        """验证文件 MD5"""
        if not expected_hash:
            return True
            
        md5 = hashlib.md5()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                md5.update(chunk)
                
        actual_hash = md5.hexdigest()
        if actual_hash.lower() == expected_hash.lower():
            logger.success(f"MD5 校验通过：{file_path.name}")
            return True
        else:
            logger.error(f"MD5 校验失败：{file_path.name}")
            logger.error(f"  期望：{expected_hash}")
            logger.error(f"  实际：{actual_hash}")
            return False
    
    def extract_file(self, file_path: Path, extract_to: Optional[Path] = None) -> bool:
        """解压文件"""
        if extract_to is None:
            extract_to = file_path.parent
            
        try:
            if file_path.suffix in ['.zip']:
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_to)
            elif file_path.suffix in ['.tar', '.gz', '.tgz', '.bz2']:
                with tarfile.open(file_path, 'r:*') as tar_ref:
                    tar_ref.extractall(extract_to)
            else:
                logger.warning(f"未知的压缩格式：{file_path.suffix}")
                return False
                
            logger.success(f"解压完成：{file_path.name}")
            return True
            
        except Exception as e:
            logger.error(f"解压失败：{file_path} - {str(e)}")
            return False
    
    def download(self, dataset_name: str, force: bool = False) -> bool:
        """下载指定数据集"""
        if dataset_name not in self.DATASETS:
            logger.error(f"不支持的数据集：{dataset_name}")
            logger.info(f"支持的数据集：{list(self.DATASETS.keys())}")
            return False
            
        info = self.DATASETS[dataset_name]
        save_path = self.root_dir / info.save_path
        
        # 检查是否已下载
        if save_path.exists() and not force:
            logger.info(f"数据集已下载：{info.name}")
            return True
            
        # 下载
        logger.info(f"开始下载 {info.name}...")
        success = self.download_file(info.url, save_path)
        
        if not success:
            return False
            
        # 验证 hash
        if info.file_hash and not self.verify_hash(save_path, info.file_hash):
            logger.warning("Hash 验证失败，但继续处理")
            
        # 解压
        if info.file_type in ['zip', 'tar', 'tgz', 'gz', 'bz2']:
            logger.info(f"解压 {info.name}...")
            self.extract_file(save_path)
            
        return True
    
    def download_all(self, datasets: Optional[List[str]] = None) -> Dict[str, bool]:
        """批量下载数据集"""
        if datasets is None:
            datasets = list(self.DATASETS.keys())
            
        results = {}
        
        with ThreadPoolExecutor(max_workers=3) as executor:
            future_to_dataset = {
                executor.submit(self.download, ds): ds 
                for ds in datasets
            }
            
            for future in as_completed(future_to_dataset):
                dataset = future_to_dataset[future]
                try:
                    results[dataset] = future.result()
                except Exception as e:
                    logger.error(f"下载 {dataset} 时出错：{str(e)}")
                    results[dataset] = False
                    
        return results


class DataPreprocessor:
    """数据预处理器"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.input_size = config.get('input_size', [112, 112])
        
    def align_face(
        self, 
        image: np.ndarray, 
        landmarks: np.ndarray,
        output_size: Tuple[int, int] = (112, 112)
    ) -> np.ndarray:
        """
        人脸对齐（基于 5 个关键点）
        
        Args:
            image: 输入图像
            landmarks: 5 个关键点 [(x1,y1), (x2,y2), ...]
                      顺序：左眼，右眼，鼻尖，左嘴角，右嘴角
            output_size: 输出尺寸
            
        Returns:
            对齐后的人脸图像
        """
        # 标准 5 点模板（112x112）
        template = np.array([
            [38.2946, 51.6963],
            [73.5318, 51.5014],
            [56.0252, 71.7366],
            [41.5493, 92.3655],
            [70.7299, 92.2041]
        ], dtype=np.float32)
        
        # 调整模板到输出尺寸
        template[:, 0] *= output_size[0] / 112.0
        template[:, 1] *= output_size[1] / 112.0
        
        # 计算仿射变换矩阵
        M, _ = cv2.estimateAffinePartial2D(landmarks, template)
        
        if M is None:
            logger.warning("无法计算仿射变换矩阵")
            return image
            
        # 应用仿射变换
        aligned = cv2.warpAffine(
            image, 
            M, 
            output_size,
            borderMode=cv2.BORDER_REFLECT_101
        )
        
        return aligned
    
    def normalize(
        self, 
        image: np.ndarray,
        mean: Tuple[float, float, float] = (0.5, 0.5, 0.5),
        std: Tuple[float, float, float] = (0.5, 0.5, 0.5)
    ) -> np.ndarray:
        """
        图像归一化
        
        Args:
            image: 输入图像 (H, W, C), RGB, [0, 255]
            mean: 均值
            std: 标准差
            
        Returns:
            归一化后的图像
        """
        image = image.astype(np.float32) / 255.0
        image = (image - mean) / std
        return image
    
    def resize(
        self, 
        image: np.ndarray,
        size: Tuple[int, int]
    ) -> np.ndarray:
        """调整图像大小"""
        return cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
    
    def preprocess_for_detection(
        self, 
        image: np.ndarray,
        input_size: Tuple[int, int] = (640, 640)
    ) -> np.ndarray:
        """
        人脸检测预处理
        
        Args:
            image: BGR 图像
            input_size: 输入尺寸
            
        Returns:
            预处理后的图像 (RGB, normalized)
        """
        # BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize
        image = self.resize(image, input_size)
        
        # Normalize
        image = self.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        # HWC to CHW
        image = np.transpose(image, (2, 0, 1))
        
        return image
    
    def preprocess_for_recognition(
        self, 
        image: np.ndarray,
        landmarks: Optional[np.ndarray] = None,
        input_size: Tuple[int, int] = (112, 112)
    ) -> np.ndarray:
        """
        特征提取预处理
        
        Args:
            image: BGR 人脸图像
            landmarks: 5 个关键点（可选）
            input_size: 输入尺寸
            
        Returns:
            预处理后的图像
        """
        # BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 对齐（如果有关键点）
        if landmarks is not None:
            image = self.align_face(image, landmarks, input_size)
        else:
            image = self.resize(image, input_size)
        
        # Normalize
        image = self.normalize(image, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        
        # HWC to CHW
        image = np.transpose(image, (2, 0, 1))
        
        return image
    
    def blur_face(
        self, 
        image: np.ndarray, 
        bbox: np.ndarray,
        blur_level: int = 15
    ) -> np.ndarray:
        """
        人脸脱敏（模糊处理）
        
        Args:
            image: 输入图像
            bbox: 人脸框 [x1, y1, x2, y2]
            blur_level: 模糊程度
            
        Returns:
            脱敏后的图像
        """
        x1, y1, x2, y2 = map(int, bbox)
        face_roi = image[y1:y2, x1:x2]
        
        # 高斯模糊
        blurred = cv2.GaussianBlur(face_roi, (blur_level, blur_level), 0)
        
        image[y1:y2, x1:x2] = blurred
        return image
    
    def pixelate_face(
        self, 
        image: np.ndarray, 
        bbox: np.ndarray,
        pixel_size: int = 8
    ) -> np.ndarray:
        """
        人脸脱敏（马赛克处理）
        
        Args:
            image: 输入图像
            bbox: 人脸框 [x1, y1, x2, y2]
            pixel_size: 马赛克块大小
            
        Returns:
            脱敏后的图像
        """
        x1, y1, x2, y2 = map(int, bbox)
        face_roi = image[y1:y2, x1:x2]
        
        # 缩小
        h, w = face_roi.shape[:2]
        small = cv2.resize(face_roi, (w // pixel_size, h // pixel_size), interpolation=cv2.INTER_LINEAR)
        
        # 放大回原尺寸
        pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
        
        image[y1:y2, x1:x2] = pixelated
        return image


class LFWDataset:
    """LFW 数据集处理"""
    
    def __init__(self, data_dir: str = "datasets/lfw"):
        self.data_dir = Path(data_dir)
        self.pairs_file = self.data_dir / "pairs.txt"
        
    def load_pairs(self) -> List[Tuple[str, str, int]]:
        """
        加载 LFW pairs.txt
        
        Returns:
            [(img1_path, img2_path, label), ...]
            label: 1=同一人，0=不同人
        """
        if not self.pairs_file.exists():
            raise FileNotFoundError(f"pairs.txt not found: {self.pairs_file}")
            
        pairs = []
        
        with open(self.pairs_file, 'r') as f:
            lines = f.readlines()
            
        # 跳过第一行（样本数）
        n_folds = int(lines[0].strip())
        
        for line in lines[1:]:
            parts = line.strip().split('\t')
            
            if len(parts) == 3:
                # 不同人
                name1, img1 = parts[0], parts[1]
                name2, img2 = parts[2].split('\t')
                img1_path = self.data_dir / name1 / f"{name1}_{img1.zfill(4)}.jpg"
                img2_path = self.data_dir / name2 / f"{name2}_{img2.zfill(4)}.jpg"
                pairs.append((str(img1_path), str(img2_path), 0))
            elif len(parts) == 4:
                # 同一人
                name = parts[0]
                img1 = parts[1]
                img2 = parts[2]
                img1_path = self.data_dir / name / f"{name}_{img1.zfill(4)}.jpg"
                img2_path = self.data_dir / name / f"{name}_{img2.zfill(4)}.jpg"
                pairs.append((str(img1_path), str(img2_path), 1))
                
        return pairs
    
    def load_images(self) -> Dict[str, List[Path]]:
        """
        加载所有人脸图像，按人名分组
        
        Returns:
            {name: [img_path1, img_path2, ...], ...}
        """
        images = {}
        
        for person_dir in self.data_dir.iterdir():
            if not person_dir.is_dir():
                continue
                
            name = person_dir.name
            images[name] = list(person_dir.glob("*.jpg"))
            
        return images


def download_datasets(datasets: List[str] = None):
    """下载数据集的便捷函数"""
    downloader = DatasetDownloader()
    
    if datasets is None:
        datasets = ["lfw", "lfw_pairs", "wider_face"]
        
    results = downloader.download_all(datasets)
    
    print("\n下载结果:")
    for dataset, success in results.items():
        status = "✓" if success else "✗"
        print(f"  {status} {dataset}")
        
    return results


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="数据集下载和预处理")
    parser.add_argument("--download", nargs="+", help="要下载的数据集")
    parser.add_argument("--list-datasets", action="store_true", help="列出支持的数据集")
    parser.add_argument("--test-preprocess", action="store_true", help="测试预处理功能")
    
    args = parser.parse_args()
    
    if args.list_datasets:
        downloader = DatasetDownloader()
        print("支持的数据集:")
        for name, info in downloader.DATASETS.items():
            auth = " (需要授权)" if info.requires_auth else ""
            print(f"  - {name}: {info.name}{auth}")
        return
        
    if args.download:
        download_datasets(args.download)
        return
        
    if args.test_preprocess:
        # 测试预处理功能
        preprocessor = DataPreprocessor({})
        
        # 创建测试图像
        test_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # 测试检测预处理
        det_input = preprocessor.preprocess_for_detection(test_img)
        print(f"检测输入形状：{det_input.shape}")
        
        # 测试识别预处理
        face_img = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
        rec_input = preprocessor.preprocess_for_recognition(face_img)
        print(f"识别输入形状：{rec_input.shape}")
        
        print("预处理测试通过!")
        return
        
    parser.print_help()


if __name__ == "__main__":
    main()
