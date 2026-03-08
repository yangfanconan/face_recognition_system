#!/usr/bin/env python3
"""
数据集准备工具

用于下载、验证和准备训练数据集
支持：LFW, WIDER Face, VGGFace2, WebFace12M
"""

import os
import sys
import argparse
import hashlib
from pathlib import Path
from typing import Dict, Optional

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))


# ============================================
# 数据集配置
# ============================================

DATASETS = {
    'lfw': {
        'name': 'LFW',
        'description': 'Labeled Faces in the Wild (评估用)',
        'url': 'http://vis-www.cs.umass.edu/lfw/lfw.tgz',
        'size': '187MB',
        'format': 'tgz',
        'output_dir': 'lfw',
        'files': {
            'images': 'lfw.tgz',
            'pairs': 'pairs.txt'
        },
        'urls': {
            'images': 'http://vis-www.cs.umass.edu/lfw/lfw.tgz',
            'pairs': 'http://vis-www.cs.umass.edu/lfw/pairs.txt'
        }
    },
    'widerface': {
        'name': 'WIDER FACE',
        'description': '人脸检测数据集',
        'url': 'https://huggingface.co/datasets/valhalla/WIDER-face/resolve/main/data/WIDER_train.zip',
        'size': '997MB',
        'format': 'zip',
        'output_dir': 'widerface',
        'urls': {
            'train': 'https://huggingface.co/datasets/valhalla/WIDER-face/resolve/main/data/WIDER_train.zip',
            'val': 'https://huggingface.co/datasets/valhalla/WIDER-face/resolve/main/data/WIDER_val.zip',
            'gt': 'https://huggingface.co/datasets/valhalla/WIDER-face/resolve/main/data/wider_face_split.zip'
        }
    },
    'cplfw': {
        'name': 'CPLFW',
        'description': '跨姿态 LFW (评估用)',
        'url': 'http://www.cbsr.ia.ac.cn/users/cpzhao/project/cplfw/CPLFW.zip',
        'size': '520MB',
        'format': 'zip',
        'output_dir': 'cplfw',
        'requires_auth': True
    },
    'rfw': {
        'name': 'RFW',
        'description': '跨种族人脸数据集',
        'url': 'http://www.whdeng.cn/RFW/Trainingdata.rar',
        'size': '约 2GB',
        'format': 'rar',
        'output_dir': 'rfw',
        'requires_auth': True
    }
}


# ============================================
# 下载工具
# ============================================

def download_file(url: str, output_path: str, chunk_size: int = 8192) -> bool:
    """下载文件"""
    import requests
    from tqdm import tqdm
    
    print(f"Downloading: {url}")
    print(f"Saving to: {output_path}")
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(output_path, 'wb') as f, tqdm(
            desc=os.path.basename(output_path),
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
        
        print(f"Download completed: {output_path}")
        return True
        
    except Exception as e:
        print(f"Download failed: {e}")
        return False


def extract_file(file_path: str, output_dir: str) -> bool:
    """解压文件"""
    import tarfile
    import zipfile
    
    print(f"Extracting: {file_path} -> {output_dir}")
    
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        if file_path.endswith('.tar.gz') or file_path.endswith('.tgz'):
            with tarfile.open(file_path, 'r:gz') as tar:
                tar.extractall(output_dir)
        elif file_path.endswith('.zip'):
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(output_dir)
        else:
            print(f"Unsupported format: {file_path}")
            return False
        
        print(f"Extraction completed: {output_dir}")
        return True
        
    except Exception as e:
        print(f"Extraction failed: {e}")
        return False


# ============================================
# 数据集准备函数
# ============================================

def prepare_lfw(output_root: str) -> bool:
    """准备 LFW 数据集"""
    print("\n" + "="*60)
    print("Preparing LFW dataset...")
    print("="*60)
    
    output_dir = os.path.join(output_root, 'lfw')
    os.makedirs(output_dir, exist_ok=True)
    
    # 下载图像
    images_path = os.path.join(output_dir, 'lfw.tgz')
    if not os.path.exists(images_path):
        success = download_file(DATASETS['lfw']['urls']['images'], images_path)
        if not success:
            return False
    else:
        print(f"Images already exist: {images_path}")
    
    # 解压
    extract_file(images_path, output_dir)
    
    # 下载 pairs.txt
    pairs_path = os.path.join(output_dir, 'pairs.txt')
    if not os.path.exists(pairs_path):
        download_file(DATASETS['lfw']['urls']['pairs'], pairs_path)
    
    # 验证
    if verify_lfw(output_dir):
        print("\n✅ LFW dataset ready!")
        return True
    else:
        print("\n❌ LFW dataset verification failed!")
        return False


def verify_lfw(lfw_dir: str) -> bool:
    """验证 LFW 数据集"""
    pairs_file = os.path.join(lfw_dir, 'pairs.txt')
    
    if not os.path.exists(pairs_file):
        print(f"Missing pairs.txt in {lfw_dir}")
        return False
    
    # 统计图像数量
    image_count = 0
    for root, dirs, files in os.walk(lfw_dir):
        for file in files:
            if file.endswith('.jpg'):
                image_count += 1
    
    print(f"Found {image_count} images in LFW")
    return image_count > 10000  # LFW 应该有 13000+ 图像


def prepare_widerface(output_root: str) -> bool:
    """准备 WIDER Face 数据集"""
    print("\n" + "="*60)
    print("Preparing WIDER FACE dataset...")
    print("="*60)
    
    output_dir = os.path.join(output_root, 'widerface')
    os.makedirs(output_dir, exist_ok=True)
    
    # 下载训练集
    train_zip = os.path.join(output_dir, 'WIDER_train.zip')
    if not os.path.exists(train_zip):
        success = download_file(DATASETS['widerface']['urls']['train'], train_zip)
        if not success:
            return False
    extract_file(train_zip, output_dir)
    
    # 下载验证集
    val_zip = os.path.join(output_dir, 'WIDER_val.zip')
    if not os.path.exists(val_zip):
        success = download_file(DATASETS['widerface']['urls']['val'], val_zip)
        if not success:
            return False
    extract_file(val_zip, output_dir)
    
    # 下载标注
    gt_zip = os.path.join(output_dir, 'wider_face_split.zip')
    if not os.path.exists(gt_zip):
        success = download_file(DATASETS['widerface']['urls']['gt'], gt_zip)
        if not success:
            return False
    extract_file(gt_zip, output_dir)
    
    # 验证
    if verify_widerface(output_dir):
        print("\n✅ WIDER FACE dataset ready!")
        return True
    else:
        print("\n❌ WIDER FACE dataset verification failed!")
        return False


def verify_widerface(widerface_dir: str) -> bool:
    """验证 WIDER Face 数据集"""
    # 检查关键文件
    required_files = [
        'wider_face_split/wider_face_train_bbx_gt.txt',
        'wider_face_split/wider_face_val_bbx_gt.txt'
    ]
    
    for file in required_files:
        file_path = os.path.join(widerface_dir, file)
        if not os.path.exists(file_path):
            print(f"Missing: {file}")
            return False
    
    # 统计图像数量
    image_count = 0
    for split in ['WIDER_train', 'WIDER_val']:
        split_dir = os.path.join(widerface_dir, split, 'images')
        if os.path.exists(split_dir):
            for root, dirs, files in os.walk(split_dir):
                for file in files:
                    if file.endswith('.jpg'):
                        image_count += 1
    
    print(f"Found {image_count} images in WIDER FACE")
    return image_count > 10000  # WIDER 应该有 32000+ 图像


# ============================================
# 主函数
# ============================================

def list_datasets():
    """列出可用数据集"""
    print("\n可用数据集:")
    print("="*70)
    
    for key, info in DATASETS.items():
        auth = " [需要授权]" if info.get('requires_auth') else ""
        print(f"\n{key}{auth}")
        print(f"  描述：{info['description']}")
        print(f"  大小：{info['size']}")
        print(f"  格式：{info['format']}")
    
    print("\n" + "="*70)
    print("\n使用示例:")
    print("  python prepare_datasets.py --dataset lfw --output-dir datasets")
    print("  python prepare_datasets.py --dataset widerface --output-dir datasets")
    print("  python prepare_datasets.py --dataset all --output-dir datasets")
    print()


def main():
    parser = argparse.ArgumentParser(description="Prepare training datasets")
    
    parser.add_argument(
        "--dataset",
        type=str,
        choices=list(DATASETS.keys()) + ['all'],
        default=None,
        help="Dataset to prepare"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="datasets",
        help="Output directory"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available datasets"
    )
    
    args = parser.parse_args()
    
    if args.list or args.dataset is None:
        list_datasets()
        return
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 准备指定数据集
    if args.dataset == 'all':
        prepare_lfw(args.output_dir)
        prepare_widerface(args.output_dir)
    elif args.dataset == 'lfw':
        prepare_lfw(args.output_dir)
    elif args.dataset == 'widerface':
        prepare_widerface(args.output_dir)
    else:
        print(f"Dataset '{args.dataset}' preparation not implemented yet")


if __name__ == "__main__":
    main()
