"""
数据集下载工具

支持数据集:
- WebFace12M
- VGGFace2
- CASIA-WebFace
- MS-Celeb-1M (清洗版)
- LFW (评估用)
- WiderFace (检测用)
"""

import os
import sys
import argparse
import logging
import hashlib
from pathlib import Path
from typing import Dict, List, Optional
import subprocess

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
        'checksum': None,  # 可选
    },
    'lfw_pairs': {
        'name': 'LFW Pairs',
        'description': 'LFW pairs.txt (验证协议)',
        'url': 'http://vis-www.cs.umass.edu/lfw/pairs.txt',
        'size': '1KB',
        'format': 'txt',
        'output_dir': 'lfw',
    },
    'widerface': {
        'name': 'WIDER FACE',
        'description': '人脸检测数据集',
        'url': 'https://huggingface.co/datasets/valhalla/WIDER-face/resolve/main/data/WIDER_train.zip',
        'size': '997MB',
        'format': 'zip',
        'output_dir': 'widerface',
    },
    'widerface_val': {
        'name': 'WIDER FACE Val',
        'description': 'WIDER FACE 验证集',
        'url': 'https://huggingface.co/datasets/valhalla/WIDER-face/resolve/main/data/WIDER_val.zip',
        'size': '997MB',
        'format': 'zip',
        'output_dir': 'widerface',
    },
    'vggface2': {
        'name': 'VGGFace2',
        'description': 'VGG 人脸数据集 (3.3M 图像)',
        'url': 'https://www.robots.ox.ac.uk/~vgg/data/vgg_face2/vggface2_train.tar.gz',
        'size': '约 50GB',
        'format': 'tar.gz',
        'output_dir': 'vggface2',
        'requires_auth': True,
    },
    'casia_webface': {
        'name': 'CASIA-WebFace',
        'description': 'CASIA 人脸数据集 (494K 图像)',
        'url': '需要申请',
        'size': '约 10GB',
        'format': 'zip',
        'output_dir': 'casia_webface',
        'requires_auth': True,
    },
    'webface12m': {
        'name': 'WebFace12M',
        'description': 'WebFace12M 数据集 (12M 图像)',
        'url': 'https://www.face-benchmark.org/download.html',
        'size': '约 100GB',
        'format': 'zip',
        'output_dir': 'webface12m',
        'requires_auth': True,
    },
    'rfw': {
        'name': 'RFW',
        'description': '跨种族人脸数据集',
        'url': 'http://www.whdeng.cn/RFW/Trainingdata.rar',
        'size': '约 2GB',
        'format': 'rar',
        'output_dir': 'rfw',
        'requires_auth': True,
    },
}


# ============================================
# 下载工具
# ============================================

def download_file(url: str, output_path: str, chunk_size: int = 8192) -> bool:
    """
    下载文件
    
    Args:
        url: 下载 URL
        output_path: 保存路径
        chunk_size: 分块大小
        
    Returns:
        success: 是否成功
    """
    import requests
    from tqdm import tqdm
    
    logging.info(f"Downloading: {url}")
    logging.info(f"Saving to: {output_path}")
    
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
        
        logging.info(f"Download completed: {output_path}")
        return True
        
    except Exception as e:
        logging.error(f"Download failed: {e}")
        return False


def extract_file(file_path: str, output_dir: str) -> bool:
    """
    解压文件
    
    Args:
        file_path: 压缩文件路径
        output_dir: 解压目录
        
    Returns:
        success: 是否成功
    """
    logging.info(f"Extracting: {file_path} -> {output_dir}")
    
    try:
        import tarfile
        import zipfile
        import rarfile
        
        os.makedirs(output_dir, exist_ok=True)
        
        if file_path.endswith('.tar.gz') or file_path.endswith('.tgz'):
            with tarfile.open(file_path, 'r:gz') as tar:
                tar.extractall(output_dir)
        elif file_path.endswith('.tar'):
            with tarfile.open(file_path, 'r:') as tar:
                tar.extractall(output_dir)
        elif file_path.endswith('.zip'):
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(output_dir)
        elif file_path.endswith('.rar'):
            with rarfile.RarFile(file_path, 'r') as rar_ref:
                rar_ref.extractall(output_dir)
        else:
            logging.error(f"Unsupported format: {file_path}")
            return False
        
        logging.info(f"Extraction completed: {output_dir}")
        return True
        
    except Exception as e:
        logging.error(f"Extraction failed: {e}")
        return False


def verify_checksum(file_path: str, expected_checksum: str) -> bool:
    """
    验证文件校验和
    
    Args:
        file_path: 文件路径
        expected_checksum: 期望的 MD5/SHA256
        
    Returns:
        valid: 是否匹配
    """
    if not expected_checksum:
        return True
    
    logging.info(f"Verifying checksum: {file_path}")
    
    hash_algo = 'sha256' if len(expected_checksum) == 64 else 'md5'
    hash_func = hashlib.sha256 if hash_algo == 'sha256' else hashlib.md5
    
    hasher = hash_func()
    
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            hasher.update(chunk)
    
    actual_checksum = hasher.hexdigest()
    
    if actual_checksum == expected_checksum:
        logging.info("Checksum verified!")
        return True
    else:
        logging.error(f"Checksum mismatch!")
        logging.error(f"  Expected: {expected_checksum}")
        logging.error(f"  Actual:   {actual_checksum}")
        return False


# ============================================
# 数据集下载函数
# ============================================

def download_lfw(output_root: str) -> bool:
    """下载 LFW 数据集"""
    logging.info("=" * 50)
    logging.info("Downloading LFW dataset...")
    
    output_dir = os.path.join(output_root, 'lfw')
    os.makedirs(output_dir, exist_ok=True)
    
    # 下载图像
    tgz_path = os.path.join(output_dir, 'lfw.tgz')
    if not os.path.exists(tgz_path):
        success = download_file(DATASETS['lfw']['url'], tgz_path)
        if not success:
            return False
    
    # 解压
    extract_file(tgz_path, output_dir)
    
    # 下载 pairs.txt
    pairs_path = os.path.join(output_dir, 'pairs.txt')
    if not os.path.exists(pairs_path):
        download_file(DATASETS['lfw_pairs']['url'], pairs_path)
    
    logging.info("LFW download completed!")
    return True


def download_widerface(output_root: str) -> bool:
    """下载 WIDER FACE 数据集"""
    logging.info("=" * 50)
    logging.info("Downloading WIDER FACE dataset...")
    
    output_dir = os.path.join(output_root, 'widerface')
    os.makedirs(output_dir, exist_ok=True)
    
    # 下载训练集
    train_zip = os.path.join(output_dir, 'WIDER_train.zip')
    if not os.path.exists(train_zip):
        success = download_file(DATASETS['widerface']['url'], train_zip)
        if not success:
            return False
        extract_file(train_zip, output_dir)
    
    # 下载验证集
    val_zip = os.path.join(output_dir, 'WIDER_val.zip')
    if not os.path.exists(val_zip):
        success = download_file(DATASETS['widerface_val']['url'], val_zip)
        if not success:
            return False
        extract_file(val_zip, output_dir)
    
    # 下载标注
    gt_url = "https://huggingface.co/datasets/valhalla/WIDER-face/resolve/main/data/wider_face_split.zip"
    gt_zip = os.path.join(output_dir, 'wider_face_split.zip')
    if not os.path.exists(gt_zip):
        download_file(gt_url, gt_zip)
        extract_file(gt_zip, output_dir)
    
    logging.info("WIDER FACE download completed!")
    return True


def download_vggface2(output_root: str, auth_url: Optional[str] = None) -> bool:
    """
    下载 VGGFace2 数据集
    
    需要先从官网申请下载链接
    """
    logging.info("=" * 50)
    logging.info("Downloading VGGFace2 dataset...")
    logging.info("Note: You need to apply for download access from:")
    logging.info("  https://www.robots.ox.ac.uk/~vgg/data/vgg_face2/")
    
    output_dir = os.path.join(output_root, 'vggface2')
    os.makedirs(output_dir, exist_ok=True)
    
    # 检查是否有授权 URL
    if auth_url:
        train_tar = os.path.join(output_dir, 'vggface2_train.tar.gz')
        if not os.path.exists(train_tar):
            success = download_file(auth_url, train_tar)
            if not success:
                return False
            extract_file(train_tar, output_dir)
    else:
        logging.info("Please provide the authorized download URL")
        logging.info("Usage: python download_datasets.py --dataset vggface2 --auth-url <URL>")
    
    return True


def download_webface12m(output_root: str, auth_url: Optional[str] = None) -> bool:
    """
    下载 WebFace12M 数据集
    
    需要先从官网申请
    """
    logging.info("=" * 50)
    logging.info("Downloading WebFace12M dataset...")
    logging.info("Note: You need to apply for download access from:")
    logging.info("  https://www.face-benchmark.org/download.html")
    
    output_dir = os.path.join(output_root, 'webface12m')
    os.makedirs(output_dir, exist_ok=True)
    
    if auth_url:
        # WebFace12M 通常是多个 zip 文件
        logging.info("Please provide the authorized download URL(s)")
    else:
        logging.info("Please apply for access and provide download URL")
    
    return True


# ============================================
# 主函数
# ============================================

def list_datasets():
    """列出可用数据集"""
    print("\n可用数据集:")
    print("=" * 70)
    
    for key, info in DATASETS.items():
        auth = " [需要授权]" if info.get('requires_auth') else ""
        print(f"\n{key}{auth}")
        print(f"  描述：{info['description']}")
        print(f"  大小：{info['size']}")
        print(f"  格式：{info['format']}")
        if info.get('requires_auth'):
            print(f"  申请：{info['url']}")
    
    print("\n" + "=" * 70)
    print("\n下载命令示例:")
    print("  python download_datasets.py --dataset lfw")
    print("  python download_datasets.py --dataset widerface")
    print("  python download_datasets.py --dataset vggface2 --auth-url <URL>")
    print()


def main():
    parser = argparse.ArgumentParser(description="Download face datasets")
    
    parser.add_argument(
        "--dataset",
        type=str,
        choices=list(DATASETS.keys()) + ['all'],
        default=None,
        help="Dataset to download"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="datasets",
        help="Output directory"
    )
    parser.add_argument(
        "--auth-url",
        type=str,
        default=None,
        help="Authorized download URL (for protected datasets)"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available datasets"
    )
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    if args.list or args.dataset is None:
        list_datasets()
        return
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 下载指定数据集
    if args.dataset == 'all':
        # 下载所有免费数据集
        download_lfw(args.output_dir)
        download_widerface(args.output_dir)
    elif args.dataset == 'lfw':
        download_lfw(args.output_dir)
    elif args.dataset == 'widerface':
        download_widerface(args.output_dir)
    elif args.dataset == 'vggface2':
        download_vggface2(args.output_dir, args.auth_url)
    elif args.dataset == 'webface12m':
        download_webface12m(args.output_dir, args.auth_url)
    else:
        logging.error(f"Unknown dataset: {args.dataset}")


if __name__ == "__main__":
    main()
