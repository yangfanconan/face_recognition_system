#!/usr/bin/env python3
"""从 HuggingFace 下载 LFW pairs.txt"""

from huggingface_hub import hf_hub_download
from pathlib import Path

def download_from_hf():
    """从 HuggingFace 下载"""
    
    print("从 HuggingFace 下载 LFW pairs.txt...")
    
    try:
        # 尝试从 HuggingFace 下载
        file_path = hf_hub_download(
            repo_id="lhoestq/lfw",
            filename="pairs.txt",
            repo_type="dataset",
            cache_dir="datasets/.cache"
        )
        
        # 复制到目标位置
        save_path = Path("datasets/lfw/pairs.txt")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        import shutil
        shutil.copy(file_path, save_path)
        
        print(f"下载成功！")
        print(f"保存路径：{save_path}")
        
        # 验证
        with open(save_path, 'r') as f:
            lines = f.readlines()
        
        print(f"行数：{len(lines)}")
        print(f"第一行：{lines[0].strip()}")
        
    except Exception as e:
        print(f"HuggingFace 下载失败：{e}")
        print("\n请手动下载:")
        print("1. 访问：http://vis-www.cs.umass.edu/lfw/pairs.txt")
        print("2. 保存到：datasets/lfw/pairs.txt")

if __name__ == "__main__":
    download_from_hf()
