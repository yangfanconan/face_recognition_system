#!/usr/bin/env python3
"""下载 LFW pairs.txt 文件"""

import requests
from pathlib import Path

def download_lfw_pairs():
    """下载 LFW pairs.txt"""
    
    # 多个备用 URL
    urls = [
        "http://vis-www.cs.umass.edu/lfw/pairs.txt",
        "https://raw.githubusercontent.com/GuilhermeFerreira/lfw/master/pairs.txt",
        "https://github.com/timesler/facenet-pytorch/raw/master/data/lfw/pairs.txt",
    ]
    
    save_path = Path("datasets/lfw/pairs.txt")
    
    # 确保目录存在
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    for url in urls:
        print(f"Trying to download pairs.txt...")
        print(f"URL: {url}")
        
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            with open(save_path, 'wb') as f:
                f.write(response.content)
            
            print(f"Download successful!")
            print(f"File size: {len(response.content)} bytes")
            
            # 验证文件内容
            with open(save_path, 'r') as f:
                lines = f.readlines()
            
            print(f"Lines: {len(lines)}")
            print(f"First line: {lines[0].strip()}")
            return
            
        except Exception as e:
            print(f"Download failed: {e}\n")
    
    raise RuntimeError("All URLs failed")

if __name__ == "__main__":
    download_lfw_pairs()
