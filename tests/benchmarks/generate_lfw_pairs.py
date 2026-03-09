#!/usr/bin/env python3
"""
根据 LFW 目录结构生成 pairs.txt

LFW pairs.txt 格式:
第一行：折叠数 (通常是 10)
后续行有两种格式:
- 同一人：name\timg1_num\timg2_num
- 不同人：name1\timg1_num\tname2\timg2_num
"""

import os
from pathlib import Path
from collections import defaultdict
import random

def generate_lfw_pairs(lfw_dir: str, output_path: str, n_folds: int = 10, seed: int = 42):
    """生成 LFW pairs.txt"""
    
    lfw_path = Path(lfw_dir)
    
    if not lfw_path.exists():
        print(f"LFW 目录不存在：{lfw_dir}")
        return False
    
    # 扫描所有人脸目录
    person_dirs = [d for d in lfw_path.iterdir() if d.is_dir()]
    print(f"找到 {len(person_dirs)} 个人")
    
    # 收集每个人的图像
    person_images = defaultdict(list)
    for person_dir in person_dirs:
        images = sorted(list(person_dir.glob("*.jpg")))
        if len(images) >= 2:
            person_images[person_dir.name] = images
    
    print(f"{len(person_images)} 个人有 2 张以上图像")
    
    # 生成同人对 (每人的所有图像对)
    same_pairs = []
    for name, images in person_images.items():
        for i in range(len(images)):
            for j in range(i + 1, len(images)):
                # 提取图像编号 (如 Aaron_Eckhart_0001.jpg -> 0001)
                img1_num = images[i].stem.split('_')[-1]
                img2_num = images[j].stem.split('_')[-1]
                same_pairs.append((name, img1_num, img2_num))
    
    print(f"同人对：{len(same_pairs)}")
    
    # 生成异人对 (随机配对不同人)
    names = list(person_images.keys())
    random.seed(seed)
    
    # 异人对数量与同人对大致相等
    diff_pairs = []
    max_attempts = len(same_pairs) * 10
    attempts = 0
    
    while len(diff_pairs) < len(same_pairs) and attempts < max_attempts:
        i, j = random.sample(range(len(names)), 2)
        
        if len(person_images[names[i]]) >= 1 and len(person_images[names[j]]) >= 1:
            img1 = random.choice(person_images[names[i]])
            img2 = random.choice(person_images[names[j]])
            
            img1_num = img1.stem.split('_')[-1]
            img2_num = img2.stem.split('_')[-1]
            
            diff_pairs.append((names[i], img1_num, names[j], img2_num))
        
        attempts += 1
    
    print(f"异人对：{len(diff_pairs)}")
    
    # 写入文件
    with open(output_path, 'w') as f:
        # 第一行：折叠数
        f.write(f"{n_folds}\n")
        
        # 写入同人对
        for name, img1, img2 in same_pairs:
            f.write(f"{name}\t{img1}\t{img2}\n")
        
        # 写入异人对
        for name1, img1, name2, img2 in diff_pairs:
            f.write(f"{name1}\t{img1}\t{name2}\t{img2}\n")
    
    print(f"\npairs.txt 已生成：{output_path}")
    print(f"总对数：{len(same_pairs) + len(diff_pairs)}")
    
    return True

if __name__ == "__main__":
    lfw_dir = "datasets/lfw"
    output_path = "datasets/lfw/pairs.txt"
    
    success = generate_lfw_pairs(lfw_dir, output_path)
    
    if success:
        # 验证文件
        with open(output_path, 'r') as f:
            lines = f.readlines()
        
        print(f"\n验证:")
        print(f"总行数：{len(lines)}")
        print(f"第一行：{lines[0].strip()}")
        print(f"第二行：{lines[1].strip() if len(lines) > 1 else 'N/A'}")
