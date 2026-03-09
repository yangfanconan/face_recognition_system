#!/usr/bin/env python3
"""调试模型输出"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import cv2
import torch
from inference.detector import Detector
from inference.recognizer import Recognizer

device = 'cuda'

# 加载模型
print("加载检测模型...")
detector = Detector(score_thresh=0.5, device=device)

print("加载识别模型...")
recognizer = Recognizer(device=device)

# 测试图像
img = cv2.imread("datasets/lfw/Aaron_Eckhart/Aaron_Eckhart_0001.jpg")
print(f"图像形状：{img.shape}")

# 检测
print("\n检测人脸...")
result = detector.detect(img)
print(f"检测结果类型：{type(result)}")
print(f"检测结果：{result}")

if isinstance(result, dict):
    print(f"keys: {result.keys()}")
    for k, v in result.items():
        print(f"  {k}: {type(v)} = {v if len(v) == 0 else len(v)}")

# 如果有检测到人脸，尝试提取特征
if isinstance(result, dict) and len(result.get('boxes', [])) > 0:
    bbox = result['boxes'][0]
    landmarks = result.get('landmarks', [None])[0]
    
    print(f"\nbbox: {bbox}")
    print(f"landmarks: {landmarks}")
    
    print("\n提取特征...")
    try:
        feature = recognizer.extract(img, bbox, landmarks)
        print(f"特征形状：{feature.shape}")
        print(f"特征：{feature[:10]}...")
    except Exception as e:
        print(f"特征提取失败：{e}")
        import traceback
        traceback.print_exc()
