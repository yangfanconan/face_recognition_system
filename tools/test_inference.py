"""
快速推理测试脚本

测试端到端推理流程
"""

import os
import sys
import time
import argparse
import logging
from pathlib import Path

import numpy as np
import cv2

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))


def create_test_image(width: int = 640, height: int = 480) -> np.ndarray:
    """创建测试图像"""
    image = np.zeros((height, width, 3), dtype=np.uint8)
    center_x, center_y = width // 2, height // 2
    face_size = min(width, height) // 4
    
    cv2.ellipse(image, (center_x, center_y), (face_size, int(face_size * 1.2)), 
                0, 0, 360, (100, 100, 100), -1)
    
    eye_y = center_y - int(face_size * 0.3)
    eye_offset = int(face_size * 0.35)
    cv2.circle(image, (center_x - eye_offset, eye_y), int(face_size * 0.1), (50, 50, 50), -1)
    cv2.circle(image, (center_x + eye_offset, eye_y), int(face_size * 0.1), (50, 50, 50), -1)
    
    nose_y = center_y + int(face_size * 0.1)
    cv2.circle(image, (center_x, nose_y), int(face_size * 0.08), (80, 80, 80), -1)
    
    mouth_y = center_y + int(face_size * 0.4)
    cv2.ellipse(image, (center_x, mouth_y), (int(face_size * 0.15), int(face_size * 0.05)),
                0, 0, 180, (60, 60, 60), -1)
    
    return image


def test_detector():
    """测试检测器"""
    print("\n" + "=" * 60)
    print("测试人脸检测器")
    print("=" * 60)
    
    try:
        from inference import Detector
        image = create_test_image()
        detector = Detector(score_thresh=0.5)
        
        for _ in range(5):
            detector.detect(image)
        
        iterations = 20
        times = []
        for _ in range(iterations):
            start = time.time()
            detections = detector.detect(image)
            end = time.time()
            times.append((end - start) * 1000)
        
        print(f"\n检测结果:")
        print(f"  检测人脸数：{len(detections)}")
        print(f"  平均推理时间：{np.mean(times):.2f}ms")
        
        return True
    except Exception as e:
        print(f"\n❌ 检测器测试失败：{e}")
        return False


def test_recognizer():
    """测试识别器"""
    print("\n" + "=" * 60)
    print("测试人脸识别器")
    print("=" * 60)
    
    try:
        from inference import Recognizer
        image = create_test_image(112, 112)
        bbox = np.array([10, 10, 100, 100])
        recognizer = Recognizer()
        
        for _ in range(5):
            recognizer.extract(image, bbox)
        
        iterations = 20
        times = []
        features = []
        
        for _ in range(iterations):
            start = time.time()
            feature = recognizer.extract(image, bbox)
            end = time.time()
            times.append((end - start) * 1000)
            features.append(feature)
        
        features = np.stack(features)
        print(f"\n识别结果:")
        print(f"  特征维度：{features[0].shape}")
        print(f"  平均推理时间：{np.mean(times):.2f}ms")
        
        return True
    except Exception as e:
        print(f"\n❌ 识别器测试失败：{e}")
        return False


def test_matcher():
    """测试匹配器"""
    print("\n" + "=" * 60)
    print("测试特征匹配器")
    print("=" * 60)
    
    try:
        from inference import Matcher
        np.random.seed(42)
        feat1 = np.random.randn(512).astype(np.float32)
        feat2 = np.random.randn(512).astype(np.float32)
        feat1 = feat1 / np.linalg.norm(feat1)
        feat2 = feat2 / np.linalg.norm(feat2)
        
        matcher = Matcher(threshold=0.6)
        
        is_same, sim = matcher.verify(feat1, feat2)
        same_sim, _ = matcher.verify(feat1, feat1)
        
        print(f"\n匹配结果:")
        print(f"  相似度：{sim:.4f}")
        print(f"  自相似度：{same_sim:.4f}")
        
        return True
    except Exception as e:
        print(f"\n❌ 匹配器测试失败：{e}")
        return False


def test_pipeline():
    """测试完整流水线"""
    print("\n" + "=" * 60)
    print("测试端到端流水线")
    print("=" * 60)
    
    try:
        from inference import FaceRecognitionPipeline
        image = create_test_image()
        pipeline = FaceRecognitionPipeline()
        
        det_result = pipeline.detect(image)
        print(f"\n检测结果：{det_result['count']} 张人脸")
        
        if det_result['count'] > 0:
            bbox = det_result['faces'][0]['bbox']
            extract_result = pipeline.extract(image, bbox=bbox)
            print(f"特征提取：{'成功' if extract_result.get('success') else '失败'}")
        
        return True
    except Exception as e:
        print(f"\n❌ 流水线测试失败：{e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Test inference pipeline")
    parser.add_argument("--test", type=str, default='all',
                        choices=['all', 'detector', 'recognizer', 'matcher', 'pipeline'])
    args = parser.parse_args()
    
    print("=" * 60)
    print("DDFD-FaceRec 推理测试")
    print("=" * 60)
    
    results = {}
    
    if args.test in ['all', 'detector']:
        results['detector'] = test_detector()
    if args.test in ['all', 'recognizer']:
        results['recognizer'] = test_recognizer()
    if args.test in ['all', 'matcher']:
        results['matcher'] = test_matcher()
    if args.test in ['all', 'pipeline']:
        results['pipeline'] = test_pipeline()
    
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    
    for name, passed in results.items():
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"  {name}: {status}")
    
    return all(results.values())


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
