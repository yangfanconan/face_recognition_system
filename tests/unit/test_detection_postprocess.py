"""
检测后处理单元测试

测试内容:
- bbox 解码修复
- NMS 修复
- 坐标裁剪
"""

import unittest
import torch
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.detection.post_process import (
    decode_bbox_fixed,
    nms_fixed,
    clip_boxes_to_image,
    filter_small_boxes,
    calculate_iou
)


class TestDecodeBbox(unittest.TestCase):
    """Bbox 解码测试"""
    
    def test_zero_offsets(self):
        """测试零偏移量解码"""
        anchors = torch.tensor([
            [100.0, 100.0, 200.0, 200.0],
            [300.0, 300.0, 400.0, 400.0],
        ])
        
        # 零偏移量
        offsets = torch.zeros(1, 2, 4)
        
        boxes = decode_bbox_fixed(offsets, anchors)
        
        # 零偏移时，解码后应该接近锚框
        expected = torch.tensor([
            [100.0, 100.0, 200.0, 200.0],
            [300.0, 300.0, 400.0, 400.0],
        ]).unsqueeze(0)
        
        self.assertTrue(torch.allclose(boxes, expected, atol=1e-5))
    
    def test_clip(self):
        """测试坐标裁剪"""
        anchors = torch.tensor([
            [0.0, 0.0, 100.0, 100.0],
        ])
        
        # 大偏移量
        offsets = torch.tensor([
            [10.0, 10.0, 10.0, 10.0],  # 会导致超大框
        ]).unsqueeze(0)
        
        boxes = decode_bbox_fixed(offsets, anchors, clip=True, max_size=640)
        
        # 坐标应该在合理范围内
        self.assertTrue((boxes >= 0).all())
        self.assertTrue((boxes <= 640).all())
    
    def test_no_negative_coordinates(self):
        """测试无负坐标"""
        anchors = torch.tensor([
            [50.0, 50.0, 100.0, 100.0],
        ])
        
        # 负偏移量
        offsets = torch.tensor([
            [-5.0, -5.0, -2.0, -2.0],
        ]).unsqueeze(0)
        
        boxes = decode_bbox_fixed(offsets, anchors, clip=True)
        
        # 坐标不应该为负
        self.assertTrue((boxes >= 0).all())


class TestNMS(unittest.TestCase):
    """NMS 测试"""
    
    def test_basic_nms(self):
        """测试基础 NMS"""
        boxes = torch.tensor([
            [0.0, 0.0, 100.0, 100.0],
            [10.0, 10.0, 110.0, 110.0],  # 与第一个框重叠
            [200.0, 200.0, 300.0, 300.0],  # 独立框
        ])
        
        scores = torch.tensor([0.9, 0.8, 0.95])
        
        keep = nms_fixed(boxes, scores, iou_threshold=0.5)
        
        # 应该保留 2 个框（索引 2 和 0 或 1）
        self.assertEqual(len(keep), 2)
        
        # 最高分数的框应该保留
        self.assertIn(2, keep.tolist())
    
    def test_nms_all_overlapping(self):
        """测试全重叠框 NMS"""
        boxes = torch.tensor([
            [0.0, 0.0, 100.0, 100.0],
            [5.0, 5.0, 105.0, 105.0],
            [10.0, 10.0, 110.0, 110.0],
        ])
        
        scores = torch.tensor([0.9, 0.8, 0.7])
        
        keep = nms_fixed(boxes, scores, iou_threshold=0.5)
        
        # 应该只保留 1 个框（最高分数）
        self.assertEqual(len(keep), 1)
        self.assertEqual(keep[0].item(), 0)
    
    def test_nms_empty_input(self):
        """测试空输入 NMS"""
        boxes = torch.tensor([]).reshape(0, 4)
        scores = torch.tensor([])
        
        keep = nms_fixed(boxes, scores, iou_threshold=0.5)
        
        self.assertEqual(len(keep), 0)
    
    def test_nms_score_threshold(self):
        """测试分数阈值"""
        boxes = torch.tensor([
            [0.0, 0.0, 100.0, 100.0],
            [200.0, 200.0, 300.0, 300.0],
        ])
        
        scores = torch.tensor([0.3, 0.9])  # 第一个框分数低于阈值
        
        keep = nms_fixed(boxes, scores, iou_threshold=0.5, score_threshold=0.5)
        
        # 应该只保留第二个框
        self.assertEqual(len(keep), 1)
        self.assertEqual(keep[0].item(), 1)


class TestClipBoxes(unittest.TestCase):
    """坐标裁剪测试"""
    
    def test_clip_to_image(self):
        """测试裁剪到图像范围"""
        boxes = torch.tensor([
            [-10.0, -10.0, 110.0, 110.0],  # 超出图像
            [50.0, 50.0, 150.0, 150.0],    # 在图像内
        ])
        
        image_size = (100, 100)  # (H, W)
        clipped = clip_boxes_to_image(boxes, image_size)
        
        # 坐标应该在图像范围内
        self.assertTrue((clipped[:, 0] >= 0).all())
        self.assertTrue((clipped[:, 1] >= 0).all())
        self.assertTrue((clipped[:, 2] <= 100).all())
        self.assertTrue((clipped[:, 3] <= 100).all())
    
    def test_clip_invalid_boxes(self):
        """测试无效框过滤"""
        boxes = torch.tensor([
            [50.0, 50.0, 40.0, 40.0],  # x2 < x1, y2 < y1
            [10.0, 10.0, 100.0, 100.0],  # 有效框
        ])
        
        image_size = (100, 100)
        clipped = clip_boxes_to_image(boxes, image_size)
        
        # 应该只保留有效框
        self.assertEqual(len(clipped), 1)


class TestFilterSmallBoxes(unittest.TestCase):
    """小框过滤测试"""
    
    def test_filter_by_size(self):
        """测试按尺寸过滤"""
        boxes = torch.tensor([
            [0.0, 0.0, 5.0, 5.0],     # 太小
            [0.0, 0.0, 20.0, 20.0],   # 足够大
        ])
        
        keep_mask = filter_small_boxes(boxes, min_size=10.0)
        
        # 应该只保留大框
        self.assertEqual(keep_mask.sum().item(), 1)
        self.assertTrue(keep_mask[1])


class TestCalculateIoU(unittest.TestCase):
    """IoU 计算测试"""
    
    def test_iou_identical(self):
        """测试相同框的 IoU"""
        boxes1 = torch.tensor([
            [0.0, 0.0, 100.0, 100.0],
        ])
        
        boxes2 = torch.tensor([
            [0.0, 0.0, 100.0, 100.0],
        ])
        
        iou = calculate_iou(boxes1, boxes2)
        
        # 相同框的 IoU 应该是 1
        self.assertAlmostEqual(iou[0, 0].item(), 1.0, places=5)
    
    def test_iou_disjoint(self):
        """测试不重叠框的 IoU"""
        boxes1 = torch.tensor([
            [0.0, 0.0, 100.0, 100.0],
        ])
        
        boxes2 = torch.tensor([
            [200.0, 200.0, 300.0, 300.0],
        ])
        
        iou = calculate_iou(boxes1, boxes2)
        
        # 不重叠框的 IoU 应该是 0
        self.assertAlmostEqual(iou[0, 0].item(), 0.0, places=5)
    
    def test_iou_partial_overlap(self):
        """测试部分重叠框的 IoU"""
        boxes1 = torch.tensor([
            [0.0, 0.0, 100.0, 100.0],
        ])
        
        boxes2 = torch.tensor([
            [50.0, 50.0, 150.0, 150.0],
        ])
        
        iou = calculate_iou(boxes1, boxes2)
        
        # 手动计算 IoU
        # 交集：50x50 = 2500
        # 并集：10000 + 10000 - 2500 = 17500
        # IoU = 2500 / 17500 = 0.1428...
        self.assertAlmostEqual(iou[0, 0].item(), 0.1428, places=3)


if __name__ == "__main__":
    # 运行测试
    unittest.main(verbosity=2)
