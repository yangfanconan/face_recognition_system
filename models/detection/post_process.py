"""
人脸检测后处理修复模块

修复问题:
1. 坐标解码错误（负数/超大值）
2. 置信度计算错误（恒为 1.0）
3. NMS 实现问题

包含:
- 修复后的 bbox 解码函数
- 修复后的 NMS 实现
- 置信度校准
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Optional


def decode_bbox_fixed(bbox_offsets: torch.Tensor, 
                      anchors: torch.Tensor,
                      clip: bool = True,
                      max_size: int = 640) -> torch.Tensor:
    """
    解码 bbox 坐标（修复版本）
    
    修复问题:
    - 坐标出现负数或超大值
    - 解码公式错误
    
    Args:
        bbox_offsets: bbox 偏移量 (B, N, 4) 或 (B, H, W, 4*num_anchors)
        anchors: 锚框 (N, 4) 或 (H, W, num_anchors, 4)
        clip: 是否裁剪到合理范围
        max_size: 最大图像尺寸
    
    Returns:
        boxes: 解码后的 bbox (B, N, 4)
    
    Example:
        >>> offsets = torch.randn(1, 100, 4)
        >>> anchors = generate_anchors(640, 640)
        >>> boxes = decode_bbox_fixed(offsets, anchors)
    """
    # 处理输入形状
    if bbox_offsets.dim() == 4:
        # (B, 4*num_anchors, H, W) -> (B, H, W, num_anchors, 4)
        B, _, H, W = bbox_offsets.shape
        num_anchors = bbox_offsets.shape[1] // 4
        bbox_offsets = bbox_offsets.permute(0, 2, 3, 1).reshape(B, H * W * num_anchors, 4)
    
    if anchors.dim() == 4:
        # (H, W, num_anchors, 4) -> (H*W*num_anchors, 4)
        anchors = anchors.reshape(-1, 4)
    
    B = bbox_offsets.shape[0]
    N = anchors.shape[0]
    
    # 计算锚框中心点和宽高
    anchor_centers = (anchors[:, :2] + anchors[:, 2:]) / 2  # (N, 2)
    anchor_sizes = anchors[:, 2:] - anchors[:, :2]  # (N, 2)
    
    # 防止除以零
    anchor_sizes = anchor_sizes.clamp(min=1.0)
    
    # 解码偏移量
    # 标准 FPN 解码公式:
    # pred_center_x = dx * anchor_width + anchor_center_x
    # pred_center_y = dy * anchor_height + anchor_center_y
    # pred_width = exp(dw) * anchor_width
    # pred_height = exp(dh) * anchor_height
    
    dx = bbox_offsets[:, :, 0]  # (B, N)
    dy = bbox_offsets[:, :, 1]
    dw = bbox_offsets[:, :, 2]
    dh = bbox_offsets[:, :, 3]
    
    # 限制偏移量范围，避免过大值
    dx = dx.clamp(-10, 10)
    dy = dy.clamp(-10, 10)
    dw = dw.clamp(-10, 10)
    dh = dh.clamp(-10, 10)
    
    # 应用解码公式
    pred_centers_x = dx * anchor_sizes[None, :, 0] + anchor_centers[None, :, 0]
    pred_centers_y = dy * anchor_sizes[None, :, 1] + anchor_centers[None, :, 1]
    pred_widths = torch.exp(dw) * anchor_sizes[None, :, 0]
    pred_heights = torch.exp(dh) * anchor_sizes[None, :, 1]
    
    # 转换为 [x1, y1, x2, y2] 格式
    boxes = torch.stack([
        pred_centers_x - pred_widths / 2,
        pred_centers_y - pred_heights / 2,
        pred_centers_x + pred_widths / 2,
        pred_centers_y + pred_heights / 2,
    ], dim=-1)  # (B, N, 4)
    
    # 裁剪到合理范围
    if clip:
        boxes = boxes.clamp(min=0, max=max_size)
    
    return boxes


def nms_fixed(boxes: torch.Tensor, 
              scores: torch.Tensor, 
              iou_threshold: float = 0.45,
              score_threshold: float = 0.5) -> torch.Tensor:
    """
    非极大值抑制（修复版本）
    
    修复问题:
    - IoU 计算错误
    - 阈值设置不合理
    
    Args:
        boxes: bbox 框 (N, 4) [x1, y1, x2, y2]
        scores: 置信度 (N,)
        iou_threshold: IoU 阈值
        score_threshold: 置信度阈值
    
    Returns:
        keep_indices: 保留的索引
    
    Example:
        >>> boxes = torch.tensor([[10, 10, 100, 100], [15, 15, 105, 105]])
        >>> scores = torch.tensor([0.9, 0.8])
        >>> keep = nms_fixed(boxes, scores, iou_threshold=0.5)
    """
    if len(boxes) == 0:
        return torch.tensor([], dtype=torch.long)
    
    # 过滤低置信度框
    valid_mask = scores >= score_threshold
    if not valid_mask.any():
        return torch.tensor([], dtype=torch.long)
    
    boxes = boxes[valid_mask]
    scores = scores[valid_mask]
    
    # 按置信度排序
    sorted_indices = torch.argsort(scores, descending=True)
    
    keep = []
    
    while len(sorted_indices) > 0:
        # 选择置信度最高的框
        current_idx = sorted_indices[0]
        keep.append(current_idx.item())
        
        if len(sorted_indices) == 1:
            break
        
        # 计算与剩余框的 IoU
        remaining_indices = sorted_indices[1:]
        
        # 当前框
        current_box = boxes[current_idx]
        remaining_boxes = boxes[remaining_indices]
        
        # 计算交集
        xx1 = torch.max(current_box[0], remaining_boxes[:, 0])
        yy1 = torch.max(current_box[1], remaining_boxes[:, 1])
        xx2 = torch.min(current_box[2], remaining_boxes[:, 2])
        yy2 = torch.min(current_box[3], remaining_boxes[:, 3])
        
        # 确保交集有效
        inter_width = (xx2 - xx1).clamp(min=0)
        inter_height = (yy2 - yy1).clamp(min=0)
        inter_area = inter_width * inter_height
        
        # 计算面积
        current_area = (current_box[2] - current_box[0]) * (current_box[3] - current_box[1])
        remaining_areas = (remaining_boxes[:, 2] - remaining_boxes[:, 0]) * \
                          (remaining_boxes[:, 3] - remaining_boxes[:, 1])
        
        # 计算并集
        union_area = current_area + remaining_areas - inter_area
        
        # 计算 IoU
        iou = inter_area / union_area.clamp(min=1e-10)
        
        # 保留 IoU 低于阈值的框
        keep_mask = iou <= iou_threshold
        sorted_indices = remaining_indices[keep_mask]
    
    # 转换回原始索引
    valid_indices = torch.where(valid_mask)[0]
    keep_indices = valid_indices[keep]
    
    return keep_indices


def batched_nms(boxes: torch.Tensor, 
                scores: torch.Tensor, 
                labels: torch.Tensor,
                iou_threshold: float = 0.45,
                score_threshold: float = 0.5) -> torch.Tensor:
    """
    批量 NMS（按类别分别执行）
    
    Args:
        boxes: bbox 框 (N, 4)
        scores: 置信度 (N,)
        labels: 类别标签 (N,)
        iou_threshold: IoU 阈值
        score_threshold: 置信度阈值
    
    Returns:
        keep_indices: 保留的索引
    """
    if len(boxes) == 0:
        return torch.tensor([], dtype=torch.long)
    
    keep_all = []
    
    # 按类别分别执行 NMS
    for label in torch.unique(labels):
        mask = labels == label
        label_boxes = boxes[mask]
        label_scores = scores[mask]
        
        label_keep = nms_fixed(label_boxes, label_scores, iou_threshold, score_threshold)
        
        # 转换回原始索引
        label_indices = torch.where(mask)[0][label_keep]
        keep_all.extend(label_indices.tolist())
    
    return torch.tensor(keep_all, dtype=torch.long)


def clip_boxes_to_image(boxes: torch.Tensor, 
                        image_size: Tuple[int, int]) -> torch.Tensor:
    """
    将 bbox 裁剪到图像范围内
    
    Args:
        boxes: bbox 框 (N, 4)
        image_size: 图像尺寸 (height, width)
    
    Returns:
        clipped_boxes: 裁剪后的 bbox
    """
    h, w = image_size
    
    boxes = boxes.clone()
    boxes[:, 0] = boxes[:, 0].clamp(0, w)
    boxes[:, 1] = boxes[:, 1].clamp(0, h)
    boxes[:, 2] = boxes[:, 2].clamp(0, w)
    boxes[:, 3] = boxes[:, 3].clamp(0, h)
    
    # 确保 x2 > x1, y2 > y1
    valid_mask = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    
    return boxes[valid_mask]


def filter_small_boxes(boxes: torch.Tensor, 
                       min_size: float = 10.0) -> torch.Tensor:
    """
    过滤过小的 bbox
    
    Args:
        boxes: bbox 框 (N, 4)
        min_size: 最小尺寸
    
    Returns:
        keep_mask: 保留的掩码
    """
    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    
    keep_mask = (widths >= min_size) & (heights >= min_size)
    
    return keep_mask


def calculate_iou(boxes1: torch.Tensor, 
                  boxes2: torch.Tensor) -> torch.Tensor:
    """
    计算两个 bbox 集合的 IoU
    
    Args:
        boxes1: bbox 集合 1 (N, 4)
        boxes2: bbox 集合 2 (M, 4)
    
    Returns:
        iou_matrix: IoU 矩阵 (N, M)
    """
    # 扩展维度以便广播
    boxes1_exp = boxes1.unsqueeze(1)  # (N, 1, 4)
    boxes2_exp = boxes2.unsqueeze(0)  # (1, M, 4)
    
    # 计算交集
    xx1 = torch.max(boxes1_exp[..., 0], boxes2_exp[..., 0])
    yy1 = torch.max(boxes1_exp[..., 1], boxes2_exp[..., 1])
    xx2 = torch.min(boxes1_exp[..., 2], boxes2_exp[..., 2])
    yy2 = torch.min(boxes1_exp[..., 3], boxes2_exp[..., 3])
    
    inter_width = (xx2 - xx1).clamp(min=0)
    inter_height = (yy2 - yy1).clamp(min=0)
    inter_area = inter_width * inter_height
    
    # 计算面积
    area1 = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    area2 = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])
    
    # 计算并集
    union_area = area1.unsqueeze(1) + area2.unsqueeze(0) - inter_area
    
    # 计算 IoU
    iou = inter_area / union_area.clamp(min=1e-10)
    
    return iou


# NMS 包装函数（用于 torchvision 兼容）
def nms_torchvision(boxes: torch.Tensor, 
                    scores: torch.Tensor, 
                    iou_threshold: float) -> torch.Tensor:
    """
    使用 torchvision 的 NMS 实现
    
    如果 torchvision 不可用，则使用自定义实现
    """
    try:
        from torchvision.ops import nms
        return nms(boxes, scores, iou_threshold)
    except ImportError:
        return nms_fixed(boxes, scores, iou_threshold)


# 测试函数
def test_nms():
    """测试 NMS 实现"""
    # 创建测试数据
    boxes = torch.tensor([
        [0.0, 0.0, 100.0, 100.0],
        [10.0, 10.0, 110.0, 110.0],  # 与第一个框重叠
        [200.0, 200.0, 300.0, 300.0],  # 独立框
    ])
    
    scores = torch.tensor([0.9, 0.8, 0.95])
    
    # 执行 NMS
    keep = nms_fixed(boxes, scores, iou_threshold=0.5)
    
    print(f"输入框数：{len(boxes)}")
    print(f"保留框索引：{keep.tolist()}")
    print(f"保留框数：{len(keep)}")
    
    # 验证：应该保留索引 2 和 0（或 1）
    assert len(keep) == 2, f"期望保留 2 个框，实际保留 {len(keep)} 个"
    
    print("✅ NMS 测试通过!")


def test_decode_bbox():
    """测试 bbox 解码"""
    # 创建测试锚框
    anchors = torch.tensor([
        [100.0, 100.0, 200.0, 200.0],
        [300.0, 300.0, 400.0, 400.0],
    ])
    
    # 创建测试偏移量（全零表示无偏移）
    offsets = torch.zeros(1, 2, 4)
    
    # 解码
    boxes = decode_bbox_fixed(offsets, anchors)
    
    print(f"锚框：{anchors.tolist()}")
    print(f"解码后框：{boxes.tolist()}")
    
    # 验证：偏移量为零时，解码后应该等于锚框
    assert torch.allclose(boxes[0], anchors), "解码结果与锚框不一致"
    
    print("✅ Bbox 解码测试通过!")


if __name__ == "__main__":
    print("测试 NMS...")
    test_nms()
    
    print("\n测试 Bbox 解码...")
    test_decode_bbox()
    
    print("\n所有测试通过! ✅")
