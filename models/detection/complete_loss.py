"""
DKGA-Det 完整损失函数

包含:
- Focal Loss (分类)
- CIoU Loss (边界框回归)
- Wing Loss (关键点)
- 标签分配器
- 组合损失
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import math


# ============================================
# Focal Loss
# ============================================

class FocalLoss(nn.Module):
    """
    Focal Loss for dense object detection
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = 'sum',
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        # Sigmoid
        p = torch.sigmoid(inputs)
        
        # BCE
        ce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none'
        )
        
        # 权重 (1 - p)^gamma for positive, p^gamma for negative
        p_t = p * targets + (1 - p) * (1 - targets)
        weight = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        weight = weight * (1 - p_t) ** self.gamma
        
        # 应用权重
        loss = weight * ce_loss
        
        if self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'mean':
            return loss.mean()
        return loss


# ============================================
# CIoU Loss
# ============================================

class CIoULoss(nn.Module):
    """
    Complete IoU Loss
    
    CIoU = IoU - (rho^2(b,b_gt)/c^2 + alpha * v)
    """
    
    def __init__(self, reduction: str = 'sum'):
        super().__init__()
        self.reduction = reduction
    
    def forward(
        self,
        pred_boxes: torch.Tensor,
        target_boxes: torch.Tensor,
    ) -> torch.Tensor:
        # pred_boxes: (N, 4) [x1, y1, x2, y2]
        # target_boxes: (N, 4)
        
        # 转换为 (cx, cy, w, h)
        pred_cx = (pred_boxes[:, 0] + pred_boxes[:, 2]) / 2
        pred_cy = (pred_boxes[:, 1] + pred_boxes[:, 3]) / 2
        pred_w = pred_boxes[:, 2] - pred_boxes[:, 0]
        pred_h = pred_boxes[:, 3] - pred_boxes[:, 1]
        
        target_cx = (target_boxes[:, 0] + target_boxes[:, 2]) / 2
        target_cy = (target_boxes[:, 1] + target_boxes[:, 3]) / 2
        target_w = target_boxes[:, 2] - target_boxes[:, 0]
        target_h = target_boxes[:, 3] - target_boxes[:, 1]
        
        # IoU
        inter_x1 = torch.max(pred_boxes[:, 0], target_boxes[:, 0])
        inter_y1 = torch.max(pred_boxes[:, 1], target_boxes[:, 1])
        inter_x2 = torch.min(pred_boxes[:, 2], target_boxes[:, 2])
        inter_y2 = torch.min(pred_boxes[:, 3], target_boxes[:, 3])
        
        inter_w = torch.clamp(inter_x2 - inter_x1, min=0)
        inter_h = torch.clamp(inter_y2 - inter_y1, min=0)
        inter_area = inter_w * inter_h
        
        pred_area = pred_w * pred_h
        target_area = target_w * target_h
        
        union_area = pred_area + target_area - inter_area
        iou = inter_area / torch.clamp(union_area, min=1e-7)
        
        # 最小外接矩形
        cx_c = torch.max(pred_boxes[:, 2], target_boxes[:, 2]) - torch.min(pred_boxes[:, 0], target_boxes[:, 0])
        cy_c = torch.max(pred_boxes[:, 3], target_boxes[:, 3]) - torch.min(pred_boxes[:, 1], target_boxes[:, 1])
        
        # 中心点距离
        rho = (pred_cx - target_cx) ** 2 + (pred_cy - target_cy) ** 2
        c = cx_c ** 2 + cy_c ** 2
        
        # IoU loss
        iou_loss = 1 - iou
        
        # 纵横比一致性
        v = (4 / (math.pi ** 2)) * torch.pow(
            torch.atan(target_w / torch.clamp(target_h, min=1e-7)) - 
            torch.atan(pred_w / torch.clamp(pred_h, min=1e-7)), 2
        )
        
        # alpha
        alpha = v / (1 - iou + v + 1e-7)
        
        # CIoU
        ciou = iou - (rho / torch.clamp(c, min=1e-7)) - alpha * v
        ciou_loss = 1 - ciou
        
        if self.reduction == 'sum':
            return ciou_loss.sum()
        elif self.reduction == 'mean':
            return ciou_loss.mean()
        return ciou_loss


# ============================================
# Wing Loss
# ============================================

class WingLoss(nn.Module):
    """
    Wing Loss for landmark detection
    
    wing(x) = w * log(1 + |x|/epsilon) if |x| < w
              |x| - C                    otherwise
    """
    
    def __init__(
        self,
        w: float = 10.0,
        epsilon: float = 2.0,
        reduction: str = 'sum',
    ):
        super().__init__()
        self.w = w
        self.epsilon = epsilon
        self.reduction = reduction
        self.C = w - w * math.log(1 + w / epsilon)
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        x = pred - target
        abs_x = torch.abs(x)
        
        loss = torch.where(
            abs_x < self.w,
            self.w * torch.log(1 + abs_x / self.epsilon),
            abs_x - self.C
        )
        
        if self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'mean':
            return loss.mean()
        return loss


# ============================================
# 标签分配器
# ============================================

class LabelAssigner:
    """
    标签分配器
    
    将 ground truth 分配给 anchor points
    """
    
    def __init__(
        self,
        num_classes: int = 1,
        strides: List[int] = [8, 16, 32],
        iou_thresh: float = 0.5,
    ):
        self.num_classes = num_classes
        self.strides = strides
        self.iou_thresh = iou_thresh
    
    def assign(
        self,
        cls_preds: List[torch.Tensor],
        reg_preds: List[torch.Tensor],
        targets: List[Dict],
        device: torch.device,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """
        分配标签
        
        Returns:
            cls_targets: 分类目标列表
            reg_targets: 回归目标列表
            kpt_targets: 关键点目标列表
            pos_masks: 正样本掩码列表
        """
        batch_size = cls_preds[0].shape[0]
        num_levels = len(cls_preds)
        
        cls_targets = []
        reg_targets = []
        kpt_targets = []
        pos_masks = []
        
        for level in range(num_levels):
            stride = self.strides[level]
            h, w = cls_preds[level].shape[2:]
            
            # 生成网格
            shifts_x = torch.arange(0, w * stride, stride, device=device)
            shifts_y = torch.arange(0, h * stride, stride, device=device)
            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing='ij')
            
            # 锚点中心
            anchors_x = shift_x.reshape(-1) + stride / 2
            anchors_y = shift_y.reshape(-1) + stride / 2
            anchors = torch.stack([anchors_x, anchors_y, anchors_x, anchors_y], dim=1)
            
            # 初始化目标
            cls_target = torch.zeros(
                (batch_size, self.num_classes, h, w),
                device=device
            )
            reg_target = torch.zeros(
                (batch_size, 4, h, w),
                device=device
            )
            pos_mask = torch.zeros(
                (batch_size, h, w),
                device=device,
                dtype=torch.bool
            )
            
            # 对每个样本分配
            for b in range(batch_size):
                gt_boxes = targets[b]['boxes']  # (N, 4)
                gt_labels = targets[b]['labels']  # (N,)
                
                if len(gt_boxes) == 0:
                    continue
                
                # 计算 IoU
                num_anchors = len(anchors)
                num_gt = len(gt_boxes)
                
                ious = self._compute_iou(anchors, gt_boxes)  # (num_anchors, num_gt)
                
                # 为每个 GT 选择最佳 anchor
                for gt_idx in range(num_gt):
                    iou = ious[:, gt_idx]
                    
                    # 选择 IoU > thresh 的 anchor
                    pos_indices = torch.where(iou > self.iou_thresh)[0]
                    
                    if len(pos_indices) == 0:
                        # 如果没有，选择最佳
                        pos_indices = torch.argmax(iou, keepdim=True)
                    
                    # 分配标签
                    for idx in pos_indices:
                        y = idx // w
                        x = idx % w
                        
                        cls_target[b, 0, y, x] = 1.0  # 人脸
                        pos_mask[b, y, x] = True
                        
                        # 回归目标 (cx, cy, w, h)
                        gt_box = gt_boxes[gt_idx]
                        anchor = anchors[idx]
                        
                        reg_target[b, 0, y, x] = (gt_box[0] - anchor[0]) / stride
                        reg_target[b, 1, y, x] = (gt_box[1] - anchor[1]) / stride
                        reg_target[b, 2, y, x] = torch.log(gt_box[2] - gt_box[0])
                        reg_target[b, 3, y, x] = torch.log(gt_box[3] - gt_box[1])
            
            cls_targets.append(cls_target)
            reg_targets.append(reg_target)
            pos_masks.append(pos_mask)
            # 关键点目标 - 使用正确的形状 (B, num_kpt, H, W)
            kpt_targets.append(torch.zeros((batch_size, 10, h, w), device=device))  # 10 个关键点坐标
        
        return cls_targets, reg_targets, kpt_targets, pos_masks
    
    def _compute_iou(
        self,
        boxes1: torch.Tensor,
        boxes2: torch.Tensor,
    ) -> torch.Tensor:
        """计算 IoU 矩阵"""
        # boxes1: (N, 4), boxes2: (M, 4)
        # 确保在同一设备上
        if boxes1.device != boxes2.device:
            boxes2 = boxes2.to(boxes1.device)
        
        N = len(boxes1)
        M = len(boxes2)
        
        # 扩展维度
        boxes1 = boxes1.unsqueeze(1).expand(N, M, 4)
        boxes2 = boxes2.unsqueeze(0).expand(N, M, 4)
        
        # 交集
        inter_min = torch.max(boxes1[..., :2], boxes2[..., :2])
        inter_max = torch.min(boxes1[..., 2:], boxes2[..., 2:])
        inter_wh = (inter_max - inter_min).clamp(0)
        inter_area = inter_wh[..., 0] * inter_wh[..., 1]
        
        # 面积
        area1 = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
        area2 = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])
        
        # IoU
        union = area1 + area2 - inter_area
        iou = inter_area / union.clamp(min=1e-7)
        
        return iou


# ============================================
# 完整检测损失
# ============================================

class DetectionLoss(nn.Module):
    """
    完整检测损失
    
    Loss = cls_weight * FocalLoss + reg_weight * CIoULoss + kpt_weight * WingLoss
    """
    
    def __init__(
        self,
        num_classes: int = 1,
        cls_weight: float = 1.0,
        reg_weight: float = 2.0,
        kpt_weight: float = 1.5,
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.25,
        wing_w: float = 10.0,
        wing_epsilon: float = 2.0,
        strides: List[int] = [8, 16, 32],
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.cls_weight = cls_weight
        self.reg_weight = reg_weight
        self.kpt_weight = kpt_weight
        self.strides = strides
        
        # 损失函数
        self.cls_loss_fn = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.reg_loss_fn = CIoULoss()
        self.kpt_loss_fn = WingLoss(w=wing_w, epsilon=wing_epsilon)
        
        # 标签分配器
        self.label_assigner = LabelAssigner(
            num_classes=num_classes,
            strides=strides,
        )
    
    def forward(
        self,
        predictions: Dict[str, List[torch.Tensor]],
        targets: List[Dict],
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            predictions: 模型输出 {cls_preds, reg_preds, kpt_preds}
            targets: 目标标注 [{boxes, labels}, ...]
        
        Returns:
            losses: 损失字典
        """
        device = predictions['cls_preds'][0].device
        
        # 分配标签
        cls_targets, reg_targets, kpt_targets, pos_masks = \
            self.label_assigner.assign(
                predictions['cls_preds'],
                predictions['reg_preds'],
                targets,
                device,
            )
        
        # 计算损失
        loss_cls = 0
        loss_reg = 0
        loss_kpt = 0
        num_pos = 0
        
        num_levels = len(predictions['cls_preds'])
        
        for level in range(num_levels):
            cls_pred = predictions['cls_preds'][level]
            reg_pred = predictions['reg_preds'][level]
            kpt_pred = predictions['kpt_preds'][level] if 'kpt_preds' in predictions else None
            
            cls_tgt = cls_targets[level]
            reg_tgt = reg_targets[level]
            pos_mask = pos_masks[level]
            
            # 分类损失
            loss_cls += self.cls_loss_fn(cls_pred, cls_tgt)
            
            # 回归损失 (仅正样本)
            if pos_mask.sum() > 0:
                # pos_mask: (B, H, W), reg_pred: (B, 4, H, W)
                # 扩展 pos_mask 到 (B, 4, H, W)
                pos_mask_4d = pos_mask.unsqueeze(1).expand_as(reg_pred)
                reg_pred_pos = reg_pred[pos_mask_4d].reshape(-1, 4)
                reg_tgt_pos = reg_tgt[pos_mask_4d].reshape(-1, 4)
                loss_reg += self.reg_loss_fn(reg_pred_pos, reg_tgt_pos)
                num_pos += pos_mask.sum()
                
                # 关键点损失
                if kpt_pred is not None:
                    kpt_tgt = kpt_targets[level]
                    # 为关键点创建正确的 mask (B, num_kpt, H, W)
                    pos_mask_kpt = pos_mask.unsqueeze(1).expand_as(kpt_pred)
                    kpt_pred_pos = kpt_pred[pos_mask_kpt].reshape(-1, kpt_pred.shape[1])
                    kpt_tgt_pos = kpt_tgt[pos_mask_kpt].reshape(-1, kpt_pred.shape[1])
                    loss_kpt += self.kpt_loss_fn(kpt_pred_pos, kpt_tgt_pos)
        
        # 归一化
        if num_pos > 0:
            loss_cls = loss_cls / num_pos
            loss_reg = loss_reg / num_pos
            loss_kpt = loss_kpt / num_pos if loss_kpt > 0 else loss_kpt
        
        # 加权
        total_loss = (
            self.cls_weight * loss_cls +
            self.reg_weight * loss_reg +
            self.kpt_weight * loss_kpt
        )
        
        return {
            'total_loss': total_loss,
            'loss_cls': loss_cls * self.cls_weight,
            'loss_reg': loss_reg * self.reg_weight,
            'loss_kpt': loss_kpt * self.kpt_weight,
            'num_pos': torch.tensor(num_pos, device=device),
        }


# ============================================
# 工厂函数
# ============================================

def build_loss(
    loss_type: str = 'detection',
    **kwargs
) -> nn.Module:
    """构建损失函数"""
    if loss_type == 'detection':
        return DetectionLoss(**kwargs)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


if __name__ == "__main__":
    # 测试损失函数
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 模拟模型输出
    batch_size = 2
    cls_preds = [
        torch.randn(batch_size, 1, 80, 80, device=device),
        torch.randn(batch_size, 1, 40, 40, device=device),
        torch.randn(batch_size, 1, 20, 20, device=device),
    ]
    reg_preds = [
        torch.randn(batch_size, 4, 80, 80, device=device),
        torch.randn(batch_size, 4, 40, 40, device=device),
        torch.randn(batch_size, 4, 20, 20, device=device),
    ]
    kpt_preds = [
        torch.randn(batch_size, 10, 80, 80, device=device),
        torch.randn(batch_size, 10, 40, 40, device=device),
        torch.randn(batch_size, 10, 20, 20, device=device),
    ]
    
    predictions = {
        'cls_preds': cls_preds,
        'reg_preds': reg_preds,
        'kpt_preds': kpt_preds,
    }
    
    # 模拟目标
    targets = [
        {
            'boxes': torch.tensor([[50, 50, 150, 150], [200, 200, 300, 300]], device=device),
            'labels': torch.tensor([1, 1], device=device),
        },
        {
            'boxes': torch.tensor([[100, 100, 200, 200]], device=device),
            'labels': torch.tensor([1], device=device),
        },
    ]
    
    # 计算损失
    loss_fn = DetectionLoss()
    losses = loss_fn(predictions, targets)
    
    print("Loss test:")
    for k, v in losses.items():
        print(f"  {k}: {v.item() if isinstance(v, torch.Tensor) else v}")
