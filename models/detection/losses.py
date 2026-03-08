"""
DKGA-Det 检测模型 - 损失函数

包含:
- Focal Loss (分类)
- CIoU Loss (边界框回归)
- Wing Loss (关键点)
- 组合损失
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


# ============================================
# Focal Loss
# ============================================

class FocalLoss(nn.Module):
    """
    Focal Loss for dense object detection
    
    Reference: https://arxiv.org/abs/1708.02002
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = 'sum',
        pos_weight: Optional[torch.Tensor] = None
    ):
        """
        Args:
            alpha: 平衡正负样本权重
            gamma: 调节难易样本权重
            reduction: 'sum' | 'mean' | 'none'
            pos_weight: 正样本权重 tensor
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.pos_weight = pos_weight
    
    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        weight: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            inputs: (N, *) 原始 logits
            targets: (N, *) 0/1 标签
            weight: (N, *) 可选的逐像素权重
            
        Returns:
            loss: 标量或 (N, *) 损失
        """
        # Sigmoid
        p = torch.sigmoid(inputs)
        
        # 计算 BCE
        ce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none', pos_weight=self.pos_weight
        )
        
        # 计算权重 (1 - p)^gamma for positive, p^gamma for negative
        p_t = p * targets + (1 - p) * (1 - targets)
        weight = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        weight = weight * (1 - p_t) ** self.gamma
        
        # 应用权重
        loss = weight * ce_loss
        
        # 应用额外的逐像素权重
        if weight is not None:
            loss = loss * weight
        
        # Reduction
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
    
    Reference: https://arxiv.org/abs/2005.03572
    
    CIoU = IoU - (rho^2(b,b_gt)/c^2) - alpha * v
    """
    
    def __init__(self, reduction: str = 'mean', eps: float = 1e-7):
        super().__init__()
        self.reduction = reduction
        self.eps = eps
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        weight: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            pred: (N, 4) [cx, cy, w, h] 或 [x1, y1, x2, y2]
            target: (N, 4)
            weight: (N,) 可选权重
            
        Returns:
            loss: CIoU loss
        """
        # 确保是 xyxy 格式
        if pred.shape[-1] == 4:
            # 检查是否是中心点格式
            if pred[..., 2:].min() > 0:  # w, h 都是正数
                # 转换为 xyxy
                pred = torch.cat([
                    pred[..., :2] - pred[..., 2:] / 2,
                    pred[..., :2] + pred[..., 2:] / 2
                ], dim=-1)
                target = torch.cat([
                    target[..., :2] - target[..., 2:] / 2,
                    target[..., :2] + target[..., 2:] / 2
                ], dim=-1)
        
        # 计算 IoU
        inter = self._intersection(pred, target)
        union = self._union(pred, target)
        iou = inter / (union + self.eps)
        
        # 计算中心点距离
        center_pred = (pred[..., :2] + pred[..., 2:]) / 2
        center_target = (target[..., :2] + target[..., 2:]) / 2
        center_dist = ((center_pred - center_target) ** 2).sum(dim=-1)
        
        # 计算对角线距离
        cw = torch.max(pred[..., 2], target[..., 2]) - torch.min(pred[..., 0], target[..., 0])
        ch = torch.max(pred[..., 3], target[..., 3]) - torch.min(pred[..., 1], target[..., 1])
        diag_dist = cw ** 2 + ch ** 2 + self.eps
        
        # 计算纵横比
        w_pred = pred[..., 2] - pred[..., 0]
        h_pred = pred[..., 3] - pred[..., 1]
        w_target = target[..., 2] - target[..., 0]
        h_target = target[..., 3] - target[..., 1]
        
        v = (4 / (math.pi ** 2)) * torch.pow(
            torch.atan(w_target / (h_target + self.eps)) - 
            torch.atan(w_pred / (h_pred + self.eps)), 2
        )
        
        with torch.no_grad():
            alpha = v / (1 - iou + v + self.eps)
        
        # CIoU
        ciou = iou - (center_dist / diag_dist) - alpha * v
        loss = 1 - ciou
        
        # 应用权重
        if weight is not None:
            loss = loss * weight
        
        # Reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss
    
    def _intersection(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """计算交集面积"""
        inter_x1 = torch.max(pred[..., 0], target[..., 0])
        inter_y1 = torch.max(pred[..., 1], target[..., 1])
        inter_x2 = torch.min(pred[..., 2], target[..., 2])
        inter_y2 = torch.min(pred[..., 3], target[..., 3])
        
        inter_w = (inter_x2 - inter_x1).clamp(0)
        inter_h = (inter_y2 - inter_y1).clamp(0)
        
        return inter_w * inter_h
    
    def _union(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """计算并集面积"""
        pred_area = (pred[..., 2] - pred[..., 0]) * (pred[..., 3] - pred[..., 1])
        target_area = (target[..., 2] - target[..., 0]) * (target[..., 3] - target[..., 1])
        
        inter = self._intersection(pred, target)
        
        return pred_area + target_area - inter


# ============================================
# Wing Loss
# ============================================

class WingLoss(nn.Module):
    """
    Wing Loss for landmark detection
    
    Reference: https://arxiv.org/abs/1711.06753
    
    wing(x) = w * log(1 + |x|/epsilon)  if |x| < w
              |x| - C                    otherwise
    where C = w - w * log(1 + w/epsilon)
    """
    
    def __init__(
        self,
        w: float = 10.0,
        epsilon: float = 2.0,
        reduction: str = 'mean'
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
        weight: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            pred: (N, *) 预测值
            target: (N, *) 目标值
            weight: (N, *) 可选权重
            
        Returns:
            loss: Wing loss
        """
        diff = torch.abs(pred - target)
        
        loss = torch.where(
            diff < self.w,
            self.w * torch.log(1 + diff / self.epsilon),
            diff - self.C
        )
        
        # 应用权重
        if weight is not None:
            loss = loss * weight
        
        # Reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


# ============================================
# Smooth L1 Loss
# ============================================

class SmoothL1Loss(nn.Module):
    """Smooth L1 Loss (Huber Loss)"""
    
    def __init__(self, beta: float = 1.0, reduction: str = 'mean'):
        super().__init__()
        self.beta = beta
        self.reduction = reduction
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        weight: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        diff = torch.abs(pred - target)
        
        loss = torch.where(
            diff < self.beta,
            0.5 * diff ** 2 / self.beta,
            diff - 0.5 * self.beta
        )
        
        if weight is not None:
            loss = loss * weight
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


# ============================================
# 组合损失
# ============================================

class DetectionLoss(nn.Module):
    """
    DKGA-Det 检测任务组合损失
    
    L = cls_weight * L_cls + reg_weight * L_reg + kpt_weight * L_kpt
    """
    
    def __init__(
        self,
        cls_weight: float = 1.0,
        reg_weight: float = 2.0,
        kpt_weight: float = 1.5,
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.25,
        wing_w: float = 10.0,
        wing_epsilon: float = 2.0,
        iou_loss_type: str = 'ciou'
    ):
        super().__init__()
        
        self.cls_weight = cls_weight
        self.reg_weight = reg_weight
        self.kpt_weight = kpt_weight
        
        # 分类损失
        self.cls_loss = FocalLoss(
            alpha=focal_alpha,
            gamma=focal_gamma,
            reduction='sum'
        )
        
        # 回归损失
        if iou_loss_type == 'ciou':
            self.reg_loss = CIoULoss(reduction='mean')
        else:
            self.reg_loss = SmoothL1Loss(beta=1.0, reduction='mean')
        
        # 关键点损失
        self.kpt_loss = WingLoss(w=wing_w, epsilon=wing_epsilon, reduction='sum')
    
    def forward(
        self,
        predictions: Dict[str, List[torch.Tensor]],
        targets: Dict[str, List[torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            predictions: 模型预测 {cls_preds, reg_preds, kpt_preds}
            targets: 目标标注 {cls_targets, reg_targets, kpt_targets, pos_mask}
            
        Returns:
            losses: 各项损失字典
        """
        cls_preds = predictions['cls_preds']
        reg_preds = predictions['reg_preds']
        kpt_preds = predictions['kpt_preds']
        
        cls_targets = targets.get('cls_targets', cls_preds)
        reg_targets = targets.get('reg_targets', reg_preds)
        kpt_targets = targets.get('kpt_targets', kpt_preds)
        pos_mask = targets.get('pos_mask', None)
        
        # 计算各项损失
        loss_cls = 0
        loss_reg = 0
        loss_kpt = 0
        num_levels = len(cls_preds)
        num_pos = 0
        
        for i in range(num_levels):
            # 分类损失
            cls_loss = self.cls_loss(cls_preds[i], cls_targets[i])
            loss_cls += cls_loss
            
            # 回归损失 (仅正样本)
            if pos_mask is not None:
                mask = pos_mask[i].bool()
                if mask.sum() > 0:
                    reg_loss = self.reg_loss(reg_preds[i][mask], reg_targets[i][mask])
                    loss_reg += reg_loss
                    num_pos += mask.sum()
                    
                    # 关键点损失
                    kpt_loss = self.kpt_loss(kpt_preds[i][mask], kpt_targets[i][mask])
                    loss_kpt += kpt_loss
            else:
                loss_reg += self.reg_loss(reg_preds[i], reg_targets[i])
                loss_kpt += self.kpt_loss(kpt_preds[i], kpt_targets[i])
        
        # 归一化
        if num_pos > 0:
            loss_reg = loss_reg / num_pos
            loss_kpt = loss_kpt / num_pos
        
        # 加权
        loss_cls = loss_cls * self.cls_weight
        loss_reg = loss_reg * self.reg_weight
        loss_kpt = loss_kpt * self.kpt_weight
        
        # 总损失
        loss_total = loss_cls + loss_reg + loss_kpt
        
        return {
            'loss_cls': loss_cls,
            'loss_reg': loss_reg,
            'loss_kpt': loss_kpt,
            'loss_total': loss_total,
            'num_pos': torch.tensor(num_pos, device=loss_total.device),
        }


# ============================================
# 标签分配 (Label Assignment)
# ============================================

class LabelAssigner:
    """
    标签分配器
    
    将 ground truth 分配给 anchor-free 的预测点
    """
    
    def __init__(
        self,
        num_classes: int = 1,
        center_radius: float = 2.5,
        iou_threshold: float = 0.2
    ):
        self.num_classes = num_classes
        self.center_radius = center_radius
        self.iou_threshold = iou_threshold
    
    def assign(
        self,
        pred_bboxes: torch.Tensor,
        gt_bboxes: torch.Tensor,
        gt_labels: torch.Tensor,
        stride: int,
        device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            pred_bboxes: (N, 4) 预测框
            gt_bboxes: (M, 4) GT 框
            gt_labels: (M,) GT 标签
            stride: 当前层级步长
            device: 设备
            
        Returns:
            cls_target: (N, num_classes) 分类目标
            reg_target: (N, 4) 回归目标
            pos_mask: (N,) 正样本掩码
        """
        N = pred_bboxes.shape[0]
        M = gt_bboxes.shape[0]
        
        # 初始化
        cls_target = torch.zeros(N, self.num_classes, device=device)
        reg_target = torch.zeros(N, 4, device=device)
        pos_mask = torch.zeros(N, dtype=torch.bool, device=device)
        
        if M == 0:
            return cls_target, reg_target, pos_mask
        
        # 计算 IoU
        ious = self._compute_iou(pred_bboxes, gt_bboxes)
        
        # 对每个 GT 选择最佳匹配
        for j in range(M):
            # 找到 IoU 最大的预测框
            iou = ious[:, j]
            best_idx = iou.argmax()
            
            if iou[best_idx] < self.iou_threshold:
                continue
            
            # 分配标签
            pos_mask[best_idx] = True
            cls_target[best_idx, gt_labels[j]] = 1.0
            reg_target[best_idx] = gt_bboxes[j]
        
        return cls_target, reg_target, pos_mask
    
    def _compute_iou(
        self,
        boxes1: torch.Tensor,
        boxes2: torch.Tensor
    ) -> torch.Tensor:
        """计算 IoU 矩阵"""
        # boxes1: (N, 4), boxes2: (M, 4)
        N = boxes1.shape[0]
        M = boxes2.shape[0]
        
        # 扩展维度
        boxes1 = boxes1.unsqueeze(1).expand(N, M, 4)
        boxes2 = boxes2.unsqueeze(0).expand(N, M, 4)
        
        # 计算交集
        inter_min = torch.max(boxes1[..., :2], boxes2[..., :2])
        inter_max = torch.min(boxes1[..., 2:], boxes2[..., 2:])
        inter_wh = (inter_max - inter_min).clamp(0)
        inter_area = inter_wh[..., 0] * inter_wh[..., 1]
        
        # 计算面积
        area1 = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
        area2 = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])
        
        # 计算 IoU
        union = area1 + area2 - inter_area
        iou = inter_area / union.clamp(min=1e-7)
        
        return iou


# ============================================
# 工厂函数
# ============================================

def build_loss_fn(
    loss_type: str = 'detection',
    **kwargs
) -> nn.Module:
    """
    构建损失函数
    
    Args:
        loss_type: 损失类型
        **kwargs: 配置参数
        
    Returns:
        损失函数
    """
    losses = {
        'detection': DetectionLoss,
        'focal': FocalLoss,
        'ciou': CIoULoss,
        'wing': WingLoss,
        'smooth_l1': SmoothL1Loss,
    }
    
    if loss_type not in losses:
        raise ValueError(f"Unknown loss type: {loss_type}")
    
    return losses[loss_type](**kwargs)


if __name__ == "__main__":
    # 测试
    cls_loss = FocalLoss(alpha=0.25, gamma=2.0)
    reg_loss = CIoULoss()
    kpt_loss = WingLoss()
    
    # 分类损失测试
    cls_pred = torch.randn(32, 1, 80, 80)
    cls_target = torch.rand(32, 1, 80, 80)
    loss_cls = cls_loss(cls_pred, cls_target)
    print(f"Focal Loss: {loss_cls.item():.4f}")
    
    # 回归损失测试
    reg_pred = torch.randn(32, 4)
    reg_target = torch.rand(32, 4) * 100
    loss_reg = reg_loss(reg_pred, reg_target)
    print(f"CIoU Loss: {loss_reg.item():.4f}")
    
    # 关键点损失测试
    kpt_pred = torch.randn(32, 10)
    kpt_target = torch.rand(32, 10) * 100
    loss_kpt = kpt_loss(kpt_pred, kpt_target)
    print(f"Wing Loss: {loss_kpt.item():.4f}")
