"""
DKGA-Det 检测模型 - 检测头

解耦头设计：分类、回归、关键点分支独立
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional

from models.common.backbone_utils import ConvBNAct


# ============================================
# 解耦头 (Decoupled Head)
# ============================================

class DecoupledHead(nn.Module):
    """
    解耦检测头
    
    分类、回归、关键点三个分支独立处理
    """
    
    def __init__(
        self,
        in_channels: int = 256,
        num_classes: int = 1,
        num_keypoints: int = 5,
        channels: int = 256,
        num_cls_convs: int = 2,
        num_reg_convs: int = 2,
        use_gn: bool = True,
        prior_prob: float = 0.01,
        activation: str = "SiLU"
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.num_keypoints = num_keypoints
        
        # 分类分支
        self.cls_convs = nn.Sequential(*[
            ConvBNAct(in_channels, channels, 3, 1, 1, activation=activation, use_gn=use_gn)
            for _ in range(num_cls_convs)
        ])
        
        # 回归分支
        self.reg_convs = nn.Sequential(*[
            ConvBNAct(in_channels, channels, 3, 1, 1, activation=activation, use_gn=use_gn)
            for _ in range(num_reg_convs)
        ])
        
        # 关键点分支
        self.kpt_convs = nn.Sequential(*[
            ConvBNAct(in_channels, channels, 3, 1, 1, activation=activation, use_gn=use_gn)
            for _ in range(num_cls_convs)
        ])
        
        # 分类输出
        self.cls_pred = nn.Conv2d(channels, num_classes, 3, padding=1)
        
        # 回归输出 (bbox: cx, cy, w, h)
        self.reg_pred = nn.Conv2d(channels, 4, 3, padding=1)
        
        # 关键点输出 (5 关键点 * 2 坐标)
        self.kpt_pred = nn.Conv2d(channels, num_keypoints * 2, 3, padding=1)
        
        self._init_weights(prior_prob)
    
    def _init_weights(self, prior_prob: float = 0.01) -> None:
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m in [self.cls_pred, self.reg_pred, self.kpt_pred]:
                    # 输出层特殊初始化
                    nn.init.normal_(m.weight, mean=0, std=0.01)
                    if m.bias is not None:
                        if m == self.cls_pred:
                            # 分类层偏置初始化为 -log((1-prior_prob)/prior_prob)
                            nn.init.constant_(m.bias, -math.log((1 - prior_prob) / prior_prob))
                        else:
                            nn.init.constant_(m.bias, 0)
                else:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    
    def forward(
        self,
        features: List[torch.Tensor]
    ) -> Dict[str, List[torch.Tensor]]:
        """
        前向传播
        
        Args:
            features: 多尺度特征列表 [(B, C, H, W), ...]
            
        Returns:
            predictions: 字典包含 cls_preds, reg_preds, kpt_preds
        """
        cls_preds = []
        reg_preds = []
        kpt_preds = []
        
        for feat in features:
            # 分类分支
            cls_feat = self.cls_convs(feat)
            cls_pred = self.cls_pred(cls_feat)
            cls_preds.append(cls_pred)
            
            # 回归分支
            reg_feat = self.reg_convs(feat)
            reg_pred = self.reg_pred(reg_feat)
            reg_preds.append(reg_pred)
            
            # 关键点分支
            kpt_feat = self.kpt_convs(feat)
            kpt_pred = self.kpt_pred(kpt_feat)
            kpt_preds.append(kpt_pred)
        
        return {
            'cls_preds': cls_preds,
            'reg_preds': reg_preds,
            'kpt_preds': kpt_preds,
        }


# ============================================
# 分类头 (带 Focal Loss 优化)
# ============================================

class ClsHead(nn.Module):
    """分类头"""
    
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        channels: int = 256,
        num_convs: int = 4,
        use_gn: bool = True
    ):
        super().__init__()
        
        convs = []
        for _ in range(num_convs):
            if use_gn:
                convs.append(nn.Sequential(
                    nn.Conv2d(in_channels, channels, 3, padding=1, bias=False),
                    nn.GroupNorm(32, channels),
                    nn.SiLU(inplace=True)
                ))
            else:
                convs.append(ConvBNAct(in_channels, channels, 3, 1, 1))
            in_channels = channels
        
        self.convs = nn.Sequential(*convs)
        self.pred = nn.Conv2d(channels, num_classes, 3, padding=1)
        
        # 初始化
        prior_prob = 0.01
        nn.init.constant_(self.pred.bias, -math.log((1 - prior_prob) / prior_prob))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.convs(x)
        x = self.pred(x)
        return x


# ============================================
# 回归头
# ============================================

class RegHead(nn.Module):
    """回归头 (Bbox + IoU)"""
    
    def __init__(
        self,
        in_channels: int,
        num_anchors: int = 1,
        channels: int = 256,
        num_convs: int = 4,
        use_gn: bool = True,
        use_iou_branch: bool = True
    ):
        super().__init__()
        
        self.use_iou_branch = use_iou_branch
        
        convs = []
        for _ in range(num_convs):
            if use_gn:
                convs.append(nn.Sequential(
                    nn.Conv2d(in_channels, channels, 3, padding=1, bias=False),
                    nn.GroupNorm(32, channels),
                    nn.SiLU(inplace=True)
                ))
            else:
                convs.append(ConvBNAct(in_channels, channels, 3, 1, 1))
            in_channels = channels
        
        self.convs = nn.Sequential(*convs)
        
        # Bbox 回归
        self.bbox_pred = nn.Conv2d(channels, num_anchors * 4, 3, padding=1)
        
        # IoU 预测分支
        if use_iou_branch:
            self.iou_pred = nn.Conv2d(channels, num_anchors, 3, padding=1)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        x = self.convs(x)
        bbox = self.bbox_pred(x)
        iou = self.iou_pred(x) if self.use_iou_branch else None
        return bbox, iou


# ============================================
# 关键点头
# ============================================

class KptHead(nn.Module):
    """关键点检测头"""
    
    def __init__(
        self,
        in_channels: int,
        num_keypoints: int = 5,
        channels: int = 256,
        num_convs: int = 4,
        use_gn: bool = True,
        use_heatmap: bool = False
    ):
        super().__init__()
        
        self.use_heatmap = use_heatmap
        self.num_keypoints = num_keypoints
        
        convs = []
        for _ in range(num_convs):
            if use_gn:
                convs.append(nn.Sequential(
                    nn.Conv2d(in_channels, channels, 3, padding=1, bias=False),
                    nn.GroupNorm(32, channels),
                    nn.SiLU(inplace=True)
                ))
            else:
                convs.append(ConvBNAct(in_channels, channels, 3, 1, 1))
            in_channels = channels
        
        self.convs = nn.Sequential(*convs)
        
        if use_heatmap:
            # 热力图输出 (每个关键点一个通道)
            self.kpt_pred = nn.Conv2d(channels, num_keypoints, 3, padding=1)
        else:
            # 坐标回归输出
            self.kpt_pred = nn.Conv2d(channels, num_keypoints * 2, 3, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.convs(x)
        x = self.kpt_pred(x)
        return x


# ============================================
# 损失函数
# ============================================

class FocalLoss(nn.Module):
    """
    Focal Loss
    
    Reference: https://arxiv.org/abs/1708.02002
    """
    
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = 'sum'
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        # Sigmoid + BCE
        inputs = torch.sigmoid(inputs)
        
        # 计算交叉熵
        ce = F.binary_cross_entropy(inputs, targets, reduction='none')
        
        # 计算权重
        weight = self.alpha * targets * (1 - inputs) ** self.gamma + \
                 (1 - self.alpha) * (1 - targets) * inputs ** self.gamma
        
        loss = ce * weight
        
        if self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'mean':
            return loss.mean()
        return loss


class CIoULoss(nn.Module):
    """
    CIoU Loss (Complete IoU)
    
    Reference: https://arxiv.org/abs/2005.03572
    """
    
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
        eps: float = 1e-7
    ) -> torch.Tensor:
        # 转换为 xyxy 格式
        pred_xy = pred[..., :2]
        pred_wh = pred[..., 2:]
        target_xy = target[..., :2]
        target_wh = target[..., 2:]
        
        # 计算 IoU
        pred_x1y1 = pred_xy - pred_wh / 2
        pred_x2y2 = pred_xy + pred_wh / 2
        target_x1y1 = target_xy - target_wh / 2
        target_x2y2 = target_xy + target_wh / 2
        
        inter_x1 = torch.max(pred_x1y1[..., 0], target_x1y1[..., 0])
        inter_y1 = torch.max(pred_x1y1[..., 1], target_x1y1[..., 1])
        inter_x2 = torch.min(pred_x2y2[..., 0], target_x2y2[..., 0])
        inter_y2 = torch.min(pred_x2y2[..., 1], target_x2y2[..., 1])
        
        inter_area = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
        pred_area = pred_wh[..., 0] * pred_wh[..., 1]
        target_area = target_wh[..., 0] * target_wh[..., 1]
        union_area = pred_area + target_area - inter_area
        
        iou = inter_area / (union_area + eps)
        
        # 计算中心点距离
        center_pred = pred_xy
        center_target = target_xy
        center_dist = ((center_pred - center_target) ** 2).sum(dim=-1)
        
        # 计算对角线距离
        cw = torch.max(pred_x2y2[..., 0], target_x2y2[..., 0]) - \
             torch.min(pred_x1y1[..., 0], target_x1y1[..., 0])
        ch = torch.max(pred_x2y2[..., 1], target_x2y2[..., 1]) - \
             torch.min(pred_x1y1[..., 1], target_x1y1[..., 1])
        diag_dist = cw ** 2 + ch ** 2 + eps
        
        # 计算纵横比
        v = (4 / (math.pi ** 2)) * torch.pow(
            torch.atan(pred_wh[..., 0] / (pred_wh[..., 1] + eps)) -
            torch.atan(target_wh[..., 0] / (target_wh[..., 1] + eps)), 2
        )
        alpha = v / (1 - iou + v + eps)
        
        # CIoU
        ciou = iou - (center_dist / diag_dist) - alpha * v
        
        loss = 1 - ciou
        
        if weight is not None:
            loss = loss * weight
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class WingLoss(nn.Module):
    """
    Wing Loss (用于关键点检测)
    
    Reference: https://arxiv.org/abs/1711.06753
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
        
        self.C = self.w - self.w * math.log(1 + self.w / self.epsilon)
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        weight: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        diff = torch.abs(pred - target)
        
        loss = torch.where(
            diff < self.w,
            self.w * torch.log(1 + diff / self.epsilon),
            diff - self.C
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
    检测任务组合损失
    """
    
    def __init__(
        self,
        cls_weight: float = 1.0,
        reg_weight: float = 2.0,
        kpt_weight: float = 1.5,
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.25,
        wing_w: float = 10.0,
        wing_epsilon: float = 2.0
    ):
        super().__init__()
        
        self.cls_weight = cls_weight
        self.reg_weight = reg_weight
        self.kpt_weight = kpt_weight
        
        self.cls_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.reg_loss = CIoULoss()
        self.kpt_loss = WingLoss(w=wing_w, epsilon=wing_epsilon)
    
    def forward(
        self,
        cls_preds: List[torch.Tensor],
        reg_preds: List[torch.Tensor],
        kpt_preds: List[torch.Tensor],
        targets: Dict
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            cls_preds: 分类预测列表
            reg_preds: 回归预测列表
            kpt_preds: 关键点预测列表
            targets: 包含 cls_targets, reg_targets, kpt_targets
            
        Returns:
            losses: 各项损失字典
        """
        loss_cls = 0
        loss_reg = 0
        loss_kpt = 0
        num_levels = len(cls_preds)
        
        for i in range(num_levels):
            # 分类损失
            if 'cls_targets' in targets:
                loss_cls += self.cls_loss(cls_preds[i], targets['cls_targets'][i])
            
            # 回归损失
            if 'reg_targets' in targets:
                loss_reg += self.reg_loss(reg_preds[i], targets['reg_targets'][i])
            
            # 关键点损失
            if 'kpt_targets' in targets:
                loss_kpt += self.kpt_loss(kpt_preds[i], targets['kpt_targets'][i])
        
        losses = {
            'loss_cls': loss_cls * self.cls_weight / num_levels,
            'loss_reg': loss_reg * self.reg_weight / num_levels,
            'loss_kpt': loss_kpt * self.kpt_weight / num_levels,
        }
        losses['loss_total'] = sum(losses.values())
        
        return losses


# ============================================
# 工厂函数
# ============================================

def build_head(
    name: str = "decoupled",
    **kwargs
) -> nn.Module:
    """
    构建检测头
    
    Args:
        name: 头类型
        **kwargs: 配置参数
        
    Returns:
        检测头模块
    """
    heads = {
        'decoupled': DecoupledHead,
        'cls': ClsHead,
        'reg': RegHead,
        'kpt': KptHead,
    }
    
    if name not in heads:
        raise ValueError(f"Unknown head: {name}")
    
    return heads[name](**kwargs)


def build_loss(
    name: str = "detection",
    **kwargs
) -> nn.Module:
    """构建损失函数"""
    losses = {
        'detection': DetectionLoss,
        'focal': FocalLoss,
        'ciou': CIoULoss,
        'wing': WingLoss,
    }
    
    if name not in losses:
        raise ValueError(f"Unknown loss: {name}")
    
    return losses[name](**kwargs)


if __name__ == "__main__":
    # 测试
    features = [
        torch.randn(2, 256, 80, 80),
        torch.randn(2, 256, 40, 40),
        torch.randn(2, 256, 20, 20),
    ]
    
    head = DecoupledHead(in_channels=256, num_classes=1, num_keypoints=5)
    outputs = head(features)
    
    print("Output shapes:")
    for key, preds in outputs.items():
        print(f"  {key}: {[p.shape for p in preds]}")
