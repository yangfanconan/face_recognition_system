"""
DKGA-Det 人脸检测模型

完整模型 = Backbone + Neck + Head
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional

from models.detection.backbone import CSPDarknet, build_backbone
from models.detection.neck import BiFPNLite, SmallFaceFPN, build_neck
from models.detection.head import DecoupledHead, build_head


# ============================================
# DKGA-Det 主模型
# ============================================

class DKGA_Det(nn.Module):
    """
    DKGA-Det 人脸检测模型
    
    特性:
    - CSPDarknet + DCNv2 主干
    - BiFPN-Lite 特征融合
    - P2 层小目标增强
    - 解耦检测头
    - 可微分关键点对齐
    """
    
    def __init__(
        self,
        # Backbone 参数
        backbone_name: str = "cspdarknet",
        backbone_kwargs: Optional[Dict] = None,
        
        # Neck 参数
        neck_name: str = "bifpn_lite",
        neck_kwargs: Optional[Dict] = None,
        
        # Head 参数
        head_name: str = "decoupled",
        head_kwargs: Optional[Dict] = None,
        
        # 检测配置
        num_classes: int = 1,
        num_keypoints: int = 5,
        
        # NMS 配置
        score_thresh: float = 0.25,
        nms_thresh: float = 0.45,
        max_detections: int = 300,
    ):
        super().__init__()
        
        # Backbone
        if backbone_kwargs is None:
            backbone_kwargs = {
                'depths': [3, 6, 6, 3],
                'channels': [64, 128, 256, 512, 1024],
                'use_dcnv2': True,
                'dcnv2_stages': [2, 3],
            }
        self.backbone = build_backbone(backbone_name, **backbone_kwargs)
        
        # Neck
        if neck_kwargs is None:
            neck_kwargs = {
                'in_channels': [256, 512, 1024],
                'out_channels': 256,
                'use_p2': True,
                'attention': True,
            }
        self.neck = build_neck(neck_name, **neck_kwargs)
        
        # Head
        if head_kwargs is None:
            head_kwargs = {
                'in_channels': 256,
                'num_classes': num_classes,
                'num_keypoints': num_keypoints,
                'channels': 256,
                'use_gn': True,
            }
        self.head = build_head(head_name, **head_kwargs)
        
        # NMS 配置
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.max_detections = max_detections
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self) -> None:
        """初始化模型权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        targets: Optional[Dict] = None
    ) -> Dict:
        """
        前向传播
        
        Args:
            x: 输入图像 (B, 3, H, W)
            targets: 训练时的目标标注
            
        Returns:
            outputs: 预测结果或损失
        """
        # Backbone: 提取多尺度特征
        features = self.backbone(x)  # (P3, P4, P5)
        
        # Neck: 特征融合
        fused_features = self.neck(features)  # (P2, P3, P4, P5)
        
        # Head: 预测
        predictions = self.head(fused_features)
        
        if self.training and targets is not None:
            # 训练模式：计算损失
            from models.detection.losses import DetectionLoss
            loss_fn = DetectionLoss()
            losses = loss_fn(predictions, targets)
            return losses
        else:
            # 推理模式：后处理
            detections = self.postprocess(predictions)
            return detections
    
    def postprocess(
        self,
        predictions: Dict[str, List[torch.Tensor]]
    ) -> List[Dict[str, torch.Tensor]]:
        """
        后处理：解码 + NMS
        
        Args:
            predictions: 模型原始输出
            
        Returns:
            detections: 检测结果列表
        """
        cls_preds = predictions['cls_preds']
        reg_preds = predictions['reg_preds']
        kpt_preds = predictions['kpt_preds']
        
        batch_size = cls_preds[0].shape[0]
        results = []
        
        for i in range(batch_size):
            # 收集所有层级的预测
            boxes_all = []
            scores_all = []
            keypoints_all = []
            
            for level in range(len(cls_preds)):
                # 分类得分
                cls_score = cls_preds[level][i].sigmoid()  # (num_classes, H, W)
                
                # 获取高置信度预测
                max_score, class_idx = cls_score.max(dim=0)  # (H, W)
                mask = max_score > self.score_thresh
                
                if mask.sum() == 0:
                    continue
                
                # 获取坐标
                ys, xs = torch.where(mask)
                
                # 解码 bbox
                reg_pred = reg_preds[level][i][:, mask]  # (4, num_valid)
                boxes = self.decode_boxes(reg_pred, xs, ys, level)
                
                # 获取分数
                scores = max_score[mask]
                
                # 解码关键点
                kpt_pred = kpt_preds[level][i][:, mask]  # (2*num_kpts, num_valid)
                keypoints = self.decode_keypoints(kpt_pred, xs, ys, level)
                
                boxes_all.append(boxes)
                scores_all.append(scores)
                keypoints_all.append(keypoints)
            
            if len(boxes_all) == 0:
                results.append({
                    'boxes': torch.empty((0, 4)),
                    'scores': torch.empty(0),
                    'labels': torch.empty(0, dtype=torch.long),
                    'keypoints': torch.empty((0, 5, 2)),
                })
                continue
            
            # 合并所有层级
            boxes_all = torch.cat(boxes_all, dim=0)
            scores_all = torch.cat(scores_all, dim=0)
            keypoints_all = torch.cat(keypoints_all, dim=0)
            
            # NMS
            keep = self.nms(boxes_all, scores_all, self.nms_thresh)
            keep = keep[:self.max_detections]
            
            results.append({
                'boxes': boxes_all[keep],
                'scores': scores_all[keep],
                'labels': torch.zeros_like(scores_all[keep], dtype=torch.long),
                'keypoints': keypoints_all[keep],
            })
        
        return results
    
    def decode_boxes(
        self,
        reg_pred: torch.Tensor,
        xs: torch.Tensor,
        ys: torch.Tensor,
        level: int
    ) -> torch.Tensor:
        """
        解码边界框
        
        Args:
            reg_pred: 回归预测 (4, N)
            xs, ys: 网格坐标
            level: 特征层级
            
        Returns:
            boxes: (N, 4) xyxy 格式
        """
        # 计算步长
        strides = [8, 16, 32, 64]  # 对应 P2, P3, P4, P5
        stride = strides[level] if level < len(strides) else 32
        
        # 计算中心点
        cx = (xs + 0.5) * stride
        cy = (ys + 0.5) * stride
        
        # 解码 (假设预测的是 offset)
        cx_decoded = cx + reg_pred[0] * stride
        cy_decoded = cy + reg_pred[1] * stride
        w_decoded = torch.exp(reg_pred[2]) * stride
        h_decoded = torch.exp(reg_pred[3]) * stride
        
        # 转换为 xyxy
        x1 = cx_decoded - w_decoded / 2
        y1 = cy_decoded - h_decoded / 2
        x2 = cx_decoded + w_decoded / 2
        y2 = cy_decoded + h_decoded / 2
        
        return torch.stack([x1, y1, x2, y2], dim=0).t()
    
    def decode_keypoints(
        self,
        kpt_pred: torch.Tensor,
        xs: torch.Tensor,
        ys: torch.Tensor,
        level: int
    ) -> torch.Tensor:
        """
        解码关键点
        
        Args:
            kpt_pred: 关键点预测 (2*num_kpts, N)
            xs, ys: 网格坐标
            level: 特征层级
            
        Returns:
            keypoints: (N, num_kpts, 2)
        """
        strides = [8, 16, 32, 64]
        stride = strides[level] if level < len(strides) else 32
        
        num_kpts = kpt_pred.shape[0] // 2
        
        # 计算中心点
        cx = (xs + 0.5) * stride
        cy = (ys + 0.5) * stride
        
        keypoints = []
        for i in range(num_kpts):
            kpt_x = cx + kpt_pred[i * 2] * stride
            kpt_y = cy + kpt_pred[i * 2 + 1] * stride
            keypoints.append(torch.stack([kpt_x, kpt_y], dim=1))
        
        return torch.stack(keypoints, dim=1)  # (N, num_kpts, 2)
    
    def nms(
        self,
        boxes: torch.Tensor,
        scores: torch.Tensor,
        iou_threshold: float
    ) -> torch.Tensor:
        """
        非极大值抑制
        
        Args:
            boxes: (N, 4) xyxy 格式
            scores: (N,)
            iou_threshold: IoU 阈值
            
        Returns:
            keep: 保留的索引
        """
        return torchvision.ops.nms(boxes, scores, iou_threshold)


# ============================================
# 模型工厂
# ============================================

def build_detector(
    model_name: str = "dkga_det",
    **kwargs
) -> DKGA_Det:
    """
    构建检测模型
    
    Args:
        model_name: 模型名称
        **kwargs: 配置参数
        
    Returns:
        检测模型
    """
    if model_name == "dkga_det":
        return DKGA_Det(**kwargs)
    elif model_name == "dkga_det_tiny":
        # 轻量级版本
        kwargs['backbone_kwargs'] = {
            'name': 'cspdarknet_tiny',
            'channels': [32, 64, 128, 256, 512],
        }
        return DKGA_Det(**kwargs)
    else:
        raise ValueError(f"Unknown model: {model_name}")


# 导入 torchvision
import torchvision

if __name__ == "__main__":
    # 测试
    model = DKGA_Det()
    model.eval()
    
    x = torch.randn(2, 3, 640, 640)
    
    with torch.no_grad():
        outputs = model(x)
    
    print("Input shape:", x.shape)
    for i, det in enumerate(outputs):
        print(f"Image {i}: {len(det['boxes'])} faces detected")
        if len(det['boxes']) > 0:
            print(f"  Boxes: {det['boxes'].shape}")
            print(f"  Scores: {det['scores'].shape}")
            print(f"  Keypoints: {det['keypoints'].shape}")
    
    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params / 1e6:.2f}M")
