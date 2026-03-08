"""
DDFD-Rec 识别模型 - 损失函数

包含:
- AdaArc Loss (自适应边界 ArcFace)
- 正交约束损失
- 属性解耦损失
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


# ============================================
# AdaArc Loss (自适应边界 ArcFace)
# ============================================

class AdaArcLoss(nn.Module):
    """
    自适应边界 ArcFace Loss
    
    核心创新:
    1. 根据样本难度动态调整边界 m
    2. 硬样本获得更大的边界以增强判别性
    3. 难度系数和硬样本指示函数
    
    L = -log(exp(s * cos(theta_y + m_i)) / (exp(s * cos(theta_y + m_i)) + sum(exp(s * cos(theta_j)))))
    
    其中 m_i = m_base + alpha * (1 - cos_theta_y) + beta * I_hard
    """
    
    def __init__(
        self,
        num_classes: int,
        embedding_size: int = 512,
        scale: float = 32.0,
        m_base: float = 0.5,
        alpha: float = 0.1,
        beta: float = 0.15,
        hard_sample_ratio: float = 0.2,
        label_smooth: float = 0.0,
        easy_margin: bool = False
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.embedding_size = embedding_size
        self.scale = scale
        self.m_base = m_base
        self.alpha = alpha
        self.beta = beta
        self.hard_sample_ratio = hard_sample_ratio
        self.label_smooth = label_smooth
        self.easy_margin = easy_margin
        
        # 权重矩阵
        self.kernel = nn.Parameter(torch.Tensor(num_classes, embedding_size))
        nn.init.xavier_uniform_(self.kernel)
        
        # 预计算三角函数值
        self._cos_m = math.cos(m_base)
        self._sin_m = math.sin(m_base)
        
        # 用于 easy_margin
        if easy_margin:
            self._th = math.cos(math.pi - m_base)
            self._mm = math.sin(math.pi - m_base) * m_base
    
    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            features: (B, embedding_size) 归一化特征
            labels: (B,) 身份标签
            
        Returns:
            loss: 标量损失
        """
        B = features.shape[0]
        
        # 归一化
        features_norm = F.normalize(features, p=2, dim=1)
        kernel_norm = F.normalize(self.kernel, p=2, dim=1)
        
        # 计算余弦相似度
        cos_theta = F.linear(features_norm, kernel_norm)
        cos_theta = cos_theta.clamp(-1, 1)
        
        # 获取目标类别的余弦值
        labels_view = labels.view(-1, 1)
        cos_y = torch.gather(cos_theta, 1, labels_view)
        
        # 计算自适应边界
        difficulty = 1 - cos_y.detach()  # 难度 = 1 - 置信度
        hard_threshold = torch.quantile(difficulty, self.hard_sample_ratio)
        hard_mask = (difficulty >= hard_threshold).float()
        
        # m_i = m_base + alpha * difficulty + beta * hard_indicator
        m = self.m_base + self.alpha * difficulty + self.beta * hard_mask
        
        # 计算 cos(theta + m)
        sin_theta = torch.sqrt(1.0 - cos_theta.pow(2))
        cos_theta_m = cos_theta * self._cos_m - sin_theta * self._sin_m
        
        # 处理边界情况
        if self.easy_margin:
            cos_theta_m = torch.where(
                cos_theta > 0, cos_theta_m, cos_theta
            )
        else:
            # 更严格的边界处理
            cos_theta_m = cos_theta - m * self._sin_m - (1 - torch.cos(m * torch.pi)) * cos_theta
        
        # One-hot 编码
        if self.label_smooth > 0:
            # 标签平滑
            one_hot = torch.zeros_like(cos_theta).scatter(
                1, labels_view, 1 - self.label_smooth
            )
            one_hot += self.label_smooth / self.num_classes
        else:
            one_hot = F.one_hot(labels, num_classes=self.num_classes).float()
        
        # 应用 margin
        output = cos_theta * (1 - one_hot) + cos_theta_m * one_hot
        
        # 缩放
        output *= self.scale
        
        # 交叉熵损失
        if self.label_smooth > 0:
            # 带标签平滑的交叉熵
            log_probs = F.log_softmax(output, dim=1)
            loss = -(one_hot * log_probs).sum(dim=1).mean()
        else:
            loss = F.cross_entropy(output, labels)
        
        return loss


# ============================================
# 正交约束损失
# ============================================

class OrthogonalLoss(nn.Module):
    """
    正交约束损失
    
    约束身份特征和属性特征正交，实现解耦
    """
    
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
    
    def forward(
        self,
        id_feat: torch.Tensor,
        attr_feat: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            id_feat: (B, id_dim) 身份特征
            attr_feat: (B, attr_dim) 属性特征
            
        Returns:
            loss: 正交约束损失
        """
        # 计算余弦相似度
        cos_sim = F.cosine_similarity(id_feat, attr_feat, dim=1)
        
        # 最小化相似度 (使其接近 0)
        loss = cos_sim.pow(2)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


# ============================================
# 属性解耦损失
# ============================================

class AttributeDisentanglementLoss(nn.Module):
    """
    属性解耦损失
    
    鼓励属性特征包含属性信息，身份特征不包含属性信息
    """
    
    def __init__(
        self,
        num_attributes: int = 5,
        attr_weights: Optional[List[float]] = None
    ):
        super().__init__()
        
        self.num_attributes = num_attributes
        self.attr_weights = attr_weights or [1.0] * num_attributes
    
    def forward(
        self,
        attr_logits: List[torch.Tensor],
        attr_targets: List[torch.Tensor],
        id_feat: Optional[torch.Tensor] = None,
        attr_feat: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            attr_logits: 属性分类 logits 列表
            attr_targets: 属性标签列表
            id_feat: 身份特征 (可选)
            attr_feat: 属性特征 (可选)
            
        Returns:
            losses: 各项损失字典
        """
        losses = {}
        total_loss = 0
        
        # 属性分类损失
        for i, (logits, targets) in enumerate(zip(attr_logits, attr_targets)):
            attr_loss = F.cross_entropy(logits, targets)
            losses[f'attr_loss_{i}'] = attr_loss
            total_loss += attr_loss * self.attr_weights[i]
        
        losses['attr_loss_total'] = total_loss
        
        # 身份特征不应包含属性信息 (对抗损失)
        if id_feat is not None and attr_feat is not None:
            # 计算互信息上界 (通过余弦相似度)
            mutual_info = F.cosine_similarity(id_feat, attr_feat, dim=1).abs().mean()
            losses['mutual_info'] = mutual_info
        
        return losses


# ============================================
# 中心损失 (Center Loss)
# ============================================

class CenterLoss(nn.Module):
    """
    Center Loss
    
    最小化类内距离，增强特征判别性
    """
    
    def __init__(
        self,
        num_classes: int,
        feature_dim: int,
        alpha: float = 0.5
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.alpha = alpha
        
        # 类别中心
        self.centers = nn.Parameter(torch.randn(num_classes, feature_dim))
    
    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            features: (B, feature_dim) 特征
            labels: (B,) 标签
            
        Returns:
            loss: center loss
        """
        B = features.shape[0]
        
        # 获取对应样本的中心
        centers_batch = self.centers[labels]
        
        # 计算距离
        diff = features - centers_batch
        loss = 0.5 * diff.pow(2).sum(dim=1).mean()
        
        # 更新中心 (动量更新)
        with torch.no_grad():
            ones = torch.ones(B, device=features.device)
            ones_sum = torch.bincount(labels, weights=ones, minlength=self.num_classes)
            ones_sum = ones_sum.clamp(min=1).view(-1, 1)
            
            # 计算每个类别的特征和
            feat_sum = torch.zeros_like(self.centers)
            feat_sum.scatter_add_(0, labels.unsqueeze(1).expand(-1, self.feature_dim), features)
            
            # 更新中心
            center_update = feat_sum / ones_sum
            self.centers.data = (1 - self.alpha) * self.centers.data + self.alpha * center_update
        
        return loss


# ============================================
# 组合损失
# ============================================

class RecognitionLoss(nn.Module):
    """
    识别任务组合损失
    
    L = L_adaarc + lambda_ortho * L_ortho + lambda_attr * L_attr
    """
    
    def __init__(
        self,
        num_classes: int,
        embedding_size: int = 512,
        scale: float = 32.0,
        m_base: float = 0.5,
        alpha: float = 0.1,
        beta: float = 0.15,
        ortho_weight: float = 0.1,
        attr_weight: float = 0.5,
        center_weight: float = 0.0,
        label_smooth: float = 0.0
    ):
        super().__init__()
        
        self.ortho_weight = ortho_weight
        self.attr_weight = attr_weight
        self.center_weight = center_weight
        
        # AdaArc Loss
        self.adaarc_loss = AdaArcLoss(
            num_classes=num_classes,
            embedding_size=embedding_size,
            scale=scale,
            m_base=m_base,
            alpha=alpha,
            beta=beta,
            label_smooth=label_smooth
        )
        
        # 正交约束损失
        self.ortho_loss = OrthogonalLoss()
        
        # 属性解耦损失
        self.attr_loss = AttributeDisentanglementLoss()
        
        # Center Loss (可选)
        if center_weight > 0:
            self.center_loss = CenterLoss(num_classes, embedding_size)
        else:
            self.center_loss = None
    
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        labels: torch.Tensor,
        attr_targets: Optional[List[torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            outputs: 模型输出字典
                - id_features: (B, id_dim) 身份特征
                - attr_features: (B, attr_dim) 属性特征
                - arcface_logits: (B, num_classes) ArcFace logits
                - attr_logits: List[(B, num_attr_cls)] 属性 logits
            labels: (B,) 身份标签
            attr_targets: 属性标签列表
            
        Returns:
            losses: 各项损失字典
        """
        losses = {}
        
        # AdaArc Loss (主损失)
        id_feat = outputs['id_features']
        loss_adaarc = self.adaarc_loss(id_feat, labels)
        losses['loss_adaarc'] = loss_adaarc
        
        # 正交约束损失
        attr_feat = outputs['attr_features']
        loss_ortho = self.ortho_loss(id_feat, attr_feat)
        losses['loss_ortho'] = loss_ortho * self.ortho_weight
        
        # 属性解耦损失
        if attr_targets is not None and 'attr_logits' in outputs:
            loss_attr_dict = self.attr_loss(
                outputs['attr_logits'], attr_targets,
                id_feat, attr_feat
            )
            losses['loss_attr'] = loss_attr_dict['attr_loss_total'] * self.attr_weight
            losses['loss_mutual_info'] = loss_attr_dict.get('mutual_info', torch.tensor(0.0))
        
        # Center Loss (可选)
        if self.center_loss is not None:
            loss_center = self.center_loss(id_feat, labels)
            losses['loss_center'] = loss_center * self.center_weight
        
        # 总损失
        losses['loss_total'] = sum(
            v for k, v in losses.items() if k.startswith('loss_') and k != 'loss_total'
        )
        
        return losses


# ============================================
# 工厂函数
# ============================================

def build_recognition_loss(
    loss_type: str = "recognition",
    **kwargs
) -> nn.Module:
    """
    构建识别损失
    
    Args:
        loss_type: 损失类型
        **kwargs: 配置参数
        
    Returns:
        损失函数
    """
    losses = {
        'recognition': RecognitionLoss,
        'adaarc': AdaArcLoss,
        'ortho': OrthogonalLoss,
        'center': CenterLoss,
    }
    
    if loss_type not in losses:
        raise ValueError(f"Unknown loss type: {loss_type}")
    
    return losses[loss_type](**kwargs)


if __name__ == "__main__":
    # 测试 AdaArc Loss
    adaarc = AdaArcLoss(num_classes=10000, embedding_size=512)
    features = torch.randn(32, 512)
    labels = torch.randint(0, 10000, (32,))
    
    loss = adaarc(features, labels)
    print(f"AdaArc Loss: {loss.item():.4f}")
    
    # 测试正交约束损失
    ortho = OrthogonalLoss()
    id_feat = torch.randn(32, 409)
    attr_feat = torch.randn(32, 103)
    
    loss_ortho = ortho(id_feat, attr_feat)
    print(f"Orthogonal Loss: {loss_ortho.item():.4f}")
    
    # 测试组合损失
    rec_loss = RecognitionLoss(num_classes=10000, embedding_size=512)
    
    outputs = {
        'id_features': id_feat,
        'attr_features': attr_feat,
        'arcface_logits': torch.randn(32, 10000),
        'attr_logits': [torch.randn(32, 10) for _ in range(5)],
    }
    
    losses = rec_loss(outputs, labels)
    print(f"\nTotal Loss: {losses['loss_total'].item():.4f}")
    for k, v in losses.items():
        print(f"  {k}: {v.item():.4f}")
