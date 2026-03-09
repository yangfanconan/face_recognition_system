"""
人脸识别损失函数实现

包含:
- ArcFace Loss (Additive Angular Margin Loss)
- CosFace Loss (Large Margin Cosine Loss)
- AM-Softmax (Additive Margin Softmax)
- Combined Loss (组合损失)

参考:
- ArcFace: https://arxiv.org/abs/1801.07698
- CosFace: https://arxiv.org/abs/1801.09414
- AM-Softmax: https://arxiv.org/abs/1801.05595
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ArcFaceLoss(nn.Module):
    """
    ArcFace Loss 实现
    
    Additive Angular Margin Loss for Deep Face Recognition
    
    核心思想: 在角度空间添加加性边界，增强特征判别性
    
    Args:
        in_features: 输入特征维度 (如 512)
        out_features: 类别数 (身份数)
        margin: 角度边界 (默认 0.5 弧度 ≈ 28.6 度)
        scale: 缩放因子 (默认 30)
    
    Example:
        >>> criterion = ArcFaceLoss(512, 10000, margin=0.5, scale=30)
        >>> features = torch.randn(32, 512)  # batch of features
        >>> labels = torch.randint(0, 10000, (32,))  # identity labels
        >>> loss = criterion(features, labels)
    """
    
    def __init__(self, in_features=512, out_features=10000, 
                 margin=0.5, scale=30.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.margin = margin
        self.scale = scale
        
        # 权重矩阵 W (out_features x in_features)
        # 每一行代表一个身份的权重向量
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        
        # 初始化权重 (Xavier 初始化)
        nn.init.xavier_uniform_(self.weight)
        
        # 预计算 cos(m) 和 sin(m)
        # 用于加速 cos(θ + m) = cos(θ)cos(m) - sin(θ)sin(m) 的计算
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        
        # 确保 φ 在正确范围内的阈值
        # cos(π - m) = -cos(m)
        self.th = math.cos(math.pi - margin)
        
        # 边界值：sin(π - m) × m
        self.mm = math.sin(math.pi - margin) * margin
    
    def forward(self, features, labels):
        """
        Args:
            features: 归一化特征 (batch_size, in_features)
            labels: 身份标签 (batch_size,)
        
        Returns:
            loss: ArcFace loss 值 (标量)
        """
        # 1. L2 归一化特征和权重
        # 确保特征和权重都在单位球面上
        features_norm = F.normalize(features, p=2, dim=1)
        weight_norm = F.normalize(self.weight, p=2, dim=1)
        
        # 2. 计算余弦相似度 cos(θ) = X · W
        # cosine shape: (batch_size, out_features)
        cosine = F.linear(features_norm, weight_norm)
        
        # 3. 计算正弦 sin(θ) = √(1 - cos²(θ))
        # 使用 clamp 避免数值误差导致负数
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2).clamp(min=0, max=1))
        
        # 4. 计算 cos(θ + m) = cos(θ)cos(m) - sin(θ)sin(m)
        # 这是 ArcFace 的核心：在角度空间添加边界
        phi = cosine * self.cos_m - sine * self.sin_m
        
        # 5. 确保 φ 在正确范围内（避免 θ+m > π）
        # 当 cos(θ) < cos(π-m) 时，使用线性插值
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        
        # 6. 创建 one-hot 标签
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        
        # 7. 只在真实类别上应用 margin
        # 其他类别保持原余弦值
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        
        # 8. 应用缩放因子并计算交叉熵损失
        output *= self.scale
        loss = F.cross_entropy(output, labels)
        
        return loss


class CosFaceLoss(nn.Module):
    """
    CosFace Loss 实现
    
    Large Margin Cosine Loss for Face Recognition
    
    核心思想: 在余弦空间添加加性边界
    
    Args:
        in_features: 输入特征维度
        out_features: 类别数
        margin: 余弦边界 (默认 0.4)
        scale: 缩放因子 (默认 30)
    """
    
    def __init__(self, in_features=512, out_features=10000, 
                 margin=0.4, scale=30.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.margin = margin
        self.scale = scale
        
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
    
    def forward(self, features, labels):
        # L2 归一化
        features_norm = F.normalize(features, p=2, dim=1)
        weight_norm = F.normalize(self.weight, p=2, dim=1)
        
        # 余弦相似度
        cosine = F.linear(features_norm, weight_norm)
        
        # 应用余弦边界 (直接减去 margin)
        phi = cosine - self.margin
        
        # one-hot 标签
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        
        # 只在真实类别上应用 margin
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        
        # 缩放并计算损失
        output *= self.scale
        loss = F.cross_entropy(output, labels)
        
        return loss


class AMSoftmaxLoss(nn.Module):
    """
    AM-Softmax (Additive Margin Softmax) 实现
    
    与 ArcFace 类似，但使用不同的边界应用方式
    
    Args:
        in_features: 输入特征维度
        out_features: 类别数
        margin: 边界 (默认 0.3)
        scale: 缩放因子 (默认 30)
    """
    
    def __init__(self, in_features=512, out_features=10000, 
                 margin=0.3, scale=30.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.margin = margin
        self.scale = scale
        
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
    
    def forward(self, features, labels):
        # 归一化
        features_norm = F.normalize(features, p=2, dim=1)
        weight_norm = F.normalize(self.weight, p=2, dim=1)
        
        # 余弦相似度
        cosine = F.linear(features_norm, weight_norm)
        
        # 应用边界
        phi = cosine - self.margin
        
        # one-hot
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        
        # 应用 margin
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        
        # 缩放和损失
        output *= self.scale
        loss = F.cross_entropy(output, labels)
        
        return loss


class CombinedLoss(nn.Module):
    """
    组合损失：ArcFace + CrossEntropy
    
    用于平滑过渡到新损失函数
    
    Args:
        in_features: 输入特征维度
        out_features: 类别数
        arcface_margin: ArcFace margin
        arcface_scale: ArcFace scale
        ce_weight: CrossEntropy 权重 (默认 0.1)
    """
    
    def __init__(self, in_features, out_features, 
                 arcface_margin=0.5, arcface_scale=30,
                 ce_weight=0.1):
        super().__init__()
        self.arcface = ArcFaceLoss(in_features, out_features, 
                                   arcface_margin, arcface_scale)
        self.ce_weight = ce_weight
        self.ce = nn.CrossEntropyLoss()
    
    def forward(self, features, logits, labels):
        """
        Args:
            features: 特征向量 (归一化后)
            logits: 原始 logits（用于 CE loss）
            labels: 标签
        """
        arcface_loss = self.arcface(features, labels)
        ce_loss = self.ce(logits, labels)
        
        return arcface_loss + self.ce_weight * ce_loss


class FocalLoss(nn.Module):
    """
    Focal Loss 实现
    
    解决类别不均衡问题
    
    Args:
        gamma: 聚焦参数 (默认 2.0)
        alpha: 类别权重 (默认 0.25)
    """
    
    def __init__(self, gamma=2.0, alpha=0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


# 损失函数工厂
def get_loss_function(name, **kwargs):
    """
    获取损失函数的工厂函数
    
    Args:
        name: 损失函数名称 ('arcface', 'cosface', 'amsoftmax', 'focal', 'crossentropy')
        **kwargs: 损失函数参数
    
    Returns:
        损失函数实例
    
    Example:
        >>> criterion = get_loss_function('arcface', in_features=512, out_features=10000)
    """
    loss_functions = {
        'arcface': ArcFaceLoss,
        'cosface': CosFaceLoss,
        'amsoftmax': AMSoftmaxLoss,
        'focal': FocalLoss,
        'crossentropy': nn.CrossEntropyLoss,
    }
    
    if name not in loss_functions:
        raise ValueError(f"未知损失函数：{name}")
    
    return loss_functions[name](**kwargs)


# 参数推荐配置
LOSS_CONFIGS = {
    'arcface': {
        'margin': 0.5,      # 角度边界（弧度）
        'scale': 30,        # 特征缩放
        'description': '推荐用于高精度人脸识别',
    },
    'cosface': {
        'margin': 0.4,      # 余弦边界
        'scale': 30,
        'description': '计算效率略高于 ArcFace',
    },
    'amsoftmax': {
        'margin': 0.3,      # 边界
        'scale': 30,
        'description': '平滑的边界应用',
    },
    'focal': {
        'gamma': 2.0,       # 聚焦参数
        'alpha': 0.25,      # 类别权重
        'description': '适用于类别不均衡场景',
    },
}
