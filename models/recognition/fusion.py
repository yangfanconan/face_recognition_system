"""
DDFD-Rec 识别模型 - 融合模块与 Transformer 编码器

包含:
- FGA 频域门控注意力融合
- Transformer Encoder
- 特征金字塔融合
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
from einops import rearrange


# ============================================
# FGA 频域门控注意力融合
# ============================================

class FrequencyGatedFusion(nn.Module):
    """
    频域门控融合模块 (FGA)
    
    自适应融合空域和频域特征
    """
    
    def __init__(
        self,
        channels: int,
        reduction: int = 4,
        gate_type: str = "sigmoid"
    ):
        super().__init__()
        
        self.channels = channels
        self.gate_type = gate_type
        
        # 空域特征处理
        self.spatial_proj = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        
        # 频域特征处理
        self.freq_proj = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        
        # 门控生成器
        self.gate_generator = nn.Sequential(
            nn.Conv2d(channels * 2, channels // reduction, 1, bias=False),
            nn.BatchNorm2d(channels // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels * 2, 1, bias=False)
        )
        
        # 激活函数
        if gate_type == "sigmoid":
            self.gate_act = nn.Sigmoid()
        elif gate_type == "softmax":
            self.gate_act = nn.Softmax(dim=1)
        else:
            self.gate_act = nn.Sigmoid()
        
        self._init_weights()
    
    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(
        self,
        x_spatial: torch.Tensor,
        x_freq: torch.Tensor
    ) -> torch.Tensor:
        """
        融合空域和频域特征
        
        Args:
            x_spatial: (B, C, H, W) 空域特征
            x_freq: (B, C, H, W) 频域特征
            
        Returns:
            x_fused: (B, C, H, W) 融合特征
        """
        B, C, H, W = x_spatial.shape
        
        # 投影
        x_spatial_proj = self.spatial_proj(x_spatial)
        x_freq_proj = self.freq_proj(x_freq)
        
        # 生成门控
        concat = torch.cat([x_spatial_proj, x_freq_proj], dim=1)
        gates = self.gate_generator(concat)
        
        # 分割门控
        if self.gate_type == "softmax":
            gates = gates.view(B, 2, C, H, W)
            gates = self.gate_act(gates)
            gate_spatial = gates[:, 0]
            gate_freq = gates[:, 1]
        else:
            gates = self.gate_act(gates)
            gate_spatial = gates[:, :C]
            gate_freq = gates[:, C:]
        
        # 加权融合
        x_fused = gate_spatial * x_spatial_proj + gate_freq * x_freq_proj
        
        return x_fused


# ============================================
# 多头自注意力
# ============================================

class MultiHeadAttention(nn.Module):
    """
    多头自注意力机制
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0
    ):
        super().__init__()
        
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # QKV 投影
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: (B, N, C) 输入序列
            mask: (N,) 或 (B, N) 可选掩码
            
        Returns:
            out: (B, N, C) 输出
        """
        B, N, C = x.shape
        
        # 计算 QKV
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # 计算注意力
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, heads, N, N)
        
        if mask is not None:
            if mask.dim() == 1:
                mask = mask.unsqueeze(0).unsqueeze(1)
            elif mask.dim() == 2:
                mask = mask.unsqueeze(1)
            attn = attn.masked_fill(mask == 0, float('-inf'))
        
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # 加权求和
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        
        return out


# ============================================
# Feed-Forward Network
# ============================================

class Mlp(nn.Module):
    """
    MLP 前馈网络
    """
    
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: nn.Module = nn.GELU,
        drop: float = 0.0
    ):
        super().__init__()
        
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# ============================================
# Transformer Block
# ============================================

class TransformerBlock(nn.Module):
    """
    Transformer 编码器块
    
    Pre-Norm 架构:
    x = x + Attention(LN(x))
    x = x + MLP(LN(x))
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.LayerNorm
    ):
        super().__init__()
        
        self.norm1 = norm_layer(dim)
        self.attn = MultiHeadAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop
        )
        
        # DropPath
        self.drop_path = nn.Dropout(drop_path) if drop_path > 0 else nn.Identity()
        
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim, hidden_features=mlp_hidden_dim,
            act_layer=act_layer, drop=drop
        )
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # 自注意力 + 残差
        x = x + self.drop_path(self.attn(self.norm1(x), mask))
        
        # MLP + 残差
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        return x


# ============================================
# Transformer Encoder
# ============================================

class TransformerEncoder(nn.Module):
    """
    Transformer 编码器
    
    用于全局特征建模
    """
    
    def __init__(
        self,
        dim: int = 256,
        depth: int = 4,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        use_cls_token: bool = False,
        num_classes: int = 0
    ):
        super().__init__()
        
        self.dim = dim
        self.depth = depth
        self.use_cls_token = use_cls_token
        
        # CLS token (可选)
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        else:
            self.cls_token = None
        
        # 位置编码
        self.pos_embed = nn.Parameter(torch.zeros(1, 197 if use_cls_token else 196, dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        # 随机深度衰减率
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[i], norm_layer=norm_layer
            )
            for i in range(depth)
        ])
        
        self.norm = norm_layer(dim)
        
        # 分类头 (可选)
        if num_classes > 0:
            self.head = nn.Linear(dim, num_classes)
        else:
            self.head = nn.Identity()
        
        # 初始化
        self._init_weights()
    
    def _init_weights(self) -> None:
        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.pos_embed, std=0.02)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    @torch.jit.ignore
    def no_weight_decay(self) -> set:
        return {'pos_embed', 'cls_token'}
    
    def forward(
        self,
        x: torch.Tensor,
        return_cls: bool = False
    ) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) 或 (B, N, C) 输入特征
            return_cls: 是否返回 CLS token
            
        Returns:
            features: 全局特征
        """
        # 处理输入
        if x.dim() == 4:
            B, C, H, W = x.shape
            x = x.flatten(2).transpose(1, 2)  # (B, H*W, C)
        
        B, N, C = x.shape
        
        # 添加 CLS token
        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)
        
        # 位置编码
        if x.shape[1] != self.pos_embed.shape[1]:
            # 插值位置编码
            pos_embed = self.pos_embed[:, 1:, :] if self.use_cls_token else self.pos_embed
            num_patches = int(math.sqrt(x.shape[1] - 1 if self.use_cls_token else x.shape[1]))
            orig_size = int(math.sqrt(pos_embed.shape[1]))
            pos_embed = pos_embed.reshape(1, orig_size, orig_size, -1).permute(0, 3, 1, 2)
            pos_embed = F.interpolate(pos_embed, size=(num_patches, num_patches), mode='bilinear')
            pos_embed = pos_embed.permute(0, 2, 3, 1).flatten(1, 2)
            
            if self.use_cls_token:
                pos_embed = torch.cat([self.pos_embed[:, :1, :], pos_embed], dim=1)
        else:
            pos_embed = self.pos_embed
        
        x = self.pos_drop(x + pos_embed)
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        
        # 输出
        if self.use_cls_token and return_cls:
            return self.head(x[:, 0])  # CLS token
        elif self.use_cls_token:
            return x[:, 1:].mean(dim=1)  # 全局平均池化
        else:
            return x.mean(dim=1)  # 全局平均池化


# ============================================
# 特征金字塔融合
# ============================================

class FeaturePyramidFusion(nn.Module):
    """
    多尺度特征金字塔融合
    
    融合来自不同阶段的特征
    """
    
    def __init__(
        self,
        in_channels: List[int],
        out_channels: int = 256,
        fusion_type: str = "concat"
    ):
        super().__init__()
        
        self.fusion_type = fusion_type
        
        if fusion_type == "concat":
            # 拼接融合
            total_channels = sum(in_channels)
            self.fusion = nn.Sequential(
                nn.Conv2d(total_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        elif fusion_type == "add":
            # 相加融合
            self.projections = nn.ModuleList([
                nn.Conv2d(ch, out_channels, 1, bias=False)
                for ch in in_channels
            ])
            self.fusion = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")
    
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            features: 多尺度特征列表 [(B, C1, H1, W1), ...]
            
        Returns:
            fused: 融合后的特征 (B, out_channels, H, W)
        """
        if self.fusion_type == "concat":
            # 上采样到相同尺寸后拼接
            target_size = features[0].shape[2:]
            features_resized = [
                F.interpolate(f, size=target_size, mode='bilinear', align_corners=False)
                for f in features
            ]
            concat = torch.cat(features_resized, dim=1)
            return self.fusion(concat)
        else:
            # 上采样后相加
            target_size = features[0].shape[2:]
            features_resized = [
                F.interpolate(proj(f), size=target_size, mode='bilinear', align_corners=False)
                for proj, f in zip(self.projections, features)
            ]
            added = sum(features_resized)
            return self.fusion(added)


# ============================================
# 工厂函数
# ============================================

def build_fusion_module(
    fusion_type: str = "fga",
    **kwargs
) -> nn.Module:
    """
    构建融合模块
    
    Args:
        fusion_type: 融合类型
        **kwargs: 配置参数
        
    Returns:
        融合模块
    """
    if fusion_type == "fga":
        return FrequencyGatedFusion(**kwargs)
    elif fusion_type == "pyramid":
        return FeaturePyramidFusion(**kwargs)
    else:
        raise ValueError(f"Unknown fusion type: {fusion_type}")


def build_transformer_encoder(
    **kwargs
) -> TransformerEncoder:
    """构建 Transformer 编码器"""
    return TransformerEncoder(**kwargs)


if __name__ == "__main__":
    # 测试 FGA 融合
    fga = FrequencyGatedFusion(channels=256)
    x_spatial = torch.randn(2, 256, 28, 28)
    x_freq = torch.randn(2, 256, 28, 28)
    
    x_fused = fga(x_spatial, x_freq)
    print("FGA Fusion output:", x_fused.shape)
    
    # 测试 Transformer Encoder
    encoder = TransformerEncoder(
        dim=256, depth=4, num_heads=8,
        mlp_ratio=4.0, drop_path_rate=0.1
    )
    
    x = torch.randn(2, 256, 14, 14)
    out = encoder(x)
    print("Transformer output:", out.shape)
    
    # 统计参数量
    total_params = sum(p.numel() for p in encoder.parameters())
    print(f"Transformer parameters: {total_params / 1e6:.2f}M")
