"""
检测模型包
"""

from .backbone import (
    ConvBNAct,
    Focus,
    Bottleneck,
    BottleneckCSP,
    DCNBottleneck,
    DCNCSPBlock,
    CSPDarknet,
    CSPDarknetTiny,
    build_backbone,
)

from .neck import (
    FPNBlock,
    PANetBlock,
    BiFPNBlock,
    BiFPNLite,
    SmallFaceFPN,
    build_neck,
)

from .head import (
    DecoupledHead,
    ClsHead,
    RegHead,
    KptHead,
    FocalLoss,
    CIoULoss,
    WingLoss,
    DetectionLoss,
    build_head,
    build_loss,
)

from .dkga_det import DKGA_Det, build_detector

from .losses import (
    FocalLoss,
    CIoULoss,
    WingLoss,
    SmoothL1Loss,
    DetectionLoss,
    LabelAssigner,
    build_loss_fn,
)

__all__ = [
    # Backbone
    "ConvBNAct",
    "Focus",
    "Bottleneck",
    "BottleneckCSP",
    "DCNBottleneck",
    "DCNCSPBlock",
    "CSPDarknet",
    "CSPDarknetTiny",
    "build_backbone",
    # Neck
    "FPNBlock",
    "PANetBlock",
    "BiFPNBlock",
    "BiFPNLite",
    "SmallFaceFPN",
    "build_neck",
    # Head
    "DecoupledHead",
    "ClsHead",
    "RegHead",
    "KptHead",
    "FocalLoss",
    "CIoULoss",
    "WingLoss",
    "DetectionLoss",
    "build_head",
    "build_loss",
    # Main model
    "DKGA_Det",
    "build_detector",
    # Losses
    "SmoothL1Loss",
    "LabelAssigner",
    "build_loss_fn",
]
