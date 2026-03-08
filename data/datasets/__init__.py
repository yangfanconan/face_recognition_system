"""
数据集包
"""

from .loader import (
    FaceDataset,
    WebFace12M,
    VGGFace2,
    CASIAWebFace,
    CustomFaceDataset,
    LFWDataset,
    build_dataloader,
)

__all__ = [
    "FaceDataset",
    "WebFace12M",
    "VGGFace2",
    "CASIAWebFace",
    "CustomFaceDataset",
    "LFWDataset",
    "build_dataloader",
]
