"""
Backbone 工具模块

包含通用的 Backbone 组件
"""

from typing import Optional

import torch.nn as nn


class ConvBNAct(nn.Module):
    """Conv + BN + Activation"""
    
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        groups=1,
        activation="SiLU",
        use_dcn=False,
        use_gn=False
    ):
        super().__init__()
        
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding,
            groups=groups, bias=False
        )
        
        if use_gn:
            self.bn = nn.GroupNorm(32, out_channels)
        else:
            self.bn = nn.BatchNorm2d(out_channels)
        
        if activation == "ReLU":
            self.act = nn.ReLU(inplace=True)
        elif activation == "SiLU":
            self.act = nn.SiLU(inplace=True)
        elif activation == "LeakyReLU":
            self.act = nn.LeakyReLU(0.1, inplace=True)
        elif activation == "GELU":
            self.act = nn.GELU()
        else:
            self.act = nn.Identity()
    
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))
