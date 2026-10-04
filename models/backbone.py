"""
2D Polar Convolutional Backbone with Multi-Scale Feature Pyramid (FPN)
"""

import torch
import torch.nn as nn
from typing import List, Tuple


class ConvBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int, stride: int = 1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class PolarBackbone(nn.Module):
    """
    Multi-scale convolutional backbone processing polar pseudo-images.
    """

    def __init__(
        self,
        in_channels: int = 64,
        layer_channels: List[int] = [64, 128, 256],
        layer_strides: List[int] = [1, 2, 2],
        upsample_channels: List[int] = [128, 128, 128]
    ):
        super().__init__()
        
        # Encoder downsampling stages
        self.block1 = ConvBlock(in_channels, layer_channels[0], stride=layer_strides[0])
        self.block2 = ConvBlock(layer_channels[0], layer_channels[1], stride=layer_strides[1])
        self.block3 = ConvBlock(layer_channels[1], layer_channels[2], stride=layer_strides[2])

        # Decoder upsampling stages
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(layer_channels[0], upsample_channels[0], kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(upsample_channels[0]),
            nn.ReLU(inplace=True)
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(layer_channels[1], upsample_channels[1], kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(upsample_channels[1]),
            nn.ReLU(inplace=True)
        )
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(layer_channels[2], upsample_channels[2], kernel_size=4, stride=4, bias=False),
            nn.BatchNorm2d(upsample_channels[2]),
            nn.ReLU(inplace=True)
        )

        self.out_channels = sum(upsample_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input: (B, in_channels, num_rings, num_sectors)
        Output: (B, total_upsample_channels, num_rings, num_sectors)
        """
        c1 = self.block1(x)
        c2 = self.block2(c1)
        c3 = self.block3(c2)

        u1 = self.deconv1(c1)
        u2 = self.deconv2(c2)
        u3 = self.deconv3(c3)

        # Concatenate multi-scale representations
        out = torch.cat([u1, u2, u3], dim=1)
        return out
