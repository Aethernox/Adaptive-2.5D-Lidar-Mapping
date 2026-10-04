"""
AdaptivePolarNet: End-to-End LiDAR Semantic Segmentation Network
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
import numpy as np

from models.polar_encoder import PolarPillarEncoder
from models.backbone import PolarBackbone


class AdaptivePolarNet(nn.Module):
    """
    End-to-End Polar LiDAR Segmentation Model.
    Architecture:
    Polar Feature Generator -> Polar Pillar MLP -> Polar 2D Conv Backbone -> Segmentation Head
    """

    def __init__(
        self,
        num_classes: int = 20,
        in_channels: int = 9,
        encoder_channels: int = 64,
        backbone_channels: list = [64, 128, 256],
        upsample_channels: list = [128, 128, 128],
        dropout: float = 0.1
    ):
        super().__init__()
        self.num_classes = num_classes
        
        # 1. Polar Pillar Encoder
        self.encoder = PolarPillarEncoder(
            in_channels=in_channels,
            out_channels=encoder_channels
        )
        
        # 2. Polar 2D Backbone
        self.backbone = PolarBackbone(
            in_channels=encoder_channels,
            layer_channels=backbone_channels,
            upsample_channels=upsample_channels
        )
        
        # 3. Segmentation Head
        feature_dim = self.backbone.out_channels
        self.head = nn.Sequential(
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(feature_dim, feature_dim // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim // 2, num_classes, kernel_size=1)
        )

    def forward(
        self,
        pseudo_image: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            pseudo_image: (B, encoder_channels, H_rings, W_sectors)
        Returns:
            logits: (B, num_classes, H_rings, W_sectors)
        """
        features = self.backbone(pseudo_image)
        logits = self.head(features)
        return logits
