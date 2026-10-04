"""
Adaptive Polar Pillar Feature Encoder
Converts polar-transformed point clouds into structured polar pillar representations.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional


class PolarPillarEncoder(nn.Module):
    """
    Encodes points grouped into Polar Pillars into fixed-dimension feature vectors.
    Input point features: [r, theta, z, intensity, r_c, theta_c, z_c, x_p, y_p] (9D)
    """

    def __init__(
        self,
        in_channels: int = 9,
        out_channels: int = 64,
        use_norm: bool = True
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, out_channels // 2, bias=False),
            nn.BatchNorm1d(out_channels // 2) if use_norm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels // 2, out_channels, bias=False),
            nn.BatchNorm1d(out_channels) if use_norm else nn.Identity(),
            nn.ReLU(inplace=True)
        )

    def forward(self, pillar_features: torch.Tensor, point_counts: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pillar_features: (num_pillars, max_pts, in_channels)
            point_counts: (num_pillars,) actual points per pillar
        Returns:
            encoded_pillars: (num_pillars, out_channels)
        """
        num_pillars, max_pts, in_dim = pillar_features.shape
        flat_features = pillar_features.view(-1, in_dim)
        
        # Pass through MLP
        encoded = self.mlp(flat_features)
        encoded = encoded.view(num_pillars, max_pts, self.out_channels)
        
        # Max-pooling across points within each pillar
        pooled, _ = torch.max(encoded, dim=1)
        return pooled
