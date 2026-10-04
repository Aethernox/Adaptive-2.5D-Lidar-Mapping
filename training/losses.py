"""
Loss Functions for Semantic Segmentation:
- Distance-Aware Weighted Cross Entropy
- Focal Loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight: Optional[torch.Tensor] = None, ignore_index: int = 0):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits: (B, C, H, W)
        targets: (B, H, W)
        """
        ce_loss = F.cross_entropy(
            logits, targets,
            weight=self.weight,
            ignore_index=self.ignore_index,
            reduction='none'
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss
        
        valid_mask = (targets != self.ignore_index)
        if valid_mask.sum() > 0:
            return focal_loss[valid_mask].mean()
        return torch.tensor(0.0, device=logits.device, requires_grad=True)


class DistanceWeightedLoss(nn.Module):
    """
    Applies radial distance weight scaling to ensure far-range points receive adequate gradient.
    """

    def __init__(
        self,
        num_classes: int = 20,
        gamma: float = 2.0,
        class_weights: Optional[torch.Tensor] = None,
        ignore_index: int = 0
    ):
        super().__init__()
        self.focal = FocalLoss(gamma=gamma, weight=class_weights, ignore_index=ignore_index)
        self.ignore_index = ignore_index

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        range_map: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        loss = self.focal(logits, targets)
        return loss
