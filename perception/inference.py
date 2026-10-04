"""
Perception Inference Engine
Authoritative implementation supporting:
- Mode 1: Ground Truth Labels
- Mode 2: Deep Learning Neural Perception (AdaptivePolarNet)
- Heuristic fallback if model weights are not found
"""

import os
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F

from core.schema import PointCloudFrame
from models.segmentation import AdaptivePolarNet
from perception.preprocessing import create_polar_pseudo_image


class PerceptionEngine:
    """
    Performs real-time semantic segmentation on incoming PointCloudFrames.
    """

    def __init__(
        self,
        model_checkpoint: Optional[str] = "checkpoints/best.pt",
        device: str = "auto",
        num_classes: int = 20,
        mode: str = "ground_truth" # 'ground_truth' or 'ai'
    ):
        self.mode = mode
        self.num_classes = num_classes
        
        # Select device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model: Optional[AdaptivePolarNet] = None
        self.is_model_loaded = False
        
        # Try loading checkpoint if in AI mode or if available
        if model_checkpoint and os.path.exists(model_checkpoint):
            self._load_model(model_checkpoint)
        elif self.mode == "ai":
            # Instantiate untrained/baseline model for AI forward pass demonstration
            self._init_fresh_model()

    def _init_fresh_model(self):
        """Initialize model architecture."""
        try:
            self.model = AdaptivePolarNet(
                num_classes=self.num_classes,
                in_channels=8,
                encoder_channels=64,
                backbone_channels=[64, 128, 256],
                upsample_channels=[128, 128, 128]
            ).to(self.device)
            self.model.eval()
            self.is_model_loaded = True
        except Exception as e:
            print(f"[PerceptionEngine] Model initialization note: {e}")
            self.is_model_loaded = False

    def _load_model(self, checkpoint_path: str):
        """Load weights from checkpoint."""
        try:
            self._init_fresh_model()
            ckpt = torch.load(checkpoint_path, map_location=self.device)
            state_dict = ckpt.get("model_state", ckpt)
            self.model.load_state_dict(state_dict)
            self.model.eval()
            self.is_model_loaded = True
            print(f"[PerceptionEngine] Loaded weights from {checkpoint_path}")
        except Exception as e:
            print(f"[PerceptionEngine] Could not load checkpoint ({e}). Falling back to heuristic/GT.")
            self.is_model_loaded = False

    def predict(
        self,
        frame: PointCloudFrame,
        force_mode: Optional[str] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run perception inference on frame.
        
        Returns:
            predicted_labels: (N,) uint8 array of learning class IDs (0..19)
            confidences: (N,) float32 array in [0.0, 1.0]
        """
        active_mode = force_mode if force_mode is not None else self.mode
        pts = frame.points
        N = len(pts)

        # MODE 1: Ground Truth
        if active_mode == "ground_truth":
            if frame.raw_labels is not None:
                sem_ids = frame.semantic_labels
                # Return mapped semantic labels if available
                # Map raw SemanticKITTI to learning class
                # Default identity / mapping
                confidences = np.ones(N, dtype=np.float32)
                return sem_ids, confidences
            else:
                return self._heuristic_segmentation(pts)

        # MODE 2: Deep Learning Model Inference
        if self.is_model_loaded and self.model is not None:
            try:
                # 1. Rasterize to polar pseudo-image (8, 128, 256)
                pseudo_img, r_idx, s_idx, valid = create_polar_pseudo_image(
                    pts, num_rings=128, num_sectors=256
                )
                
                # 2. PyTorch Tensor
                tensor_in = torch.from_numpy(pseudo_img).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    logits = self.model(tensor_in) # (1, 20, 128, 256)
                    probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy() # (20, 128, 256)

                # 3. Project back to points
                grid_preds = np.argmax(probs, axis=0) # (128, 256)
                grid_confs = np.max(probs, axis=0)   # (128, 256)

                point_labels = np.zeros(N, dtype=np.uint8)
                point_confs = np.zeros(N, dtype=np.float32)

                if np.any(valid):
                    point_labels[valid] = grid_preds[r_idx[valid], s_idx[valid]]
                    point_confs[valid] = grid_confs[r_idx[valid], s_idx[valid]]

                return point_labels, point_confs
            except Exception as e:
                return self._heuristic_segmentation(pts)

        # Fallback
        return self._heuristic_segmentation(pts)

    def _heuristic_segmentation(self, pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Fast heuristic geometric classifier for robust fallback."""
        N = len(pts)
        z = pts[:, 2]
        r = np.sqrt(pts[:, 0]**2 + pts[:, 1]**2)

        labels = np.zeros(N, dtype=np.uint8)
        confidences = np.full(N, 0.85, dtype=np.float32)

        # Terrain / Road height threshold
        is_ground = (z < -1.2) & (z > -2.5)
        labels[is_ground] = 9 # road

        # Obstacles
        is_obstacle = (z >= -1.2) & (z < 2.0)
        labels[is_obstacle] = 13 # building / static structure

        # High vegetation / poles
        is_high = (z >= 2.0)
        labels[is_high] = 15 # vegetation

        return labels, confidences
