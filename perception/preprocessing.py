"""
Polar Point Cloud Preprocessing & Pseudo-Image Rasterization
"""

from typing import Tuple, Optional
import numpy as np
import torch


def create_polar_pseudo_image(
    points: np.ndarray,
    num_rings: int = 128,
    num_sectors: int = 256,
    max_range: float = 100.0,
    min_range: float = 1.0,
    min_z: float = -3.0,
    max_z: float = 5.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Construct a dense polar pseudo-image feature map for ConvNet backbone.
    Channels: [mean_z, max_z, min_z, mean_intensity, density, std_z, mean_r, mean_theta] (8 channels)
    
    Returns:
        pseudo_image: (8, num_rings, num_sectors) float32
        point_ring_idx: (N,) int32
        point_sec_idx: (N,) int32
        valid_point_mask: (N,) bool
    """
    N = len(points)
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    intensity = points[:, 3] if points.shape[1] > 3 else np.zeros(N, dtype=np.float32)

    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    theta_0_2pi = np.where(theta < 0, theta + 2 * np.pi, theta)

    # Valid bounds
    valid = (r >= min_range) & (r <= max_range) & (z >= min_z) & (z <= max_z)
    
    # Ring & sector addressing
    ring_idx = np.floor((r - min_range) / ((max_range - min_range) / num_rings)).astype(np.int32)
    ring_idx = np.clip(ring_idx, 0, num_rings - 1)
    
    sec_idx = np.floor(theta_0_2pi * (num_sectors / (2.0 * np.pi))).astype(np.int32)
    sec_idx = np.clip(sec_idx, 0, num_sectors - 1)

    pseudo_image = np.zeros((8, num_rings, num_sectors), dtype=np.float32)
    
    if not np.any(valid):
        return pseudo_image, ring_idx, sec_idx, valid

    v_r = ring_idx[valid]
    v_s = sec_idx[valid]
    v_z = z[valid]
    v_i = intensity[valid]
    v_rad = r[valid]
    v_th = theta[valid]

    lin_idx = v_r * num_sectors + v_s
    unique_cells, split_idx, counts = np.unique(
        np.sort(lin_idx), return_index=True, return_counts=True
    )
    splits = np.split(np.argsort(lin_idx), split_idx[1:])

    for cell_id, s in zip(unique_cells, splits):
        cr = cell_id // num_sectors
        cs = cell_id % num_sectors
        
        pts_z = v_z[s]
        pts_i = v_i[s]
        
        mean_z = np.mean(pts_z)
        max_z_val = np.max(pts_z)
        min_z_val = np.min(pts_z)
        mean_i = np.mean(pts_i)
        std_z = np.std(pts_z) if len(pts_z) > 1 else 0.0
        
        pseudo_image[0, cr, cs] = mean_z
        pseudo_image[1, cr, cs] = max_z_val
        pseudo_image[2, cr, cs] = min_z_val
        pseudo_image[3, cr, cs] = mean_i
        pseudo_image[4, cr, cs] = len(s) / 50.0 # normalized density
        pseudo_image[5, cr, cs] = std_z
        pseudo_image[6, cr, cs] = np.mean(v_rad[s]) / max_range
        pseudo_image[7, cr, cs] = np.mean(v_th[s]) / np.pi

    return pseudo_image, ring_idx, sec_idx, valid
