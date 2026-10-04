"""
Point Cloud Preprocessing and Coordinate Transforms
"""

from typing import Tuple, Optional
import numpy as np


def filter_point_cloud(
    points: np.ndarray,
    raw_labels: Optional[np.ndarray] = None,
    min_range: float = 1.0,
    max_range: float = 100.0,
    min_z: float = -4.0,
    max_z: float = 6.0
) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    """
    Filter invalid points, NaNs, Infs, near-field ego vehicle occlusion, and range bounds.
    
    Args:
        points: (N, 4) array [x, y, z, intensity]
        raw_labels: Optional (N,) array uint32
        min_range: Minimum range in meters (ego vehicle boundary)
        max_range: Maximum range in meters
        min_z: Minimum Z bound
        max_z: Maximum Z bound
        
    Returns:
        filtered_points: (M, 4)
        filtered_labels: (M,) or None
        valid_mask: (N,) boolean mask
    """
    if points is None or len(points) == 0:
        empty_pts = np.empty((0, 4), dtype=np.float32)
        empty_lbl = np.empty((0,), dtype=np.uint32) if raw_labels is not None else None
        return empty_pts, empty_lbl, np.empty((0,), dtype=bool)

    # 1. Finite check (no NaNs, no Infs)
    finite_mask = np.isfinite(points[:, 0]) & np.isfinite(points[:, 1]) & np.isfinite(points[:, 2])
    
    # 2. Range calculation: r = sqrt(x^2 + y^2) (XY plane) or 3D range
    r_xy = np.sqrt(points[:, 0]**2 + points[:, 1]**2)
    range_mask = (r_xy >= min_range) & (r_xy <= max_range)
    
    # 3. Z bounds
    z_mask = (points[:, 2] >= min_z) & (points[:, 2] <= max_z)
    
    valid_mask = finite_mask & range_mask & z_mask
    
    filtered_points = points[valid_mask]
    filtered_labels = raw_labels[valid_mask] if raw_labels is not None else None
    
    return filtered_points, filtered_labels, valid_mask


def cartesian_to_polar(points_xyz: np.ndarray) -> np.ndarray:
    """
    Convert (N, 3) or (N, >=3) [x, y, z, ...] to [r, theta, z, ...]
    
    where:
        r = sqrt(x^2 + y^2) in [0, inf)
        theta = atan2(y, x) in [-pi, pi)
        z = z
        
    Returns:
        polar_points: (N, >=3) array where first 3 columns are [r, theta, z]
    """
    x = points_xyz[:, 0]
    y = points_xyz[:, 1]
    z = points_xyz[:, 2]
    
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)  # in [-pi, pi)
    
    result = points_xyz.copy()
    result[:, 0] = r
    result[:, 1] = theta
    result[:, 2] = z
    return result


def polar_to_cartesian(polar_rtz: np.ndarray) -> np.ndarray:
    """
    Convert (N, 3) or (N, >=3) [r, theta, z, ...] to [x, y, z, ...]
    
    where:
        x = r * cos(theta)
        y = r * sin(theta)
        z = z
    """
    r = polar_rtz[:, 0]
    theta = polar_rtz[:, 1]
    z = polar_rtz[:, 2]
    
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    
    result = polar_rtz.copy()
    result[:, 0] = x
    result[:, 1] = y
    result[:, 2] = z
    return result


def transform_points(points: np.ndarray, transform_matrix_4x4: np.ndarray) -> np.ndarray:
    """
    Apply 4x4 homogeneous transformation to (N, 3) or (N, 4) points.
    Preserves extra feature columns (intensity, etc.).
    """
    xyz = points[:, :3]
    ones = np.ones((len(xyz), 1), dtype=xyz.dtype)
    xyz_hom = np.hstack([xyz, ones])
    
    transformed_xyz_hom = (transform_matrix_4x4 @ xyz_hom.T).T
    transformed_xyz = transformed_xyz_hom[:, :3]
    
    result = points.copy()
    result[:, :3] = transformed_xyz
    return result
