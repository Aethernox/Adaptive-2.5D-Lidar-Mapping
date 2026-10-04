"""
Unit Tests for Adaptive Polar Grid Engine
Tests cell address math, half-open intervals, multi-statistic aggregation, and rasterization.
"""

import numpy as np
import pytest

from mapping.adaptive_grid import AdaptivePolarGrid
from core.schema import TierSchedule, TierSpec


def test_cell_addressing_math():
    grid = AdaptivePolarGrid()
    
    # 1. Test point at (3.0, 4.0) -> r = 5.0m, theta = atan2(4, 3) ~ 0.927 rad
    x = np.array([3.0], dtype=np.float32)
    y = np.array([4.0], dtype=np.float32)
    
    tier_idx, ring_idx, sec_idx, valid = grid.lookup_cell_indices(x, y)
    
    assert valid[0] == True
    assert tier_idx[0] == 0 # Tier 0: 0-10m
    # Tier 0 delta_r = 0.05m -> ring = floor(5.0 / 0.05) = 100
    assert ring_idx[0] == 100
    assert sec_idx[0] >= 0 and sec_idx[0] < 1024


def test_tier_seam_boundary_half_open_interval():
    grid = AdaptivePolarGrid()
    
    # Exactly at r = 10.0m (boundary between Tier 0 and Tier 1)
    # Under [r_min, r_max) convention, r = 10.0m belongs to Tier 1 [10.0, 25.0)
    x = np.array([10.0, 0.0, 25.0, 50.0], dtype=np.float32)
    y = np.array([0.0, 10.0, 0.0, 0.0], dtype=np.float32)
    
    tier_idx, ring_idx, sec_idx, valid = grid.lookup_cell_indices(x, y)
    
    assert tier_idx[0] == 1 # Tier 1: [10, 25)
    assert ring_idx[0] == 0 # First ring of Tier 1
    assert tier_idx[1] == 1 # Tier 1: [10, 25)
    assert tier_idx[2] == 2 # Tier 2: [25, 50)
    assert tier_idx[3] == 3 # Tier 3: [50, 100)


def test_multi_statistic_cell_aggregation():
    grid = AdaptivePolarGrid()
    
    # Create synthetic cluster of points in a single cell (around r=5m, angle=0)
    N = 50
    x = np.full(N, 5.02, dtype=np.float32)
    y = np.full(N, 0.01, dtype=np.float32)
    
    # Heights from -1.5m (ground) to +2.0m (obstacle roof)
    z = np.linspace(-1.5, 2.0, N, dtype=np.float32)
    pts = np.stack([x, y, z, np.ones(N)], axis=1)
    
    # 30 road points (class 9), 20 building points (class 13)
    labels = np.array([9]*30 + [13]*20, dtype=np.uint8)
    confs = np.full(N, 0.9, dtype=np.float32)
    
    grid.update_from_points(pts, labels, confs, clear_first=True)
    
    t0_data = grid.tier_grids[0]
    # Cell at ring=100, sector=0
    assert t0_data.occupied_mask[100, 0] == True
    # Majority class should be road (class 9, 30 votes vs 20)
    assert t0_data.semantic_class[100, 0] == 9
    # Ground height should be near -1.5m
    assert -1.6 <= t0_data.ground_height[100, 0] <= -1.0
    # Obstacle top height should be near +2.0m
    assert 1.5 <= t0_data.obstacle_top_height[100, 0] <= 2.1


def test_rasterization():
    grid = AdaptivePolarGrid()
    pts = np.array([[5.0, 5.0, 0.0, 1.0]], dtype=np.float32)
    labels = np.array([9], dtype=np.uint8)
    grid.update_from_points(pts, labels)
    
    img = grid.rasterize_to_cartesian(grid_size=100, extent_m=50.0)
    assert img.shape == (100, 100)
    assert np.any(img > 0)


if __name__ == "__main__":
    test_cell_addressing_math()
    test_tier_seam_boundary_half_open_interval()
    test_multi_statistic_cell_aggregation()
    test_rasterization()
    print("All adaptive grid tests passed!")
