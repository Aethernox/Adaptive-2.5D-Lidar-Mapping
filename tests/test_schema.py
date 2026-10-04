"""
Unit Tests for Core Data Contracts and Schemas
"""

import numpy as np
import pytest

from core.schema import (
    PointCloudFrame,
    Pose,
    TierSpec,
    TierSchedule,
    TierGridData,
    Track,
    TrackSet
)


def test_pointcloud_frame():
    points = np.array([
        [1.0, 2.0, 3.0, 0.5],
        [4.0, 5.0, 6.0, 0.8],
        [7.0, 8.0, 9.0, 0.1]
    ], dtype=np.float32)
    
    # Pack semantic=10 (car), instance=3 -> (3 << 16) | 10 = 196618
    raw_labels = np.array([(3 << 16) | 10, 10, 0], dtype=np.uint32)
    
    frame = PointCloudFrame(
        seq=1,
        stamp_ns=100000000,
        frame_id="velodyne",
        points=points,
        raw_labels=raw_labels
    )
    
    assert frame.num_points == 3
    assert frame.semantic_labels[0] == 10
    assert frame.instance_ids[0] == 3
    assert frame.semantic_labels[1] == 10
    assert frame.instance_ids[1] == 0


def test_pose_conversions():
    # Identity test
    T_eye = np.eye(4, dtype=np.float64)
    pose_eye = Pose.from_matrix(T_eye)
    assert np.allclose(pose_eye.to_matrix(), T_eye)
    
    # Pure translation
    T_trans = np.eye(4, dtype=np.float64)
    T_trans[0, 3] = 12.5
    T_trans[1, 3] = -4.2
    T_trans[2, 3] = 1.8
    pose_trans = Pose.from_matrix(T_trans)
    assert np.allclose(pose_trans.to_matrix(), T_trans)
    
    # 90 deg rotation around Z
    T_rot = np.array([
        [0.0, -1.0, 0.0, 5.0],
        [1.0,  0.0, 0.0, 2.0],
        [0.0,  0.0, 1.0, 0.0],
        [0.0,  0.0, 0.0, 1.0]
    ], dtype=np.float64)
    pose_rot = Pose.from_matrix(T_rot)
    assert np.allclose(pose_rot.to_matrix(), T_rot, atol=1e-5)


def test_tier_schedule_math():
    schedule = TierSchedule.default()
    assert len(schedule.tiers) == 4
    
    # Tier 0: 0-10m @ 0.05m -> 200 rings, 1024 sectors -> 204,800 cells
    t0 = schedule.tiers[0]
    assert t0.num_rings == 200
    assert t0.azimuth_bins == 1024
    assert t0.total_cells == 204800
    
    # Tier 1: 10-25m @ 0.15m -> 100 rings, 512 sectors -> 51,200 cells
    t1 = schedule.tiers[1]
    assert t1.num_rings == 100
    assert t1.total_cells == 51200
    
    # Tier 2: 25-50m @ 0.30m -> 83 rings, 256 sectors -> 21,248 cells
    t2 = schedule.tiers[2]
    assert t2.num_rings == 83
    assert t2.total_cells == 83 * 256
    
    # Tier 3: 50-100m @ 0.50m -> 100 rings, 128 sectors -> 12,800 cells
    t3 = schedule.tiers[3]
    assert t3.num_rings == 100
    assert t3.total_cells == 12800
    
    # Total adaptive cells: ~290k cells vs 16M uniform cells
    total = schedule.total_cells
    assert 285000 <= total <= 295000


def test_tier_grid_data():
    grid = TierGridData.create_empty(tier_id=0, num_rings=200, num_sectors=1024)
    assert grid.ground_height.shape == (200, 1024)
    assert np.all(np.isnan(grid.ground_height))
    assert grid.semantic_class.shape == (200, 1024)
    assert np.all(grid.semantic_class == 0)


if __name__ == "__main__":
    test_pointcloud_frame()
    test_pose_conversions()
    test_tier_schedule_math()
    test_tier_grid_data()
    print("All schema tests passed!")
