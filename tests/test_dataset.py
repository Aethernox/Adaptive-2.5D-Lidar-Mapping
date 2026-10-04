"""
Unit Tests for SemanticKITTI Dataset Loader and Transforms
"""

import numpy as np
import pytest
from pathlib import Path

from datasets.semantic_kitti import SemanticKittiDataset
from datasets.transforms import filter_point_cloud, cartesian_to_polar, polar_to_cartesian, transform_points


def test_transforms_polar_cartesian_roundtrip():
    # Test random points
    np.random.seed(42)
    pts = np.random.uniform(-50.0, 50.0, size=(1000, 4)).astype(np.float32)
    # Ensure some non-zero XY
    pts[:, 0] += 0.5
    pts[:, 1] += 0.5
    
    polar = cartesian_to_polar(pts)
    assert polar.shape == pts.shape
    
    # Check r is positive
    assert np.all(polar[:, 0] >= 0)
    # Check theta is in [-pi, pi]
    assert np.all((polar[:, 1] >= -np.pi) & (polar[:, 1] <= np.pi))
    # Check z is unchanged
    assert np.allclose(polar[:, 2], pts[:, 2])
    # Check intensity is unchanged
    assert np.allclose(polar[:, 3], pts[:, 3])
    
    # Reconstruct cartesian
    reconstructed = polar_to_cartesian(polar)
    assert np.allclose(reconstructed[:, :3], pts[:, :3], atol=1e-5)


def test_filter_point_cloud():
    pts = np.array([
        [0.1, 0.1, 0.0, 0.5],    # Too close (r < 1.0m) -> ego self-occlusion
        [5.0, 5.0, 0.0, 0.8],    # Valid (r ~ 7.07m)
        [80.0, 80.0, 0.0, 0.2],  # Too far (r > 100m)
        [np.nan, 2.0, 0.0, 0.0], # NaN
        [3.0, 4.0, 10.0, 0.1],   # Z too high (z > 6.0m)
        [3.0, 4.0, -1.5, 0.4]    # Valid (r = 5.0m, z = -1.5m)
    ], dtype=np.float32)
    
    lbls = np.array([10, 20, 30, 40, 50, 60], dtype=np.uint32)
    
    filtered_pts, filtered_lbls, mask = filter_point_cloud(
        pts, lbls, min_range=1.0, max_range=100.0, min_z=-4.0, max_z=6.0
    )
    
    # Expect only indices 1 and 5 to survive
    assert len(filtered_pts) == 2
    assert len(filtered_lbls) == 2
    assert np.allclose(filtered_pts[0], pts[1])
    assert np.allclose(filtered_pts[1], pts[5])
    assert filtered_lbls[0] == 20
    assert filtered_lbls[1] == 60


def test_semantic_kitti_loader():
    dataset = SemanticKittiDataset(
        dataset_root="kitti_dataset",
        sequences=["00"],
        subset="debug",
        max_frames=10
    )
    
    assert len(dataset) > 0
    assert len(dataset) <= 10
    
    # Load first frame
    frame = dataset[0]
    assert frame is not None
    assert frame.num_points > 0
    assert frame.points.shape[1] == 4
    assert frame.raw_labels is not None
    assert len(frame.raw_labels) == frame.num_points
    
    # Test semantic class mapping
    sem_ids = frame.semantic_labels
    assert sem_ids is not None
    learning_ids = dataset.raw_to_learning(sem_ids)
    assert len(learning_ids) == len(sem_ids)
    assert np.all((learning_ids >= 0) & (learning_ids <= 19))
    
    # Test project category mapping
    proj_cats = dataset.learning_to_project(learning_ids)
    assert len(proj_cats) == len(learning_ids)
    assert np.all((proj_cats >= 0) & (proj_cats <= 6))


if __name__ == "__main__":
    test_transforms_polar_cartesian_roundtrip()
    test_filter_point_cloud()
    test_semantic_kitti_loader()
    print("All dataset tests passed!")
