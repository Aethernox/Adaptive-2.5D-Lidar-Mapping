"""
Unit Tests for Kalman Filtering & Multi-Object Tracking
"""

import numpy as np
import pytest

from tracking.kalman import KalmanBoxTracker
from tracking.association import associate_detections_to_trackers
from tracking.tracker import MultiObjectTracker


def test_kalman_constant_velocity_predict_update():
    # Initial box: [x=10, y=0, z=0, l=4, w=2, h=1.5, yaw=0]
    bbox = np.array([10.0, 0.0, 0.0, 4.0, 2.0, 1.5, 0.0])
    tracker = KalmanBoxTracker(bbox, class_id=1, dt=0.1)
    
    # Simulate forward motion along X: x increases by 1.0m each step (10 m/s = 36 km/h)
    for step in range(1, 6):
        pred = tracker.predict()
        meas = np.array([10.0 + step * 1.0, 0.0, 0.0, 4.0, 2.0, 1.5, 0.0])
        tracker.update(meas)

    # Velocity estimate should converge towards vx ~ 10 m/s
    vx, vy, vz = tracker.velocity
    assert 7.0 <= vx <= 13.0
    assert abs(vy) < 1.0
    assert tracker.speed >= 7.0


def test_data_association():
    dets = np.array([
        [10.0, 5.0, 0.0, 4.0, 2.0, 1.5, 0.0],
        [25.0, -10.0, 0.0, 4.0, 2.0, 1.5, 0.0]
    ])
    trks = np.array([
        [10.2, 5.1, 0.0, 4.0, 2.0, 1.5, 0.0], # Close to det 0
        [50.0, 0.0, 0.0, 4.0, 2.0, 1.5, 0.0]   # Far away from all
    ])

    matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets, trks, distance_threshold=2.5)
    
    assert len(matched) == 1
    assert matched[0][0] == 0 and matched[0][1] == 0
    assert 1 in unmatched_dets
    assert 1 in unmatched_trks


def test_multi_object_tracker_lifecycle():
    tracker = MultiObjectTracker(min_hits=2, max_age=3)
    
    # Frame 1: dynamic cluster at (15, 0, 0)
    pts1 = np.random.uniform(-0.5, 0.5, size=(30, 4))
    pts1[:, 0] += 15.0
    lbls1 = np.full(30, 1, dtype=np.uint8) # car
    
    ts1 = tracker.update(pts1, lbls1, stamp_ns=0)
    
    # Frame 2: dynamic cluster moved to (16, 0, 0)
    pts2 = np.random.uniform(-0.5, 0.5, size=(30, 4))
    pts2[:, 0] += 16.0
    lbls2 = np.full(30, 1, dtype=np.uint8)
    
    ts2 = tracker.update(pts2, lbls2, stamp_ns=100000000)
    
    assert len(ts2.tracks) >= 1
    trk = ts2.tracks[0]
    assert trk.class_id == 1
    assert 14.0 <= trk.cx <= 18.0


if __name__ == "__main__":
    test_kalman_constant_velocity_predict_update()
    test_data_association()
    test_multi_object_tracker_lifecycle()
    print("All tracker tests passed!")
