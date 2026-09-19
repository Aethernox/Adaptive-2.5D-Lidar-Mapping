import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import NUM_CLASSES
from perception.model import PointSegModel
from sim.lidar import LidarSimulator
from mapping.tracker import ObjectTracker
from pipeline import Pipeline


def test_model_predict_shapes():
    model = PointSegModel(seed=0)
    pts = np.random.uniform(-10, 10, size=(200, 4))
    cls, conf = model.predict(pts)
    assert cls.shape == (200,)
    assert conf.shape == (200,)
    assert cls.min() >= 0 and cls.max() < NUM_CLASSES
    assert (conf >= 0).all() and (conf <= 1).all()


def test_model_predict_empty_points():
    model = PointSegModel(seed=0)
    cls, conf = model.predict(np.zeros((0, 4)))
    assert len(cls) == 0 and len(conf) == 0


def test_lidar_sim_produces_points_with_range_spread():
    sim = LidarSimulator(seed=1)
    frame = sim.sweep(10.0)
    pts = frame["points"]
    assert pts.shape[1] == 4
    r = np.hypot(pts[:, 0], pts[:, 1])
    assert r.min() >= 0
    assert r.max() <= 100.5
    assert len(frame["true_labels"]) == len(pts)


def test_tracker_tracks_moving_actor_across_frames():
    tracker = ObjectTracker()
    # simulate a vehicle moving at 2 m/s along x, sampled as a small cluster
    for step in range(4):
        t = step * 0.5
        cx = 10.0 + 2.0 * t
        cluster = np.random.normal(loc=[cx, 0.0, 0.5], scale=[0.5, 0.5, 0.1], size=(30, 3))
        cls = np.full(30, 5)  # vehicle class id
        objs = tracker.update(cluster, cls, t, dt=0.5)
    assert len(objs) >= 1
    veh = objs[0]
    assert veh["kind"] == "vehicle"
    # after several frames of consistent motion, velocity estimate should be
    # roughly in the direction of travel
    assert veh["vx"] > 0.2


def test_pipeline_end_to_end_runs_and_returns_valid_frame():
    p = Pipeline(seed=7)
    out = p.step()
    assert "image_b64" in out and len(out["image_b64"]) > 100
    assert out["n_points"] > 0
    assert 0 < len(out["point_cloud"]) <= 3600
    assert all(len(point) == 6 for point in out["point_cloud"][:10])
    assert out["fps"] > 0
    assert out["memory"]["reduction_factor"] > 1
    assert len(out["accuracy_by_range"]) == 4
    # run a second step to make sure state (grid persistence, tracker) is stable
    out2 = p.step()
    assert out2["frame_idx"] == out["frame_idx"] + 1
