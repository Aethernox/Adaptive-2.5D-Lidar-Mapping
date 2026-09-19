"""
Orchestrates one frame of the full pipeline:

    raw points --(perception model)--> per-point class+confidence
               --(grid engine)--> variable-resolution 2.5D map
               --(tracker)--> sparse dynamic-object overlay
               --(rasterizer)--> RGB image for the dashboard

Also computes the live metrics panel: FPS / per-stage latency, memory
footprint vs a uniform-grid baseline, and range-bucketed classification
accuracy (using the simulator's ground-truth labels, since this prototype
has no held-out real dataset to score against at runtime).
"""
import os
import time
import base64
import io
import numpy as np
from PIL import Image

from config import RANGE_BUCKETS, CLASSES, NUM_CLASSES, DYNAMIC_CLASS_IDS, TIERS
from sim.lidar import LidarSimulator
from perception.model import PointSegModel
from perception.train import train as train_model, WEIGHTS_PATH
from mapping.grid_engine import VariableResolutionGrid
from mapping.tracker import ObjectTracker
from mapping.rasterizer import GridRasterizer

SIM_DT = 0.45  # seconds of simulated ego motion per pipeline step
MAX_CLOUD_POINTS = 3600


def _load_or_train_model():
    if os.path.exists(WEIGHTS_PATH):
        return PointSegModel.load(WEIGHTS_PATH)
    return train_model(verbose=False)


class Pipeline:
    def __init__(self, seed=42):
        self.sim = LidarSimulator(seed=seed)
        self.model = _load_or_train_model()
        self.grid = VariableResolutionGrid()
        self.tracker = ObjectTracker()
        self.rasterizer = GridRasterizer()
        self.t = 0.0
        self.frame_idx = 0
        self._acc_history = []  # rolling per-range-bucket accuracy

    def reset(self):
        self.__init__()

    def step(self):
        stage_t = {}
        t_frame0 = time.perf_counter()

        t0 = time.perf_counter()
        frame = self.sim.sweep(self.t)
        stage_t["sense_ms"] = (time.perf_counter() - t0) * 1000

        points = frame["points"]
        true_labels = frame["true_labels"]

        t0 = time.perf_counter()
        pred_cls, pred_conf = self.model.predict(points)
        stage_t["infer_ms"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        # Dynamic-class points are NOT densified into the persistent grid —
        # they live only in the sparse tracked-object overlay (see
        # mapping/tracker.py) so the grid's memory budget isn't spent
        # re-storing a handful of moving boxes into every cell they touch.
        static_mask = ~np.isin(pred_cls, list(DYNAMIC_CLASS_IDS))
        self.grid.update(points[static_mask], pred_cls[static_mask], pred_conf[static_mask])
        stage_t["fuse_ms"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        objects = self.tracker.update(points, pred_cls, self.t, SIM_DT)
        stage_t["track_ms"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        img = self.rasterizer.render(self.grid, dynamic_objects=objects)
        stage_t["render_ms"] = (time.perf_counter() - t0) * 1000

        total_ms = (time.perf_counter() - t_frame0) * 1000
        fps = 1000.0 / total_ms if total_ms > 0 else 0.0

        acc_by_bucket = self._range_bucket_accuracy(points, pred_cls, true_labels)

        mem = self.grid.memory_stats()

        self.t += SIM_DT
        self.frame_idx += 1

        return dict(
            frame_idx=self.frame_idx,
            image_b64=self._encode_png(img),
            point_cloud=self._point_cloud_payload(points, pred_cls, pred_conf),
            ego=dict(x=frame["ego_x"], y=frame["ego_y"], heading=frame["heading"]),
            n_points=int(len(points)),
            objects=objects,
            adaptive_grid=dict(
                scheme="tiered_log_polar",
                cells=self.grid.visualization_cells(),
                tiers=[dict(name=t["name"], r_min=t["r_min"], r_max=t["r_max"],
                            dr=t["dr"], n_sectors=t["n_sectors"]) for t in TIERS],
            ),
            latency_ms=stage_t,
            total_ms=total_ms,
            fps=fps,
            memory=mem,
            accuracy_by_range=acc_by_bucket,
            class_names=CLASSES,
        )

    @staticmethod
    def _encode_png(img_array):
        buf = io.BytesIO()
        Image.fromarray(img_array, mode="RGB").save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii")

    @staticmethod
    def _point_cloud_payload(points, class_ids, confidences):
        """Return a bounded, JSON-friendly point sample for the live 3D view."""
        if len(points) <= MAX_CLOUD_POINTS:
            indices = np.arange(len(points))
        else:
            indices = np.linspace(0, len(points) - 1, MAX_CLOUD_POINTS, dtype=np.int32)
        sampled = points[indices, :3]
        return np.column_stack((
            np.round(sampled, 2),
            class_ids[indices].astype(np.int16),
            np.round(confidences[indices], 3),
            np.round(points[indices, 3], 3),
        )).tolist()

    def _range_bucket_accuracy(self, points, pred_cls, true_labels):
        r = np.hypot(points[:, 0], points[:, 1])
        out = []
        for lo, hi in RANGE_BUCKETS:
            m = (r >= lo) & (r < hi)
            n = int(m.sum())
            if n == 0:
                out.append(dict(range=f"{lo}-{hi}m", n_points=0, accuracy=None, miou=None))
                continue
            acc = float((pred_cls[m] == true_labels[m]).mean())
            ious = []
            for c in range(NUM_CLASSES):
                pm, tm = pred_cls[m] == c, true_labels[m] == c
                union = (pm | tm).sum()
                if union == 0:
                    continue
                ious.append((pm & tm).sum() / union)
            miou = float(np.mean(ious)) if ious else None
            out.append(dict(range=f"{lo}-{hi}m", n_points=n, accuracy=acc, miou=miou))
        return out
