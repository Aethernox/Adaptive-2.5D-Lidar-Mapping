"""Unified synthetic/KITTI replay pipeline preserving the grid/tracker contract."""
from __future__ import annotations
import base64
import io
import os
import time
import numpy as np
from PIL import Image

from config import (CLASSES, DATASET_TYPE, DYNAMIC_CLASS_IDS, KITTI_DATASET_ROOT, KITTI_MAX_FRAMES, KITTI_SEQUENCE,
                    KITTI_START_FRAME, MAX_CLOUD_POINTS, MAX_INFERENCE_POINTS, NUM_CLASSES, RANGE_BUCKETS,
                    RNG_SEED, TIERS, TRAINED_MODEL_PATH)
from data.kitti import KittiDataError
from data.sources import KittiFrameSource, SyntheticFrameSource, load_kitti_model_artifact
from perception.model import PointSegModel
from perception.features import extract_features
from perception.train import WEIGHTS_PATH, train as train_synthetic_model
from mapping.grid_engine import VariableResolutionGrid
from mapping.tracker import ObjectTracker
from mapping.rasterizer import GridRasterizer


def _load_synthetic_model():
    return PointSegModel.load(WEIGHTS_PATH) if os.path.exists(WEIGHTS_PATH) else train_synthetic_model(verbose=False)


class Pipeline:
    def __init__(self, seed=RNG_SEED, dataset_type=DATASET_TYPE, dataset_root=KITTI_DATASET_ROOT,
                 sequence=KITTI_SEQUENCE, model_path=TRAINED_MODEL_PATH, max_frames=KITTI_MAX_FRAMES):
        self.seed, self.dataset_type, self.dataset_root, self.sequence, self.model_path = seed, dataset_type.lower(), dataset_root, sequence, model_path
        if self.dataset_type == "kitti":
            if not dataset_root:
                raise KittiDataError("KITTI_DATASET_ROOT or --dataset-root is required for KITTI replay")
            self.source = KittiFrameSource(dataset_root, sequence, KITTI_START_FRAME, max_frames=max_frames)
            self.model, self.model_metadata = load_kitti_model_artifact(model_path)
        elif self.dataset_type == "synthetic":
            self.source, self.model, self.model_metadata = SyntheticFrameSource(seed=seed), _load_synthetic_model(), None
        else:
            raise ValueError("dataset_type must be 'synthetic' or 'kitti'")
        self.grid, self.tracker, self.rasterizer = VariableResolutionGrid(), ObjectTracker(), GridRasterizer()
        self._step_counter, self._previous_time = 0, None

    def reset(self):
        """Reset runtime state without re-training or reloading a model."""
        self.source.reset(); self.grid = VariableResolutionGrid(); self.tracker = ObjectTracker()
        self._step_counter = 0; self._previous_time = None

    def seek(self, target: int):
        target = max(0, int(target))
        if self.dataset_type == "kitti":
            target = self.source.index_for_frame_id(target)
        elif self.source.total_frames is not None:
            target = min(target, self.source.total_frames - 1)
        self.reset(); out = None
        for _ in range(target + 1):
            out = self.step()
        return out

    @property
    def total_frames(self): return self.source.total_frames

    def config_payload(self):
        current_frame = self._step_counter
        frame_min, frame_max = 1, 500
        if self.dataset_type == "kitti":
            frame_min, frame_max = self.source.frame_ids[0], self.source.frame_ids[-1]
            current_frame = frame_min if self._step_counter == 0 else self.source.frame_ids[max(0, self.source.index - 1)]
        return {"dataset_type": self.dataset_type, "current_sequence": getattr(self.source, "sequence", "DETERMINISTIC"),
                "available_sequences": self.source.available_sequences, "total_frames": self.total_frames,
                "current_frame": current_frame, "frame_min": frame_min, "frame_max": frame_max,
                "ground_truth_available": bool(getattr(self.source, "dataset", None) and self.source.dataset.has_labels)}

    def step(self):
        timings, frame_started = {}, time.perf_counter()
        t0 = time.perf_counter(); frame = self.source.next_frame(); timings["data_load_ms"] = (time.perf_counter() - t0) * 1000
        timings["sense_ms"] = timings["data_load_ms"]  # legacy dashboard name
        original_points = frame.points
        indices = np.arange(len(original_points))
        if MAX_INFERENCE_POINTS > 0 and len(indices) > MAX_INFERENCE_POINTS:
            indices = np.linspace(0, len(indices) - 1, MAX_INFERENCE_POINTS, dtype=np.int32)
        points = original_points[indices]
        true_labels = None if frame.true_labels is None else frame.true_labels[indices]
        instances = None if frame.instance_labels is None else frame.instance_labels[indices]

        # Training and inference both use this exact feature extractor.
        t0 = time.perf_counter(); features = extract_features(points)
        timings["preprocessing_ms"] = (time.perf_counter() - t0) * 1000
        t0 = time.perf_counter(); pred_cls, pred_conf = self.model.predict_features(features)
        timings["inference_ms"] = timings["infer_ms"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter(); static_mask = ~np.isin(pred_cls, list(DYNAMIC_CLASS_IDS))
        self.grid.update(points[static_mask], pred_cls[static_mask], pred_conf[static_mask])
        timings["grid_fusion_ms"] = timings["fuse_ms"] = (time.perf_counter() - t0) * 1000

        dt = 0.1 if self._previous_time is None else max(frame.timestamp - self._previous_time, 1e-3)
        t0 = time.perf_counter(); objects = self.tracker.update(points, pred_cls, frame.timestamp, dt)
        timings["tracking_ms"] = timings["track_ms"] = (time.perf_counter() - t0) * 1000
        t0 = time.perf_counter(); image = self.rasterizer.render(self.grid, dynamic_objects=objects)
        timings["render_ms"] = (time.perf_counter() - t0) * 1000
        total_ms = (time.perf_counter() - frame_started) * 1000
        self._previous_time, self._step_counter = frame.timestamp, self._step_counter + 1
        frame_number = frame.frame_idx if self.dataset_type == "kitti" else self._step_counter
        return {
            "frame_idx": int(frame_number), "dataset_type": self.dataset_type, "sequence": frame.sequence,
            "sensor": "VELODYNE" if self.dataset_type == "kitti" else "LIDAR · 360°", "mode": "REPLAY" if self.dataset_type == "kitti" else "DETERMINISTIC",
            "total_frames": self.total_frames, "image_b64": self._encode_png(image),
            "point_cloud": self._point_cloud_payload(points, pred_cls, pred_conf, true_labels, instances,
                                                       include_ground_truth=self.dataset_type == "kitti"),
            "ego": frame.ego, "n_points": int(len(original_points)), "inference_points": int(len(points)), "objects": objects,
            "adaptive_grid": {"scheme": "tiered_log_polar", "cells": self.grid.visualization_cells(),
                              "tiers": [{k: t[k] for k in ("name", "r_min", "r_max", "dr", "n_sectors")} for t in TIERS]},
            "latency_ms": timings, "total_ms": total_ms, "fps": 1000.0 / total_ms if total_ms else 0.0,
            "memory": self.grid.memory_stats(), "accuracy_by_range": self._range_bucket_accuracy(points, pred_cls, true_labels),
            "class_names": CLASSES, "ground_truth_available": true_labels is not None,
            "at_end": self.dataset_type == "kitti" and frame.frame_idx == self.source.frame_ids[-1],
        }

    @staticmethod
    def _encode_png(image):
        buffer = io.BytesIO(); Image.fromarray(image, mode="RGB").save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("ascii")

    @staticmethod
    def _point_cloud_payload(points, class_ids, confidences, true_labels=None, instance_ids=None,
                             include_ground_truth=False):
        indices = np.arange(len(points)) if len(points) <= MAX_CLOUD_POINTS else np.linspace(0, len(points)-1, MAX_CLOUD_POINTS, dtype=np.int32)
        base = np.column_stack((np.round(points[indices, :3], 2), class_ids[indices].astype(np.int16),
                                np.round(confidences[indices], 3), np.round(points[indices, 3], 3)))
        # Preserve the six-field synthetic dashboard payload. KITTI additionally
        # carries GT semantic/instance fields solely for development rendering.
        if not include_ground_truth or true_labels is None:
            return base.tolist()
        gt = true_labels[indices].astype(np.int16)
        instances = np.full(len(indices), -1, dtype=np.int32) if instance_ids is None else instance_ids[indices].astype(np.int32)
        return np.column_stack((base, gt, instances)).tolist()

    @staticmethod
    def _range_bucket_accuracy(points, prediction, truth):
        if truth is None:
            return [{"range": f"{lo}-{hi}m", "n_points": 0, "accuracy": None, "miou": None} for lo, hi in RANGE_BUCKETS]
        planar_range = np.hypot(points[:, 0], points[:, 1]); output = []
        for lo, hi in RANGE_BUCKETS:
            valid = (planar_range >= lo) & (planar_range < hi) & (truth >= 0)
            if not valid.any():
                output.append({"range": f"{lo}-{hi}m", "n_points": 0, "accuracy": None, "miou": None}); continue
            p, y = prediction[valid], truth[valid]; ious = []
            for class_id in range(NUM_CLASSES):
                union = np.logical_or(p == class_id, y == class_id).sum()
                if union: ious.append(float(np.logical_and(p == class_id, y == class_id).sum() / union))
            output.append({"range": f"{lo}-{hi}m", "n_points": int(len(y)), "accuracy": float((p == y).mean()),
                           "miou": float(np.mean(ious)) if ious else None})
        return output
