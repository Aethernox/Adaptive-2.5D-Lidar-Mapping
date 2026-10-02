"""Unified frame sources for the unchanged grid/tracker/dashboard pipeline."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np

from config import (CLASS_MAPPING_VERSION, FEATURE_NORMALIZATION_VERSION, KITTI_MAX_FRAMES, KITTI_SEQUENCE,
                    NUM_CLASSES, TRAINED_MODEL_PATH)
from data.kitti import KittiDataError, KittiSequenceDataset, available_sequences
from data.semantic_kitti import map_to_prototype
from sim.lidar import LidarSimulator


@dataclass
class UnifiedFrame:
    points: np.ndarray
    frame_idx: int
    timestamp: float
    ego: dict
    true_labels: np.ndarray | None = None
    instance_labels: np.ndarray | None = None
    sequence: str = "DETERMINISTIC"
    source: str = "synthetic"


class SyntheticFrameSource:
    source = "synthetic"
    def __init__(self, seed=42, dt=0.45):
        self.sim, self.dt, self.index = LidarSimulator(seed=seed), dt, 0

    @property
    def total_frames(self): return None
    @property
    def available_sequences(self): return []

    def reset(self): self.index = 0
    def next_frame(self) -> UnifiedFrame:
        raw = self.sim.sweep(self.index * self.dt)
        ego_dict = raw.get("ego_state", {"x": raw["ego_x"], "y": raw["ego_y"], "heading": raw["heading"]})
        frame = UnifiedFrame(raw["points"], self.index, self.index * self.dt,
                             ego_dict, raw["true_labels"], None, "DETERMINISTIC", self.source)
        self.index += 1
        return frame


class KittiFrameSource:
    source = "kitti"
    def __init__(self, dataset_root: str, sequence: str = KITTI_SEQUENCE, start_frame: int = 0,
                 max_frames: int = KITTI_MAX_FRAMES):
        self.dataset = KittiSequenceDataset(dataset_root, sequence=sequence, require_labels=False)
        self.frame_indices = list(range(len(self.dataset)))
        if max_frames > 0:
            self.frame_indices = self.frame_indices[:max_frames]
        self.index = max(0, min(start_frame, len(self.frame_indices) - 1))

    @property
    def total_frames(self): return len(self.frame_indices)
    @property
    def available_sequences(self): return available_sequences(self.dataset.root)
    @property
    def sequence(self): return self.dataset.sequence

    def reset(self): self.index = 0
    @property
    def frame_ids(self): return [self.dataset.frame_ids[i] for i in self.frame_indices]

    def index_for_frame_id(self, frame_id: int) -> int:
        """Resolve a native KITTI frame number to its indexed replay position."""
        ids = np.asarray(self.frame_ids)
        exact = np.flatnonzero(ids == int(frame_id))
        if exact.size:
            return int(exact[0])
        # A scrubber can emit an integer between sparse source frame names.
        return int(np.abs(ids - int(frame_id)).argmin())

    def seek(self, index): self.index = max(0, min(int(index), len(self.frame_indices) - 1))
    def next_frame(self) -> UnifiedFrame:
        raw = self.dataset.get_frame(self.frame_indices[self.index], require_labels=False)
        true_labels = None if raw.semantic_labels is None else map_to_prototype(raw.semantic_labels)
        ego = {"x": 0.0, "y": 0.0, "heading": 0.0}  # points remain in KITTI Velodyne ego coordinates
        index = self.index
        self.index = min(self.index + 1, len(self.frame_indices) - 1)
        return UnifiedFrame(raw.points, raw.frame_idx, index * 0.1, ego, true_labels, raw.instance_labels,
                            raw.sequence, self.source)


def load_kitti_model_artifact(path: str | Path = TRAINED_MODEL_PATH):
    """Load and validate a real-data model; never silently fall back to GT."""
    from perception.model import PointSegModel
    path = Path(path)
    metadata_path = path.parent / "metadata.json"
    if not path.exists() or not metadata_path.exists():
        raise KittiDataError(f"KITTI model not trained. Expected weights and metadata at {path.parent}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    mismatches = []
    if metadata.get("num_classes") != NUM_CLASSES: mismatches.append("class count")
    if metadata.get("feature_normalization_version") != FEATURE_NORMALIZATION_VERSION: mismatches.append("feature normalization version")
    if metadata.get("class_mapping_version") != CLASS_MAPPING_VERSION: mismatches.append("class mapping version")
    if mismatches:
        raise KittiDataError("Incompatible KITTI model artifact: " + ", ".join(mismatches))
    return PointSegModel.load(path), metadata
