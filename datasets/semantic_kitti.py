"""
SemanticKITTI Dataset Loader and Sequence Replay Interface
"""

import os
import glob
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import yaml

from core.schema import PointCloudFrame, Pose
from datasets.transforms import filter_point_cloud


class SemanticKittiDataset:
    """
    SemanticKITTI Dataset Loader with support for:
    - Raw LiDAR point clouds (.bin)
    - Semantic & instance labels (.label)
    - Sensor calibration (calib.txt)
    - 6-DoF odometry/poses (poses.txt)
    - Class mapping to learning classes and high-level project categories
    """

    def __init__(
        self,
        dataset_root: Optional[Union[str, Path]] = None,
        sequences: Optional[List[str]] = None,
        split_mode: str = "all",  # 'train', 'val', 'test', 'all'
        subset: str = "full",     # 'debug', 'small', 'full'
        max_frames: Optional[int] = None,
        config_path: Optional[str] = "configs/dataset.yaml",
        classes_config_path: Optional[str] = "configs/classes.yaml",
        apply_filter: bool = True
    ):
        self.apply_filter = apply_filter
        
        # 1. Load configuration files
        self.config = self._load_yaml(config_path) if config_path and os.path.exists(config_path) else {}
        self.classes_cfg = self._load_yaml(classes_config_path) if classes_config_path and os.path.exists(classes_config_path) else {}
        
        # 2. Determine dataset root path
        self.dataset_root = self._resolve_dataset_root(dataset_root)
        
        # 3. Setup class mappings and colors
        self._setup_class_mappings()
        
        # 4. Resolve sequences
        if sequences is not None:
            self.sequences = [str(s).zfill(2) for s in sequences]
        else:
            self.sequences = self._resolve_sequences_for_split(split_mode)
            
        # 5. Resolve max frames limit
        self.max_frames = self._resolve_max_frames(subset, max_frames)
        
        # 6. Index all frames across sequences
        self.frames_index: List[Dict[str, Any]] = []
        self.calibrations: Dict[str, Dict[str, np.ndarray]] = {}
        self.poses: Dict[str, List[np.ndarray]] = {}
        
        self._index_dataset()

    def _load_yaml(self, path: str) -> Dict[str, Any]:
        with open(path, "r") as f:
            return yaml.safe_load(f)

    def _resolve_dataset_root(self, root_arg: Optional[Union[str, Path]]) -> Path:
        """Find the root directory containing 'sequences'."""
        candidates = []
        if root_arg:
            candidates.append(Path(root_arg))
        if "DATASET_ROOT" in os.environ:
            candidates.append(Path(os.environ["DATASET_ROOT"]))
        if "dataset" in self.config and "dataset_root" in self.config["dataset"]:
            candidates.append(Path(self.config["dataset"]["dataset_root"]))
        
        # Default local search paths
        candidates.extend([
            Path("kitti_dataset"),
            Path("data/SemanticKITTI"),
            Path("../kitti_dataset"),
            Path(".")
        ])
        
        for p in candidates:
            if p.exists() and (p / "sequences").exists():
                return p.resolve()
            if p.name == "sequences" and p.exists():
                return p.parent.resolve()
                
        # If not found directly, check if current directory has sequences
        if Path("sequences").exists():
            return Path(".").resolve()
            
        # Return first candidate or default
        return (candidates[0] if candidates else Path("kitti_dataset")).resolve()

    def _setup_class_mappings(self):
        """Construct raw_to_learning and learning_to_project lookup tables."""
        raw_to_learn = self.classes_cfg.get("raw_to_learning_map", {})
        # Pre-allocate numpy lookup table for speed (max raw label ID ~ 300)
        self.raw_to_learning_lut = np.zeros(300, dtype=np.uint8)
        for raw_id, learn_id in raw_to_learn.items():
            if int(raw_id) < 300:
                self.raw_to_learning_lut[int(raw_id)] = int(learn_id)
                
        # Learning class metadata
        self.learning_classes = self.classes_cfg.get("learning_classes", {})
        self.project_categories = self.classes_cfg.get("project_categories", {})
        self.project_category_colors = self.classes_cfg.get("project_category_colors", {})
        
        # Project category lookup: learning_id -> project_category_id (0..6)
        cat_name_to_id = {v: k for k, v in self.project_categories.items()}
        self.learning_to_project_cat = np.zeros(256, dtype=np.uint8)
        for learn_id, info in self.learning_classes.items():
            cat_name = info.get("project_category", "UNKNOWN")
            self.learning_to_project_cat[int(learn_id)] = cat_name_to_id.get(cat_name, 0)

    def _resolve_sequences_for_split(self, split_mode: str) -> List[str]:
        cfg_ds = self.config.get("dataset", {})
        if split_mode == "train":
            return [str(s).zfill(2) for s in cfg_ds.get("train_sequences", ["00"])]
        elif split_mode == "val":
            return [str(s).zfill(2) for s in cfg_ds.get("val_sequences", ["08"])]
        elif split_mode == "test":
            return [str(s).zfill(2) for s in cfg_ds.get("test_sequences", ["09", "10"])]
        else:
            # All available sequences
            seq_dir = self.dataset_root / "sequences"
            if seq_dir.exists():
                found = sorted([d.name for d in seq_dir.iterdir() if d.is_dir()])
                if found:
                    return found
            return ["00"]

    def _resolve_max_frames(self, subset: str, max_frames_arg: Optional[int]) -> Optional[int]:
        if max_frames_arg is not None:
            return max_frames_arg
        cfg_max = self.config.get("dataset", {}).get("max_frames", {})
        if subset == "debug":
            return cfg_max.get("debug", 100)
        elif subset == "small":
            return cfg_max.get("small", 500)
        return cfg_max.get("full", None)

    def _index_dataset(self):
        """Index point cloud scans, labels, calibrations, and poses."""
        for seq in self.sequences:
            seq_dir = self.dataset_root / "sequences" / seq
            if not seq_dir.exists():
                continue
                
            # Load Calibration
            calib_file = seq_dir / "calib.txt"
            calib = self._load_calib(calib_file)
            self.calibrations[seq] = calib
            
            # Load Poses
            poses = self._load_poses(seq_dir, seq, calib.get("Tr"))
            self.poses[seq] = poses
            
            # Find scans and labels
            velo_files = sorted(glob.glob(str(seq_dir / "velodyne" / "*.bin")))
            label_files = sorted(glob.glob(str(seq_dir / "labels" / "*.label")))
            
            # Match frames
            label_dict = {Path(p).stem: p for p in label_files}
            
            frames_for_seq = []
            for frame_idx, bin_path in enumerate(velo_files):
                stem = Path(bin_path).stem
                lbl_path = label_dict.get(stem, None)
                pose_mat = poses[frame_idx] if frame_idx < len(poses) else np.eye(4)
                
                frames_for_seq.append({
                    "seq": seq,
                    "frame_idx": frame_idx,
                    "stem": stem,
                    "bin_path": bin_path,
                    "label_path": lbl_path,
                    "pose": pose_mat
                })
                
                if self.max_frames and len(frames_for_seq) >= self.max_frames:
                    break
                    
            self.frames_index.extend(frames_for_seq)

    def _load_calib(self, calib_path: Path) -> Dict[str, np.ndarray]:
        """Parse KITTI calib.txt file."""
        calib = {}
        if not calib_path.exists():
            # Identity defaults
            calib["Tr"] = np.eye(4, dtype=np.float64)
            return calib
            
        with open(calib_path, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                parts = line.strip().split(":")
                if len(parts) != 2:
                    continue
                key = parts[0].strip()
                values = np.fromstring(parts[1].strip(), sep=" ", dtype=np.float64)
                
                if key == "Tr":
                    # 3x4 to 4x4
                    T = np.eye(4, dtype=np.float64)
                    T[:3, :4] = values.reshape(3, 4)
                    calib["Tr"] = T
                elif key.startswith("P"):
                    calib[key] = values.reshape(3, 4)
                    
        if "Tr" not in calib:
            calib["Tr"] = np.eye(4, dtype=np.float64)
            
        return calib

    def _load_poses(self, seq_dir: Path, seq: str, Tr: Optional[np.ndarray]) -> List[np.ndarray]:
        """
        Load poses from sequences/<seq>/poses.txt or kitti_dataset/poses/<seq>.txt.
        Converts camera poses to LiDAR frame poses:
            T_velo_world = T_cam0_world * Tr_velo_to_cam0
        """
        pose_candidates = [
            seq_dir / "poses.txt",
            self.dataset_root / "poses" / f"{seq}.txt",
            self.dataset_root / f"{seq}.txt"
        ]
        
        pose_file = None
        for p in pose_candidates:
            if p.exists():
                pose_file = p
                break
                
        poses = []
        if pose_file:
            with open(pose_file, "r") as f:
                for line in f:
                    if not line.strip():
                        continue
                    vals = np.fromstring(line.strip(), sep=" ", dtype=np.float64)
                    if len(vals) == 12:
                        T_cam = np.eye(4, dtype=np.float64)
                        T_cam[:3, :4] = vals.reshape(3, 4)
                        
                        if Tr is not None:
                            # T_velo = T_cam * Tr
                            T_velo = T_cam @ Tr
                        else:
                            T_velo = T_cam
                        poses.append(T_velo)
        
        if not poses:
            # Fallback: estimate/generate identity poses or simple constant forward motion
            velo_count = len(glob.glob(str(seq_dir / "velodyne" / "*.bin")))
            if velo_count == 0:
                velo_count = 100
            for i in range(velo_count):
                T = np.eye(4, dtype=np.float64)
                # Small synthetic forward translation (0.1m per frame @ 10Hz = 1 m/s = 3.6 km/h) for testing
                T[0, 3] = i * 0.10
                poses.append(T)
                
        return poses

    def __len__(self) -> int:
        return len(self.frames_index)

    def __getitem__(self, idx: int) -> PointCloudFrame:
        meta = self.frames_index[idx]
        
        # 1. Load point cloud
        raw_pts = np.fromfile(meta["bin_path"], dtype=np.float32).reshape(-1, 4)
        
        # 2. Load labels if present
        raw_lbls = None
        if meta["label_path"] and os.path.exists(meta["label_path"]):
            raw_lbls = np.fromfile(meta["label_path"], dtype=np.uint32)
            if len(raw_lbls) != len(raw_pts):
                raw_lbls = None  # Mismatched length fallback
                
        # 3. Apply optional preprocessing filter
        if self.apply_filter:
            min_r = self.config.get("dataset", {}).get("min_range", 1.0)
            max_r = self.config.get("dataset", {}).get("max_range", 100.0)
            min_z = self.config.get("dataset", {}).get("min_z", -4.0)
            max_z = self.config.get("dataset", {}).get("max_z", 6.0)
            pts, lbls, _ = filter_point_cloud(raw_pts, raw_lbls, min_r, max_r, min_z, max_z)
        else:
            pts, lbls = raw_pts, raw_lbls

        # 4. Construct PointCloudFrame
        stamp_ns = int(meta["frame_idx"] * 1e8)  # 10 Hz = 100ms
        return PointCloudFrame(
            seq=meta["frame_idx"],
            stamp_ns=stamp_ns,
            frame_id="velodyne",
            points=pts,
            raw_labels=lbls
        )

    def get_pose(self, idx: int) -> Pose:
        """Get ego-vehicle pose for frame idx."""
        meta = self.frames_index[idx]
        stamp_ns = int(meta["frame_idx"] * 1e8)
        return Pose.from_matrix(meta["pose"], stamp_ns=stamp_ns, reference_frame="odom")

    def raw_to_learning(self, raw_semantic_ids: np.ndarray) -> np.ndarray:
        """Map raw SemanticKITTI semantic IDs to learning class IDs (0..19)."""
        safe_ids = np.clip(raw_semantic_ids, 0, 299)
        return self.raw_to_learning_lut[safe_ids]

    def learning_to_project(self, learning_ids: np.ndarray) -> np.ndarray:
        """Map learning class IDs (0..19) to high-level project category IDs (0..6)."""
        return self.learning_to_project_cat[learning_ids]
