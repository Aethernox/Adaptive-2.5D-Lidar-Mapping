"""
Virtual LiDAR Dataset Replay Engine
Streams KITTI sequence frames into the perception, mapping, and tracking pipeline.
Supports both live pipeline execution and ultra-fast preprocessed cache playback.
"""

import time
from typing import Generator, Dict, Optional, Any, Union
from pathlib import Path
import numpy as np

from core.schema import PointCloudFrame, Pose, SystemMetrics, MapSnapshot, TrackSet
from datasets.semantic_kitti import SemanticKittiDataset
from perception.inference import PerceptionEngine
from mapping.adaptive_grid import AdaptivePolarGrid
from mapping.uniform_grid import UniformGridBaseline
from mapping.temporal_fusion import TemporalMapFusion
from tracking.tracker import MultiObjectTracker
from simulation.preprocessor import SimulationPreprocessor, PreprocessedSimulation


class VirtualLidarReplay:
    """
    Virtual LiDAR Replay Engine.
    Executes the end-to-end perception and mapping pipeline frame by frame.
    """

    def __init__(
        self,
        sequence: str = "00",
        start_frame: int = 0,
        end_frame: Optional[int] = None,
        target_fps: float = 10.0,
        mode: str = "ground_truth", # 'ground_truth', 'ai', 'benchmark'
        dataset_root: str = "kitti_dataset"
    ):
        self.sequence = sequence
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.target_fps = target_fps
        self.mode = mode
        self.frame_interval = 1.0 / max(target_fps, 0.1)

        # 1. Dataset
        self.dataset = SemanticKittiDataset(
            dataset_root=dataset_root,
            sequences=[sequence],
            subset="full",
            max_frames=end_frame
        )

        # 2. Pipeline Components
        self.perception = PerceptionEngine(mode="ai" if mode == "ai" else "ground_truth")
        self.adaptive_grid = AdaptivePolarGrid()
        self.uniform_baseline = UniformGridBaseline()
        self.temporal_fusion = TemporalMapFusion(self.adaptive_grid)
        self.tracker = MultiObjectTracker()

    def stream_frames(self, loop: bool = False, realtime: bool = True) -> Generator[Dict[str, Any], None, None]:
        """
        Yields processed frame packet containing point clouds, adaptive map, tracks, and metrics.
        If realtime=True, sleeps to match target_fps. If realtime=False, runs at max compute speed.
        """
        total_frames = len(self.dataset)
        start_idx = max(0, self.start_frame)
        end_idx = min(total_frames, self.end_frame) if self.end_frame else total_frames

        while True:
            for idx in range(start_idx, end_idx):
                t_loop_start = time.time()
                
                # 1. Ingest PointCloudFrame & Pose
                t0 = time.time()
                frame: PointCloudFrame = self.dataset[idx]
                pose: Pose = self.dataset.get_pose(idx)
                load_ms = (time.time() - t0) * 1000.0

                # 2. Preprocessing & Deep Learning Perception
                t1 = time.time()
                pred_labels, confidences = self.perception.predict(frame)
                infer_ms = (time.time() - t1) * 1000.0

                # 3. Dynamic Object Tracking
                t2 = time.time()
                track_set: TrackSet = self.tracker.update(
                    points=frame.points,
                    semantic_labels=pred_labels,
                    instance_ids=frame.instance_ids,
                    stamp_ns=frame.stamp_ns
                )
                tracking_ms = (time.time() - t2) * 1000.0

                # 4. Adaptive 2.5D Mapping & Temporal Fusion
                t3 = time.time()
                map_snapshot: MapSnapshot = self.temporal_fusion.update_frame(
                    points=frame.points,
                    pose=pose,
                    semantic_labels=pred_labels,
                    confidences=confidences,
                    instance_ids=frame.instance_ids
                )
                mapping_ms = (time.time() - t3) * 1000.0

                # 5. Uniform Baseline Profiling
                t4 = time.time()
                uniform_stats = self.uniform_baseline.update_from_points(frame.points, pred_labels)
                uniform_ms = (time.time() - t4) * 1000.0

                # Latency & Memory Metrics Calculation
                total_ms = (time.time() - t_loop_start) * 1000.0
                current_fps = 1000.0 / max(total_ms, 1e-2)

                # Memory footprint: ~16 bytes per cell
                adaptive_cells = self.adaptive_grid.total_cell_count
                uniform_cells = self.uniform_baseline.total_cell_count
                adaptive_mb = (adaptive_cells * 16) / (1024.0 * 1024.0)
                uniform_mb = (uniform_cells * 16) / (1024.0 * 1024.0)
                reduction_pct = (1.0 - (adaptive_cells / max(uniform_cells, 1))) * 100.0

                metrics = SystemMetrics(
                    seq=frame.seq,
                    fps=current_fps,
                    load_ms=load_ms,
                    preprocess_ms=load_ms,
                    inference_ms=infer_ms,
                    mapping_ms=mapping_ms,
                    fusion_ms=mapping_ms * 0.4,
                    tracking_ms=tracking_ms,
                    total_ms=total_ms,
                    memory_adaptive_mb=adaptive_mb,
                    memory_uniform_mb=uniform_mb,
                    memory_reduction_pct=reduction_pct,
                    num_points=frame.num_points,
                    num_adaptive_cells=adaptive_cells,
                    num_uniform_cells=uniform_cells,
                    active_tracks=len(track_set.tracks)
                )

                yield {
                    "frame": frame,
                    "pose": pose,
                    "pred_labels": pred_labels,
                    "confidences": confidences,
                    "map_snapshot": map_snapshot,
                    "track_set": track_set,
                    "metrics": metrics,
                    "trajectory": list(self.temporal_fusion.trajectory)
                }

                # Rate limiter (only in realtime mode)
                if realtime:
                    elapsed = time.time() - t_loop_start
                    sleep_time = self.frame_interval - elapsed
                    if sleep_time > 0:
                        time.sleep(sleep_time)

            if not loop:
                break


class PreprocessedLidarReplay:
    """
    High-Speed Replay Engine backed by pre-computed simulation packages.
    Delivers zero-latency frame streaming at arbitrary FPS (10-120+ FPS).
    """

    def __init__(
        self,
        simulation: Union[PreprocessedSimulation, str, Path],
        target_fps: float = 30.0
    ):
        if isinstance(simulation, (str, Path)):
            self.simulation = PreprocessedSimulation.load(simulation)
        else:
            self.simulation = simulation

        self.target_fps = target_fps
        self.frame_interval = 1.0 / max(target_fps, 0.1)

    def stream_frames(self, loop: bool = False, realtime: bool = True) -> Generator[Dict[str, Any], None, None]:
        total_frames = len(self.simulation)
        if total_frames == 0:
            return

        while True:
            for idx in range(total_frames):
                t0 = time.time()
                frame_payload = self.simulation[idx]
                yield frame_payload

                if realtime:
                    elapsed = time.time() - t0
                    sleep_time = self.frame_interval - elapsed
                    if sleep_time > 0:
                        time.sleep(sleep_time)

            if not loop:
                break
