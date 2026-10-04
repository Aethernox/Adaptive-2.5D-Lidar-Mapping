"""
Dashboard Data Bridge & Packet Serializer
Converts pipeline objects into lightweight JSON and binary payloads for WebSocket streaming.
"""

from typing import Dict, Any, List
import numpy as np
import base64
import json

from core.schema import PointCloudFrame, Pose, MapSnapshot, TrackSet, SystemMetrics


class DashboardBridge:
    """
    Serializes pipeline state into compact JSON/base64 payloads for the web dashboard.
    """

    def __init__(self, subsample_ratio: int = 4):
        self.subsample_ratio = subsample_ratio # Downsample dense points for smooth 60fps browser rendering

    def create_frame_payload(
        self,
        frame: PointCloudFrame,
        pose: Pose,
        pred_labels: np.ndarray,
        map_snapshot: MapSnapshot,
        track_set: TrackSet,
        metrics: SystemMetrics,
        trajectory: List[Any],
        classes_cfg: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Builds a comprehensive real-time visualization payload.
        """
        pts = frame.points
        N = len(pts)

        # 1. Subsample point cloud for web transmission
        step = max(1, self.subsample_ratio)
        pts_sub = pts[::step]
        labels_sub = pred_labels[::step] if pred_labels is not None else np.zeros(len(pts_sub), dtype=np.uint8)

        # Pack XYZ and intensity/label into flat list or base64 float32 array
        # Flat list of [x, y, z, label, intensity]
        x = pts_sub[:, 0].astype(np.float32)
        y = pts_sub[:, 1].astype(np.float32)
        z = pts_sub[:, 2].astype(np.float32)
        i = pts_sub[:, 3].astype(np.float32) if pts_sub.shape[1] > 3 else np.zeros(len(pts_sub), dtype=np.float32)
        l = labels_sub.astype(np.float32)

        # 2. Serialize Dynamic 3D Tracks
        tracks_data = []
        for trk in track_set.tracks:
            tracks_data.append({
                "id": trk.id,
                "class_id": trk.class_id,
                "cx": trk.cx, "cy": trk.cy, "cz": trk.cz,
                "l": trk.length, "w": trk.width, "h": trk.height,
                "heading": trk.heading,
                "vx": trk.vx, "vy": trk.vy,
                "is_moving": trk.is_moving,
                "confidence": trk.confidence
            })

        # 3. Serialize Adaptive 2.5D Grid summary (occupied cell coordinates & semantics)
        grid_cells_data = []
        for tier_data in map_snapshot.tiers:
            occ_r, occ_s = np.where(tier_data.occupied_mask)
            if len(occ_r) == 0:
                continue
                
            # Sample occupied cells for top-down radar display
            t_id = tier_data.tier_id
            for r_idx, s_idx in zip(occ_r[::2], occ_s[::2]):
                grid_cells_data.append([
                    int(t_id),
                    int(r_idx),
                    int(s_idx),
                    int(tier_data.semantic_class[r_idx, s_idx]),
                    float(np.nan_to_num(tier_data.ground_height[r_idx, s_idx], nan=0.0)),
                    float(np.nan_to_num(tier_data.obstacle_top_height[r_idx, s_idx], nan=0.0)),
                    int(tier_data.confidence[r_idx, s_idx])
                ])

        # 4. Serialize System Metrics
        metrics_data = {
            "seq": metrics.seq,
            "fps": round(metrics.fps, 1),
            "load_ms": round(metrics.load_ms, 2),
            "preprocess_ms": round(metrics.preprocess_ms, 2),
            "inference_ms": round(metrics.inference_ms, 2),
            "mapping_ms": round(metrics.mapping_ms, 2),
            "fusion_ms": round(metrics.fusion_ms, 2),
            "tracking_ms": round(metrics.tracking_ms, 2),
            "total_ms": round(metrics.total_ms, 2),
            "memory_adaptive_mb": round(metrics.memory_adaptive_mb, 2),
            "memory_uniform_mb": round(metrics.memory_uniform_mb, 2),
            "memory_reduction_pct": round(metrics.memory_reduction_pct, 1),
            "num_points": metrics.num_points,
            "num_adaptive_cells": metrics.num_adaptive_cells,
            "num_uniform_cells": metrics.num_uniform_cells,
            "active_tracks": metrics.active_tracks
        }

        # 5. Ego Pose & Trajectory
        pose_data = {
            "px": pose.px, "py": pose.py, "pz": pose.pz,
            "qx": pose.qx, "qy": pose.qy, "qz": pose.qz, "qw": pose.qw
        }
        recent_trajectory = trajectory[-100:] if len(trajectory) > 0 else []

        payload = {
            "type": "frame_update",
            "seq": frame.seq,
            "stamp_ns": frame.stamp_ns,
            "num_points": N,
            "subsampled_points": len(pts_sub),
            "points": {
                "x": np.round(x, 2).tolist(),
                "y": np.round(y, 2).tolist(),
                "z": np.round(z, 2).tolist(),
                "label": l.astype(int).tolist()
            },
            "tracks": tracks_data,
            "grid_cells": grid_cells_data,
            "metrics": metrics_data,
            "pose": pose_data,
            "trajectory": recent_trajectory
        }

        return payload
