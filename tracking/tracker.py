"""
Multi-Object LiDAR Dynamic Tracker
Authoritative implementation matching SOFTWARE_ARCHITECTURE.md §5 (M3) and ADR-7.
"""

from typing import List, Dict, Tuple, Optional
import numpy as np

from core.schema import Track, TrackSet
from tracking.kalman import KalmanBoxTracker
from tracking.association import associate_detections_to_trackers


class MultiObjectTracker:
    """
    Tracks dynamic objects across frames using 3D Kalman filtering and association.
    Manages track confirmation, velocity estimation, and sparse dynamic overlay.
    """

    def __init__(
        self,
        max_age: int = 5,
        min_hits: int = 3,
        distance_threshold: float = 2.5,
        min_moving_speed: float = 0.5, # m/s (1.8 km/h)
        dynamic_classes: Tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8)
    ):
        self.max_age = max_age
        self.min_hits = min_hits
        self.distance_threshold = distance_threshold
        self.min_moving_speed = min_moving_speed
        self.dynamic_classes = dynamic_classes
        
        self.trackers: List[KalmanBoxTracker] = []
        self.frame_count: int = 0

    def reset(self):
        self.trackers.clear()
        self.frame_count = 0
        KalmanBoxTracker.count = 0

    def extract_clusters_from_points(
        self,
        points: np.ndarray,
        semantic_labels: np.ndarray,
        instance_ids: Optional[np.ndarray] = None
    ) -> List[Dict]:
        """
        Extract 3D oriented bounding boxes from semantic dynamic points or instance IDs.
        
        Returns:
            detections: List of dicts with keys 'bbox_3d' [x, y, z, l, w, h, yaw], 'class_id', 'points'
        """
        detections = []
        if points is None or len(points) == 0 or semantic_labels is None:
            return detections

        dynamic_mask = np.isin(semantic_labels, self.dynamic_classes)
        if not np.any(dynamic_mask):
            return detections

        dyn_pts = points[dynamic_mask]
        dyn_lbls = semantic_labels[dynamic_mask]
        dyn_insts = instance_ids[dynamic_mask] if instance_ids is not None else np.zeros(len(dyn_pts), dtype=np.uint16)

        # 1. If instance IDs are present and distinct (> 0), group by instance ID
        unique_insts = np.unique(dyn_insts)
        has_real_instances = len(unique_insts[unique_insts > 0]) > 0

        if has_real_instances:
            for inst_id in unique_insts:
                if inst_id == 0:
                    continue
                mask_i = (dyn_insts == inst_id)
                cluster_pts = dyn_pts[mask_i]
                if len(cluster_pts) < 15: # Filter tiny noise clusters
                    continue
                
                class_id = int(np.bincount(dyn_lbls[mask_i]).argmax())
                bbox = self._compute_bbox_3d(cluster_pts)
                detections.append({
                    "bbox_3d": bbox,
                    "class_id": class_id,
                    "num_points": len(cluster_pts)
                })
        else:
            # 2. Fast spatial Euclidean clustering by class
            for cls_id in self.dynamic_classes:
                cls_mask = (dyn_lbls == cls_id)
                if not np.any(cls_mask):
                    continue
                pts_cls = dyn_pts[cls_mask]
                if len(pts_cls) < 15:
                    continue
                
                # Grid-based fast connected components
                clusters = self._fast_spatial_cluster(pts_cls, cluster_dist=1.5)
                for c_pts in clusters:
                    if len(c_pts) >= 15:
                        bbox = self._compute_bbox_3d(c_pts)
                        detections.append({
                            "bbox_3d": bbox,
                            "class_id": int(cls_id),
                            "num_points": len(c_pts)
                        })

        return detections

    def _compute_bbox_3d(self, pts: np.ndarray) -> np.ndarray:
        """Compute axis-aligned / robust 3D bounding box [cx, cy, cz, l, w, h, yaw]."""
        min_xyz = np.min(pts[:, :3], axis=0)
        max_xyz = np.max(pts[:, :3], axis=0)
        
        center = (min_xyz + max_xyz) / 2.0
        extents = np.maximum(max_xyz - min_xyz, 0.2) # Minimum 20cm extent
        
        # Heading estimation from PCA in XY plane
        xy_centered = pts[:, :2] - center[:2]
        if len(xy_centered) > 3:
            cov = np.cov(xy_centered.T)
            eigenvals, eigenvecs = np.linalg.eigh(cov)
            principal_axis = eigenvecs[:, -1]
            yaw = float(np.arctan2(principal_axis[1], principal_axis[0]))
        else:
            yaw = 0.0

        return np.array([
            center[0], center[1], center[2],
            extents[0], extents[1], extents[2],
            yaw
        ], dtype=np.float32)

    def _fast_spatial_cluster(self, pts: np.ndarray, cluster_dist: float = 1.5) -> List[np.ndarray]:
        """Fast grid hashing clustering for spatial point groupings."""
        voxel_size = cluster_dist
        coords = np.floor(pts[:, :3] / voxel_size).astype(np.int64)
        
        # Hash 3D coords to 1D keys
        keys = (coords[:, 0] * 73856093) ^ (coords[:, 1] * 19349663) ^ (coords[:, 2] * 83492791)
        unique_keys, split_idx = np.unique(keys, return_index=True)
        splits = np.split(np.arange(len(keys)), split_idx[1:])
        
        return [pts[s] for s in splits if len(s) >= 10]

    def update(
        self,
        points: np.ndarray,
        semantic_labels: np.ndarray,
        instance_ids: Optional[np.ndarray] = None,
        stamp_ns: int = 0
    ) -> TrackSet:
        """
        Process current frame detections, update Kalman filters, and return active TrackSet.
        """
        self.frame_count += 1
        
        # 1. Extract 3D detections
        detections = self.extract_clusters_from_points(points, semantic_labels, instance_ids)
        
        # 2. Predict next state for existing trackers
        predicted_boxes = []
        for trk in self.trackers:
            predicted_boxes.append(trk.predict())
        predicted_boxes = np.array(predicted_boxes) if len(predicted_boxes) > 0 else np.empty((0, 7))

        # 3. Associate detections to trackers
        det_boxes = np.array([d["bbox_3d"] for d in detections]) if len(detections) > 0 else np.empty((0, 7))
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(
            det_boxes, predicted_boxes, self.distance_threshold
        )

        # 4. Update matched trackers
        for det_idx, trk_idx in matched:
            self.trackers[trk_idx].update(det_boxes[det_idx])
            self.trackers[trk_idx].class_id = detections[det_idx]["class_id"]

        # 5. Create new trackers for unmatched detections
        for det_idx in unmatched_dets:
            new_trk = KalmanBoxTracker(
                bbox_3d=det_boxes[det_idx],
                class_id=detections[det_idx]["class_id"]
            )
            self.trackers.append(new_trk)

        # 6. Cull dead trackers and assemble active TrackSet
        active_tracks: List[Track] = []
        surviving_trackers: List[KalmanBoxTracker] = []

        for trk in self.trackers:
            if trk.time_since_update <= self.max_age:
                surviving_trackers.append(trk)
                
                # Confirm track if seen sufficiently many times
                if trk.hits >= self.min_hits or self.frame_count <= self.min_hits:
                    state = trk.get_state()
                    vx, vy, vz = trk.velocity
                    speed = trk.speed
                    is_moving = (speed >= self.min_moving_speed)
                    
                    active_tracks.append(Track(
                        id=trk.id,
                        class_id=trk.class_id,
                        cx=float(state[0]), cy=float(state[1]), cz=float(state[2]),
                        length=float(state[3]), width=float(state[4]), height=float(state[5]),
                        heading=float(state[6]),
                        vx=vx, vy=vy,
                        confidence=min(1.0, 0.5 + trk.hits * 0.1),
                        age_frames=trk.age,
                        misses=trk.time_since_update,
                        is_moving=is_moving
                    ))

        self.trackers = surviving_trackers
        return TrackSet(stamp_ns=stamp_ns, tracks=active_tracks)
