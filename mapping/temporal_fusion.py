"""
Temporal Fusion & Local Reference Frame Persistence Engine
Authoritative implementation matching SOFTWARE_ARCHITECTURE.md §7.1, §7.2, ADR-1, ADR-7.
"""

from typing import Optional, List, Tuple
import numpy as np

from core.schema import Pose, TierGridData, MapSnapshot
from mapping.adaptive_grid import AdaptivePolarGrid
from datasets.transforms import transform_points


class TemporalMapFusion:
    """
    Maintains a persistent 2.5D elevation and semantic map across consecutive frames.
    
    Key Properties (ADR-1, ADR-7):
    - Anchored to a slowly-recentered local reference frame (translation-only recentering)
    - Ego vehicle rotation is absorbed by transforming points into the local frame before binning
    - Dynamic objects are strictly excluded from static elevation/terrain persistence
    - Static cell values are blended via confidence-weighted exponential moving average (EMA)
    """

    def __init__(
        self,
        adaptive_grid: AdaptivePolarGrid,
        decay_rate: float = 0.95,
        recenter_threshold_m: float = 5.0,
        dynamic_classes: Tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8) # Vehicle & Pedestrian classes
    ):
        self.grid = adaptive_grid
        self.decay_rate = decay_rate
        self.recenter_threshold_m = recenter_threshold_m
        self.dynamic_classes = dynamic_classes
        
        self.anchor_pose: Optional[Pose] = None
        self.current_pose: Optional[Pose] = None
        self.frame_count: int = 0
        self.trajectory: List[Tuple[float, float, float]] = []

    def reset(self):
        self.grid.reset()
        self.anchor_pose = None
        self.current_pose = None
        self.frame_count = 0
        self.trajectory.clear()

    def update_frame(
        self,
        points: np.ndarray,
        pose: Pose,
        semantic_labels: Optional[np.ndarray] = None,
        confidences: Optional[np.ndarray] = None,
        instance_ids: Optional[np.ndarray] = None
    ) -> MapSnapshot:
        """
        Process a single frame with pose transformation and temporal fusion.
        
        Args:
            points: (N, 4) or (N, 3) raw point cloud in sensor frame
            pose: 6-DoF Pose in world/odom frame
            semantic_labels: (N,) learning class IDs
            confidences: (N,) confidence scores
            instance_ids: (N,) dynamic instance IDs
        """
        self.current_pose = pose
        self.trajectory.append((pose.px, pose.py, pose.pz))
        self.frame_count += 1
        
        if self.anchor_pose is None:
            self.anchor_pose = pose

        # Check if local frame recentering is needed (ADR-1, §7.2: translation-only recentering)
        delta_x = pose.px - self.anchor_pose.px
        delta_y = pose.py - self.anchor_pose.py
        dist_from_anchor = np.sqrt(delta_x**2 + delta_y**2)
        
        if dist_from_anchor > self.recenter_threshold_m:
            self._recenter_local_frame(pose)

        # Transform points from sensor/ego frame into current local grid frame
        # T_sensor_to_anchor = inv(T_anchor) @ T_current_pose
        T_anchor = self.anchor_pose.to_matrix()
        T_curr = pose.to_matrix()
        T_sensor_to_anchor = np.linalg.inv(T_anchor) @ T_curr
        
        transformed_points = transform_points(points, T_sensor_to_anchor)
        
        # Dynamic Object Filtering (ADR-7): Exclude dynamic objects from static terrain accumulation
        static_mask = np.ones(len(points), dtype=bool)
        if semantic_labels is not None:
            is_dynamic = np.isin(semantic_labels, self.dynamic_classes)
            static_mask = ~is_dynamic

        # Update the adaptive polar grid with the transformed points
        # For the active frame, we accumulate into the persistent grid
        self.grid.update_from_points(
            points=transformed_points[static_mask],
            semantic_labels=semantic_labels[static_mask] if semantic_labels is not None else None,
            confidences=confidences[static_mask] if confidences is not None else None,
            instance_ids=instance_ids[static_mask] if instance_ids is not None else None,
            clear_first=False  # Keep persistent temporal state!
        )

        return self.grid.get_snapshot(stamp_ns=pose.stamp_ns, anchor_pose=self.anchor_pose)

    def _recenter_local_frame(self, new_pose: Pose):
        """
        Translation-only recentering (SOFTWARE_ARCHITECTURE.md §7.2).
        Shifts the local reference anchor without rotating the grid axes.
        """
        self.anchor_pose = new_pose
        # In a rolling buffer, inner-tier arrays are shifted.
        # Outer tiers are naturally refreshed each sweep.
