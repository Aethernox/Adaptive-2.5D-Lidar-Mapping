"""
Core Data Contracts and Message Schemas
Authoritative implementation matching SOFTWARE_ARCHITECTURE.md §6 and ADR-6.
Process-agnostic, typed, and serialized for zero-drift cross-module communication.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
import numpy as np
import json


@dataclass
class PointCloudFrame:
    """Raw point cloud frame received from sensor or replay engine."""
    seq: int                     # Monotonic per-sensor sequence number
    stamp_ns: int                # Timestamp in nanoseconds
    frame_id: str                # Reference frame (e.g. 'velodyne', 'base_link')
    points: np.ndarray           # Shape (N, 4): [x, y, z, intensity] float32
    raw_labels: Optional[np.ndarray] = None  # Shape (N,): raw uint32 (lower 16=semantic, upper 16=instance)
    
    @property
    def num_points(self) -> int:
        return len(self.points) if self.points is not None else 0

    @property
    def semantic_labels(self) -> Optional[np.ndarray]:
        if self.raw_labels is None:
            return None
        return (self.raw_labels & 0xFFFF).astype(np.uint16)

    @property
    def instance_ids(self) -> Optional[np.ndarray]:
        if self.raw_labels is None:
            return None
        return (self.raw_labels >> 16).astype(np.uint16)


@dataclass
class Pose:
    """Ego vehicle 6-DoF pose at a specific timestamp."""
    stamp_ns: int
    reference_frame: str = "odom"
    px: float = 0.0
    py: float = 0.0
    pz: float = 0.0
    qx: float = 0.0
    qy: float = 0.0
    qz: float = 0.0
    qw: float = 1.0
    position_cov_trace: float = 0.0  # Confidence proxy; high trace flags pose as unreliable
    matrix: Optional[np.ndarray] = None  # 4x4 homogeneous transformation matrix

    def to_matrix(self) -> np.ndarray:
        """Return or construct 4x4 homogeneous transformation matrix."""
        if self.matrix is not None:
            return self.matrix
        
        # Convert quaternion to rotation matrix
        x, y, z, w = self.qx, self.qy, self.qz, self.qw
        R = np.array([
            [1 - 2*(y**2 + z**2), 2*(x*y - z*w),     2*(x*z + y*w)],
            [2*(x*y + z*w),     1 - 2*(x**2 + z**2), 2*(y*z - x*w)],
            [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x**2 + y**2)]
        ], dtype=np.float64)
        
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = [self.px, self.py, self.pz]
        return T

    @classmethod
    def from_matrix(cls, T: np.ndarray, stamp_ns: int = 0, reference_frame: str = "odom") -> 'Pose':
        """Construct Pose from 4x4 homogeneous matrix."""
        px, py, pz = float(T[0, 3]), float(T[1, 3]), float(T[2, 3])
        R = T[:3, :3]
        
        # Quaternion from rotation matrix
        trace = np.trace(R)
        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            qw = 0.25 / s
            qx = (R[2, 1] - R[1, 2]) * s
            qy = (R[0, 2] - R[2, 0]) * s
            qz = (R[1, 0] - R[0, 1]) * s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            qw = (R[2, 1] - R[1, 2]) / s
            qx = 0.25 * s
            qy = (R[0, 1] + R[1, 0]) / s
            qz = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            qw = (R[0, 2] - R[2, 0]) / s
            qx = (R[0, 1] + R[1, 0]) / s
            qy = 0.25 * s
            qz = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            qw = (R[1, 0] - R[0, 1]) / s
            qx = (R[0, 2] + R[2, 0]) / s
            qy = (R[1, 2] + R[2, 1]) / s
            qz = 0.25 * s

        return cls(
            stamp_ns=stamp_ns,
            reference_frame=reference_frame,
            px=px, py=py, pz=pz,
            qx=float(qx), qy=float(qy), qz=float(qz), qw=float(qw),
            matrix=T
        )


@dataclass
class TierSpec:
    """Specification of a single resolution tier."""
    id: int
    r_min: float
    r_max: float
    delta_r: float
    azimuth_bins: int
    description: str = ""

    @property
    def num_rings(self) -> int:
        return int(np.round((self.r_max - self.r_min) / self.delta_r))

    @property
    def total_cells(self) -> int:
        return self.num_rings * self.azimuth_bins


@dataclass
class TierSchedule:
    """Hot-reloadable tier configuration schedule."""
    schema_version: int = 1
    max_range: float = 100.0
    tiers: List[TierSpec] = field(default_factory=list)

    @property
    def total_cells(self) -> int:
        return sum(t.total_cells for t in self.tiers)

    @classmethod
    def default(cls) -> 'TierSchedule':
        return cls(
            schema_version=1,
            max_range=100.0,
            tiers=[
                TierSpec(id=0, r_min=0.0, r_max=10.0, delta_r=0.05, azimuth_bins=1024, description="Tier 0: 0-10m"),
                TierSpec(id=1, r_min=10.0, r_max=25.0, delta_r=0.15, azimuth_bins=512, description="Tier 1: 10-25m"),
                TierSpec(id=2, r_min=25.0, r_max=50.0, delta_r=0.30, azimuth_bins=256, description="Tier 2: 25-50m"),
                TierSpec(id=3, r_min=50.0, r_max=100.0, delta_r=0.50, azimuth_bins=128, description="Tier 3: 50-100m"),
            ]
        )


@dataclass
class TierGridData:
    """Structure-of-Arrays (SoA) layout for a single tier grid."""
    tier_id: int
    num_rings: int
    num_sectors: int
    ground_height: np.ndarray      # Shape (num_rings, num_sectors) float32
    obstacle_top_height: np.ndarray# Shape (num_rings, num_sectors) float32
    semantic_class: np.ndarray     # Shape (num_rings, num_sectors) uint8
    confidence: np.ndarray         # Shape (num_rings, num_sectors) uint8 (0-255)
    point_density: np.ndarray      # Shape (num_rings, num_sectors) float32
    dynamic_instance_id: np.ndarray# Shape (num_rings, num_sectors) uint16
    occupied_mask: np.ndarray      # Shape (num_rings, num_sectors) bool

    @classmethod
    def create_empty(cls, tier_id: int, num_rings: int, num_sectors: int) -> 'TierGridData':
        return cls(
            tier_id=tier_id,
            num_rings=num_rings,
            num_sectors=num_sectors,
            ground_height=np.full((num_rings, num_sectors), np.nan, dtype=np.float32),
            obstacle_top_height=np.full((num_rings, num_sectors), np.nan, dtype=np.float32),
            semantic_class=np.zeros((num_rings, num_sectors), dtype=np.uint8),
            confidence=np.zeros((num_rings, num_sectors), dtype=np.uint8),
            point_density=np.zeros((num_rings, num_sectors), dtype=np.float32),
            dynamic_instance_id=np.zeros((num_rings, num_sectors), dtype=np.uint16),
            occupied_mask=np.zeros((num_rings, num_sectors), dtype=bool)
        )


@dataclass
class Track:
    """Tracked dynamic object representation."""
    id: int
    class_id: int
    cx: float; cy: float; cz: float        # Centroid coordinates in meters
    length: float; width: float; height: float # Bounding box extents
    heading: float                         # Yaw angle in radians
    vx: float = 0.0; vy: float = 0.0       # Velocity estimate in m/s
    confidence: float = 1.0
    age_frames: int = 1
    misses: int = 0
    is_moving: bool = False                # Confirmed moving vs stationary vehicle


@dataclass
class TrackSet:
    """Set of all active tracks at current frame."""
    stamp_ns: int
    tracks: List[Track] = field(default_factory=list)


@dataclass
class MapSnapshot:
    """Published immutable snapshot of the 2.5D persistent grid."""
    stamp_ns: int
    schema_version: int
    tier_schedule_version: int
    anchor_pose: Pose
    tiers: List[TierGridData]
    unanchored: bool = False
    stale: bool = False


@dataclass
class SystemMetrics:
    """Real-time latency, memory, and performance metrics."""
    seq: int
    fps: float
    load_ms: float
    preprocess_ms: float
    inference_ms: float
    mapping_ms: float
    fusion_ms: float
    tracking_ms: float
    total_ms: float
    memory_adaptive_mb: float
    memory_uniform_mb: float
    memory_reduction_pct: float
    num_points: int
    num_adaptive_cells: int
    num_uniform_cells: int
    active_tracks: int
    range_iou: Dict[str, float] = field(default_factory=dict)
