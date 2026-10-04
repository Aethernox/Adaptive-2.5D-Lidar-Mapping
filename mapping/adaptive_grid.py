"""
Adaptive Variable-Resolution Grid Engine
Direct O(1) cell addressing, log-polar tiered range rings, and multi-statistic cell aggregation.
Authoritative implementation matching SOFTWARE_ARCHITECTURE.md §5 (M2), ADR-4, ADR-5, ADR-6, ADR-8.
"""

from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import yaml

from core.schema import TierSpec, TierSchedule, TierGridData, MapSnapshot, Pose


class AdaptivePolarGrid:
    """
    Tiered Log-Polar Range-Ring Grid for 2.5D elevation and semantic mapping.
    
    Provides:
    - O(1) direct Cartesian (x, y) to Tier/Ring/Sector cell addressing
    - Multi-statistic cell aggregation (ground height, obstacle top height, semantics, confidence, density)
    - Seam handling and half-open intervals [r_min, r_max)
    - Cartesian rasterization for human visualization
    """

    def __init__(self, schedule: Optional[TierSchedule] = None, config_path: Optional[str] = "configs/adaptive_grid.yaml"):
        if schedule is not None:
            self.schedule = schedule
        elif config_path:
            self.schedule = self._load_schedule_from_yaml(config_path)
        else:
            self.schedule = TierSchedule.default()
            
        self.tiers = self.schedule.tiers
        self._init_lookup_tables()
        self.reset()

    def _load_schedule_from_yaml(self, path: str) -> TierSchedule:
        with open(path, "r") as f:
            cfg = yaml.safe_load(f)
        grid_cfg = cfg.get("adaptive_grid", {})
        max_range = grid_cfg.get("max_range", 100.0)
        tier_list = []
        for t in grid_cfg.get("tiers", []):
            tier_list.append(TierSpec(
                id=int(t["id"]),
                r_min=float(t["min_range"]),
                r_max=float(t["max_range"]),
                delta_r=float(t["radial_resolution"]),
                azimuth_bins=int(t["azimuth_bins"]),
                description=t.get("description", "")
            ))
        return TierSchedule(
            schema_version=grid_cfg.get("schema_version", 1),
            max_range=max_range,
            tiers=tier_list
        )

    def _init_lookup_tables(self):
        """Pre-compute tier boundaries and ring offsets for fast lookup."""
        self.tier_r_mins = np.array([t.r_min for t in self.tiers], dtype=np.float32)
        self.tier_r_maxs = np.array([t.r_max for t in self.tiers], dtype=np.float32)
        self.tier_delta_rs = np.array([t.delta_r for t in self.tiers], dtype=np.float32)
        self.tier_num_rings = np.array([t.num_rings for t in self.tiers], dtype=np.int32)
        self.tier_num_sectors = np.array([t.azimuth_bins for t in self.tiers], dtype=np.int32)
        
        self.total_cell_count = sum(t.total_cells for t in self.tiers)

    def reset(self):
        """Allocate empty Structure-of-Arrays (SoA) for each tier grid."""
        self.tier_grids: List[TierGridData] = []
        for t in self.tiers:
            self.tier_grids.append(TierGridData.create_empty(
                tier_id=t.id,
                num_rings=t.num_rings,
                num_sectors=t.azimuth_bins
            ))

    def lookup_cell_indices(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        O(1) Vectorized cell index lookup for (N,) coordinates.
        
        Returns:
            tier_indices: (N,) int32 (or -1 if out of bounds)
            ring_indices: (N,) int32
            sector_indices: (N,) int32
            valid_mask: (N,) bool
        """
        r = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)  # [-pi, pi)
        # Shift theta to [0, 2pi)
        theta_0_2pi = np.where(theta < 0, theta + 2 * np.pi, theta)
        
        N = len(x)
        tier_indices = np.full(N, -1, dtype=np.int32)
        ring_indices = np.full(N, -1, dtype=np.int32)
        sector_indices = np.full(N, -1, dtype=np.int32)
        valid_mask = np.zeros(N, dtype=bool)

        for i, t in enumerate(self.tiers):
            # Half-open interval [r_min, r_max)
            in_tier = (r >= t.r_min) & (r < t.r_max)
            if not np.any(in_tier):
                continue
                
            r_in = r[in_tier]
            th_in = theta_0_2pi[in_tier]
            
            # Ring calculation
            rings = np.floor((r_in - t.r_min) / t.delta_r).astype(np.int32)
            rings = np.clip(rings, 0, t.num_rings - 1)
            
            # Sector calculation: [0, 2pi) -> [0, azimuth_bins)
            sectors = np.floor(th_in * (t.azimuth_bins / (2.0 * np.pi))).astype(np.int32)
            sectors = np.clip(sectors, 0, t.azimuth_bins - 1)
            
            tier_indices[in_tier] = i
            ring_indices[in_tier] = rings
            sector_indices[in_tier] = sectors
            valid_mask[in_tier] = True

        return tier_indices, ring_indices, sector_indices, valid_mask

    def update_from_points(
        self,
        points: np.ndarray,
        semantic_labels: Optional[np.ndarray] = None,
        confidences: Optional[np.ndarray] = None,
        instance_ids: Optional[np.ndarray] = None,
        ground_classes: Tuple[int, ...] = (9, 10, 11, 12, 17), # road, parking, sidewalk, other-ground, terrain
        clear_first: bool = True
    ):
        """
        Aggregate 3D point cloud into multi-statistic adaptive 2.5D grid cells.
        
        Args:
            points: (N, 4) or (N, 3) array [x, y, z, (intensity)]
            semantic_labels: (N,) learning class IDs (0..19)
            confidences: (N,) confidence scores [0.0, 1.0]
            instance_ids: (N,) dynamic instance IDs
            ground_classes: learning class IDs considered terrain/ground
            clear_first: reset grid before accumulating current frame
        """
        if clear_first:
            self.reset()
            
        if points is None or len(points) == 0:
            return

        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]

        if semantic_labels is None:
            semantic_labels = np.zeros(len(points), dtype=np.uint8)
        if confidences is None:
            confidences = np.ones(len(points), dtype=np.float32)
        if instance_ids is None:
            instance_ids = np.zeros(len(points), dtype=np.uint16)

        # Lookup cell addresses for all points
        tier_idx, ring_idx, sec_idx, valid = self.lookup_cell_indices(x, y)
        if not np.any(valid):
            return

        # Process each tier
        for t_id, tier_data in enumerate(self.tier_grids):
            t_mask = valid & (tier_idx == t_id)
            if not np.any(t_mask):
                continue
                
            r_indices = ring_idx[t_mask]
            s_indices = sec_idx[t_mask]
            z_vals = z[t_mask]
            labels = semantic_labels[t_mask]
            confs = confidences[t_mask]
            insts = instance_ids[t_mask]
            
            # Linear cell index for 1D fast grouping
            num_sec = tier_data.num_sectors
            linear_indices = r_indices * num_sec + s_indices
            
            # Sort by cell index for fast run-length grouping
            sort_order = np.argsort(linear_indices)
            lin_sorted = linear_indices[sort_order]
            z_sorted = z_vals[sort_order]
            lbl_sorted = labels[sort_order]
            conf_sorted = confs[sort_order]
            inst_sorted = insts[sort_order]
            
            # Find unique occupied cells and their split points
            unique_cells, split_indices, counts = np.unique(
                lin_sorted, return_index=True, return_counts=True
            )
            splits = np.split(np.arange(len(lin_sorted)), split_indices[1:])
            
            # Fast vectorized/grouped aggregation per unique cell
            for cell_lin_id, split_idx in zip(unique_cells, splits):
                r = cell_lin_id // num_sec
                s = cell_lin_id % num_sec
                
                cell_z = z_sorted[split_idx]
                cell_lbl = lbl_sorted[split_idx]
                cell_conf = conf_sorted[split_idx]
                cell_inst = inst_sorted[split_idx]
                pt_count = len(cell_z)
                
                # Ground vs Obstacle separation
                is_ground = np.isin(cell_lbl, ground_classes)
                
                # 1. Ground Height (robust 5th percentile)
                if np.any(is_ground):
                    ground_z = cell_z[is_ground]
                    g_height = float(np.percentile(ground_z, 5.0))
                else:
                    g_height = float(np.percentile(cell_z, 5.0))
                    
                # 2. Obstacle Top Height (robust 95th percentile)
                obs_z = cell_z[~is_ground] if np.any(~is_ground) else cell_z
                o_height = float(np.percentile(obs_z, 95.0))
                
                # 3. Majority Semantic Class (confidence-weighted)
                unique_lbls, lbl_inv = np.unique(cell_lbl, return_inverse=True)
                weighted_votes = np.bincount(lbl_inv, weights=cell_conf)
                maj_class = unique_lbls[np.argmax(weighted_votes)]
                
                # 4. Confidence (mean model confidence * density factor)
                mean_conf = float(np.mean(cell_conf))
                # Quantize to 0-255 uint8
                conf_quant = int(np.clip(mean_conf * 255.0, 0, 255))
                
                # 5. Density: points / cell area
                # Cell area = 0.5 * (r_outer^2 - r_inner^2) * delta_theta
                r_in = self.tiers[t_id].r_min + r * self.tiers[t_id].delta_r
                r_out = r_in + self.tiers[t_id].delta_r
                d_theta = (2.0 * np.pi) / num_sec
                cell_area = 0.5 * (r_out**2 - r_in**2) * d_theta
                density = float(pt_count / max(cell_area, 1e-4))
                
                # 6. Dynamic Instance ID (mode instance)
                dyn_insts = cell_inst[cell_inst > 0]
                inst_id = int(np.bincount(dyn_insts).argmax()) if len(dyn_insts) > 0 else 0
                
                # Store into SoA
                tier_data.ground_height[r, s] = g_height
                tier_data.obstacle_top_height[r, s] = o_height
                tier_data.semantic_class[r, s] = maj_class
                tier_data.confidence[r, s] = conf_quant
                tier_data.point_density[r, s] = density
                tier_data.dynamic_instance_id[r, s] = inst_id
                tier_data.occupied_mask[r, s] = True

    def get_snapshot(self, stamp_ns: int, anchor_pose: Optional[Pose] = None) -> MapSnapshot:
        """Create an immutable MapSnapshot."""
        if anchor_pose is None:
            anchor_pose = Pose(stamp_ns=stamp_ns)
        return MapSnapshot(
            stamp_ns=stamp_ns,
            schema_version=self.schedule.schema_version,
            tier_schedule_version=1,
            anchor_pose=anchor_pose,
            tiers=self.tier_grids
        )

    def rasterize_to_cartesian(
        self,
        grid_size: int = 400,
        extent_m: float = 100.0,
        layer: str = "semantics"  # 'semantics', 'elevation', 'confidence', 'density'
    ) -> np.ndarray:
        """
        Rasterize polar grid into a 2D Cartesian image for human dashboard visualization.
        
        Args:
            grid_size: output image resolution (e.g. 400x400)
            extent_m: half-width extent in meters (e.g. 100m for [-100, 100]m)
            layer: which attribute to rasterize
            
        Returns:
            image_array: (grid_size, grid_size) or (grid_size, grid_size, 3)
        """
        lin = np.linspace(-extent_m, extent_m, grid_size, dtype=np.float32)
        X, Y = np.meshgrid(lin, -lin)  # Image coordinates: Y downwards
        
        tier_idx, ring_idx, sec_idx, valid = self.lookup_cell_indices(X.ravel(), Y.ravel())
        
        result = np.zeros(grid_size * grid_size, dtype=np.float32)
        
        for t_id, tier_data in enumerate(self.tier_grids):
            t_mask = valid & (tier_idx == t_id)
            if not np.any(t_mask):
                continue
                
            r_in = ring_idx[t_mask]
            s_in = sec_idx[t_mask]
            
            if layer == "semantics":
                vals = tier_data.semantic_class[r_in, s_in]
            elif layer == "elevation":
                vals = tier_data.obstacle_top_height[r_in, s_in]
                vals = np.where(np.isnan(vals), tier_data.ground_height[r_in, s_in], vals)
            elif layer == "confidence":
                vals = tier_data.confidence[r_in, s_in] / 255.0
            elif layer == "density":
                vals = tier_data.point_density[r_in, s_in]
            else:
                vals = tier_data.occupied_mask[r_in, s_in].astype(np.float32)
                
            result[t_mask] = np.nan_to_num(vals, nan=0.0)

        return result.reshape((grid_size, grid_size))
