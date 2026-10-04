"""
Uniform Cartesian 2.5D Grid Baseline Engine
Used for comparative memory, cell count, latency, and FPS benchmarking.
Matches Section 15 & SOFTWARE_ARCHITECTURE.md §11.
"""

from typing import Dict, Tuple, Optional
import numpy as np


class UniformGridBaseline:
    """
    Fixed-resolution uniform Cartesian 2.5D grid baseline.
    Default: 200m x 200m area [-100, 100]m at 0.05m (5cm) resolution.
    Creates 4000 x 4000 = 16,000,000 cells.
    """

    def __init__(
        self,
        x_range: Tuple[float, float] = (-100.0, 100.0),
        y_range: Tuple[float, float] = (-100.0, 100.0),
        resolution: float = 0.05,  # 5 cm uniform cells
        bytes_per_cell: int = 16
    ):
        self.x_min, self.x_max = x_range
        self.y_min, self.y_max = y_range
        self.resolution = resolution
        self.bytes_per_cell = bytes_per_cell

        self.width = int(np.round((self.x_max - self.x_min) / self.resolution))
        self.height = int(np.round((self.y_max - self.y_min) / self.resolution))
        self.total_cell_count = self.width * self.height
        
        # Theoretical and actual memory calculations
        self.theoretical_memory_bytes = self.total_cell_count * self.bytes_per_cell
        self.theoretical_memory_mb = self.theoretical_memory_bytes / (1024.0 * 1024.0)
        
        # Note: allocating a full 16M array of 16-byte structs takes ~256MB.
        # For fast execution in benchmarking, we provide both dense allocation and sparse indexed aggregation.
        self.reset()

    def reset(self):
        # Structure-of-arrays representation or empty tracker
        self.occupied_indices = set()
        self.ground_height = None
        self.obstacle_top_height = None
        self.semantic_class = None

    def lookup_cell_indices(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Convert Cartesian (x, y) to (ix, iy) cell indices."""
        ix = np.floor((x - self.x_min) / self.resolution).astype(np.int32)
        iy = np.floor((y - self.y_min) / self.resolution).astype(np.int32)
        
        valid = (ix >= 0) & (ix < self.width) & (iy >= 0) & (iy < self.height)
        ix = np.clip(ix, 0, self.width - 1)
        iy = np.clip(iy, 0, self.height - 1)
        return ix, iy, valid

    def update_from_points(
        self,
        points: np.ndarray,
        semantic_labels: Optional[np.ndarray] = None,
        confidences: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Aggregate points into the uniform grid and return profiling stats.
        """
        if points is None or len(points) == 0:
            return {"occupied_cells": 0, "total_cells": self.total_cell_count}

        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]

        ix, iy, valid = self.lookup_cell_indices(x, y)
        if not np.any(valid):
            return {"occupied_cells": 0, "total_cells": self.total_cell_count}

        ix_v = ix[valid]
        iy_v = iy[valid]
        linear_idx = iy_v * self.width + ix_v
        
        unique_cells = np.unique(linear_idx)
        num_occupied = len(unique_cells)
        
        return {
            "occupied_cells": num_occupied,
            "total_cells": self.total_cell_count,
            "memory_mb": self.theoretical_memory_mb
        }
