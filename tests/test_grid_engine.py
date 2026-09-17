import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import TIERS, MAX_RANGE, NUM_CLASSES
from mapping.grid_engine import VariableResolutionGrid


def test_locate_o1_addressing_matches_tier_bounds():
    grid = VariableResolutionGrid()
    # A point at r=5 should land in tier0 (0-10m)
    tier_idx, ring, sector = grid.locate(np.array([5.0]), np.array([0.0]))
    assert tier_idx[0] == 0
    assert 0 <= ring[0] < TIERS[0]["n_rings"]
    assert 0 <= sector[0] < TIERS[0]["n_sectors"]

    # A point at r=30 should land in tier2 (25-50m)
    tier_idx, ring, sector = grid.locate(np.array([30.0]), np.array([0.0]))
    assert tier_idx[0] == 2


def test_locate_out_of_range_is_dropped():
    grid = VariableResolutionGrid()
    tier_idx, _, _ = grid.locate(np.array([150.0]), np.array([0.0]))
    assert tier_idx[0] == -1


def test_locate_no_seam_gaps_at_tier_boundary():
    """Points just inside/outside a tier boundary must not both be dropped
    or double counted -- exactly one tier claims each range."""
    grid = VariableResolutionGrid()
    boundary = TIERS[0]["r_max"]
    just_inside = boundary - 1e-6
    just_outside = boundary + 1e-6
    t_in, _, _ = grid.locate(np.array([just_inside]), np.array([0.0]))
    t_out, _, _ = grid.locate(np.array([just_outside]), np.array([0.0]))
    assert t_in[0] == 0
    assert t_out[0] == 1


def test_update_populates_cells_and_no_crash_on_empty_frame():
    grid = VariableResolutionGrid()
    grid.update(np.zeros((0, 3)), np.zeros((0,), dtype=int), np.zeros((0,)))
    pts = np.array([[3.0, 0.0, 0.02], [3.0, 0.0, 0.02]])
    cls = np.array([0, 0])
    conf = np.array([0.9, 0.9])
    grid.update(pts, cls, conf)
    stats = grid.cell_stats_at(3.0, 0.0)
    assert stats is not None
    assert stats["point_count"] >= 1
    assert stats["class_id"] == 0


def test_memory_stats_reduction_factor_over_50x():
    grid = VariableResolutionGrid()
    stats = grid.memory_stats()
    assert stats["adaptive_cells"] < stats["uniform_cells"]
    assert stats["reduction_factor"] > 50  # headline claim from the design doc


def test_cell_class_layer_within_valid_range():
    grid = VariableResolutionGrid()
    pts = np.random.uniform(-20, 20, size=(500, 3))
    pts[:, 2] = 0.0
    cls = np.random.randint(0, NUM_CLASSES, size=500)
    conf = np.random.uniform(0.5, 1.0, size=500)
    grid.update(pts, cls, conf)
    for tier_cells in grid.cells:
        assert np.all(tier_cells[..., 2] >= 0)
        assert np.all(tier_cells[..., 2] < NUM_CLASSES)
