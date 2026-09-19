"""
Global configuration: semantic classes, colors, and the variable-resolution
tier schedule used by both the perception pipeline and the grid engine.
"""
import numpy as np

# ---------------------------------------------------------------------------
# Semantic classes
# ---------------------------------------------------------------------------
CLASSES = [
    "drivable_terrain",   # 0
    "non_drivable_terrain",  # 1
    "static_structure",   # 2  (wall / building)
    "static_pole",        # 3  (pole / thin obstacle)
    "dynamic_pedestrian",  # 4
    "dynamic_vehicle",     # 5
]
NUM_CLASSES = len(CLASSES)
DYNAMIC_CLASS_IDS = {4, 5}
STATIC_OBSTACLE_CLASS_IDS = {2, 3}
TERRAIN_CLASS_IDS = {0, 1}

CLASS_COLORS = {
    0: (60, 140, 60),      # drivable terrain - green
    1: (150, 150, 90),     # non-drivable terrain - olive
    2: (150, 60, 60),      # structure - dark red
    3: (200, 140, 30),     # pole - orange
    4: (230, 40, 200),     # pedestrian - magenta
    5: (40, 120, 230),     # vehicle - blue
}
UNKNOWN_COLOR = (35, 35, 40)      # low point-density / low-confidence cell
EMPTY_COLOR = (15, 15, 18)        # no data in cell at all

# ---------------------------------------------------------------------------
# Variable-resolution tier schedule (log-polar / range-ring grid)
# Ratios follow the reference architecture doc; radial cell sizes are exact
# multiples across tier boundaries so no fractional cell occurs at a seam.
# ---------------------------------------------------------------------------
TIERS = [
    # name,   r_min, r_max, dr(m), n_sectors
    dict(name="tier0", r_min=0.0,  r_max=10.0,  dr=0.05, n_sectors=360),
    dict(name="tier1", r_min=10.0, r_max=25.0,  dr=0.15, n_sectors=180),
    dict(name="tier2", r_min=25.0, r_max=50.0,  dr=0.30, n_sectors=90),
    dict(name="tier3", r_min=50.0, r_max=100.0, dr=0.50, n_sectors=45),
]

for _t in TIERS:
    _t["n_rings"] = int(round((_t["r_max"] - _t["r_min"]) / _t["dr"]))
    _t["dtheta"] = 2 * np.pi / _t["n_sectors"]
    _t["n_cells"] = _t["n_rings"] * _t["n_sectors"]

MAX_RANGE = TIERS[-1]["r_max"]
RANGE_BUCKETS = [(0, 10), (10, 25), (25, 50), (50, 100)]
DISPLAY_IMAGE_SIZE = 384

# Uniform-grid baseline used purely for the memory-savings comparison shown
# on the dashboard (same overall extent and radial cell size as the finest
# tier, applied everywhere out to MAX_RANGE).
UNIFORM_CELL_SIZE = TIERS[0]["dr"]

# Cell layers stored per cell (see mapping/grid_engine.py):
# 0 ground_height, 1 obstacle_top_height, 2 class_id, 3 confidence,
# 4 point_count
N_LAYERS = 5
LAYER_GROUND_H, LAYER_TOP_H, LAYER_CLASS, LAYER_CONF, LAYER_COUNT = range(5)

RNG_SEED = 42
