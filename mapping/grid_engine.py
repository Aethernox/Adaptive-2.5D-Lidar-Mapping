"""
Variable-Resolution Grid Engine (Module 2 of the reference architecture).

Implements the tiered log-polar ("range-ring") grid: cell address for a
point (x, y) in the ego frame is computed in O(1) — a tier lookup + two
floor-divisions, no tree traversal. Ring width grows tier-by-tier with
range so "high resolution near, low resolution far" is a property of the
coordinate system itself.

Each cell stores multiple statistics (not a single collapsed height), so
overhangs and potholes are not lost the way a naive max-z 2.5D map would
lose them:
    - ground_height     robust min z of terrain-labeled points in the cell
    - obstacle_top_h     max z of non-terrain points in the cell
    - class_id           confidence-weighted majority vote
    - confidence         point_count / expected_count_at_this_range
    - point_count        raw count (also drives the confidence layer)

Dynamic objects are *not* densified into the grid — they live in a sparse
object list (see mapping/tracker.py) — so the memory win from the tiered
grid is not spent re-storing a handful of moving boxes into every cell they
touch.
"""
import numpy as np
from config import (TIERS, N_LAYERS, LAYER_GROUND_H, LAYER_TOP_H, LAYER_CLASS,
                     LAYER_CONF, LAYER_COUNT, NUM_CLASSES, TERRAIN_CLASS_IDS,
                     UNIFORM_CELL_SIZE, MAX_RANGE)

# Expected point count per cell at tier's mid-range, used to normalize the
# confidence/point-density layer (a coarse but adequate heuristic: the
# simulator's point density falls roughly as 1/(1+(r/18)^2), see sim/lidar).
_EXPECTED_DENSITY_PER_M2 = 6.0
_TIER_MAX_RANGES = np.array([t["r_max"] for t in TIERS], dtype=np.float32)


class VariableResolutionGrid:
    def __init__(self):
        self.tiers = TIERS
        # One (n_rings, n_sectors, N_LAYERS) float32 array per tier.
        self.cells = [np.zeros((t["n_rings"], t["n_sectors"], N_LAYERS), dtype=np.float32)
                      for t in self.tiers]
        # class-vote accumulator per cell, reset each frame, used to compute
        # the confidence-weighted majority vote before collapsing to class_id
        self._class_votes = [np.zeros((t["n_rings"], t["n_sectors"], NUM_CLASSES), dtype=np.float32)
                              for t in self.tiers]
        self._expected_counts = []
        for t in self.tiers:
            ring_mid = t["r_min"] + (np.arange(t["n_rings"], dtype=np.float32) + 0.5) * t["dr"]
            cell_area = ring_mid * t["dr"] * t["dtheta"]
            expected = np.maximum(cell_area * _EXPECTED_DENSITY_PER_M2, 1e-3)
            self._expected_counts.append(np.repeat(expected, t["n_sectors"]))
        self._ema_alpha = 0.06  # confidence decay rate for cells with no return this frame

    # ------------------------------------------------------------------
    # O(1) cell addressing
    # ------------------------------------------------------------------
    @staticmethod
    def locate(x, y):
        """Vectorized point(s) (x,y) in ego frame -> (tier_idx, ring, sector).
        tier_idx == -1 marks points outside MAX_RANGE (dropped)."""
        r = np.hypot(x, y)
        theta = np.mod(np.arctan2(y, x), 2 * np.pi)
        tier_idx = np.searchsorted(_TIER_MAX_RANGES, r, side="right").astype(np.int32)
        tier_idx[(r < TIERS[0]["r_min"]) | (r >= TIERS[-1]["r_max"])] = -1
        ring = np.zeros(r.shape, dtype=np.int32)
        sector = np.zeros(r.shape, dtype=np.int32)
        for i, t in enumerate(TIERS):
            m = tier_idx == i
            if np.any(m):
                ring[m] = np.clip(((r[m] - t["r_min"]) / t["dr"]).astype(np.int32),
                                  0, t["n_rings"] - 1)
                sector[m] = np.mod((theta[m] / t["dtheta"]).astype(np.int32), t["n_sectors"])
        return tier_idx, ring, sector

    # ------------------------------------------------------------------
    # Per-frame aggregation ("3D -> 2.5D projection without data loss")
    # ------------------------------------------------------------------
    def update(self, points, class_ids, confidences):
        """points: (N,3+) x,y,z[,...] in ego frame (already motion-compensated
        into the CURRENT ego frame by the caller). class_ids/confidences: (N,)
        from the perception model."""
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        tier_idx, ring, sector = self.locate(x, y)
        valid = tier_idx >= 0

        frame_has_data = [np.zeros((t["n_rings"], t["n_sectors"]), dtype=bool) for t in self.tiers]

        for i, t in enumerate(self.tiers):
            m = valid & (tier_idx == i)
            if not np.any(m):
                self._decay_tier(i, np.zeros((t["n_rings"], t["n_sectors"]), dtype=bool))
                continue
            ring_i, sec_i = ring[m], sector[m]
            flat = ring_i * t["n_sectors"] + sec_i
            n_cells = t["n_rings"] * t["n_sectors"]

            counts = np.bincount(flat, minlength=n_cells).astype(np.float32)

            terrain_mask = np.isin(class_ids[m], list(TERRAIN_CLASS_IDS))
            ground_sum = np.bincount(flat[terrain_mask], weights=z[m][terrain_mask],
                                      minlength=n_cells)
            ground_cnt = np.bincount(flat[terrain_mask], minlength=n_cells)
            ground_mean = np.divide(ground_sum, np.maximum(ground_cnt, 1),
                                     out=np.zeros(n_cells), where=ground_cnt > 0)

            top_h = np.full(n_cells, -np.inf, dtype=np.float32)
            np.maximum.at(top_h, flat, z[m])
            top_h[np.isinf(top_h)] = np.nan

            votes = self._class_votes[i].reshape(n_cells, NUM_CLASSES)
            votes *= 0.5  # decay previous-frame votes (temporal smoothing)
            w = confidences[m].astype(np.float32)
            for c in range(NUM_CLASSES):
                cm = class_ids[m] == c
                if np.any(cm):
                    votes[:, c] += np.bincount(flat[cm], weights=w[cm], minlength=n_cells)
            best_cls = votes.argmax(axis=1)

            # Confidence layer: observed density vs expected density at this range
            conf = np.clip(counts / self._expected_counts[i], 0, 1)

            has_data = counts > 0
            frame_has_data[i] = has_data.reshape(t["n_rings"], t["n_sectors"])

            g = self.cells[i].reshape(n_cells, N_LAYERS)
            g[has_data, LAYER_GROUND_H] = np.where(
                ground_cnt[has_data] > 0, ground_mean[has_data], g[has_data, LAYER_GROUND_H])
            g[has_data, LAYER_TOP_H] = np.where(
                ~np.isnan(top_h[has_data]), top_h[has_data], g[has_data, LAYER_TOP_H])
            g[has_data, LAYER_CLASS] = best_cls[has_data]
            g[has_data, LAYER_CONF] = conf[has_data]
            g[has_data, LAYER_COUNT] = counts[has_data]

            self._decay_tier(i, frame_has_data[i])

    def _decay_tier(self, tier_i, has_data_2d):
        """Cells with no returns this frame keep their last value but their
        confidence decays (EMA) — reflects growing staleness rather than
        instantly forgetting, matching the reference doc's temporal-
        persistence intent without the full rolling-buffer implementation
        (documented as a prototype simplification)."""
        g = self.cells[tier_i]
        empty = ~has_data_2d
        g[..., LAYER_CONF][empty] *= (1 - self._ema_alpha)

    # ------------------------------------------------------------------
    # Memory accounting (the headline "adaptive vs uniform" comparison)
    # ------------------------------------------------------------------
    def memory_stats(self):
        adaptive_cells = sum(t["n_cells"] for t in self.tiers)
        adaptive_bytes = adaptive_cells * N_LAYERS * 4  # float32
        uniform_side = int(2 * MAX_RANGE / UNIFORM_CELL_SIZE)
        uniform_cells = uniform_side * uniform_side
        uniform_bytes = uniform_cells * N_LAYERS * 4
        return dict(
            adaptive_cells=adaptive_cells, adaptive_bytes=adaptive_bytes,
            uniform_cells=uniform_cells, uniform_bytes=uniform_bytes,
            reduction_factor=uniform_cells / adaptive_cells,
        )

    def cell_stats_at(self, x, y):
        tier_idx, ring, sector = self.locate(np.array([x]), np.array([y]))
        if tier_idx[0] < 0:
            return None
        c = self.cells[tier_idx[0]][ring[0], sector[0]]
        return dict(tier=self.tiers[tier_idx[0]]["name"], ground_height=float(c[LAYER_GROUND_H]),
                    obstacle_top_height=float(c[LAYER_TOP_H]), class_id=int(c[LAYER_CLASS]),
                    confidence=float(c[LAYER_CONF]), point_count=float(c[LAYER_COUNT]))

    def visualization_cells(self, max_cells=3500, min_confidence=0.08):
        """A bounded view of real populated cells for the browser overlay.

        The processing grid remains entirely server-side; this deliberately
        sends only observed cells and caps the payload so the diagnostic view
        cannot become the pipeline's bottleneck.  Each record is the centre
        of an actual log-polar cell, plus its actual dimensions and layers.
        """
        chunks = []
        for tier_i, tier in enumerate(self.tiers):
            cells = self.cells[tier_i]
            occupied = cells[..., LAYER_CONF] >= min_confidence
            ring, sector = np.nonzero(occupied)
            if not len(ring):
                continue
            radius = tier["r_min"] + (ring.astype(np.float32) + 0.5) * tier["dr"]
            angle = (sector.astype(np.float32) + 0.5) * tier["dtheta"]
            vals = cells[ring, sector]
            chunks.append(np.column_stack((
                radius * np.cos(angle), radius * np.sin(angle),
                np.full(len(ring), tier_i),
                vals[:, LAYER_CLASS], vals[:, LAYER_CONF],
                np.full(len(ring), tier["dr"]),
                np.full(len(ring), tier["dtheta"]),
            )))
        if not chunks:
            return []
        result = np.concatenate(chunks)
        if len(result) > max_cells:
            # Evenly distributed selection preserves the radial profile better
            # than returning only the nearest, densest tier.
            result = result[np.linspace(0, len(result) - 1, max_cells, dtype=np.int32)]
        return np.round(result, 3).tolist()
