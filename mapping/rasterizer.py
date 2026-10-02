"""
Cartesian rasterization of the tiered polar grid for display.

Generates high-contrast, multi-layer top-down 2.5D perception map:
  - Vectorized O(1) table lookup from Cartesian pixels to log-polar grid cells.
  - Multi-class color rendering with confidence blending and elevation relief.
  - Ego vehicle orientation heading cone and footprint.
  - Polar tier boundary rings and range indicators.
  - Dynamic object tracking boxes with velocity vectors.
"""
import numpy as np
from config import (TIERS, MAX_RANGE, DISPLAY_IMAGE_SIZE, CLASS_COLORS, UNKNOWN_COLOR, EMPTY_COLOR,
                     LAYER_GROUND_H, LAYER_TOP_H, LAYER_CLASS, LAYER_CONF, LAYER_COUNT)
from mapping.grid_engine import VariableResolutionGrid

IMG_SIZE = DISPLAY_IMAGE_SIZE  # pixels, square, covers [-MAX_RANGE, MAX_RANGE]

CONF_EMPTY_THRESH = 0.03
CONF_UNKNOWN_THRESH = 0.20

# Vivid modern palette for crisp 2.5D display
VIVID_COLORS = {
    0: (46, 184, 92),     # drivable terrain - vibrant emerald green
    1: (180, 160, 80),    # non-drivable / curb / sidewalk - warm amber/gold
    2: (215, 65, 75),     # static structure - crimson ruby
    3: (255, 145, 30),    # static pole - bright orange
    4: (255, 60, 180),    # pedestrian - neon magenta
    5: (30, 150, 255),    # vehicle - electric cyan
}


class GridRasterizer:
    def __init__(self, img_size=IMG_SIZE):
        self.img_size = img_size
        half = MAX_RANGE
        xs = np.linspace(-half, half, img_size)
        ys = np.linspace(half, -half, img_size)
        gx, gy = np.meshgrid(xs, ys)
        tier_idx, ring, sector = VariableResolutionGrid.locate(gx, gy)
        self.tier_idx = tier_idx
        self.ring = ring
        self.sector = sector
        self.in_range = tier_idx >= 0
        self._pixel_indices = []
        self._cell_indices = []
        flat_tiers = tier_idx.ravel()
        flat_rings = ring.ravel()
        flat_sectors = sector.ravel()
        
        for i, t in enumerate(TIERS):
            pixel_indices = np.flatnonzero(flat_tiers == i)
            self._pixel_indices.append(pixel_indices)
            self._cell_indices.append(flat_rings[pixel_indices] * t["n_sectors"] + flat_sectors[pixel_indices])
            
        self.color_lut = np.array([VIVID_COLORS.get(c, CLASS_COLORS[c]) for c in sorted(CLASS_COLORS)], dtype=np.float32)
        self._ring_pixels = [self._build_ring_pixels(t["r_max"]) for t in TIERS]

    def render(self, grid: VariableResolutionGrid, dynamic_objects=None):
        H = W = self.img_size
        img = np.zeros((H, W, 3), dtype=np.uint8)
        img[:, :] = (8, 12, 18)  # Deep cyber slate background

        flat_img = img.reshape(-1, 3)
        conf_full = np.zeros(H * W, dtype=np.float32)
        cls_full = np.zeros(H * W, dtype=np.int32)
        top_h_full = np.full(H * W, np.nan, dtype=np.float32)

        for i, t in enumerate(TIERS):
            pixels = self._pixel_indices[i]
            if pixels.size == 0:
                continue
            cells = grid.cells[i]
            vals = cells.reshape(-1, cells.shape[-1])[self._cell_indices[i]]
            conf_full[pixels] = vals[:, LAYER_CONF]
            cls_full[pixels] = vals[:, LAYER_CLASS].astype(np.int32)
            top_h_full[pixels] = vals[:, LAYER_TOP_H]

        in_range = self.in_range.ravel()
        known = in_range & (conf_full >= CONF_UNKNOWN_THRESH)
        low_conf = in_range & (conf_full >= CONF_EMPTY_THRESH) & (conf_full < CONF_UNKNOWN_THRESH)
        empty = in_range & (conf_full < CONF_EMPTY_THRESH)

        flat_img[empty] = (14, 18, 26)
        flat_img[low_conf] = (28, 35, 45)

        if np.any(known):
            base = self.color_lut[np.clip(cls_full[known], 0, len(self.color_lut) - 1)]
            brightness = 0.60 + 0.40 * np.clip(conf_full[known], 0, 1)[:, None]
            h = np.nan_to_num(top_h_full[known], nan=0.0)
            shade = np.clip(1.0 + 0.08 * h, 0.75, 1.45)[:, None]
            colored = np.clip(base * brightness * shade, 0, 255).astype(np.uint8)
            flat_img[known] = colored

        # Grid tier boundary rings
        for rows, cols in self._ring_pixels:
            img[rows, cols] = (45, 65, 85)

        # Crosshairs / Axes
        cx = cy = self.img_size // 2
        img[cy, max(0, cx - 40):min(W, cx + 40)] = (35, 55, 75)
        img[max(0, cy - 40):min(H, cy + 40), cx] = (35, 55, 75)

        # Ego vehicle footprint & forward heading chevron
        img[cy - 5:cy + 5, cx - 3:cx + 3] = (0, 230, 255)
        img[cy - 8:cy - 5, cx - 1:cx + 1] = (255, 255, 255)  # forward nose

        if dynamic_objects:
            self._draw_objects(img, dynamic_objects)

        return img

    def world_to_px(self, x, y):
        half = MAX_RANGE
        col = int((x + half) / (2 * half) * self.img_size)
        row = int((half - y) / (2 * half) * self.img_size)
        return col, row

    def _build_ring_pixels(self, r):
        theta = np.linspace(0, 2 * np.pi, 720, endpoint=False)
        cols = ((r * np.cos(theta) + MAX_RANGE) / (2 * MAX_RANGE) * self.img_size).astype(np.int32)
        rows = ((MAX_RANGE - r * np.sin(theta)) / (2 * MAX_RANGE) * self.img_size).astype(np.int32)
        valid = (rows >= 0) & (rows < self.img_size) & (cols >= 0) & (cols < self.img_size)
        return rows[valid], cols[valid]

    def _draw_objects(self, img, objects):
        for obj in objects:
            col, row = self.world_to_px(obj["x"], obj["y"])
            hw = max(int(obj.get("w", 1.8) / (2 * MAX_RANGE) * self.img_size), 2)
            hd = max(int(obj.get("d", 4.0) / (2 * MAX_RANGE) * self.img_size), 2)
            
            r0, r1 = max(row - hd, 0), min(row + hd, img.shape[0])
            c0, c1 = max(col - hw, 0), min(col + hw, img.shape[1])
            
            if obj.get("kind") == "vehicle":
                color = (0, 220, 255)
            elif obj.get("kind") == "pedestrian":
                color = (255, 80, 220)
            elif obj.get("kind") == "cyclist":
                color = (255, 200, 40)
            elif obj.get("kind") == "truck":
                color = (80, 160, 255)
            elif obj.get("kind") == "pole":
                color = (255, 140, 20)
            else:
                color = (230, 80, 80)
                
            img[r0:r1, c0:c1] = self._blend_box_edge(img[r0:r1, c0:c1], color)
            
            # Velocity vector
            vx, vy = obj.get("vx", 0), obj.get("vy", 0)
            speed = np.hypot(vx, vy)
            if speed > 0.15:
                ex, ey = obj["x"] + vx * 1.5, obj["y"] + vy * 1.5
                ecol, erow = self.world_to_px(ex, ey)
                self._draw_line(img, col, row, ecol, erow, (255, 255, 255))

    @staticmethod
    def _blend_box_edge(patch, color):
        out = patch.copy()
        if out.size == 0:
            return out
        out[0, :] = color
        out[-1, :] = color
        out[:, 0] = color
        out[:, -1] = color
        return out

    @staticmethod
    def _draw_line(img, x0, y0, x1, y1, color, steps=24):
        H, W = img.shape[:2]
        for s in range(steps + 1):
            t = s / steps
            x = int(x0 + (x1 - x0) * t)
            y = int(y0 + (y1 - y0) * t)
            if 0 <= y < H and 0 <= x < W:
                img[y, x] = color
