"""
Cartesian rasterization of the tiered polar grid for display.

The reference doc explicitly calls this out as a "simple resample, since
display doesn't need the compute savings the polar structure gives
elsewhere" — the pixel->(tier,ring,sector) lookup table is built ONCE at
startup, so per-frame rendering is a handful of vectorized fancy-index
gathers, not a per-pixel Python loop.
"""
import numpy as np
from config import (TIERS, MAX_RANGE, DISPLAY_IMAGE_SIZE, CLASS_COLORS, UNKNOWN_COLOR, EMPTY_COLOR,
                     LAYER_GROUND_H, LAYER_TOP_H, LAYER_CLASS, LAYER_CONF, LAYER_COUNT)
from mapping.grid_engine import VariableResolutionGrid

IMG_SIZE = DISPLAY_IMAGE_SIZE  # pixels, square, covers [-MAX_RANGE, MAX_RANGE]

CONF_EMPTY_THRESH = 0.03
CONF_UNKNOWN_THRESH = 0.22


class GridRasterizer:
    def __init__(self, img_size=IMG_SIZE):
        self.img_size = img_size
        half = MAX_RANGE
        # pixel (row=py maps to -y i.e. image "up"=+y forward-left convention;
        # col=px maps to +x = ego forward)
        xs = np.linspace(-half, half, img_size)
        ys = np.linspace(half, -half, img_size)
        gx, gy = np.meshgrid(xs, ys)  # gx varies along columns, gy along rows
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
        # class -> color LUT (for fast vectorized gather)
        self.color_lut = np.array([CLASS_COLORS[c] for c in sorted(CLASS_COLORS)], dtype=np.float32)
        self._ring_pixels = [self._build_ring_pixels(t["r_max"]) for t in TIERS]

    def render(self, grid: VariableResolutionGrid, dynamic_objects=None):
        H = W = self.img_size
        img = np.zeros((H, W, 3), dtype=np.uint8)
        img[:, :] = (10, 10, 12)  # out-of-range background

        flat_img = img.reshape(-1, 3)
        conf_full = np.zeros(H * W, dtype=np.float32)
        cls_full = np.zeros(H * W, dtype=np.int32)
        top_h_full = np.full(H * W, np.nan, dtype=np.float32)

        for i, t in enumerate(TIERS):
            pixels = self._pixel_indices[i]
            if pixels.size == 0:
                continue
            cells = grid.cells[i]  # (n_rings, n_sectors, N_LAYERS)
            vals = cells.reshape(-1, cells.shape[-1])[self._cell_indices[i]]
            conf_full[pixels] = vals[:, LAYER_CONF]
            cls_full[pixels] = vals[:, LAYER_CLASS].astype(np.int32)
            top_h_full[pixels] = vals[:, LAYER_TOP_H]

        in_range = self.in_range.ravel()
        known = in_range & (conf_full >= CONF_UNKNOWN_THRESH)
        low_conf = in_range & (conf_full >= CONF_EMPTY_THRESH) & (conf_full < CONF_UNKNOWN_THRESH)
        empty = in_range & (conf_full < CONF_EMPTY_THRESH)

        flat_img[empty] = EMPTY_COLOR
        flat_img[low_conf] = UNKNOWN_COLOR

        if np.any(known):
            base = self.color_lut[np.clip(cls_full[known], 0, len(self.color_lut) - 1)]
            brightness = 0.55 + 0.45 * np.clip(conf_full[known], 0, 1)[:, None]
            # subtle elevation shading using obstacle-top-height where available
            h = np.nan_to_num(top_h_full[known], nan=0.0)
            shade = np.clip(1.0 + 0.06 * h, 0.75, 1.3)[:, None]
            colored = np.clip(base * brightness * shade, 0, 255).astype(np.uint8)
            flat_img[known] = colored

        # grid tier boundary rings (faint) for visual reference
        for rows, cols in self._ring_pixels:
            img[rows, cols] = (70, 70, 75)

        # ego marker
        cx = cy = self.img_size // 2
        img[cy - 4:cy + 4, cx - 4:cx + 4] = (255, 255, 255)

        if dynamic_objects:
            self._draw_objects(img, dynamic_objects)

        return img

    def world_to_px(self, x, y):
        half = MAX_RANGE
        col = int((x + half) / (2 * half) * self.img_size)
        row = int((half - y) / (2 * half) * self.img_size)
        return col, row

    def _draw_range_ring(self, img, r, color=(70, 70, 75)):
        rows, cols = self._build_ring_pixels(r)
        img[rows, cols] = color

    def _build_ring_pixels(self, r):
        theta = np.linspace(0, 2 * np.pi, 720, endpoint=False)
        cols = ((r * np.cos(theta) + MAX_RANGE) / (2 * MAX_RANGE) * self.img_size).astype(np.int32)
        rows = ((MAX_RANGE - r * np.sin(theta)) / (2 * MAX_RANGE) * self.img_size).astype(np.int32)
        valid = (rows >= 0) & (rows < self.img_size) & (cols >= 0) & (cols < self.img_size)
        return rows[valid], cols[valid]

    def _draw_objects(self, img, objects):
        for obj in objects:
            col, row = self.world_to_px(obj["x"], obj["y"])
            half_px = max(int(obj.get("w", 1.0) / (2 * MAX_RANGE) * self.img_size * 2), 3)
            r0, r1 = max(row - half_px, 0), min(row + half_px, img.shape[0])
            c0, c1 = max(col - half_px, 0), min(col + half_px, img.shape[1])
            color = (255, 230, 60) if obj["kind"] == "vehicle" else (255, 80, 220)
            img[r0:r1, c0:c1] = self._blend_box_edge(img[r0:r1, c0:c1], color)
            vx, vy = obj.get("vx", 0), obj.get("vy", 0)
            speed = np.hypot(vx, vy)
            if speed > 0.05:
                ex, ey = obj["x"] + vx * 1.5, obj["y"] + vy * 1.5
                ecol, erow = self.world_to_px(ex, ey)
                self._draw_line(img, col, row, ecol, erow, color)

    @staticmethod
    def _blend_box_edge(patch, color):
        out = patch.copy()
        if out.size == 0:
            return out
        out[0, :] = color; out[-1, :] = color
        out[:, 0] = color; out[:, -1] = color
        return out

    @staticmethod
    def _draw_line(img, x0, y0, x1, y1, color, steps=20):
        H, W = img.shape[:2]
        for s in range(steps + 1):
            t = s / steps
            x = int(x0 + (x1 - x0) * t)
            y = int(y0 + (y1 - y0) * t)
            if 0 <= y < H and 0 <= x < W:
                img[y, x] = color
