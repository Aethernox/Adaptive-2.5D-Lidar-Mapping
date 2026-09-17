"""
Synthetic driving-scene generator.

There is no real Lidar hardware or public dataset available in this
sandbox, so we build a small procedural "world" that stands in for
SemanticKITTI: a ground surface with a curb and a pothole, a handful of
static structures/poles, and a few dynamic pedestrians/vehicles moving on
simple trajectories. The world exposes ground-truth semantic labels for
every surface point, which is what lets us both *train* the perception
model and *evaluate* its accuracy per range bucket later on.
"""
import numpy as np
from config import RNG_SEED

CLASS_DRIVABLE, CLASS_NONDRIVABLE, CLASS_STRUCTURE, CLASS_POLE, CLASS_PED, CLASS_VEH = range(6)


def ground_height(x, y):
    """Height field for the terrain: gentle rolling noise + a curb strip
    (raised, non-drivable) + a pothole (localized depression)."""
    z = 0.02 * np.sin(x * 0.15) * np.cos(y * 0.1)
    # Curb: a raised strip running parallel to the road at y ~ 6m
    curb = np.where((y > 5.7) & (y < 6.3), 0.15, 0.0)
    # Pothole: localized depression near (8, -2)
    d2 = (x - 8.0) ** 2 + (y + 2.0) ** 2
    pothole = np.where(d2 < 1.2 ** 2, -0.12 * (1 - d2 / 1.2 ** 2), 0.0)
    return z + curb + pothole


def ground_class(x, y):
    """0 = drivable, 1 = non-drivable (curb / off-road / pothole rim)."""
    on_curb = (y > 5.7) & (y < 6.3)
    off_road = (y > 6.3) | (y < -8.0)
    d2 = (x - 8.0) ** 2 + (y + 2.0) ** 2
    on_pothole_rim = (d2 < 1.6 ** 2) & (d2 >= 0.9 ** 2)
    non_drivable = on_curb | off_road | on_pothole_rim
    return np.where(non_drivable, CLASS_NONDRIVABLE, CLASS_DRIVABLE)


class StaticObject:
    """Axis-aligned box (walls) or vertical cylinder (poles) in world frame."""

    def __init__(self, kind, cx, cy, w=0.3, d=0.3, h=1.8, base_z=None):
        self.kind = kind  # "wall" or "pole"
        self.cx, self.cy = cx, cy
        self.w, self.d, self.h = w, d, h
        self.base_z = base_z if base_z is not None else ground_height(np.array([cx]), np.array([cy]))[0]
        self.cls = CLASS_STRUCTURE if kind == "wall" else CLASS_POLE

    def sample_points(self, n):
        """Sample n points on the visible (outward-facing) surface."""
        if self.kind == "pole":
            theta = np.random.uniform(0, 2 * np.pi, n)
            z = np.random.uniform(0, self.h, n)
            r = self.w / 2
            x = self.cx + r * np.cos(theta)
            y = self.cy + r * np.sin(theta)
        else:  # wall: a flat vertical face facing the origin/ego side
            u = np.random.uniform(-self.w / 2, self.w / 2, n)
            z = np.random.uniform(0, self.h, n)
            x = self.cx + u
            y = self.cy + np.zeros(n)
        return x, y, self.base_z + z


class DynamicActor:
    """Simple constant-velocity box actor (pedestrian or vehicle)."""

    def __init__(self, kind, x0, y0, vx, vy, w, d, h):
        self.kind = kind
        self.x0, self.y0 = x0, y0
        self.vx, self.vy = vx, vy
        self.w, self.d, self.h = w, d, h
        self.cls = CLASS_PED if kind == "pedestrian" else CLASS_VEH

    def pose(self, t):
        return self.x0 + self.vx * t, self.y0 + self.vy * t

    def sample_points(self, t, n):
        cx, cy = self.pose(t)
        base_z = ground_height(np.array([cx]), np.array([cy]))[0]
        u = np.random.uniform(-self.w / 2, self.w / 2, n)
        v = np.random.uniform(-self.d / 2, self.d / 2, n)
        z = np.random.uniform(0.05, self.h, n)
        return cx + u, cy + v, base_z + z, cx, cy


class World:
    def __init__(self, seed=RNG_SEED):
        rng = np.random.RandomState(seed)
        self.statics = [
            StaticObject("wall", cx=14.0, cy=9.0, w=6.0, h=2.2),
            StaticObject("wall", cx=-10.0, cy=-9.0, w=8.0, h=2.5),
            StaticObject("wall", cx=40.0, cy=15.0, w=10.0, h=3.0),
        ]
        pole_positions = [(6, -6), (18, 6.6), (30, -10), (55, 20), (70, -25), (5, 6.6), (45, -18)]
        for (px, py) in pole_positions:
            self.statics.append(StaticObject("pole", cx=px, cy=py, w=0.25, h=2.0))

        self.dynamics = [
            DynamicActor("pedestrian", x0=6.0, y0=-1.5, vx=0.9, vy=0.05, w=0.5, d=0.5, h=1.7),
            DynamicActor("pedestrian", x0=20.0, y0=3.0, vx=-0.4, vy=0.3, w=0.5, d=0.5, h=1.6),
            DynamicActor("vehicle", x0=25.0, y0=-4.0, vx=-2.5, vy=0.0, w=1.9, d=4.5, h=1.6),
            DynamicActor("vehicle", x0=-15.0, y0=4.0, vx=3.0, vy=0.0, w=1.9, d=4.5, h=1.6),
        ]
        self._rng = rng
