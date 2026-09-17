"""
Synthetic Lidar sweep generator: produces, for a given simulation time t, a
point cloud (in the EGO frame, which is what the rest of the pipeline
consumes) together with ground-truth per-point semantic labels used for
training / evaluation.

This is a stand-in for a real spinning-Lidar driver + SemanticKITTI replay.
It is not physically exact (no true ray occlusion), but it reproduces the
properties that matter for this prototype: point density falling off with
range, terrain + static + dynamic classes, and a moving ego vehicle whose
frame the grid engine must stay aligned with.
"""
import numpy as np
from config import MAX_RANGE, RNG_SEED
from sim.world import World, ground_height, ground_class

EGO_SPEED = 4.0  # m/s


def ego_pose(t):
    """Ego position + heading (rad) at time t. Mild lateral weave so the
    heading (and hence the grid orientation) actually changes over time,
    exercising the ego-motion-compensation path in the grid engine."""
    x = -20.0 + EGO_SPEED * t
    y = 2.0 * np.sin(t * 0.12)
    dx = EGO_SPEED
    dy = 2.0 * 0.12 * np.cos(t * 0.12)
    heading = np.arctan2(dy, dx)
    return x, y, heading


def world_to_ego(wx, wy, wz, ego_x, ego_y, ego_z, heading):
    dx, dy, dz = wx - ego_x, wy - ego_y, wz - ego_z
    c, s = np.cos(heading), np.sin(heading)
    xr = dx * c + dy * s
    yr = -dx * s + dy * c
    return xr, yr, dz


def ego_to_world(xr, yr, ego_x, ego_y, heading):
    c, s = np.cos(heading), np.sin(heading)
    dx = xr * c - yr * s
    dy = xr * s + yr * c
    return ego_x + dx, ego_y + dy


class LidarSimulator:
    def __init__(self, world: World = None, seed=RNG_SEED):
        self.world = world or World()
        self.rng = np.random.RandomState(seed)

    def _sample_ground(self, ego_x, ego_y, ego_z, heading, n_azimuth=720):
        rng = self.rng
        theta = np.linspace(0, 2 * np.pi, n_azimuth, endpoint=False)
        # Non-uniform range sampling per azimuth: more samples close-in.
        max_r_samples = 26
        # Sample fractional positions with density decaying with range.
        u = rng.uniform(0, 1, size=(n_azimuth, max_r_samples))
        r = MAX_RANGE * u ** 2.2  # bias toward small r
        theta_grid = np.repeat(theta[:, None], max_r_samples, axis=1)

        r = r.ravel()
        th = theta_grid.ravel()
        # Random thinning to emulate beam divergence / return dropout at range
        keep_prob = 1.0 / (1.0 + (r / 18.0) ** 2)
        keep = rng.uniform(0, 1, size=r.shape) < keep_prob
        r, th = r[keep], th[keep]
        r = np.clip(r, 0.3, MAX_RANGE - 0.01)

        xr = r * np.cos(th)
        yr = r * np.sin(th)
        wx, wy = ego_to_world(xr, yr, ego_x, ego_y, heading)
        wz = ground_height(wx, wy)

        # Exclude ground samples that fall inside a static/dynamic footprint
        # (that surface is emitted by the object sampler instead).
        mask = np.ones(wx.shape, dtype=bool)
        for obj in self.world.statics:
            if obj.kind == "pole":
                d = np.hypot(wx - obj.cx, wy - obj.cy)
                mask &= d > (obj.w / 2 + 0.05)
            else:
                mask &= ~((np.abs(wx - obj.cx) < obj.w / 2 + 0.1) & (np.abs(wy - obj.cy) < 0.3))
        cls = ground_class(wx, wy)
        xr_e, yr_e, zr_e = world_to_ego(wx, wy, wz, ego_x, ego_y, ego_z, heading)
        return xr_e[mask], yr_e[mask], zr_e[mask], cls[mask]

    def _sample_statics(self, ego_x, ego_y, ego_z, heading):
        xs, ys, zs, cs = [], [], [], []
        for obj in self.world.statics:
            d = np.hypot(obj.cx - ego_x, obj.cy - ego_y)
            if d > MAX_RANGE:
                continue
            n = int(np.clip(400 / (1 + (d / 15.0) ** 2), 4, 400))
            wx, wy, wz = obj.sample_points(n)
            xr_e, yr_e, zr_e = world_to_ego(wx, wy, wz, ego_x, ego_y, ego_z, heading)
            r = np.hypot(xr_e, yr_e)
            keep = r < MAX_RANGE
            xs.append(xr_e[keep]); ys.append(yr_e[keep]); zs.append(zr_e[keep])
            cs.append(np.full(keep.sum(), obj.cls))
        if not xs:
            return (np.array([]),) * 3 + (np.array([], dtype=int),)
        return (np.concatenate(xs), np.concatenate(ys), np.concatenate(zs),
                np.concatenate(cs).astype(int))

    def _sample_dynamics(self, t, ego_x, ego_y, ego_z, heading):
        xs, ys, zs, cs, meta = [], [], [], [], []
        for i, act in enumerate(self.world.dynamics):
            cx, cy = act.pose(t)
            d = np.hypot(cx - ego_x, cy - ego_y)
            if d > MAX_RANGE:
                continue
            n = int(np.clip(300 / (1 + (d / 15.0) ** 2), 6, 300))
            wx, wy, wz, ccx, ccy = act.sample_points(t, n)
            xr_e, yr_e, zr_e = world_to_ego(wx, wy, wz, ego_x, ego_y, ego_z, heading)
            r = np.hypot(xr_e, yr_e)
            keep = r < MAX_RANGE
            xs.append(xr_e[keep]); ys.append(yr_e[keep]); zs.append(zr_e[keep])
            cs.append(np.full(keep.sum(), act.cls))
            cvx, cvy, _ = world_to_ego(cx + act.vx, cy + act.vy, 0, ego_x, ego_y, 0, heading)
            cxr, cyr, _ = world_to_ego(cx, cy, 0, ego_x, ego_y, 0, heading)
            meta.append(dict(actor_id=i, kind=act.kind, cx=cxr, cy=cyr,
                              vx=cvx - cxr, vy=cvy - cyr, w=act.w, d=act.d))
        if not xs:
            return (np.array([]),) * 3 + (np.array([], dtype=int),), meta
        return (np.concatenate(xs), np.concatenate(ys), np.concatenate(zs),
                np.concatenate(cs).astype(int)), meta

    def sweep(self, t):
        """Return a full frame: points (N,3) in ego frame, true labels (N,),
        ego pose, and dynamic-actor ground truth (for tracker sanity checks)."""
        ego_x, ego_y, heading = ego_pose(t)
        ego_z = 0.0
        gx, gy, gz, gc = self._sample_ground(ego_x, ego_y, ego_z, heading)
        sx, sy, sz, sc = self._sample_statics(ego_x, ego_y, ego_z, heading)
        (dx, dy, dz, dc), dyn_meta = self._sample_dynamics(t, ego_x, ego_y, ego_z, heading)

        x = np.concatenate([gx, sx, dx])
        y = np.concatenate([gy, sy, dy])
        z = np.concatenate([gz, sz, dz])
        true_cls = np.concatenate([gc, sc, dc])
        intensity = self.rng.uniform(0.1, 1.0, size=x.shape)

        points = np.stack([x, y, z, intensity], axis=1).astype(np.float32)
        return dict(points=points, true_labels=true_cls.astype(int),
                    ego_x=ego_x, ego_y=ego_y, ego_z=ego_z, heading=heading,
                    dyn_meta=dyn_meta, t=t)
