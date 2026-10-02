"""
Synthetic LiDAR sweep generator with real-time autonomous path planning,
multi-street navigation, dynamic obstacle avoidance, and high-density point clouds.
"""
import numpy as np
from config import MAX_RANGE, RNG_SEED
from sim.world import World, ground_height, ground_class, get_circuit_pose, CIRCUIT_LENGTH


def compute_ego_state(t):
    """
    Intelligent Autonomous Vehicle Navigation & Decision Controller.
    
    Operates across the 4-street city circuit (Grand Ave, South Blvd,
    Industrial Way, Park Lane) with 90° corner turns occurring every ~35m.
    """
    base_speed = 4.2  # m/s (~15.1 km/h)
    s = base_speed * t
    s_mod = np.mod(s, CIRCUIT_LENGTH)
    
    lateral_offset = -1.8  # default right-lane center
    speed = base_speed
    steer_angle = 0.0
    
    # 1. Street 1 (Grand Avenue: s in [0, 35.0m])
    if s_mod < 35.0:
        if 14.0 <= s_mod <= 28.0:
            # Construction cone & stalled car avoidance swerve
            decision = "AVOIDING_OBSTACLE_LANE_CHANGE"
            if s_mod < 19.0:
                p = (s_mod - 14.0) / 5.0
                lateral_offset = -1.8 + 3.4 * (3 * p ** 2 - 2 * p ** 3)
                steer_angle = 6.5 * np.sin(p * np.pi)
            elif s_mod <= 23.0:
                lateral_offset = 1.6  # clear overtaking lane
                steer_angle = 0.0
            else:
                p = (s_mod - 23.0) / 5.0
                lateral_offset = 1.6 - 3.4 * (3 * p ** 2 - 2 * p ** 3)
                steer_angle = -6.5 * np.sin(p * np.pi)
        elif 8.0 <= s_mod <= 14.0:
            decision = "SLOWING_FOR_CROSSING_PEDESTRIAN"
            speed = 2.8
            steer_angle = -1.0
        else:
            decision = "CRUISING_GRAND_AVENUE"

    # Turn 1: Grand Ave -> South Blvd (s in [35.0, 58.56])
    elif s_mod < 58.56:
        decision = "NAVIGATING_90_DEG_TURN_SOUTH_BLVD"
        speed = 3.2
        steer_angle = -18.0

    # 2. Street 2: South Boulevard (s in [58.56, 93.56])
    elif s_mod < 93.56:
        if 72.0 <= s_mod <= 82.0:
            decision = "CROSSING_INTERSECTION_PEDESTRIAN_YIELD"
            speed = 2.8
            steer_angle = 0.5
        else:
            decision = "CRUISING_SOUTH_BOULEVARD"

    # Turn 2: South Blvd -> Industrial Way (s in [93.56, 117.12])
    elif s_mod < 117.12:
        decision = "NAVIGATING_90_DEG_TURN_INDUSTRIAL_WAY"
        speed = 3.2
        steer_angle = -18.0

    # 3. Street 3: Industrial Way (s in [117.12, 152.12])
    elif s_mod < 152.12:
        if 125.0 <= s_mod <= 142.0:
            decision = "AVOIDING_CONSTRUCTION_ZONE_LANE_CHANGE"
            if s_mod < 131.0:
                p = (s_mod - 125.0) / 6.0
                lateral_offset = -1.8 + 3.4 * (3 * p ** 2 - 2 * p ** 3)
                steer_angle = 6.0 * np.sin(p * np.pi)
            elif s_mod <= 136.0:
                lateral_offset = 1.6
                steer_angle = 0.0
            else:
                p = (s_mod - 136.0) / 6.0
                lateral_offset = 1.6 - 3.4 * (3 * p ** 2 - 2 * p ** 3)
                steer_angle = -6.0 * np.sin(p * np.pi)
        else:
            decision = "CRUISING_INDUSTRIAL_WAY"

    # Turn 3: Industrial Way -> Park Lane (s in [152.12, 175.68])
    elif s_mod < 175.68:
        decision = "NAVIGATING_90_DEG_TURN_PARK_LANE"
        speed = 3.2
        steer_angle = -18.0

    # 4. Street 4: Park Lane (s in [175.68, 210.68])
    elif s_mod < 210.68:
        decision = "CRUISING_PARK_LANE_URBAN_PARK"

    # Turn 4: Park Lane -> Grand Ave (s in [210.68, 234.24])
    else:
        decision = "NAVIGATING_90_DEG_TURN_GRAND_AVE"
        speed = 3.2
        steer_angle = -18.0

    # Centerline pose & street name
    rx, ry, rth, street_name = get_circuit_pose(s)
    nx, ny = -np.sin(rth), np.cos(rth)
    ego_x = rx + nx * lateral_offset
    ego_y = ry + ny * lateral_offset
    
    heading = rth + np.radians(steer_angle * 0.35)
    ego_z = float(ground_height(np.array([ego_x]), np.array([ego_y]))[0])

    # 2. Planned 3D Collision-Free Trajectory Waypoints (ego frame)
    planned_waypoints_ego = []
    c, s_ang = np.cos(heading), np.sin(heading)
    
    for i in range(1, 24):
        look_dist = i * 1.6
        look_s = s + look_dist
        wrx, wry, wrth, _ = get_circuit_pose(look_s)
        wnx, wny = -np.sin(wrth), np.cos(wrth)
        
        look_s_mod = np.mod(look_s, CIRCUIT_LENGTH)
        look_lat = -1.8
        
        # Mirror lane change maneuvers in planned path
        if (14.0 <= look_s_mod <= 28.0) or (125.0 <= look_s_mod <= 142.0):
            look_lat = 1.6
            
        wx = wrx + wnx * look_lat
        wy = wry + wny * look_lat
        wz = float(ground_height(np.array([wx]), np.array([wy]))[0])
        
        dx, dy, dz = wx - ego_x, wy - ego_y, wz - ego_z
        xe = dx * c + dy * s_ang
        ye = -dx * s_ang + dy * c
        planned_waypoints_ego.append([float(xe), float(ye), float(dz + 0.12)])

    return {
        "x": float(ego_x),
        "y": float(ego_y),
        "z": float(ego_z),
        "heading": float(heading),
        "speed": float(speed),
        "speed_kmh": float(speed * 3.6),
        "steer_angle": float(steer_angle),
        "decision": decision,
        "street_name": street_name,
        "planned_path": planned_waypoints_ego
    }


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
        self.world = world or World(seed=seed)
        self.rng = np.random.RandomState(seed)

    def _sample_ground(self, ego_x, ego_y, ego_z, heading, statics, n_azimuth=960):
        rng = self.rng
        theta = np.linspace(0, 2 * np.pi, n_azimuth, endpoint=False)
        max_r_samples = 32
        u = rng.uniform(0, 1, size=(n_azimuth, max_r_samples))
        r = MAX_RANGE * u ** 1.85
        theta_grid = np.repeat(theta[:, None], max_r_samples, axis=1)

        r = r.ravel()
        th = theta_grid.ravel()
        keep_prob = 1.0 / (1.0 + (r / 25.0) ** 2)
        keep = rng.uniform(0, 1, size=r.shape) < keep_prob
        r, th = r[keep], th[keep]
        r = np.clip(r, 0.35, MAX_RANGE - 0.05)

        xr = r * np.cos(th)
        yr = r * np.sin(th)
        wx, wy = ego_to_world(xr, yr, ego_x, ego_y, heading)
        wz = ground_height(wx, wy)

        # Exclude ground inside obstacle footprints
        mask = np.ones(wx.shape, dtype=bool)
        for obj in statics:
            if obj.kind in ("pole", "tree", "cone", "sign"):
                d = np.hypot(wx - obj.cx, wy - obj.cy)
                mask &= d > (obj.w / 2.0 + 0.05)
            else:
                mask &= ~((np.abs(wx - obj.cx) < obj.w / 2.0 + 0.1) & (np.abs(wy - obj.cy) < obj.d / 2.0 + 0.1))

        cls = ground_class(wx, wy)
        xr_e, yr_e, zr_e = world_to_ego(wx, wy, wz, ego_x, ego_y, ego_z, heading)
        
        intensities = np.where(
            cls[mask] == 0,
            rng.uniform(0.32, 0.58, size=mask.sum()),
            rng.uniform(0.14, 0.35, size=mask.sum())
        )
        is_marking = (cls[mask] == 0) & ((np.abs(yr_e[mask]) < 0.12) | (np.abs(np.abs(yr_e[mask]) - 3.6) < 0.12))
        intensities[is_marking] = rng.uniform(0.85, 0.98, size=is_marking.sum())

        return xr_e[mask], yr_e[mask], zr_e[mask], cls[mask], intensities

    def _sample_statics(self, ego_x, ego_y, ego_z, heading, statics):
        xs, ys, zs, cs, ints = [], [], [], [], []
        rng = self.rng
        
        for obj in statics:
            d = np.hypot(obj.cx - ego_x, obj.cy - ego_y)
            if d > MAX_RANGE:
                continue
            
            if obj.kind in ("cone", "sign"):
                base_pts = 95
            elif obj.kind in ("pole", "tree"):
                base_pts = 190
            elif obj.kind == "parked_car":
                base_pts = 420
            else:
                base_pts = 520
                
            n = int(np.clip(base_pts / (1.0 + (d / 20.0) ** 2), 8, base_pts))
            wx, wy, wz = obj.sample_points(n)
            xr_e, yr_e, zr_e = world_to_ego(wx, wy, wz, ego_x, ego_y, ego_z, heading)
            r = np.hypot(xr_e, yr_e)
            keep = r < MAX_RANGE
            
            if not np.any(keep):
                continue
                
            k_sum = keep.sum()
            xs.append(xr_e[keep])
            ys.append(yr_e[keep])
            zs.append(zr_e[keep])
            cs.append(np.full(k_sum, obj.cls, dtype=int))
            
            if obj.kind in ("cone", "sign"):
                ints.append(rng.uniform(0.88, 1.0, size=k_sum))
            elif obj.kind == "parked_car":
                ints.append(rng.uniform(0.58, 0.95, size=k_sum))
            elif obj.kind in ("pole", "tree"):
                ints.append(rng.uniform(0.35, 0.75, size=k_sum))
            else:
                ints.append(rng.uniform(0.28, 0.65, size=k_sum))

        if not xs:
            return (np.array([]),) * 3 + (np.array([], dtype=int), np.array([]))
        return (np.concatenate(xs), np.concatenate(ys), np.concatenate(zs),
                np.concatenate(cs).astype(int), np.concatenate(ints).astype(np.float32))

    def _sample_dynamics(self, t, ego_x, ego_y, ego_z, heading, dynamics):
        xs, ys, zs, cs, ints, meta = [], [], [], [], [], []
        rng = self.rng
        
        for i, act in enumerate(dynamics):
            cx, cy, current_vx, current_vy = act.pose(t)
            d = np.hypot(cx - ego_x, cy - ego_y)
            if d > MAX_RANGE:
                continue
            
            base_pts = 140 if act.kind in ("pedestrian", "cyclist") else (520 if act.kind in ("truck", "bus") else 380)
            n = int(np.clip(base_pts / (1.0 + (d / 20.0) ** 2), 10, base_pts))
            wx, wy, wz, ccx, ccy = act.sample_points(t, n)
            xr_e, yr_e, zr_e = world_to_ego(wx, wy, wz, ego_x, ego_y, ego_z, heading)
            r = np.hypot(xr_e, yr_e)
            keep = r < MAX_RANGE
            
            if not np.any(keep):
                continue
                
            k_sum = keep.sum()
            xs.append(xr_e[keep])
            ys.append(yr_e[keep])
            zs.append(zr_e[keep])
            cs.append(np.full(k_sum, act.cls, dtype=int))
            
            if act.kind in ("vehicle", "truck", "bus"):
                ints.append(rng.uniform(0.55, 0.96, size=k_sum))
            else:
                ints.append(rng.uniform(0.28, 0.65, size=k_sum))
                
            cvx, cvy, _ = world_to_ego(cx + current_vx, cy + current_vy, 0, ego_x, ego_y, 0, heading)
            cxr, cyr, _ = world_to_ego(cx, cy, 0, ego_x, ego_y, 0, heading)
            
            meta.append(dict(
                actor_id=i,
                kind=act.kind,
                cx=float(cxr),
                cy=float(cyr),
                vx=float(cvx - cxr),
                vy=float(cvy - cyr),
                w=float(act.w),
                d=float(act.d),
                h=float(act.h)
            ))

        if not xs:
            return (np.array([]),) * 3 + (np.array([], dtype=int), np.array([])), meta
        return (np.concatenate(xs), np.concatenate(ys), np.concatenate(zs),
                np.concatenate(cs).astype(int), np.concatenate(ints).astype(np.float32)), meta

    def sweep(self, t):
        """Return full high-density LiDAR frame in the moving/turning ego sensor frame."""
        ego_state = compute_ego_state(t)
        ego_x = ego_state["x"]
        ego_y = ego_state["y"]
        ego_z = ego_state["z"]
        heading = ego_state["heading"]
        
        statics = self.world.get_statics_near(ego_x, ego_y, MAX_RANGE)
        dynamics = self.world.get_dynamics_near(t, ego_x, ego_y, MAX_RANGE)
            
        gx, gy, gz, gc, gi = self._sample_ground(ego_x, ego_y, ego_z, heading, statics)
        sx, sy, sz, sc, si = self._sample_statics(ego_x, ego_y, ego_z, heading, statics)
        (dx, dy, dz, dc, di), dyn_meta = self._sample_dynamics(t, ego_x, ego_y, ego_z, heading, dynamics)

        x = np.concatenate([gx, sx, dx])
        y = np.concatenate([gy, sy, dy])
        z = np.concatenate([gz, sz, dz])
        true_cls = np.concatenate([gc, sc, dc])
        intensity = np.concatenate([gi, si, di])

        points = np.stack([x, y, z, intensity], axis=1).astype(np.float32)
        return dict(
            points=points,
            true_labels=true_cls.astype(int),
            ego_x=ego_x,
            ego_y=ego_y,
            ego_z=ego_z,
            heading=heading,
            ego_state=ego_state,
            dyn_meta=dyn_meta,
            t=t
        )
