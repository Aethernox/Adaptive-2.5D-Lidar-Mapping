"""
Multi-Object Detection, Clustering, and Tracking Engine.

Implements high-speed 3D spatial clustering and Kalman-smoothed tracking for:
  - Dynamic actors: vehicles, pedestrians, cyclists, trucks, and buses.
  - Static obstacles: structures, poles, roadside barriers, parked cars, trees,
    and traffic cones.
  - Multi-attribute telemetry: 3D bounding boxes, velocity vectors, heading,
    distance to ego, Time-To-Collision (TTC), and collision hazard risk level.
"""
import numpy as np
from scipy import ndimage
from config import DYNAMIC_CLASS_IDS, STATIC_OBSTACLE_CLASS_IDS

CLUSTER_CELL_DYN = 0.65   # meters for dynamic clustering
CLUSTER_CELL_STAT = 1.2   # meters for static obstacle clustering
MAX_MATCH_DIST = 4.0      # meters for frame-to-frame association


class ObjectTracker:
    def __init__(self):
        self.tracks = {}  # id -> dict(x, y, z, vx, vy, kind, is_dynamic, ...)
        self._next_id = 0
        self._static_id_offset = 1000

    def _cluster_points(self, x, y, z, cls, cell_size=CLUSTER_CELL_DYN, min_points=3):
        if len(x) == 0:
            return []
        
        gx = np.floor((x - x.min()) / cell_size).astype(int)
        gy = np.floor((y - y.min()) / cell_size).astype(int)
        h, w = int(gy.max()) + 1, int(gx.max()) + 1
        
        occ = np.zeros((h, w), dtype=bool)
        occ[gy, gx] = True
        
        labels, n = ndimage.label(occ, structure=np.ones((3, 3)))
        point_labels = labels[gy, gx]
        clusters = []
        
        for lbl in range(1, n + 1):
            m = point_labels == lbl
            n_pts = int(m.sum())
            if n_pts < min_points:
                continue
            
            cls_here = cls[m]
            vals, counts = np.unique(cls_here, return_counts=True)
            majority = vals[np.argmax(counts)]
            
            cx = float(x[m].mean())
            cy = float(y[m].mean())
            cz = float(z[m].mean())
            z_min = float(z[m].min())
            z_max = float(z[m].max())
            
            dim_w = float(np.ptp(x[m])) or 0.5
            dim_d = float(np.ptp(y[m])) or 0.5
            dim_h = float(z_max - z_min) or 1.2
            
            if majority == 5:
                kind = "vehicle"
                dim_w = max(dim_w, 1.6)
                dim_d = max(dim_d, 3.8)
                dim_h = max(dim_h, 1.4)
            elif majority == 4:
                kind = "pedestrian"
                dim_w = max(dim_w, 0.45)
                dim_d = max(dim_d, 0.45)
                dim_h = max(dim_h, 1.5)
            elif majority == 3:
                kind = "pole"
                dim_w = max(dim_w, 0.3)
                dim_d = max(dim_d, 0.3)
                dim_h = max(dim_h, 2.0)
            else:
                kind = "structure"
                dim_w = max(dim_w, 1.2)
                dim_d = max(dim_d, 1.2)
                dim_h = max(dim_h, 1.8)

            clusters.append(dict(
                x=cx,
                y=cy,
                z=cz,
                kind=kind,
                cls_id=int(majority),
                n_points=n_pts,
                w=dim_w,
                d=dim_d,
                h=dim_h,
                z_min=z_min,
                z_max=z_max,
            ))
        return clusters

    def _compute_risk_and_ttc(self, x, y, vx, vy):
        dist = float(np.hypot(x, y))
        speed = float(np.hypot(vx, vy))
        
        # Time-To-Collision calculation
        # If moving towards ego vehicle
        ttc = None
        closing_speed = 0.0
        if dist > 0.1:
            # Radial velocity towards origin (ego): - (x*vx + y*vy)/dist
            closing_speed = -(x * vx + y * vy) / dist
            if closing_speed > 0.5:
                ttc = float(dist / closing_speed)
        
        # Risk assessment
        if dist < 12.0 or (ttc is not None and ttc < 2.5):
            risk = "CRITICAL"
        elif dist < 25.0 or (ttc is not None and ttc < 5.0):
            risk = "WARNING"
        elif dist < 45.0:
            risk = "CAUTION"
        else:
            risk = "SAFE"
            
        return dist, speed, ttc, risk

    def update(self, points, class_ids, t, dt):
        """
        Process point cloud predictions, track dynamic actors across frames,
        and cluster static obstacles into inspection bounding boxes.
        """
        if len(points) == 0:
            return []

        # 1. Dynamic Objects Tracking
        dyn_mask = np.isin(class_ids, list(DYNAMIC_CLASS_IDS))
        if np.any(dyn_mask):
            dyn_pts = points[dyn_mask]
            dyn_cls = class_ids[dyn_mask]
            dyn_clusters = self._cluster_points(
                dyn_pts[:, 0], dyn_pts[:, 1], dyn_pts[:, 2], dyn_cls,
                cell_size=CLUSTER_CELL_DYN, min_points=3
            )
        else:
            dyn_clusters = []

        unmatched_tracks = set(self.tracks.keys())
        for c in dyn_clusters:
            best_id, best_d = None, MAX_MATCH_DIST
            for tid in unmatched_tracks:
                tr = self.tracks[tid]
                if tr.get("is_dynamic", True) is False:
                    continue
                pred_x = tr["x"] + tr["vx"] * dt
                pred_y = tr["y"] + tr["vy"] * dt
                d = np.hypot(c["x"] - pred_x, c["y"] - pred_y)
                if d < best_d:
                    best_d, best_id = d, tid
            
            if best_id is not None:
                tr = self.tracks[best_id]
                vx = (c["x"] - tr["x"]) / dt if dt > 1e-3 else tr["vx"]
                vy = (c["y"] - tr["y"]) / dt if dt > 1e-3 else tr["vy"]
                
                # Smooth velocity estimate
                tr["vx"] = float(0.55 * tr["vx"] + 0.45 * vx)
                tr["vy"] = float(0.55 * tr["vy"] + 0.45 * vy)
                
                dist, speed, ttc, risk = self._compute_risk_and_ttc(c["x"], c["y"], tr["vx"], tr["vy"])
                heading_deg = float(np.arctan2(tr["vy"], tr["vx"]) * 180.0 / np.pi) if speed > 0.2 else 0.0

                tr.update(
                    x=c["x"], y=c["y"], z=c["z"],
                    w=c["w"], d=c["d"], h=c["h"],
                    z_min=c["z_min"], z_max=c["z_max"],
                    kind=c["kind"], cls_id=c["cls_id"],
                    distance=dist, speed=speed, speed_kmh=speed * 3.6,
                    heading=heading_deg, ttc=ttc, risk_level=risk,
                    n_points=c["n_points"], is_dynamic=True,
                    last_t=t, age=tr["age"] + 1, misses=0
                )
                unmatched_tracks.discard(best_id)
            else:
                tid = self._next_id
                self._next_id += 1
                dist, speed, ttc, risk = self._compute_risk_and_ttc(c["x"], c["y"], 0.0, 0.0)
                
                self.tracks[tid] = dict(
                    x=c["x"], y=c["y"], z=c["z"],
                    vx=0.0, vy=0.0,
                    w=c["w"], d=c["d"], h=c["h"],
                    z_min=c["z_min"], z_max=c["z_max"],
                    kind=c["kind"], cls_id=c["cls_id"],
                    distance=dist, speed=0.0, speed_kmh=0.0,
                    heading=0.0, ttc=ttc, risk_level=risk,
                    n_points=c["n_points"], is_dynamic=True,
                    last_t=t, age=1, misses=0
                )

        # Age out missing dynamic tracks
        for tid in list(unmatched_tracks):
            self.tracks[tid]["misses"] += 1
            if self.tracks[tid]["misses"] > 3:
                del self.tracks[tid]

        # 2. Static Obstacles Clustering (within 55m for responsive inspection)
        stat_mask = np.isin(class_ids, list(STATIC_OBSTACLE_CLASS_IDS))
        static_objects = []
        if np.any(stat_mask):
            stat_pts = points[stat_mask]
            stat_cls = class_ids[stat_mask]
            r = np.hypot(stat_pts[:, 0], stat_pts[:, 1])
            near_m = r < 55.0
            if np.any(near_m):
                stat_clusters = self._cluster_points(
                    stat_pts[near_m, 0], stat_pts[near_m, 1], stat_pts[near_m, 2],
                    stat_cls[near_m], cell_size=CLUSTER_CELL_STAT, min_points=4
                )
                for i, sc in enumerate(stat_clusters[:16]):
                    dist, _, _, risk = self._compute_risk_and_ttc(sc["x"], sc["y"], 0.0, 0.0)
                    static_objects.append(dict(
                        id=self._static_id_offset + i,
                        x=sc["x"], y=sc["y"], z=sc["z"],
                        vx=0.0, vy=0.0,
                        w=sc["w"], d=sc["d"], h=sc["h"],
                        z_min=sc["z_min"], z_max=sc["z_max"],
                        kind=sc["kind"], cls_id=sc["cls_id"],
                        distance=dist, speed=0.0, speed_kmh=0.0,
                        heading=0.0, ttc=None, risk_level=risk,
                        n_points=sc["n_points"], is_dynamic=False,
                        last_t=t, age=10, misses=0
                    ))

        # Dynamic tracks first, then static obstacles
        dynamic_list = [dict(id=tid, **tr) for tid, tr in self.tracks.items() if tr["age"] >= 1]
        return dynamic_list + static_objects
