"""
Dynamic-object detection + tracking (Module 1 Head-B / Module 3's "sparse
dynamic overlay" in the reference doc), simplified for the prototype:

  1. Cluster points predicted as pedestrian/vehicle using a grid-based
     connected-components pass (cheap substitute for the reference's
     CenterPoint-style heatmap head).
  2. Track cluster centroids frame-to-frame with nearest-neighbor
     association + a constant-velocity estimate (substitute for the
     Kalman filter + Hungarian assignment in the reference doc).

Objects are kept as a small sparse list {id, class, x, y, vx, vy, footprint}
— never densified into the grid — matching the reference doc's rationale
for why dynamic objects shouldn't spend the grid's memory budget.
"""
import numpy as np
from scipy import ndimage
from config import DYNAMIC_CLASS_IDS

CLUSTER_CELL = 0.75  # meters, coarse grid used only for clustering
MAX_MATCH_DIST = 3.0  # meters, frame-to-frame association gate


class ObjectTracker:
    def __init__(self):
        self.tracks = {}  # id -> dict(x,y,vx,vy,kind,last_t,age)
        self._next_id = 0

    def _cluster(self, x, y, cls):
        if len(x) == 0:
            return []
        gx = np.floor((x - x.min()) / CLUSTER_CELL).astype(int)
        gy = np.floor((y - y.min()) / CLUSTER_CELL).astype(int)
        h, w = gy.max() + 1, gx.max() + 1
        occ = np.zeros((h, w), dtype=bool)
        occ[gy, gx] = True
        labels, n = ndimage.label(occ, structure=np.ones((3, 3)))
        point_labels = labels[gy, gx]
        clusters = []
        for lbl in range(1, n + 1):
            m = point_labels == lbl
            if m.sum() < 3:
                continue
            cls_here = cls[m]
            vals, counts = np.unique(cls_here, return_counts=True)
            majority = vals[np.argmax(counts)]
            clusters.append(dict(
                x=float(x[m].mean()), y=float(y[m].mean()),
                kind="vehicle" if majority == 5 else "pedestrian",
                n_points=int(m.sum()),
                w=float(x[m].max() - x[m].min()) or 0.5,
                d=float(y[m].max() - y[m].min()) or 0.5,
            ))
        return clusters

    def update(self, points, class_ids, t, dt):
        dyn_mask = np.isin(class_ids, list(DYNAMIC_CLASS_IDS))
        x, y = points[dyn_mask, 0], points[dyn_mask, 1]
        clusters = self._cluster(x, y, class_ids[dyn_mask])

        unmatched_tracks = set(self.tracks.keys())
        for c in clusters:
            best_id, best_d = None, MAX_MATCH_DIST
            for tid in unmatched_tracks:
                tr = self.tracks[tid]
                if tr["kind"] != c["kind"]:
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
                # light smoothing of the velocity estimate
                tr["vx"] = 0.6 * tr["vx"] + 0.4 * vx
                tr["vy"] = 0.6 * tr["vy"] + 0.4 * vy
                tr.update(x=c["x"], y=c["y"], w=c["w"], d=c["d"],
                          last_t=t, age=tr["age"] + 1, misses=0)
                unmatched_tracks.discard(best_id)
            else:
                tid = self._next_id; self._next_id += 1
                self.tracks[tid] = dict(x=c["x"], y=c["y"], vx=0.0, vy=0.0,
                                         kind=c["kind"], w=c["w"], d=c["d"],
                                         last_t=t, age=1, misses=0)

        # age out tracks that went unmatched this frame
        for tid in list(unmatched_tracks):
            self.tracks[tid]["misses"] += 1
            if self.tracks[tid]["misses"] > 3:
                del self.tracks[tid]

        return [dict(id=tid, **tr) for tid, tr in self.tracks.items() if tr["age"] >= 1]
