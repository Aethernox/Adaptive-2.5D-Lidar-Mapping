"""Per-point feature extraction shared by training and inference.

Features are intentionally simple (x, y, z, intensity, range, height-ish
signal) — the point of this prototype's DL component is to demonstrate a
*real, trained, end-to-end segmentation network* sitting in the pipeline
where PointNet++/a sparse-conv backbone would go in production, not to
reach state-of-the-art accuracy on real Lidar data. See
SOFTWARE_ARCHITECTURE.md section "Perception backbone substitution".
"""
import numpy as np

FEATURE_NAMES = ["x", "y", "z", "intensity", "range"]
N_FEATURES = len(FEATURE_NAMES)

# Fitted once on a representative sample and reused everywhere so
# train/inference normalization always matches.
FEATURE_MEAN = np.array([20.0, 0.0, 0.4, 0.55, 25.0], dtype=np.float32)
FEATURE_STD = np.array([25.0, 15.0, 0.6, 0.3, 22.0], dtype=np.float32)


def extract_features(points: np.ndarray) -> np.ndarray:
    """points: (N,4) = x,y,z,intensity in ego frame -> (N,5) normalized feats."""
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 4:
        raise ValueError(f"points must have shape (N, 4), received {points.shape}")
    x, y, z, intensity = points[:, 0], points[:, 1], points[:, 2], points[:, 3]
    r = np.hypot(x, y)
    feats = np.stack([x, y, z, intensity, r], axis=1).astype(np.float32)
    return (feats - FEATURE_MEAN) / FEATURE_STD


def feature_configuration() -> dict:
    """Serializable feature contract saved beside real-data model weights."""
    return {"names": FEATURE_NAMES, "mean": FEATURE_MEAN.tolist(), "std": FEATURE_STD.tolist()}
