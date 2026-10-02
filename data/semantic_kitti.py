"""SemanticKITTI packed-label decoding and explicit prototype mapping."""
from __future__ import annotations

import numpy as np

from config import IGNORE_LABEL, SEMANTICKITTI_TO_PROTOTYPE

# Official SemanticKITTI learning labels, plus the raw IDs frequently seen in
# semantic labels.  Unknown IDs are retained and reported by callers.
SEMANTICKITTI_CLASS_NAMES = {
    0: "unlabeled", 1: "outlier", 10: "car", 11: "bicycle", 13: "bus",
    15: "motorcycle", 16: "on-rails", 18: "truck", 20: "other-vehicle",
    30: "person", 31: "bicyclist", 32: "motorcyclist", 40: "road",
    44: "parking", 48: "sidewalk", 49: "other-ground", 50: "building",
    51: "fence", 52: "other-structure", 60: "lane-marking", 70: "vegetation",
    71: "trunk", 72: "terrain", 80: "pole", 81: "traffic-sign", 99: "other-object",
    252: "moving-car", 253: "moving-bicyclist", 254: "moving-person",
    255: "moving-motorcyclist", 256: "moving-on-rails", 257: "moving-bus",
    258: "moving-truck", 259: "moving-other-vehicle",
}
_PROTOTYPE_LOOKUP = np.full(1 << 16, IGNORE_LABEL, dtype=np.int16)
for _raw_id, _class_id in SEMANTICKITTI_TO_PROTOTYPE.items():
    _PROTOTYPE_LOOKUP[_raw_id] = _class_id


def _as_u32(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels)
    if labels.dtype.kind not in "iu":
        raise ValueError("SemanticKITTI labels must be an integer array")
    return labels.astype(np.uint32, copy=False)


def decode_semantic_labels(labels: np.ndarray) -> np.ndarray:
    """Return the low 16-bit semantic ID from packed uint32 labels."""
    return (_as_u32(labels) & np.uint32(0xFFFF)).astype(np.uint16)


def decode_instance_labels(labels: np.ndarray) -> np.ndarray:
    """Return the high 16-bit instance ID from packed uint32 labels."""
    return (_as_u32(labels) >> np.uint32(16)).astype(np.uint16)


def map_to_prototype(raw_semantic_ids: np.ndarray) -> np.ndarray:
    """Map raw semantic IDs to 0..5, retaining unmapped IDs as -1."""
    raw = np.asarray(raw_semantic_ids)
    if raw.ndim != 1:
        raise ValueError("semantic IDs must be a 1D array")
    if raw.dtype.kind not in "iu":
        raise ValueError("semantic IDs must be an integer array")
    # A lookup table avoids scanning an entire LiDAR sweep once per native
    # class (important when validating millions of points across a sequence).
    mapped = np.full(raw.shape, IGNORE_LABEL, dtype=np.int16)
    valid = (raw >= 0) & (raw < len(_PROTOTYPE_LOOKUP))
    mapped[valid] = _PROTOTYPE_LOOKUP[raw[valid].astype(np.intp, copy=False)]
    return mapped


def class_name(raw_id: int) -> str:
    return SEMANTICKITTI_CLASS_NAMES.get(int(raw_id), f"unknown_{int(raw_id)}")
