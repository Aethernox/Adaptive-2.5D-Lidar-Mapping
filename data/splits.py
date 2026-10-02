"""Deterministic frame/sequence splits; frames never leak between partitions."""
from __future__ import annotations
import json
from pathlib import Path


def deterministic_frame_split(n_frames: int) -> dict[str, list[int]]:
    if n_frames < 3:
        raise ValueError("At least 3 frames are required for train/validation/test splitting")
    n_train = max(1, int(n_frames * 0.70))
    n_val = max(1, int(n_frames * 0.15))
    if n_train + n_val >= n_frames:
        n_val = 1
        n_train = n_frames - 2
    return {"train": list(range(0, n_train)), "validation": list(range(n_train, n_train + n_val)),
            "test": list(range(n_train + n_val, n_frames))}


def save_split(path: str | Path, split: dict, metadata: dict | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"split": split, "metadata": metadata or {}}, indent=2), encoding="utf-8")
