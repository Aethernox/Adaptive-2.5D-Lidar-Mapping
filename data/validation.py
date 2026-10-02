"""Dataset validation entry point: ``python -m data.validation --dataset-root ...``."""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import numpy as np

from config import IGNORE_LABEL, KITTI_DATASET_ROOT
from data.kitti import KittiDataError, KittiSequenceDataset
from data.semantic_kitti import class_name, map_to_prototype


def validate_dataset(dataset_root: str | Path, sequence: str = "00", max_frames: int = 0) -> dict:
    ds = KittiSequenceDataset(dataset_root, sequence=sequence, require_labels=True)
    if not ds.calibration:
        raise KittiDataError(f"Calibration file is required but missing: {ds.calib_path}")
    selected = list(range(len(ds))) if max_frames <= 0 else list(range(min(len(ds), max_frames)))
    raw_counts: Counter[int] = Counter()
    mapped = ignored = points_total = 0
    intensity_min, intensity_max = np.inf, -np.inf
    xyz_min = np.full(3, np.inf); xyz_max = np.full(3, -np.inf)
    for index in selected:
        frame = ds.get_frame(index, require_labels=True)
        assert frame.semantic_labels is not None
        labels = frame.semantic_labels
        mapped_labels = map_to_prototype(labels)
        points_total += len(frame.points)
        mapped += int((mapped_labels != IGNORE_LABEL).sum())
        ignored += int((mapped_labels == IGNORE_LABEL).sum())
        class_ids, class_counts = np.unique(labels, return_counts=True)
        raw_counts.update({int(class_id): int(count) for class_id, count in zip(class_ids, class_counts)})
        intensity_min = min(intensity_min, float(frame.intensity.min(initial=np.inf)))
        intensity_max = max(intensity_max, float(frame.intensity.max(initial=-np.inf)))
        xyz_min = np.minimum(xyz_min, frame.points[:, :3].min(axis=0))
        xyz_max = np.maximum(xyz_max, frame.points[:, :3].max(axis=0))
    return {
        "root": str(ds.root), "sequence": ds.sequence, "frames_discovered": len(ds), "valid_frames": len(selected),
        "invalid_frames": 0, "average_points_per_frame": points_total / max(1, len(selected)),
        "points_total": points_total, "mapped_points": mapped, "ignored_points": ignored,
        "mapped_coverage": mapped / points_total if points_total else 0.0,
        "ignored_coverage": ignored / points_total if points_total else 0.0,
        "raw_class_counts": {str(k): v for k, v in sorted(raw_counts.items())},
        "raw_class_names": {str(k): class_name(k) for k in sorted(raw_counts)},
        "unknown_raw_ids": [k for k in sorted(raw_counts) if class_name(k).startswith("unknown_")],
        "intensity_range": [float(intensity_min), float(intensity_max)],
        "xyz_min": xyz_min.tolist(), "xyz_max": xyz_max.tolist(),
        "calibration_present": bool(ds.calibration), "poses_present": ds.poses is not None,
    }


def print_report(report: dict) -> None:
    print("KITTI DATASET VALIDATION\n------------------------")
    print(f"Root: {report['root']}\nSequence: {report['sequence']}")
    print(f"Frames discovered: {report['frames_discovered']}\nValid frames: {report['valid_frames']}\nInvalid frames: {report['invalid_frames']}")
    print(f"\nAverage points/frame: {report['average_points_per_frame']:.1f}")
    print(f"Calibration: {'present' if report['calibration_present'] else 'missing'}; poses: {'present' if report['poses_present'] else 'not available'}")
    print("\nLabel coverage:")
    print(f"    mapped: {report['mapped_coverage'] * 100:.2f}%\n    ignored: {report['ignored_coverage'] * 100:.2f}%")
    print("\nClasses:")
    for raw_id, count in report["raw_class_counts"].items():
        print(f"    {report['raw_class_names'][raw_id]} ({raw_id}): {count}")
    if report["unknown_raw_ids"]:
        print(f"Unknown/unmapped raw IDs: {report['unknown_raw_ids']}")
    print(f"Intensity range: {report['intensity_range'][0]:.4f} .. {report['intensity_range'][1]:.4f}")
    print(f"XYZ min/max: {report['xyz_min']} / {report['xyz_max']}")
    print("\nSTATUS: PASS")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a SemanticKITTI sequence frame by frame.")
    parser.add_argument("--dataset-root", default=KITTI_DATASET_ROOT, required=not bool(KITTI_DATASET_ROOT))
    parser.add_argument("--sequence", default="00")
    parser.add_argument("--max-frames", type=int, default=0)
    args = parser.parse_args()
    try:
        print_report(validate_dataset(args.dataset_root, args.sequence, args.max_frames))
    except KittiDataError as exc:
        print(f"STATUS: FAIL\n{exc}")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
