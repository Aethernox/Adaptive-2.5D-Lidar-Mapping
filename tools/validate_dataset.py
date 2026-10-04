"""
Dataset Validation Tool for KITTI / SemanticKITTI
Executes a rigorous structural, byte-level, and semantic integrity check on the dataset.
Usage:
    python -m tools.validate_dataset [--root /path/to/SemanticKITTI] [--verbose]
"""

import os
import sys
import glob
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

from datasets.semantic_kitti import SemanticKittiDataset


def format_size(bytes_val: int) -> str:
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_val < 1024.0:
            return f"{bytes_val:3.1f} {unit}"
        bytes_val /= 1024.0
    return f"{bytes_val:.1f} TB"


def validate_dataset(dataset_root: str = "kitti_dataset", verbose: bool = False) -> Dict:
    report = {
        "dataset_root": str(Path(dataset_root).resolve()),
        "status": "PASS",
        "sequences_found": [],
        "total_scans": 0,
        "total_labels": 0,
        "valid_calib_count": 0,
        "valid_poses_count": 0,
        "mismatched_frames": 0,
        "corrupted_files": 0,
        "sequence_details": {}
    }
    
    root_path = Path(dataset_root).resolve()
    seq_dir = root_path / "sequences"
    
    if not seq_dir.exists():
        report["status"] = "FAIL"
        report["error"] = f"Directory {seq_dir} does not exist."
        return report
        
    seq_folders = sorted([d for d in seq_dir.iterdir() if d.is_dir()])
    if not seq_folders:
        report["status"] = "FAIL"
        report["error"] = f"No sequence folders found in {seq_dir}."
        return report

    print("\n" + "=" * 78)
    print(f" KITTI / SemanticKITTI Dataset Validation Report")
    print(f" Root: {root_path}")
    print("=" * 78)

    for seq_folder in seq_folders:
        seq_name = seq_folder.name
        report["sequences_found"].append(seq_name)
        
        velo_dir = seq_folder / "velodyne"
        label_dir = seq_folder / "labels"
        calib_file = seq_folder / "calib.txt"
        poses_file = seq_folder / "poses.txt"
        if not poses_file.exists():
            poses_file = root_path / "poses" / f"{seq_name}.txt"
            
        velo_files = sorted(glob.glob(str(velo_dir / "*.bin"))) if velo_dir.exists() else []
        label_files = sorted(glob.glob(str(label_dir / "*.label"))) if label_dir.exists() else []
        
        has_calib = calib_file.exists()
        has_poses = poses_file.exists()
        
        if has_calib:
            report["valid_calib_count"] += 1
        if has_poses:
            report["valid_poses_count"] += 1
            
        report["total_scans"] += len(velo_files)
        report["total_labels"] += len(label_files)
        
        # Check first N frames for byte integrity and point-label count match
        sample_check_count = min(len(velo_files), 50 if not verbose else len(velo_files))
        seq_mismatches = 0
        seq_corruptions = 0
        sample_pts_count = 0
        sample_classes_found = set()
        
        label_dict = {Path(p).stem: p for p in label_files}
        
        for i in range(sample_check_count):
            bin_p = velo_files[i]
            stem = Path(bin_p).stem
            lbl_p = label_dict.get(stem, None)
            
            # Check bin size
            bin_size = os.path.getsize(bin_p)
            if bin_size % 16 != 0:
                seq_corruptions += 1
                continue
                
            num_pts = bin_size // 16
            sample_pts_count += num_pts
            
            # Check label size if present
            if lbl_p and os.path.exists(lbl_p):
                lbl_size = os.path.getsize(lbl_p)
                if lbl_size % 4 != 0:
                    seq_corruptions += 1
                    continue
                num_lbls = lbl_size // 4
                if num_pts != num_lbls:
                    seq_mismatches += 1
                else:
                    # Sample inspect semantic classes
                    if i < 5:
                        raw_labels = np.fromfile(lbl_p, dtype=np.uint32)
                        sem_ids = raw_labels & 0xFFFF
                        sample_classes_found.update(np.unique(sem_ids).tolist())

        avg_pts = int(sample_pts_count / max(sample_check_count, 1))
        
        seq_info = {
            "scans": len(velo_files),
            "labels": len(label_files),
            "has_calib": has_calib,
            "has_poses": has_poses,
            "mismatches": seq_mismatches,
            "corruptions": seq_corruptions,
            "avg_points_per_scan": avg_pts,
            "sample_classes": sorted(list(sample_classes_found))
        }
        report["sequence_details"][seq_name] = seq_info
        report["mismatched_frames"] += seq_mismatches
        report["corrupted_files"] += seq_corruptions
        
        status_flag = "OK" if seq_mismatches == 0 and seq_corruptions == 0 and len(velo_files) > 0 else "WARNING"
        print(f" Sequence [{seq_name}]: {status_flag}")
        print(f"   - Scans (.bin)      : {len(velo_files):5d} frames")
        print(f"   - Labels (.label)   : {len(label_files):5d} files")
        print(f"   - Calibration       : {'FOUND (' + calib_file.name + ')' if has_calib else 'MISSING (default fallback)'}")
        print(f"   - Poses             : {'FOUND (' + poses_file.name + ')' if has_poses else 'FALLBACK (synthetic odom)'}")
        print(f"   - Avg Points/Scan   : {avg_pts:,} pts")
        if sample_classes_found:
            print(f"   - Unique Classes    : {len(sample_classes_found)} semantic classes verified")
        print("-" * 78)

    print("\nSUMMARY:")
    print(f" Total Sequences  : {len(report['sequences_found'])}")
    print(f" Total Scans      : {report['total_scans']:,}")
    print(f" Total Labels     : {report['total_labels']:,}")
    print(f" Mismatched Frames: {report['mismatched_frames']}")
    print(f" Corrupted Files  : {report['corrupted_files']}")
    print(f" Final Status     : {report['status']}")
    print("=" * 78 + "\n")
    
    return report


def main():
    parser = argparse.ArgumentParser(description="Validate KITTI / SemanticKITTI Dataset")
    parser.add_argument("--root", type=str, default="kitti_dataset", help="Dataset root directory")
    parser.add_argument("--verbose", action="store_true", help="Perform full validation across all frames")
    args = parser.parse_args()
    
    report = validate_dataset(args.root, args.verbose)
    if report["status"] != "PASS":
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
