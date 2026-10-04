"""
Terminal Raw Data Batch Preprocessor & Simulation Packager CLI
Pre-processes full raw LiDAR sequences in the terminal, performing perception, 2.5D adaptive
polar mapping, dynamic tracking, and metric calculation ahead of time for high-FPS localhost simulation.

Usage:
    python -m tools.preprocess_sequence --sequence 00 --start-frame 0 --end-frame 200 --mode ground_truth
"""

import sys
import argparse
from pathlib import Path
from simulation.preprocessor import SimulationPreprocessor


def main():
    parser = argparse.ArgumentParser(
        description="Pre-process Raw LiDAR Sequence in Terminal for High-FPS Localhost Simulation"
    )
    parser.add_argument("--sequence", type=str, default="00", help="KITTI Sequence ID (default: 00)")
    parser.add_argument("--start-frame", type=int, default=0, help="Starting frame index (default: 0)")
    parser.add_argument("--end-frame", type=int, default=200, help="Ending frame index (default: 200, or -1 for all)")
    parser.add_argument("--mode", type=str, default="ground_truth", choices=["ground_truth", "ai", "benchmark"], help="Perception mode")
    parser.add_argument("--subsample", type=int, default=3, help="Point cloud subsample ratio for web rendering (default: 3)")
    parser.add_argument("--cache-dir", type=str, default="data_cache", help="Directory to save preprocessed simulation package")
    parser.add_argument("--dataset-root", type=str, default="kitti_dataset", help="Path to KITTI dataset root")
    parser.add_argument("--force", action="store_true", help="Force re-processing even if cache already exists")
    
    args = parser.parse_args()
    end_frame = None if args.end_frame < 0 else args.end_frame

    preprocessor = SimulationPreprocessor(
        dataset_root=args.dataset_root,
        cache_dir=args.cache_dir,
        subsample_ratio=args.subsample
    )

    if not args.force and preprocessor.has_cache(args.sequence, args.start_frame, end_frame, args.mode, args.subsample):
        print(f"\n[!] Preprocessed simulation package already exists for Sequence {args.sequence} (frames {args.start_frame}->{end_frame}).")
        print(f"    Path: {preprocessor.get_cache_path(args.sequence, args.start_frame, end_frame, args.mode, args.subsample)}")
        print(f"    Use '--force' if you wish to re-compute raw data.\n")
        print("To launch high-FPS localhost simulation now:")
        print(f"    python -m tools.replay_kitti --sequence {args.sequence} --dashboard")
        return

    sim = preprocessor.process_sequence(
        sequence=args.sequence,
        start_frame=args.start_frame,
        end_frame=end_frame,
        mode=args.mode,
        subsample_ratio=args.subsample,
        save_cache=True,
        show_progress=True
    )

    print("Next step: Launch the High-FPS Localhost Simulation Server:")
    print(f"    python -m tools.replay_kitti --sequence {args.sequence} --dashboard\n")


if __name__ == "__main__":
    main()
