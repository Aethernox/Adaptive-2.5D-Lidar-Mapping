"""
KITTI Sequence Replay CLI Tool & Simulation Launcher
Supports both batch terminal preprocessing and high-FPS zero-latency localhost simulation.

Usage:
    # 1. Interactive High-FPS Web Dashboard (Terminal Pre-processing -> Localhost 60+ FPS):
    python -m tools.replay_kitti --sequence 00 --start-frame 0 --end-frame 200 --dashboard

    # 2. Terminal Pre-processing Only:
    python -m tools.replay_kitti --sequence 00 --start-frame 0 --end-frame 200 --preprocess-only

    # 3. CLI Real-Time Replay:
    python -m tools.replay_kitti --sequence 00 --start-frame 0 --end-frame 200 --fps 30
"""

import sys
import time
import argparse
from typing import Optional

from simulation.kitti_replay import VirtualLidarReplay, PreprocessedLidarReplay
from simulation.preprocessor import SimulationPreprocessor
from visualization.server import start_server


def run_cli_replay(
    sequence: str = "00",
    start_frame: int = 0,
    end_frame: Optional[int] = 200,
    fps: float = 30.0,
    mode: str = "ground_truth",
    loop: bool = False,
    use_cache: bool = True
):
    print("\n" + "=" * 80)
    print(" VIRTUAL LIDAR REPLAY ENGINE (CLI MODE)")
    print(f" Sequence: {sequence} | Frames: [{start_frame} -> {end_frame}] | Mode: {mode.upper()} | Target FPS: {fps}")
    print("=" * 80 + "\n")

    preprocessor = SimulationPreprocessor()
    if use_cache and preprocessor.has_cache(sequence, start_frame, end_frame, mode):
        print(f"[CLI Replay] Loading cached simulation package for fast replay...")
        sim = preprocessor.load_cache(sequence, start_frame, end_frame, mode)
        replay = PreprocessedLidarReplay(simulation=sim, target_fps=fps)
        
        print(f"{'FRAME':<8} | {'POINTS':<8} | {'INFER':<10} | {'MAPPING':<10} | {'TRACKS':<8} | {'FPS':<6} | {'MODE'}")
        print("-" * 80)
        try:
            for payload in replay.stream_frames(loop=loop, realtime=True):
                m = payload["metrics"]
                infer_str = f"{m['inference_ms']:.1f}ms"
                map_str = f"{m['mapping_ms']:.1f}ms"
                fps_str = f"{fps:.1f}"
                print(
                    f"{m['seq']:<8d} | {m['num_points']:<8d} | {infer_str:<10} | "
                    f"{map_str:<10} | {m['active_tracks']:<8d} | {fps_str:<6} | CACHED"
                )
        except KeyboardInterrupt:
            print("\n[Replay] Stopped by user.")
    else:
        replay = VirtualLidarReplay(
            sequence=sequence,
            start_frame=start_frame,
            end_frame=end_frame,
            target_fps=fps,
            mode=mode
        )

        print(f"{'FRAME':<8} | {'POINTS':<8} | {'PREPROCESS':<10} | {'INFERENCE':<10} | {'MAPPING':<10} | {'TRACKS':<8} | {'FPS':<6} | {'STATUS'}")
        print("-" * 80)

        try:
            for item in replay.stream_frames(loop=loop, realtime=True):
                m = item["metrics"]
                status = "OK"
                print(
                    f"{m.seq:<8d} | {m.num_points:<8d} | {f'{m.preprocess_ms:.1f}ms':<10} | "
                    f"{f'{m.inference_ms:.1f}ms':<10} | {f'{m.mapping_ms:.1f}ms':<10} | "
                    f"{m.active_tracks:<8d} | {f'{m.fps:.1f}':<6} | {status}"
                )
        except KeyboardInterrupt:
            print("\n[Replay] Stopped by user.")

    print("\n" + "=" * 80)
    print(" Replay completed successfully!")
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Virtual LiDAR Dataset Replay & Simulation Launcher")
    parser.add_argument("--sequence", type=str, default="00", help="Sequence ID (e.g. 00)")
    parser.add_argument("--start-frame", type=int, default=0, help="Starting frame index")
    parser.add_argument("--end-frame", type=int, default=200, help="Ending frame index (or -1 for all)")
    parser.add_argument("--fps", type=float, default=30.0, help="Target simulation playback FPS (default: 30.0)")
    parser.add_argument("--mode", type=str, default="ground_truth", choices=["ground_truth", "ai", "benchmark"], help="Perception mode")
    parser.add_argument("--subsample", type=int, default=3, help="Point cloud subsample ratio (default: 3)")
    parser.add_argument("--dashboard", action="store_true", help="Launch high-FPS localhost interactive dashboard at http://localhost:8080")
    parser.add_argument("--preprocess-only", action="store_true", help="Pre-process sequence in terminal and exit")
    parser.add_argument("--no-cache", action="store_true", help="Force terminal re-processing even if cache exists")
    parser.add_argument("--loop", action="store_true", help="Loop CLI playback continuously")
    args = parser.parse_args()

    end_frame = None if args.end_frame < 0 else args.end_frame

    if args.preprocess_only:
        preprocessor = SimulationPreprocessor(subsample_ratio=args.subsample)
        preprocessor.process_sequence(
            sequence=args.sequence,
            start_frame=args.start_frame,
            end_frame=end_frame,
            mode=args.mode,
            subsample_ratio=args.subsample,
            save_cache=True,
            show_progress=True
        )
    elif args.dashboard:
        start_server(
            port=8080,
            sequence=args.sequence,
            start_frame=args.start_frame,
            end_frame=end_frame,
            fps=args.fps,
            mode=args.mode,
            subsample_ratio=args.subsample,
            force_preprocess=args.no_cache
        )
    else:
        run_cli_replay(
            sequence=args.sequence,
            start_frame=args.start_frame,
            end_frame=end_frame,
            fps=args.fps,
            mode=args.mode,
            loop=args.loop,
            use_cache=not args.no_cache
        )


if __name__ == "__main__":
    main()
