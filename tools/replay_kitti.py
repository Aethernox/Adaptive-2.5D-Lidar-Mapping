"""
KITTI Sequence Replay CLI Tool
Matches Section 17: Replay simulation engine with live console metrics and optional browser dashboard.
Usage:
    python -m tools.replay_kitti --sequence 00 --start-frame 0 --end-frame 200 --fps 10 [--dashboard]
"""

import sys
import time
import argparse
from typing import Optional

from simulation.kitti_replay import VirtualLidarReplay
from visualization.server import start_server


def run_cli_replay(
    sequence: str = "00",
    start_frame: int = 0,
    end_frame: Optional[int] = 200,
    fps: float = 10.0,
    mode: str = "ground_truth",
    loop: bool = False
):
    print("\n" + "=" * 78)
    print(f" VIRTUAL LIDAR REPLAY ENGINE")
    print(f" Sequence: {sequence} | Frames: {start_frame} -> {end_frame} | Mode: {mode} | Target FPS: {fps}")
    print("=" * 78 + "\n")

    replay = VirtualLidarReplay(
        sequence=sequence,
        start_frame=start_frame,
        end_frame=end_frame,
        target_fps=fps,
        mode=mode
    )

    print(f"{'FRAME':<8} | {'POINTS':<8} | {'PREPROCESS':<10} | {'INFERENCE':<10} | {'MAPPING':<10} | {'TRACKS':<8} | {'FPS':<6} | {'STATUS'}")
    print("-" * 78)

    try:
        for item in replay.stream_frames(loop=loop):
            m = item["metrics"]
            status = "OK"
            print(
                f"{m.seq:<8d} | {m.num_points:<8d} | {f'{m.preprocess_ms:.1f}ms':<10} | "
                f"{f'{m.inference_ms:.1f}ms':<10} | {f'{m.mapping_ms:.1f}ms':<10} | "
                f"{m.active_tracks:<8d} | {f'{m.fps:.1f}':<6} | {status}"
            )
    except KeyboardInterrupt:
        print("\n[Replay] Stopped by user.")

    print("\n" + "=" * 78)
    print(" Replay finished successfully!")
    print("=" * 78 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Virtual LiDAR Dataset Replay")
    parser.add_argument("--sequence", type=str, default="00", help="Sequence ID (e.g. 00)")
    parser.add_argument("--start-frame", type=int, default=0, help="Starting frame index")
    parser.add_argument("--end-frame", type=int, default=200, help="Ending frame index")
    parser.add_argument("--fps", type=float, default=10.0, help="Target playback FPS")
    parser.add_argument("--mode", type=str, default="ground_truth", choices=["ground_truth", "ai", "benchmark"], help="Perception mode")
    parser.add_argument("--dashboard", action="store_true", help="Launch interactive browser dashboard at http://localhost:8080")
    parser.add_argument("--loop", action="store_true", help="Loop playback continuously")
    args = parser.parse_args()

    if args.dashboard:
        start_server(port=8080, sequence=args.sequence)
    else:
        run_cli_replay(
            sequence=args.sequence,
            start_frame=args.start_frame,
            end_frame=args.end_frame,
            fps=args.fps,
            mode=args.mode,
            loop=args.loop
        )


if __name__ == "__main__":
    main()
