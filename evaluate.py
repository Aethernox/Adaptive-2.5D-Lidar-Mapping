"""
Offline evaluation: range-bucketed accuracy / mIoU (the problem statement's
"accuracy across varying distances" requirement) plus a latency and memory
report, run over a batch of held-out synthetic frames (different seed/time
range from training).

Run: python3 evaluate.py
"""
import argparse
import time
import numpy as np

from config import RANGE_BUCKETS, CLASSES, NUM_CLASSES
from sim.lidar import LidarSimulator
from perception.model import PointSegModel
from perception.train import WEIGHTS_PATH
from mapping.grid_engine import VariableResolutionGrid


def evaluate(n_frames=25, seed=999):
    model = PointSegModel.load(WEIGHTS_PATH)
    sim = LidarSimulator(seed=seed)

    bucket_correct = {b: 0 for b in RANGE_BUCKETS}
    bucket_total = {b: 0 for b in RANGE_BUCKETS}
    bucket_intersection = {b: np.zeros(NUM_CLASSES) for b in RANGE_BUCKETS}
    bucket_union = {b: np.zeros(NUM_CLASSES) for b in RANGE_BUCKETS}

    infer_times = []
    for i in range(n_frames):
        t = 2000.0 + i * 1.3
        frame = sim.sweep(t)
        pts, true = frame["points"], frame["true_labels"]

        t0 = time.perf_counter()
        pred, conf = model.predict(pts)
        infer_times.append(time.perf_counter() - t0)

        r = np.hypot(pts[:, 0], pts[:, 1])
        for (lo, hi) in RANGE_BUCKETS:
            m = (r >= lo) & (r < hi)
            if not np.any(m):
                continue
            bucket_correct[(lo, hi)] += int((pred[m] == true[m]).sum())
            bucket_total[(lo, hi)] += int(m.sum())
            for c in range(NUM_CLASSES):
                pm, tm = pred[m] == c, true[m] == c
                bucket_intersection[(lo, hi)][c] += (pm & tm).sum()
                bucket_union[(lo, hi)][c] += (pm | tm).sum()

    print(f"\nEvaluated {n_frames} held-out frames "
          f"(mean inference time {1000*np.mean(infer_times):.2f} ms/frame)\n")
    print(f"{'Range':<10}{'Points':>10}{'Accuracy':>12}{'mIoU':>10}")
    print("-" * 42)
    for b in RANGE_BUCKETS:
        lo, hi = b
        total = bucket_total[b]
        if total == 0:
            print(f"{lo}-{hi}m{'':<4}{'0':>10}{'--':>12}{'--':>10}")
            continue
        acc = bucket_correct[b] / total
        ious = [bucket_intersection[b][c] / bucket_union[b][c]
                for c in range(NUM_CLASSES) if bucket_union[b][c] > 0]
        miou = np.mean(ious) if ious else float("nan")
        print(f"{lo}-{hi}m{'':<4}{total:>10}{acc*100:>11.1f}%{miou*100:>9.1f}%")

    # Memory comparison (deterministic, independent of the frame data)
    grid = VariableResolutionGrid()
    mem = grid.memory_stats()
    print("\nMemory footprint:")
    print(f"  Adaptive tiered grid : {mem['adaptive_cells']:>10,} cells "
          f"({mem['adaptive_bytes']/1e6:.2f} MB)")
    print(f"  Uniform 5cm grid     : {mem['uniform_cells']:>10,} cells "
          f"({mem['uniform_bytes']/1e6:.1f} MB)")
    print(f"  Reduction factor     : {mem['reduction_factor']:.0f}x")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--dataset", choices=("synthetic", "kitti"), default="synthetic")
    args, remaining = parser.parse_known_args()
    if args.dataset == "kitti":
        from evaluate_kitti import main as kitti_main
        import sys
        sys.argv = [sys.argv[0], *remaining]
        kitti_main()
    else:
        evaluate()
