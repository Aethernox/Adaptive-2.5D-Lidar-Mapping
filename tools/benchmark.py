"""
Benchmark CLI Tool: Adaptive Variable-Resolution vs. Uniform Grid
Measures actual latency, memory footprint, FPS, and range-bucketed metrics.
Outputs results/benchmark.json and results/benchmark.csv.
"""

import os
import sys
import time
import json
import csv
import argparse
from pathlib import Path
import numpy as np

from datasets.semantic_kitti import SemanticKittiDataset
from mapping.adaptive_grid import AdaptivePolarGrid
from mapping.uniform_grid import UniformGridBaseline
from perception.inference import PerceptionEngine
from tracking.tracker import MultiObjectTracker
from evaluation.range_metrics import RangeBucketedEvaluator


def run_benchmark(
    sequence: str = "00",
    max_frames: int = 50,
    dataset_root: str = "kitti_dataset",
    output_dir: str = "results"
) -> dict:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 78)
    print(f" BENCHMARK: Adaptive Variable-Resolution vs Uniform 5cm Grid")
    print(f" Sequence: {sequence} | Frames: {max_frames} | Root: {dataset_root}")
    print("=" * 78)

    dataset = SemanticKittiDataset(dataset_root=dataset_root, sequences=[sequence], max_frames=max_frames)
    perception = PerceptionEngine(mode="ground_truth")
    adaptive_grid = AdaptivePolarGrid()
    uniform_grid = UniformGridBaseline()
    tracker = MultiObjectTracker()
    evaluator = RangeBucketedEvaluator()

    adaptive_latencies = []
    uniform_latencies = []
    frame_metrics_list = []

    for idx in range(min(len(dataset), max_frames)):
        frame = dataset[idx]
        
        # 1. Perception
        t_p0 = time.time()
        preds, confs = perception.predict(frame)
        p_time = (time.time() - t_p0) * 1000.0

        # Ground truth labels for range evaluation
        raw_sem = frame.semantic_labels
        learning_gts = dataset.raw_to_learning(raw_sem) if raw_sem is not None else preds
        evaluator.update(frame.points, preds, learning_gts)

        # 2. Tracking
        t_tr0 = time.time()
        track_set = tracker.update(frame.points, preds, frame.instance_ids, frame.stamp_ns)
        tr_time = (time.time() - t_tr0) * 1000.0

        # 3. Adaptive Grid Mapping
        t_a0 = time.time()
        adaptive_grid.update_from_points(frame.points, preds, confs, frame.instance_ids, clear_first=True)
        adapt_map_ms = (time.time() - t_a0) * 1000.0
        total_adapt_ms = p_time + tr_time + adapt_map_ms
        adaptive_latencies.append(total_adapt_ms)

        # 4. Uniform Grid Mapping
        t_u0 = time.time()
        uniform_stats = uniform_grid.update_from_points(frame.points, preds, confs)
        unif_map_ms = (time.time() - t_u0) * 1000.0
        total_unif_ms = p_time + tr_time + unif_map_ms
        uniform_latencies.append(total_unif_ms)

        frame_metrics_list.append({
            "frame": idx,
            "points": frame.num_points,
            "adaptive_ms": total_adapt_ms,
            "uniform_ms": total_unif_ms,
            "adapt_fps": 1000.0 / max(total_adapt_ms, 1e-2),
            "unif_fps": 1000.0 / max(total_unif_ms, 1e-2)
        })

    # Summary Calculations
    mean_adapt_ms = float(np.mean(adaptive_latencies))
    mean_unif_ms = float(np.mean(uniform_latencies))
    adapt_fps = 1000.0 / max(mean_adapt_ms, 1e-2)
    unif_fps = 1000.0 / max(mean_unif_ms, 1e-2)

    adapt_cells = adaptive_grid.total_cell_count
    unif_cells = uniform_grid.total_cell_count
    adapt_mb = (adapt_cells * 16) / (1024.0 * 1024.0)
    unif_mb = (unif_cells * 16) / (1024.0 * 1024.0)
    mem_reduction = (1.0 - (adapt_cells / unif_cells)) * 100.0
    mem_ratio = unif_cells / max(adapt_cells, 1)

    range_accuracy = evaluator.compute_metrics()

    benchmark_summary = {
        "sequence": sequence,
        "frames_evaluated": len(frame_metrics_list),
        "adaptive_grid": {
            "cell_count": adapt_cells,
            "memory_mb": round(adapt_mb, 2),
            "mean_latency_ms": round(mean_adapt_ms, 2),
            "fps": round(adapt_fps, 1)
        },
        "uniform_grid_5cm": {
            "cell_count": unif_cells,
            "memory_mb": round(unif_mb, 2),
            "mean_latency_ms": round(mean_unif_ms, 2),
            "fps": round(unif_fps, 1)
        },
        "comparative_advantage": {
            "memory_reduction_pct": round(mem_reduction, 2),
            "memory_ratio": f"{mem_ratio:.1f}x",
            "fps_speedup": f"{adapt_fps / max(unif_fps, 0.1):.2f}x"
        },
        "range_accuracy": range_accuracy
    }

    # Print Table
    print("\nRESULTS TABLE:")
    print("-" * 78)
    print(f" {'Metric':<30} | {'Adaptive Grid':<20} | {'Uniform 5cm Grid':<20}")
    print("-" * 78)
    print(f" {'Cell Count':<30} | {adapt_cells:<20,d} | {unif_cells:<20,d}")
    print(f" {'Memory Footprint':<30} | {f'{adapt_mb:.2f} MB':<20} | {f'{unif_mb:.2f} MB':<20}")
    print(f" {'Memory Reduction':<30} | {f'{mem_reduction:.1f}% ({mem_ratio:.1f}x)':<20} | {'Baseline (0%)':<20}")
    print(f" {'Mean Latency / Frame':<30} | {f'{mean_adapt_ms:.2f} ms':<20} | {f'{mean_unif_ms:.2f} ms':<20}")
    print(f" {'Effective Throughput':<30} | {f'{adapt_fps:.1f} FPS':<20} | {f'{unif_fps:.1f} FPS':<20}")
    print("-" * 78)

    print("\nRANGE-BUCKETED ACCURACY BREAKDOWN:")
    print("-" * 78)
    print(f" {'Range Tier':<25} | {'mIoU (%)':<15} | {'Point Accuracy (%)':<20}")
    print("-" * 78)
    for tier_name, metrics in range_accuracy.items():
        print(f" {tier_name:<25} | {metrics['mIoU']:<15.2f} | {metrics['point_accuracy_pct']:<20.2f}")
    print("=" * 78 + "\n")

    # Save JSON & CSV
    json_path = Path(output_dir) / "benchmark.json"
    with open(json_path, "w") as f:
        json.dump(benchmark_summary, f, indent=2)

    csv_path = Path(output_dir) / "benchmark.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["frame", "points", "adaptive_ms", "uniform_ms", "adapt_fps", "unif_fps"])
        writer.writeheader()
        writer.writerows(frame_metrics_list)

    print(f"Saved benchmark summary to: {json_path}")
    print(f"Saved per-frame metrics to: {csv_path}\n")

    return benchmark_summary


def main():
    parser = argparse.ArgumentParser(description="Run Adaptive vs Uniform Grid Benchmark")
    parser.add_argument("--sequence", type=str, default="00")
    parser.add_argument("--frames", type=int, default=50)
    parser.add_argument("--root", type=str, default="kitti_dataset")
    parser.add_argument("--out", type=str, default="results")
    args = parser.parse_args()

    run_benchmark(args.sequence, args.frames, args.root, args.out)


if __name__ == "__main__":
    main()
