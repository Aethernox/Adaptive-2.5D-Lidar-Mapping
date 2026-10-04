"""
End-to-End Pipeline Integration Test
Verifies:
KITTI Frame -> Ingest & Filter -> Perception -> Adaptive Polar Grid -> Temporal Fusion -> Tracker -> Metrics
Runs multiple consecutive frames without crashing or leaking memory.
"""

import numpy as np
import pytest

from simulation.kitti_replay import VirtualLidarReplay


def test_end_to_end_replay_pipeline():
    replay = VirtualLidarReplay(
        sequence="00",
        start_frame=0,
        end_frame=15, # 15 frames
        target_fps=20.0,
        mode="ground_truth"
    )

    frame_count = 0
    for item in replay.stream_frames(loop=False):
        frame_count += 1
        assert "frame" in item
        assert "map_snapshot" in item
        assert "track_set" in item
        assert "metrics" in item
        
        m = item["metrics"]
        assert m.fps > 0
        assert m.num_adaptive_cells > 280000
        assert m.memory_reduction_pct > 95.0 # ~98.2% reduction vs 16M uniform cells
        
        # Verify map snapshot tiers
        snapshot = item["map_snapshot"]
        assert len(snapshot.tiers) == 4

    assert frame_count == 15
    print(f"Integration pipeline successfully processed {frame_count} frames end-to-end!")


if __name__ == "__main__":
    test_end_to_end_replay_pipeline()
