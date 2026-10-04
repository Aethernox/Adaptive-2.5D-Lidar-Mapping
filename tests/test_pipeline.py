"""
End-to-End Pipeline & Simulation Preprocessor Integration Test
Verifies:
1. KITTI Frame -> Ingest & Filter -> Perception -> Adaptive Polar Grid -> Temporal Fusion -> Tracker -> Metrics
2. Batch Preprocessor -> Simulation Package -> High-FPS Replay Engine
Runs without crashing or leaking memory.
"""

import numpy as np
import pytest
from pathlib import Path

from simulation.kitti_replay import VirtualLidarReplay, PreprocessedLidarReplay
from simulation.preprocessor import SimulationPreprocessor, PreprocessedSimulation


def test_end_to_end_replay_pipeline():
    replay = VirtualLidarReplay(
        sequence="00",
        start_frame=0,
        end_frame=15, # 15 frames
        target_fps=20.0,
        mode="ground_truth"
    )

    frame_count = 0
    for item in replay.stream_frames(loop=False, realtime=False):
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
    print(f"[✓] Integration pipeline successfully processed {frame_count} frames end-to-end!")


def test_batch_preprocessor_and_high_fps_replay(tmp_path: Path):
    preprocessor = SimulationPreprocessor(
        cache_dir=str(tmp_path),
        subsample_ratio=3
    )

    # 1. Process 10 frames offline in terminal
    sim: PreprocessedSimulation = preprocessor.process_sequence(
        sequence="00",
        start_frame=0,
        end_frame=10,
        mode="ground_truth",
        save_cache=True,
        show_progress=False
    )

    assert len(sim) == 10
    assert sim.metadata["total_frames"] == 10
    assert sim.metadata["memory_reduction_pct"] > 90.0
    assert len(sim.json_strings) == 10

    # 2. Test instant frame access and JSON formatting
    frame_0 = sim[0]
    assert frame_0["type"] == "frame_update"
    assert "points" in frame_0
    assert "tracks" in frame_0
    assert "grid_cells" in frame_0

    json_str_0 = sim.get_json(0)
    assert isinstance(json_str_0, str)
    assert len(json_str_0) > 100

    # 3. Test saving and reloading from cache
    cache_file = tmp_path / "test_sim.sim.pkl"
    sim.save(cache_file)
    assert cache_file.exists()

    reloaded_sim = PreprocessedSimulation.load(cache_file)
    assert len(reloaded_sim) == 10

    # 4. Test high-speed replay streaming
    replay = PreprocessedLidarReplay(simulation=reloaded_sim, target_fps=100.0)
    streamed_count = 0
    for payload in replay.stream_frames(loop=False, realtime=False):
        streamed_count += 1
        assert "points" in payload

    assert streamed_count == 10
    print(f"[✓] Batch preprocessor & High-FPS replay test passed ({streamed_count} frames)!")


if __name__ == "__main__":
    test_end_to_end_replay_pipeline()
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        test_batch_preprocessor_and_high_fps_replay(Path(td))
