# Adaptive Variable-Resolution 2.5D Lidar Mapping — Prototype

A working, runnable prototype of the pipeline described in
`SOFTWARE_ARCHITECTURE.md`: synthetic Lidar → point-wise semantic
segmentation (terrain / static obstacles / dynamic objects) → a
tiered log-polar variable-resolution 2.5D grid → a live dashboard showing
the map, FPS/latency, memory savings vs. a uniform grid, and range-bucketed
accuracy.

No real Lidar hardware, GPU, or public dataset is required — see
"Assumptions" in `SOFTWARE_ARCHITECTURE.md` for why (no network access to
install `torch`, no real Lidar datasets available in this sandbox).

## Setup

```bash
pip install -r requirements.txt
```

## 1. Train the segmentation model (~15–20s on CPU)

```bash
python3 -m perception.train
```

Trains a small point-wise MLP on synthetic frames with ground-truth labels
and writes `perception/weights.npz`. The dashboard/pipeline will
auto-train on first run if this file is missing, so this step is optional.

## 2. Run the live dashboard

```bash
python3 -m server.app
```

Open **http://localhost:5050**. Each poll advances the simulated ego vehicle
one step and updates an interactive WebGL semantic point cloud plus the
adaptive 2.5D map. Drag the point-cloud viewport to orbit and use the wheel to
zoom. Live FPS, per-stage latency, adaptive-vs-uniform memory footprint,
tracked objects, and range-bucketed accuracy are shown beside the views.

The display raster uses a 384 px backend image that is upscaled to the
dashboard's 560 px presentation size. This keeps the adaptive grid's full
mapping resolution intact while reducing render and PNG-transfer cost; the
resolution is configurable as `config.DISPLAY_IMAGE_SIZE`.

## 3. Run the offline range-bucketed evaluation

```bash
python3 evaluate.py
```

Prints per-range-bucket accuracy/mIoU and the memory comparison table on a
held-out batch of synthetic frames.

## 4. Run tests

```bash
python3 run_tests.py
# or, if pytest is available (needs network to install):
# pytest tests/ -v
```

## Project layout

```
config.py                 classes, colors, tier schedule (single source of truth)
sim/world.py               procedural static+dynamic world (stand-in for SemanticKITTI)
sim/lidar.py                ego motion + synthetic point-cloud sweep generator
perception/features.py       per-point feature extraction
perception/model.py           point-wise segmentation MLP (numpy, manual backprop)
perception/train.py            training loop
mapping/grid_engine.py         tiered log-polar variable-resolution grid engine
mapping/tracker.py              dynamic-object clustering + frame-to-frame tracking
mapping/rasterizer.py            grid -> RGB image for the dashboard
pipeline.py                       orchestrates one frame end-to-end + metrics
server/app.py, templates/         Flask dashboard
evaluate.py                        offline range-bucketed evaluation
tests/, run_tests.py                unit tests + fallback runner (no pytest needed)
```
