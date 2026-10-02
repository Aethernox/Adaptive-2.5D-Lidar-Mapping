# Adaptive Variable-Resolution LiDAR Mapping

A runnable spatial-intelligence prototype for real-time LiDAR perception. It
supports deterministic synthetic driving scenes and KITTI/SemanticKITTI replay
through the same semantic model, variable-resolution 2.5D grid, tracker, Flask
API, and interactive WebGL dashboard.

Live KITTI replay predicts from `.bin` points and a saved model only. SemanticKITTI
labels are reserved for validation, training, evaluation, and optional ground-truth
comparison; native instance IDs are never used as tracker IDs.

## Features

- Point-wise semantic segmentation for drivable terrain, non-drivable terrain,
  static structures, poles, pedestrians, and vehicles.
- Tiered log-polar adaptive grid: high resolution near the ego vehicle and
  progressively coarser resolution at distance.
- Sparse dynamic-object clustering and frame-to-frame tracking with velocity
  estimates.
- Interactive GPU point-cloud dashboard with prediction, optional ground truth
  and error comparison, intensity, height, distance, and dynamic-state modes.
- Temporal point accumulation, motion trails, object inspection, adaptive-grid
  map, playback controls, presentation mode, and measured telemetry.
- Offline range-bucketed accuracy/mIoU evaluation and lightweight unit tests.

## Architecture

```text
LidarSimulator
  └─ synthetic point cloud + ground-truth semantic labels
       └─ PointSegModel
            ├─ static points ──> VariableResolutionGrid ──> GridRasterizer
            └─ dynamic points ─> ObjectTracker
                                      │
Pipeline.step() ──────────────────────┴─> live frame payload
                                             │
Flask API + WebGL dashboard <───────────────┘
```

`Pipeline` is the single orchestration layer. Every `GET /api/frame` request
advances the deterministic simulation, runs inference, fuses the adaptive grid,
updates tracks, calculates metrics, and returns the visualisation payload.

## Requirements

- Python 3.9 or newer (Python 3.10+ recommended)
- A browser with WebGL support for the dashboard

Runtime dependencies are listed in `requirements.txt`:

- Flask
- NumPy
- SciPy
- Pillow

## Installation

```bash
git clone <your-repository-url>
cd lidar_prototype

python -m venv .venv
```

Activate the environment:

```bash
# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Optional, for pytest discovery instead of the built-in runner:

```bash
python -m pip install pytest
```

## Usage

### Start the dashboard

```bash
python -m server.app
```

Open [http://127.0.0.1:5050](http://127.0.0.1:5050).

The first start uses `perception/weights.npz` when it exists. If it does not,
the pipeline trains the small local NumPy model automatically before serving
frames. The generated weights file is intentionally ignored by Git.

Dashboard controls include:

- Drag to orbit; Shift+drag to pan; use the mouse wheel to zoom.
- Select semantic, intensity, height, distance, or dynamic rendering modes.
- Switch between ego, orbit, top, front, rear, and free cameras.
- Click a tracked 3D box to inspect its measured position, dimensions, speed,
  and track state.
- Select a temporal accumulation window, toggle motion trails and adaptive-grid
  overlay, filter classes/ranges/dynamic state, and press `P` for presentation
  mode.

### Validate SemanticKITTI data

```bash
python -m data.validation --dataset-root <dataset-root> --sequence 00
```

The reader accepts standard `sequences/00/{velodyne,labels}` data, a flat
directory of `.bin` frames for label-free replay, and a read-only archive layout
containing a `velodyne` directory plus `*labels*.zip` and `*calib*.zip`.
Validation remains strict: it verifies `N x 4` float32 points, packed label
decoding, calibration, poses, and a matching point/label count per frame.

### Train on SemanticKITTI

```bash
python -m perception.train_kitti --dataset-root <dataset-root> --sequences 00 \
  --epochs 20 --batch-size 512 --learning-rate 0.05 --points-per-frame 20000 \
  --checkpoint artifacts/kitti/model/model.npz --device cpu
```

Training streams frames with a deterministic non-overlapping train/validation/test
split and saves weights, feature contract, class mapping, split, and metrics.
The CPU NumPy model uses the same `extract_features()` implementation as live
inference.

### Replay KITTI in the dashboard

```powershell
$env:DATASET_TYPE = 'kitti'
$env:KITTI_DATASET_ROOT = '<dataset-root>'
$env:KITTI_SEQUENCE = '00'
$env:TRAINED_MODEL_PATH = 'artifacts/kitti/model/model.npz'
python -m server.app
```

The dashboard uses prediction by default. GT and error modes are comparison-only
and enable only when labels are present. `MAX_INFERENCE_POINTS` and
`MAX_CLOUD_POINTS` independently bound inference and browser payload work.

### Train the local synthetic model explicitly

```bash
python -m perception.train
```

This regenerates `perception/weights.npz` from simulated training sweeps.

### Evaluate

```bash
python evaluate.py
```

The evaluation reports held-out synthetic-frame accuracy and mIoU by range,
mean inference time, and the adaptive-versus-uniform grid memory comparison.

Evaluate real data with:

```bash
python evaluate.py --dataset kitti --dataset-root <dataset-root> --sequence 00 \
  --model-path artifacts/kitti/model/model.npz
```

This reports overall accuracy, per-class precision/recall/IoU, mIoU, and
metrics by planar range: 0–10 m, 10–25 m, 25–50 m, and 50–100 m.

### Test

```bash
python run_tests.py

# Optional, when pytest is installed
python -m pytest tests -v
```

## Configuration

`config.py` is the central configuration module. It defines:

- Semantic class names and display colours.
- Dynamic and terrain class sets.
- Adaptive-grid tiers, radial resolution, sectors, and maximum range.
- Display raster resolution and range-accuracy buckets.
- The deterministic random seed and optional `DATASET_TYPE`,
  `KITTI_DATASET_ROOT`, `KITTI_SEQUENCE`, `KITTI_MAX_FRAMES`,
  `MAX_INFERENCE_POINTS`, `MAX_CLOUD_POINTS`, and `TRAINED_MODEL_PATH`
  environment settings.

The server port is `5050` and is defined in `server/app.py`.

## API

All endpoints are served by the Flask application at `http://127.0.0.1:5050`.

| Method | Endpoint | Description |
| --- | --- | --- |
| `GET` | `/` | Serves the dashboard. |
| `GET` | `/api/frame` | Advances the pipeline by one frame and returns the live perception payload. |
| `POST` | `/api/reset` | Resets simulator, adaptive-grid, tracker, and frame state. |
| `POST` | `/api/seek` | Rebuilds grid/tracker state through a requested frame. Body: `{"frame": 96}`. |
| `GET` | `/api/config` | Current source, sequences, frame bounds, and GT availability. |
| `POST` | `/api/dataset/select` | Select synthetic or KITTI source without changing payload schema. |

`/api/frame` and `/api/seek` return a JSON object containing:

```json
{
  "frame_idx": 1,
  "point_cloud": "[[x, y, z, class_id, confidence, intensity], ...]",
  "ego": {"x": 0.0, "y": 0.0, "heading": 0.0},
  "objects": "tracked dynamic-object records",
  "adaptive_grid": "tier definition and bounded observed-cell payload",
  "image_b64": "base64-encoded adaptive-map PNG",
  "latency_ms": "per-stage timings in milliseconds",
  "fps": 0.0,
  "memory": "adaptive/uniform grid memory statistics",
  "accuracy_by_range": "per-range accuracy and mIoU"
}
```

The quoted array/object descriptions above denote variable-length JSON data;
refer to `pipeline.py` for the exact response construction.

## Project structure

```text
.
├── config.py                 # Semantic classes, colours, grid tiers, constants
├── pipeline.py               # End-to-end per-frame orchestration
├── evaluate.py               # Offline synthetic evaluation
├── run_tests.py              # Dependency-light test runner
├── sim/
│   ├── world.py              # Procedural static and dynamic driving scene
│   └── lidar.py              # Ego pose and synthetic LiDAR sweep generation
├── perception/
│   ├── features.py           # Point feature extraction
│   ├── model.py              # NumPy point-wise MLP
│   └── train.py              # Training entry point
├── mapping/
│   ├── grid_engine.py        # Variable-resolution log-polar grid
│   ├── tracker.py            # Dynamic clustering and tracking
│   └── rasterizer.py         # Adaptive grid to 2.5D image
├── server/
│   ├── app.py                # Flask routes and replay control
│   ├── templates/index.html  # Dashboard layout
│   └── static/               # WebGL renderer and dashboard styles
└── tests/                    # Grid, simulator, model, tracker, pipeline tests
```

## Git hygiene

`.gitignore` excludes local environments, secrets, Python caches, test and
tool caches, build output, generated model weights, datasets, runtime output,
and common IDE/OS files. Source code, configuration, documentation, tests, and
dashboard assets remain version-controlled.

## Limitations and next steps

This is an SIH prototype with a compact CPU NumPy point MLP, rather than a
production sparse-convolution LiDAR model. Seeking far into a full KITTI
sequence correctly rebuilds persistent grid/tracker state and is therefore more
expensive than advancing a frame.
