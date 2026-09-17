# Adaptive Variable-Resolution 2.5D Lidar Mapping — Software Architecture

**Status:** Working prototype, implemented and verified (see "Definition of Done").
**Code:** `lidar_prototype/` (delivered alongside this document).

---

## 1. Problem Understanding

Autonomous vehicles need to perceive their surroundings from raw Lidar in
real time. Full 3D point clouds are too expensive to process/store at high
uniform resolution; plain 2D occupancy grids throw away height information
needed for curbs, potholes, and overhangs. The task asks for a **foveated,
variable-resolution 2.5D map**: high detail near the vehicle, coarser detail
far away, built from a deep-learning perception pipeline that separates
terrain (drivable/non-drivable), static obstacles (walls, poles), and
dynamic objects (pedestrians, vehicles), with a live dashboard and
performance/accuracy metrics as evidence.

An existing design document (`adaptive-lidar-mapping-architecture.md`,
provided as a starting point) proposes a production-grade version of this:
a PointPillars/Cylinder3D-style sparse-conv backbone with a CenterPoint
detection head and Kalman tracking, trained on SemanticKITTI/nuScenes, run
on Jetson-class hardware, visualized via Foxglove/rerun. That is a sound
*target* architecture, but it assumes GPU deep-learning frameworks, real
driving datasets, and robotics visualization tooling — none of which are
available in this sandboxed environment (no network access to install
`torch`/`spconv`/ROS, no dataset downloads). Per the task instructions,
the existing document is a starting point, not a constraint: this
prototype keeps its core **data-structure idea** (the tiered log-polar
grid) intact and faithful, while substituting lighter-weight,
dependency-free components everywhere a heavy ML/robotics stack was
assumed, so the whole thing is genuinely runnable and testable here.

---

## 2. Core Requirements (from the problem statement)

1. **Terrain analysis** — classify drivable vs. non-drivable surface.
2. **Object detection** — classify static obstacles (walls, poles) and
   dynamic objects (pedestrians, vehicles).
3. **Adaptive spatial representation** — a non-uniform grid whose cell size
   grows with range, without alignment errors or data loss during 3D→2.5D
   projection.
4. **Real-time visualization dashboard** — color-coded 2.5D map, showing
   memory savings vs. a uniform high-resolution grid.
5. **Performance metrics** — FPS/latency and per-distance-bucket
   classification accuracy.

---

## 3. Assumptions

These are the concrete, documented judgment calls made to turn the problem
statement into something buildable and testable in this sandbox:

| # | Assumption | Rationale |
|---|---|---|
| A1 | No GPU/`torch`/`spconv`/ROS available, no network to install them or download SemanticKITTI/nuScenes. | Verified in-sandbox: `pip install torch` fails (no network), only `numpy`, `scipy`, `Pillow`, `Flask`, `matplotlib` are preinstalled. |
| A2 | A **procedurally generated synthetic world** (ground plane with a curb and pothole, static walls/poles, a few pedestrians/vehicles on simple trajectories) stands in for a real driving-scene dataset. | Gives exact, always-available ground-truth labels for both training and evaluation, at the cost of not testing on real sensor noise/occlusion. Documented as the prototype's principal limitation (see §12). |
| A3 | The "deep learning model" is a **small fully-connected network trained with manual numpy backprop** (5→64→32→6, ReLU, softmax cross-entropy), not PointNet++/a sparse-conv backbone. | Preserves the *interface* a real backbone would expose — `predict(points) -> (class_ids, confidence)` — so a real PointPillars/Cylinder3D model can be substituted later (see §11) without touching the grid engine, tracker, or dashboard. Achieves ~95% point-wise test accuracy on synthetic data, enough to demonstrate a genuine trained-model-in-the-loop pipeline, not a hand-coded classifier. |
| A4 | Dashboard is a **Flask web app + a raster PNG image refreshed by polling**, not Foxglove/rerun/ROS2 pub-sub. | No ROS/robotics-viewer stack available; a plain web dashboard is simpler, has zero extra install dependencies, and is directly demonstrable in a browser. |
| A5 | Object tracking uses **grid-based connected-components clustering + nearest-neighbor constant-velocity association**, not a CenterPoint heatmap head + Kalman filter + Hungarian assignment. | Much simpler to implement correctly without a training/annotation pipeline for box regression, while still producing the required sparse object list (id, class, position, velocity) that separates "dynamic" from "static-but-detected." |
| A6 | The rolling-buffer / odometry-fused temporal persistence from the reference doc's Module 3 is simplified to a **per-cell confidence exponential decay** (cells fade out if not re-observed, rather than a full rolling-index-shift buffer + IMU pose fusion). | The point being demonstrated — that the map persists across frames rather than being wiped every frame, and that staleness is tracked — is preserved; the specific O(edge)-cost rolling-buffer optimization is a performance refinement, not a correctness requirement, for a CPU prototype at this frame count. Documented as future work (§11). |
| A7 | Tier schedule is the same *shape* as the reference doc (4 tiers, 0–10/10–25/25–50/50–100 m, ~5cm→50cm cells) but with somewhat coarser azimuth binning (360/180/90/45 sectors instead of 1024/512/256/128) to keep the cell count and rendering cost CPU-friendly for a live dashboard. | Preserves the "fine near, coarse far" property and the O(1) addressing scheme exactly; only changes the *resolution knob*, which is explicitly called out as tunable in the reference doc. Still yields a **~157× memory reduction** vs. the uniform-grid baseline (exceeds the reference doc's own ~55× estimate). |
| A8 | "Real-time" for this CPU-only prototype means **~15–20 FPS** end-to-end (measured, not simulated), not the Jetson-class ~27 FPS target in the reference doc. | Reasonable for a single-threaded Python/numpy CPU pipeline; the reference doc's own numbers are explicitly marked "design targets, not measured numbers" on GPU-class embedded hardware, which isn't available here. |

---

## 4. Prototype Scope

**In scope (implemented and working):**
- Synthetic Lidar sweep generation with ground-truth semantic labels.
- Trained point-wise semantic segmentation (6 classes) — a real, gradient-descent-trained neural network in the loop, not a lookup table.
- Tiered log-polar variable-resolution grid engine: O(1) cell addressing, multi-statistic cells (ground height, obstacle-top height, class, confidence, count), no tier-boundary seams.
- Sparse dynamic-object detection + frame-to-frame tracking (id, class, position, velocity), kept out of the dense grid.
- Live web dashboard: color-coded 2.5D map, FPS, per-stage latency breakdown, adaptive-vs-uniform memory comparison, range-bucketed accuracy/mIoU.
- Offline evaluation script producing the same range-bucketed metrics on a held-out batch.
- Unit tests covering grid addressing/seams/memory accounting, the model, the tracker, and full pipeline execution.

**Out of scope (explicitly deferred, see §11):**
- Real sensor data ingestion (PCAP/ROS bag replay), real datasets (SemanticKITTI/nuScenes), GPU inference, TensorRT/FP16 deployment.
- Full rolling-buffer + IMU/odometry pose-fusion persistence.
- CenterPoint-style anchor-free detection head + Kalman filter + Hungarian assignment.
- Multi-vehicle / multi-sensor fusion, HD-map integration, path planning consumption of the map.

---

## 5. Target Architecture

```
sim.world / sim.lidar               perception.model (trained MLP)      mapping.grid_engine
┌───────────────────┐   points      ┌─────────────────────────┐  class,  ┌──────────────────────────┐
│ Synthetic ego      │ ──(x,y,z,i)─▶│ Point-wise segmentation  │ conf ──▶│ Tiered log-polar grid:    │
│ motion + Lidar     │   (N,4)      │ (terrain/static/dynamic) │         │ O(1) address, multi-stat  │
│ sweep generator    │               └─────────────────────────┘         │ cells, confidence decay   │
└───────────────────┘                          │                         └──────────────────────────┘
                                                │ dynamic-class points               │
                                                ▼                                    ▼
                                     mapping.tracker                        mapping.rasterizer
                                ┌───────────────────────┐            ┌───────────────────────────┐
                                │ Cluster + NN tracking  │            │ Grid + objects -> RGB PNG │
                                │ -> sparse object list  │            └───────────────────────────┘
                                └───────────────────────┘                          │
                                                │                                    ▼
                                                └──────────────────▶  pipeline.Pipeline.step()
                                                                              │
                                                                              ▼
                                                                     server.app (Flask)
                                                                     /api/frame, /api/reset
                                                                              │
                                                                              ▼
                                                                     Browser dashboard
                                                                (map image + live metrics)
```

Everything is a single Python process, single in-memory pipeline object,
no queues/services/databases — appropriate for a CPU prototype at this
scale (see "Implementation Rules": avoid unnecessary infrastructure).

### Components and responsibilities

| Module | File(s) | Responsibility |
|---|---|---|
| **Config** | `config.py` | Single source of truth for classes, colors, tier schedule — everything else imports from here so the grid, model, and rasterizer never disagree. |
| **World / Sensor sim** | `sim/world.py`, `sim/lidar.py` | Procedural scene + ego motion; produces `(points, true_labels, ego_pose)` per timestep, in the ego frame. |
| **Perception (features)** | `perception/features.py` | Raw points → normalized per-point feature vectors, shared by train and inference so they can never drift apart. |
| **Perception (model)** | `perception/model.py`, `perception/train.py` | Trained MLP segmentation model; `PointSegModel.predict(points) -> (class_ids, confidence)` is the stable interface. |
| **Grid engine** | `mapping/grid_engine.py` | The core data structure: `VariableResolutionGrid.locate()` (O(1) addressing), `.update()` (3D→2.5D fusion), `.memory_stats()`. |
| **Tracker** | `mapping/tracker.py` | Clusters dynamic-class points, tracks centroids frame-to-frame, returns the sparse object overlay. |
| **Rasterizer** | `mapping/rasterizer.py` | Grid + objects → RGB image, using a precomputed pixel→cell lookup table (built once, O(pixels) per frame). |
| **Pipeline** | `pipeline.py` | Orchestrates one frame end-to-end, times each stage, computes accuracy/memory/FPS, returns the API response payload. |
| **Dashboard** | `server/app.py`, `server/templates/index.html` | Flask endpoints + polling JS front end. |
| **Evaluation** | `evaluate.py` | Offline range-bucketed accuracy/mIoU + memory report on held-out frames. |
| **Tests** | `tests/`, `run_tests.py` | Unit tests; `run_tests.py` is a pytest-compatible fallback runner (no `pytest` available offline). |

---

## 6. Data Models

**Point cloud frame** (`sim.lidar.LidarSimulator.sweep`):
```
points:       (N, 4) float32   — x, y, z, intensity, in the EGO frame
true_labels:  (N,)   int        — ground-truth class id (sim only)
ego_x/y, heading, t              — ego pose at this frame
dyn_meta                        — ground-truth actor states (sanity-check use)
```

**Semantic classes** (`config.CLASSES`, id = index):
`0 drivable_terrain · 1 non_drivable_terrain · 2 static_structure ·
3 static_pole · 4 dynamic_pedestrian · 5 dynamic_vehicle`

**Grid cell** (per tier, per `(ring, sector)`, `config.N_LAYERS = 5` float32 channels):
```
[0] ground_height        mean z of terrain-labeled points in the cell
[1] obstacle_top_height  max z of non-terrain points in the cell
[2] class_id             confidence-weighted majority vote (float, stored as int-valued)
[3] confidence           observed / expected point density at this range, decayed over time
[4] point_count          raw point count this frame
```
Cell address: `tier = lookup(r); ring = floor((r - tier.r_min)/tier.dr); sector = floor(theta/tier.dtheta) mod n_sectors` — O(1), matching the reference doc exactly.

**Tracked object** (`mapping.tracker.ObjectTracker`):
```
{id, kind ("pedestrian"|"vehicle"), x, y, vx, vy, w, d, age, misses}
```
Kept as a small Python list, never written into the dense grid.

**API response** (`GET /api/frame`):
```json
{
  "frame_idx": 12, "image_b64": "...", "ego": {"x":.., "y":.., "heading":..},
  "n_points": 10248, "objects": [ ... ], "latency_ms": {"sense_ms":.., "infer_ms":.., "fuse_ms":.., "track_ms":.., "render_ms":..},
  "total_ms": 55.1, "fps": 18.1,
  "memory": {"adaptive_cells":.., "adaptive_bytes":.., "uniform_cells":.., "uniform_bytes":.., "reduction_factor":..},
  "accuracy_by_range": [ {"range":"0-10m","n_points":..,"accuracy":..,"miou":..}, ... ],
  "class_names": [...]
}
```

---

## 7. APIs / Interfaces

| Interface | Contract |
|---|---|
| `PointSegModel.predict(points: (N,4)) -> (class_ids: (N,) int, confidence: (N,) float)` | The one interface a real PointPillars/Cylinder3D/torch model must satisfy to be swapped in. |
| `VariableResolutionGrid.locate(x, y) -> (tier_idx, ring, sector)` | Static/vectorized O(1) addressing, used identically by `update()` and the rasterizer's pixel LUT. |
| `VariableResolutionGrid.update(points, class_ids, confidences)` | Fuses one frame's (already ego-frame) points into the persistent grid. |
| `VariableResolutionGrid.memory_stats() -> dict` | Adaptive vs. uniform cell/byte counts — the dashboard's headline number. |
| `ObjectTracker.update(points, class_ids, t, dt) -> list[dict]` | Returns the current sparse dynamic-object list. |
| `GET /api/frame` | Advances the pipeline one frame; returns the full JSON payload above. |
| `POST /api/reset` | Resets simulation time, grid, and tracker state. |

---

## 8. Core Workflows

**A. Training** (`perception/train.py`): generate labeled synthetic frames
across many `(t, ego-pose)` samples → minibatch SGD on the MLP → save
`perception/weights.npz`. Run once; the pipeline auto-trains on first use
if weights are missing.

**B. Live frame processing** (`pipeline.Pipeline.step()`, called by
`/api/frame`):
1. `sim.lidar.sweep(t)` → raw points + true labels (ego frame).
2. `model.predict(points)` → per-point class + confidence.
3. Dynamic-class points are **excluded** from the grid update (see A→grid
   dataflow diagram) and instead go to the tracker; static/terrain points
   go to `grid.update()`.
4. `tracker.update()` → sparse object list with ids/velocities.
5. `rasterizer.render()` → RGB image (grid cells + object boxes/velocity arrows).
6. Metrics computed: per-stage wall-clock latency, FPS, memory stats
   (static, doesn't depend on frame content), range-bucketed accuracy
   (using the frame's ground truth, since this is a synthetic-data
   prototype with no separate held-out real dataset at runtime).
7. `t += SIM_DT`; response JSON returned to the browser.

**C. Offline evaluation** (`evaluate.py`): same as B steps 1–2 on a
held-out batch (different seed/time range than training), aggregated into
a range-bucketed accuracy/mIoU table plus the memory comparison.

---

## 9. Important Edge Cases

| Edge case | Handling |
|---|---|
| Empty point cloud (no returns this frame) | `PointSegModel.predict` and `VariableResolutionGrid.update` both short-circuit on `len(points) == 0`; covered by `test_model_predict_empty_points` / `test_update_populates_cells_and_no_crash_on_empty_frame`. |
| Point exactly on a tier boundary | Tiers are half-open `[r_min, r_max)`, assigned in order with a "claimed" mask so exactly one tier claims each point — no double-count, no gap; covered by `test_locate_no_seam_gaps_at_tier_boundary`. |
| Point beyond max range (100m) | `locate()` returns `tier_idx = -1`; such points are dropped from the grid (never crash); covered by `test_locate_out_of_range_is_dropped`. |
| Cell never observed | Confidence stays at 0, rendered as the distinct "empty" color rather than a false class — this is the safety-relevant "flag as unknown, don't guess" property called out in the reference doc. |
| Cell observed once, then not re-observed | Confidence decays exponentially each frame it's missed (not instant wipe, not permanent memory) — models sensor/occlusion staleness. |
| Under-sampled far cells | Confidence = observed/expected density is naturally low at range (bigger cells, sparser returns), so far cells trend toward "unknown" rather than a confident wrong label — directly demonstrates the reference doc's stated hypothesis (mIoU drops with range, but the system should be "less certain rather than confidently wrong"), and `evaluate.py`'s output confirms this pattern on this prototype's data. |
| Curb / pothole (non-flat terrain) | Ground truth height field includes a raised curb and a local depression; cells store both ground height and obstacle-top height so a 2.5D max-z collapse never occurs. |
| Dynamic object briefly occluded / no cluster this frame | Tracker keeps a track alive for up to 3 consecutive missed frames (`misses > 3` before deletion) rather than dropping ids on a single gap. |
| Concurrent dashboard requests | Flask app uses a single `threading.Lock()` around `pipeline.step()` — simple, correct, sufficient for a single-user prototype (see §11 for multi-user scaling). |

---

## 10. Error Handling

- **Feature/model interface boundary**: `perception/features.py` is the *only* place normalization constants live; both training and inference import it, preventing train/serve skew.
- **Grid bincount safety**: all per-cell aggregations use `np.divide(..., where=...)` / explicit `has_data` masks rather than dividing by zero-count cells.
- **Startup robustness**: `pipeline.py`'s `_load_or_train_model()` auto-trains if `weights.npz` is missing, so the dashboard is runnable from a clean checkout with one command.
- **Server-level**: Flask's built-in error responses cover malformed requests; the two endpoints are read-mostly (`/api/frame`) or idempotent (`/api/reset`), so no partial-state-corruption paths exist under normal use.
- **Test-time validation**: `mapping/grid_engine.py`'s class layer is asserted to stay within `[0, NUM_CLASSES)` (`test_cell_class_layer_within_valid_range`) — guards against a silent aggregation bug producing an invalid class id that would crash the rasterizer's color LUT.

---

## 11. Testing Approach

- **Unit tests** (`tests/test_grid_engine.py`, `tests/test_pipeline.py`, 11 tests total, all passing):
  - Grid: O(1) addressing correctness, no seam gaps/double-counts at tier boundaries, out-of-range dropping, memory-reduction factor, cell-update correctness on a synthetic frame, class-id validity invariant.
  - Model: prediction shapes/ranges, empty-input handling.
  - Sim: point cloud shape/range invariants.
  - Tracker: a synthetically moving cluster is tracked with a plausible velocity sign after a few frames.
  - Pipeline: a full `step()` call returns a well-formed response and advances state correctly on a second call (grid persistence / tracker continuity).
- **No `pytest` available offline** (no network to install it in this sandbox) — tests are written in plain pytest-compatible style (bare `test_*` functions + `assert`) and also run via `run_tests.py`, a tiny custom runner; `pytest tests/ -v` will work unmodified wherever pytest is installed.
- **Offline evaluation** (`evaluate.py`) is the "accuracy across varying distances" validation the problem statement explicitly asks for, run on a held-out batch (different RNG seed/time range than training data) — this is the closest thing to an integration/regression test for model quality.
- **Manual end-to-end verification performed**: trained the model (~95% test accuracy), ran the Flask server, fetched `/`, multiple `/api/frame` calls (confirmed FPS ~15-20, frame index advancing, image decodes to a visually correct color-coded map — inspected directly), and `/api/reset` (confirmed frame index resets to 1).

---

## 12. Future Scalability Considerations

These map the prototype's simplifications back onto the reference
document's original (production-grade) design, so the path forward is
explicit rather than a rewrite:

| Prototype today | Production evolution | What has to change |
|---|---|---|
| Numpy MLP, `predict(points)` interface | PointPillars/Cylinder3D sparse-conv backbone (as in the reference doc), torch + spconv/torchsparse, trained on SemanticKITTI/nuScenes | Only `perception/model.py` — the interface (`points -> class_ids, confidence`) is already the contract the rest of the system expects. |
| Confidence-decay persistence | Full rolling-buffer inner tier (O(edge) update) + IMU/odometry-fused outer-tier re-anchoring | `mapping/grid_engine.py`'s update/decay logic; the cell-address scheme and cell layout don't need to change. |
| Cluster + nearest-neighbor tracker | CenterPoint-style heatmap head + Kalman filter + Hungarian assignment | `mapping/tracker.py`, plus a detection head added to the model; the sparse-object-list contract to the rasterizer/dashboard stays the same. |
| Flask + polling PNG | WebSocket push / Foxglove / rerun.io pub-sub, for lower-latency multi-client streaming | `server/` only; `pipeline.step()` output is already a clean serializable payload. |
| Single-process, single-lock | Separate perception process (GPU) from a lightweight mapping/serving process, if the model becomes GPU-bound | Would introduce a process boundary/queue between `perception` and `mapping` — deliberately *not* done now, per the instruction to avoid premature infrastructure. |
| Synthetic world | Real Lidar (PCAP/ROS bag) ingestion, SemanticKITTI/nuScenes training/eval splits | `sim/` is isolated behind the same `sweep(t) -> points, labels` shape a real dataset loader would expose. |
| 360/180/90/45 azimuth bins | 1024/512/256/128 (reference doc's schedule), once rendering/compute is GPU-accelerated | `config.TIERS` — a one-line change; everything else (grid, rasterizer LUT, tests) is parametric on this table already. |

---

## 13. Key Architectural Decisions

1. **Preserve the tiered log-polar grid exactly** — this is the part of the
   reference design that genuinely solves the stated "hard data-structure
   problem" (O(1) addressing, exact-multiple ring widths avoiding seams,
   multi-statistic cells avoiding 2.5D data loss). Nothing about it
   requires a GPU or a specific ML framework, so it was implemented as
   specified rather than simplified.
2. **Substitute, don't fake, the deep-learning component** — rather than a
   hand-written rule-based classifier pretending to be "the model," the
   prototype trains an actual neural network with gradient descent on
   labeled data and evaluates it out-of-sample. This keeps the "deep
   learning pipeline" requirement honest while working around the missing
   `torch`/GPU dependency.
3. **Single-process, in-memory pipeline** — no message queue, no database,
   no microservices. A Lidar pipeline processing one sweep at a time on a
   single machine has no need for distributed infrastructure at prototype
   scale, and adding it would only obscure the actual novelty (the grid
   engine) being demonstrated.
4. **Dynamic objects never enter the dense grid** — implemented exactly as
   the reference doc argues (§4.4): a sparse object list is both cheaper
   and semantically correct (a moving pedestrian isn't "terrain" at any of
   the cells it passes through over time).
5. **Config as single source of truth** — `config.py` is imported by the
   sim, model, grid, and rasterizer specifically so that class ids/colors/
   tier definitions can never drift apart between modules, a common source
   of subtle bugs in pipelines with this many stages.

---

## 14. Definition of Done

All of the following were verified in this sandbox before calling the
prototype complete (not just asserted):

- [x] `python3 -m perception.train` runs to completion and produces a model with **~95% held-out point-wise accuracy**.
- [x] `python3 -m server.app` starts a working Flask server; `GET /`, `GET /api/frame` (multiple times, confirming state advances and FPS is reported), and `POST /api/reset` (confirmed frame index resets) were all exercised via `curl` and their responses inspected.
- [x] A rendered map frame was decoded and **visually inspected** — confirmed correct color-coded terrain (drivable/non-drivable), static obstacles, tracked dynamic-object boxes with velocity arrows, and visibly coarser cell resolution at range.
- [x] `python3 evaluate.py` runs and produces the range-bucketed accuracy/mIoU table required by the problem statement, on held-out data — output matches the reference doc's own hypothesized shape (accuracy stays high, mIoU degrades with range).
- [x] Memory comparison: **~157× reduction** (≈2.04 MB adaptive vs. ≈320 MB uniform-grid equivalent) — the explicit "significant reduction" the dashboard requirement asks for.
- [x] `python3 run_tests.py` — **11/11 tests pass**, covering grid addressing/seams/memory, model, sim, tracker, and full pipeline execution.
- [x] No placeholder/TODO logic remains in any core-path module (`sim`, `perception`, `mapping`, `pipeline`, `server`).
