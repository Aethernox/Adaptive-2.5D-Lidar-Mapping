# Software Architecture — Adaptive Variable-Resolution 2.5D Lidar Mapping

Status: **Target architecture (greenfield — no implementation exists yet)**
Supersedes: `adaptive-lidar-mapping-architecture.md` (research/pitch document, retained as prior art — see §3)

---

## 1. Problem Understanding

Strip the problem statement to its actual engineering requirements:

1. **Input**: a stream of raw 3D lidar point clouds (~100–150K points/sweep at ~10 Hz).
2. **Output**: a **2.5D grid** (elevation + semantic layers) around the ego vehicle that
   - is **variable resolution** (fine near the sensor, coarse far away),
   - does **not** lose overhangs/potholes the way a naive single-height 2.5D map does,
   - carries a semantic label per cell (drivable / non-drivable / static obstacle) and a **separate dynamic-object layer** (pedestrians, vehicles),
   - updates in real time (the problem statement's "high FPS" requirement) and stays spatially **coherent as the vehicle moves**.
3. **Deep learning component**: point cloud → semantic segmentation (terrain/static/dynamic) and object detection, sized so it doesn't defeat the whole point of using a variable-resolution structure (i.e., far-range points should cost less to *classify*, not just less to *store*).
4. **Demonstrable evidence**: a live dashboard and quantitative memory/latency/accuracy numbers, bucketed by range.

This is fundamentally a **real-time robotics perception system**, not a CRUD web application. That distinction drives almost every decision below: "API design" here mostly means internal streaming data contracts between pipeline stages, not REST resources over a database; "scalability" means sustaining a fixed per-frame compute/latency budget on embedded hardware, not horizontal request scaling; "idempotency/retries" apply narrowly to the control-plane (config, model updates), not to the sensor hot path.

---

## 2. Requirements and Assumptions

The problem statement leaves the following underspecified. Assumptions are stated explicitly so they can be revisited without re-deriving the whole design:

| # | Assumption | Rationale |
|---|---|---|
| A1 | Target hardware is an embedded GPU SoC (e.g., Jetson AGX Orin class), not a server GPU. | Implied by "real-time" + "autonomous navigation"; everything here still works on a server GPU, just with more headroom. |
| A2 | Single spinning/solid-state lidar, ~100–150K pts/frame @ 10 Hz. Multi-sensor fusion is a documented extension, not MVP scope. | Matches the problem statement's framing; keeps the core pipeline's contracts sensor-count-agnostic (see §5, M1). |
| A3 | A pose estimate (odometry/IMU-fusion, optionally GNSS) is available as an **input** to this system. | The problem statement scopes *mapping*, not *localization*. Building a SLAM/odometry stack is out of scope; this system consumes poses, it doesn't produce them. |
| A4 | Target is a real-time **soft** deadline (~30 FPS), not a safety-certified (ISO 26262 / ASIL) hard real-time system. | The problem statement asks for "low latency / high FPS" evidence, not functional-safety certification. This is called out explicitly so "production-ready" isn't misread as "safety-certified" — a certified deployment needs additional redundancy/certification work not addressed here. |
| A5 | No existing codebase. "Migration" (§16) means a build-out sequence, not a refactor. | Confirmed by the uploaded document being a design proposal, not code. |
| A6 | Stack: Python for training/eval/orchestration, a compiled hot path (C++/CUDA or Numba/CuPy) for the per-frame grid engine and inference. | Justified in ADR-2 (§17); avoids Python's GC/GIL on the one part of the system with a hard latency budget. |
| A7 | Dashboard consumers are engineers/evaluators on a trusted local network, not public multi-tenant users. | Sets the bar for the auth model in §10 — real but intentionally lightweight. |
| A8 | Training data: SemanticKITTI (primary, has sequential frames + odometry needed to test persistence), optional nuScenes-lidarseg fine-tuning. | Retained from the source document — both are standard, appropriate public benchmarks for this task. |

---

## 3. Current Architecture Assessment

The uploaded document (`adaptive-lidar-mapping-architecture.md`) is a strong **research pitch**: the core insight — make the variable-resolution grid the network's native coordinate system rather than a post-hoc downsample — is sound and is **retained** in this architecture. Three real weaknesses need fixing before it's a buildable system:

1. **Temporal persistence is internally inconsistent (the most important fix).** §5 of the source doc borrows the "rolling buffer" trick from Cartesian costmaps (shift indices by modulo arithmetic as the vehicle translates), but the grid it describes is **anchored to the ego vehicle's heading** (a polar grid whose sectors are defined by the vehicle's own azimuth axis). When the vehicle *rotates*, every point's sector index changes in a way index-shifting cannot fix — the modulo trick only works for axis-aligned translation, not rotation. As written, Module 3 would either silently misalign the map on every turn or require an expensive full re-bin, contradicting the "rolling buffer, O(edge) not O(area)" claim. §7 (Key Workflows) and ADR-1 (§17) fix this.
2. **No data contracts.** The document specifies *algorithms* (polar addressing, pillar encoding) but never the actual message schemas crossing module boundaries, versioning, or what happens when a downstream consumer (dashboard, logger, future planner) needs a stable interface while the model iterates. §6 and §17/ADR-6 fix this.
3. **No failure-mode design.** Sensor dropout, GPU OOM, missed deadlines, pose loss, and malformed packets are not addressed — for a system whose whole premise is safety-relevant perception, "what happens when something goes wrong" is not optional. §9 fixes this.

Everything else in the source document — the rejection of quadtree/octree (§4.1 of source), the multi-statistic cell instead of single-height z-buffer (§4.4), the sparse dynamic-object overlay instead of densifying moving objects (§4.4), the explicit confidence/uncertainty layer — is architecturally correct and is **kept unchanged**, with citations back to it in §17.

---

## 4. Proposed Architecture

```
                          ┌────────────────────────────────────────────────────────┐
                          │                     Perception Pipeline                  │
 Lidar ─┐                 │                                                          │
 driver │  PointCloudFrame │  ┌────────┐   ┌───────────┐   ┌────────────┐            │
        ├────────────────▶│  │Ingest &│──▶│  Backbone  │──▶│  Seg + Det │            │
 Pose   │                 │  │Prep    │   │ (M1)       │   │  Heads     │            │
 source ┘  Pose            │  └────────┘   └───────────┘   └─────┬──────┘            │
                          │                                       │ SegResult,        │
                          │                                       │ DetectionSet      │
                          │                                       ▼                   │
                          │                              ┌─────────────────┐          │
                          │                              │ Grid Fusion      │          │
                          │                              │ Engine (M2)      │          │
                          │                              └────────┬────────┘          │
                          │                                       │ MapSnapshot        │
                          │                              ┌────────▼────────┐          │
                          │                              │ Tracker (M3)     │          │
                          │                              └────────┬────────┘          │
                          └───────────────────────────────────────┼───────────────────┘
                                                                   │  MapSnapshot + TrackSet
                                                    ┌──────────────┼──────────────┐
                                                    ▼              ▼              ▼
                                           ┌────────────┐  ┌────────────┐ ┌─────────────┐
                                           │ Dashboard  │  │  Metrics /  │ │ Async Logger │
                                           │  (M4)      │  │ Diagnostics │ │  (bounded,   │
                                           └────────────┘  └────────────┘ │ non-blocking)│
                                                                          └─────────────┘
```

Five logical modules (M1–M4 as in the source doc, plus an explicit M5 for cross-cutting infra). Module **boundaries are process-agnostic** — they're defined by data contracts (§6), not by which runtime hosts them. This matters because of a deliberate two-topology decision (ADR-2, §17):

- **MVP / hackathon topology**: single process, multiple threads, in-memory queues. Simplest thing that hits the FPS target; zero IPC/serialization overhead; fastest to build and debug.
- **Production / vehicle-integration topology**: one process per module group, ROS2 + DDS (or equivalent pub/sub) for transport, so the perception stack composes with a real AV stack (planner, control) that almost certainly already speaks ROS2, and so a crash in M4 (dashboard) can never take down M1–M3.

Because both topologies share the same message schemas (§6), moving from one to the other is a deployment/config change, not a rewrite.

---

## 5. Components and Responsibilities

| Module | Responsibility | Notes vs. source doc |
|---|---|---|
| **M1 — Ingest & Preprocessing** | Deserialize raw packets, validate/sanitize, remove ego-vehicle self-occlusion points (near-field mask), transform points into the local reference frame using the current pose (§7), polar-bin into pillars per the active tier schedule. | New: explicit validation + near-field mask + the frame-transform-before-binning step that fixes the persistence bug. |
| **M1 — Perception Backbone** | Polar pillar encoder → sparse-conv backbone → (a) semantic segmentation head, (b) CenterPoint-style detection head. | Kept as designed in the source doc §3 — this part was already sound. |
| **M2 — Grid Fusion Engine** | Aggregate per-pillar/point predictions into the multi-statistic 2.5D cells (ground height, obstacle-top height, class, confidence, density) defined in source doc §4.4; blend into the persistent local-reference-frame grid; handle tier-boundary blending. | Kept, but now writes into a grid that is anchored to a slowly-recentered local frame (§7), not vehicle heading. |
| **M3 — Tracker** | Associate current-frame detections to existing tracks (Hungarian assignment), constant-velocity Kalman update, spawn/kill tracks, maintain the sparse dynamic-object overlay. | Kept as designed in the source doc §3.2 Head B. |
| **M4 — Dashboard & Metrics** | Render the fused grid + tracked objects; live FPS/latency/memory panel; range-bucketed accuracy panel. | Kept; interface now goes over the versioned schema in §6 instead of being tool-specific. |
| **M5 — Runtime Infra** (new, cross-cutting) | Config loader/hot-reload, health/metrics exporter, async bounded-queue logger, model/config version management. | Not present in the source doc at all; required for anything called "production-ready." |

---

## 6. Data & API Design

All cross-module messages are defined once, versioned, and shared by every consumer (training code, runtime, dashboard) — this is ADR-6 (§17): a schema drift between the model's training-time output shape and the runtime's expected input shape is one of the most common real-world causes of silent perception bugs, so the contract is the single source of truth, not something reimplemented per-language.

```protobuf
// schema v1 — illustrative; use protobuf or flatbuffers in the real repo
message PointCloudFrame {
  uint64 seq;              // monotonic per-sensor sequence number
  int64  stamp_ns;
  string frame_id;         // sensor/base_link frame this cloud is expressed in
  repeated float xyz_i;    // flat [x,y,z,intensity] x N, N inferred from length
}

message Pose {
  int64  stamp_ns;
  string reference_frame;  // e.g. "odom" — see §7 for why this matters
  double px, py, pz;
  double qx, qy, qz, qw;
  float  position_cov_trace;   // cheap confidence proxy; low-cov flags "pose unreliable" (§9)
}

message TierSchedule {           // hot-reloadable config, not hardcoded
  message Tier { float r_min, r_max, delta_r; uint32 azimuth_bins; }
  repeated Tier tiers;
  uint32 schema_version;
}

message GridCell {               // stored as Structure-of-Arrays, not this AoS shape —
  float  ground_height;          // this message is the *logical* per-cell contract
  float  obstacle_top_height;
  uint8  semantic_class;
  uint8  confidence;             // quantized 0-255
  uint16 dynamic_instance_id;    // 0 = none; non-zero indexes into TrackSet
}

message MapSnapshot {
  int64  stamp_ns;
  uint32 schema_version;
  uint32 tier_schedule_version;  // catches "grid built under an old tier schedule" bugs
  Pose   anchor_pose;            // pose of the grid's local reference frame at snapshot time
  bytes  tier0_cells;            // packed GridCell arrays, one blob per tier
  bytes  tier1_cells;
  bytes  tier2_cells;
  bytes  tier3_cells;
}

message Track {
  uint32 id; uint8 class_id;
  float cx, cy, cz, l, w, h, heading, vx, vy;
  float confidence; uint32 age_frames, misses;
}
message TrackSet { int64 stamp_ns; repeated Track tracks; }
```

**Interfaces:**

| Interface | Transport (MVP) | Transport (production) | Contract |
|---|---|---|---|
| Sensor → Ingest | in-process call / SPSC ring buffer | DDS topic, `best_effort`, depth=1 | `PointCloudFrame` |
| Pose source → Ingest, Fusion | in-process | DDS topic, `best_effort`, depth=1 | `Pose` |
| Fusion → Dashboard/Logger/Metrics | in-process pub (atomic pointer swap, §8) | DDS topic, `best_effort`, depth=1 (only the latest map matters) | `MapSnapshot`, `TrackSet` |
| Dashboard client → Runtime | WebSocket, binary-framed schema above | same | read-only stream |
| Operator → Runtime control plane | local REST/gRPC, `PUT` semantics | same | `TierSchedule` update, restart, metrics query — **idempotent by design** (§8) |

Dashboard/control-plane auth, transport security: see §10.

---

## 7. Key Workflows

### 7.1 Per-frame perception → fusion (steady state)

```
1. Ingest receives PointCloudFrame + best-available Pose.
2. Validate: range bounds, NaN/Inf strip, packet sequence check (drop stale/out-of-order frames).
3. Apply near-field self-occlusion mask (points closer than vehicle-geometry radius → discarded).
4. Transform points from sensor/base_link frame into the grid's LOCAL REFERENCE FRAME
   using the current Pose. <-- this single step is what fixes the rotation bug (§3, ADR-1):
   the grid's own axes never rotate frame-to-frame; only the point transform changes.
5. Polar-bin into pillars per the active TierSchedule (O(1) address per point, source doc §4.2).
6. Backbone forward pass → SegResult (per-pillar/point class) + raw Detections.
7. Tracker update: associate Detections to existing Tracks, Kalman predict+update, age tracks.
8. Grid Fusion: write per-cell statistics (source doc §4.4) into the persistent grid.
   - Static/terrain layers: confidence-weighted blend (EMA) with existing cell values.
   - Dynamic layer: NOT blended — tracked objects live only in TrackSet, referenced by
     dynamic_instance_id (source doc's sparse-overlay decision, retained).
9. Publish MapSnapshot + TrackSet (single-writer atomic swap, §8).
10. Dashboard/Logger/Metrics consume asynchronously; none can block step 1-9.
```

### 7.2 Grid re-centering (periodic, not per-frame)

The persistent grid's local reference frame is only **translated**, never rotated, and only when the vehicle has moved more than one inner-tier ring-width from the grid's current origin (same amortized-O(edge) trick as the source doc intended, now applied correctly to translation only): shift the SoA arrays by whole ring/sector counts, clear the newly-exposed edge cells. Outer tiers (low persistence value per source doc §5) are simply re-derived predominantly from the current sweep each frame, so they don't need re-centering logic at all.

### 7.3 Config update (control plane)

Config changes (e.g., a new `TierSchedule`) are double-buffered and swapped **only at a frame boundary** — never applied mid-frame, which would make in-flight cell addresses inconsistent. This is the one place classic idempotent-API design (`PUT`, not `POST`, with a request id) applies in this system.

---

## 8. Concurrency, Idempotency & Failure Handling

**Concurrency model (MVP topology):** pipeline-parallel across frames — while the GPU runs backbone inference for frame *N*, the CPU runs grid fusion for frame *N-1*. The only shared mutable state read by other threads (dashboard/logger) is the latest `MapSnapshot`, published via a single-writer/multi-reader **atomic pointer swap** (RCU-style) — readers never block the writer and never see a torn snapshot. No locks on the hot path.

**Concurrency model (production topology):** process/node isolation gives this for free — DDS QoS (`best_effort`, `keep_last(1)`) means a slow dashboard subscriber simply misses frames, it never backpressures the perception nodes.

**Idempotency/retries — scoped correctly:** there is no meaningful "retry a missed lidar sweep" (the next sweep simply arrives 100ms later); retry semantics apply only to:
- **Offline replay/debugging**: same rosbag + same config + same model checksum → byte-identical output. This determinism requirement is what makes regression testing (§14) possible at all.
- **Control-plane operations** (§7.3): `PUT`-style, idempotency-keyed, so a dropped ack + client retry can't double-apply a config change.

**Failure handling — every failure degrades gracefully, never crashes the process:**

| Failure | Detection | Response |
|---|---|---|
| Sensor dropout (no frames for N cycles) | watchdog timer | Hold last `MapSnapshot`, monotonically decay confidence per cell, set `stale=true` flag, raise diagnostic |
| Malformed/corrupt packet | schema/range validation in Ingest | Drop packet, increment counter metric, do not propagate downstream |
| Pose unavailable / low confidence (`position_cov_trace` above threshold) | check in Ingest before step 4 (§7.1) | Skip persistence blend for this frame (treat as single-frame/unanchored), flag map `unanchored=true` in `MapSnapshot` |
| GPU OOM / inference exceeds deadline | wall-clock check around backbone call | Skip frame's inference, reuse previous `SegResult`/`Detections` with age-based confidence decay; never block the loop waiting for a slow kernel |
| Out-of-order/duplicate frame | `seq` monotonicity check | Drop |
| Tier-boundary rounding at exact `r_min`/`r_max` | half-open interval `[r_min, r_max)` convention, unit-tested (§14) | N/A by construction |
| Concurrent config change mid-frame | double-buffered config, swap only at frame boundary (§7.3) | N/A by construction |

---

## 9. Edge Cases & Failure Scenarios

Beyond the table above, scenario-level cases worth naming explicitly because they map directly to test cases (§14):

- **Overhanging obstacle directly above drivable terrain** (the case naive 2.5D maps lose): must appear via the `obstacle_top_height` layer while `ground_height`/class in the same cell still reads drivable — validated in the multi-statistic cell design (source doc §4.4, retained).
- **Pothole**: a cell whose robust-min ground height is meaningfully below its neighbors' — the "robust min / local plane-fit" aggregation (retained from source doc) is specifically what prevents a single bad return from being mistaken for a pothole.
- **Under-sampled far cell**: point density below expected-for-range → flagged `unknown`, never defaults to `drivable`. This is the single most safety-relevant behavior in the whole system and is called out again in §18 (Definition of Done).
- **Parked vs. moving vehicle** (same detector class, different track behavior): distinguished purely by the tracker's velocity estimate over several frames, not by the segmentation head — a vehicle only enters the dynamic overlay once the tracker confirms motion, avoiding flicker between "static obstacle" and "dynamic object" for a car that's simply idling.
- **Long straight-line drive without turning**: exercises the re-centering path (§7.2) repeatedly — a natural stress test for edge-cell clearing correctness.
- **Sharp turn / roundabout**: exercises the per-frame frame-transform (§7.1 step 4) under large rotation — this is exactly the case the original rolling-buffer design would have gotten wrong (§3).

---

## 10. Security & Authorization

Scope: this module is a perception subsystem, not a full vehicle security architecture (CAN bus, actuator security are out of scope). Within scope:

- **Input sanitization at the trust boundary**: Ingest (§5) treats every incoming packet as untrusted — bounds-checked ranges, rejects malformed/oversized packets, rate-limited — the first line of defense against a malfunctioning or spoofed sensor.
- **Model & config artifact integrity**: model weights and `TierSchedule`/runtime config are loaded only after checksum verification against a signed manifest; a mismatch refuses to start rather than silently running an unexpected model.
- **Dashboard/control-plane access**: token-based auth over the WebSocket/REST interfaces (A7, §2 — local trusted network, so this is deliberately lightweight rather than full multi-tenant IAM); the dashboard connection is **read-only** — it can never issue control commands, only the separate, explicitly authenticated control-plane endpoint can.
- **Inter-process transport (production topology)**: if nodes communicate over a real network rather than shared memory, use the DDS security plugin (or TLS-wrapped transport) rather than plaintext — required the moment "production" means more than one host.
- **Supply chain**: pin all DL framework / CUDA / DDS dependency versions; scan container images for CVEs in CI (§15); no dynamic dependency resolution at deploy time on the vehicle.
- **Logged data**: recorded frames used for offline training/eval may contain pedestrians/other vehicles — apply the organization's existing data-retention/anonymization policy to the async logger's output (§11); this system doesn't invent a new policy, it just needs to respect one.

---

## 11. Performance & Scalability

**Latency budget** (targets, per source doc §7 — to be measured on real target hardware, not claimed as already-achieved):

| Stage | Uniform-grid baseline (est.) | Adaptive grid (target) |
|---|---|---|
| Ingest + polar binning | ~18 ms | ~4 ms |
| Backbone inference | ~55 ms | ~22 ms |
| Grid fusion + tracker update | ~12 ms | ~6 ms |
| Rendering (dashboard, async, off hot path) | — | — |
| **Total / FPS (hot path only)** | ~85 ms (~12 FPS) | ~32 ms (**~31 FPS**) |

**Memory** (source doc §4.3, retained — the address-scheme fix in §7 doesn't change cell counts): ~290K adaptive cells vs. 16M for a uniform 5cm grid over the same 200m×200m area, ≈**55× reduction**, ~4.6 MB vs. ~256 MB for map layers at ~16 bytes/cell.

**Scalability — two genuinely different axes, kept architecturally separate (ADR not to conflate them, §17):**
- **Per-sensor scalability** (more lidars on one vehicle): early point-level fusion into the shared local reference frame before binning, with per-point timestamp correction, so M1–M3 stay single-input-stream in design even with multiple physical sensors. Out of MVP scope (A2) but the contract in §6 doesn't need to change to support it.
- **Fleet scalability** (many vehicles' data aggregated for retraining): a batch/offline concern entirely decoupled from the real-time hot path — async logger (§8, §9) ships to object storage; nothing about scaling a fleet's data pipeline should ever touch the per-frame latency budget above. Conflating these two is a common design mistake this architecture deliberately avoids.

**Reliability**: graceful degradation table in §8 is the primary reliability mechanism — the system is designed to never stop producing *a* map, only to become less confident about it.

---

## 12. Caching & Background Processing

- **Static allocation, no per-frame malloc**: pillar/tensor buffers sized to `max_pillars`/`max_points_per_pillar` and allocated once at startup — dynamic GPU allocation per frame is a latency-jitter anti-pattern on embedded hardware and is explicitly disallowed on the hot path.
- **Model weights**: loaded once at startup, memory-mapped/pinned.
- **Config**: double-buffered, hot-reloadable (§7.3), no restart required for a `TierSchedule` change.
- **Background workers (all off the hot path, bounded queue + drop-oldest backpressure — never block M1–M3):**
  - Async disk logger (rosbag-equivalent) for offline replay/eval.
  - Telemetry/metrics batch upload.
  - Periodic (every N frames) class-distribution sampling for drift/health signal (§13) — cheap, async, never gates a frame.

---

## 13. Observability, Logging & Monitoring

- **Structured logs**, correlation id = frame `seq`.
- **Metrics** (Prometheus-style exporter): FPS, per-stage latency histograms, dropped-frame counter, points/frame, track count, confidence distribution, memory footprint (live comparison against the uniform-grid baseline — this is the "receipts, live, on screen" panel from the source doc §6, kept as a strong demo moment and now backed by an actual metrics pipeline instead of being dashboard-only).
- **Health checks**: liveness (process alive + processed a frame within 3× deadline), readiness (model loaded + first valid frame processed).
- **Diagnostics integration** (production topology): standard node-health aggregation (e.g. ROS2 diagnostics) so a fleet operator sees this module's health the same way as every other vehicle subsystem.
- **Alerting**: on sustained dropped-frame rate, sustained `unanchored`/`stale` map state, or model output class-distribution drift beyond a threshold.

---

## 14. Testing Strategy

Fast tests dominate the day-to-day loop; expensive tests are gated to nightly/pre-release so they don't slow development:

| Layer | What | Cadence |
|---|---|---|
| **Unit** | Cell-address math (property-based: address→inverse round-trips, no overlapping tier ranges, half-open-interval boundary correctness), grid-aggregation functions (robust-min ground height, majority-vote class on synthetic point sets), frame-transform correctness (identity when pose delta = 0; known rotation/translation produce the expected result), tracker (Kalman update math, Hungarian assignment on synthetic sequences, id continuity across occlusion gaps) | every commit |
| **Model evaluation** | Range-bucketed mIoU + detection AP on held-out SemanticKITTI sequences; CI gate fails the build if mIoU regresses beyond a set threshold vs. the last accepted checkpoint | nightly (compute-heavy) |
| **Integration** | Full pipeline against curated recorded rosbag fixtures with known ground truth; asserts FPS ≥ target, schema validity, no crash, no memory growth over a looped N-frame run | pre-merge |
| **Performance regression** | p95 per-stage latency and memory footprint asserted against thresholds on reference/target hardware | nightly / pre-release |
| **Fault injection** | Simulated sensor dropout, corrupt packets, GPU-OOM, pose loss — asserts the §8/§9 degrade paths trigger correctly, not just happy-path behavior | pre-release |
| **Simulation** | Scenario replay (e.g., CARLA) for specific edge cases: occluded pedestrian, overhang, pothole | pre-release, pre-hardware |

The evaluation protocol itself (range-bucketed IoU, source doc §7) is retained unchanged — bucketing by range rather than reporting a single mIoU is the correct way to satisfy the problem statement's "accuracy across varying distances" requirement, and the source doc's hypothesis (confidence should stay well-calibrated even as raw mIoU degrades with range) is exactly what §9's "under-sampled → unknown, never drivable" behavior is meant to produce — that hypothesis becomes a concrete, checkable test.

---

## 15. Deployment & Infrastructure

- **Containerization**: NVIDIA L4T base image matched to the target JetPack version; multi-stage build — a heavy training/export stage, and a lean runtime image (TensorRT engine + minimal runtime deps only) to shrink both image size and attack surface (§10).
- **CI/CD**: lint → type-check → unit test → build → model export/quantize (TensorRT/FP16) → integration test → package → (optional) hardware-in-the-loop job on a real Jetson rig → tag/publish.
- **Config-as-code**: `TierSchedule` and runtime thresholds checked into the repo as versioned files, layered `base + environment overlay` — not scattered environment variables.
- **Rollout**: shadow mode first (new model/config version runs alongside the active one, outputs logged and compared, never actuates anything) before promotion. Because this system has **no persistent database** — state is process-local/ephemeral, only ever checkpointed for offline logging — rollback is simply "redeploy the previous tagged model+config bundle," with no data migration step. This is a deliberate architectural property, not an oversight (§17, ADR: stateless-by-design).

---

## 16. Migration Plan

There is no legacy code to migrate (A5); this is a build-out sequence, restructured from the source doc's roadmap (§9 there) to fix a sequencing risk: the original roadmap starts building the model before the interfaces between modules are fixed, which risks rework once real contracts emerge. The key addition here is a **walking-skeleton phase before the ML phase**, so the team has an end-to-end demo (with a dumb classifier) well before the real model is ready, decoupling integration risk from ML research risk:

| Phase | Deliverable |
|---|---|
| **0 — Contracts** | Repo scaffold, CI skeleton, schemas from §6 locked (protobuf/flatbuffers), `TierSchedule` config format defined |
| **1 — Grid engine (pure math, no ML)** | Polar binning, tiered addressing, multi-statistic cell aggregation, local-reference-frame transform + re-centering (§7) — fully unit-testable in isolation |
| **2 — Walking skeleton** | Wire a placeholder classifier (e.g., RANSAC ground-plane fit for terrain/non-terrain) through Ingest → Grid Fusion → Dashboard, end-to-end, before the DL model exists |
| **3 — Real model** | Train/integrate the polar-pillar + sparse-conv backbone (source doc §3), swap in behind the same interface from Phase 2 — dashboard/grid code untouched |
| **4 — Tracker** | CenterPoint-style detection head + Kalman tracker, dynamic overlay wired into Grid Fusion |
| **5 — Dashboard & metrics** | Live map rendering, FPS/latency/memory panel, range-bucketed accuracy panel |
| **6 — Hardening** | Fault-injection suite (§9), deployment packaging (§15), performance tuning on target hardware |
| **7 — Stretch** | Multi-sensor fusion (§11), fleet telemetry pipeline (§11) |

---

## 17. Architectural Decisions & Trade-offs (ADRs)

| ID | Decision | Why |
|---|---|---|
| **ADR-1** | Persistent grid is anchored to a slowly-recentered **local reference frame** (translation-only re-centering); vehicle rotation is absorbed by transforming incoming points into that frame *before* binning (§7.1 step 4), not by rotating the grid. | Fixes the rotation-vs-rolling-buffer contradiction in the source document (§3). Standard practice in robotics local costmaps (persist against an `odom`-like frame, not the instantaneous vehicle-heading frame). |
| **ADR-2** | Two deployment topologies sharing one set of data contracts: single-process/multi-thread for MVP, multi-process ROS2/DDS for production vehicle integration. | Simplest thing that works for a hackathon demo; doesn't paint the team into a corner when integrating with a real AV stack later. Rejected: building custom IPC/lifecycle infra from scratch (reinvents what ROS2 already provides) and committing to ROS2 for the MVP (unnecessary complexity/dependency weight when a single process hits the FPS target). |
| **ADR-3** | Versioned, language-agnostic schemas (protobuf/flatbuffers) for every cross-module message, independent of any single middleware choice. | Prevents training/runtime shape drift; makes the dashboard interoperable (cheap to decode in a browser) regardless of whether the backend is ROS2 or a bare process. |
| **ADR-4** | Structure-of-Arrays layout for grid cells (not the AoS "packed struct" framing implied by the source doc's byte-count table). | Same ~16 bytes/cell memory number, but SoA is the correct layout for SIMD/GPU batch aggregation — an implementation refinement, not a contradiction of the source doc's memory math. |
| **ADR-5** | Log-polar tiered grid, O(1) cell addressing, no tree traversal. | **Retained from source doc (§4.1–4.2)** — rejected alternative: quadtree/octree, correctly identified as a poor fit for 10Hz online insertion due to pointer-chasing and per-frame subdivision cost. |
| **ADR-6** | Multi-statistic cells (ground height, obstacle-top height, class, confidence, density) instead of single-height z-buffer. | **Retained from source doc (§4.4)** — this is what actually prevents losing overhangs/potholes, which is the core complaint the problem statement makes about naive 2.5D maps. |
| **ADR-7** | Sparse dynamic-object list overlay instead of a dense per-cell dynamic channel. | **Retained from source doc (§4.4)** — dynamic objects are spatially sparse; densifying them into every touched cell would erode the memory win the whole adaptive-grid design exists to capture. |
| **ADR-8** | Explicit confidence/point-density layer; under-sampled cells flagged `unknown`, never defaulted to `drivable`. | **Retained from source doc (§4.4, §7)** — the single most safety-relevant property of the system; elevated to a Definition-of-Done criterion (§18) rather than left as a nice-to-have. |
| **ADR-9** | Walking-skeleton build order: heuristic classifier before the real DL model. | New — decouples integration risk (does the pipeline plumbing work?) from ML research risk (does the model hit accuracy targets?), so both can be debugged independently. |
| **ADR-10** | All disk/network IO (logging, telemetry) is async, bounded-queue, drop-oldest — never able to block the hot path. | New — a slow disk or network link must never turn into a dropped real-time frame budget. |
| **ADR-11** | No persistent database; state is process-local/ephemeral, checkpointed only for offline logging. | Deliberate simplification (§15) — this system doesn't need transactional storage, and not having one removes an entire category of migration/consistency concerns a typical service would have. |

---

## 18. Definition of Done

- [ ] All modules (M1–M5) implemented behind the versioned interfaces in §6; MVP topology (§4) runs end-to-end on recorded data.
- [ ] Unit test suite covers grid-address math, aggregation functions, and the frame-transform/re-centering logic (§14) with property-based boundary tests, all green in CI.
- [ ] Range-bucketed evaluation (§7 of source doc, retained) run on held-out SemanticKITTI sequences; results reported per bucket, not as a single mIoU.
- [ ] Confidence-degrades-before-accuracy-drops property (§9, §14) demonstrated with an explicit test, not just asserted in prose.
- [ ] Sustained target FPS demonstrated in a soak test (multi-minute loop) on target/reference hardware with no crash and no memory growth.
- [ ] Every failure mode in §8's table has a corresponding fault-injection test demonstrating graceful degradation, not a crash.
- [ ] Dashboard renders the map with the required color-coding and a live FPS/latency/memory panel; the memory-vs-uniform-grid comparison is computed live, not hardcoded.
- [ ] Turning/rotation scenario (§9) specifically exercised and shown not to misalign the map — this is the direct regression test for the bug fixed in ADR-1.
- [ ] CI pipeline (lint, type-check, unit, integration, build) green; container image builds and boots on the target hardware profile.
- [ ] This document and `MASTER_PROMPT.md` are kept in sync with the implementation as it evolves (docs-as-code, not a one-time artifact).
