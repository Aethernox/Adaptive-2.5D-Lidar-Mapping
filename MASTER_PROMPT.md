# MASTER PROMPT — Implement the Adaptive Variable-Resolution 2.5D Lidar Mapping Prototype

Use this prompt to have Claude Code implement (or re-implement, or extend)
the prototype from scratch, strictly from `SOFTWARE_ARCHITECTURE.md`. Paste
everything below to Claude Code as-is, with `SOFTWARE_ARCHITECTURE.md`
(and, if available, the reference doc `adaptive-lidar-mapping-architecture.md`)
present in the repository root.

---

## PROMPT

You are implementing a working software prototype from a design document.
`SOFTWARE_ARCHITECTURE.md` in this repository is the **source of truth** for
what to build. If a reference architecture document is also present
(e.g. `adaptive-lidar-mapping-architecture.md`), treat it only as
background context for *why* certain design choices were made — never let
it override `SOFTWARE_ARCHITECTURE.md` where the two differ; the
architecture doc has already made the simplification calls needed to make
this buildable and testable in a real environment.

### 0. Repository inspection (do this first, every time)

1. Read `SOFTWARE_ARCHITECTURE.md` in full before writing any code.
2. Check whether an implementation already exists (e.g. a `lidar_prototype/`
   directory, or the modules named in the architecture doc's "Components
   and responsibilities" table). If it exists:
   - Run its tests and dashboard first to see what already works.
   - Prefer extending/fixing over rewriting from scratch.
   - If asked to add a feature, find the smallest set of files that need
     to change per the architecture doc's module boundaries.
3. If no implementation exists, confirm what's actually available in this
   environment before assuming anything from the architecture doc's
   "Assumptions" section still holds:
   - `python3 -c "import torch"` — if this succeeds and GPU deep-learning
     tooling is genuinely usable, you may implement the real PointPillars/
     Cylinder3D-style backbone described in the reference doc instead of
     the numpy-MLP substitute — but only if you can also actually train
     and run it end-to-end here. Don't half-implement a heavier backbone
     you can't verify.
   - `pip install <pkg>` — test whether network access exists before
     assuming any package beyond what's already installed is available.
   - If nothing has changed since the architecture doc was written (no
     torch, no network), implement exactly the substitutions it documents.

### 1. Implementation sequence

Follow this order — each stage should be runnable and testable before
moving to the next one (don't build the dashboard before the grid engine
works; don't build the grid engine before you have points flowing):

1. **`config.py`** — classes, colors, tier schedule. Get this right first;
   every other module imports from it.
2. **`sim/world.py`, `sim/lidar.py`** — synthetic scene + ego motion +
   sweep generator producing `(points, true_labels, ego_pose)`. Verify by
   printing point counts / range distributions, no visualization needed yet.
3. **`perception/features.py`, `perception/model.py`, `perception/train.py`**
   — feature extraction, model, training loop. Verify: train and print
   held-out accuracy; it should clear ~90%+ on this synthetic data (if it
   doesn't, something is wrong with features/labels, not the model
   capacity — debug there first).
4. **`mapping/grid_engine.py`** — the core data structure. This is the
   highest-value, most bug-prone module — write its unit tests *alongside*
   it (tier-boundary seam behavior, out-of-range handling, memory
   accounting), not after.
5. **`mapping/tracker.py`** — dynamic object clustering + tracking.
6. **`mapping/rasterizer.py`** — grid → image. Verify by saving a PNG to
   disk and actually looking at it (view the image file) — a rasterizer
   bug that produces a technically-valid-but-wrong-looking image is easy
   to miss by only checking array shapes.
7. **`pipeline.py`** — wire the above into one `step()` call with timing
   and metrics. Verify end-to-end before touching the web layer.
8. **`server/app.py`, `server/templates/index.html`** — Flask + dashboard.
9. **`evaluate.py`** — offline range-bucketed evaluation.
10. **`tests/`, `run_tests.py`** — if not already written alongside each
    module in steps 2–7, backfill now. If `pytest` isn't installable
    (no network), write a small stdlib-only runner rather than skipping
    tests — see `run_tests.py` pattern in the architecture doc if present.

### 2. Architecture rules (do not violate these)

- **Follow `SOFTWARE_ARCHITECTURE.md`'s module boundaries exactly** —
  each listed component's file(s) and responsibility. Don't merge modules
  for convenience (e.g. don't inline the grid engine into `pipeline.py`);
  don't split one module's responsibility across files either.
- **No infrastructure beyond what the architecture doc specifies.** No
  message queues, no databases, no microservices, no Docker-compose stack,
  unless the architecture doc's scope section calls for it. A single
  Python process with a Flask dev server is the correct scale for this
  prototype.
- **Preserve the O(1) cell-addressing contract** in the grid engine
  exactly as documented (tier lookup → floor-division ring/sector, no tree
  traversal, no per-point dynamic allocation). If you must deviate,
  say so explicitly in your summary and explain why.
- **Dynamic-class points do not get densified into the grid.** They are
  tracked separately as a sparse object list. This is a specific,
  intentional architectural decision (see architecture doc §13) — don't
  "simplify" it away by writing dynamic objects into grid cells too.
- **Keep the train/inference feature path shared** (one
  `features.py`/equivalent, imported by both), so training and serving
  can never silently drift apart.
- **Any substitution you make for an unavailable dependency must preserve
  the interface** the real component would expose (e.g. a segmentation
  model must still expose `predict(points) -> (class_ids, confidence)`)
  so it can be swapped later without touching downstream modules.

### 3. Coding guidelines

- Prefer plain, direct code over abstractions/patterns that don't earn
  their complexity at this scale (no repository pattern, no dependency
  injection framework, no plugin system).
- Vectorize with numpy where the architecture doc implies performance
  matters (grid updates, rasterization) — a per-point Python loop over
  thousands of points will visibly fail the "real-time" requirement.
- Every module that has non-obvious *why* (a substitution, a
  simplification, a deliberately-rejected alternative) should say so in a
  short module-level docstring — mirror the style already in the
  architecture doc's own reasoning, don't just describe *what* the code
  does.
- Do not leave `TODO`/`NotImplementedError`/stub functions in any code
  path that's part of the core end-to-end flow. If something is genuinely
  out of scope, it must be listed in the architecture doc's "out of
  scope" / "future work" sections, not silently stubbed in code.
- Add input validation and clear error handling at real boundaries: model
  inputs, grid cell addressing (out-of-range points), empty point clouds,
  Flask request handling — not defensively everywhere.

### 4. Testing and validation (do this before declaring anything done)

Run every one of these and actually read the output — don't assume:

1. `pip install -r requirements.txt` (or confirm equivalent packages are
   already present) — check what's really usable in this environment.
2. Train the model; confirm held-out accuracy is reasonable (>85–90% on
   this synthetic data — if far lower, debug before proceeding).
3. Run the grid engine's unit tests specifically — tier-boundary seam
   behavior is the single easiest place to introduce an off-by-one that
   silently drops or double-counts points.
4. Start the dashboard server, and using `curl` (or an HTTP client):
   - Fetch `/` and confirm it returns HTML.
   - Fetch `/api/frame` multiple times in a row; confirm `frame_idx`
     increments, `fps`/`latency_ms` are populated and sane (not zero, not
     absurdly large), and `image_b64` decodes to a valid, non-trivial PNG.
   - **Decode and actually view at least one returned image** — confirm
     colors correspond to the legend, near-field looks finer-grained than
     far-field, and tracked objects appear where expected. A grid/
     rasterizer bug is very easy to miss by only checking JSON shape.
   - Fetch `/api/reset` and confirm state actually resets (`frame_idx`
     back to a starting value).
5. Run `evaluate.py`; confirm the accuracy/mIoU-by-range table is
   populated for all range buckets and the memory comparison shows a
   substantial (order-of-magnitude) reduction.
6. Run the full test suite (`pytest` if available, else the fallback
   runner) and confirm all tests pass — fix, don't ignore, any failure.
7. If background/long-running processes are used for manual testing
   (e.g. the Flask server), launch them detached (e.g. `setsid ... &`,
   fully redirected) so they survive between tool calls, and always tear
   them down (`pkill`) once validation is complete.

### 5. Debugging guidance

- If accuracy is poor: check the feature/label alignment first (most
  common bug class in this kind of pipeline is a silent index mismatch
  between points and labels after masking/filtering), then check
  normalization constants, then model capacity/training length last.
- If the rendered map looks entirely empty or entirely one color: check
  the confidence-decay/threshold constants before suspecting the grid
  math — a decay rate that's too aggressive relative to per-frame point
  density will visually look like a broken renderer even though the
  underlying cell addressing is correct.
- If FPS is far below expectations: profile per-stage latency (the
  pipeline should already report this) before optimizing blindly — it is
  usually rendering or an unvectorized aggregation loop, not model
  inference.
- If a background server process seems to "disappear" between tool calls,
  it was likely started without full detachment (missing `setsid`/output
  redirection) and was killed when its parent shell exited — restart it
  properly rather than assuming the code itself crashed.

### 6. Self-review (before declaring completion)

Go through this checklist explicitly and fix anything that doesn't pass —
don't just assert it does:

- [ ] Every module listed in the architecture doc's component table exists
      and matches its stated responsibility.
- [ ] No core-path file contains a stub, placeholder, or
      `NotImplementedError`.
- [ ] The grid engine's O(1) addressing, multi-statistic cells, and
      tier-boundary seam handling are implemented as specified, with tests
      proving it (not just asserted in prose).
- [ ] Dynamic objects are confirmed (by reading the grid-update call site)
      to never be written into the dense grid.
- [ ] The full pipeline was run end-to-end at least once outside of unit
      tests (i.e., through the actual dashboard or a standalone script),
      and its output was inspected, not just "it didn't crash."
- [ ] All tests pass, run just now, not from memory of an earlier pass.
- [ ] The memory-reduction and range-bucketed-accuracy numbers the
      problem statement asks for are both actually present in some
      runnable output (dashboard and/or `evaluate.py`), not only
      described in documentation.
- [ ] Any deviation from `SOFTWARE_ARCHITECTURE.md` made during
      implementation (because something in it turned out to be wrong,
      infeasible, or outdated relative to what's actually installed) is
      called out explicitly in your final summary, with the reasoning —
      silent deviation is worse than a documented one.

### 7. Completion criteria

Declare the task complete only when **all** of the following are true, and
state explicitly in your summary how each was verified:

1. Every core workflow in `SOFTWARE_ARCHITECTURE.md` §8 ("Core Workflows")
   runs successfully end-to-end.
2. All unit tests pass.
3. The dashboard is confirmed working via real HTTP requests (not just
   "the code looks right") — including a visually inspected map image.
4. `evaluate.py` (or equivalent) produces the range-bucketed accuracy and
   memory-comparison numbers the problem statement explicitly requires.
5. Documentation (`SOFTWARE_ARCHITECTURE.md`, and this file if you changed
   the design) reflects what was actually built — if you deviated from
   the plan, update the docs to match reality rather than leaving them
   describing something that no longer exists in the code.
6. No component was left half-implemented "to save time" — if scope had
   to be cut, it was cut at a clean module boundary and recorded in the
   architecture doc's scope/future-work sections, not left as broken code.

Do not report the prototype as complete or working without having
actually run it and observed the output described above.
