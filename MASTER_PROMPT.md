# Master Prompt — Implement the Adaptive Lidar Mapping System

**Purpose**: paste this into Claude Code (or re-issue it at the start of any implementation session) to drive the build against `SOFTWARE_ARCHITECTURE.md`. It's written to be reusable across the whole project, not a one-shot task — re-run it whenever you start a new phase or a new session.

---

## 0. Source of Truth

`SOFTWARE_ARCHITECTURE.md` (in the repo root) is the authoritative design. If anything below conflicts with it, the architecture doc wins — update it first, then implement, never the reverse. If you (Claude Code) find that implementation reality requires deviating from the architecture doc, **stop and update the doc in the same change**, with a short rationale, before writing more code. Docs and code drift apart by default; don't let that happen here.

Before writing or changing any code, in this order:

1. **Read `SOFTWARE_ARCHITECTURE.md` in full.** Identify which Phase (§16 of that doc) the repo is currently in.
2. **Inspect the actual repository state**: directory layout, existing modules, existing tests, existing CI config, any `TODO`/`FIXME` markers, and the last few commits' intent. Do not assume the repo matches the architecture doc's Phase table — verify.
3. **Trace data flow for whatever you're about to touch**: which module produces the message, which consumes it, which schema version is in play (§6 of the architecture doc). If a schema field's real usage doesn't match §6, that's a signal to fix the doc or fix the code — decide which is correct and say so before proceeding.
4. **Validate assumptions against the codebase**, not against the architecture doc's prose — e.g., if §2's assumption A6 (stack: Python + compiled hot path) isn't actually what's in the repo, don't silently work around it; flag it.
5. Only then start implementing.

---

## 1. Implementation Sequence

Follow the Phase order in `SOFTWARE_ARCHITECTURE.md` §16 (`Migration Plan`) unless the repo is already past Phase 0. Within whichever phase is current:

- **Phase 0 (Contracts)**: define the message schemas from §6 first, as actual code (protobuf/flatbuffers definitions or equivalent typed dataclasses), before any pipeline logic. Nothing in later phases should invent an ad hoc shape for a message that §6 already defines.
- **Phase 1 (Grid engine)**: implement polar binning, tiered addressing, and multi-statistic cell aggregation as **pure, deterministic functions** with no ML dependency — this phase should be fully unit-testable (§14 of the architecture doc) without a trained model. Implement the local-reference-frame transform and re-centering logic from §7 here; this is the fix for the rotation bug identified in ADR-1 (§17) — get it right and tested before anything is built on top of it.
- **Phase 2 (Walking skeleton)**: wire a placeholder/heuristic classifier through the full pipeline end-to-end (Ingest → Grid Fusion → Dashboard) before the real model exists. The goal of this phase is a working, demoable pipeline, not classification accuracy.
- **Phase 3 (Real model)**: implement/train the polar-pillar + sparse-conv backbone (architecture doc §5, M1) and swap it in behind the exact interface Phase 2 established. Do not change the Grid Fusion or Dashboard code to accommodate the model — if you find yourself needing to, the interface from Phase 2 was wrong; fix the interface deliberately and update §6.
- **Phase 4 (Tracker)**: detection head + Kalman tracker + dynamic overlay.
- **Phase 5 (Dashboard & metrics)**: live rendering + the FPS/latency/memory panel described in architecture doc §13.
- **Phase 6 (Hardening)**: implement every failure-handling behavior in architecture doc §8/§9 as actual code paths with fault-injection tests, not just documentation.
- **Phase 7 (Stretch)**: multi-sensor fusion, fleet telemetry — only after Phases 0–6 are done and validated.

Work incrementally within a phase: small, buildable, testable commits. Never leave the repo in a state that doesn't build or doesn't pass its existing tests, even mid-phase.

---

## 2. Coding & Architecture Rules

- **Modify existing code over duplicating it.** If a function/module already does most of what's needed, extend it; don't create a parallel implementation "to be safe."
- **Remove obsolete code when replacing an approach.** If Phase 3's real model replaces Phase 2's heuristic classifier, delete the heuristic (or clearly relocate it to a test-fixtures/baseline location) — don't leave dead code paths.
- **Keep interfaces, types, schemas, validation, and error handling consistent** with §6 and §8 of the architecture doc across every module that touches them. A schema change in one place is a schema-version bump everywhere, not a silent divergence.
- **Don't introduce a new library, framework, or service unless the current stack genuinely can't solve the problem.** In particular: don't add a message broker, database, or ML framework beyond what §2/A6 of the architecture doc already specifies without first updating that doc with the justification.
- **No dynamic allocation on the per-frame hot path** (architecture doc §12) — buffers for pillars/tensors are sized once at startup. If you add a new per-frame data structure, size it statically or explain in the PR/commit message why that's not possible.
- **No blocking IO on the hot path** (architecture doc §8, §12) — logging, telemetry, and metrics export are async, bounded-queue, drop-oldest. If you're tempted to write a `print`/log call inside the Ingest→Fusion→Tracker path that isn't already going through the async logger, don't.
- **Follow the half-open interval convention** (`[r_min, r_max)`) for tier boundaries exactly as specified in architecture doc §7/§9 — this is a named edge case with a named test; don't reintroduce an off-by-one at tier seams.
- **Graceful degradation, never a crash**, for every failure mode in architecture doc §8's table. A new failure mode you discover during implementation gets added to that table (and a test), not just handled inline and forgotten.
- **Config changes apply only at frame boundaries** (double-buffered, architecture doc §7.3) — never mutate live tier-schedule/config state mid-frame.

---

## 3. Validation Requirements

Run these before considering any change complete — adapt the exact commands to whatever the repo's actual tooling turns out to be (inspect `pyproject.toml`/`CMakeLists.txt`/CI config first rather than assuming):

- **Lint**: Python (`ruff`/`flake8`) and, if a compiled hot path exists, the C++/CUDA equivalent (`clang-tidy` or the repo's configured linter).
- **Type-check**: `mypy` (or the repo's configured type checker) on all Python; equivalent static analysis on any compiled code.
- **Build**: the full build must succeed cleanly, including the compiled hot-path module if present.
- **Unit tests**: run the full suite, not just tests near your change — the grid-math and frame-transform tests in particular (architecture doc §14) are cheap and must always be green.
- Fix issues discovered during validation yourself, in the same change — don't report a lint/type/test failure and stop; resolve it, then re-run validation to confirm.

---

## 4. Testing Requirements

Follow the test pyramid in architecture doc §14 exactly — don't skip a layer because it's inconvenient, and don't add heavyweight tests where a unit test would do:

- New pure-math or aggregation logic → **unit test**, including boundary/property-based cases (tier seams, identity transforms, etc.), runs on every commit.
- New/changed model behavior → update the **range-bucketed evaluation** harness and check the regression gate (fail if mIoU/AP drops beyond the configured threshold vs. the last accepted checkpoint).
- New/changed pipeline wiring → **integration test** against the repo's curated recorded fixtures; assert schema validity, FPS ≥ target, no crash, no memory growth over a looped run.
- New/changed failure-handling code → a corresponding **fault-injection test** (sensor dropout, corrupt packet, GPU-OOM/deadline miss, pose loss) proving graceful degradation per architecture doc §8 — not just a happy-path test.
- Never mark work complete on the strength of "it ran once and looked right" — the soak-test/no-memory-growth and fault-injection requirements in the Definition of Done (architecture doc §18) are real gates, not aspirational.

---

## 5. Security & Performance Requirements

- Every new external input (packet field, config value, model output tensor) gets bounds/schema validation at the trust boundary it crosses (architecture doc §10) — assume malformed input, don't assume the upstream component is well-behaved.
- Any new model weight file or config bundle is checksum/signature-verified before load; a mismatch must refuse to start, not warn and continue.
- Any new dashboard/control-plane endpoint follows the read-only-dashboard / authenticated-control-plane split in architecture doc §10 — never let a visualization/query path also be able to mutate runtime state.
- Any new per-frame code path is measured against the latency budget in architecture doc §11 (per-stage, not just end-to-end) — if you add work to the hot path, show the before/after latency, don't assume it's negligible.
- Any new background/IO work goes through the bounded async queue pattern (architecture doc §12) — never a new unbounded queue, never a new synchronous disk/network call from a hot-path thread.

---

## 6. Self-Review & Completion Criteria

Before declaring a phase, task, or PR complete, go back through this checklist against the actual diff — not from memory:

1. Re-read the relevant section(s) of `SOFTWARE_ARCHITECTURE.md` and confirm the implementation matches what's written — if it doesn't, either the code or the doc is wrong; fix whichever is incorrect, in the same change.
2. Confirm every item in §3 (Validation) above actually passed — don't state completion without having run them.
3. Confirm every item in §4 (Testing) above that applies to this change has a real, passing test — not a placeholder or a `TODO`.
4. Check the relevant rows of the failure-handling table (architecture doc §8) and the edge-case list (architecture doc §9) — does this change touch any of them? If so, is there a test for it?
5. Check the Definition of Done (architecture doc §18) — if this change closes out one of its checkboxes, say so explicitly and make sure the corresponding evidence (test, benchmark, demo) actually exists in the repo, not just in a commit message.
6. If this change deviated from the architecture doc in any way, confirm the doc was updated in the same change (§0 above) — never leave the two out of sync.

Do not claim a task, phase, or the overall project is complete without having walked through this list against what's actually in the repository.
