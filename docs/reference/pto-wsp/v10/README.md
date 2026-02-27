# Future Plans

This folder contains forward-looking design/implementation plans beyond v9.

## Keeping v10 docs up to date (required)

The v10 documents in this folder are **living specs**. When real work is executed (code, tests, integration), update the
relevant v10 docs in the same change:

- `v10_plan.md` and `v10_tracker.md` must reflect the current execution plan and progress.
- Vision/spec docs (`v10_analysis.md`, `v10_features.md`, `v10_spec.md`) must reflect the current backend/target model
  (including pto-runtime as a codegen target) and must not drift from what the codebase actually supports.
- `v10_implementation.md` must include a dated “done vs todo” snapshot and should be refreshed as milestones land.

- `docs/future/v10_analysis.md` — v10 design analysis (WHY)
- `docs/future/v10_features.md` — v10 feature catalog (WHAT, at a glance)
- `docs/future/v10_spec.md` — v10 spec draft (semantics + API surface)
- `docs/future/v10_implementation.md` — v10 implementation plan (HOW)
- `docs/future/v10_plan.md` — v10 goals, workstreams, milestones
- `docs/future/v10_tracker.md` — checklist tracker for v10 execution
- `docs/future/v10_compiler_pipeline.md` — v10 Python-driven compiler/codegen pipeline (AST passes + intrinsics + optional MLIR islands)
- `docs/future/v10_pto_runtime_interface.md` — interface contract / hazards checkpoint for PTO‑WSP ↔ pto-runtime
- `docs/reference/pto_runtime/README.md` — reference index for pto-runtime notes (for v10)
- `docs/reference/pto_runtime/analysis.md` — notes on the decoupled `pto-runtime` runtime architecture (reference)
- `docs/reference/pto_runtime/integration.md` — how PTO‑WSP should integrate with `pto-runtime` (reference)
- `docs/reference/pto_runtime/gaps.md` — explicit pto-runtime gaps PTO‑WSP v10 cares about (reference)
- `docs/reference/pto_runtime/task_buffer.md` — task-buffer / bounded runtime direction (reference)
- `docs/reference/18_pypto.md` — reference notes on PyPTO’s python-driven pipeline and artifact packaging patterns
