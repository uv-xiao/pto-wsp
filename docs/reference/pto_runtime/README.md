# Reference: pto-runtime (for PTO-WSP v10)

This folder contains reference notes and design context about `pto-runtime`, which PTO-WSP v10 targets as its primary runtime
substrate for Ascend simulation and device execution.

Local repository clone (gitignored): `references/pto-runtime/` (and the tracked submodule at `3rdparty/pto-runtime/`).

## Contents

- `docs/reference/pto_runtime/analysis.md` — high-level analysis notes of pto-runtime architecture and current state
- `docs/reference/pto_runtime/integration.md` — PTO-WSP ↔ pto-runtime integration notes (v10 direction)
- `docs/reference/pto_runtime/gaps.md` — explicit gaps / missing features PTO-WSP needs (semantics-honest checklist)
- `docs/reference/pto_runtime/task_buffer.md` — preview/reference: task-buffer direction for bounded execution and true backpressure

## How this relates to v10 docs

v10 living specs remain in `docs/future/` (plan/spec/tracker). Those docs should link here for background and reference
material, while keeping v10 requirements and checkpoints in `docs/future/`.
