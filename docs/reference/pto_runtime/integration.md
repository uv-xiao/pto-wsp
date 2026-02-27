# Reference: PTO‑WSP ↔ pto-runtime Integration Notes (v10 direction)

This document is a **reference note** for how PTO‑WSP v10 should integrate with `pto-runtime` (source clone in
`references/pto-runtime/` and submodule in `3rdparty/pto-runtime/`).

Interface checkpoint (v10 living spec):
- `docs/future/v10_pto_runtime_interface.md`

## 0) Goals and non-negotiables

- PTO‑WSP remains responsible for: **DSL + compilation + artifact semantics** (workload–schedule, dispatch/task_window,
  CSP/CSPT, predicates/slots).
- pto-runtime provides: **execution substrate** (host + AICPU + AICore) and build/run toolchain.
- v10 is **always codegen**: regardless of backend, compile kernels + orchestration/schedule payload into an executable
  artifact (C++ plus backend toolchain outputs).
- Python may orchestrate compilation and launch artifacts, but must not “drive execution semantics”.

## 1) Dependency shape

### Option A: Git submodule (recommended, current)

Pinned runtime for reproducibility:
- `3rdparty/pto-runtime/`

### Option B: External checkout

Allow pointing PTO‑WSP to a runtime checkout via config/env var, but keep submodule as the default for CI.

## 2) Runtime flavors and the integration ladder

pto-runtime has multiple runtimes under `references/pto-runtime/src/runtime/`. For PTO‑WSP v10, treat these as a pragmatic
ladder, not competing designs:

1) `host_build_graph`: host orchestration builds full task graph (fast path to “runnable”)
2) `aicpu_build_graph`: more graph build moved onto AICPU (bridge)
3) `tensormap_and_ringbuffer` (PTO2): bounded task-buffer runtime (target for true `task_window` backpressure)

## 3) Packaging boundary (what PTO‑WSP should emit)

For `pto_runtime_*` backends, PTO‑WSP should emit an artifact directory that matches pto-runtime’s example conventions:

- `kernels/orchestration/*.cpp`
- `kernels/aiv/*.cpp`, `kernels/aic/*.cpp` (or platform-specific variants)
- `kernels/kernel_config.py` (manifest/config: runtime name, orchestration entry, kernels list, runtime config knobs)

This “visible source tree artifact” pattern is a key takeaway from both pto-runtime and PyPTO:
- deterministic structure
- reviewable diff
- backend toolchain can compile it independently of PTO‑WSP’s internal IR

## 4) Phase mapping (PTO‑WSP v10)

### Phase 1: runnable correctness backend via `host_build_graph` (`a2a3sim` + `a2a3`)

Target:
- `pto_runtime_a2a3sim`: CI/default correctness backend for scheduling/CSP semantics
- `pto_runtime_a2a3`: real-device execution when toolchain exists (gated)

Mechanism:
- PTO‑WSP emits a `host_build_graph`-shaped source tree and **wraps pto-runtime’s Python tooling** to build+run it.

Source of truth:
- pto-runtime runner + compilers:
  - `references/pto-runtime/python/runtime_builder.py`
  - `references/pto-runtime/python/runtime_compiler.py`
  - `references/pto-runtime/python/kernel_compiler.py`

### Phase 2: semantics-complete bounded runtime via PTO2 (`tensormap_and_ringbuffer`)

PTO‑WSP’s long-term semantics (true backpressure, dynamic expansion, channel blocking) naturally want a bounded runtime.
pto-runtime already has the core pieces in its PTO2 runtime (`tensormap_and_ringbuffer`):

- bounded shared memory (task ring + dep list pool + flow-control pointers)
- private TensorMap for dependency discovery
- scope-based lifetime rules (fanout refs)
- explicit backpressure points

For v10 design, this is the “correct landing zone” for:
- `task_window(mode=STALL)` as *real* backpressure to orchestration/expansion,
- deadlock-aware diagnostics (bounded resources + CSP blocking),
- “orchestrator on device” execution for low-latency dynamic scheduling.

Reference:
- `docs/reference/pto_runtime/task_buffer.md`

## 5) Semantic mapping checklist (what still needs explicit contracts)

### 5.1 `dispatch(policy)` (scheduler assignment, not `core_type`)

PTO‑WSP intent:
- `dispatch(policy)` selects **which AICPU scheduler domain** issues/schedules the task.

What exists in pto-runtime today:
- pto-runtime can run multiple AICPU threads (and in PTO2 runtime, “3 schedulers + 1 orchestrator” is a first-class setup),
  but there is no stable, user-controlled ABI surface for “task belongs to scheduler shard k” that PTO‑WSP can target.

Track explicitly as a v10 gap:
- `docs/reference/pto_runtime/gaps.md`

### 5.2 `task_window(mode=STALL)` (bounded in-flight tasks)

PTO‑WSP requirement:
- `task_window` is a semantics knob, not metadata.

pto-runtime status:
- `host_build_graph` can schedule a finite graph but cannot backpressure orchestration that already ran to completion.
- PTO2 runtime (`tensormap_and_ringbuffer`) has an explicit task-window ring and flow-control pointers designed for this.

### 5.3 CSP/CSPT

PTO‑WSP requirement:
- Channels are explicit blocking edges; time is derived from kernel cycle reports + channel latency model.

pto-runtime status:
- does not yet model channels as first-class readiness constraints.
- PTO2 bounded runtime is the plausible place to integrate channel waits (in scheduler readiness checks) and diagnostics.

### 5.4 Predicates/slots and tensor→scalar materialization

PTO‑WSP requirement:
- a versioned slot/symbol table ABI carried by the artifact, updatable between runs without recompiling kernels.

pto-runtime status:
- no general slot ABI yet; would need to be added to the package contract and runtime state.

## 6) Practical guidance for PTO‑WSP implementation

- Treat the emitted artifact directory as the canonical interface boundary for `pto_runtime_*` backends.
- Reuse pto-runtime Python toolchain for compilation and execution (don’t duplicate platform toolchain logic in PTO‑WSP).
- Keep “Phase 1 runnable” and “Phase 2 semantics-complete” explicitly distinct in docs and capability matrices.

