# PTO‑RT v10: Implementation Plan (draft)

This document outlines *how* we can evolve the current codebase to meet v10 goals.

## 1. Organizing principle

v10 must avoid “one backend = one runtime”. Instead:

- **one runtime core** implements task windowing, dependency tracking, CSP, and policy hooks
- backends provide:
  - kernel compilation/lowering
  - executor integration
  - architecture model instance and capability matrix

## 1.1 Runtime layering (ASCII sketch)

The goal is to make “what runs where” unambiguous:

```
Python API
  |
  v
Typed IR (C++ IR layer)
  |
  v
Backend codegen (cpu_sim / ascend / aie)
  |
  v
Artifact package (per target)
  - kernels (PTO‑ISA / vendor)
  - orchestration bytecode/IR (CSP bodies, control flow)
  - schedule metadata + policy IDs
  - slots/symbols ABI (dynamic axes, predicates)
  - ArchModel instance + capability matrix
  |
  v
Artifact runtime core (shared semantics)
  - orchestrator (build/submit, scopes, slots)
  - scheduler (deps, ready queues, retirement)
  - workers/executors (kernel launch + cycle reports)
```

And the shared “ArchModel” should be used in both compiler and runtime:

```
            +------------------+
            |   ArchModel      |
            | (mem/exec/sync)  |
            +------------------+
               ^            ^
               |            |
        codegen decisions   runtime policy/cycles
```

## 2. Proposed code structure (incremental)

Suggested new modules (names are indicative):

- `include/pto/wsp/arch/` and `src/pto/wsp/arch/`
  - architecture model types (memory, exec units, sync, dispatch model)
  - concrete instances:
    - `ascend_arch.*`
    - `aie_arch.*`

- `include/pto/wsp/runtime/` and `src/pto/wsp/runtime/`
  - runtime core:
    - task window ring buffer
    - dependency tracking:
      - tensor-map inference
      - explicit CSP edges
    - scheduler/policy hooks
    - deadlock diagnostics
  - expected internal split (conceptual):
    - **orchestrator**: executes orchestration/CSP bodies, allocates buffers, builds deps, submits tasks (may stall)
    - **scheduler**: maintains ready queues and dependency counters, advances retirement pointers
    - **workers/executors**: execute kernels and report cycles/completion events

- `include/pto/wsp/backend/` and `src/pto/wsp/backend/`
  - backends are “thin”:
    - `cpu_sim` uses runtime core + PTO‑ISA CPU sim kernels
    - `ascend_npu` uses runtime core + CANN integration
    - `aie` uses runtime core + AIE toolchain integration

## 3. Key implementation milestones

### M1) Architecture model layer

Deliver:
- `ArchModel` representation with minimal required fields
- `BackendCapabilities` matrix and a consistent reporting format

Why first:
- the arch model drives both CSP and dispatch/issue semantics across backends

### M2) Runtime core: bounded execution + diagnostics

Deliver:
- **multi-ring bounded runtime resources** (not just task window):
  - task ring (task window)
  - tensormap pool
  - deplist pool
  - heap/buffer arena
  - ready queues
- consistent “stall-only” semantics across all bounded resources
- flow-control stats (stall counts/time, high-water marks, current stall reason)
- deadlock detection hooks + actionable diagnostics

Conceptual flow-control shape (inspired by pto-isa-lh runtime2):

```
orchestrator
  |  (alloc/submit; may stall)
  +--> task ring  (task_window)
  +--> heap arena (packed outputs / intermediates)
  +--> tensormap pool + deplist pool (deps)
  |
  v
scheduler
  +--> ready queue(s) ---> workers/executors ---> completion events
  |
  +--> retirement pointers (frees task slots, dep entries, heap tail)
```

Note:
- Use pto-isa-lh’s runtime “flow control” patterns as a reference:
  - multiple bounded rings (task table, heap, dep list, tensormap pool)
  - backpressure between orchestrator and scheduler
  - actionable diagnostics when bounds are violated
  - scope-driven liveness (scope_begin/scope_end) and its interaction with task-window sizing

Reference pointers:
- `references/pto-isa-lh/docs/runtime_buffer_manager_methods.md`
- `references/pto-isa-lh/src/runtime/pto_runtime_common.h`
- `references/pto-isa-lh/src/runtime2/`

### M3) Cross-backend CSP runtime

Deliver:
- artifact-level CSP runtime implemented once in runtime core
- channel operations integrated into dependency tracking and scheduling
- CSPT time computation integrated into stats

### M4) Ascend backend: from emit-only to runnable

Deliver:
- keep emission for inspection, but add a runnable path in proper CANN environments:
  - build host runtime + device binaries
  - load and execute artifacts
  - report cycles and (optionally) profiling

### M5) AIE backend: runnable target with toolchain integration

Deliver:
- project emitter and build/run wrapper compatible with an AIE toolchain environment
- external-kernel integration for complex compute
- stream/dataflow safety:
  - represent channels as FIFO/stream edges in the emitted artifact
  - validate the stream graph is a DAG (or explicitly reject cyclic graphs until capacity modeling exists)
  - schedule kernel launches in topological order where required by the toolchain/runtime

Allo’s AIE backend is a reference for:
- project structuring and environment detection
- external kernel compilation and integration

Reference pointers:
- `references/allo/allo/backend/aie/__init__.py`
- `references/allo/allo/backend/aie/utils.py`
- `references/allo/docs/source/backends/aie/`

## 4. Validation plan

### 4.1 Local (always-on): CPU-sim

- all examples must remain self-checking (golden vs pto-wsp output)
- CSP examples must be included

### 4.2 Toolchain environments (optional locally, required for v10 “done”)

- Ascend/CANN validation suite:
  - correctness vs golden for supported examples
  - CSP correctness (no missed waits, no silent deadlocks)
  - timing/cycle sanity (tolerance-based if needed)
  - practical execution:
    - allow running tests via SSH on a device server (repo sync + build/run)
    - keep server details out of git: `.ASCEND_SERVER.toml` (ignored) with an `.example` template

- AIE validation suite:
  - correctness vs golden
  - CSP correctness
  - timing/cycle sanity

## 5. Risks

- CSP + bounded resources can deadlock; diagnostics must be excellent.
- Multi-NPU support requires disciplined separation: if backend glue leaks into runtime core, the model will fork.
- Toolchain variability (CANN / AIE) demands robust build wrappers and clear capability reporting.
