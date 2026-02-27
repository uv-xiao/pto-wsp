# PTO‑WSP v10: Implementation Plan (draft)

This document outlines *how* we can evolve the current codebase to meet v10 goals.

## Progress snapshot (2026-02-02)

**What exists now (done):**
- `pto-runtime` submodule at `3rdparty/pto-runtime`
- pto-runtime Python import bridge: `python/pto_wsp/pto_runtime_bridge.py`
- Phase 1 runnable integration (host_build_graph):
  - codegen targets emit a visible source tree:
    - `target="a2a3sim_codegen"` / `target="a2a3_codegen"`
    - emitter entrypoint: `include/pto/wsp/codegen/pto_runtime_host_build_graph.hpp`
  - orchestration codegen emits a runnable task graph (`runtime->add_task` / `runtime->add_successor`) for the initial supported subset
  - platform-correct kernel sources are emitted:
    - `kernels/aiv_sim/*.cpp` (a2a3sim)
    - `kernels/aiv/*.cpp` (a2a3)
  - runnable PTO‑WSP targets exist:
    - `target="pto_runtime_a2a3sim"` (end-to-end test passes)
    - `target="pto_runtime_a2a3"` (wired; toolchain-gated)
  - PTO‑WSP wraps pto-runtime tooling to build+run after emission:
    - `python/pto_wsp/pto_runtime_runner.py`
- sandbox-safe codegen cache default: `build/.pto_wsp_codegen_cache` (override via `PTO_WSP_CODEGEN_CACHE_DIR`)
- CMake option `PTO_RUNTIME_PATH` (default: `3rdparty/pto-runtime`)

**What is next (todo):**
- harden Phase 1 correctness and honesty:
  - make the supported-subset boundary explicit in code (fail fast with clear diagnostics)
  - tighten dependency generation beyond “conservative sequential chaining” (tensor read/write analysis)
  - validate `target="pto_runtime_a2a3"` in a real Ascend/CANN environment (toolchain + device)
- define the v10 manifest/ABI as needed for Phase 2 (task-buffer) and long-term portability
- map `dispatch(policy)` to multi-AICPU scheduling semantics (currently a documented gap)
- implement CSP channel semantics + diagnostics on the pto-runtime path (Phase 2 target)

**Diagram (done → next):**

```
PTO‑WSP Python
  |  (build IR, choose target)
  v
PTO‑WSP C++ codegen
  |  DONE: emits host_build_graph sources (a2a3sim_codegen)
  |  TODO: emits host_build_graph sources + metadata (visible artifact)
  v
pto-runtime tooling (Python)
  |  DONE: import bridge exists
  |  TODO: PTO‑WSP wraps builder/compiler to build+run for a2a3sim / a2a3
  v
pto-runtime runtime (a2a3sim / a2a3)
```

## 0. Interface checkpoint (gating doc)

For v10, “implementing backends” is inseparable from defining a clean, versioned boundary with `pto-runtime`. Treat the
interface contract as a gating specification:

- `docs/future/v10_pto_runtime_interface.md`

## 1. Organizing principle

v10 must keep the architecture explicitly decoupled:

1) **DSL layer (Python)**: workload–schedule as the portable surface; kernels as backend-specific intrinsic programs.  
2) **Compilation layer**: source-AST pass pipeline + optional internal AST↔MLIR bridge for pass-selected regions.  
3) **Emitter/Backend layer**: backend registry + artifact packaging + toolchain build/run integration.  

Within the backend/runtime story, v10 must avoid “one backend = one runtime”. Instead:

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
Backend codegen (pto_runtime_* / aie; cpu_sim legacy)
  |
  v
Artifact package (per target)
  - kernels (PTO‑ISA / vendor)
  - orchestration bytecode/IR (CSP bodies, control flow)
  - schedule metadata + policy IDs
  - slots/symbols ABI (dynamic axes, predicates)
  - ArchModel instance + capability matrix
  - package manifest + ABI version (consumable by `pto-runtime`)
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
    - `pto_runtime_*` targets emit artifacts consumable by `pto-runtime` (`a2a3sim` for local validation; `a2a3` for devices)
    - `ascend_npu` uses runtime core + CANN integration (via `pto-runtime` on the runnable path)
    - `aie` uses runtime core + AIE toolchain integration
    - legacy note: native `cpu_sim` is a v9 backend and is planned to be removed after `pto_runtime_a2a3sim` parity validation

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

Conceptual flow-control shape (aligned to pto-runtime task-buffer direction):

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
- Target the decoupled `pto-runtime` as the execution substrate:
  - v10 should avoid building a parallel bespoke runtime core inside PTO‑WSP.
  - use `a2a3sim` for local semantics tests (no toolchain) and `a2a3` for real device.
  - align bounded-resource (“multi-ring”) behavior to the pto-runtime task-buffer direction.

Reference pointers:
- `references/pto-runtime/README.md`
- `docs/reference/pto_runtime/task_buffer.md`
- `references/pto-runtime/src/runtime/host_build_graph/runtime/runtime.h`

### M2.1) pto-runtime codegen pipeline (Phase 1)

Phase 1 integration should be implemented as a **C++ codegen target** in PTO‑WSP that emits a pto-runtime-compatible
`host_build_graph` source tree (mirroring pto-runtime examples):

- `kernels/orchestration/*.cpp` — orchestration entry (build task graph via pto-runtime `Runtime`)
- `kernels/*/*.cpp` — executor-specific kernel sources (AIV/AIC, etc.)
- `kernels/kernel_config.py` — declares orchestration symbol + kernel list (`func_id`, `core_type`)

At the Python level, PTO‑WSP should **import pto-runtime tooling** (from the submodule, via a small bridge helper) to build
and run the generated tree:

- `RuntimeBuilder(platform="a2a3sim")` for local semantics testing
- `RuntimeBuilder(platform="a2a3")` for real-device runs (toolchain-gated)

Missing pto-runtime capabilities (multi-AICPU dispatch mapping, CSP channels, task-buffer backpressure) must be treated as
explicit gaps; do not claim enforcement until the runtime exposes the necessary APIs.

### M3) Cross-backend CSP runtime

Deliver:
- artifact-level CSP runtime implemented once in runtime core
- channel operations integrated into dependency tracking and scheduling
- CSPT time computation integrated into stats

### M4) Ascend backend: from emit-only to runnable

Deliver:
- keep emission for inspection, but add a runnable path in proper CANN environments by targeting `pto-runtime`:
  - build pto-runtime host runtime + device binaries (`a2a3`)
  - emit/register kernels and the schedule/orchestration binary
  - load and execute
  - report cycles and (optionally) profiling

Phasing recommendation:
- Phase 1: generate a **host orchestration `.so`** that builds the task graph via pto-runtime APIs (unblocks runnable backend).
- Phase 2: move expansion/orchestration onto AICPU using the task-buffer direction (restores PTO‑WSP’s on-device expansion thesis).

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
    - if `pto-runtime` is integrated as a submodule, ensure remote scripts sync the submodule too

- AIE validation suite:
  - correctness vs golden
  - CSP correctness
  - timing/cycle sanity

## 5. Risks

- CSP + bounded resources can deadlock; diagnostics must be excellent.
- Multi-NPU support requires disciplined separation: if backend glue leaks into runtime core, the model will fork.
- Toolchain variability (CANN / AIE) demands robust build wrappers and clear capability reporting.
