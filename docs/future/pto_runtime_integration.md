# PTO‑WSP ↔ pto-runtime Integration (v10 design notes)

This doc proposes how PTO‑WSP v10 should integrate with the decoupled `pto-runtime` project.

Interface checkpoint (gating spec): `docs/future/v10_pto_runtime_interface.md`.

## 0) Goals

- Reuse `pto-runtime` as the **runtime execution substrate** (Ascend real device + host simulation).
- Keep PTO‑WSP responsible for:
  - workload/schedule programming model,
  - IR + codegen,
  - artifact semantics (CSP/CSPT, dispatch, task_window, predicates).
- Make local validation possible without Ascend toolchains via `pto-runtime` `a2a3sim`.

## 1) Integration shape options

### Option A — Git submodule (recommended)

Add `pto-runtime` as a submodule under `3rdparty/pto-runtime` (tracked).

Pros:
- reproducible builds (pinned commit)
- CI/dev tools can build `a2a3sim` and run scheduler-level tests
- clear ownership boundary (pto-runtime remains its own repo)

Cons:
- submodule UX overhead (updates, sync)

### Option B — External dependency (workspace-local)

Keep `pto-runtime` out of the repo, require an env var (e.g., `PTO_RUNTIME_PATH`) or config file.

Pros:
- no submodule management

Cons:
- non-reproducible for most users
- harder to script and test reliably

### Option C — Vendor/import code (not recommended)

Copy runtime sources into PTO‑WSP.

Cons:
- duplicates the runtime project and creates merge debt

## 2) Artifact boundary: what PTO‑WSP should produce for pto-runtime

PTO‑WSP already has a codegen-first artifact concept. In v10, for Ascend+sim backends, the artifact should be packaged in a
format that `pto-runtime` can run.

At minimum, PTO‑WSP should be able to emit:

1) **AICore kernel binaries**
   - generated from `pto.*` lowering or `ptoisa.*` (PTO‑ISA tile kernels)
   - plus file-based custom kernels (escape hatch)

2) **AICPU scheduler/orchestrator binary**
   - runs the schedule semantics:
     - `dispatch(policy)` executor selection
     - `task_window` stall-only backpressure
     - CSP channel waits/signals
     - predicate/slot evaluation boundaries

3) **Host runtime library glue**
   - to load/register kernels and launch the AICPU binary
   - to manage device memory + H2D/D2H

`pto-runtime` already provides “host runtime + AICPU + AICore” scaffolding; v10 integration is primarily about:

- defining the ABI for “task buffer / plan” and
- generating the right binaries for `a2a3` and `a2a3sim`.

## 2.1) Integration method (Phase 1): C++ codegen emits pto-runtime sources

For Phase 1, PTO‑WSP integrates with pto-runtime by **generating a pto-runtime-compatible source tree** (host_build_graph),
then using pto-runtime tooling to build and run it.

Concretely, PTO‑WSP C++ codegen should emit a tree shaped like pto-runtime examples:

1) `kernels/orchestration/<name>.cpp`
   - A host-build-graph orchestration function that uses pto-runtime `Runtime` APIs to:
     - allocate buffers,
     - build tasks (`add_task`) and deps (`add_successor`),
     - record output tensors for copy-back.

2) `kernels/<core_type>/*.cpp`
   - Kernel sources (AIV/AIC or other executor types supported by the selected platform).

3) `kernels/kernel_config.py`
   - Declares the orchestration symbol name and kernel list (`func_id`, `core_type`), matching pto-runtime config
     conventions.

At the Python level, PTO‑WSP should **import pto-runtime tooling** (from the submodule, via a small bridge helper) to:

- build host runtime + AICPU + AICore binaries and run for `platform="a2a3sim"` locally, and
- build the real-device binaries and run for `platform="a2a3"` in proper CANN environments.

PTO‑WSP should wrap this as part of its own “compile + run” flow for `pto_runtime_*` targets, while still keeping the emitted
source tree as the primary visible artifact (reviewable output).

### Phase 1 “complete integration” definition (not just emit stubs)

For v10, “Phase 1 integration is done” does **not** mean “we emitted placeholder sources”. It means:

- PTO‑WSP emits a **visible host_build_graph source tree** artifact, and
- PTO‑WSP wraps pto-runtime tooling to **build+run** that artifact end-to-end:
  - `target="pto_runtime_a2a3sim"` must be runnable in CI/local dev (no Ascend toolchain required)
  - `target="pto_runtime_a2a3"` must be wired (toolchain-gated by `ASCEND_HOME_PATH`)
- the generated sources are **codegen-complete** for an initial supported subset:
  - orchestration C++ builds a runnable task graph (`add_task`/`add_successor`, plus tensor H2D/D2H hooks)
  - kernels are real for both:
    - `a2a3sim` (plain C++ simulation kernels), and
    - `a2a3` (incore kernels; toolchain-gated compilation)

Implementation plan and tracker are kept in: `docs/plans/2026-02-02-pto-runtime-integration-v10.md`.

### Platform split in emitted sources (a2a3sim vs a2a3)

To keep Phase 1 runnable without toolchains while still supporting real device runs, PTO‑WSP should emit:

- `kernels/aiv_sim/*.cpp` for `a2a3sim` (plain C++ loop kernels; compiled by `g++` via pto-runtime tooling)
- `kernels/aiv/*.cpp` for `a2a3` (incore kernels; compiled by `ccec` via pto-runtime tooling)

`kernels/kernel_config.py` should select the correct kernel source list based on a platform hint (e.g.
`PTO_RUNTIME_PLATFORM=a2a3sim|a2a3`) that the PTO‑WSP runner sets before importing the config.

**Non-negotiable:** schedule/CSP semantics must not be implemented by Python “driving” execution. Python may build IR,
invoke compilation, and launch runtime; semantics must live in artifacts/runtime logic.

## 3) Mapping PTO‑WSP semantics into pto-runtime mechanisms

### 3.1 dispatch(policy)

- **PTO‑WSP intent (v10):** `dispatch(policy)` decides **which AICPU scheduler instance** a task belongs to (i.e., host-side
  task division across AICPU schedulers / queues), not merely how an already-assigned task is executed.
- This is **not** the same thing as per-task `core_type` (AIC/AIV) selection:
  - `core_type` chooses *what kind of executor* runs the task (AICore vs AIV, etc.).
  - `dispatch` chooses *which AICPU scheduler* issues/schedules the task.
- **Current pto-runtime gap:** pto-runtime’s `host_build_graph` path can represent task deps and (at most) per-task executor
  metadata, but does not yet expose a first-class notion of “task belongs to AICPU #k” (multi-AICPU scheduling/partitioning).
  Track this explicitly as a v10 integration gap (see `docs/future/pto_runtime_gaps.md`).

### 3.2 task_window(stall-only)

- PTO‑WSP task_window must be enforced by the runtime, not treated as metadata.
- The upcoming task-buffer design in `docs/future/pto-runtime-task-buffer.md` is the correct direction:
  - bounded in-flight task table (ring)
  - backpressure when full

### 3.3 CSP/CSPT

- Channels become explicit scheduling edges.
- Channel operations are expressed as:
  - additional deps / wait conditions in the AICPU scheduler
  - explicit ready-queue eligibility checks (empty/full)
- **Implementation note:** CSP requires **auto-generated orchestrator/scheduler logic** from PTO‑WSP codegen. The integration
  must not rely on Python to “drive” the CSP execution at runtime.
- CSPT time:
  - kernel time comes from PTO‑ISA cycle reports (AICore execution)
  - channel latency is constant (default 0) and accumulates as stalls

### 3.4 Runtime predicates + tensor→scalar

- Slots/symbol tables live in runtime state shared by orchestration/scheduler.
- Tensor→scalar materialization must be implemented as:
  - a kernel producing a small tensor output, and
  - a runtime “load element → slot” operation, avoiding host recompilation.

## 4) “cpu_sim” becomes “a2a3sim” (recommendation)

For v10 semantics testing, `a2a3sim` should become the canonical local backend because it exercises:

- AICPU scheduler logic,
- AICore kernel invocation,
- handshake/queueing behavior,
- (future) task-buffer backpressure,

without requiring Ascend toolchains. This is more semantically faithful than a single-process dlopen runner.

## 4.1 Phased integration plan (pragmatic)

There is a real tension between:

- PTO‑WSP v9 direction: on-device task expansion (AICPU “expander”) + compact plan + runtime symbols/slots, vs
- pto-runtime today: host-built graph (`host_build_graph`) with an orchestration `.so` executed by the host runtime.

So v10 should plan a phased approach:

### Phase 1 (unblock runnable backend fast)

- Treat PTO‑WSP codegen as producing a **host orchestration `.so`** that builds the runtime task graph using pto-runtime’s
  `Runtime`/C API.
- Use pto-runtime `a2a3sim` to validate schedule semantics locally.
- Use pto-runtime `a2a3` to run on real Ascend device.

This gets correctness + runnable paths without forcing “on-device expansion” to land immediately.

### Phase 2 (restore PTO‑WSP’s on-device expansion direction)

- Move orchestration/expansion onto AICPU using the pto-runtime task-buffer direction:
  - compact plan + slots/symbols ABI,
  - bounded task-buffer backpressure (`task_window`),
  - CSP channel waits integrated into the AICPU scheduler.

This is where PTO‑WSP’s differentiators (dynamic schedule, CSP-first execution) should fully live.

## 5) Practical workflow and configs

- In PTO‑WSP repo, keep local device server configs untracked (already supported):
  - `.ASCEND_SERVER.toml` (ignored) + `.ASCEND_SERVER.toml.example`
- When using a submodule, remote test scripts should sync it too.

## 6) v10 execution plan impacts

Adopting pto-runtime changes the v10 plan:

- “backend maturity” is no longer “port from older runtime patterns”, but:
  - align PTO‑WSP artifact ABI with pto-runtime task-buffer ABI,
  - add schedule/CSP/predicate semantics to the pto-runtime scheduler path,
  - validate in `a2a3sim` and on real Ascend device.

See updates in:
- `docs/future/v10_plan.md`
- `docs/future/v10_implementation.md`
- `docs/future/v10_tracker.md`
