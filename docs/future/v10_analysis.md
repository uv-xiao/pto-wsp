# PTO‑RT v10: Analysis (draft)

> **Goal:** clarify *why* v10 exists and what architectural shifts are necessary (without rewriting PTO‑RT into another project).

## 1. v9 recap: what works and what doesn’t

v9 established the core direction:

- **Codegen-first execution**: CPU-sim runs as a compiled artifact.
- **CSP/CSPT semantics on CPU-sim**: CSP pipelines execute inside the artifact; semantic time is PTO‑ISA cycles + constant channel latency.
- **Scheduling core**: `dispatch(...)` + `task_window(stall-only)` are behavior-changing; most other knobs are API-only/diagnosed.
- **NPU**: emission preserves schedule metadata, but is **emit-only** in environments without CANN.

v10 is not about inventing CSP or schedule; it’s about:

1) making CSP and scheduling semantics **cross-backend**,  
2) making NPU backends **fully runnable** (in appropriate toolchain environments), and  
3) introducing a first-class **NPU architecture model** so “Ascend vs AIE vs future NPU” does not imply “fork the compiler”.

## 2. Backend maturity: what “mature” means for PTO‑RT

When we say “mature backend” in v10, we mean:

1) **Explicit resource-bounded runtime**: bounded task metadata, bounded queues, bounded intermediate storage, and explicit flow control.
2) **Clear concurrency model**: orchestrator vs scheduler vs workers, and what can run concurrently.
3) **Deadlock-aware execution**: if bounded resources are exhausted, the runtime must (a) stall safely or (b) provide actionable diagnostics.
4) **Deterministic semantics**: “semantic time” is the artifact’s cycle model, not host wall-clock, and should be replayable.
5) **Backend-independent runtime core**: scheduling and CSP machinery should not be reimplemented per backend.

### 2.1 Reference: pto-isa-lh runtime patterns to learn from

We should treat pto-isa-lh as a reference for runtime *mechanisms*, not as a frontend model to copy.

Key runtime patterns that map directly to PTO‑RT v10 needs:

- **Task window / ring-buffer flow control**: avoid “build all tasks then run”; enforce a bounded in-flight window with stall/backpressure.
- **TensorMap dependency inference with bounded pools**: keep dependency tracking bounded and fast with lazy invalidation.
- **Multi-threaded separation of concerns**: orchestrator thread building tasks while scheduler/workers execute, enabling memory reuse.
- **Deadlock detection + diagnostics**: when window sizing or consumer accounting deadlocks progress, fail with a detailed message and guidance.

These patterns are highly relevant because CSP introduces additional blocking edges that must coexist with task-window backpressure.

#### 2.1.1 “Five rings” as an explicit v10 design target

pto-isa-lh’s flow-control design is not “just a task window”; it is a coordinated set of bounded resources that all provide
backpressure (stall) to orchestration when exhausted. The important conceptual shape is:

1) **Task ring** (task metadata window)  
2) **TensorMap pool** (dependency inference state)  
3) **DepList pool** (fanin/fanout edges)  
4) **Heap ring** (intermediate buffers / packed outputs)  
5) **Ready queue(s)** (work queues per worker type)  

Alongside this, there are two essential requirements for maturity:

- **Flow-control stats** (stall counts, stall time, high-water marks) for tuning and regression detection.
- **Deadlock diagnostics** that are actionable when bounded resources + blocking semantics stop progress.

PTO‑RT v10 should treat these as first-class runtime concepts (even if sizing differs per backend).

Two concrete implementation details from pto-isa-lh’s `runtime2` that are worth copying as *mechanisms*:

- **Orchestrator-side scopes affect liveness**: task outputs are held live by “scope references” until `scope_end()` decrements
  fanout refcounts; if `task_window` is too small for the active scope, the orchestrator can deadlock itself (it cannot reach
  `scope_end()` while stalled on the task ring).
- **Stall reasons are explicit and measurable**: stall is not a vague “backpressure”; it has explicit reasons (task ring /
  tensormap pool / deplist pool / heap ring / ready queue), plus counters, time, and high-water marks for diagnosis and tuning.

Concrete reference entrypoints in `references/pto-isa-lh/`:
- `references/pto-isa-lh/docs/runtime_buffer_manager_methods.md` (flow control, ring buffers, deadlock diagnostics, runtime2 discussion)
- `references/pto-isa-lh/src/runtime/pto_runtime_common.h` (TensorMap, fanin/fanout, ready queues, stall statistics)
- `references/pto-isa-lh/src/runtime2/` (threaded scheduler/worker/orchestrator split; task window sizing)

## 3. CSP across backends: what must remain invariant

v10 CSP is a *semantic contract*:

- **CSP nodes are explicit** (send/consume), not inferred from tensor reads/writes.
- **Time semantics are unified**:
  - kernel time comes from backend cycle reports (PTO‑ISA cycles on CPU-sim/Ascend; AIE cycles on AIE)
  - channel latency is a constant 0 cycles by default for v10 (stall-only), configurable later
- **Progress and boundedness are explicit**:
  - a bounded task window is a global constraint
  - channels are token-based (sync-only payload model); v10 starts with a small default logical capacity (1 token) and can grow later

The key shift from v9 to v10 is not “CSP exists”, but “CSP exists everywhere”: CPU-sim, Ascend, AIE.

## 4. Programmable dispatch/issue across backends

In v9, we intentionally limited enforcement to a small subset. In v10:

- `dispatch` and `task_window` remain core semantics, but
- the scheduling surface must be **policy-extensible** (registry of policies) rather than a fixed set of hard-coded branches.

This is required for multi-NPU support because different NPUs have different:

- worker types / exec units (vector/matrix/DMA)
- queueing models
- launch/dispatch granularity (core array vs wavefront)
- costs (DMA vs compute)

## 5. Why we need an NPU architecture representation

To make Ascend and AIE both runnable under one compiler, we need a shared representation of:

- **Memory hierarchy**: global memory, caches, scratchpads, register files; address spaces and access rules.
- **Execution units**: vector vs matrix/cube, DMA/MTE engines, their concurrency constraints.
- **Synchronization model**: barriers, flags, ordering domains.
- **Dispatch model**: core grid, waves, queues, and how tasks map to them.
- **Cycle accounting hooks**: how kernels report time; how communication/queueing contributes to semantic time.

Without this, every backend becomes a special-case pipeline, and CSP/dispatch semantics will drift.

## 6. Reference: Dato and Allo for AIE/dataflow direction

Dato’s model is an important conceptual guide for multi-NPU support:

- explicit streams/channels
- layout as type refinement
- mapping from virtual tasks/streams to physical resources

Allo’s AIE backend is a practical reference for:

- what the AIE toolchain expects (project structure, build/run workflow)
- the role of external kernels for complex compute on AIE cores
- dataflow transformations (e.g., buffer→FIFO transformations and canonicalization passes)
- **data-driven deadlock risk**: for stream-driven execution, launching kernels in topological order and/or enforcing a DAG
  stream graph is a pragmatic constraint (until buffered channels / capacity modeling is introduced)

Concrete reference entrypoints in `references/allo/`:
- `references/allo/allo/ir/types.py` (`Stream` type)
- `references/allo/allo/autoscheduler/passes.py` (dataflow canonicalization + buffer→FIFO extraction)
- `references/allo/allo/backend/aie/__init__.py` (AIE toolchain integration, build+run wrapper)
- `references/allo/docs/source/backends/aie/` (environment, usage, profiling/trace notes)

For PTO‑RT v10, the takeaway is:

- treat AIE as a **dataflow accelerator** where streams are first-class
- keep CSP semantics stable across backends, while allowing per-arch mapping strategies

## 7. Summary: v10 thesis

v10 is the “multi-backend maturity” release:

- Adopt bounded-runtime patterns (task window + flow control + deadlock diagnostics).
- Make CSP/CSPT semantics cross-backend and runnable (CPU-sim, Ascend, AIE).
- Make dispatch/issue policy extensible and arch-aware.
- Introduce an NPU architecture model that unifies backend codegen/runtime behavior.
