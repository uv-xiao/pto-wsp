# PTO‑RT v10: Specification (draft)

> This document is a **draft spec** for v10. It is intended to be aligned with implementation as v10 is built.

## 1. Non-negotiable semantic rules

### 1.1 Codegen-first execution

- If an API is described as “implemented”, its semantics are implemented in the **generated artifact**, not in Python fallback code.
- Python may build IR and launch artifacts, but must not become the execution engine for v10.

### 1.2 Canonical time

- `Program.stats.total_cycles` is the only canonical time metric.
- Backends provide kernel cycle reports:
  - CPU-sim/Ascend: PTO‑ISA cycle reports
  - AIE: AIE cycle reports (or equivalent hardware/emulator timing counter)

### 1.3 Schedule enforcement and diagnostics

- Each backend must declare a capability matrix:
  - **enforced**: changes artifact behavior
  - **recorded**: serialized/preserved in artifact metadata but not behavior-changing
  - **diagnosed**: ignored but explicitly reported once
  - **unsupported**: compile-time error (reserved for safety-critical cases)

## 2. CSP/CSPT semantics (cross-backend)

### 2.1 Core entities

- **Channel**: named edge between processes.
- **Process**: sequential actor with a declarative body.
- **send(channel, value)**: publish a value/event into a channel.
- **consume(channel, fn)**: receive values/events and run `fn` on each.

### 2.2 Blocking semantics

v10 defines CSP blocking at the artifact runtime level:

- `send` may stall if the channel is full.
- `consume` stalls if the channel is empty.
- For v10, the default channel model is **sync-only**:
  - values are “tokens” (e.g., task IDs) not arbitrary payload buffers
  - **logical capacity is 1 token** by default (producer can be at most 1 token ahead)
  - default channel latency is **0 cycles** (stall-only)

Backends may implement channels using deeper physical FIFOs/queues, but must enforce the **logical** capacity declared by the
artifact semantics (e.g., by counting outstanding tokens).

### 2.3 Termination

v10 must define how pipelines end:

- produced channels can be `close()`d by the producing process after it completes
- consumers exit when a channel is closed and empty

### 2.4 CSPT time semantics

Within a pipeline:

- each process has a local logical time `t_p`
- executing a kernel increments `t_p` by the kernel’s reported cycles
- channel operations may add latency (v10 default 0)
- pipeline makespan is `max_p(t_p)`

## 3. Dispatch/issue scheduling semantics (cross-backend)

### 3.1 dispatch(policy)

`dispatch` maps tasks (kernel invocations) to executors.

v10 requires at minimum:

- `round_robin(num_workers)`
- one additional policy that is meaningfully different (e.g., affinity or hash)

Policies must be implementable as modules, not hard-coded if/else trees.

### 3.2 task_window(TaskWindow(..., unit="tasks", mode=STALL))

`task_window` bounds the number of in-flight task metadata entries.

- `mode=STALL` is the semantic baseline: when the window is full, issue stalls until a slot is freed.
- other modes (abort/benchmark) may be supported per backend, but must be explicit in capability matrix.

### 3.3 Deadlock diagnostics

If CSP + bounded task window + channel waits results in deadlock (no progress), the runtime must provide:

- a detection mechanism (e.g., no progress for N steps / spins / time)
- a structured diagnostic report:
  - window utilization and high-water mark
  - which channel(s) are blocking (empty/full)
  - which process is waiting and why
  - suggested mitigation knobs (e.g., increase task_window or channel capacity)

## 3.4 Bounded runtime resources (“multi-ring” flow control)

v10 requires the artifact runtime to be **resource-bounded by construction**. “Task window” is necessary but not sufficient:
orchestration can also exhaust dependency-tracking pools, intermediate buffer arenas, and ready queues.

The runtime must treat the following bounded resources as first-class (names are conceptual; per-backend sizing differs):

1) **Task ring**: in-flight task metadata (the `task_window` bound is a direct constraint here)  
2) **TensorMap pool**: dependency inference entries (producer mapping / lazy invalidation)  
3) **DepList pool**: fanin/fanout adjacency nodes  
4) **Heap ring / buffer arena**: runtime-managed intermediate allocations / packed outputs  
5) **Ready queue(s)**: per-worker/per-exec-unit queues for runnable tasks  

### 3.4.1 Stall-only semantics (baseline)

When any bounded resource is exhausted, orchestration must **STALL** (backpressure) until space is freed. v10 does not allow
“silently grow” or “fallback to host Python execution” behavior as a recovery path.

### 3.4.2 Flow-control statistics (required)

The artifact runtime must expose flow-control stats sufficient for:

- regression detection (e.g., a change introduces unexpected stalls),
- tuning task window/pool sizing, and
- diagnosing deadlocks.

At minimum:

- stall counts per bounded resource,
- stall time per bounded resource (backend-defined clock; cycles preferred where available),
- high-water marks per bounded resource,
- current stall reason (for live debugging).

The stall reasons should be stable and backend-independent in meaning (e.g., “TASK_RING full”, “TENSORMAP_POOL full”, etc.).

Recommended minimal stall-reason vocabulary for v10 diagnostics:

- `TASK_RING_FULL`
- `TENSORMAP_POOL_FULL`
- `DEPLIST_POOL_FULL`
- `HEAP_ARENA_FULL`
- `READY_QUEUE_FULL`
- `CHANNEL_EMPTY_WAIT`
- `CHANNEL_FULL_WAIT`

### 3.4.3 Scope / liveness diagnostics (required)

v10 assumes orchestration constructs (including CSP process bodies) may create nested “scopes” that keep task outputs live
until the scope ends. This interacts with bounded resources:

- if a scope produces too many in-flight tasks, `task_window` stalling can prevent reaching `scope_end`, which prevents freeing
  outputs, which prevents task retirement (a deadlock).

When deadlock diagnostics trigger, the report must include at least:

- current scope depth (or equivalent liveness state),
- whether “scope-held outputs” are preventing consumption/retirement,
- recommended mitigations (e.g., increase `task_window`, reduce scope size, restructure orchestration).

## 4. Runtime predicates and tensor→scalar conversion

### 4.1 ScalarExpr

- ScalarExpr must be evaluable inside artifacts.
- ScalarExpr values can come from:
  - constants
  - dynamic symbols
  - runtime slots

### 4.2 Tensor→slot materialization

To support data-dependent control without recompilation:

- a kernel can write to a tensor
- artifact runtime can **load** selected tensor elements into u64 slots
- control flow (cond / scheduling keys) can read slots via ScalarExpr

This is the only supported “dynamic value injection” mechanism in v10; it avoids hidden host-side retracing.

## 5. NPU architecture model (multi-NPU support)

v10 introduces an architecture model that is shared across all backends.

### 5.1 Required fields (conceptual)

- **Memory**: address spaces, capacities, access granularity
- **Exec units**: vector/matrix/DMA, concurrency constraints
- **Sync**: barriers/flags and their ordering domains
- **Dispatch model**: core grid, waves, queues, worker types
- **Timing hooks**: how kernels report cycles and how to compose them

### 5.2 Capability matrix

Each backend must expose:

- which schedule directives are enforced
- which CSP features are supported (sync-only vs payload channels)
- which predicate mechanisms are supported

## 6. Backend targets (v10)

### 6.1 cpu_sim

- runnable and validated locally
- serves as the reference correctness backend

### 6.2 ascend_npu

- runnable in a proper Ascend/CANN environment
- must preserve v10 CSP + dispatch semantics

### 6.3 aie

- runnable in an AIE-capable environment (hardware/emulator)
- must preserve v10 CSP + dispatch semantics
- local emit-only fallback is permitted when toolchains are absent, but output must remain structured
- since AIE is dataflow/stream-driven, v10 requires:
  - explicit stream/channel edges in the emitted artifact, and
  - compile-time validation that the stream graph is a DAG **or** a clear “unsupported” error for cyclic stream graphs until
    buffered channels/capacity modeling is introduced.
