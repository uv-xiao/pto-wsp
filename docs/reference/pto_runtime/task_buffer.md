# Reference: pto-runtime “task-buffer” direction (PTO2 / bounded runtime)

This note summarizes the **bounded runtime** design implemented in pto-runtime’s `tensormap_and_ringbuffer` runtime
(a.k.a. **PTO2** in headers). This is the most relevant direction for PTO‑WSP v10’s:

- `task_window(mode=STALL)` as **real backpressure** to orchestration/expansion
- bounded dependency tracking + bounded intermediate allocation
- actionable deadlock/stall diagnostics (especially once CSP blocking is added)

Primary sources:
- `references/pto-runtime/src/runtime/tensormap_and_ringbuffer/runtime/pto_runtime2.h`
- `references/pto-runtime/src/runtime/tensormap_and_ringbuffer/runtime/pto_shared_memory.h`
- `references/pto-runtime/src/runtime/tensormap_and_ringbuffer/runtime/pto_orchestrator.h`
- `references/pto-runtime/src/runtime/tensormap_and_ringbuffer/runtime/pto_scheduler.h`
- `references/pto-runtime/src/runtime/tensormap_and_ringbuffer/aicpu/aicpu_executor.cpp`

## 0) What “task-buffer” means here

PTO2 is not “just a task window”. It is a coordinated bounded design:

- **Task ring**: bounded in-flight task descriptors (the `task_window_size`)
- **Dep-list pool**: bounded dependency edge storage (fanout lists)
- **Heap ring**: bounded intermediate/output allocation arena (GM heap allocation with ring pointers)
- **TensorMap**: bounded producer lookup structure used for dependency discovery (private)
- **Scope stack**: explicit lifetime / fanout refcount management (private)
- **Ready queues**: bounded queues for runnable tasks (private; and an “orchestrator-ready” queue fast path)

The key property PTO‑WSP v10 wants: when any bounded resource is full, **the orchestrator stalls** (backpressure), which is
the correct semantics foundation for `task_window(mode=STALL)` and for future CSP channel blocking.

## 1) Orchestrator ↔ Scheduler split (why the shared memory exists)

PTO2 splits responsibilities:

- **Orchestrator** (Turing-complete control): submits tasks, allocates intermediate buffers, manages scopes.
- **Scheduler**: turns dependency readiness into worker dispatch (AIC/AIV), tracks completion, frees resources.

They communicate via a **bounded shared memory layout** intended to be device-friendly:

- `PTO2SharedMemoryHeader`: flow-control pointers + layout info
- `PTO2TaskDescriptor[]`: task ring buffer
- `PTO2DepListEntry[]`: dep-list pool (fanout lists)

Pointer/reference:
- `references/pto-runtime/src/runtime/tensormap_and_ringbuffer/runtime/pto_shared_memory.h`

## 2) The “five rings” mapping (PTO‑WSP vocabulary ↔ PTO2 structures)

PTO‑WSP v10 often talks about “five rings” for boundedness. PTO2 already contains the essential shape:

1) **Task ring** → `PTO2TaskDescriptor[]` + `current_task_index` / `last_task_alive`
2) **DepList pool** → `PTO2DepListEntry[]` (bounded fanout edges)
3) **Heap ring** → GM heap arena + `heap_top` / `heap_tail` (allocation + free pointers)
4) **TensorMap pool** → `PTO2TensorMap` entry pool (bounded producer lookup)
5) **Ready queues** → scheduler ready queues + orchestrator-ready queue fast path

The important *design* point is that these are **coupled**: stalling on any one of them should be observable and should
become a first-class stall reason in v10 diagnostics.

## 3) Scope-based liveness (a key hazard PTO‑WSP must preserve)

PTO2 uses explicit scopes:

- `scope_begin()` starts a new scope and causes tasks inside to be lifetime-owned by that scope.
- `scope_end()` releases the scope’s references (fanout refcounts), enabling buffer release once all consumers finish.

Consequence for `task_window` sizing:
- If orchestration cannot reach `scope_end()` because it is stalled on bounded resources (task ring / dep pool / heap ring),
  the runtime can deadlock itself. This is a semantic hazard PTO‑WSP v10 must surface as diagnostics, not as a silent hang.

Pointer:
- `references/pto-runtime/src/runtime/tensormap_and_ringbuffer/runtime/pto_orchestrator.h`

## 4) Dependency discovery via TensorMap (why this matters for PTO‑WSP)

Instead of requiring explicit dependency edges, PTO2 supports a TensorMap-style mechanism:

- outputs are registered into TensorMap as “produced by task X”
- inputs perform a TensorMap lookup to discover producers and build fanin/fanout relationships

This is a strong fit for PTO‑WSP’s “artifact runtime enforces schedule semantics” direction because:
- dependency discovery is kept inside the orchestration/runtime boundary,
- boundedness is enforced by the same ring/pool design,
- it provides a concrete place to later integrate **CSP channels** as additional readiness constraints (beyond tensor deps).

Pointer:
- `references/pto-runtime/src/runtime/tensormap_and_ringbuffer/runtime/pto_tensormap.h`

## 5) Multi-threaded AICPU scheduling (what exists, what’s still missing for v10)

In PTO2 runtime, `kernel_config.py` commonly configures:
- multiple AICPU threads (e.g., 3 scheduler threads + 1 orchestrator thread),
- a worker partitioning model (AIC vs AIV cores).

This is the right “shape” for PTO‑WSP’s multi-scheduler intent, but v10 still needs:
- a stable ABI for user-controlled `dispatch(policy)` (task → scheduler-domain assignment),
- and a stable policy registry contract.

Pointers:
- `references/pto-runtime/examples/tensormap_and_ringbuffer/vector_example/kernels/kernel_config.py`
- `references/pto-runtime/src/runtime/tensormap_and_ringbuffer/aicpu/aicpu_executor.cpp`
- `docs/reference/pto_runtime/gaps.md`

## 6) Implications for PTO‑WSP v10 backend design

- Treat `tensormap_and_ringbuffer` / PTO2 as the **target runtime model** for “semantics-complete” v10 (true backpressure).
- Keep `host_build_graph` as a pragmatic Phase 1 for fast runnable integration and regression tests.
- Design PTO‑WSP artifact packaging so it can evolve from “host orchestration `.so`” (Phase 1) to “device orchestration +
  bounded shared-memory contract” (Phase 2) without changing the front-end DSL.

