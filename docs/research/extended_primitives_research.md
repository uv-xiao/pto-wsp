Below is a v9 design for four **extended schedule primitives** that (a) **subsume** pto-isa-lh’s dual-queue + sliding-window behavior and (b) add capabilities pto-isa-lh cannot express, while fitting cleanly into the existing `dispatch / streams / timing` model.

---

## 1) `dispatch_threshold()` — multi-level, threshold-selected dispatch (binary-expansion aware)

### Goal
Make dispatch a **piecewise policy** selected by a **level** derived from either:
- a **program-level** runtime parameter (e.g., `seq_len`, `num_tiles`), mirroring binary-expansion thresholds; and/or
- a **task-level** metric (e.g., `tile_rows`, `kernel.estimated_cycles`), mirroring adaptive tiling and residual handling.

### Python API (proposed)
```python
schedule = schedule.dispatch_threshold(
    metric=Metric.num_tiles(),                  # or Metric.seq_len(), Metric.task(lambda t: ...)
    thresholds=[256, 512, 1024, 2048, 4096],    # typically power-of-2 (binary expansion), but not required
    policy_per_level={
        0:    DispatchPolicy.round_robin(1),
        256:  DispatchPolicy.round_robin(2),
        512:  DispatchPolicy.round_robin(4),
        1024: DispatchPolicy.work_steal(),
    },
    default=DispatchPolicy.work_steal(),
    # optional but important for “dual-queue++”:
    pool_by=PoolBy.exec_unit(),                 # routes Vector/Cube/Either to separate executor pools
)
```

### Semantics
- Compute `v = metric(ctx)` (program-level) or `v = metric(task, ctx)` (task-level).
- Select `level = max({t in thresholds | v >= t}, default=0)`.
- Apply `policy_per_level[level]` (or `default`) to map the task into a target executor **within its pool**.
- If `pool_by` is present, dispatch becomes **2-stage**:
  1) choose pool (e.g., Vector vs Cube workers) from kernel traits,
  2) choose target within that pool using the level-selected policy.

### Why strictly more powerful than pto-isa-lh
- pto-isa-lh has a **hard-coded** dual-queue split (`is_cube`) and a single routing scheme; this generalizes to:
  - N pools (not just 2),
  - per-level scaling (more workers for larger work),
  - per-task adaptive behavior (e.g., residual tiles dispatched differently).

### IR / serialization (v9)
Add `DispatchThresholdNode { metric, thresholds[], entries[level]->DispatchNode, pool_by }`.
- **Compile-time**: if metric resolves to constant (static axes), fold to a single `DispatchNode`.
- **Runtime**: evaluate once per program (program-level metric) or per task (task-level metric).

---

## 2) `pipeline_depth()` — controlled in-flight issuing (double/triple buffering)

### Goal
Provide explicit control over **how many tasks may be in-flight** per scope, enabling:
- **double buffering** (`depth=2`),
- **triple buffering** (`depth=3`),
- and more general bounded pipelining that pto-isa-lh does not have.

### Python API (proposed)
```python
schedule = schedule.pipeline_depth(
    depth=2,                        # 1 disables pipelining
    scope="per_stream",             # "global" | "per_stream" | "per_target" | "per_pool" | "per_key"
    key=Key.kernel_id(),            # only used when scope="per_key"
    on_full="stall",                # "stall" | "spill_to_readyq" | "error"
)
```

### Semantics
The issuer maintains a token/semaphore per scope:
- issuing consumes 1 token,
- completion returns 1 token,
- if empty → `on_full` behavior triggers.

This is **orthogonal** to:
- `streams(n)` (how many concurrent queues exist),
- `timing(...)` (when to consider issuing),
- `dispatch...` (where tasks go).

### Why strictly more powerful than pto-isa-lh
pto-isa-lh can stall due to window exhaustion, but it cannot express “keep exactly 2–3 iterations in flight per stream/queue” as a first-class scheduling contract.

### IR
`PipelineDepthNode { depth, scope, key_kind, on_full }`.

---

## 3) `task_window()` — explicit task-metadata / dependency-state window sizing

### Goal
Generalize pto-isa-lh’s fixed `PTO_TASK_WINDOW_SIZE=8K` sliding window into a configurable, multi-scope mechanism that controls:
- how many task records are retained,
- how long TensorMap/dependency state is kept,
- and what happens on overflow.

### Python API (proposed)
```python
schedule = schedule.task_window(
    size=8192,
    unit="tasks",                   # "tasks" | "bytes" | "cycles_est"
    scope="global",                 # or "per_pool"/"per_stream"
    overflow="stall",               # "stall" | "evict_completed" | "evict_producer_state" | "error"
    min_size=1024, max_size=262144, # optional bounds for runtime auto-tuning
)
```

### Semantics (reference implementation model)
- Maintain a ring of task slots per scope with **generation counters** (safe reuse).
- Retire slots when tasks complete; optionally retire producer TensorMap entries when no longer needed.
- On overflow:
  - `stall`: wait for completions (matches pto-isa-lh EXECUTE/SIMULATE behavior),
  - `error`: matches pto-isa-lh DUMP_GRAPH abort,
  - eviction modes can match pto-isa-lh BENCHMARK_ONLY cleanup, but are now explicit.

### Why strictly more powerful than pto-isa-lh
- pto-isa-lh exposes one hard-coded window size and mode-dependent behavior.
- v9 exposes programmable scope, units, overflow policy, and (optionally) adaptive resizing.

### IR
`TaskWindowNode { size_expr, unit, scope, overflow, bounds }`.

---

## 4) `batch_deps()` — batched dependency resolution / edge compression

### Goal
Reduce dependency-tracking overhead (TensorMap lookups + fanin/fanout updates) by:
- deferring resolution to batches for locality/vectorization,
- and optionally compressing regular dependency patterns (common in tiled workloads).

### Python API (proposed)
```python
schedule = schedule.batch_deps(
    threshold=128,                  # batch size before resolving deps
    by="kernel",                    # "kernel" | "stream" | "dispatch_target" | "none"
    mode="defer",                   # "defer" | "compress_ranges"
    flush_on=["barrier", "kernel_change", "stream_boundary"],
)
```

### Semantics
- Accumulate submitted tasks into a batch.
- Resolve dependencies for the batch in a tight loop:
  - build a local “recent producers” table for outputs in the batch,
  - resolve inputs against (a) batch-local producers, then (b) global TensorMap,
  - commit edges/fanin updates in bulk.
- `compress_ranges` allows representing patterned deps (e.g., per-tile stencil / fixed offsets) as compact descriptors internally, expanding only if needed.

### Why strictly more powerful than pto-isa-lh
pto-isa-lh resolves deps eagerly per `pto_task_add_input/output`; batching (and optional compression) enables new performance regimes and new representations, while still allowing the eager behavior with `threshold=1`.

### IR
`BatchDepsNode { threshold, by_kind, mode, flush_triggers }`.

---

## Putting it together (covers pto-isa-lh + more)
A v9 schedule that mirrors **binary-expansion thresholds + adaptive tiles**, supports **dual-queue routing**, enforces **double buffering**, keeps a **sliding window**, and batches deps:

```python
program = (workload
  .dispatch_threshold(
      metric=Metric.num_tiles(),                 # binary-expansion style thresholds
      thresholds=[256,512,1024,2048,4096],
      policy_per_level={0: RR(1), 256: RR(2), 512: RR(4), 1024: WS()},
      pool_by=PoolBy.exec_unit())                # dual-queue generalized
  .streams(4)
  .pipeline_depth(2, scope="per_stream")         # double buffering
  .task_window(size=8192, overflow="stall")      # sliding window (explicit)
  .batch_deps(threshold=128, by="kernel")        # reduce overhead
  .timing(TimingPolicy.immediate)
  .compile())
```

If you want, I can also propose the matching `.pto` assembly syntax and the minimal C++ `ScheduleIR` node definitions needed so these are serializable (i.e., not relying on Python lambdas for anything that must round-trip).