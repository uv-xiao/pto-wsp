# Task Graph vs Stream Execution Research for PTO-RT v9

## Overview

This document compares stream-based and task-graph-based execution models to design the `task_graph()` schedule primitive that provides **strict coverage** of pto-isa-lh capabilities (requirement R9).

---

## 1) pto-isa-lh Task Graph Model (STRICT COVERAGE)  
Source-of-truth is the runtime headers/impl, not just the summary doc: `references/pto-isa-lh/src/runtime/pto_runtime_common.h#L36`, `references/pto-isa-lh/src/runtime/pto_runtime_common.c#L174`, `references/pto-isa-lh/src/runtime/pto_runtime_arm64.c#L151`, plus the unified wrapper `references/pto-isa-lh/src/runtime/pto_runtime.h`.

- **Core representation = “task record + explicit DAG counters”**  
  - `PendingTask` stores `task_id`, `func_name/func_ptr`, `args[]`, **`fanin`**, **`fanout[]`**, plus execution hints (`is_cube`) and simulation timing (`earliest_start_cycle`, `end_cycle`). See `references/pto-isa-lh/src/runtime/pto_runtime_common.h#L99`.  
  - Runtime keeps a **sliding window ring** of task slots: `PTO_TASK_WINDOW_SIZE=8192`, slot = `task_id & (WINDOW_SIZE-1)`. See `references/pto-isa-lh/src/runtime/pto_runtime_common.h#L37`.

- **Dependency inference = TensorMap exact-match lookup keyed by region**  
  - A tensor access is modeled as a **2D region** `(raw_tensor, row_offset, col_offset, rows, cols)` in `TensorRegion`. See `references/pto-isa-lh/src/runtime/pto_runtime_common.h#L73`.  
  - TensorMap entry maps **exact region → producer task_id** (`TensorMapEntry.region`, `producer_id`). See `references/pto-isa-lh/src/runtime/pto_runtime_common.h#L134`.  
  - **Hash + equality**: hashed over `(ptr, row_off, col_off, rows, cols)`; equality requires all fields match exactly. See `references/pto-isa-lh/src/runtime/pto_runtime_common.c#L182`.  
  - **Staleness rule**: an entry is “invalid” if `producer_id < window_oldest_pending` (producer known completed and retired). See `references/pto-isa-lh/src/runtime/pto_runtime_common.c#L209`.

- **Graph construction API (the critical flow)**  
  1) `pto_task_alloc_impl(...)` allocates an increasing `task_id`, initializes slot fields, and enforces window behavior by mode. See `references/pto-isa-lh/src/runtime/pto_runtime_common.c#L291`.  
     - Modes (hard behavior differences):  
       - `PTO_MODE_DUMP_GRAPH`: abort orchestration when window full.  
       - `PTO_MODE_EXECUTE/SIMULATE`: **stall** when window full, waiting on `window_not_full`.  
       - `PTO_MODE_BENCHMARK_ONLY`: “fake-advance” `window_oldest_pending` to bound TensorMap growth (since tasks never complete).  
       See the switch in `references/pto-isa-lh/src/runtime/pto_runtime_common.c#L291`.  
  2) `pto_task_add_output(...)` appends an output `TaskArg` and **inserts** `(region → task_id)` into TensorMap. See `references/pto-isa-lh/src/runtime/pto_runtime_common.c#L433`.  
  3) `pto_task_add_input(...)` appends an input `TaskArg`, does TensorMap lookup, and if a producer exists:  
     - adds consumer to producer’s `fanout[]` and increments consumer `fanin`.  
     - protected by `task_mutex`; also checks `producer->is_complete` to avoid a pipelined race.  
     See `references/pto-isa-lh/src/runtime/pto_runtime_common.c#L368`.  
  4) `pto_task_submit(...)` enqueues the task **iff `fanin==0`**; otherwise it just stays pending until deps resolve. See `references/pto-isa-lh/src/runtime/pto_runtime_arm64.c#L151`.  
  5) On completion, runtime iterates the producer’s `fanout[]`, decrements each dependent’s `fanin`, and when it hits zero, the dependent becomes ready. See `references/pto-isa-lh/src/runtime/pto_runtime_arm64.c#L174`.

- **Execution model = “ready queues + workers”, with heterogeneity support**  
  - ARM64: single ready queue; completion directly decrements dependents (“distributed dependency management”). See `references/pto-isa-lh/src/runtime/pto_runtime_arm64.c#L174`.  
  - A2A3: dual ready queues (vector vs cube) keyed by `PendingTask.is_cube`; completion routes newly-ready tasks to the right queue (“dedicated dependency module”). (Implementation lives in `references/pto-isa-lh/src/runtime/pto_runtime_a2a3.c`.)  
  - **Pipelined build+execute**: workers can start before orchestration ends, gated by `execution_task_threshold` / `execution_started` (so dependency inference must be thread-safe; that’s why `pto_task_add_input` checks `producer->is_complete`). See the blocking-get logic around `execution_task_threshold` in `references/pto-isa-lh/src/runtime/pto_runtime_arm64.c#L76`.

- **Simulation capability (still part of “coverage”)**  
  - Tasks can carry a `cycle_func`; workers record traces and enforce `actual_start = max(worker_cycle, earliest_start_cycle)`; completion propagates `earliest_start_cycle` to dependents. Struct + fields: `references/pto-isa-lh/src/runtime/pto_runtime_common.h#L125`.

- **Important limitations you must match (or consciously extend) in v9**  
  - Dependency inference is **exact-region equality** (no overlap detection; no canonical “bytes touched”).  
  - TensorMap key includes **shape**; the same memory accessed with a different `(rows, cols)` descriptor will **not match** (false negatives).  
  - The mechanism is fundamentally “last producer for this exact region”; hazards like WAW/WAR are only handled if the program models them via inputs (e.g., `InOut` as input+output).  
  - Fixed maxima: `PTO_MAX_ARGS=16`, `PTO_MAX_FANOUT=512`, window=8192. See `references/pto-isa-lh/src/runtime/pto_runtime_common.h#L41`.

---

**2) Current v9 stream-based issuing (“streams() model”)**  
The spec-defined surface API is in `docs/spec.md#L838`, and the intended two-phase model is consistent with older v8 analysis (`docs/archive/v8/analysis.md#L223`). In this model:

- **Schedule splits into** `dispatch(...)` (Task→AICPU) + **issue via streams** (`streams(n)`, `stream_by(...)`, `timing(...)`). See `docs/spec.md#L846`.  
- The v8/v9 stream semantics are effectively: “ordered within a stream; concurrent across streams” (CUDA-like), with explicit cross-stream sync via Events/Channels (Events are unified with CSP channels in the spec). See `docs/archive/v8/analysis.md#L260` and Event API `docs/spec.md#L1279`.  
- **Dependencies are currently structural**, coming from workload constructors like `for_each`/`sequential`/CSP channels, not from data hazards on tensors. (`3rdparty/pto-isa/include/pto/rt/schedule.hpp` even states schedule doesn’t encode data deps; deps come from workload types.)

---

**3) Task graph vs streams: what you gain/lose**  

- **Expressiveness**  
  - Streams are great for “mostly-linear per-key pipelines” but awkward for general DAG patterns with **fanin** (task needs multiple parents) and **fanout** (one producer feeds many consumers) unless you add explicit Events/joins everywhere.  
  - pto-isa-lh’s task graph naturally supports arbitrary DAGs because readiness is purely `fanin==0`.

- **Dependency authoring cost**  
  - Streams: dependencies are mostly manual (structure + Events).  
  - Task graph (pto-isa-lh style): dependencies are cheap to author because they’re *inferred* from tensor region usage during orchestration (TensorMap).

- **Runtime overhead vs schedule clarity**  
  - Task graph pays bookkeeping: TensorMap lookups, fanin/fanout updates, window management.  
  - Streams can be lighter-weight, but only if you don’t need DAG inference.

---

**4) Proposed `task_graph()` schedule primitive (alternative to `streams()`)**  
Goal: provide a first-class “pto-isa-lh-compatible” execution mode in v9 (explicit DAG counters + TensorMap inference), while still allowing a stream mode for simple cases.

I recommend modeling `task_graph()` as selecting a different **issue engine**:

```python
program = (workload
  .dispatch(...)                 # unchanged: placement / target selection
  .task_graph(                   # NEW: graph-based issuer (alternative to streams())
      deps=Deps.infer_tensor_map_exact(),     # matches pto-isa-lh behavior
      window=TaskWindow(tasks=8192, overflow="stall"|"abort"|"benchmark"),
      pools=Pools.by_exec_unit(),             # covers is_cube dual-queue
      ready=ReadyPolicy.work_steal()|fifo()|priority(...),
      start=StartPolicy.threshold(n)|after_orchestration(),
      trace=TracePolicy.none()|cycles(...),
  )
  .compile())
```

**Semantics (designed to match pto-isa-lh exactly by default):**
- Each emitted task becomes a `PendingTask`-like record with:
  - `inputs[]` / `outputs[]` tensor regions (v9 should generalize 2D→ND, but keep a fast-path for 2D).  
  - `fanin` integer + `fanout` adjacency (or an equivalent compressed structure).  
  - `exec_unit` trait (generalizing `is_cube`), and optional `cycle_cost` trait (for simulation).  
- **Dependency inference (default = strict pto-isa-lh)**: on task submit:
  - for each input region, do TensorMap exact-key lookup; if found and producer still in-window/not-complete, add edge producer→consumer and increment consumer `fanin`.  
  - for each output region, insert mapping region→this task.  
- **Readiness**: task becomes runnable when `fanin==0`; completion decrements dependents.  
- **Windowing**: enforce a bounded “task metadata + producer map” window with the same modes:
  - `stall` (execute/simulate), `abort` (dump graph), `benchmark` (fake-advance).  
- **Heterogeneous pools**: `Pools.by_exec_unit()` must reproduce A2A3’s dual-queue semantics (vector vs cube), but generalized to N pools.

This gives you a primitive that is strictly capable of reproducing pto-isa-lh’s runtime behavior, while leaving room to add v9-only optimizations like `batch_deps()` later (see `docs/research/extended_primitives_research.md#L92`).

---

**5) Explicit vs inferred dependencies (hybrid model)**  

- **Inferred deps (pto-isa-lh-compatible)**  
  - `Deps.infer_tensor_map_exact()` uses region keys that match pto-isa-lh: `(buffer_id/raw_ptr, offsets, extents)` with **extents included in the key** (important for strict coverage).  
  - Requires v9 tasks to carry **typed access roles**: `In`, `Out`, `InOut` (already in `docs/spec.md` kernel signature discussion), so `InOut` can be treated as “add_input + add_output” to model read-modify-write correctly.

- **Explicit deps (needed for control dependencies / non-tensor hazards)**  
  - Keep structural dependencies (`sequential`, `for_each`, CSP Events/Channels).  
  - Add a compact per-task override for edge injection (conceptually): `task(..., after=[...])` or a builder-level `depends(a, b)` for graph mode.  
  - This lets you express “must run after” even when there is no tensor-region relationship (e.g., side-effects, resource lifetimes, host callbacks).

- **Hybrid rule (recommended default in task_graph mode)**  
  - `Deps.hybrid(infer=..., explicit=True)` = union of:
    - structural/workload deps (today’s system), plus
    - TensorMap inferred RAW deps, plus
    - user-specified explicit edges.  
  This avoids forcing everything through TensorMap (which will always have false negatives in some aliasing/shape-mismatch cases).

---

**6) Redo dispatch/issue design (concise + powerful)**  
Today’s surface API in `docs/spec.md#L846` mixes “issue configuration” across `streams()`, `stream_by()`, `issue(...)`, and `timing(...)`. A cleaner split that also accommodates task graphs:

- Keep `dispatch(...)` as “where does the task run” (AICPU / device / pool routing).  
- Make **one** method the home for *all* “how are tasks released over time” decisions:

```python
program = (workload
  .dispatch(DispatchPolicy.work_steal())   # placement
  .issue(Issue.streams(count=4, by=..., timing=...))          # old model
  # or
  .issue(Issue.task_graph(deps=..., window=..., ready=...))   # new model
  .compile())
```

Then:
- `.streams(n)` / `.stream_by(fn)` / `.timing(p)` become sugar for `issue(Issue.streams(...))`.  
- `.task_graph(...)` becomes sugar for `issue(Issue.task_graph(...))`.  
- Bonus: you can support a third issuer later (e.g., “static topological batches”) without multiplying combinators.

---

**7) Coverage checklist: pto-isa-lh → v9 `task_graph()`**  

- `PendingTask` fields (`fanin/fanout`, args, exec hint, timing): covered by the task-graph issuer record. (`references/pto-isa-lh/src/runtime/pto_runtime_common.h#L99`)  
- TensorMap exact key + stale-by-window rule: covered by `Deps.infer_tensor_map_exact()` + `TaskWindow(tasks=8192, ...)`. (`references/pto-isa-lh/src/runtime/pto_runtime_common.c#L182`)  
- Sliding window modes (stall/abort/benchmark fake-advance): covered by `TaskWindow(..., overflow=..., mode=...)`. (`references/pto-isa-lh/src/runtime/pto_runtime_common.c#L291`)  
- Ready-queue behavior (enqueue only when `fanin==0`, decrement fanin on completion): core semantics of task-graph issuer. (`references/pto-isa-lh/src/runtime/pto_runtime_arm64.c#L151`)  
- Dual-queue (vector/cube): covered by `Pools.by_exec_unit()` / generalized pool routing. (`references/pto-isa-lh/src/runtime/pto_runtime_common.h#L225`)  
- Pipelined execution start threshold + producer-complete race handling: covered by `StartPolicy.threshold(n)` and ensuring dependency insert checks “producer already completed” before adding an edge (same as pto-isa-lh). (`references/pto-isa-lh/src/runtime/pto_runtime_common.c#L406`)  
- Cycle simulation / trace: covered by `TracePolicy.cycles(...)` and storing `earliest_start_cycle/end_cycle`. (`references/pto-isa-lh/src/runtime/pto_runtime_common.h#L125`)

---

**Recommendations (pragmatic next steps)**  
- Implement `Issue.task_graph(...)` as the *new* issuer backend and make `.task_graph(...)` sugar; leave `.streams(...)` as sugar for `Issue.streams(...)` for continuity with `docs/spec.md#L838`.  
- Default `task_graph` dependency inference to **exact TensorMap** (pto-isa-lh compatible), but expose an opt-in `Deps.infer_bytes_overlap()` later to address the known false-negative cases from “shape-in-key” and “overlap not checked”.  
- Add `Deps.hybrid(...)` early; it’s the escape hatch that prevents users from fighting TensorMap limitations.

If you want, I can draft a concrete v9 API block (ready to paste into `docs/spec.md` Section 5) with the exact names/types for `Issue`, `Deps`, `TaskWindow`, `Pools`, and `ReadyPolicy`, plus 2–3 end-to-end examples (softmax DAG, flash-attn tiled fanin, and a dual-queue vector/cube case).
