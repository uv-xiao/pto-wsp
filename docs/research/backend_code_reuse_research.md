## 1) Shared data structure design (TaskGraph, TaskNode, TensorMap, ReadyQueue)

### Design goals
- One *graph/runtime core* usable by **CPU sim** and **Ascend AICPU** (both are “host schedulers”), while still supporting **AICore/AIE** “device execution”.
- Match proven patterns:
  - **Sliding window task table + TensorMap GC** from `references/pto-isa-lh/src/runtime/pto_runtime_common.h`.
  - **Multi-threaded ready-queue + fanin atomic** from `references/pto-isa-wc/runtime/aicpu/graph_executor.cpp`.
  - **Device-visible Task POD + kernel-id dispatch** from `references/pto-isa-wc/runtime/graph/graph.h` + `references/pto-isa-wc/runtime/aicore/kernel.cpp`.

### Core types (portable, C-compatible layout)
Make a “portable core” header that avoids STL/exceptions so it can be compiled into AICPU/AICore builds.

```cpp
namespace pto::wsp::graph {

// Hierarchy: where does the task execute?
enum class ExecDomain : uint8_t { HostCPU, AscendAICore, AMDAIETile };

// Pool: how is it scheduled on the host scheduler?
// (CPU sim: vector/cube thread pools; Ascend: AIV/AIC pools)
enum class ExecPool : uint8_t { Vector=0, Cube=1, Any=255 };

using TaskId   = uint32_t;
using KernelId = uint16_t;
using StreamId = uint16_t;   // used by pipeline_depth scope="per_stream"
using TargetId = uint16_t;   // per-core/per-tile affinity if needed

struct TensorRegion2D {
  uint64_t base;      // pointer/handle (device-visible)
  int32_t  row_off, col_off;
  int32_t  rows, cols;
};

struct TaskIO {
  TensorRegion2D region;
  uint8_t is_output;  // 0=input, 1=output
};

struct TaskNodePod {
  TaskId    id;
  KernelId  kernel;
  ExecDomain domain;
  ExecPool   pool;
  StreamId   stream;
  TargetId   affinity;     // optional

  // deps
  int32_t fanin;           // treated atomically by host scheduler
  uint32_t fanout_begin;   // into a flat fanout_edges[]
  uint16_t fanout_count;

  // args/IO
  uint16_t num_u64_args;
  uint16_t num_io;
  uint64_t u64_args[/*MAX_ARGS*/];
  TaskIO   io[/*MAX_IO*/];

  // schedule metadata (extended primitives)
  uint16_t sched_tags;     // bitfield: barrier, flush_batch, etc.
};
}
```

### TaskGraph (two-layer: “built graph” + “runtime state”)
Split representation to maximize reuse:

1) **TaskGraphStorage** (mostly immutable after build; device-copyable)
- `tasks[] : TaskNodePod`
- `fanout_edges[] : TaskId` (flat adjacency list)
- (optional) `task_offsets[]` for faster iteration
- (optional) `kernel_table[]` mapping `KernelId -> symbol/name/ABI`

2) **TaskGraphRuntime** (host-scheduler state; not device-copied)
- `TensorMap producer_map` for dependency inference during build (and for pipelined build/execute).
- `ReadyQueueSet ready` for scheduling.
- `WindowState window` to implement `task_window`.
- `IssueGates gates` to implement `pipeline_depth`.
- `DepBatcher batcher` to implement `batch_deps`.
- Counters/tracing hooks.

This directly mirrors what works in pto-isa-lh (`PTORuntime` combines these; you’re just giving it a cleaner, backend-neutral shape).

### TensorMap (dependency inference + GC)
Implement two interchangeable backends:
- **FixedTensorMap** (AICPU + fast CPU sim mode): hash table + entry pool (pto-isa-lh style).
- **DynamicTensorMap** (optional): `unordered_map` for easier debugging.

Key API:
```cpp
struct ProducerRef { TaskId producer; uint32_t generation; };

class TensorMap {
public:
  void insert_output(TensorRegion2D r, ProducerRef p);
  // For inputs: find latest producer whose region overlaps (exact match first; overlap optional)
  bool find_producer(TensorRegion2D r, ProducerRef* out) const;

  // task_window support:
  void gc_before(TaskId oldest_live_task);
};
```

### ReadyQueue (multi-queue + dual-queue support)
Unify CPU-sim dual-queue and Ascend AIV/AIC dispatch by making ready queues *a set*:

```cpp
class ReadyQueueSet {
public:
  void push(ExecPool pool, TaskId tid);
  bool try_pop(ExecPool pool, TaskId* tid);     // per-pool worker
  bool try_pop_any(TaskId* tid);                // host dispatcher / steal
};
```

Provide two implementations selectable by backend/options:
- `ReadyQueueMPMC` (mutex+cv or lock-free ring), for CPU sim and AICPU multi-thread schedulers.
- `ReadyQueueSPSC` (optional) for per-worker fast paths.

---

## 2) Backend plugin interface (Backend base class, compilation options)

### Two-phase compilation: share lowering, specialize emission
To maximize reuse, define a backend-neutral lowered form:

```cpp
namespace pto::wsp::backend {

struct CompileOptions {
  std::string target;              // cpu_sim | ascend_npu | amd_aie
  int opt_level = 2;
  bool debug = false;
  bool profiling = false;

  // Hierarchical knobs (host scheduler vs device execution)
  struct Host {
    int num_threads = 0;           // CPU sim
    int num_aicpu_threads = 0;     // Ascend
  } host;

  struct ScheduleRuntime {
    // lowered extended primitives
    DispatchThresholdConfig dispatch;
    PipelineDepthConfig     pipeline;
    TaskWindowConfig        window;
    BatchDepsConfig         batch_deps;
  } sched;
};

struct LoweredPlan {
  graph::TaskGraphStorage graph;
  // kernel ABI table, constant pools, etc.
  KernelBundle kernels;
  // artifacts needed by backend emitters (templates/MLIR/etc.)
};

class Backend {
public:
  virtual ~Backend() = default;
  virtual std::string name() const = 0;
  virtual bool supports(const ir::Module&) const = 0;

  // Shared lowering happens once (or in a shared helper):
  virtual LoweredPlan lower(const ir::Module&, const CompileOptions&) = 0;

  // Backend-specific emission/build:
  virtual std::unique_ptr<Program> compile(const LoweredPlan&, const CompileOptions&) = 0;
};
}
```

### Plugin/registry
Keep a simple static registry (like `docs/backend-arch.md` already sketches), but register *factories* so backends are link-optional:

```cpp
using BackendFactory = std::unique_ptr<Backend>(*)();

void register_backend(std::string name, BackendFactory f);
Backend& get_backend(std::string name);
```

(If you later want `.so` plugins: add an `extern "C" pto_wsp_backend_init()` entrypoint returning factories; no interface changes needed.)

---

## 3) Code generation template patterns

### Principle: generate only the “thin glue”, link the shared runtime
For Ascend, don’t generate the scheduler; generate:
- AICPU **orchestration** (build graph + call shared executor),
- AICore **kernel dispatch wrappers** (kernel-id switch + ABI unpack),
and link against a shared “graph runtime core” library where possible.

### Recommended template set
- `templates/ascend/aicpu_orchestrate.cc.in`
  - emits: task emission loop(s), schedule config constants, batch flush points, launches executor.
- `templates/ascend/aicore_kernel_dispatch.cc.in`
  - emits: `switch(kernel_id)` calling wrappers (pattern from `references/pto-isa-wc/runtime/aicore/kernel.cpp`).
- `templates/common/kernel_ids.h.in` (or X-macro list)
  - single source of truth for `KernelId` enum + signature metadata.
- `templates/common/task_abi.h.in`
  - defines `TaskNodePod`/`TaskIO` layout and pack/unpack helpers.

### Pattern 1: X-macro kernel list (avoids duplicate switches)
Generate one `KERNEL_LIST(X)` and reuse it in both AICPU and AICore compilation units to create:
- `enum KernelId`
- `dispatch(KernelId, TaskNodePod*)`
- optional name table for profiling

### Pattern 2: “Builder API” calls in templates (not hand-built arrays)
Even for generated code, prefer:
- `builder.alloc_task(kernel, domain, pool, stream, …)`
- `builder.add_input/output(region)`
- `builder.submit(tid)`
Then share the hard parts (TensorMap deps, fanin/fanout wiring, batching) in common code.

### Pattern 3: backend-neutral “schedule runtime config” constants
Lower extended primitives into compact structs and embed them as `constexpr` in generated code so AICPU can run the *same* policy logic as CPU sim.

---

## 4) How extended primitives lower to each backend

### `dispatch_threshold`
- **CPU sim (dual-queue)**: evaluate metric per scope → choose `DispatchPolicy` → route tasks into `ReadyQueueSet(Vector/Cube)` and pick worker target (RR/WS). Two-stage dispatch when `pool_by=exec_unit`: first pool, then policy within pool.
- **Ascend NPU**: metric selects (a) active core set (AIV/AIC), (b) per-pool dispatch policy, (c) possibly AICPU-thread/core assignment strategy (extend the approach in `references/pto-isa-wc/runtime/aicpu/graph_executor.cpp`).
- **AMD AIE**: metric selects **mapping strategy** (grid size/tiling, replication vs sharding). If metric known at compile time: pick one. If runtime-known: compile a small set of variants and select at runtime.

### `pipeline_depth`
- **CPU sim**: implement token gates per `scope` (`global|per_stream|per_pool|per_target`). “Ready but no token” tasks go to a deferred queue; completion returns token and promotes deferred tasks.
- **Ascend NPU**: gate dispatch in AICPU executor using `in_flight < depth` per scope (extend `tasks_in_flight_` into per-stream/per-pool counters); only assign `Handshake.task` when a token is available.
- **AMD AIE**: lower to **FIFO/buffer depth** between tiles/processes (objectfifo depth / double-buffered DMA scheduling), plus host submission throttling if needed.

### `task_window`
- **CPU sim**: adopt pto-isa-lh sliding window semantics (`PTO_TASK_WINDOW_SIZE`, `PTO_TASK_SLOT`, GC of TensorMap); `overflow=stall|evict_completed|error`.
- **Ascend NPU**: treat as codegen-time capacity for task metadata in AICPU-visible memory; with `stall`, enable pipelined build/execute (start executor once threshold reached, like pto-isa-lh `execution_task_threshold`); with `error`, abort generation.
- **AMD AIE**: mainly a host-side bound for descriptor queues and debug dumping; most AIE schedules are static, so it often becomes a compile-time validation limit.

### `batch_deps`
- **CPU sim**: batch dependency inference (local batch map first, then global TensorMap), optionally compress patterned deps into ranges; flush on barriers/stream boundaries/kernel changes.
- **Ascend NPU**: generate “emit tasks → flush deps → run executor” structure; batching reduces AICPU overhead (fewer TensorMap probes / fanout updates) and keeps task metadata contiguous.
- **AMD AIE**: typically compile-time only (dependencies become explicit channels/edges in MLIR); batching is a compile-time optimization pass, not a runtime mechanism.

---

## 5) What to share vs keep backend-specific

### Share (high ROI)
- `graph/`: `TaskNodePod`, `TaskGraphStorage`, `TensorRegion`, `TensorMap` (fixed + optional dynamic), `ReadyQueueSet`.
- `runtime/`: windowing (`task_window`), gating (`pipeline_depth`), dispatch policy evaluation (`dispatch_threshold`), dep batching (`batch_deps`) as backend-neutral utilities.
- `lowering/`: IR → “lowered plan” (kernel IDs, task metadata, schedule-runtime config).
- `codegen/`: template engine + common templates (kernel list, ABI, orchestration skeleton).

### Backend-specific (must diverge)
- **CPU sim**: thread pool implementation details (including dual-queue worker assignment), optional cycle simulation/tracing.
- **Ascend NPU**: handshake protocol, AICPU/AICore build toolchain, device memory APIs, cache-coherency primitives (as in `references/pto-isa-wc/runtime/graph/handshake.h` and `runtime/aicore/kernel.cpp`).
- **AMD AIE**: spatial mapper/layout analyzer, MLIR lowering, XRT/runtime integration, tile placement/stream routing.

If you want, I can turn this into a concrete repo layout + header/API sketch (file-by-file) that matches your existing `docs/backend-arch.md` and the pto-isa-lh/wc reference code paths.