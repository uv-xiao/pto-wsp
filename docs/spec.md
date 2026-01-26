# PTO Workload-Schedule Programming (PTO-WSP) v9: API Specification

## 1. Overview

This specification defines the v9 API for PTO-ISA runtime extension, supporting:

1. **Python frontend** with declarative and combinator-style APIs
2. **Typed workload expressions** preserved from v8
3. **CSP pipeline-parallel** with `Channel`, `Process`, `consume`
4. **Spatial schedule primitives** for dataflow architectures
5. **Multi-backend compilation** (CPU sim, NPU, AIE)

### 1.1 Modules

```python
# Core module
from pto_wsp import (
    # Axis types
    Dense, DenseDyn, Ragged, Sparse,

    # Workload definition
    workload,  # @workload decorator
    P,         # Loop constructors namespace (P, P.seq, P.sel, P.pipe, P.when)

    # Workload primitives
    combine, sequential,

    # Kernel definition (NEW in v9 - R7)
    kernel,                     # @kernel decorator (builder-style)
    jit_kernel,                 # @jit_kernel decorator (Triton-style, RECOMMENDED)
    tl,                         # Tile language primitives (tl.load, tl.store, tl.matmul, etc.)
    In, Out, InOut, Constexpr,  # Type annotations for kernel signatures
    Tile, Scalar,               # Typed kernel parameters

    # CSP primitives
    Channel, process, send, consume, connect, replicate,

    # Schedule policies
    DispatchPolicy, IssuePolicy, TimingPolicy,

    # Task graph configuration
    Deps, ReadyPolicy, StartPolicy, TracePolicy, Pools, TaskWindow,

    # Tensor layout (R10)
    TensorLayout, TensorShard, TensorReplicate, MemLayout,
    relayout, allreduce, allgather, reduce_scatter,

    # Spatial layout
    Shard, Replicate,

    # Execution
    Program,

    # Legacy (deprecated)
    # register_kernel, ExternalKernel, npu, NPUFunction
)
```

### 1.2 Namespace (C++ IR)

```cpp
namespace pto::wsp::ir {
    // IR node types
    // See ir-design.md for details
}

namespace pto::wsp::backend {
    // Backend interface
    // See backend-arch.md for details
}
```

---

## 2. Type System

### 2.1 Axis Types

Axes describe iteration spaces.

```python
from pto_wsp import Dense, DenseDyn, Ragged, Sparse

# Static size axis
batch = Dense[4]        # 4 elements, known at compile time
heads = Dense[8]        # 8 elements

# Dynamic size axis
batch = DenseDyn(batch_size)    # Runtime size
seq = DenseDyn(seq_len)

# Ragged axis (variable lengths per outer element)
tokens = Ragged(batch_size, lengths)  # lengths[i] = tokens in batch i

# Sparse axis (CSR format)
routing = Sparse(batch_size, indptr, indices)  # MoE routing
```

**Type Definitions:**

```python
class Dense(Generic[N]):
    """Static size axis."""
    size: int  # Compile-time constant N

class DenseDyn:
    """Dynamic size axis."""
    size: int  # Runtime value

    def __init__(self, size: int | IntVar): ...

class Ragged:
    """Variable length per outer element."""
    outer_size: int
    lengths: list[int]

    def __init__(self, outer_size: int, lengths: list[int]): ...
    def length(self, idx: int) -> int: ...
    def total(self) -> int: ...

class Sparse:
    """CSR format for sparse iteration."""
    outer_size: int
    indptr: list[int]   # Row pointers
    indices: list[int]  # Column indices

    def __init__(self, outer_size: int, indptr: list[int], indices: list[int]): ...
    def nnz(self) -> int: ...
    def row_nnz(self, row: int) -> int: ...
```

### 2.2 Tensor Type

```python
from pto_wsp import Tensor, DType, Location

class DType(Enum):
    F16 = "f16"
    BF16 = "bf16"
    F32 = "f32"
    F64 = "f64"
    I8 = "i8"
    I16 = "i16"
    I32 = "i32"
    I64 = "i64"
    U8 = "u8"
    U16 = "u16"
    U32 = "u32"
    U64 = "u64"

class Location(Enum):
    Global = "global"   # Global memory (HBM)
    L2 = "l2"          # L2 cache
    UB = "ub"          # Unified Buffer (AICore local)
    L1 = "l1"          # L1 buffer

class Tensor:
    data: Any           # Pointer/handle to data
    shape: tuple[int, ...]
    dtype: DType
    location: Location
    layout: Layout      # NEW: Layout refinement (R8, R10)

    def __getitem__(self, idx: int) -> Tensor: ...
    def slice(self, start: int, end: int) -> Tensor: ...
    def nbytes(self) -> int: ...
```

### 2.3 Layout Types (New in v9 - R8, R10)

Layout is a **type-level refinement** (not a schedule primitive). See `docs/research/type_system_research.md` for rationale.

```python
from pto_wsp import Layout, Shard, Replicate, MemLayout

# Distribution types (Dato-style)
class Replicate:
    """Full data on each tile/worker."""
    pass

class Shard:
    """Partitioned along mesh axis."""
    def __init__(self, mesh_axis: int):
        self.mesh_axis = mesh_axis

# Memory layout (Triton-style composable layout)
class MemLayout:
    """Physical memory arrangement."""
    strides: tuple[int, ...]
    order: tuple[int, ...]       # Iteration order (contiguity)
    swizzle: Optional[str] = None  # Bank conflict avoidance

    def compose(self, other: MemLayout) -> MemLayout: ...
    def permute(self, perm: tuple[int, ...]) -> MemLayout: ...
    def tile(self, tile_shape: tuple[int, ...]) -> MemLayout: ...

# Unified Layout = Distribution × Memory
class Layout:
    """Layout type with distribution and memory facets."""
    dist: tuple[Replicate | Shard, ...]  # Per-dimension distribution
    mem: Optional[MemLayout] = None       # Physical layout (optional)

    def __init__(self, *dist: Replicate | Shard, mem: MemLayout = None):
        self.dist = dist
        self.mem = mem

# Tensor with layout refinement
Q: Tensor[F16, (B, H, S, D), TensorLayout(TensorShard(0), TensorReplicate(), TensorReplicate(), TensorReplicate())]
K: Tensor[F16, (B, S, D), TensorLayout(TensorReplicate(), TensorShard(1), TensorReplicate())]
```

**Layout Compatibility (Dato rules):**

```python
def tensor_layout_join(a: TensorLayout, b: TensorLayout) -> TensorLayout:
    """Join layouts for elementwise operations."""
    # R ⊔ S(i) = S(i)
    # S(i) ⊔ S(j) = error if i != j
    ...

# Explicit redistribution required when incompatible
y = relayout(x, to=TensorLayout.sharded(dim=0, rank=2))  # Explicit!
z = allreduce(partial_sum)                                # Explicit collective
```

### 2.4 Task Type

```python
from pto_wsp import Task, KernelRef

class Task:
    """Single kernel invocation."""
    kernel: KernelRef        # Kernel reference (not string! - R7)
    params: list[Any]        # Parameters (axis values)
    resources: list[Tensor]  # Input/output tensors (with Layout types)
    id: int                  # Unique ID (assigned at enumeration)

    def get(self, axis: str) -> int:
        """Get axis value for dispatch/stream policies."""
        ...
```

---

## 3. Workload Primitives

Workloads are constructed using the **`@workload` decorator with symbolic loops** (RECOMMENDED). This TileLang-inspired syntax is concise while maintaining explicit parallelism semantics.

### 3.0 Recommended Syntax: `@workload` + Symbolic Loops

```python
from pto_wsp import workload, P, kernel, In, Out, Tensor

# Define kernel using @kernel decorator (JIT-style - R7)
@kernel
def attn(Q: In[Tensor], K: In[Tensor], V: In[Tensor], O: Out[Tensor]):
    ...

# RECOMMENDED: @workload + symbolic loops
@workload
def attention(batch, heads):
    for b, h in P(batch, heads):        # P = Parallel grid
        attn[b, h](Q=Q[b,h], K=K[b], V=V[b], O=O[b,h])

# Type: Workload[DenseDyn × Dense[8], AttnTask, Independent]
```

**Why this syntax:**
- **Concise**: ~3 lines vs ~7 lines for context manager style
- **TileLang-inspired**: Single-letter namespace `P` like TileLang's `T`
- **Explicit semantics**: `P()` = parallel, `P.seq()` = sequential
- **Direct kernel calls**: `kernel[axes](...)` instead of string-based `task("name", ...)`

### 3.0.1 Loop Constructors (`P` Namespace)

The `P` namespace provides loop constructors with explicit parallelism semantics:

| Constructor | Semantics | Dependency Type |
|-------------|-----------|-----------------|
| `P(axes...)` | Parallel grid (= `P.grid()`) | `Independent` |
| `P.seq(axis)` | Sequential iteration | `Sequential` |
| `P.sel(sparse)` | Sparse iteration | `Selected` |
| `P.pipe(stages, depth=N)` | Pipelined stages | `Pipeline` |
| `P.when(pred)` | Conditional branch | `Conditional` |

**Examples:**

```python
from pto_wsp import workload, P

# P() - Parallel grid (concise alias for P.grid())
@workload
def attention(batch, heads):
    for b, h in P(batch, heads):       # All B×H iterations run in parallel
        attn[b, h](Q[b,h], K[b], V[b], O[b,h])

# P.seq() - Sequential iteration
@workload
def scan(seq_len):
    for i in P.seq(seq_len):           # Must run in order
        scan_step[i](input[i], output[i])

# P.sel() - Sparse iteration (MoE routing)
@workload
def moe(batch, routing):
    for b in P(batch):
        for e in P.sel(routing[b]):    # Only selected experts
            expert[b, e](tokens[b], weights[e])

# Nested with mixed semantics
@workload
def staged_compute(stages, tiles):
    for s in P.seq(stages):            # Sequential across stages
        for t in P(tiles):             # Parallel within stage
            compute[s, t](...)
```

**Conditional Workloads:**

```python
@workload
def tiered_attention(batch, seq_lens):
    for b in P(batch):
        with P.when(seq_lens[b] <= 2048):
            attn_2k[b](...)
        with P.otherwise():
            attn_8k[b](...)
```

**Implementation Note**: The `@workload` decorator wraps the function to create an implicit builder context. Symbolic loops (`P()`, `P.seq()`, etc.) yield iteration variables and emit workload nodes on exit.


### 3.1 Parallel Primitives

**`P(axes...)` / `P.grid(axes...)`** - Independent iteration (all can run in parallel):

```python
@workload
def attention(batch, heads):
    for b, h in P(batch, heads):           # Type: Independent
        attn[b, h](Q[b,h], K[b], V[b], O[b,h])
```

### 3.2 Sequential Primitives

**`P.seq(axis)`** - Sequential iteration (task[i] depends on task[i-1]):

```python
@workload
def scan(seq_len):
    for i in P.seq(seq_len):               # Type: Sequential
        scan_step[i](input[i], output[i])
```

### 3.3 Selection Primitives

**`P.sel(sparse)`** - Sparse iteration over selected indices (MoE routing):

```python
@workload
def moe(batch, routing):
    for b in P(batch):
        for e in P.sel(routing[b]):        # Only selected experts
            expert[b, e](tokens[b], weights[e])
```

### 3.4 Conditional Primitives

**`P.when(pred)` / `P.otherwise()`** - Conditional workload selection:

```python
@workload
def tiered_attention(batch, seq_lens):
    for b in P(batch):
        with P.when(seq_lens[b] <= 2048):
            attn_2k[b](...)
        with P.when(seq_lens[b] <= 8192):  # Nested condition
            attn_8k[b](...)
        with P.otherwise():
            attn_32k[b](...)
```

### 3.5 Kernel Calls (Leaf Workloads)

**Direct kernel call** `kernel[axes](args...)` is the recommended leaf workload:

```python
@kernel
def attn(Q: In[Tensor], K: In[Tensor], V: In[Tensor], O: Out[Tensor]): ...

@workload
def attention(batch, heads):
    for b, h in P(batch, heads):
        attn[b, h](Q=Q[b,h], K=K[b], V=V[b], O=O[b,h])  # Direct kernel call
```

### 3.6 Composition Primitives

**`combine(w1, w2, ...)`** - Unordered composition (schedule determines order):

```python
layer = combine(rms_norm, attention, ffn)  # Type: Combined
```

**`sequential(w1, w2, ...)`** - Ordered composition (w2 waits for w1):

```python
pipeline = sequential(load, compute, store)  # Type: Sequential
```

### 3.7 CSP Primitives (Pipeline-Parallel)

CSP (Communicating Sequential Processes) primitives enable pipeline-parallel workloads via channels and processes. CSP is part of the workload model (established in v7/v8).

**Channel** - Typed, bounded communication:

```python
ch = Channel[Tile](depth=2)   # Double-buffered channel
event = Channel(depth=0)      # Event (rendezvous)
```

**P.pipe()** - Pipeline scope with channels (concise syntax):

```python
@workload
def pipeline(tiles):
    ch = Channel[Tile](depth=2)

    with P.pipe():                          # Pipeline scope
        for i in P.seq(tiles):              # Producer
            send(ch, load[i](data))

        with consume(ch) as tile:           # Consumer (replaces recv())
            compute(tile)
```

**connect()** - Wire processes into pipeline (builder syntax):

```python
loader = process("loader").produces(ch).body(...)
computer = process("computer").consumes(ch).produces(out).body(...)
storer = process("storer").consumes(out).body(...)

pipeline = connect([loader, computer, storer], [ch, out])
pipeline.start()
pipeline.join()
```

**replicate()** - Multiple process instances:

```python
connect([generator, replicate(worker, 4)], [work_queue])
```

---

## 4. Schedule API

Schedules use **combinator style**: each method returns a new schedule with the policy applied, enabling type-safe chaining.

### 4.1 Combinator-Style Schedule

```python
# Create schedule from workload using combinator style
program = (workload
    .dispatch(DispatchPolicy.round_robin(4))
    .streams(2)
    .stream_by(lambda t: t.params[0] % 2)
    .timing(TimingPolicy.immediate)
    .compile())

# Or step-by-step for readability
schedule = workload.dispatch(DispatchPolicy.affinity(lambda t: t.batch))
schedule = schedule.streams(4)
schedule = schedule.stream_by(lambda t: t.head % 4)
schedule = schedule.timing(TimingPolicy.interleaved(4))

# Spatial primitives (new in v9)
schedule = schedule.spatial_map(grid=(4, 4))
schedule = schedule.layout(Q, Shard(dim=0), Replicate())

# Compile to executable program
program = schedule.compile()
```

**Key difference from v8:** The schedule is bound to the workload via combinator methods, enabling better type checking. Each method returns a new schedule with accumulated policies.

### 4.2 Dispatch Policies

```python
from pto_wsp import DispatchPolicy

# Round-robin across AICPUs
schedule.dispatch(DispatchPolicy.round_robin(num_aicpus))

# Same axis value → same AICPU
schedule.dispatch(DispatchPolicy.affinity(lambda t: t.batch))

# Hash-based
schedule.dispatch(DispatchPolicy.hash(lambda t: t.key))

# Dynamic load balancing
schedule.dispatch(DispatchPolicy.work_steal())

# Custom function
schedule.dispatch(DispatchPolicy.dispatch_by(lambda t: custom_logic(t)))
```

### 4.3 Issue Policies

```python
from pto_wsp import IssuePolicy

# Stream assignment by key
schedule.stream_by(lambda t: t.head % num_streams)

# All tasks in single stream
schedule.issue(IssuePolicy.single_stream())

# Per-axis stream assignment
schedule.issue(IssuePolicy.per_axis(axis))

# Priority ordering
schedule.issue(IssuePolicy.priority(lambda t: t.seq_len))
```

### 4.4 Timing Policies

```python
from pto_wsp import TimingPolicy

# Issue immediately when dependencies satisfied
schedule.timing(TimingPolicy.immediate)

# Batch N tasks before issuing
schedule.timing(TimingPolicy.batched(n))

# Interleaved across streams
schedule.timing(TimingPolicy.interleaved(streams))

# Rate limiting
schedule.timing(TimingPolicy.rate_limit(tasks_per_ms))
```

### 4.5 Spatial Primitives (New in v9)

```python
# Spatial mapping: workload → tile grid (defines the Mesh environment)
schedule.spatial_map(grid=(4, 4))  # 4x4 tile array
```

Note: Layout is a **type-level refinement** on tensors (see Section 2.3), not a schedule primitive. `spatial_map(grid=...)` defines the Mesh environment for validating `Shard(mesh_axis=...)` refinements.

### 4.6 Extended Primitives (New in v9)

These primitives are **strictly more powerful** than pto-isa-lh's runtime, providing fine-grained control over dispatch, pipelining, memory management, and dependency resolution.

#### 4.6.1 dispatch_threshold()

Multi-level dispatch based on thresholds (like binary expansion):

```python
from pto_wsp import Metric, PoolBy

schedule = schedule.dispatch_threshold(
    metric=Metric.num_tiles(),                  # or Metric.seq_len(), Metric.task(...)
    thresholds=[256, 512, 1024, 2048, 4096],    # Typically power-of-2
    policy_per_level={
        0:    DispatchPolicy.round_robin(1),
        256:  DispatchPolicy.round_robin(2),
        512:  DispatchPolicy.round_robin(4),
        1024: DispatchPolicy.work_steal(),
    },
    default=DispatchPolicy.work_steal(),
    pool_by=PoolBy.exec_unit(),                 # Routes Vector/Cube to separate pools
)
```

**Semantics**: Computes `v = metric(ctx)`, selects `level = max({t in thresholds | v >= t})`, applies corresponding policy. With `pool_by`, dispatch becomes 2-stage: choose pool (Vector/Cube), then target within pool.

#### 4.6.2 pipeline_depth()

Controlled in-flight issuing for double/triple buffering:

```python
schedule = schedule.pipeline_depth(
    depth=2,                        # 1 disables pipelining
    scope="per_stream",             # "global" | "per_stream" | "per_target" | "per_pool"
    on_full="stall",                # "stall" | "spill_to_readyq" | "error"
)
```

**Semantics**: Maintains token/semaphore per scope. Issuing consumes 1 token; completion returns 1 token. Orthogonal to `streams()`, `timing()`, and `dispatch()`.

#### 4.6.3 task_window()

Explicit task-metadata / dependency-state window sizing:

```python
schedule = schedule.task_window(
    size=8192,
    unit="tasks",                   # "tasks" | "bytes" | "cycles_est"
    scope="global",                 # or "per_pool" | "per_stream"
    overflow="stall",               # "stall" | "evict_completed" | "error"
    min_size=1024, max_size=262144, # Optional bounds for auto-tuning
)
```

**Semantics**: Maintains ring of task slots per scope with generation counters. On overflow, applies specified policy.

#### 4.6.4 batch_deps()

Batched dependency resolution for reduced overhead:

```python
schedule = schedule.batch_deps(
    threshold=128,                  # Batch size before resolving deps
    by="kernel",                    # "kernel" | "stream" | "dispatch_target" | "none"
    mode="defer",                   # "defer" | "compress_ranges"
    flush_on=["barrier", "kernel_change", "stream_boundary"],
)
```

**Semantics**: Accumulates tasks, resolves dependencies in batch for locality/vectorization. `compress_ranges` represents patterned deps as compact descriptors.

#### 4.6.5 Complete Example with Extended Primitives

```python
program = (workload
    .dispatch_threshold(
        metric=Metric.num_tiles(),
        thresholds=[256, 512, 1024, 2048, 4096],
        policy_per_level={0: RR(1), 256: RR(2), 512: RR(4), 1024: WS()},
        pool_by=PoolBy.exec_unit())
    .streams(4)
    .pipeline_depth(2, scope="per_stream")
    .task_window(size=8192, overflow="stall")
    .batch_deps(threshold=128, by="kernel")
    .timing(TimingPolicy.immediate)
    .compile())
```

### 4.7 Task Graph Execution (New in v9 - R9)

`task_graph()` provides an alternative to stream-based issuing, using **pto-isa-lh-compatible** DAG execution with TensorMap-based dependency inference. This achieves **strict coverage** of pto-isa-lh runtime capabilities.

**When to use task_graph vs streams:**

| Criterion | Use `streams()` | Use `task_graph()` |
|-----------|----------------|-------------------|
| Dependency pattern | Linear pipelines, per-key ordering | General DAGs with fanin/fanout |
| Dependency specification | Manual (structural + Events) | Automatic (inferred from tensor regions) |
| Overhead | Lower | Higher (TensorMap, fanin/fanout tracking) |
| pto-isa-lh compatibility | Partial | Full |

#### 4.7.1 task_graph() Primitive

```python
from pto_wsp import Deps, TaskWindow, Pools, ReadyPolicy, StartPolicy, TracePolicy

program = (workload
    .dispatch(DispatchPolicy.work_steal())
    .task_graph(
        # Dependency inference (pto-isa-lh compatible by default)
        deps=Deps.infer_tensor_map_exact(),

        # Sliding window (matches pto-isa-lh PTO_TASK_WINDOW_SIZE)
        window=TaskWindow(tasks=8192, overflow="stall"),

        # Pool routing (generalizes is_cube dual-queue)
        pools=Pools.by_exec_unit(),

        # Ready queue policy
        ready=ReadyPolicy.work_steal(),

        # Pipelined execution start
        start=StartPolicy.threshold(1024),

        # Cycle simulation/tracing
        trace=TracePolicy.cycles(cost_fn=my_cost_fn),
    )
    .compile())
```

#### 4.7.2 Dependency Inference Modes

```python
class Deps:
    @staticmethod
    def infer_tensor_map_exact() -> Deps:
        """pto-isa-lh compatible: exact region key (ptr, offsets, extents)."""
        ...

    @staticmethod
    def infer_bytes_overlap() -> Deps:
        """Enhanced: detect overlapping byte ranges (handles shape mismatches)."""
        ...

    @staticmethod
    def explicit_only() -> Deps:
        """No inference; only structural + user-specified edges."""
        ...

    @staticmethod
    def hybrid(infer: Deps = infer_tensor_map_exact(), explicit: bool = True) -> Deps:
        """Union of inferred + structural + explicit edges (recommended)."""
        ...
```

#### 4.7.3 TaskWindow Configuration

```python
class TaskWindow:
    def __init__(
        self,
        tasks: int = 8192,           # Max pending tasks (like PTO_TASK_WINDOW_SIZE)
        overflow: str = "stall",     # "stall" | "abort" | "benchmark"
    ):
        ...

# Overflow modes (match pto-isa-lh):
# - "stall": Block orchestration when full (PTO_MODE_EXECUTE/SIMULATE)
# - "abort": Stop orchestration when full (PTO_MODE_DUMP_GRAPH)
# - "benchmark": Fake-advance window (PTO_MODE_BENCHMARK_ONLY)
```

#### 4.7.4 Pool Routing

```python
class Pools:
    @staticmethod
    def single() -> Pools:
        """Single ready queue (ARM64 mode)."""
        ...

    @staticmethod
    def by_exec_unit() -> Pools:
        """Dual queues: Vector vs Cube (A2A3 mode)."""
        ...

    @staticmethod
    def custom(route_fn: Callable[[Task], int], num_pools: int) -> Pools:
        """N-way routing by custom function."""
        ...
```

#### 4.7.5 Ready Queue Policies

```python
class ReadyPolicy:
    @staticmethod
    def fifo() -> ReadyPolicy:
        """First-in-first-out."""
        ...

    @staticmethod
    def work_steal() -> ReadyPolicy:
        """Work stealing across workers."""
        ...

    @staticmethod
    def priority(key: Callable[[Task], int]) -> ReadyPolicy:
        """Priority queue by key function."""
        ...
```

#### 4.7.6 Unified Issue API (Alternative Design)

For cleaner API, both streams and task_graph can be accessed via `.issue()`:

```python
# Stream-based (existing)
program = workload.dispatch(...).issue(Issue.streams(count=4, by=..., timing=...)).compile()

# Task-graph-based (new)
program = workload.dispatch(...).issue(Issue.task_graph(deps=..., window=...)).compile()

# Sugar methods (preserve backward compatibility)
workload.streams(4)        # = workload.issue(Issue.streams(count=4))
workload.task_graph(...)   # = workload.issue(Issue.task_graph(...))
```

#### 4.7.7 pto-isa-lh Coverage Checklist

| pto-isa-lh Feature | v9 Coverage |
|--------------------|-------------|
| `PendingTask` (fanin/fanout, args, is_cube) | Task record in task_graph issuer |
| TensorMap exact-key lookup | `Deps.infer_tensor_map_exact()` |
| Sliding window (8192 slots) | `TaskWindow(tasks=8192, ...)` |
| Window overflow modes | `overflow="stall"/"abort"/"benchmark"` |
| Dual-queue (vector/cube) | `Pools.by_exec_unit()` |
| Pipelined start threshold | `StartPolicy.threshold(n)` |
| Cycle simulation | `TracePolicy.cycles(...)` |

### 4.8 Backend Applicability

Different schedule primitives may not be applicable to all backends. The framework validates schedule primitives at compile time based on target backend capabilities.

#### 4.8.1 Backend Capability Table

| Primitive | CPU Sim | Ascend NPU | AMD AIE |
|-----------|---------|------------|---------|
| `dispatch(round_robin)` | ✓ | ✓ | ✗ |
| `dispatch(work_steal)` | ✓ | ✓ | ✗ |
| `streams()` | ✓ | ✓ | ✗ |
| `task_graph()` | ✓ | ✓ | ✓ |
| `spatial_map()` | ✗ | ✗ | ✓ |
| `pipeline_depth()` | ✓ (tasks) | ✓ (tasks) | ✓ (FIFO) |

#### 4.8.2 Compile-Time Validation

```python
# Errors at compile time if schedule uses unsupported primitives
program = (workload
    .dispatch(DispatchPolicy.work_steal())  # ⚠️ Not supported on AIE
    .spatial_map(grid=(4, 4))
    .compile(target="amd_aie"))             # Error raised here
```

**Error message:**
```
ScheduleError: dispatch(work_steal) not supported for target 'amd_aie'
  Hint: AMD AIE uses spatial_map() for task placement, not dispatch policies
```

#### 4.8.3 Backend-Specific Extensions

```python
# Backend extensions via qualified names
from pto.rt.backend.ascend import double_buffer, prefetch
from pto.rt.backend.aie import tile_placement

program = (workload
    .dispatch(DispatchPolicy.round_robin(4))
    .streams(2)
    .extend(double_buffer(scope="L1"))      # Ascend-specific
    .compile(target="ascend_npu"))
```

#### 4.8.4 Backend Interface

```cpp
class Backend {
public:
    virtual bool supports(ir::NodeKind kind) const = 0;
    virtual std::vector<std::string> supported_targets() const = 0;
    // ...
};
```

---

## 5. Compilation and Execution

### 5.1 compile()

```python
# Compile to executable program
program = schedule.compile()

# With options
program = schedule.compile(
    target="cpu_sim",  # or "ascend_npu", "amd_aie"
    enable_profiling=True,
    optimization_level=2
)
```

### 5.2 Program Class

```python
class Program:
    def execute(self) -> None:
        """Execute the program."""
        ...

    def execute_async(self) -> None:
        """Start execution asynchronously."""
        ...

    def synchronize(self) -> None:
        """Wait for completion."""
        ...

    def is_complete(self) -> bool:
        """Check if execution is done."""
        ...

    def elapsed(self) -> float:
        """Return execution time in seconds."""
        ...

    def stats(self) -> ProgramStats:
        """Return execution statistics."""
        ...

class ProgramStats:
    num_tasks: int
    num_streams: int
    num_aicpus: int
    compile_time_ms: float
    execute_time_ms: float
```

---

## 6. Kernel Definition (JIT-Style)

PTO-RT v9 supports **JIT-style kernel definitions** that eliminate string-based task names (R7). Kernels are defined as decorated Python functions and called directly within workload contexts.

### 6.1 The @jit_kernel Decorator (RECOMMENDED)

The `@jit_kernel` decorator provides a Triton-style programming model with typed `Value` objects and `tl.*` primitives. This is the **recommended** approach for new code.

```python
from pto_wsp import jit_kernel, tl, In, Out, Tile, Scalar
from pto_wsp import F16, F32

@jit_kernel
def rmsnorm(
    x: In[Tile[32, 128, F16]],
    out: Out[Tile[32, 128, F16]],
    eps: Scalar[F32] = 1e-6
):
    """RMS normalization kernel using typed Value objects.

    No string-based references - all operations return typed Values.
    """
    # All operations return Value objects (no strings!)
    sq = tl.mul(x, x)              # sq: Value
    mean = tl.rowmean(sq)          # mean: Value
    rsqrt_val = tl.rsqrt(mean)     # rsqrt_val: Value
    tl.store(out, tl.mul(x, rsqrt_val))
```

**Tile Language Primitives (`tl.*`):**

| Category | Primitives |
|----------|------------|
| **Memory** | `tl.load(src)`, `tl.store(dst, src)`, `tl.alloc(shape, dtype)` |
| **Arithmetic** | `tl.add(a, b)`, `tl.sub(a, b)`, `tl.mul(a, b)`, `tl.div(a, b)` |
| **Math** | `tl.exp(x)`, `tl.log(x)`, `tl.sqrt(x)`, `tl.rsqrt(x)` |
| **Reductions** | `tl.sum(x, axis)`, `tl.max(x, axis)`, `tl.rowmean(x)`, `tl.rowmax(x)` |
| **Matrix** | `tl.matmul(a, b)`, `tl.transpose(x)` |
| **Control** | `tl.where(cond, a, b)`, `tl.broadcast(x, shape)` |

**Key Features:**
- **Typed Values**: All `tl.*` operations return `Value` objects with type information
- **No String References**: Unlike the legacy NPU builder, no string-based tile names
- **Compile to Backend**: `kernel.compile(target="ascend_npu")` generates backend code

### 6.2 The @kernel Decorator (Builder-Style)

The `@kernel` decorator is for simpler workload integration without explicit tile operations:

```python
from pto_wsp import kernel, In, Out, Tensor

@kernel
def flash_attn(
    Q: In[Tensor],        # Input tensor
    K: In[Tensor],
    V: In[Tensor],
    O: Out[Tensor],       # Output tensor
):
    """Flash attention kernel.

    The body can be:
    1. Empty (external implementation)
    2. Python implementation (for CPU sim)
    """
    pass  # Implementation provided externally or via CPU sim
```

**Type Annotations:**

| Annotation | Meaning |
|------------|---------|
| `In[Tensor]` | Input tensor (read-only) |
| `Out[Tensor]` | Output tensor (write-only) |
| `InOut[Tensor]` | Input/output tensor (read-write) |
| `Constexpr[T]` | Compile-time constant (specialization key) |
| `AxisVar` | Iteration axis variable (from outer `parallel_for`) |

### 6.2 Using Kernels in Workloads (Direct Call)

Instead of `task("kernel_name", ...)`, call kernels directly:

```python
from pto_wsp import workload, P, DenseDyn, Dense

# Define axes
batch = DenseDyn(batch_size)
heads = Dense[8]

# Use kernel directly in workload
@workload
def attention(batch, heads):
    for b, h in P(batch, heads):
        flash_attn[b, h](Q=Q[b,h], K=K[b], V=V[b], O=O[b,h])
```

**Note**: The subscript syntax `kernel[b, h](args)` binds axis values to the kernel invocation.

### 6.3 Kernel Reference (KernelRef)

The `@kernel` decorator returns a `KernelRef` object:

```python
class KernelRef:
    """Reference to a registered kernel."""

    # Identity
    symbol: str                    # Internal name for linking
    module: Optional[Module]       # Parent module (if specified)
    signature: KernelSignature     # Inferred from type annotations

    # Methods
    def at(self, **axes: AxisVar) -> BoundKernel:
        """Bind iteration axes (TileLang-style)."""
        ...

    def __getitem__(self, axes: tuple[AxisVar, ...]) -> BoundKernel:
        """Bind iteration axes (Triton-style subscript)."""
        ...

    def __call__(self, **bindings: Tensor) -> None:
        """Emit task to current workload builder (requires axis binding first)."""
        ...

    def lower(self, target: str = "cpu_sim") -> NPUFunction:
        """Lower to NPU function IR."""
        ...
```

### 6.4 Compilation Model

| Mode | Triggered | Use Case |
|------|-----------|----------|
| **AOT (default)** | `workload.compile(target=...)` | Deterministic builds |
| **Runtime JIT** | First `program.execute()` | Interactive development |
| **Cached** | Specialization cache hit | Both modes |

Specialization key: `(target, dtypes, layouts, constexpr_values)`

```python
# AOT compilation (default)
program = attention.compile(target="ascend_npu")

# Force eager compilation of specific kernel
flash_attn.compile(target="ascend_npu", BLOCK_M=128, BLOCK_N=64)
```

### 6.5 JIT Kernel Compilation

`@jit_kernel` functions can be compiled to backend-specific code:

```python
from pto_wsp import jit_kernel, tl

@jit_kernel
def my_kernel(x: In[Tile[M, N, F16]], out: Out[Tile[M, N, F16]]):
    tl.store(out, tl.mul(x, x))

# Compile to specific target
compiled = my_kernel.compile(target="ascend_npu")

# Access generated IR
print(my_kernel.ir.dump())  # View kernel IR

# Generated code
print(compiled.code)  # Backend-specific code
```

### 6.6 Legacy: External Kernel (DEPRECATED)

> **Note**: This API is deprecated. Use `@jit_kernel` for new code.

For backward compatibility with external kernels:

```python
from pto_wsp import ExternalKernel

# Legacy external kernel registration
flash_attn_v2 = ExternalKernel(
    name="flash_attn_v2",
    impl_path="kernels/flash_attn_v2.cpp",
    inputs=["Q", "K", "V"],
    outputs=["O"],
)
```

---

## 7. Event API

Events are unbuffered channels (rendezvous semantics).

```python
from pto_wsp import Event, record, synchronize, query

# Create event
event = Event("sync")

# Signal completion (blocks until receiver ready)
record(event)

# Wait for completion (blocks until sender ready)
synchronize(event)

# Non-blocking check
if query(event):
    print("Event signaled")
```

---

## 8. Complete Examples

### 8.1 Attention with Variable Lengths

```python
from pto_wsp import workload, P, kernel, In, Out, Tensor, DenseDyn, Dense
from pto_wsp import DispatchPolicy, TimingPolicy

# Data
batch_size = 4
num_heads = 8
seq_lens = [512, 2048, 8192, 32768]

# Axes
batch = DenseDyn(batch_size)
heads = Dense[8]

# Define kernel
@kernel
def attn_kernel(Q: In[Tensor], K: In[Tensor], V: In[Tensor], O: Out[Tensor]):
    ...

# Define workload (@workload + P namespace)
@workload
def attention(batch, heads):
    for b, h in P(batch, heads):
        attn_kernel[b, h](Q=Q[b,h], K=K[b], V=V[b], O=O[b,h])

# Schedule (combinator style)
program = (attention(batch, heads)
    .dispatch(DispatchPolicy.affinity(lambda t: t.get("batch")))
    .streams(2)
    .stream_by(lambda t: t.get("head") % 2)
    .timing(TimingPolicy.immediate)
    .compile())

# Execute
program.execute()
program.synchronize()
```

### 8.2 MoE with Sparse Routing

```python
from pto_wsp import workload, P, kernel, In, Out, Tensor, DenseDyn, Sparse
from pto_wsp import DispatchPolicy

# Routing (CSR format)
indptr = [0, 2, 5, 7, 10]
indices = [1, 3, 0, 2, 4, 1, 5, 0, 3, 7]
routing = Sparse(4, indptr, indices)

# Axes
batch = DenseDyn(batch_size)

# Define kernel
@kernel
def expert(tokens: In[Tensor], weights: In[Tensor], out: Out[Tensor]):
    ...

# Define workload (@workload + P.sel for sparse)
@workload
def moe(batch, routing):
    for b in P(batch):
        for e in P.sel(routing[b]):           # Only selected experts
            expert[b, e](tokens=tokens[b], weights=weights[e], out=out[b,e])

# Schedule with work stealing (combinator style)
program = (moe(batch, routing)
    .dispatch(DispatchPolicy.work_steal())
    .streams(8)
    .stream_by(lambda t: t.get("expert"))     # expert → stream
    .compile())

program.execute()
```

### 8.3 Megakernel Pipeline

```python
from pto_wsp import workload, P, kernel, In, Out, Tensor, DenseDyn
from pto_wsp import Channel, send, consume, connect, process
from pto_wsp import DispatchPolicy

# Define kernels
@kernel
def load(input: In[Tensor], buf: Out[Tensor]): ...

@kernel
def compute(buf: In[Tensor], result: Out[Tensor]): ...

@kernel
def store(result: In[Tensor], output: Out[Tensor]): ...

# Channels (depth=2 for double buffering)
l2c = Channel("load_to_compute", depth=2)
c2s = Channel("compute_to_store", depth=2)

# Define pipeline workload using P.pipe() and CSP
@workload
def pipeline(num_tiles):
    tiles = DenseDyn(num_tiles)

    # Loader process (sequential tile loading)
    with P.pipe():
        for i in P.seq(tiles):
            send(l2c, load[i](input=input[i], buf=tile_buf))

    # Computer process (consumes from load, produces to store)
    with consume(l2c) as t:
        send(c2s, compute[t](buf=tile_buf, result=result_buf))

    # Storer process (consumes from compute)
    with consume(c2s) as r:
        store[r](result=result_buf, output=output)

# Schedule (combinator style)
program = (pipeline(num_tiles)
    .dispatch(DispatchPolicy.round_robin(1))    # Single AICPU
    .compile())

program.execute()
program.synchronize()
```

### 8.4 Spatial Attention (AIE)

```python
from pto_wsp import workload, P, kernel, In, Out, Tensor, Dense
from pto_wsp import Layout, Shard, Replicate

# Axes
batch = Dense[4]
heads = Dense[4]

# Define kernel with explicit layout annotations
@kernel
def attn_kernel(
    Q: In[Tensor[F16, [N, D], TensorLayout(TensorShard(0), TensorReplicate())]],     # sharded on batch
    K: In[Tensor[F16, [N, D], TensorLayout(TensorReplicate(), TensorShard(1))]],     # sharded on heads
    V: In[Tensor[F16, [N, D], TensorLayout(TensorReplicate(), TensorShard(1))]],     # sharded on heads
    O: Out[Tensor[F16, [N, D], TensorLayout(TensorShard(0), TensorShard(1))]]        # sharded on both
): ...

# Define workload (@workload + P namespace)
@workload
def spatial_attention(batch, heads):
    for b, h in P(batch, heads):
        attn_kernel[b, h](Q=Q[b,h], K=K[b], V=V[b], O=O[b,h])

# Spatial schedule for 4x4 AIE array (combinator style)
program = (spatial_attention(batch, heads)
    .spatial_map(grid=(4, 4))
    .compile(target="amd_aie"))

program.execute()
```

---

## 9. Error Handling

```python
from pto_wsp import (
    RuntimeError,
    CompileError,
    ExecutionError,
    ChannelError,
    ChannelClosed,
)

try:
    program = schedule.compile()
    program.execute()
except CompileError as e:
    print(f"Compilation failed: {e}")
except ExecutionError as e:
    print(f"Execution failed: {e}")
except ChannelClosed as e:
    print(f"Channel closed: {e}")
```

---

## 10. Summary

### 10.1 Primitives

| Category | Primitives |
|----------|------------|
| **Axes** | `Dense[N]`, `DenseDyn`, `Ragged`, `Sparse` |
| **Workload Definition** | `@workload` decorator with `P` namespace |
| **Loop Constructors** | `P()`, `P.seq()`, `P.sel()`, `P.pipe()`, `P.when()` |
| **Kernel Definition** | `@kernel` decorator with `In`, `Out`, `InOut`, `Constexpr` |
| **Composition** | `combine`, `sequential` |
| **CSP** | `Channel`, `send`, `consume`, `connect`, `replicate` |
| **Schedule** | `dispatch`, `streams`, `task_graph`, `stream_by`, `timing` (combinator methods) |
| **Spatial** | `spatial_map`, `Shard`, `Replicate` (Layout as type, not schedule) |
| **Execution** | `compile`, `execute`, `synchronize` |

### 10.2 Key Properties

1. **Concise Syntax**: `@workload` decorator with `P` namespace for parallel/sequential loops (TileLang-inspired)
2. **Direct Kernel Calls**: `kernel[axes](args)` instead of string-based `task("name", ...)`
3. **Combinator Schedule**: `workload.dispatch(...).streams(...)` for type-safe chaining
4. **Type-safe**: Dependency types (Independent, Sequential, etc.) inferred from structure
5. **Separable**: Workload vs Schedule separation enables reuse
6. **Multi-backend**: Same workload targets CPU sim, NPU, AIE
7. **Human-in-the-loop**: Programmer controls dispatch and issue policies
8. **Backend Applicability**: Compile-time validation of schedule primitives per target

---

*Version: 9.3*
*Last Updated: 2026-01-25*

Changes in 9.3:
- Concise workload syntax: `@workload` + `P` namespace (TileLang-inspired)
- Direct kernel calls: `kernel[axes](args)` instead of string-based `task()`
- CSP unified under workload primitives (Section 3.7)
- Backend applicability mechanism for schedule validation (Section 4.8)
- Layout as type refinement, not schedule primitive (Section 2.3)
- Removed deprecated: context manager style, lambda syntax, string-based task()
