# PTO Workload-Schedule Programming (PTO-WSP) v9: Design Analysis

## 1. Executive Summary

This document describes the design rationale for PTO Workload-Schedule Programming (PTO-WSP) v9. The v9 release extends v8's typed workload expression model with:

1. **Python frontend** - Declarative workload definition with concise `@workload` + `P` namespace syntax
2. **C++ IR layer** - Explicit intermediate representation for multi-backend targeting
3. **Multi-backend support** - CPU simulation, Ascend NPU, and AMD AIE (spatial)
4. **JIT kernel definition** - `@kernel` decorator enabling direct kernel calls in workloads
5. **Unified type system** - Layout as type refinement (`Tensor[DType, Shape, Layout]`)
6. **Task graph execution** - pto-isa-lh compatible DAG execution via `task_graph()`
7. **Backend applicability** - Compile-time validation of schedule primitives per target

The design preserves v8's core strengths (typed workloads, CSP pipelines, schedule separation) while adding concise syntax and infrastructure for production deployment.

---

## 2. Background and Motivation

### 2.1 v8 Achievements

The v8 runtime extension established:

| Achievement | Description |
|-------------|-------------|
| **Typed Workloads** | `Workload[Axes, Task, Deps]` - workloads as types with compile-time guarantees |
| **Data-Parallel** | `parallel_for`, `for_each`, `select`, `cond` - declarative iteration |
| **Pipeline-Parallel** | CSP with `Channel`, `Process`, `consume` - structured concurrency |
| **Schedule Separation** | `dispatch_by`, `stream_by`, `timing` - programmer control |
| **Unified Sync** | `Event = Channel<Signal, 0>` - rendezvous semantics |

### 2.2 v8 Limitations

From research reports 13-17, we identified key gaps:

| Limitation | Impact | Solution in v9 |
|------------|--------|----------------|
| C++ only | Hard to integrate with ML frameworks | Python frontend |
| Implicit IR | Can't serialize, debug, or target multiple backends | Explicit C++ IR |
| No CPU sim | Slow iteration during development | CPU simulation backend |
| No NPU code gen | Can't deploy to Ascend | NPU backend |
| No spatial support | Can't target AMD AIE / FPGA | Spatial schedule primitives |

### 2.3 Reference Analysis Summary

From our Phase 1 research:

| Reference | Key Contribution to v9 |
|-----------|----------------------|
| **pto-isa-lh** | Python builder pattern, task graph runtime, .pto assembly format |
| **pto-isa-wc** | AICPU-AICore handshake, Ascend C code generation |
| **allo** | Schedule API (split/reorder/fuse), MLIR-based IR, AIE backend |
| **dato** | Stream/Layout types, Virtual Mapping Graph, linear type checking |

---

## 3. Design Principles

### 3.1 Preserve v8 Semantics

v9 must express everything v8 could express. The typed workload model is the foundation:

```
v8 Expression → v9 IR → Backend Code
```

Type safety is preserved: `parallel_for` always produces `Independent` dependencies; `for_each` always produces `Sequential`. This is enforced at IR construction time.

### 3.2 Separation of Concerns

```
┌─────────────────┐
│  Python DSL     │ ← User-facing, ergonomic
├─────────────────┤
│  C++ IR Layer   │ ← Backend-agnostic, serializable
├─────────────────┤
│  Backend        │ ← Target-specific code generation
└─────────────────┘
```

Each layer has clear responsibilities:
- **Python DSL**: User ergonomics, type checking, builder pattern
- **C++ IR**: Canonical representation, optimization, serialization
- **Backend**: Target-specific lowering, code generation

### 3.3 Concise + Declarative Syntax

From v8 design analysis and TileLang research:

> Declarative primitives (vs imperative loops) enable:
> - Parallel task enumeration
> - JIT analysis and optimization
> - Dependency inference from structure

v9 adopts **concise symbolic loop syntax** inspired by TileLang:

```python
from pto_wsp import workload, P, kernel

@kernel
def attn(Q, K, V, O): ...

@workload
def attention(batch, heads):
    for b, h in P(batch, heads):    # P = Parallel grid
        attn[b, h](Q[b,h], K[b], V[b], O[b,h])
```

**Key design decisions:**
- **`@workload` decorator** - Concise, returns `Workload` directly (not `with workload()...w.finish()`)
- **`P` namespace** - Single-letter like TileLang's `T`, contains `P()`, `P.seq()`, `P.sel()`, `P.pipe()`
- **Direct kernel calls** - `kernel[axes](args...)` instead of string-based `task("name", ...)`
- **Explicit parallelism** - `P()` = parallel, `P.seq()` = sequential (explicit semantics)

### 3.4 Human-in-the-Loop Scheduling

Programmers control execution through combinator-style Schedule:

```python
# Combinator style: each method returns new schedule, enabling type-safe chaining
program = (workload
    .dispatch(DispatchPolicy.affinity(lambda t: t.batch))  # Task → AICPU
    .streams(2)                        # Concurrent streams
    .stream_by(lambda t: t.head % 2)   # Task → Stream
    .timing(TimingPolicy.immediate)    # When to issue
    .compile())
```

The runtime respects these choices rather than making automatic decisions.

---

## 4. Architecture Overview

### 4.1 Full Stack

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Python Frontend                                    │
│                                                                              │
│  @workload                            # Combinator-style schedule            │
│  def attention(batch, heads):        program = (attention(batch, heads)     │
│      for b, h in P(batch, heads):       .dispatch(affinity(...))            │
│          kernel[b, h](...)              .spatial_map(grid=(4,4))            │
│                                          .compile())                         │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                              to_ir() │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              C++ IR Layer                                    │
│                                                                              │
│  WorkloadIR               ScheduleIR              AssemblyFormat            │
│  ├─ ParallelForNode       ├─ DispatchNode         ├─ print()               │
│  ├─ ForEachNode           ├─ StreamNode           └─ parse()               │
│  ├─ SelectNode            ├─ SpatialMapNode                                 │
│  ├─ TaskNode              └─ LayoutNode                                     │
│  └─ ChannelNode                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    │ lower()       │ lower()       │ lower()
                    ▼               ▼               ▼
┌─────────────────────┐ ┌─────────────────────┐ ┌─────────────────────┐
│   CPU Simulation    │ │   Ascend NPU        │ │   AMD AIE           │
│                     │ │                     │ │                     │
│ - TaskGraphRuntime  │ │ - AICPU codegen     │ │ - MLIR lowering     │
│ - ThreadPool        │ │ - AICore wrapper    │ │ - Tile mapping      │
│ - Pure C execution  │ │ - Handshake proto   │ │ - Stream routing    │
└─────────────────────┘ └─────────────────────┘ └─────────────────────┘
```

### 4.2 Data Flow

1. **Definition**: User writes workload and schedule in Python
2. **IR Construction**: Python DSL calls `to_ir()` to create C++ IR
3. **Serialization** (optional): IR can be saved as `.pto` assembly
4. **Compilation**: `schedule.compile()` invokes backend-specific lowering
5. **Execution**: `program.execute()` runs the generated code

### 4.3 Hierarchical Workload-Schedule Model (New in v9)

v9 introduces a two-level hierarchical model to support both outer orchestration and inner kernel scheduling.

#### 4.3.1 Two-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  Level 0: Outer Schedule (CPU/AICPU Orchestration)                          │
│                                                                              │
│    Workload IR (task graph) → Executors (threads, AICPU, device streams)    │
│                                                                              │
│    Primitives: dispatch(), streams(), stream_by(), timing()                 │
│    Schedule: Task routing, stream assignment, issue timing                  │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │  Level 1: Inner Schedule (NPU/InCore Kernel)                            │ │
│  │                                                                          │ │
│  │    NPUFunction IR → Memory hierarchy + pipelines (GM/L2/UB/L1/L0)       │ │
│  │                                                                          │ │
│  │    Primitives: tile_policy(), double_buffer(), pipeline()               │ │
│  │    Schedule: Tile sizing, buffer allocation, DMA/compute overlap        │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Rationale**: This matches pto-isa-lh's "orchestration builds tasks; InCore functions define kernels" and the Ascend A2/A3 model (vector vs cube cores + MTE/compute pipe overlap).

#### 4.3.2 Level Depth Decision

**Decision**: 2 levels as the first-class IR contract

| Alternative | Pros | Cons | Decision |
|-------------|------|------|----------|
| 2 levels | Clean split: graph vs kernel | May need extension later | **CHOSEN** |
| 3 levels | Device placement as separate level | Complexity, unclear boundaries | Rejected |
| Arbitrary | Maximum flexibility | Unbounded complexity | Rejected |

**Escape Hatch**: Additional levels can be added without IR redesign:
- Multi-device placement → outer schedule refinement (`spatial_map`, `dispatch_by(device_id)`)
- Microcode scheduling → backend lowering detail (Ascend flags/barriers, AIE ping-pong DMA)

#### 4.3.3 Outer-Inner Linkage

**Design**: Tasks reference kernels via stable `KernelHandle`, not strings.

```python
# Outer schedule can use inner kernel traits
program = (workload
    .dispatch(DispatchPolicy.by(lambda t: t.kernel.is_cube))  # Route by exec unit
    .stream_by(lambda t: t.kernel.name_hash)  # Balance by kernel
    .compile())
```

**KernelDef** stores:
- Signature (inputs/outputs/resources)
- Traits (`ExecUnit`, SRAM budget, estimated cycles)
- Inner IR (`NPUFunction`) + schedule hints
- Per-backend artifacts (compiled binary, cost model)

#### 4.3.4 Inner Schedule Specification

**Design**: Functional IR (portable) + Schedule annotations (backend-specific)

| Layer | Contents | Portability |
|-------|----------|-------------|
| Functional IR | `tile/memref`, `load/store`, `compute`, `loops/if` | All backends |
| Schedule Annotations | `tile_policy`, `double_buffer`, `pipeline` | Backend-meaningful |

**Double Buffering** compiles to explicit async + sync:
- Directive: `.double_buffer("q", depth=2)`
- Lowering: ping/pong buffers + `load_async(tag)` + `wait(tag)` + swap

**Tile Size Selection**:
- Prefer kernel variants (`attn_32x128`, `attn_64x256`) when SRAM constraints are tight
- Allow symbolic shapes only if backend can specialize at compile time

#### 4.3.5 Execution Unit Classification

**Design**: 3-state `ExecUnit` trait (not boolean)

```python
class ExecUnit(Enum):
    VectorOnly = "vector"   # Element-wise, reductions
    CubeOnly = "cube"       # Matrix multiply
    Either = "either"       # Can run on either (rare)
```

**Inference**: Operations like `.matmul(use_cube=True)` automatically set `ExecUnit.CubeOnly`

**Backend Mapping**:
- CPU sim: Ignored (all on CPU)
- Ascend: Routes to AIV (vector) or AIC (cube) queue
- AMD AIE: Maps to different kernel implementations

#### 4.3.6 Backend Mapping Rules

| Level | CPU Sim | Ascend NPU | AMD AIE |
|-------|---------|------------|---------|
| **Outer** | Threads | AICPU → AICores | Tiles/columns |
| Executors | Thread pool | Hardware streams | Spatial array |
| Dispatch | Thread selection | Stream + core group | Column assignment |
| **Inner** | NumPy/Torch ops | PTO-ISA → AICore | AIE kernel code |
| Schedule hints | Cost model only | MTE/compute overlap | Software pipeline |
| Double buffering | Ignored | Ping/pong + async DMA | DMA channels |

---

## 5. Type System (Preserved from v8)

### 5.1 Axis Types

```cpp
namespace pto::wsp {
    template<int64_t N> struct Dense;   // Static size
    struct DenseDyn;                     // Dynamic size
    struct Ragged;                       // Variable lengths
    struct Sparse;                       // CSR format
}
```

### 5.2 Workload Type

```cpp
template<typename Axes, typename Task, typename Deps>
class Workload;

// Dependency types (unchanged from v8)
struct Independent;  // parallel_for produces this
struct Sequential;   // for_each produces this
struct ChannelDep;   // CSP channel dependency
struct Combined;     // combine() produces this
struct None;         // single task
```

### 5.3 Type Safety Guarantees

| Loop Constructor | Produces | Guarantee |
|------------------|----------|-----------|
| `for ... in P(axes)` | `Independent` | All tasks can run in parallel |
| `for ... in P.seq(axis)` | `Sequential` | Task[i] depends on Task[i-1] |
| `for ... in P.sel(sparse)` | `Independent` | Selected tasks independent |
| `combine(w1, w2)` | `Combined` | Schedule determines order |
| `sequential(w1, w2)` | `Sequential` | w2 waits for w1 |

---

## 6. New in v9

### 6.1 Python Frontend

**Design Choice**: Concise Workloads + Combinator-Style Schedule

From v9.3 refinement and TileLang-inspired syntax research:
- `@workload` decorator with `P` namespace is concise (~3 lines vs ~7 for context managers)
- Direct kernel calls `kernel[axes](args)` instead of string-based `task("name", ...)`
- Combinator-style `workload.dispatch(...).streams(...)` provides type-safe chaining

```python
from pto_wsp import workload, P, kernel, In, Out, Tensor, Dense, DenseDyn
from pto_wsp import DispatchPolicy, TimingPolicy

# Define kernel
@kernel
def attn_kernel(Q: In[Tensor], K: In[Tensor], V: In[Tensor], O: Out[Tensor]): ...

# Concise workload using @workload + P namespace
batch = DenseDyn(batch_size)
heads = Dense[8]

@workload
def attention(batch, heads):
    for b, h in P(batch, heads):
        attn_kernel[b, h](Q=Q[b,h], K=K[b], V=V[b], O=O[b,h])

# Combinator style for schedule (type-safe chaining)
program = (attention(batch, heads)
    .dispatch(DispatchPolicy.affinity(lambda t: t.get("batch")))
    .streams(2)
    .timing(TimingPolicy.immediate)
    .compile())

program.execute()
```

### 6.2 C++ IR

**Design Choice**: Custom C++ IR (not MLIR)

Rationale from Report 15 (allo) and Report 17 (gap analysis):
- MLIR has steep learning curve and dependency overhead
- Our IR is simpler than general compiler IR
- Can migrate to MLIR later if needed

IR node hierarchy:

```cpp
namespace pto::wsp::ir {

// Base class
struct IRNode {
    virtual ~IRNode() = default;
    virtual void print(std::ostream& os) const = 0;
    virtual std::string kind() const = 0;
};

// Workload nodes
struct AxisNode : IRNode { /* Dense, DenseDyn, Ragged, Sparse */ };
struct TaskNode : IRNode { /* kernel, params, resources */ };
struct ParallelForNode : IRNode { /* axis, body */ };
struct ForEachNode : IRNode { /* axis, body */ };
struct SelectNode : IRNode { /* sparse, body */ };
struct CombineNode : IRNode { /* workloads */ };
struct SequentialNode : IRNode { /* workloads */ };

// CSP nodes
struct ChannelNode : IRNode { /* name, type, capacity */ };
struct ProcessNode : IRNode { /* name, consumes, produces, body */ };
struct SendNode : IRNode { /* channel, value */ };
struct ConsumeNode : IRNode { /* channel, body */ };

// Schedule nodes
struct DispatchNode : IRNode { /* policy */ };
struct StreamNode : IRNode { /* count, key_fn */ };
struct TimingNode : IRNode { /* policy */ };
struct SpatialMapNode : IRNode { /* grid */ };    // NEW in v9
struct LayoutNode : IRNode { /* tensor, layout */ }; // NEW in v9

}
```

### 6.3 Assembly Format

**Design Choice**: Custom .pto format (aligned with pto-isa-lh)

```
// PTO Module: attention_v9
// Version: 9.0
// Target: cpu_sim | ascend_npu | amd_aie

// === Type Definitions ===
!dense8 = Dense[8]
!dyndense = DenseDyn
!channel = Channel[Task, 2]

// === Workload Definition ===
@workload attention(%batch: !dyndense, %heads: !dense8) {
  parallel_for %b in %batch {
    parallel_for %h in %heads {
      %t = task @attn_kernel(%b, %h)
          resources(%Q[%b][%h], %K[%b], %V[%b], %O[%b][%h])
      yield %t
    }
  }
}

// === Schedule Definition ===
@schedule attention_sched for @attention {
  dispatch = affinity(%batch)
  streams = 2
  stream_by = %head mod 2
  timing = immediate
}

// === CSP Pipeline ===
@pipeline megakernel {
  channel %l2c : !channel
  channel %c2s : !channel

  process @loader produces(%l2c) {
    for_each %i in DenseDyn(%num_tiles) {
      %t = task @load_kernel(%i) resources(%input[%i], %buf)
      send %l2c, %t
    }
  }

  process @computer consumes(%l2c) produces(%c2s) {
    consume %l2c as %tile {
      %r = task @compute_kernel(%tile) resources(%buf, %out)
      send %c2s, %r
    }
  }

  process @storer consumes(%c2s) {
    consume %c2s as %result {
      task @store_kernel(%result) resources(%out, %final)
    }
  }
}
```

### 6.4 Spatial Schedule Primitives

**Design Choice**: New primitives inspired by dato paper

From Report 16 (dato), we learned that spatial architectures need:
1. **Spatial mapping**: Assign tasks to physical tiles
2. **Layout annotations**: Specify data distribution

```python
# Spatial mapping: workload → tile grid
schedule.spatial_map(attention, grid=(4, 4))

# Data layout: tensor → distribution
schedule.layout(Q, Shard(dim=0), Replicate())  # Q sharded on batch
schedule.layout(K, Replicate(), Shard(dim=1))  # K sharded on seq
schedule.layout(V, Replicate(), Shard(dim=1))  # V sharded on seq
```

These primitives integrate with existing Schedule API and are lowered to target-specific code by the AIE backend.

---

## 7. Backend Architecture

### 7.1 Backend Interface

```cpp
namespace pto::wsp::backend {

class Backend {
public:
    virtual ~Backend() = default;

    // Compile IR to executable
    virtual Program compile(const ir::WorkloadIR& workload,
                           const ir::ScheduleIR& schedule) = 0;

    // Query backend capabilities
    virtual bool supports(ir::NodeKind kind) const = 0;
    virtual std::string name() const = 0;
};

class Program {
public:
    virtual ~Program() = default;
    virtual void execute() = 0;
    virtual void synchronize() = 0;
    virtual bool is_complete() const = 0;
};

}
```

### 7.2 CPU Simulation Backend

From Report 13 (pto-isa-lh), we adopt the task graph runtime:

```cpp
class CPUSimBackend : public Backend {
    // Task graph with fanin/fanout edges
    struct TaskNode {
        int32_t task_id;
        int32_t fanin;           // Dependencies remaining
        std::vector<int32_t> fanout; // Downstream tasks
        std::function<void()> kernel;
    };

    // Thread pool for execution
    ThreadPool pool;

    // Dependency tracking
    TensorMap producer_map;  // (tensor, region) → task_id
};
```

Target: ~5000+ tasks/ms throughput (matching pto-isa-lh)

### 7.3 Ascend NPU Backend

From Report 14 (pto-isa-wc), we adopt:

```cpp
class AscendNPUBackend : public Backend {
    // Generate AICPU orchestration code
    std::string generate_aicpu_code(const ir::ScheduleIR& sched);

    // Generate AICore kernel wrappers
    std::string generate_aicore_wrapper(const ir::TaskNode& task);

    // AICPU-AICore handshake
    void emit_kernel_wait(int kernel_id);
    void emit_signal_complete();
};
```

Key patterns:
- Multi-threaded graph executor
- Handshake protocol for coordination
- Ascend C code generation

### 7.4 AMD AIE Backend (P1)

From Report 15 (allo) and Report 16 (dato):

```cpp
class AMDAIEBackend : public Backend {
    // Lower to MLIR AIE dialect
    mlir::ModuleOp lower_to_mlir(const ir::WorkloadIR& workload,
                                  const ir::ScheduleIR& schedule);

    // Spatial mapping
    void apply_spatial_map(const ir::SpatialMapNode& map);

    // Data layout
    void apply_layout(const ir::LayoutNode& layout);
};
```

This backend is P1 (should have) priority. Initial focus on CPU sim + NPU.

### 7.5 Backend Applicability Validation

**Design Choice**: Compile-time validation of schedule primitives per target

Different schedule primitives have different applicability:

| Primitive | CPU Sim | Ascend NPU | AMD AIE |
|-----------|---------|------------|---------|
| `dispatch(work_steal)` | ✓ | ✓ | ✗ |
| `streams()` | ✓ | ✓ | ✗ |
| `task_graph()` | ✓ | ✓ | ✓ |
| `spatial_map()` | ✗ | ✗ | ✓ |

The `Backend::supports(ir::NodeKind)` interface enables compile-time validation:

```python
# Error at compile time
program = (workload
    .dispatch(DispatchPolicy.work_steal())  # ⚠️ Not supported
    .spatial_map(grid=(4, 4))
    .compile(target="amd_aie"))             # Error raised here
```

Backend-specific extensions use qualified names (`pto.rt.backend.ascend.double_buffer`).

---

## 8. Migration from v8

### 8.1 API Compatibility

v9.3 introduces concise Python syntax. Migration from v8 C++ patterns:

| v8 C++ | v9.3 Python | Notes |
|--------|-------------|-------|
| `parallel_for(axis, body)` | `for x in P(axis):` | Symbolic loop with `P` namespace |
| `for_each(axis, body)` | `for x in P.seq(axis):` | Sequential via `P.seq()` |
| `task(kernel, params, res)` | `kernel[axes](args)` | Direct kernel call (no strings) |
| `Channel<T, N>` | `Channel(name, depth=N)` | Named channels |
| `workload.schedule()` | `workload.dispatch(...).streams(...)` | Combinator style |
| `schedule.compile()` | `schedule.compile()` | Unchanged |

### 8.2 Migration Path

1. **Phase 1**: Use v9.3 Python API for new workloads (`@workload` + `P` namespace)
2. **Phase 2**: Port existing v8 C++ workloads to Python
3. **Phase 3**: Deprecate direct v8 C++ API (keep for embedding)

---

## 9. Summary

### 9.1 What v9 Adds

| Feature | Status | Priority |
|---------|--------|----------|
| Python frontend (`@workload` + `P`) | P0 | Must have |
| C++ IR layer | P0 | Must have |
| JIT kernel definition (`@kernel`) | P0 | Must have |
| Unified type system (Layout types) | P0 | Must have |
| CPU simulation backend | P0 | Must have |
| Ascend NPU backend | P0 | Must have |
| `task_graph()` execution | P0 | Must have |
| Backend applicability validation | P0 | Must have |
| `spatial_map` primitive | P1 | Should have |
| AMD AIE backend | P2 | Nice to have |

### 9.2 What v9 Preserves

- Typed workload expressions (`Workload[Axes, Task, Deps]`)
- CSP pipeline-parallel model (channels, processes)
- Schedule separation (workload defines what, schedule defines how)
- Event = Channel<Signal, 0>

### 9.3 Design Decision Summary

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Workload syntax | `@workload` + `P` namespace | Concise like TileLang, explicit parallelism |
| Kernel calls | `kernel[axes](args)` | Type-safe, no string-based task names |
| Layout | Type refinement (not schedule) | Compile-time checking, cleaner separation |
| Issue mode | `streams()` or `task_graph()` | Flexibility for different dependency patterns |
| Backend check | Compile-time validation | Clear errors, no runtime surprises |

### 9.4 JIT Kernel API Design Rationale (L3, L11)

The v9 release introduces `@jit_kernel` with Triton-style `tl.*` primitives, replacing the string-based NPU function builder.

**Problem with string-based API (legacy `npu()` builder):**

```python
# LEGACY (deprecated) - string-based, error-prone
npu_func = (npu("rmsnorm")
    .tile("x", 32, 128)          # String name "x"
    .tile("sq", 32, 128)         # String name "sq"
    .mul("sq", "x", "x")         # String refs - no type checking!
    .rowmean("mean", "sq")
    .build())
```

Issues:
- No compile-time type checking on tile/memref names
- Easy to make typos in string references
- Hard to trace data flow through operations

**Solution: Typed Value objects (`@jit_kernel` + `tl.*`):**

```python
# NEW (recommended) - typed Values, no strings
@jit_kernel
def rmsnorm(x: In[Tile[32, 128, F16]], out: Out[Tile[32, 128, F16]]):
    sq = tl.mul(x, x)              # sq: Value (typed!)
    mean = tl.rowmean(sq)          # mean: Value
    tl.store(out, tl.mul(x, tl.rsqrt(mean)))
```

Benefits:
- **Type safety**: All operations return typed `Value` objects
- **IDE support**: Autocomplete, type hints, refactoring
- **Clear data flow**: Values track dependencies
- **Triton familiarity**: Familiar API for GPU programmers

### 9.5 Concurrent Mechanism Design Rationale (L12)

The v9 release extracts common concurrency patterns from pto-isa-lh and pto-isa-wc.

**Two execution models:**

| Feature | FIFO (lh-style) | Work-Stealing (wc-style) |
|---------|-----------------|--------------------------|
| Queue structure | Single shared queue | Per-worker deques |
| Task ordering | Strict FIFO | Local LIFO, steal FIFO |
| Cache locality | Lower | Higher (local tasks) |
| Load balancing | Natural | Requires stealing |
| Use case | Single-stream NPU | Multi-core CPU, multi-stream |

**Python API mapping:**

```python
# FIFO execution (pto-isa-lh compatible)
program = workload.task_graph(ready=ReadyPolicy.fifo())

# Work-stealing execution (pto-isa-wc style)
program = workload.task_graph(ready=ReadyPolicy.work_steal())
```

**C++ implementation** (in `include/pto/rt/concurrent/utilities.hpp`):
- `CompletionCounter`: Atomic counter with completion notification
- `Latch`: Single-use barrier for synchronization
- `Barrier`: Reusable multi-phase synchronization
- `BoundedQueue`: MPMC queue for work distribution
- `ThreadPool`: Fixed-size worker pool
- `DomainHandshake`: CPU↔Accelerator coordination

### 9.6 Linear Layout Design Rationale (L8)

The v9 release adds Linear Layout support based on Triton's F₂ binary matrix formalism (arXiv:2505.23819).

**Key concepts:**

- Tensor layouts represented as F₂ (binary field) matrices
- Basis vectors specify how indices map to memory addresses
- Composition rules enable layout propagation through operations
- Automatic swizzling for bank conflict avoidance

**Implementation** (in `python/pto_wsp/linear_layout.py`):

```python
from pto_wsp import LinearLayout

# Create row-major layout
layout = LinearLayout.row_major([32, 128])

# Compose with transpose
transposed = layout.transpose_dims([1, 0])

# Propagate through reshape
reshaped = propagate_reshape(layout, [32, 128], [4096])
```

**Integration with TensorLayout:**

```python
# Convert between representations
tensor_layout = linear_layout.to_tensor_layout()
linear_layout = LinearLayout.from_tensor_layout(tensor_layout)
```

---

## References

- v8 Analysis: `docs/archive/v8/analysis.md`
- v8 Spec: `docs/archive/v8/spec.md`
- Research Report 13: `docs/research/13_pto_isa_lh.md`
- Research Report 14: `docs/research/14_pto_isa_wc.md`
- Research Report 15: `docs/research/15_allo.md`
- Research Report 16: `docs/research/16_dato.md`
- Gap Analysis: `docs/research/17_v9_gap_analysis.md`
