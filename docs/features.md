# PTO Workload-Schedule Programming (PTO-WSP) framework v9: Feature Catalog

This document catalogs all features in the current version with concise explanations and links to detailed documentation or code.

---

## Overview

PTO Workload-Schedule Programming (PTO-WSP) framework enables dynamic LLM workloads on Ascend NPU and other accelerators. Features are organized in layers:

1. **Core Types** - Foundation: Tensors, Axes, DTypes
2. **Workload Definition** - @workload, @kernel, P namespace
3. **NPU Programming** - JIT kernel API with typed values
4. **Scheduling** - Dispatch, streams, task graph
5. **CSP Pipeline** - Channels and processes
6. **Type System** - Layout types, type checking
7. **C++ Backend** - IR, codegen, backends

---

## Quick Reference

| Feature | Module | Status | Links |
|---------|--------|--------|-------|
| Axis Types | `pto_wsp.types` | ✓ Done | [Details](#1-axis-types) |
| Tensor & DType | `pto_wsp.types` | ✓ Done | [Details](#2-tensor-and-dtype) |
| @workload decorator | `pto_wsp.builder` | ✓ Done | [Details](#3-workload-decorator) |
| @kernel decorator | `pto_wsp.builder` | ✓ Done | [Details](#4-kernel-decorator) |
| P namespace | `pto_wsp.p_namespace` | ✓ Done | [Details](#5-p-namespace) |
| JIT Kernel API | `pto_wsp.kernel` | ✓ Done | [Details](#6-jit-kernel-api) |
| NPU Function Builder (Legacy) | `pto_wsp.npu` | Legacy | [Details](#7-npu-function-builder-legacy) |
| Schedule Combinator API | `pto_wsp.workload` | ✓ Done | [Details](#8-schedule-combinator-api) |
| Task Graph (R9) | `pto_wsp.schedule` | ✓ Done | [Details](#9-task-graph-r9) |
| Extended Schedule Primitives | `pto_wsp.schedule` | ✓ Done | [Details](#10-extended-schedule-primitives) |
| CSP Primitives | `pto_wsp.csp` | ✓ Done | [Details](#11-csp-primitives) |
| Layout Types (R10) | `pto_wsp.types` | ✓ Done | [Details](#12-layout-types-r10) |
| Linear Layout (F₂) | `pto_wsp.linear_layout` | ✓ Done | [Details](#13-linear-layout-f2) |
| Type Checker | `pto_wsp.type_checker` | ✓ Done | [Details](#14-type-checker) |
| C++ IR | `include/pto/rt/ir/` | ✓ Done | [Details](#15-c-ir) |
| Backend Architecture | `include/pto/rt/backend/` | ✓ Done | [Details](#16-backend-architecture) |
| Concurrent Utilities | `include/pto/rt/concurrent/` | ✓ Done | [Details](#17-concurrent-utilities) |

---

## Layer 1: Core Types

### 1. Axis Types

**Purpose**: Describe iteration domain shapes for workloads.

| Type | Description | Example |
|------|-------------|---------|
| `Dense[N]` | Static compile-time size | `Dense[8]()` for 8 heads |
| `DenseDyn` | Dynamic runtime size | `DenseDyn(batch_size)` |
| `Ragged` | Variable per outer index | `Ragged(4, lengths)` for variable seqs |
| `Sparse` | CSR-format sparse axis | `Sparse(4, indptr, indices)` for MoE |

```python
from pto_wsp import Dense, DenseDyn, Ragged, Sparse

batch = DenseDyn(32)           # Dynamic batch size
heads = Dense[8]()             # Fixed 8 attention heads
seq_lens = Ragged(32, lengths) # Variable-length sequences
experts = Sparse(8, indptr, indices)  # MoE expert selection
```

**Code**: `python/pto_wsp/types.py`
**Docs**: `docs/spec.md` Section 2.1

---

### 2. Tensor and DType

**Purpose**: First-class tensor type with optional layout refinement.

```python
from pto_wsp import Tensor, DType, Location

# Basic tensor
t = Tensor(
    data=np.zeros((32, 128)),
    shape=(32, 128),
    dtype=DType.F16,
    location=Location.Global
)

# With layout (see Layout Types section)
t = Tensor(..., layout=TensorLayout.sharded(dim=0, rank=2))
```

**DTypes**: `F16`, `F32`, `BF16`, `I8`, `I16`, `I32`, `I64`
**Locations**: `Global`, `L2`, `UB`, `L1`

**Code**: `python/pto_wsp/types.py`
**Docs**: `docs/spec.md` Section 2.3

---

## Layer 2: Workload Definition

### 3. @workload Decorator

**Purpose**: Define typed workload expressions using Python syntax.

```python
from pto_wsp import workload, kernel, P, Dense, DenseDyn

@kernel
def attention(Q, K, V, O):
    pass

@workload
def multi_head_attention(batch, heads):
    for b, h in P(batch, heads):
        attention[b, h](Q, K, V, O)

# Create workload instance
w = multi_head_attention(DenseDyn(4), Dense[8]())
```

**Key features**:
- Declarative workload definition (no lambdas required)
- Type inference for workload structure
- Built-in type checking integration

**Code**: `python/pto_wsp/builder.py`
**Docs**: `docs/spec.md` Section 3

---

### 4. @kernel Decorator

**Purpose**: JIT-style kernel definitions with typed signatures.

```python
from pto_wsp import kernel, In, Out, InOut, Constexpr, Tensor

@kernel
def matmul(A: In[Tensor], B: In[Tensor], C: Out[Tensor], alpha: Constexpr[float]):
    pass

# Axis binding: kernel[batch, head]
bound = matmul[b, h]

# Call with keyword arguments
bound(A=input_a, B=input_b, C=output, alpha=1.0)
```

**Annotations**:
- `In[T]`: Read-only input
- `Out[T]`: Write-only output
- `InOut[T]`: Read-modify-write
- `Constexpr[T]`: Compile-time constant

**Code**: `python/pto_wsp/builder.py`
**Docs**: `docs/spec.md` Section 4

---

### 5. P Namespace

**Purpose**: Symbolic loop constructors for workload iteration.

```python
from pto_wsp import P

# Parallel loop - independent tasks
for b, h in P(batch, heads):
    kernel[b, h](...)

# Sequential loop - ordered dependencies
for i in P.seq(seq_len):
    scan[i](...)

# Selection - sparse iteration
for e in P.sel(expert_mask):
    expert[e](...)

# Pipeline scope for CSP
with P.pipe():
    ...

# Conditional branch
with P.when(seq_len > threshold):
    large_kernel(...)
with P.otherwise():
    small_kernel(...)
```

**Loop constructors**:
| Constructor | Dependency | Use case |
|-------------|------------|----------|
| `P(axes)` | Independent | Data parallelism |
| `P.seq(axis)` | Sequential (i→i+1) | Scan, recurrence |
| `P.sel(axis)` | Independent sparse | MoE routing |
| `P.pipe()` | Pipeline stages | Megakernels |
| `P.when(cond)` | Conditional | Tiered kernels |

**Code**: `python/pto_wsp/p_namespace.py`
**Docs**: `docs/spec.md` Section 3.2

---

## Layer 3: NPU Programming

### 6. JIT Kernel API

**Purpose**: Triton-style JIT kernel programming with typed values (recommended approach).

This is the primary API for NPU kernel programming, replacing the legacy string-based builder (L3 requirement).

```python
from pto_wsp.kernel import jit_kernel, tl, Value, DType

@jit_kernel
def rmsnorm_kernel(x, out):
    """RMSNorm using typed operations (not string-based)."""
    sq = tl.mul(x, x)              # Typed binary op
    mean = tl.rowmean(sq)          # Typed reduction
    rsqrt_val = tl.rsqrt(mean)     # Typed unary op
    result = tl.mul(x, rsqrt_val)
    tl.store(out, result)

# Trace kernel with typed Values
x = Value.tile(32, 128, DType.F16)
out = Value.tile(32, 128, DType.F16)
ir = rmsnorm_kernel(x=x, out=out)

# Compile for Ascend
compiled = rmsnorm_kernel.compile(target="ascend")
print(compiled.code)  # Generated PTO-ISA code
```

**Tile Language Primitives** (`tl.*`):

| Category | Operations |
|----------|------------|
| Binary | `tl.add`, `tl.mul`, `tl.sub`, `tl.div`, `tl.max`, `tl.min` |
| Unary | `tl.exp`, `tl.rsqrt`, `tl.neg`, `tl.abs`, `tl.tanh`, `tl.sigmoid` |
| Reduction | `tl.rowsum`, `tl.rowmax`, `tl.rowmean`, `tl.colsum` |
| Memory | `tl.load`, `tl.store` |
| MatMul | `tl.matmul` |

**Key advantage**: Uses `Value` objects with integer IDs instead of string names, enabling type checking at trace time.

**Code**: `python/pto_wsp/kernel.py`
**Docs**: `docs/design/npu-design.md`

---

### 7. NPU Function Builder (Legacy)

> **Note**: This is the legacy API. Use `@jit_kernel` with `tl.*` primitives instead (Section 6).

**Purpose**: String-based NPU function construction (deprecated in favor of JIT Kernel API).

```python
from pto_wsp import npu, DType

# Legacy: Uses string-based tile/memref names
rmsnorm = (npu("rmsnorm")
    .tile("x", 32, 128, dtype=DType.F16)
    .tile("out", 32, 128, dtype=DType.F16)
    .memref("input", DType.F16, is_input=True)
    .memref("output", DType.F16, is_output=True)
    .load("x", "input")
    .mul("sq", "x", "x")
    .rowmean("mean", "sq")
    .rsqrt("rsqrt_val", "mean")
    .rowexpandmul("out", "x", "rsqrt_val")
    .store("output", "out")
    .build())
```

**Why deprecated**: String-based references are error-prone and don't support compile-time type checking.

**Code**: `python/pto_wsp/npu.py`

---

## Layer 4: Scheduling

### 8. Schedule Combinator API

**Purpose**: Type-safe schedule composition through method chaining.

```python
program = (workload
    .dispatch(DispatchPolicy.round_robin(4))  # Task→executor
    .streams(2)                                # Concurrent streams
    .stream_by(lambda t: t.get("head") % 2)   # Stream assignment
    .timing(TimingPolicy.immediate)            # Issue timing
    .compile(target="cpu_sim"))                # Compile
```

**Available methods**:

| Method | Purpose |
|--------|---------|
| `.dispatch(policy)` | Task→executor assignment |
| `.streams(count)` | Number of concurrent streams |
| `.stream_by(fn)` | Stream assignment function |
| `.timing(policy)` | Task issue timing |
| `.spatial_map(grid)` | Tile grid mapping |
| `.task_graph(...)` | DAG-based execution (see next) |
| `.compile(target)` | Compile to executable |

**Code**: `python/pto_wsp/workload.py`
**Docs**: `docs/spec.md` Section 5

---

### 9. Task Graph (R9)

**Purpose**: DAG-based execution as alternative to streams, matching pto-isa-lh capabilities.

```python
from pto_wsp import (
    Deps, TaskWindow, WindowMode, Pools,
    ReadyPolicy, StartPolicy, TracePolicy
)

# Basic task graph (pto-isa-lh compatible)
program = (workload
    .dispatch(DispatchPolicy.round_robin(4))
    .task_graph()
    .compile())

# Full configuration
program = (workload
    .dispatch(DispatchPolicy.work_steal())
    .task_graph(
        deps=Deps.infer_tensor_map_exact(),
        window=TaskWindow(8192, "tasks", WindowMode.STALL),
        pools=Pools.by_exec_unit(),
        ready=ReadyPolicy.work_steal(),
        start=StartPolicy.threshold(100),
        trace=TracePolicy.cycles()
    )
    .compile())
```

**Configuration options**:

| Parameter | Options | Default |
|-----------|---------|---------|
| `deps` | `infer_tensor_map_exact()`, `infer_bytes_overlap()`, `explicit()`, `hybrid()` | tensor_map_exact |
| `window` | `TaskWindow(size, unit, mode)` | 8192 tasks, stall |
| `pools` | `single()`, `by_exec_unit()`, `custom()` | single |
| `ready` | `fifo()`, `work_steal()`, `priority(fn)` | fifo |
| `start` | `after_orchestration()`, `threshold(n)` | after_orchestration |
| `trace` | `none()`, `cycles(cost_fn)` | none |

**FIFO vs Work-Steal** (L9):
- **FIFO**: Single shared queue; tasks dequeued in order. Simple, predictable, good for homogeneous tasks.
- **Work-Steal**: Per-worker deques; workers steal from others when idle. Better load balancing for heterogeneous tasks.

**Code**: `python/pto_wsp/schedule.py`
**Docs**: `docs/research/task_graph_design_research.md`

---

### 10. Extended Schedule Primitives

**Purpose**: Advanced dispatch and issue control for large workloads.

```python
# Multi-level dispatch based on workload size
workload.dispatch_threshold(
    thresholds=[256, 1024, 4096],
    policies={
        256: DispatchPolicy.round_robin(1),
        1024: DispatchPolicy.affinity(lambda t: t.batch),
        4096: DispatchPolicy.work_steal(),
    }
)

# In-flight task control (double/triple buffering)
workload.pipeline_depth(2, scope="per_stream")

# Task metadata window management
workload.task_window(8192, unit="tasks", mode="stall")

# Batched dependency resolution
workload.batch_deps(64, range_compression=True)
```

**Note**: Per L10 requirement, consider using only `dispatch_threshold` for simplicity.

**Code**: `python/pto_wsp/schedule.py`
**Docs**: `docs/research/extended_primitives_research.md`

---

## Layer 5: CSP Pipeline

### 11. CSP Primitives

**Purpose**: Pipeline-parallel programming with channels and processes.

```python
from pto_wsp import Channel, Event, process, send, consume, connect, replicate
from pto_wsp import record, synchronize, query

# Create channels
ch = Channel("data", depth=2)  # Buffered channel
event = Event("sync")          # Unbuffered (rendezvous)

# Define processes
loader = (process("loader")
    .produces(ch)
    .body(for_each(tiles, lambda i: send(ch, load[i](...)))))

computer = (process("computer")
    .consumes(ch_in)
    .produces(ch_out)
    .body(consume(ch_in, lambda t: send(ch_out, compute[t](...)))))

# Connect pipeline
pipeline = connect([loader, computer], [ch])

# Replicate for parallelism
workers = replicate(worker, 4)

# Event synchronization
record(event)      # Signal completion
synchronize(event) # Wait for signal
if query(event):   # Non-blocking check
    ...
```

**Primitives**:

| Primitive | Purpose |
|-----------|---------|
| `Channel(name, depth)` | Bounded FIFO queue |
| `Event(name)` | Unbuffered sync (depth=0) |
| `process(name)` | Process builder |
| `send(ch, val)` | Send to channel |
| `consume(ch, fn)` | Receive and process |
| `connect(procs, chs)` | Wire pipeline |
| `replicate(proc, n)` | Create n instances |
| `record(event)` | Signal event |
| `synchronize(event)` | Wait for event |
| `query(event)` | Non-blocking check |

**Code**: `python/pto_wsp/csp.py`
**Docs**: `docs/spec.md` Section 6

---

## Layer 6: Type System

### 12. Layout Types (R10)

**Purpose**: Layout as tensor type refinement (Dato-style), not schedule primitive.

```python
from pto_wsp import (
    TensorLayout, TensorShard, TensorReplicate, MemLayout,
    relayout, allreduce, allgather, reduce_scatter
)

# Distribution facet (Dato-style)
TensorShard(mesh_axis=0)  # S(0) - sharded on mesh axis 0
TensorReplicate()          # R - replicated

# Memory facet (Triton-style)
MemLayout.row_major((M, N))
MemLayout.col_major((M, N))
MemLayout.permute((M, N), (1, 0))

# Combined layout
layout = TensorLayout(
    dist=(TensorShard(0), TensorReplicate()),
    mem=MemLayout.row_major((M, N))
)

# Layout transitions (explicit collectives)
sharded = relayout(replicated, TensorLayout.sharded(dim=0, rank=2))
reduced = allreduce(partial, mesh_axis=0, op="sum")
gathered = allgather(sharded, dim=0, mesh_axis=0)
scattered = reduce_scatter(full, dim=0, mesh_axis=0)
```

**Join rules** (Dato-style):
- R ⊔ R = R
- R ⊔ S(i) = S(i)
- S(i) ⊔ S(i) = S(i)
- S(i) ⊔ S(j) = ERROR (i≠j)

**Code**: `python/pto_wsp/types.py`
**Docs**: `docs/research/type_system_research.md`

---

### 13. Linear Layout (F₂)

**Purpose**: Triton-style linear layout representation using F₂ binary matrices for efficient tensor memory layouts.

Based on [arXiv:2505.23819](https://arxiv.org/abs/2505.23819) - Triton Linear Layout system.

```python
from pto_wsp.linear_layout import (
    LinearLayout, f2_dot, f2_matmul, f2_rank,
    propagate_transpose, propagate_reshape, propagate_broadcast
)

# Create blocked layout (threads process contiguous elements)
blocked = LinearLayout.blocked(
    total_elements=4096,
    num_threads=256,
    block_size=16
)

# Create strided layout (threads access interleaved elements)
strided = LinearLayout.strided(
    total_elements=4096,
    num_threads=256
)

# Apply layout to compute physical address
logical_idx = 42
physical_addr = blocked.apply_index(logical_idx)

# Compose layouts (row_major then blocked)
row = LinearLayout.row_major(64, 64)
composed = row.compose(blocked)

# Compute swizzle for bank conflict avoidance
swizzled = LinearLayout.compute_swizzle(
    layout=blocked,
    bank_bits=4,     # 16 banks
    vector_bits=3    # 8-element vectors
)
```

**F₂ Matrix Operations**:

| Function | Purpose |
|----------|---------|
| `f2_dot(a, b)` | Dot product over F₂ (XOR of ANDs) |
| `f2_matmul(A, B)` | Matrix multiply over F₂ |
| `f2_rank(A)` | Rank of binary matrix |
| `propagate_transpose(L, perm)` | Propagate transpose through layout |
| `propagate_reshape(L, old, new)` | Propagate reshape through layout |
| `propagate_broadcast(L, dim)` | Propagate broadcast through layout |

**Factory Methods**:

| Method | Description |
|--------|-------------|
| `LinearLayout.identity(bits)` | Identity layout matrix |
| `LinearLayout.blocked(total, threads, block)` | Blocked/tiled layout |
| `LinearLayout.strided(total, threads)` | Strided/cyclic layout |
| `LinearLayout.row_major(M, N)` | Row-major memory layout |
| `LinearLayout.col_major(M, N)` | Column-major memory layout |

**Code**: `python/pto_wsp/linear_layout.py`
**Docs**: `docs/research/linear_layout.md`

---

### 14. Type Checker

**Purpose**: Builder-time validation of workload and kernel calls.

```python
from pto_wsp import TypeChecker, validate_axis_index, check_layouts_compatible

# Automatic in @workload
@workload(type_check=True, fail_on_type_error=True)
def strict_workload(batch, heads):
    ...

# Manual validation
axis = Dense[8]()
validate_axis_index(axis, 5)   # OK
validate_axis_index(axis, 10)  # Raises TypeError

# Layout compatibility
result = check_layouts_compatible(tensor1, tensor2)
if result is None:
    print("Need relayout() between operands")
```

**Checks performed**:
- Kernel arity match
- Argument type validation
- Layout compatibility (Dato-style join)
- Axis bounds validation
- Shape compatibility

**Code**: `python/pto_wsp/type_checker.py`
**Docs**: `docs/design/type-system.md`

---

## Layer 7: C++ Backend

### 15. C++ IR

**Purpose**: Multi-level IR for CPU and NPU workloads.

```cpp
namespace pto::wsp::ir {
    // Core IR nodes
    class IRNode;
    class WorkloadNode;
    class TaskNode;
    class ParallelForNode;
    class ForEachNode;

    // NPU IR
    class NPUFunction;
    class NPUOp;

    // Visitor pattern
    class IRVisitor;

    // Type checking
    class TypeCheckPass;
}
```

**Components**:
- Workload IR: Task generation patterns
- NPU IR: InCore operations
- Visitor pattern for traversal
- Type checking pass

**Code**: `include/pto/rt/ir/`, `src/pto/rt/ir/`
**Docs**: `docs/design/ir-design.md`

---

### 16. Backend Architecture

**Purpose**: Pluggable backend system for multi-target compilation.

```cpp
namespace pto::wsp::backend {
    // Backend interface
    class Backend;

    // Code generation
    class CodeEmitter;
    class CodeGenerator;
    class EmitterRegistry;

    // Concrete backends
    class CPUSimBackend;
    class AscendNPUEmitter;
}
```

**Supported backends**:

| Backend | Target | Purpose |
|---------|--------|---------|
| CPU Sim | `cpu_sim` | Debug and validation |
| Ascend NPU | `ascend_npu` | Production deployment |

**Code**: `include/pto/rt/backend/`, `src/pto/rt/backend/`
**Docs**: `docs/design/backend-arch.md`

---

### 17. Concurrent Utilities

**Purpose**: Common concurrency primitives extracted from pto-isa-lh/wc for reuse across backends.

Implements L12 requirement for backend-neutral concurrency utilities.

```cpp
#include "pto/rt/concurrent/utilities.hpp"

namespace pto::wsp::concurrent {
    // Completion tracking
    CompletionCounter counter(100);
    counter.increment();
    counter.decrement();
    counter.wait_for_zero_with_backoff();

    // Single-use barrier
    Latch latch(3);
    latch.count_down();
    latch.wait();

    // Reusable barrier
    Barrier barrier(4);
    barrier.arrive_and_wait();

    // Thread-safe queue
    BoundedQueue<Task> queue(4096);
    queue.push(task);
    auto task = queue.pop();
    queue.close();

    // Thread pool
    ThreadPool pool(8);
    pool.start();
    pool.submit([](){ /* work */ });
    pool.shutdown();

    // Multi-domain handshake (CPU↔Accelerator)
    DomainHandshake handshake;
    handshake.cpu_ready();
    handshake.wait_accelerator_ready();
    handshake.transfer_complete();
}
```

**Utilities**:

| Class | Purpose |
|-------|---------|
| `CompletionCounter` | Atomic counter with completion notification |
| `Latch` | Single-use synchronization barrier |
| `Barrier` | Reusable multi-phase barrier |
| `BoundedQueue<T>` | MPMC bounded queue with blocking ops |
| `ThreadPool` | Fixed-size worker thread pool |
| `DomainHandshake` | CPU↔Accelerator coordination (L12) |
| `parallel_for` | Data-parallel loop utility |

**Execution Domains**:

```cpp
enum class ExecDomain {
    HostCPU,       // CPU simulation
    AscendAICore,  // Ascend NPU core
    AMDAIETile,    // AMD AIE tile
    Generic
};
```

**Code**: `include/pto/rt/concurrent/utilities.hpp`

---

## Test Coverage

| Component | Test File | Count |
|-----------|-----------|-------|
| Python Frontend | `tests/test_python_frontend.py` | 118 |
| End-to-End | `tests/test_e2e.py` | 33 |
| Linear Layout | `tests/test_linear_layout.py` | 32 |
| Pybind Integration | `tests/test_pybind_integration.py` | 32 |
| C++ IR | `tests/test_ir.cpp` | 29 |
| C++ Graph | `tests/test_graph.cpp` | 16 |
| C++ Backend | `tests/test_backend.cpp` | 21 |
| **Total** | | **281** |

---

## Related Documents

| Category | Documents |
|----------|-----------|
| Design | `docs/analysis.md` (WHY), `docs/spec.md` (WHAT) |
| Detailed Design | `docs/design/ir-design.md`, `docs/design/backend-arch.md`, `docs/design/type-system.md`, `docs/design/npu-design.md` |
| Research | `docs/reference/` (external), `docs/research/` (working) |
| Planning | `docs/task_plan.md`, `docs/comments.md` |
