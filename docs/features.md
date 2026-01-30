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

### Status vocabulary (v9)

- **✓ Done**: feature exists; see per-section notes for whether it is **enforced** vs **API-only** in v9 artifacts.
- **Partial**: feature surface exists but enforcement is limited (usually “diagnosed/ignored”).
- **Removed**: legacy surface intentionally removed in v9.

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
| NPU Function Builder (Removed) | N/A | Removed | [Details](#7-npu-function-builder-removed) |
| Schedule Combinator API | `pto_wsp.workload` | Partial (v9 runtime subset) | [Details](#8-schedule-combinator-api) |
| Task Graph (R9) | `pto_wsp.schedule` | Partial (v9: task_window stall) | [Details](#9-task-graph-r9) |
| Schedule Configuration Classes | `pto_wsp.schedule` | API-only / not enforced | [Details](#10-schedule-configuration-classes) |
| CSP Primitives | `pto_wsp.csp` | Partial (CPU-sim CSPT) | [Details](#11-csp-primitives) |
| Layout Types (R10) | `pto_wsp.types` | ✓ Done | [Details](#12-layout-types-r10) |
| Linear Layout (F₂) | `pto_wsp.linear_layout` | ✓ Done | [Details](#13-linear-layout-f2) |
| Type Checker | `pto_wsp.type_checker` | ✓ Done | [Details](#14-type-checker) |
| C++ IR | `include/pto/wsp/ir/` | ✓ Done | [Details](#15-c-ir) |
| Backend Architecture | `include/pto/wsp/backend/` | ✓ Done | [Details](#16-backend-architecture) |
| Concurrent Utilities | `include/pto/wsp/concurrent/` | ✓ Done | [Details](#17-concurrent-utilities) |

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
from pto_wsp import DType, In, Out, Tile, kernel, pto

@kernel
def rmsnorm_kernel(x: In[Tile[32, 128, DType.F16]], out: Out[Tile[32, 128, DType.F16]]):
    sq = pto.mul(x, x)
    mean = pto.rowmean(sq)
    rsqrt_val = pto.rsqrt(mean)
    pto.store(out, pto.mul(x, rsqrt_val))
```

**Tile Language Primitives** (`pto.*`):

The tracing IR supports a broad set of operations, but v9 **codegen-first CPU-sim** correctness is validated for the
subset exercised by examples/tests. Supported-by-codegen operations include:

| Category | Operations |
|----------|------------|
| Binary | `pto.add`, `pto.mul`, `pto.sub`, `pto.div`, `pto.max`, `pto.min` |
| Unary | `pto.exp`, `pto.rsqrt` |
| Reduction | `pto.rowsum`, `pto.rowmax`, `pto.rowmean` |
| Memory | `pto.load`, `pto.store` |
| MatMul | `pto.matmul` |

**Not supported in v9 codegen-first mode:** `Kernel.compile()` (compile kernels in isolation).
Use `workload.compile(target=...)` to build runnable artifacts from the C++ IR module.

**Path A (custom PTO‑ISA kernels):**
Some functionality (e.g., TopK/sorting primitives) is intentionally **not** exposed as a PTO‑RT primitive op.
Instead, author kernels using PTO‑ISA tile primitives and compile them into the codegen artifact via either:

- `@ptoisa_kernel` (Python authoring → emits a C++ kernel body automatically), or
- `@kernel(cpp_src=..., cpp_includes=...)` (manual C++ body; escape hatch).

Additionally, if you want to keep kernel code in a standalone C++ file, use:

- `@kernel(cpp_body_path="path/to/snippet.cpp")` (file contains a *body snippet* inserted into the generated wrapper), or
- `@kernel(cpp_tu_path="path/to/kernel.cpp")` (file contains a full translation unit with an `extern "C"` kernel definition).

**Three kernel-authoring modes (v9):**
- `pto.*` IR-traced kernels lowered to PTO‑ISA calls (default)
- `ptoisa.*` instruction-traced kernels (`@ptoisa_kernel`)
- file-based custom C++ kernels (`cpp_body_path` / `cpp_tu_path`)

**Code**: `python/pto_wsp/kernel.py`
**Docs**: `docs/design/npu-design.md`

---

### 7. NPU Function Builder (Removed)

> **Note**: The legacy string-based NPU function builder (`npu()`) has been removed in v9.
> Use `@jit_kernel` with `pto.*` primitives instead (Section 6).

The legacy API used string-based tile/memref names which were error-prone:

```python
# REMOVED in v9 - use @jit_kernel instead
# rmsnorm = npu("rmsnorm").tile("x", ...).mul("sq", "x", "x")...

# NEW (v9) - Use typed Values with pto.* primitives:
from pto_wsp import jit_kernel, pto, In, Out, Tile, DType

@jit_kernel
def rmsnorm(x: In[Tile[32, 128, DType.F16]], out: Out[Tile[32, 128, DType.F16]]):
    sq = pto.mul(x, x)              # Returns Value (typed!)
    mean = pto.rowmean(sq)          # No string refs
    rsqrt_val = pto.rsqrt(mean)
    pto.store(out, pto.mul(x, rsqrt_val))
```

**Why removed**: String-based references were error-prone and didn't support compile-time type checking. The `@jit_kernel` decorator with `pto.*` primitives provides full type safety and IDE support.

---

## Layer 4: Scheduling

### 8. Schedule Combinator API

**Purpose**: Type-safe schedule composition through method chaining.

**v9 runtime support (codegen-first CPU-sim artifacts):**
- **Supported semantics:** `dispatch(...)` and `task_graph(window=TaskWindow(..., mode=STALL))` (stall-only task_window).
- **Metrics:** `Program.stats.total_cycles` is derived from **PTO-ISA kernel cycle reports** (makespan / CSPT time).
- **Not enforced in v9 artifacts:** `stream_by`, `timing`, advanced task-graph config (deps/pools/ready/start/trace). These are either ignored or explicitly diagnosed.
- **NPU emission:** preserves `dispatch` + `task_window`; other schedule knobs are explicitly marked unsupported in emitted sources.
- **Tensor-driven predicates (v9):** artifacts provide runtime u64 slots for data-dependent control via:
  - `slot_load_u64(slot, tensor_view, row=0, col=0)` (load tensor element into slot)
  - `slot_set_u64(slot, value)` (tests/debug)
  - `slot_u64(i)` in `ScalarExpr` predicates/keys (e.g., for `cond(...)`)

```python
from pto_wsp import DispatchPolicy, TaskWindow, WindowMode

program = (workload
    .dispatch(DispatchPolicy.round_robin(4))   # Task→worker assignment
    .task_graph(window=TaskWindow(8192, "tasks", WindowMode.STALL))  # task_window (stall-only)
    .compile(target="cpu_sim"))
```

**Available methods**:

| Method | Purpose |
|--------|---------|
| `.dispatch(policy)` | Task→executor assignment |
| `.streams(count)` | Number of concurrent streams |
| `.stream_by(fn)` | Stream assignment function |
| `.timing(policy)` | Task issue timing |
| `.spatial_map(grid)` | Tile grid mapping (API-only / no-op in v9 artifacts) |
| `.task_graph(...)` | DAG-based execution (see next) |
| `.compile(target)` | Compile to executable |

**Code**: `python/pto_wsp/workload.py`
**Docs**: `docs/spec.md` Section 5

---

### 9. Task Graph (R9)

**Purpose**: DAG-based execution as alternative to streams, matching pto-isa-lh capabilities.

**v9 runtime status:** Only `TaskWindow(..., mode=STALL, unit="tasks")` is enforced as a `task_window` constraint in the
codegen-first CPU-sim artifact. Other task-graph configuration parameters are currently **not** enforced by the v9 artifact
runtime and should be treated as API placeholders (they may be ignored/diagnosed).

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

### 10. Schedule Configuration Classes

**Purpose**: Task graph configuration primitives for fine-grained control.

The extended schedule primitives (dispatch_threshold, pipeline_depth, etc.) from early v9 design were simplified.

**v9 runtime status:** these configuration classes exist for API completeness, but (beyond `TaskWindow(..., mode=STALL, unit="tasks")`)
they are **not enforced** by the codegen-first v9 artifact runtime yet.

```python
from pto_wsp import (
    Deps, TaskWindow, WindowMode, Pools,
    ReadyPolicy, StartPolicy, TracePolicy
)

# Configure task graph execution with fine-grained control
program = (workload
    .dispatch(DispatchPolicy.work_steal())
    .task_graph(
        deps=Deps.infer_tensor_map_exact(),      # Dependency inference
        window=TaskWindow(8192, "tasks", WindowMode.STALL),  # Window management
        pools=Pools.by_exec_unit(),              # Pool routing
        ready=ReadyPolicy.work_steal(),          # Ready queue policy
        start=StartPolicy.threshold(100),        # Execution start
        trace=TracePolicy.cycles()               # Cycle simulation
    )
    .compile())
```

**Available configuration classes**:

| Class | Purpose | Options |
|-------|---------|---------|
| `Deps` | Dependency inference | `infer_tensor_map_exact()`, `infer_bytes_overlap()`, `explicit()`, `hybrid()` |
| `TaskWindow` | Metadata window | `TaskWindow(size, unit, mode)` |
| `WindowMode` | Overflow behavior | `STALL`, `ABORT`, `BENCHMARK` |
| `Pools` | Execution pools | `single()`, `by_exec_unit()`, `custom()` |
| `ReadyPolicy` | Ready queue | `fifo()`, `work_steal()`, `priority(fn)` |
| `StartPolicy` | Start timing | `after_orchestration()`, `threshold(n)` |
| `TracePolicy` | Tracing | `none()`, `cycles(cost_fn)` |

**Code**: `python/pto_wsp/schedule.py`

---

## Layer 5: CSP Pipeline

### 11. CSP Primitives

**Purpose**: Pipeline-parallel programming with channels and processes.

**v9 runtime support:**
- **CPU-sim (codegen-first):** CSP pipelines execute inside the generated artifact with CSPT time semantics.
  - Timebase: **PTO-ISA kernel cycle reports** + constant channel latency (default `0`).
  - Latency override without rebuild: runtime symbol `__pto_wsp_channel_latency_cycles`.
- **NPU:** emission preserves `dispatch` + `task_window` scheduling info; CSP runtime execution on-device requires CANN/toolchain (not runnable in this environment).

**Not codegenned in v9:** the Python convenience helpers `record/synchronize/query` are not part of the codegen-first CSP path.
Use `Channel(depth=0)` rendezvous channels + `send/consume` to express synchronization in workloads.

```python
from pto_wsp import Channel, process, send, consume, connect, replicate

# Create channels
ch = Channel("data", depth=2)  # Buffered channel
event = Channel("sync", depth=0)  # Unbuffered (rendezvous)

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

```

**Primitives**:

| Primitive | Purpose |
|-----------|---------|
| `Channel(name, depth)` | Bounded FIFO queue |
| `process(name)` | Process builder |
| `send(ch, val)` | Send to channel |
| `consume(ch, fn)` | Receive and process |
| `connect(procs, chs)` | Wire pipeline |
| `replicate(proc, n)` | Create n instances |
| `Channel(name, depth=0)` | Rendezvous-style sync |

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

**Code**: `include/pto/wsp/ir/`, `src/pto/wsp/ir/`
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

**Code**: `include/pto/wsp/backend/`, `src/pto/wsp/backend/`
**Docs**: `docs/design/backend-arch.md`

---

### 17. Concurrent Utilities

**Purpose**: Common concurrency primitives extracted from pto-isa-lh/wc for reuse across backends.

Implements L12 requirement for backend-neutral concurrency utilities.

```cpp
#include "pto/wsp/concurrent/utilities.hpp"

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
    Generic
};
```

**Code**: `include/pto/wsp/concurrent/utilities.hpp`

---

### 18. On-Device Task Generation

**Purpose**: Efficient task generation on AICPU instead of host-side expansion.

**Key insight**: Workload is a **program** that generates tasks, not a **list** of tasks.

```
Host-side expansion (pto-isa-lh):
- Expand parallel_for → O(N) task structures
- Transfer ~400MB to device
- High latency before execution starts

On-device generation (PTO-WSP v9):
- Compile workload → ~4KB bytecode
- Transfer bytecode to each AICPU
- Each AICPU generates only its own tasks
- Pipelined with execution (low latency)
```

**Bytecode format:**

```cpp
enum class WLOpcode : uint8_t {
    PARALLEL_FOR    = 0x10,  // axis_id, body_offset
    TASK            = 0x20,  // kernel_id, params, resources
    DISPATCH_FILTER = 0x40,  // which AICPU owns this task
};

struct WorkloadBytecode {
    uint32_t magic;           // 0x50544F57 = "PTOW"
    uint32_t version;
    uint32_t num_instructions;
    // Followed by instructions
};
```

**Interpreter on AICPU:**

```cpp
void WorkloadInterpreter::interpret() {
    while (pc < bytecode->num_instructions) {
        switch (instr->opcode) {
        case WLOpcode::PARALLEL_FOR:
            for (int32_t i = 0; i < extent; i++) {
                push_loop(i);
                interpret_body();  // Recursive
                pop_loop();
            }
            break;

        case WLOpcode::TASK:
            if (evaluate_dispatch_predicate()) {
                emit_task(instr);  // Only my tasks
            }
            break;
        }
    }
}
```

**Benefits:**

| Metric | Host Expansion | On-Device Gen |
|--------|----------------|---------------|
| Data Transfer | ~400MB | ~4KB |
| Startup Latency | High | Low (pipelined) |
| Dynamic Shapes | Rebuild graph | Same bytecode |
| Memory | O(tasks) on host | O(1) bytecode |

**Code**: `include/pto/wsp/backend/` (on-device generation infrastructure)
**Docs**: `docs/design/on-device-task-gen.md`

---

## Test Coverage

| Component | Test File | Count |
|-----------|-----------|-------|
| Python Frontend | `tests/test_python_frontend.py` | 134 |
| End-to-End | `tests/test_e2e.py` | 40 |
| Linear Layout | `tests/test_linear_layout.py` | 32 |
| Pybind Integration | `tests/test_pybind_integration.py` | 32 |
| C++ IR | `tests/test_ir.cpp` | 30 |
| C++ Graph | `tests/test_graph.cpp` | 17 |
| C++ Backend | `tests/test_backend.cpp` | 28 |
| **Total** | | **313** |

---

---

## Implementation Notes

This section is intentionally high-level. For the authoritative “as-built” guide (code pointers + diagrams), see
`docs/implementation.md`.

Key v9 behaviors:

- **Codegen-first execution:** semantics live in generated artifacts (CPU-sim) / emitted source tree (NPU).
- **Cycles/time:** derived strictly from PTO-ISA kernel cycle reports (no separate estimator).
- **Scheduling (CPU-sim):** `dispatch` + `task_window` (stall-only) are behavior-changing; other knobs are ignored/diagnosed.
- **CSP/CSPT (CPU-sim):** pipeline workloads execute inside the artifact; channel latency is constant (default `0`, symbol override).

---

## Related Documents

| Category | Documents |
|----------|-----------|
| Design | `docs/analysis.md` (WHY), `docs/spec.md` (WHAT) |
| Detailed Design | `docs/design/ir-design.md`, `docs/design/backend-arch.md`, `docs/design/type-system.md`, `docs/design/npu-design.md` |
| Research | `docs/reference/` (external), `docs/research/` (working) |
| Historical notes | `docs/archive/comments.md` |
