# PTO Workload-Schedule Programming (PTO-WSP) v9: API Specification

> **Design Version:** v9.3 | **Implementation Version:** 0.1.0 (Prototype)

## 1. Overview

This specification defines the v9 API for PTO-ISA runtime extension, supporting:

1. **Python frontend** with declarative and combinator-style APIs
2. **Typed workload expressions** preserved from v8
3. **CSP pipeline-parallel** with `Channel`, `Process`, `consume` *(Partial)*
4. **Spatial schedule primitives** for dataflow architectures
5. **Multi-backend compilation** (CPU sim, NPU)

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
    jit_kernel,                 # @jit_kernel decorator (Triton-style)
    pto,                        # PTO tile language primitives (pto.load, pto.store, etc.)
    In, Out, InOut, Constexpr,  # Type annotations for kernel signatures
    Tile, Scalar,               # Typed kernel parameters

    # CSP primitives
    Channel, process, send, consume, connect, replicate,

    # Schedule policies
    DispatchPolicy, TimingPolicy,  # IssuePolicy removed (unused)

    # Task graph configuration (Experimental)
    Deps, ReadyPolicy, StartPolicy, TracePolicy, Pools,

    # Tensor layout (R10)
    TensorLayout, TensorShard, TensorReplicate, MemLayout,
    relayout, allreduce, allgather, reduce_scatter,

    # Spatial layout (use TensorShard/TensorReplicate instead)
    # Shard, Replicate,  # Deprecated

    # Execution
    Program,

    # Legacy (deprecated - do not use)
    # register_kernel, ExternalKernel, npu, NPUFunction, kernel_legacy
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
    layout: TensorLayout      # NEW: Layout refinement (R8, R10)

    def __getitem__(self, idx: int) -> Tensor: ...
    def slice(self, start: int, end: int) -> Tensor: ...
    def nbytes(self) -> int: ...
```

### 2.3 Layout Types (New in v9 - R8, R10)

Layout is a **type-level refinement** (not a schedule primitive). See `docs/research/type_system_research.md` for rationale.

```python
from pto_wsp import TensorLayout, TensorShard, TensorReplicate, MemLayout

# TensorLayout = (distribution per dim) × (optional MemLayout)
#
# - Distribution facet: per-dimension elements of {R, S(mesh_axis)}
# - Memory facet: optional MemLayout (strides/order/swizzle)

# All-replicated by default
layout0 = TensorLayout.default(rank=4)

# Shard only dim=0 onto mesh axis 0 (others replicated)
layout1 = TensorLayout.sharded(dim=0, rank=4, mesh_axis=0)

# Full control (explicit dist tuple + optional memory layout)
layout2 = TensorLayout(
    (TensorShard(0), TensorReplicate(), TensorReplicate(), TensorReplicate()),
    mem=MemLayout.row_major((B, H, S, D)),
)
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

**v9 runtime semantics (codegen-first):**
- Predicates are compiled as a structured expression IR (**ScalarExpr**) and evaluated inside the generated artifact.
- Predicates may depend on task-local values (axis vars / params), runtime symbols, and runtime **slots** (see below).
- Do not rely on Python booleans or host-side “driver loops” for conditional behavior in v9.

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

### 3.8 Slot Primitives (Tensor → scalar bridge)

Dynamic workloads often need **data-dependent control flow** (e.g., routing decisions) where a kernel produces tensor data,
but `cond`/schedule keys need scalars/bools. v9 exposes a minimal “slot” facility to bridge tensor values into ScalarExpr:

- `slot_set_u64(slot, value)` — debug/tests
- `slot_load_u64(slot, tensor_view, row=0, col=0)` — load a tensor element into a runtime slot
- `slot_u64(i)` — use the slot value in ScalarExpr predicates/keys (via `pto_wsp.scalar_expr`)

These are compiled and executed inside artifacts (CPU-sim), enabling data-dependent branching without recompiling.

---

## 4. Schedule API

Schedules use **combinator style**: each method returns a new schedule with the policy applied, enabling type-safe chaining.

### 4.1 Combinator-Style Schedule

```python
# Create schedule from workload using combinator style
program = (workload
    .dispatch(DispatchPolicy.round_robin(4))
    .task_graph(window=TaskWindow(8192, "tasks", WindowMode.STALL))
    .compile(target="cpu_sim"))

# Notes (v9):
# - `streams/stream_by/timing` exist as schedule API but are not fully enforced in v9 artifacts.
# - Layout is a tensor type refinement; use `TensorLayout` + `relayout(...)` (not `schedule.layout(...)`).
```

**Key difference from v8:** The schedule is bound to the workload via combinator methods, enabling better type checking. Each method returns a new schedule with accumulated policies.

### 4.2 Dispatch Policies

```python
from pto_wsp import DispatchPolicy

# Round-robin across AICPUs
schedule.dispatch(DispatchPolicy.round_robin(num_aicpus=4))

# Same axis value → same AICPU
schedule.dispatch(DispatchPolicy.affinity(lambda t: t.batch))

# Hash-based
schedule.dispatch(DispatchPolicy.hash(lambda t: t.key))

# Custom function
schedule.dispatch(DispatchPolicy.dispatch_by(lambda t: custom_logic(t)))

# Note (v9): DispatchPolicy.work_steal() exists as API but is currently diagnosed/unsupported in v9 artifacts.
```

### 4.3 Stream Assignment

Stream assignment uses the `stream_by()` combinator method:

```python
# Stream assignment by key function
schedule = workload.streams(4).stream_by(lambda t: t.get("head") % 4)
```

> **Note (v9)**: `streams()` may affect worker count as a fallback, but `stream_by()` is not enforced in v9 artifacts.

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

> **Note (v9)**: `timing(...)` is currently diagnosed/ignored in v9 artifacts (it does not affect `total_cycles`).

### 4.5 Spatial Primitives (New in v9)

```python
# Spatial mapping: workload → tile grid (defines the Mesh environment)
schedule.spatial_map(grid=(4, 4))  # 4x4 tile array
```

Note: Layout is a **type-level refinement** on tensors (see Section 2.3), not a schedule primitive. `spatial_map(grid=...)` defines the Mesh environment for validating `Shard(mesh_axis=...)` refinements.

### 4.6 Complete Example

```python
program = (workload
    .dispatch(DispatchPolicy.round_robin(4))
    .task_graph(window=TaskWindow(8192, "tasks", WindowMode.STALL))
    .compile(target="cpu_sim"))
```

### 4.7 Task Graph Execution (New in v9 - R9)

`task_graph()` configures DAG-style execution. In v9 codegen-first artifacts, only the **task_window (stall-only, unit=tasks)**
behavior is enforced; other advanced task-graph knobs are currently API-only/diagnostic.

Minimal v9 usage:

```python
program = (workload
    .dispatch(DispatchPolicy.round_robin(4))
    .task_graph(window=TaskWindow(8192, "tasks", WindowMode.STALL))
    .compile(target="cpu_sim"))
```

Advanced task-graph features (dependency inference modes, pools, ready/start/trace policies) are documented in `docs/design/`,
but are not fully enforced in v9 artifacts.

### 4.8 Backend Applicability

Different schedule primitives may not be applicable to all backends. In v9, PTO‑RT follows a
**codegen-first** model: backend behavior is realized by the generated artifact, and unsupported
schedule directives are either ignored with diagnostics or explicitly rejected (see below).

#### 4.8.1 v9 Enforcement Summary (as-built)

| Primitive | CPU-sim artifact | Ascend NPU emission (this env) |
|-----------|------------------|--------------------------------|
| `dispatch(round_robin/hash/affinity/custom)` | **enforced** | **preserved in plan** |
| `dispatch(work_steal)` | **unsupported** (diagnosed) | **unsupported** (annotated) |
| `task_graph(window=TaskWindow(..., mode=STALL))` | **enforced** (stall-only) | **preserved in plan** |
| `streams(n)` | **partial** (used only as worker-count fallback) | **unsupported** (annotated) |
| `stream_by(...)` | **unsupported** (diagnosed) | **unsupported** (annotated) |
| `timing(...)` | **unsupported** (diagnosed) | **unsupported** (annotated) |

v9 does not guarantee deterministic interleavings; tests validate invariants (correctness + cycle-time accounting).

---

## 5. Compilation and Execution

### 5.0 Execution Path

PTO‑RT v9 uses a **codegen-first** compilation pipeline:

1. Python authoring builds a workload tree + kernel IR (`@workload`, `@kernel`, `pto.*`, `ptoisa.*`).
2. `python/pto_wsp/ir_bridge.py` converts the workload into a C++ `ir::Module` (via `pto_ir_cpp` bindings).
3. C++ compilation/codegen emits backend artifacts:
   - `target="cpu_sim"`: emits C++ sources, builds a cached `.so`, and executes it via `dlopen`.
   - `target="ascend_npu"`: emits a host/AICPU/AICore source tree for inspection (device build/run requires Ascend/CANN).

There is no Python threadpool fallback executor in v9.

### 5.1 compile()

```python
# Compile to executable program (CPU simulation)
program = workload.compile(target="cpu_sim")

# Emit NPU sources (emit-only in this environment)
program = workload.compile(target="ascend_npu")
print(program.codegen_artifact_dir)
```

### 5.2 Program Class

```python
class Program:
    def execute(self) -> None:
        """Execute the program."""
        ...

    def synchronize(self) -> None:
        """Wait for completion."""
        ...

    def is_complete(self) -> bool:
        """Check if execution is done."""
        ...

    @property
    def stats(self) -> ProgramStats:
        """Execution statistics (includes total_cycles)."""
        ...

    @property
    def codegen_artifact_dir(self) -> str | None:
        """For codegen-only targets (e.g. ascend_npu emit-only), emitted directory."""
        ...

class ProgramStats:
    compile_time_ms: float
    execute_time_ms: float
    task_count: int
    total_cycles: int
```

### 5.3 NPU emission and on-device expansion (v9)

For `target="ascend_npu"`, v9 emits an artifact source tree that includes an AICPU “expander” translation unit that expands
tasks on-device from a compact plan + runtime symbols. Full device build/execution is toolchain-gated (Ascend/CANN).

`docs/design/on-device-task-gen.md` contains design exploration of a bytecode interpreter approach; treat that as future work unless
explicitly implemented and validated in this repo.

### 5.4 Deprecated / removed APIs (v9)

- `Kernel.compile()` is not supported in v9 codegen-first mode; compile at the workload level via `workload.compile(target=...)`.
- `Program.register_kernel(...)` is not supported in codegen-first mode.
- `Workload.layout(...)` is deprecated (emits `DeprecationWarning`); use `TensorLayout` + `relayout(...)`.
- `tl.*` is a deprecated alias for `pto.*`.
- `task(kernel: str, ...)` exists for legacy enumerate-only workflows; prefer direct `@kernel` calls inside `@workload`.

---

## 6. Kernel Definition (JIT-Style)

PTO-WSP v9 supports **JIT-style kernel definitions** that eliminate string-based task names (R7). Kernels are defined as decorated Python functions and called directly within workload contexts.

### 6.1 The @jit_kernel Decorator (RECOMMENDED)

The `@jit_kernel` decorator provides a Triton-style programming model with typed `Value` objects and `pto.*` primitives. This is the **recommended** approach for new code.

```python
from pto_wsp import jit_kernel, pto, In, Out, Tile, Scalar, DType

@jit_kernel
def rmsnorm(
    x: In[Tile[32, 128, DType.F16]],
    out: Out[Tile[32, 128, DType.F16]],
    eps: Scalar[DType.F32] = 1e-6
):
    """RMS normalization kernel using typed Value objects.

    No string-based references - all operations return typed Values.
    """
    # All operations return Value objects (no strings!)
    sq = pto.mul(x, x)              # sq: Value
    mean = pto.rowmean(sq)          # mean: Value
    rsqrt_val = pto.rsqrt(mean)     # rsqrt_val: Value
    pto.store(out, pto.mul(x, rsqrt_val))
```

**Tile Language Primitives (`pto.*`):**

| Category | Primitives |
|----------|------------|
| **Memory** | `pto.load(src)`, `pto.store(dst, src)`, `pto.alloc(shape, dtype)` |
| **Arithmetic** | `pto.add(a, b)`, `pto.sub(a, b)`, `pto.mul(a, b)`, `pto.div(a, b)`, `pto.maximum(a, b)`, `pto.minimum(a, b)` |
| **Math** | `pto.exp(x)`, `pto.log(x)`, `pto.sqrt(x)`, `pto.rsqrt(x)`, `pto.tanh(x)`, `pto.sigmoid(x)`, `pto.sin(x)`, `pto.cos(x)` |
| **Activations** | `pto.relu(x)`, `pto.gelu(x)`, `pto.silu(x)`, `pto.neg(x)`, `pto.abs(x)` |
| **Reductions** | `pto.rowsum(x)`, `pto.rowmax(x)`, `pto.rowmean(x)`, `pto.colsum(x)`, `pto.colmax(x)` |
| **Broadcast** | `pto.rowmul(tile, vec)`, `pto.rowadd(tile, vec)`, `pto.colmul(tile, vec)`, `pto.coladd(tile, vec)`, `pto.broadcast(x, shape)` |
| **Matrix** | `pto.matmul(a, b, acc=None)` |
| **Special** | `pto.constant(value, dtype)`, `pto.slice_even(x)`, `pto.slice_odd(x)`, `pto.interleave(a, b)` |

> **v9 note (Path A):** `TopK` is not a PTO‑RT primitive (`pto.topk` is not supported). Implement TopK-style logic as a **custom kernel** using PTO‑ISA tile primitives via either `@ptoisa_kernel` (Python → emitted C++ body) or `@kernel(cpp_src=..., cpp_includes=...)` (manual C++).

**Key Features:**
- **Typed Values**: All `pto.*` operations return `Value` objects with type information
- **No String References**: Unlike the legacy NPU builder, no string-based tile names
- **Compile to Backend**: use `workload.compile(target=...)` to build codegen artifacts from the C++ IR module (**v9 does not support** `Kernel.compile()`).

### 6.2 The @kernel Decorator (Builder-Style + Custom C++ Kernels)

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
    1. Traced `pto.*` ops (JIT kernel IR)
    2. Empty + a custom implementation (Path A):
       - `@ptoisa_kernel` (Python authoring → emitted C++ body), or
       - `@kernel(cpp_src=..., cpp_includes=...)` (manual C++ body)
    """
    pass  # Implementation provided externally or via CPU sim
```

**Custom kernel (Path A):**

```python
@kernel(
  cpp_includes=["pto/cpu/TMrgSort.hpp"],
  cpp_src=\"\"\"/* C++ body compiled into the artifact */\"\"\",
)
def topk_kernel(...):  # annotated params required
    pass
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
from pto_wsp import jit_kernel, pto, DType

@jit_kernel
def my_kernel(x: In[Tile[M, N, DType.F16]], out: Out[Tile[M, N, DType.F16]]):
    pto.store(out, pto.mul(x, x))

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

---

## 9. Error Handling

```python
from pto_wsp import (
    PtoError,
    CompileError,
    TypeCheckError,
    IRConversionError,
    ExecutionError,
    KernelError,
    ScheduleError,
    ChannelError,
    ChannelClosed,
    ChannelFull,
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
6. **Multi-backend**: Same workload targets CPU sim and Ascend NPU
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
