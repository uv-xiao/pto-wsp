# PTO-RT v9 Tutorial

A step-by-step guide to writing typed workload expressions for dynamic LLM workloads.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Defining Kernels](#defining-kernels)
3. [Creating Workloads](#creating-workloads)
4. [Scheduling and Execution](#scheduling-and-execution)
5. [Pipeline Parallelism with CSP](#pipeline-parallelism-with-csp)
6. [Profiling and Tracing](#profiling-and-tracing)
7. [Advanced Topics](#advanced-topics)

---

## Getting Started

### Installation

```bash
# Clone the repository
git clone https://github.com/your-org/pto-wsp.git
cd pto-wsp

# Install Python package
pip install -e .
```

### Basic Concepts

PTO-RT provides typed workload expressions for expressing parallel computations:

- **Kernels**: Tile-level operations (NPU/GPU primitives)
- **Workloads**: Parallel/sequential task compositions
- **Schedules**: Dispatch policies, stream allocation, timing control
- **Programs**: Compiled workloads ready for execution

### Hello World

```python
from pto_wsp import (
    kernel, workload, P,
    Dense, Tensor, DType,
    tl, In, Out, Tile,
)

# 1. Define a kernel with tile operations
F32 = DType.F32

@kernel
def add_kernel(a: In[Tile[32, 32, F32]], b: In[Tile[32, 32, F32]], c: Out[Tile[32, 32, F32]]):
    x = tl.load(a)
    y = tl.load(b)
    z = tl.add(x, y)
    tl.store(c, z)

# 2. Define axes (iteration dimensions)
batch = Dense[4]()

# 3. Define a workload using the kernel
@workload
def simple_add():
    for b in P(batch):
        add_kernel[b](a=A[b], b=B[b], c=C[b])

# 4. Compile and execute
program = simple_add().compile()
program.execute()
program.synchronize()

print(f"Executed {program.stats.task_count} tasks")
```

---

## Defining Kernels

### The @kernel Decorator

Kernels are NPU/GPU tile operations defined with the `@kernel` decorator:

```python
from pto_wsp import kernel, tl, In, Out, InOut, Tile, Scalar, DType

F16 = DType.F16
F32 = DType.F32

@kernel
def matmul_tile(
    a: In[Tile[128, 64, F16]],     # Input tile
    b: In[Tile[64, 128, F16]],     # Input tile
    c: Out[Tile[128, 128, F32]],   # Output tile
):
    """Tiled matrix multiplication."""
    a_data = tl.load(a)
    b_data = tl.load(b)
    result = tl.matmul(a_data, b_data)
    tl.store(c, result)
```

### Parameter Annotations

| Annotation | Description |
|------------|-------------|
| `In[Tile[M, N, dtype]]` | Read-only tile input |
| `Out[Tile[M, N, dtype]]` | Write-only tile output |
| `InOut[Tile[M, N, dtype]]` | Read-write tile |
| `Scalar[dtype]` | Scalar parameter |
| `Constexpr` | Compile-time constant |

### Tile Language Primitives (`tl.*`)

```python
# Memory operations
data = tl.load(ptr)        # Load from memory
tl.store(ptr, data)        # Store to memory

# Arithmetic
z = tl.add(x, y)           # Element-wise add
z = tl.sub(x, y)           # Element-wise subtract
z = tl.mul(x, y)           # Element-wise multiply
z = tl.div(x, y)           # Element-wise divide

# Matrix operations
c = tl.matmul(a, b)        # Matrix multiplication

# Reductions
s = tl.rowsum(x)           # Sum across rows
m = tl.rowmax(x)           # Max across rows
mean = tl.rowmean(x)       # Mean across rows

# Elementwise functions
y = tl.exp(x)              # Exponential
y = tl.rsqrt(x)            # Reciprocal square root
y = tl.silu(x)             # SiLU activation
y = tl.softmax(x)          # Softmax
```

### Kernel Tracing

Inspect the generated IR:

```python
@kernel
def rmsnorm(x: In[Tile[32, 128, F16]], out: Out[Tile[32, 128, F16]]):
    data = tl.load(x)
    sq = tl.mul(data, data)
    mean = tl.rowmean(sq)
    rsqrt_val = tl.rsqrt(mean)
    result = tl.mul(data, rsqrt_val)
    tl.store(out, result)

# Trace kernel to inspect IR
ir = rmsnorm.trace()
print(f"Operations: {len(ir.ops)}")
for op in ir.ops:
    print(f"  {op}")
```

---

## Creating Workloads

### The @workload Decorator and P Namespace

Workloads define parallel task structures using the `P` namespace for loop iteration:

```python
from pto_wsp import workload, P, Dense, DenseDyn

# Static axis (size known at definition)
heads = Dense[8]()

# Dynamic axis (size known at runtime)
batch = DenseDyn(4)

@workload
def attention():
    for b in P(batch):
        for h in P(heads):
            attention_kernel[b, h](Q=Q[b][h], K=K[b], V=V[b], O=O[b][h])
```

### Axis Types

| Type | Description | Example |
|------|-------------|---------|
| `Dense[N]` | Static size | `Dense[8]()` |
| `DenseDyn` | Runtime size | `DenseDyn(n)` |
| `Ragged` | Variable per-row | `Ragged(offsets)` |
| `Sparse` | Sparse indices | `Sparse(indices)` |

### Nested Loops

```python
@workload
def tiled_matmul():
    for bm in P(tile_m):
        for bn in P(tile_n):
            for bk in P(tile_k):
                matmul_tile[bm, bn, bk](
                    a=A[bm][bk],
                    b=B[bk][bn],
                    c=C[bm][bn]
                )
```

### Combining Workloads

```python
from pto_wsp import combine, sequential

# Independent workloads (can run in parallel)
combined = combine(workload_a(), workload_b())

# Sequential workloads (must run in order)
pipeline = sequential(load_stage(), compute_stage(), store_stage())
```

### Task Enumeration

Enumerate tasks before execution:

```python
w = attention()
tasks = w.enumerate()

print(f"Total tasks: {len(tasks)}")
for task in tasks[:5]:
    print(f"  Kernel: {task.kernel}, Batch: {task.get('b')}, Head: {task.get('h')}")
```

---

## Scheduling and Execution

### Combinator-Style Scheduling

Apply scheduling policies using method chaining:

```python
from pto_wsp import DispatchPolicy, TimingPolicy

program = (attention()
    .dispatch(DispatchPolicy.round_robin(4))   # 4 executors
    .streams(2)                                 # 2 concurrent streams
    .timing(TimingPolicy.immediate)             # Execute ASAP
    .compile())
```

### Dispatch Policies

```python
# Round-robin across N executors
DispatchPolicy.round_robin(4)

# Affinity-based (tasks with same key go to same executor)
DispatchPolicy.affinity(lambda t: t.get("batch"))

# Hash-based distribution
DispatchPolicy.hash(lambda t: (t.get("batch"), t.get("head")))

# Work-stealing for dynamic load balancing
DispatchPolicy.work_steal()
```

### Timing Policies

```python
# Execute immediately when ready
TimingPolicy.immediate

# Batch tasks before execution
TimingPolicy.batched(batch_size=16)

# Interleave with other streams
TimingPolicy.interleaved(priority=1)
```

### Task Graph Mode

For fine-grained dependency control:

```python
from pto_wsp import Deps, DepsMode, ReadyPolicy, Pools

program = (attention()
    .task_graph(
        deps=Deps(mode=DepsMode.tensor_map_exact),  # Exact tensor dependency
        ready=ReadyPolicy.work_steal(),              # Work-stealing scheduler
        pools=Pools.by_exec_unit()                   # Per-unit task pools
    )
    .compile())
```

### Execution

```python
# Compile
program = workload().compile()

# Execute (non-blocking)
program.execute()

# Wait for completion
program.synchronize()

# Check statistics
print(f"Tasks: {program.stats.task_count}")
print(f"Compile time: {program.stats.compile_time_ms:.2f}ms")
print(f"Execute time: {program.stats.execute_time_ms:.2f}ms")
```

---

## Pipeline Parallelism with CSP

### Channels and Processes

```python
from pto_wsp import Channel, process, send, consume, connect

# Create channels
load_to_compute = Channel(capacity=4)
compute_to_store = Channel(capacity=4)

# Define processes
loader = (process("loader")
    .produces(load_to_compute)
    .body(for_each(tiles, lambda t:
        send(load_to_compute, task("load", [t], [...]))))
)

computer = (process("computer")
    .consumes(load_to_compute)
    .produces(compute_to_store)
    .body(consume(load_to_compute, lambda t:
        send(compute_to_store, task("compute", [t], [...]))))
)

storer = (process("storer")
    .consumes(compute_to_store)
    .body(consume(compute_to_store, lambda t:
        task("store", [t], [...])))
)

# Connect processes into pipeline
pipeline = connect([loader, computer, storer], [load_to_compute, compute_to_store])
```

### Event Synchronization

```python
from pto_wsp import Event, record, synchronize, query

# Create event
completion = Event()

# Record event after operation
record(completion, after=compute_task)

# Check if event completed (non-blocking)
if query(completion):
    print("Compute done!")

# Wait for event (blocking)
synchronize(completion)
```

---

## Profiling and Tracing

### Enable Tracing

```python
from pto_wsp import TraceLevel

program = workload().compile()
program.enable_tracing(TraceLevel.TIMING)  # or FULL for all details

program.execute()
program.synchronize()

# Print summary
program.trace.print_summary()
```

### Trace Levels

| Level | Description |
|-------|-------------|
| `TraceLevel.NONE` | No tracing (default) |
| `TraceLevel.SUMMARY` | Summary statistics only |
| `TraceLevel.TIMING` | Per-task timing |
| `TraceLevel.FULL` | Full trace with metadata |

### Analyzing Traces

```python
# Get trace events
task_events = program.trace.get_task_events()
kernel_events = program.trace.get_kernel_events("matmul_tile")

# Get statistics
summary = program.trace.summary()
print(f"Total kernel time: {summary['kernel_time_ms']:.2f}ms")
print(f"Avg task time: {summary['avg_task_time_us']:.2f}us")
```

### Exporting for Visualization

Export to Chrome Tracing format for use in `chrome://tracing` or Perfetto:

```python
import json

chrome_trace = program.trace.to_chrome_trace()
with open("trace.json", "w") as f:
    json.dump(chrome_trace, f)

# Open chrome://tracing and load trace.json
```

---

## Advanced Topics

### Tensor Layouts

```python
from pto_wsp import Tensor, TensorLayout, TensorShard, relayout

# Create sharded tensor
A = Tensor(
    shape=(4096, 4096),
    dtype=DType.F16,
    layout=TensorLayout.sharded(dim=0, rank=2, mesh_axis=0)
)

# Layout transitions
B = relayout(A, TensorLayout.replicated(rank=2))  # Shard → Replicate
```

### Collective Operations

```python
from pto_wsp import allreduce, allgather, reduce_scatter

# All-reduce across mesh
result = allreduce(tensor, op="sum", mesh_axis=0)

# All-gather sharded tensor
gathered = allgather(sharded_tensor, mesh_axis=0)

# Reduce-scatter
scattered = reduce_scatter(tensor, op="sum", mesh_axis=0)
```

### IR Passes

```python
from pto_wsp import PassManager, FlattenCombinePass, PrintPass

pm = PassManager()
pm.add(PrintPass("before"))
pm.add(FlattenCombinePass())
pm.add(PrintPass("after"))

transformed = pm.run(workload(), verbose=True)
```

### Custom Passes

```python
from pto_wsp import Pass, Workload

class MyOptimizationPass(Pass):
    def run(self, workload: Workload) -> Workload:
        # Transform workload
        return transformed_workload

pm.add(MyOptimizationPass())
```

---

## Next Steps

- See `examples/` for complete working examples
- Read `docs/spec.md` for detailed API specification
- Check `docs/analysis.md` for design rationale
- Explore `docs/design/` for architecture details

## Getting Help

- GitHub Issues: Report bugs and request features
- Documentation: `docs/` directory
- Examples: `examples/` directory with runnable code
