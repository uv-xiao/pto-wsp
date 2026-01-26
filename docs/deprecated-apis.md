# Deprecated APIs and Migration Guide

This document lists deprecated APIs in PTO-RT v9 and provides migration guidance.

## Overview

PTO-RT v9 introduced several new APIs while deprecating older patterns:
- JIT Kernel API (`@kernel` with `tl.*`) replaces string-based NPU builder
- Unified `@kernel` decorator replaces separate `@jit_kernel`
- Task graph configuration replaces extended schedule primitives
- Canonical layout types replace duplicate implementations

## Deprecated APIs

### 1. NPU Function Builder (`npu()`)

**Status**: DEPRECATED since v9

**Old Pattern** (deprecated):
```python
from pto_wsp import npu

rmsnorm_kernel = (npu("rmsnorm")
    .tile("x", 32, 128, F16)
    .tile("out", 32, 128, F16)
    .scalar("eps", F32)
    .body("""
        sq = mul(x, x)
        mean = rowmean(sq)
        rsqrt_val = rsqrt(add(mean, eps))
        out = mul(x, rsqrt_val)
    """)
    .build())
```

**New Pattern** (recommended):
```python
from pto_wsp import kernel, tl, In, Out, Tile, Scalar

@kernel
def rmsnorm(x: In[Tile[32, 128, F16]], out: Out[Tile[32, 128, F16]], eps: Scalar[F32] = 1e-6):
    data = tl.load(x)
    sq = tl.mul(data, data)
    mean = tl.rowmean(sq)
    rsqrt_val = tl.rsqrt(mean)  # Note: eps handling in tl.rsqrt
    result = tl.mul(data, rsqrt_val)
    tl.store(out, result)
```

**Migration**:
1. Replace `npu("name")` with `@kernel` decorator
2. Convert `.tile()`, `.scalar()` declarations to type annotations
3. Convert string body to `tl.*` operations
4. String refs like `"x"` become typed `Value` objects

---

### 2. Separate `@jit_kernel` Decorator

**Status**: MERGED into `@kernel` since v9

**Old Pattern**:
```python
from pto_wsp import jit_kernel, tl

@jit_kernel
def my_kernel(a: In[Tile], b: Out[Tile]):
    data = tl.load(a)
    tl.store(b, data)
```

**New Pattern** (recommended):
```python
from pto_wsp import kernel, tl

@kernel
def my_kernel(a: In[Tile], b: Out[Tile]):
    data = tl.load(a)
    tl.store(b, data)
```

**Migration**:
- Simply replace `@jit_kernel` with `@kernel`
- `@jit_kernel` is kept as an alias for backward compatibility

---

### 3. `register_kernel()` Function

**Status**: DEPRECATED

**Old Pattern** (deprecated):
```python
from pto_wsp import register_kernel

register_kernel("my_kernel", my_impl)
```

**New Pattern** (recommended):
```python
from pto_wsp import kernel

@kernel
def my_kernel(a, b):
    # Implementation with tl.* operations
    pass

# CPU simulation uses pto-isa backend automatically
```

**Migration**:
- Define kernels with `@kernel` decorator
- CPU simulation handled by pto-isa backend
- No separate registration needed

---

### 4. Extended Schedule Primitives

**Status**: DEPRECATED (L10)

**Old Patterns** (deprecated):
```python
workload.pipeline_depth(2)
workload.task_window(8192)
workload.batch_deps(128)
```

**New Pattern** (recommended):
```python
from pto_wsp import TaskWindow, Deps, DepsMode

workload.task_graph(
    window=TaskWindow(max_outstanding=8192, mode=WindowMode.stall),
    deps=Deps(mode=DepsMode.tensor_map_exact),
)
```

**Migration**:
- `pipeline_depth(n)` → `task_graph(ready=ReadyPolicy.fifo())`
- `task_window(n)` → `task_graph(window=TaskWindow(max_outstanding=n))`
- `batch_deps(n)` → `task_graph(deps=Deps(batch_size=n))`

---

### 5. `spatial.Shard` and `spatial.Replicate`

**Status**: DEPRECATED

**Old Pattern** (deprecated):
```python
from pto_wsp import Shard, Replicate  # from spatial.py

layout = Shard(dim=0)
```

**New Pattern** (recommended):
```python
from pto_wsp import TensorShard, TensorReplicate, TensorLayout

# For tensor layout specification
layout = TensorLayout.sharded(dim=0, rank=2, mesh_axis=0)

# Or for type checking
from pto_wsp import LayoutShard, LayoutReplicate, Layout

layout = Layout.sharded(dim=0, rank=2, mesh_axis=0)
```

**Migration**:
- `spatial.Shard(dim=n)` → `TensorShard(mesh_axis=n)` (note semantic change)
- `spatial.Replicate()` → `TensorReplicate()`
- For layout composition, use `TensorLayout` or `Layout`

**Important**: `spatial.Shard.dim` refers to tensor dimension, while
`TensorShard.mesh_axis` refers to mesh axis. These have different semantics.

---

### 6. `Workload.layout()` Method

**Status**: DEPRECATED

**Old Pattern** (deprecated):
```python
workload.layout(tensor_name, layout_spec)
```

**New Pattern** (recommended):
```python
from pto_wsp import Tensor, TensorLayout

# Specify layout at tensor creation
tensor = Tensor(
    data=data,
    shape=(4, 4),
    dtype=DType.F16,
    layout=TensorLayout.sharded(dim=0, rank=2)
)
```

**Migration**:
- Specify layout when creating Tensor objects
- Use `relayout()`, `allreduce()`, `allgather()`, `reduce_scatter()` for transitions

---

## API Status Summary

| API | Status | Replacement |
|-----|--------|-------------|
| `npu()` builder | Deprecated | `@kernel` with `tl.*` |
| `@jit_kernel` | Alias | `@kernel` (same functionality) |
| `register_kernel()` | Deprecated | `@kernel` decorator |
| `pipeline_depth()` | Deprecated (L10) | `.task_graph(...)` |
| `task_window()` | Deprecated (L10) | `.task_graph(window=...)` |
| `batch_deps()` | Deprecated (L10) | `.task_graph(deps=...)` |
| `spatial.Shard` | Deprecated | `TensorShard` or `LayoutShard` |
| `spatial.Replicate` | Deprecated | `TensorReplicate` or `LayoutReplicate` |
| `Workload.layout()` | Deprecated | Layout at Tensor creation |

## Current Recommended APIs

### Kernel Definition
```python
from pto_wsp import kernel, tl, In, Out, Tile, Scalar

@kernel
def my_kernel(a: In[Tile[M, N, F16]], b: Out[Tile[M, N, F16]]):
    data = tl.load(a)
    result = tl.mul(data, data)
    tl.store(b, result)
```

### Workload Definition
```python
from pto_wsp import workload, P, Dense, DenseDyn

batch = DenseDyn(4)
heads = Dense[8]()

@workload
def attention():
    for b in P(batch):
        for h in P(heads):
            my_kernel[b, h](a=Q[b][h], b=O[b][h])
```

### Schedule Configuration
```python
program = (attention()
    .dispatch(DispatchPolicy.round_robin(4))
    .streams(2)
    .timing(TimingPolicy.immediate)
    .compile())
```

### Task Graph Mode
```python
from pto_wsp import Deps, DepsMode, ReadyPolicy, Pools

program = (attention()
    .task_graph(
        deps=Deps(mode=DepsMode.tensor_map_exact),
        ready=ReadyPolicy.work_steal(),
        pools=Pools.by_exec_unit()
    )
    .compile())
```

### Layout Types
```python
from pto_wsp import Tensor, TensorLayout, TensorShard, TensorReplicate

# Replicated tensor
tensor = Tensor(data=None, shape=(4, 4), dtype=DType.F16)

# Sharded tensor
tensor = Tensor(
    data=None,
    shape=(4, 4),
    dtype=DType.F16,
    layout=TensorLayout.sharded(dim=0, rank=2, mesh_axis=0)
)
```

## Version History

- **v9**: Introduced unified `@kernel`, deprecated NPU builder, extended schedule primitives
- **v8**: Introduced Linear Layout, consolidated layout types
- **v7**: Introduced task graph API
