# Concise Workload Design for PTO-WSP v9

## Overview

This document addresses user requirements for:
1. **Concise workload syntax** - TileLang-inspired brevity
2. **CSP as part of workload** - unified hierarchy (v8 decision)
3. **Hierarchical workload-schedule organization** - CPU/NPU workload together
4. **Backend-specific schedule applicability** - mechanism for unsupported primitives

---

## 1. Concise Workload Syntax

### 1.1 Design Principles

- **Symbolic `for` loops** instead of context managers (TileLang-style)
- **Single namespace `P`** for all loop constructors (parallel, sequential, grid)
- **Direct kernel calls** via `kernel[axes](...)` instead of `task("name", ...)`
- **Explicit parallelism semantics** in loop constructors

### 1.2 Proposed Concise API

```python
from pto_wsp import workload, P, kernel, Tensor, In, Out

# P = Parallel/Pipeline namespace (like TileLang's T)
# P(...) = parallel grid, P.seq(...) = sequential, P.pipe(...) = pipeline

@kernel
def attn(Q: In[Tensor], K: In[Tensor], V: In[Tensor], O: Out[Tensor]): ...

@workload
def attention(batch, heads):
    for b, h in P(batch, heads):    # Parallel grid iteration
        attn[b, h](Q[b,h], K[b], V[b], O[b,h])
```

### 1.3 Loop Constructors

| Constructor | Semantics | Dependency |
|-------------|-----------|------------|
| `P(axes...)` | Parallel grid | Independent |
| `P.seq(axis)` | Sequential iteration | Sequential |
| `P.pipe(stages)` | Pipeline stages | Pipeline |
| `P.sel(pred, cases)` | Selection | Varies |
| `P.when(pred)` | Conditional | Context-dependent |

### 1.4 Comparison: Before vs After

**Before (verbose context manager):**
```python
with workload() as w:
    with parallel_for(batch) as b:
        with parallel_for(heads) as h:
            task("attn_kernel", [b, h], [Q[b][h], K[b], V[b], O[b][h]])
attention = w.finish()
```

**After (concise symbolic loop):**
```python
@workload
def attention(batch, heads):
    for b, h in P(batch, heads):
        attn[b, h](Q[b,h], K[b], V[b], O[b,h])
```

---

## 2. CSP as Part of Workload

### 2.1 Design Decision (from v8)

CSP primitives (`Channel`, `Process`, `send`, `consume`) are **workload nodes**, not a separate category. A `Pipeline` is a specialized `Workload` that connects processes via channels.

### 2.2 Unified Workload Hierarchy

```
Workload
├── Parallel primitives: P(...), parallel_for
├── Sequential primitives: P.seq(...), for_each, sequential
├── Selection primitives: P.sel(...), select, cond
├── CSP primitives: P.pipe(...), Channel, send, recv
└── Leaf: kernel[axes](...)
```

### 2.3 CSP Concise Syntax

```python
@workload
def pipeline(tiles):
    ch = Channel[Tile](depth=2)

    with P.pipe():                    # Pipeline scope
        for i in P.seq(tiles):        # Producer (sequential)
            send(ch, load[i](data))

        for tile in recv(ch):         # Consumer (channel-driven)
            compute(tile)
```

---

## 3. Hierarchical Workload-Schedule Organization

### 3.1 Proposed Spec Structure

```
1. Overview
2. Type System (Axis, Tensor, Layout)
3. Workload Primitives (unified)
   3.1 Loop Constructors (P namespace)
   3.2 Parallel Primitives
   3.3 Sequential Primitives
   3.4 Selection Primitives
   3.5 CSP Primitives (channels, processes, pipelines)
   3.6 Barrier and Sync
4. Kernel Definition
   4.1 @kernel decorator
   4.2 Using kernels in workloads
   4.3 External kernels (NPU/C++)
5. Schedule API
   5.1 Dispatch (where tasks run)
   5.2 Issue (how tasks are released)
       - streams()
       - task_graph()
   5.3 Spatial (dataflow mapping)
   5.4 Backend Applicability
6. Compilation and Execution
7. Complete Examples
8. Error Handling
```

### 3.2 Key Changes

1. **Move CSP** from Section 4 into Section 3 (Workload Primitives)
2. **Move Kernel Definition** before Schedule (kernels define what, schedule defines how)
3. **Add Backend Applicability** section under Schedule
4. **Consolidate** three syntax styles into one recommended: `@workload` + symbolic loops

---

## 4. Backend-Specific Schedule Applicability

### 4.1 Problem

Different backends support different schedule primitives:
- `spatial_map()` - only AMD AIE
- `streams()` - CPU sim, Ascend NPU
- `task_graph()` - all backends
- `pipeline_depth()` - different semantics per backend

### 4.2 Solution: Backend Capabilities

```python
class BackendCapabilities:
    """Declares what schedule primitives a backend supports."""

    supported_dispatch: set[DispatchPolicy]
    supported_issue: set[IssuePolicy]  # streams, task_graph
    supported_spatial: bool

    def check(self, schedule: Schedule) -> list[Warning | Error]: ...
```

### 4.3 Backend Capability Table

| Primitive | CPU Sim | Ascend NPU | AMD AIE |
|-----------|---------|------------|---------|
| `dispatch(round_robin)` | ✓ | ✓ | ✗ |
| `dispatch(work_steal)` | ✓ | ✓ | ✗ |
| `streams()` | ✓ | ✓ | ✗ |
| `task_graph()` | ✓ | ✓ | ✓ |
| `spatial_map()` | ✗ | ✗ | ✓ |
| `pipeline_depth()` | ✓ (tasks) | ✓ (tasks) | ✓ (FIFO) |

### 4.4 Compile-Time Checking

```python
program = (workload
    .dispatch(DispatchPolicy.work_steal())  # Not supported on AIE
    .spatial_map(grid=(4, 4))               # Only AIE
    .compile(target="amd_aie"))             # Error at compile time
```

**Error:**
```
ScheduleError: dispatch(work_steal) not supported for target 'amd_aie'
  Hint: AMD AIE uses spatial_map() for task placement, not dispatch policies
```

### 4.5 Backend-Specific Extensions

```python
# Backend extensions via qualified names
from pto.rt.backend.ascend import double_buffer, prefetch
from pto.rt.backend.aie import tile_placement, dma_channel

program = (workload
    .dispatch(...)
    .streams(2)
    .extend(double_buffer(scope="L1"))  # Ascend-specific
    .compile(target="ascend_npu"))
```

---

## 5. Implementation Recommendations

1. **Phase 1**: Update `docs/spec.md` with new structure
   - Move CSP under Workload Primitives
   - Consolidate syntax styles to `@workload` + `P` namespace
   - Add Backend Applicability section

2. **Phase 2**: Update `docs/analysis.md`
   - Reflect concise syntax decision
   - Document backend capability model

3. **Phase 3**: Update Python implementation
   - Add `P` namespace with `__iter__` for symbolic loops
   - Add `BackendCapabilities` checking at compile time

---

## 6. Rationale

### Why `@workload` decorator?
- More concise than context manager + `w.finish()`
- Returns `Workload` directly
- Aligns with `@kernel` for consistency

### Why `P` namespace?
- Single letter like TileLang's `T`
- Mnemonic: **P**arallel/**P**ipeline
- Contains all loop constructors for discoverability

### Why unify CSP under Workload?
- CSP was always part of workload model (v7/v8)
- `Pipeline` is just a `Workload` with channel dependencies
- Simplifies mental model: one section for "what to compute"

### Why backend capabilities?
- Clear error messages at compile time
- Documents what each backend supports
- Enables backend-specific extensions cleanly
