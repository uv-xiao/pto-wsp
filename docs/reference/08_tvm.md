# Research Note 8: TVM - End-to-End Deep Learning Compiler

## Overview

TVM is an automated end-to-end optimizing compiler for deep learning that provides performance portability across diverse hardware back-ends (CPUs, GPUs, FPGAs, accelerators). It introduces **tensor expression language** with **schedule-based optimization** and **ML-based auto-tuning**.

**Source**: TVM: An Automated End-to-End Optimizing Compiler for Deep Learning (OSDI 2018)

## 1. System Architecture

### 1.1 Two-Level IR Design

TVM uses a hierarchical IR:

```
┌─────────────────────────────────────────────────────────────────┐
│                     High-Level: Computational Graph              │
│  - Nodes = operators (conv2d, relu, matmul)                     │
│  - Edges = data dependencies (tensors)                          │
│  - Optimizations: fusion, constant folding, layout transform    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Low-Level: Tensor Expression                 │
│  - Declarative tensor computation                               │
│  - Schedule-based optimization                                  │
│  - Code generation to LLVM/CUDA/etc                            │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 Key Insight

> "Decouple what to compute from how to compute it"

This is the Halide principle applied to deep learning with extensions for:
- GPU-specific optimizations (shared memory, thread cooperation)
- Accelerator-specific features (tensorization, latency hiding)
- Automated schedule search (ML-based cost model)

## 2. Tensor Expression Language

### 2.1 Core Concept

Tensor expressions describe **what** to compute without specifying **how**:

```python
# Tensor expression for matrix multiplication
m, n, h = t.var('m'), t.var('n'), t.var('h')
A = t.placeholder((m, h), name='A')
B = t.placeholder((n, h), name='B')
k = t.reduce_axis((0, h), name='k')
C = t.compute((m, n), lambda y, x: t.sum(A[k, y] * B[k, x], axis=k))
#              ↑ result shape       ↑ computing rule
```

### 2.2 Properties

| Property | Description |
|----------|-------------|
| **Declarative** | Specifies result, not implementation |
| **Index-based** | Uses lambda functions with indices |
| **Composable** | Expressions can be chained |
| **Hardware-agnostic** | Same expression works on CPU/GPU/accelerator |

### 2.3 Operations Supported

- Element-wise: `compute((m, n), lambda i, j: A[i,j] + B[i,j])`
- Reduction: `compute((m,), lambda i: t.sum(A[i, k], axis=k))`
- Stencil: `compute((m, n), lambda i, j: A[i-1,j] + A[i,j] + A[i+1,j])`
- Broadcasting: Implicit shape inference

## 3. Schedule Primitives

### 3.1 From Halide (Reused)

| Primitive | Effect | Example |
|-----------|--------|---------|
| `split(x, factor)` | Split loop into outer/inner | `xo, xi = s.split(x, 32)` |
| `fuse(x, y)` | Fuse two loops | `xy = s.fuse(x, y)` |
| `reorder(...)` | Change loop order | `s.reorder(xo, yo, xi, yi)` |
| `tile(x, y, xf, yf)` | 2D tiling | `xo, yo, xi, yi = s.tile(x, y, 32, 32)` |
| `compute_at(s, x)` | Inline computation | `s[B].compute_at(s[C], x)` |
| `parallel(x)` | Parallelize loop | `s.parallel(x)` |
| `vectorize(x)` | Vectorize loop | `s.vectorize(xi)` |
| `unroll(x)` | Unroll loop | `s.unroll(xi)` |

### 3.2 TVM Extensions

#### Special Memory Scopes (GPU)

```python
# Cooperative data loading to shared memory
AS = s.cache_read(A, "shared", [C])  # Mark as shared memory
s[AS].compute_at(s[C], ko)           # Compute at outer loop
# Compiler auto-inserts barriers
```

**Key Insight**: Memory scopes let threads cooperatively load data into shared memory, enabling cross-thread data reuse.

#### Tensorization (Hardware Intrinsics)

```python
# Declare hardware intrinsic behavior
w, x = t.placeholder((8, 8)), t.placeholder((8, 8))
k = t.reduce_axis((0, 8))
y = t.compute((8, 8), lambda i, j: t.sum(w[i, k] * x[j, k], axis=k))

# Lowering rule
def gemm_intrin_lower(inputs, outputs):
    return t.hardware_intrin("gemm8x8", inputs[0], inputs[1], outputs[0])

gemm8x8 = t.decl_tensor_intrin(y.op, gemm_intrin_lower)

# Use in schedule
s[C].tensorize(yi, gemm8x8)
```

**Key Insight**: Tensorization separates hardware intrinsic declaration from schedule, making it extensible to new accelerators.

#### Latency Hiding (Decoupled Access-Execute)

For accelerators with DAE (Decoupled Access-Execute) architecture:

```
┌─────────────────────────────────────────────────────────────────┐
│ Without Latency Hiding:                                          │
│   ld0 → ex0 → ld1 → ex1 → ld2 → ex2 → ...  (sequential)         │
│                                                                  │
│ With Latency Hiding (DAE Pipeline):                              │
│   ld0 → ld1 → ld2 → ...                                         │
│         ex0 → ex1 → ex2 → ...  (overlapped)                     │
└─────────────────────────────────────────────────────────────────┘
```

TVM uses **virtual threads** to express parallelism:
1. Write high-level threaded program
2. Compiler inserts synchronization (push_dep/pop_dep)
3. Interleave into single instruction stream
4. Hardware recovers pipeline parallelism

```python
# High-level virtual threaded code
for vthread tx in range(2):
    acc_buffer CL[8]
    for k in range(128):
        ld.dma_copy2d(AL, A[k][tx*8:tx*8+8])
        ex.accumulate(AL, CL)
```

**Result**: 70% → 88% compute utilization on FPGA accelerator.

## 4. Graph-Level Optimizations

### 4.1 Operator Fusion

TVM classifies operators into four categories:

| Category | Description | Example | Fusion Rule |
|----------|-------------|---------|-------------|
| **Injective** | 1-to-1 map | add, relu | Can fuse with each other |
| **Reduction** | Many-to-one | sum, max | Can fuse with injective inputs |
| **Complex-out-fusable** | Can fuse output | conv2d | Can fuse injective outputs |
| **Opaque** | Cannot fuse | sort | No fusion |

**Fusion Example**:
```
Before: conv2d → bn → relu (3 kernels)
After:  conv2d+bn+relu (1 fused kernel)
```

**Speedup**: 1.2× to 2× from reduced memory traffic.

### 4.2 Data Layout Transformation

Different hardware prefers different layouts:
- CPU: NCHW (batch, channel, height, width)
- GPU: NCHW or NHWC depending on operation
- Accelerator: Might require 4×4 tiled layout

TVM automatically inserts layout transformations between operators with different preferred layouts.

## 5. ML-Based Cost Model

### 5.1 The Problem

| Approach | Data Cost | Model Bias | Hardware Info | Learn from History |
|----------|-----------|------------|---------------|-------------------|
| Blackbox auto-tuning | High | None | No | No |
| Predefined cost model | None | High | Yes | No |
| **ML-based** | **Low** | **Low** | **No** | **Yes** |

### 5.2 Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│ Schedule Explorer                                                │
│   1. Propose candidate configurations                           │
│   2. Query ML model for performance prediction                  │
│   3. Run top-k on hardware                                      │
│   4. Update ML model with measurements                          │
└─────────────────────────────────────────────────────────────────┘
```

### 5.3 Feature Extraction

From lowered loop AST:
- Memory access count per buffer per loop level
- Reuse ratio per buffer
- Loop annotations (vectorize, unroll, parallel)

```
Loop AST → Feature Extraction → XGBoost → Cost Prediction
```

### 5.4 Exploration Algorithm

**Simulated Annealing**:
1. Start with random configuration
2. Random walk to nearby configuration
3. Accept if cost decreases (per ML model)
4. Likely reject if cost increases
5. Converges to low-cost configurations

**Result**: ML-based optimizer finds better configs much faster than blackbox auto-tuning.

## 6. Key Insights for PTO Workload-Schedule Programming (PTO-WSP)

### 6.1 Tensor Expression Analogy

TVM's tensor expression for **operators** is analogous to our **Workload** for **task graphs**:

| TVM | PTO-ISA v7 |
|-----|------------|
| Tensor expression | Workload specification |
| Schedule primitives | Schedule primitives (dispatch, issue, etc.) |
| Loop transformations | Task graph transformations |
| Tensorization | Task mapping to kernels |

### 6.2 Schedule Primitives to Adopt

From TVM, we can adopt analogous schedule primitives:

| TVM Primitive | v7 Analogy | Description |
|---------------|------------|-------------|
| `tile(x, factor)` | `tile(axis, size)` | Tile iteration space |
| `reorder(...)` | `reorder(axes)` | Reorder task generation |
| `compute_at` | `colocate` | Keep related tasks together |
| `parallel` | `parallel_for` | Already in workload |
| `cache_read` | `prefetch` | Already in schedule |
| `tensorize` | `bind(kernel)` | Map computation to kernel |
| `virtual_thread` | CSP processes | Latency hiding |

### 6.3 New Insights for v7

1. **Operator Fusion** → **Task Fusion**
   - Fuse multiple small tasks into one
   - Reduce dispatch overhead
   - Categories: injective, reduction, complex-out-fusable, opaque

2. **Memory Scopes** → **Buffer Scopes**
   - Explicit UB (Unified Buffer) management
   - Cross-task data sharing on AICore

3. **ML-based Schedule Search**
   - Automatically find good schedules
   - Learn from runtime measurements

4. **Data Layout** → **Tensor Layout**
   - Automatic layout transformation between tasks
   - AICore-friendly tiled layouts

### 6.4 Latency Hiding Pattern

TVM's virtual thread lowering for DAE accelerators directly applies to AICPU→AICore:

```
High-level:                    Low-level:
for vthread in range(N):       interleaved single stream with
    load(data)           →     push_dep/pop_dep for
    compute(data)              fine-grained synchronization
```

This is already supported by our CSP model with channels!

## 7. Specific Improvements for v7

### 7.1 Add Tensor Expression-like Task Specification

Instead of just referencing kernels, allow inline computation:

```cpp
// Current v7: reference kernel
Workload::task(attention_kernel, params, resources)

// Enhanced: inline tensor expression
Workload::compute({M, N}, [](i, j) {
    return sum(A[i, k] * B[k, j], k);
}, resources)
```

### 7.2 Add More Schedule Primitives

```cpp
// TVM-inspired schedule primitives
Schedule sched = workload
    .tile(batch, 4)              // Tile batch dimension
    .reorder(batch_o, head, batch_i)  // Reorder iteration
    .fuse(batch_o, head)         // Fuse for dispatch
    .dispatch(round_robin(4))
    .prefetch(2)                 // Prefetch 2 tasks ahead
    .issue(batch_affinity());
```

### 7.3 Add Task Fusion

```cpp
// Task fusion categories
enum class TaskCategory {
    INJECTIVE,           // 1-to-1: elementwise ops
    REDUCTION,           // Many-to-1: sum, softmax
    COMPLEX_OUT_FUSABLE, // Can fuse output: GEMM, conv
    OPAQUE               // Cannot fuse: custom kernels
};

// Fusion as schedule primitive
Schedule sched = workload
    .fuse_injective()    // Auto-fuse injective tasks
    .dispatch(...);
```

### 7.4 Add ML-based Schedule Search

```cpp
// Automatic schedule optimization
Schedule sched = AutoScheduler::search(
    workload,
    target,              // AICPU + AICore config
    num_trials=1000,
    cost_model=XGBoost   // or TreeRNN
);
```

## 8. Summary

| TVM Concept | v7 Application |
|-------------|----------------|
| Tensor expression | Task specification with inline compute |
| Schedule primitives | Extended with tile, reorder, fuse |
| Operator fusion | Task fusion by category |
| Memory scopes | Buffer scopes for UB management |
| Tensorization | Task-to-kernel binding |
| Virtual threads | CSP processes (already supported) |
| ML-based search | AutoScheduler for schedule optimization |
| Data layout | Tensor layout transformation |

## References

- [TVM Paper (OSDI 2018)](https://www.usenix.org/conference/osdi18/presentation/chen)
- [TVM Documentation](https://tvm.apache.org/docs/)
- [Halide Schedule Primitives](research/03_pl_design.md)
- [Megakernels Latency Hiding](research/06_megakernels.md)

---
*Version: 1.0*
*Last Updated: 2025-01-17*
