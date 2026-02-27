# Research Note 10: Relax - Composable Abstractions for End-to-End Dynamic ML

## Overview

Relax is a compiler abstraction for optimizing end-to-end dynamic machine learning workloads. Key innovations: **cross-level abstraction** that encapsulates computational graphs, tensor programs, and libraries in one representation, and **first-class symbolic shapes** for tracking dynamic shape computations globally.

**Source**: Relax: Composable Abstractions for End-to-End Dynamic Machine Learning (ASPLOS 2025)

## 1. Problem: Dynamic Shapes in ML

LLMs and modern ML models have pervasive dynamic shapes:
- Variable-sized input messages
- KV-cache context length
- Variable batch sizes
- MoE routing

**Challenge**: Traditional ML compilers handle dynamic shapes at each level within each function, but cross-level optimizations are difficult because:
1. AOT compilation needs full-program optimization across functions
2. Custom operators need graph-level optimizations aware of foreign functions
3. Single-shot lowering prevents using tensor program analysis to inform graph optimization

## 2. System Architecture

### 2.1 Cross-Level Abstraction

```
┌─────────────────────────────────────────────────────────────────┐
│                   Unified Cross-Level IR                         │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐   │
│  │ Computational   │  │ Loop-Level      │  │ External        │   │
│  │ Graph           │  │ Tensor Programs │  │ Libraries       │   │
│  │ (high-level ops)│  │ (TensorIR)      │  │ (cuBLAS, etc.)  │   │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘   │
│           │                    │                     │           │
│           └────────────────────┼─────────────────────┘           │
│                                │                                 │
│                    call_tir / call_dps_library                   │
└─────────────────────────────────────────────────────────────────┘
```

**Key Insight**: Instead of single-shot lowering between levels, keep all levels in one IR and incrementally optimize/lower.

### 2.2 Single-Shot vs Cross-Level

```
Traditional Multi-Level:           Relax Cross-Level:
┌──────────────────┐               ┌──────────────────────────────┐
│ Computational    │               │ Unified IR with:              │
│ Graph IR         │               │  - Graph operators            │
└────────┬─────────┘               │  - Tensor programs            │
         │ single-shot             │  - Library calls              │
         ▼                         │                               │
┌──────────────────┐               │ Cross-level optimizations:    │
│ Tensor Program   │               │  - Partial lowering           │
│ IR               │               │  - Analysis feedback          │
└────────┬─────────┘               │  - Cross-level transforms     │
         │ single-shot             └──────────────────────────────┘
         ▼
┌──────────────────┐
│ Libraries        │
└──────────────────┘
```

## 3. Language Constructs

### 3.1 Annotations (Types)

| Annotation | Example | Description |
|------------|---------|-------------|
| `Object` | `Object` | Any runtime value |
| `Shape` | `Shape([n, 4])` | Symbolic shape value |
| `Tensor` | `Tensor((n, 4), "f32")` | Tensor with symbolic shape |
| `Tuple` | `Tuple[Tensor, Object]` | Tuple of values |
| `Callable` | `Callable([...], ...)` | Function type with shapes |

### 3.2 Dataflow Blocks

Side-effect-free regions without control flow:

```python
def main(x: Tensor(("n", 4), "f32")):
    n = sym_var()
    with dataflow():  # Pure computation region
        lv0: Tensor((n, 4), "f32") = exp(x)
        lv1: Tensor((n, 4), "f32") = relu(lv0)
    return lv1
```

**Benefits**:
- Safe to apply graph transformations
- Dead code elimination without worrying about side effects
- Clear boundary for optimizations

### 3.3 Cross-Level Function Calls

```python
def main(x: Tensor(("n", 128), "f32"), w: Tensor((128, 256), "f32")):
    n = sym_var()
    with dataflow():
        # Call tensor program
        lv0: Tensor((n, 256), "f32") = call_tir(
            mm, [x, w], Tensor((n, 256), "f32")
        )
        # Call external library
        lv1: Tensor((n, 256), "f32") = call_dps_library(
            "cutlass.rms_norm", [lv0], Tensor((n, 256), "f32")
        )
    return lv1

@tensorir_function
def mm(X: Buffer(("n", 128), "f32"), W: Buffer((128, 256), "f32"),
       Y: Buffer(("n", 256), "f32")):
    n = sym_var()
    for i, j, k in grid(n, 256, 128):
        with block():
            with init():
                Y[i, j] = 0
            Y[i, j] += X[i, k] * W[k, j]
```

**Destination-Passing Style (DPS)**: Tensor programs take output buffer as parameter, matching low-level calling conventions while graph-level appears pure.

```python
def call_tir(tir_func, args, annotation, sym_args):
    output = alloc_tensor(annotation.shape, annotation.dtype)
    tir_func(*args, output, *sym_args)  # DPS call
    return output
```

## 4. First-Class Symbolic Shapes

### 4.1 Problem with "Unknown" Annotations

```python
# Unknown annotation approach (loses information)
def any_shape_fn(x: Tensor((?, 2, 2), "f32")):
    lv0: Tensor((?, 4), "f32") = reshape(x, ...)
    lv1: Tensor((?,), "f32") = flatten(lv0)
    # Lost: lv1 has 4x elements of x's first dim
```

```python
# First-class symbolic shape (preserves relations)
def symbolic_shape_fn(x: Tensor(("n", 2, 2), "f32")):
    n = sym_var()
    lv0: Tensor((n, 4), "f32") = reshape(x, shape(n, 4))
    lv1: Tensor((n * 4,), "f32") = flatten(lv0)
    # Known: lv1 has n*4 elements = 4*n = original total
```

### 4.2 match_cast for Data-Dependent Shapes

```python
def fn(x: Tensor(("n", 2, 2), "f32")):
    n, m = sym_var(), sym_var()
    lv0: Tensor((n * 4,), "f32") = flatten(x)
    lv1: Tensor(ndim=1, dtype="f32") = unique(lv0)  # Data-dependent!
    lv2 = match_cast(lv1, Tensor((m,), "f32"))  # Assert shape (m,)
    lv3: Tensor((m,), "f32") = exp(lv2)
    return lv3
```

### 4.3 Shape Deduction Principles

1. **Isolated symbolic relations at function boundaries**
   - Function signatures have parameter and return annotations
   - Enables first-class Callable with shape inference

2. **Forward symbolic deduction**
   - Deduce annotation based on inputs
   - Falls back to coarse-grained when specific info unavailable

3. **Support symbolic expressions in parameter annotations**
   - Function parameters can have expressions like `(2*n,)`
   - Pass extra shape parameters when needed

```python
# After operator fusion
def fused_add_relu(
    x: Tensor(("n * 2",), "f32"),  # Expression in annotation
    y: Tensor(("n * 2",), "f32"),
    s: Shape(["n"])  # Extra parameter to pass n
) -> Tensor(("n * 2",), "f32"):
    lv0 = add(x, y)
    return relu(lv0)
```

## 5. Cross-Level Optimization Patterns

### 5.1 Partial Lowering

Instead of single-shot lowering, make incremental decisions:

```
Before:  lv0 → lv1 → lv2
                ↓
After:   lv0 → lv1 → lv2
              (lib)  (tir)
```

**Benefit**: Can dispatch some operators to libraries, others to generated code.

### 5.2 Analysis Feedback

Analyze tensor programs to annotate operator properties:

```python
@tensorir_function
def relu(X: Buffer(...), Y: Buffer(...)):
    for i:
        Y[i] = max(X[i], 0)
# Analysis: ElementWise pattern

# Graph level now knows relu is element-wise without manual annotation
```

**Pattern Categories**:
- `ElementWise`: 1-to-1 mapping
- `Broadcast`: Read subset of write indices
- `Injective`: 1-to-1 but reordered
- `Reduction`: Many-to-one
- `OutputEwiseFusible`: Can fuse element-wise to output (matmul, conv)
- `Opaque`: Cannot analyze

### 5.3 Cross-Level Transforms

Joint transformation of graph and tensor program levels:

```python
# Before: Tensor program allocates workspace internally
@tensorir_function
def mm_split_k(...):
    workspace = alloc_buffer(8*1024*1024, "f32", "global")
    ...

# After: Workspace lifted to graph level
def main(...):
    workspace = alloc_tensor((8*1024*1024,), "f32")
    lv0 = call_tir(mm_split_k, [x, w, workspace], ...)

@tensorir_function
def mm_split_k(X, W, workspace, Y):  # workspace is now a parameter
    ...
```

**Benefit**: Workspace participates in global memory planning.

## 6. Dynamic Shape-Aware Optimizations

### 6.1 Operator Fusion Pipeline

```
Initial Program
    ↓ Compute pattern analysis (Algorithm 1)
    ↓ FuseOps - group into subgraph functions (Algorithm 2)
    ↓ FuseTensorIR - merge tensor programs
Fused Program
```

**Key**: All steps handle symbolic shapes by tracking variables and generating extra shape parameters.

### 6.2 Dynamic Shape-Aware Memory Planning

```
Before planning:
x → lv0 → lv1 → lv2 → lv3
    (2,n) (n,2) (n,2) (2,n)
    [alloc] [alloc] [alloc] [alloc]

After planning:
x → lv0 → lv1 → lv2 → lv3
    [s0]   [s1]   [s0]   [s1]  (reuse via symbolic equality)
```

**Algorithm**:
1. Lower `call_tir`/`call_dps_library` to explicit allocation
2. Use symbolic expression analysis to prove equality
3. Reuse storage when shapes match symbolically
4. Take upper bounds for static pre-allocation

### 6.3 CUDA Graph Offloading

**Challenge**: CUDA Graph requires static memory allocation.

**Solution**: With static memory planning, pre-allocate all memory even for dynamic shapes, then enable CUDA Graph.

## 7. Key Insights for PTO Workload-Schedule Programming (PTO-WSP)

### 7.1 Cross-Level Abstraction Analogy

Relax's cross-level abstraction maps directly to our Workload-Schedule design:

| Relax Level | v7 Analogy |
|-------------|------------|
| Computational Graph | Workload specification |
| Tensor Programs | Kernel implementations |
| External Libraries | Pre-compiled kernels |
| `call_tir` | Task invocation |
| `call_dps_library` | External kernel call |

### 7.2 First-Class Symbolic Shape for Tasks

Apply symbolic shape tracking to task parameters:

```cpp
// First-class symbolic task dimensions
Workload attn = [](Tensor(("n", "h", "d"), "f16") Q,
                   Tensor(("n", "kv_len", "d"), "f16") K,
                   Tensor(("n", "kv_len", "d"), "f16") V) {
    auto n = sym_var();
    auto kv_len = sym_var();
    return parallel_for(n, [&](auto b) {
        return task(attention_kernel, {Q[b], K[b], V[b]},
                    Tensor((kv_len,), "f16"));  // Output shape tracked
    });
};
```

### 7.3 Dataflow Blocks for Task Regions

Borrow the dataflow block concept for pure task regions:

```cpp
Workload w = [&]() {
    with_dataflow([&]() {  // Pure task region
        auto lv0 = task(kernel1, inputs);
        auto lv1 = task(kernel2, {lv0});
        return lv1;
    });
};
```

### 7.4 Destination-Passing Style for Kernels

Bridge between workload (pure) and kernel (DPS) semantics:

```cpp
// Workload level: appears pure
auto result = task(gemm_kernel, {A, B}, Tensor((M, N), dtype));

// Lowered to kernel level: DPS
// void gemm_kernel(A, B, C) { C = A @ B; }
auto result = alloc_tensor({M, N}, dtype);
call_dps(gemm_kernel, {A, B, result});
```

### 7.5 Analysis Feedback for Task Properties

Analyze kernels to determine task properties:

```cpp
// Kernel analysis pass
TaskCategory analyze_kernel(Kernel k) {
    // Analyze loop patterns in kernel
    if (is_element_wise(k)) return INJECTIVE;
    if (has_reduction(k)) return REDUCTION;
    if (is_fuse_multiply_add(k)) return OUTPUT_EWISE_FUSIBLE;
    return OPAQUE;
}

// Use in fusion decisions
Schedule sched = workload
    .auto_annotate_task_categories()  // Analysis feedback
    .fuse_by_category()               // Fuse based on properties
    .dispatch(...);
```

### 7.6 Dynamic Shape-Aware Task Dispatch

Use symbolic shape for dispatch decisions:

```cpp
Schedule sched = workload
    .symbolic_shape_aware()           // Enable symbolic tracking
    .dispatch([](TaskInfo t) {
        // Can reason about symbolic shapes
        if (t.output_shape == t.input_shape) {
            return dispatch_strategy::same_core();  // Affinity
        }
        return dispatch_strategy::load_balanced();
    });
```

### 7.7 Cross-Level Workspace Lifting

Lift task workspace to workload level for global planning:

```cpp
// Before: each task allocates workspace
Kernel k = [](auto in, auto out) {
    auto workspace = alloc_local(1024);  // Hidden allocation
    ...
};

// After: workspace lifted to workload
Workload w = [&]() {
    auto workspace = alloc_shared(1024);  // Visible at workload level
    return parallel_for(n, [&](auto i) {
        return task(k, {in[i], workspace}, out[i]);
    });
};
```

## 8. Specific Improvements for v7

### 8.1 Add Symbolic Shape to Task Specification

```cpp
// Symbolic shape variables
using SymVar = int64_t;  // Runtime: concrete int

// Symbolic expressions
struct SymExpr {
    variant<SymVar, int64_t, BinaryOp<SymExpr>> value;
    static SymExpr var(string name);
    static SymExpr constant(int64_t v);
    SymExpr operator+(SymExpr other);
    SymExpr operator*(SymExpr other);
};

// Tensor with symbolic shape
template<typename T>
struct SymTensor {
    vector<SymExpr> shape;
    string dtype;
};

// Workload with symbolic shapes
Workload attn(SymTensor<f16> Q, SymTensor<f16> K) {
    auto n = Q.shape[0];
    auto kv_len = K.shape[1];
    return parallel_for(n, [&](auto b) {
        return task(attention_kernel, {Q[b], K[b]},
                    SymTensor<f16>{{kv_len}, "f16"});
    });
}
```

### 8.2 Add Dataflow Blocks

```cpp
// Dataflow block: pure region without side effects
Workload w = dataflow([&]() {
    auto t1 = task(k1, {x});
    auto t2 = task(k2, {t1});
    return t2;
});

// Enables:
// - Safe dead code elimination
// - Reordering within block
// - Fusion across tasks
```

### 8.3 Add Foreign Function Call Primitives

```cpp
// call_tir: call loop-level tensor program
auto result = call_tir(tensorir_func, args, output_annotation);

// call_dps_library: call external library
auto result = call_dps_library("cublas.gemm", args, output_annotation);

// Unified dispatch
Schedule sched = workload
    .partial_lower_to_library("cublas.gemm", gemm_pattern)
    .remaining_to_tir()
    .dispatch(...);
```

### 8.4 Add match_cast for Data-Dependent Shapes

```cpp
// For MoE routing with unknown expert counts
Workload moe(Tensor<i32> routing) {
    auto [batch, experts] = route(inputs, routing);
    // experts has data-dependent shape

    // match_cast asserts shape with new symbolic var
    auto experts_cast = match_cast(experts, {batch, sym_var("m")});

    return parallel_for(experts_cast.shape[1], [&](auto e) {
        return task(expert_kernel, {experts_cast[e]});
    });
}
```

### 8.5 Add Shape Deduction Rules

```cpp
// Register shape deduction per operator
ShapeDeduceRule reshape_deduce = [](auto inputs, auto params) {
    return params["new_shape"];  // Shape from parameters
};

ShapeDeduceRule concat_deduce = [](auto inputs, auto params) {
    int axis = params["axis"];
    auto out_shape = inputs[0].shape;
    for (int i = 1; i < inputs.size(); i++) {
        out_shape[axis] = out_shape[axis] + inputs[i].shape[axis];
    }
    return out_shape;
};

// System deduces shapes through forward propagation
```

## 9. Summary

| Relax Concept | v7 Application |
|---------------|----------------|
| Cross-level abstraction | Unified Workload-Schedule-Kernel representation |
| First-class symbolic shape | Symbolic task dimensions |
| Dataflow blocks | Pure task regions |
| call_tir/call_dps_library | Task invocation / library dispatch |
| Destination-passing style | Bridge workload to kernel semantics |
| Annotations | Task/Tensor type with shape |
| Shape deduction | Forward symbolic shape propagation |
| match_cast | Assert shape for data-dependent dims |
| Partial lowering | Incremental dispatch decisions |
| Analysis feedback | Automatic task category annotation |
| Cross-level transforms | Joint workload-kernel optimization |
| Memory planning | Dynamic shape-aware buffer reuse |
| CUDA Graph offloading | Static execution graph for dynamic shapes |

## References

- [Relax Paper (ASPLOS 2025)](https://doi.org/10.1145/3676641.3716249)
- [Apache TVM](https://tvm.apache.org/)
- [TensorIR](research/08_tvm.md)
- [SparseTIR](research/09_sparsetir.md)

---
*Version: 1.0*
*Last Updated: 2025-01-17*
