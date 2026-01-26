# PTO Workload-Schedule Programming (PTO-WSP): Conceptual Analysis (v7.1 - TVM Series Enhanced)

## Executive Summary

v7.1 enhances the **Workload-Schedule** separation paradigm with insights from the TVM compiler ecosystem (TVM, SparseTIR, Relax). Key additions:

1. **Axis-Based Task Specification** (from SparseTIR): Express task iteration spaces using composable axes
2. **First-Class Symbolic Shapes** (from Relax): Track dynamic dimensions symbolically across transformations
3. **Task Fusion by Category** (from TVM): Automatic fusion based on operator patterns
4. **Three-Stage Compilation** (from SparseTIR): Coordinate → Position → Code lowering
5. **Cross-Level Abstraction** (from Relax): Unified representation for workload, schedule, and kernels

## 1. Enhanced Workload Specification

### 1.1 Axis-Based Iteration Space (From SparseTIR)

SparseTIR introduces **axes** with two orthogonal attributes:
- **Dense/Sparse**: Whether indices are contiguous
- **Fixed/Variable**: Whether the count is known at compile time

**v7.1 adopts this for task iteration:**

```cpp
// Axis types for task spaces
namespace Axis {
    // Dense-Fixed: regular dimension (batch, heads)
    template<int N>
    struct DenseFixed { static constexpr int size = N; };

    // Dense-Variable: ragged dimension (sequence lengths)
    struct DenseVariable {
        int* lengths;  // Size per outer element
    };

    // Sparse-Variable: CSR-like (MoE routing)
    struct SparseVariable {
        int* indptr;   // [n+1] row pointers
        int* indices;  // [nnz] column indices
    };

    // Sparse-Fixed: ELL-like (fixed experts per token)
    template<int NNZ>
    struct SparseFixed {
        int* indices;  // [n * NNZ] indices
    };
}

// Task iteration with axes
Workload moe = task_iter(
    {Axis::DenseFixed<BATCH>{},
     Axis::SparseVariable{routing_indptr, routing_indices}},
    "SR",  // Spatial, Reduction
    [](auto batch, auto expert) {
        return task(expert_kernels[expert], {batch, expert});
    }
);
```

**Benefits:**
- Express MoE routing naturally as sparse iteration
- Reuse auxiliary data (indptr, indices) across workloads
- Enable format-aware scheduling

### 1.2 First-Class Symbolic Shapes (From Relax)

Track dynamic dimensions symbolically throughout the IR:

```cpp
// Symbolic expression type
class SymExpr {
public:
    static SymExpr var(const char* name);
    static SymExpr constant(int64_t v);

    SymExpr operator+(SymExpr other);
    SymExpr operator*(SymExpr other);
    SymExpr operator/(SymExpr other);

    // Symbolic comparison (returns SymBool)
    SymBool operator<(SymExpr other);
    SymBool operator==(SymExpr other);
};

// Tensor with symbolic shape
struct SymTensor {
    vector<SymExpr> shape;
    DType dtype;
};

// Workload with symbolic shapes
auto batch = SymExpr::var("batch");
auto seq_len = SymExpr::var("seq_len");
auto heads = SymExpr::var("heads");

Workload attn = parallel_for(range(batch), [&](auto b) {
    return parallel_for(range(heads), [&](auto h) {
        return task(attention_kernel,
            {Q[b][h], K[b], V[b]},  // Inputs
            SymTensor{{seq_len}, f16}  // Output shape tracked
        );
    });
});
```

**Benefits:**
- Shape relations preserved across transformations
- Enable symbolic memory planning
- Support data-dependent shapes via `match_cast`

### 1.3 Dataflow Blocks (From Relax)

Mark pure regions for safe optimization:

```cpp
// Dataflow block: side-effect-free region
Workload w = dataflow([&]() {
    auto t1 = task(gemm, {A, B});
    auto t2 = task(relu, {t1});
    auto t3 = task(gemm, {t2, C});
    return t3;
});

// Inside dataflow blocks:
// - Safe to reorder tasks
// - Dead code elimination
// - Automatic fusion
```

### 1.4 match_cast for Data-Dependent Shapes (From Relax)

Handle shapes only known at runtime:

```cpp
Workload moe = [&]() {
    // Routing result has data-dependent shape
    auto routed = task(router, {tokens});  // Returns variable-size result

    // Assert shape with new symbolic variable
    auto m = SymExpr::var("num_selected");
    auto routed_cast = match_cast(routed, SymTensor{{batch, m}, i32});

    // Now can iterate over m
    return parallel_for(range(m), [&](auto e) {
        return task(expert_kernel, {routed_cast[e]});
    });
};
```

## 2. Enhanced Schedule Primitives

### 2.1 Task Fusion by Category (From TVM)

TVM classifies operators for fusion. v7.1 applies this to tasks:

```cpp
enum class TaskCategory {
    INJECTIVE,           // 1-to-1 element-wise (relu, add)
    REDUCTION,           // Many-to-1 (sum, softmax)
    COMPLEX_OUT_FUSABLE, // Can fuse output (gemm, conv)
    OPAQUE               // Cannot fuse (custom kernels)
};

// Automatic category annotation via kernel analysis
Schedule sched = workload
    .auto_annotate()              // Analyze kernels → categories
    .fuse_by_category()           // Fuse injective chains
    .dispatch(round_robin(4));
```

**Fusion Rules:**
| Producer | Consumer | Fusable? |
|----------|----------|----------|
| Injective | Injective | ✓ |
| Any | Injective | ✓ (at output) |
| Complex-out | Injective | ✓ |
| Reduction | Injective | ✓ (at output) |
| Any | Reduction | ✗ |
| Any | Opaque | ✗ |

### 2.2 Additional Schedule Primitives (From TVM)

```cpp
Schedule sched = workload
    // TVM-inspired primitives
    .tile(batch, 4)                    // Tile iteration space
    .reorder(batch_outer, head, batch_inner)  // Reorder axes
    .fuse(batch_outer, head)           // Fuse for dispatch

    // Existing primitives
    .dispatch(round_robin(4))
    .issue(batch_affinity())
    .pipeline(2);
```

### 2.3 Format Decomposition (From SparseTIR)

Decompose workloads into composable formats for load balancing:

```cpp
// hyb(c, k) format: partition + bucket for load balance
Schedule sched = workload
    .decompose(TaskFormat::hyb(
        column_partition = 4,    // Partition by expert groups
        bucket_factor = 3        // Bucket by task count
    ))
    .dispatch(bucket_aware_dispatch(4))
    .issue(bucket_affinity());

// Effect: Tasks grouped by similar sizes
// - Bucket 0: 1-2 tasks/token
// - Bucket 1: 3-4 tasks/token
// - Bucket 2: 5-8 tasks/token
// Each bucket scheduled separately for better load balance
```

### 2.4 Partial Lowering (From Relax)

Make incremental dispatch decisions:

```cpp
Schedule sched = workload
    // Partial lowering: some ops to library, others to generated code
    .partial_lower()
        .to_library("cublas.gemm", pattern::gemm())
        .to_library("flashattn", pattern::attention())
        .remaining_to_tir()

    // Continue scheduling
    .dispatch(round_robin(4));
```

## 3. Three-Stage Compilation (From SparseTIR)

### 3.1 Stage Overview

```
┌─────────────────────────────────────────────────────────────────┐
│ Stage I: Coordinate Space (Workload Specification)              │
│   - Logical task iteration (batch, head, expert)                │
│   - Sparse iterations with coordinates                          │
│   - Format decomposition, axis transformations                  │
└─────────────────────────────────────────────────────────────────┘
                              │ Task Iteration Lowering
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ Stage II: Position Space (Schedule Application)                 │
│   - Physical task enumeration (positions in arrays)             │
│   - Loop structures with task queues                            │
│   - Dispatch/issue primitives, parallelization                  │
└─────────────────────────────────────────────────────────────────┘
                              │ Code Generation
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ Stage III: AICPU Binary                                         │
│   - JIT-compiled executable                                     │
│   - No high-level constructs                                    │
│   - Direct hardware interaction                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Stage I: Coordinate Space

High-level task specification with logical coordinates:

```cpp
// Coordinate space: logical task iteration
Workload moe = task_iter(
    {batch_axis, expert_axis},  // Axes define iteration space
    [](auto b, auto e) {
        return task(expert_kernels[e], {tokens[b], weights[e]});
    }
);

// Stage I transformations
Workload transformed = moe
    .decompose(TaskFormat::hyb(4, 3))  // Format decomposition
    .sparse_reorder({expert, batch})   // Reorder sparse axes
    .sparse_fuse({batch, expert});     // Fuse for single loop
```

### 3.3 Stage II: Position Space

After lowering, work with physical positions:

```cpp
// Position space: physical task enumeration
// After task_iter lowering:
for (int b_pos = 0; b_pos < batch; b_pos++) {
    for (int e_pos = expert_indptr[b_pos]; e_pos < expert_indptr[b_pos+1]; e_pos++) {
        int e = expert_indices[e_pos];  // Coordinate from position
        enqueue_task(expert_kernels[e], {tokens[b_pos], weights[e]});
    }
}

// Stage II transformations (schedule primitives)
Schedule sched = pos_workload
    .fuse(b_pos, e_pos)        // Fuse loops
    .bind(fused, "aicpu")      // Bind to AICPU threads
    .pipeline(2);              // Enable pipelining
```

### 3.4 Stage III: AICPU Binary

Final compiled code:

```cpp
// Generated AICPU binary (pseudo-assembly)
define @runtime_main() {
    %id = call @get_aicpu_id()
    %start = mul %id, %tasks_per_cpu
    %end = add %start, %tasks_per_cpu

    for %pos = %start to %end {
        %task = call @decode_task(%pos)
        call @issue_to_aicore(%task)
    }
}
```

## 4. Cross-Level Abstraction (From Relax)

### 4.1 Unified Representation

Keep workload, schedule, and kernel in one IR:

```cpp
// Cross-level IR
Module m = {
    // Workload level
    Workload attn = parallel_for(batch, [](b) {
        return task(attention_tir, {Q[b], K[b], V[b]});
    }),

    // Kernel level (TensorIR)
    TensorIR attention_tir = @tensorir {
        for i, j, k in grid(seq_len, heads, dim):
            with block():
                // Attention computation
    },

    // External library
    ExternalFunc cublas_gemm = "cublas.gemm"
};
```

### 4.2 Cross-Level Transformations

Joint optimization across levels:

```cpp
// Workspace lifting: move allocation from kernel to workload
Module transformed = m
    .lift_workspace(attention_tir)  // Workspace visible at workload level
    .fuse_kernels(gemm1, gemm2)     // Fuse kernel implementations
    .inline_small_kernels();        // Inline small kernels into caller
```

### 4.3 Analysis Feedback

Use kernel analysis to inform workload optimization:

```cpp
// Analyze kernels to determine task properties
Module analyzed = m.analyze_kernels();

// Access analysis results at workload level
for (auto& task : workload.tasks()) {
    TaskCategory cat = analyzed.get_category(task.kernel);
    AccessPattern pat = analyzed.get_access_pattern(task.kernel);

    // Use for fusion/scheduling decisions
    if (cat == INJECTIVE) {
        // Safe to fuse with neighbors
    }
}
```

## 5. Enhanced CSP Model

### 5.1 Symbolic Time

Extend CSP with symbolic time for analysis:

```cpp
// Process with symbolic time tracking
Process scheduler = Process::create([&](auto& ctx) {
    while (true) {
        select(ctx)
            .recv(tasks, [&](Task t) {
                ctx.time.advance(task_gen_latency);  // Symbolic
                queues[t.batch % NUM].push(t);
            })
            .timeout(100_ns, [&]() {
                ctx.time.advance(100_ns);
            })
            .wait();
    }
});

// Analyze for deadlock, latency bounds
Analysis result = analyze_csp(scheduler, generator, issuer);
assert(result.no_deadlock());
assert(result.max_latency() < 10_ms);
```

### 5.2 Channel Types with Shape

Channels carry tensors with symbolic shapes:

```cpp
// Typed channel with shape
Channel<SymTensor> task_results = Channel::create<SymTensor>(capacity);

// Shape preserved through channel
auto result = task_results.recv();  // result.shape is symbolic
```

## 6. Memory Planning Enhancements

### 6.1 Symbolic Memory Planning (From Relax)

Plan memory using symbolic shape analysis:

```cpp
// Symbolic memory planning
MemoryPlan plan = workload
    .analyze_shapes()           // Deduce all symbolic shapes
    .compute_liveness()         // Liveness analysis with symbols
    .allocate_buffers();        // Allocate with symbolic sizes

// Result: buffers with symbolic sizes
// Buffer A: shape = (batch, seq_len, 128)
// Buffer B: shape = (batch, seq_len, 128)  // Same shape!
// → Can share storage when lifetimes don't overlap
```

### 6.2 Workspace Sharing

Share workspace across tasks:

```cpp
// Lift workspace to workload level
Workload w_lifted = workload.lift_workspace();

// Workspace now visible for global planning
Schedule sched = w_lifted
    .share_workspace(task_group1, task_group2)  // Share between groups
    .dispatch(round_robin(4));
```

## 7. Complete Enhanced Example

### 7.1 MoE with All Enhancements

```cpp
// Symbolic dimensions
auto batch = SymExpr::var("batch");
auto tokens = SymExpr::var("tokens");
auto experts = SymExpr::var("num_experts");

// Axis-based iteration with sparse expert routing
Axis batch_axis = Axis::DenseVariable{seq_lens};
Axis expert_axis = Axis::SparseVariable{routing_indptr, routing_indices};

// Workload specification (Stage I: Coordinate Space)
Workload moe = dataflow([&]() {
    return task_iter(
        {batch_axis, expert_axis},
        "SR",  // Spatial, Reduction
        [&](auto b, auto e) {
            return task(expert_kernels[e],
                {tokens[b]},
                SymTensor{{hidden_dim}, f16}
            );
        }
    );
});

// Schedule specification (Stage I → II → III)
Schedule sched = moe
    // Stage I transforms
    .auto_annotate()                        // Analyze task categories
    .decompose(TaskFormat::hyb(4, 3))       // Load-balanced decomposition

    // Partial lowering
    .partial_lower()
        .to_library("cublas.gemm", pattern::gemm())

    // Stage II transforms
    .dispatch(DispatchPolicy::dynamic())
    .issue(IssueOrder::priority([](Task& t) {
        return -expert_load[t.expert_idx()];
    }))
    .colocate(expert_axis)
    .pipeline(3)
    .steal();

// Compile (Stage III)
CompiledProgram prog = Compiler::compile(sched, {
    .opt_level = OptLevel::O3,
    .enable_symbolic_planning = true,
    .enable_workspace_sharing = true
});

// Execute
prog.bind("batch", batch_size);
prog.bind("tokens", token_count);
prog.bind("routing_indptr", routing_indptr);
prog.bind("routing_indices", routing_indices);
prog.execute(&ctx);
```

## 8. Summary of Enhancements

| Source | Feature | v7.1 Addition |
|--------|---------|---------------|
| SparseTIR | Axes | `Axis::Dense/Sparse × Fixed/Variable` |
| SparseTIR | Format decomposition | `TaskFormat::hyb()`, `.decompose()` |
| SparseTIR | Three-stage IR | Coordinate → Position → Binary |
| SparseTIR | Sparse iterations | `task_iter()` with sparse axes |
| Relax | Symbolic shapes | `SymExpr`, `SymTensor` |
| Relax | Dataflow blocks | `dataflow([&](){...})` |
| Relax | match_cast | `match_cast(tensor, shape)` |
| Relax | Partial lowering | `.partial_lower().to_library()` |
| Relax | Analysis feedback | `.auto_annotate()` |
| Relax | Cross-level | Unified workload-kernel IR |
| TVM | Fusion categories | `TaskCategory`, `.fuse_by_category()` |
| TVM | Schedule primitives | `.tile()`, `.reorder()`, `.fuse()` |
| TVM | Memory scopes | Workspace lifting and sharing |

## 9. References

- [TVM Research Note](research/08_tvm.md)
- [SparseTIR Research Note](research/09_sparsetir.md)
- [Relax Research Note](research/10_relax.md)
- [v7 Analysis](analysis.md)
- [v7 Specification](spec.md)

---
*Version: 7.1*
*Last Updated: 2025-01-17*
