# Research Note 9: SparseTIR - Composable Abstractions for Sparse Compilation

## Overview

SparseTIR is a sparse tensor compilation abstraction that offers **composable formats** and **composable transformations** for deep learning workloads. Key insight: a single sparse format cannot maximize hardware efficiency, so decompose into multiple formats for different parts.

**Source**: SparseTIR: Composable Abstractions for Sparse Compilation in Deep Learning (ASPLOS 2023)

## 1. System Architecture

### 1.1 Three-Stage IR Design

```
┌─────────────────────────────────────────────────────────────────┐
│ Stage I: Coordinate-Space Computation                           │
│   - Sparse iterations over non-zero elements                    │
│   - Access sparse buffers in coordinate space                   │
│   - Format decomposition, sparse_reorder, sparse_fuse           │
└─────────────────────────────────────────────────────────────────┘
                              │ Sparse Iteration Lowering
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ Stage II: Position-Space Computation                            │
│   - Loop structures with sparse buffers                         │
│   - Access buffers in position space (non-zero index)           │
│   - Loop manipulation, memory hierarchy, parallelization        │
└─────────────────────────────────────────────────────────────────┘
                              │ Sparse Buffer Lowering
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ Stage III: Loop-Level IR                                        │
│   - Compatible with TensorIR/TVM                                │
│   - Flattened buffers, no sparse constructs                     │
│   - Target-specific code generation                             │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 Key Insight: Composability

**Format Composability**: Store different parts of a sparse matrix in different formats that best fit their local patterns.

```
Single Format:                Composable Formats:
┌─────────────┐              ┌─────────────┐
│ A_csr       │      →       │ A_bsr + A_ell│
│ (one format)│              │ (decomposed) │
└─────────────┘              └─────────────┘

Y = A_csr * X        →       Y = A_bsr * X + A_ell * X
```

**Transformation Composability**: Instead of single-shot compilation, apply transformations step-by-step and stage-by-stage.

## 2. Language Constructs

### 2.1 Axes

Axes define sparse iteration spaces with two orthogonal attributes:

| Attribute | Options | Description |
|-----------|---------|-------------|
| **dense/sparse** | dense, sparse | Whether indices are contiguous |
| **fixed/variable** | fixed, variable | Whether non-zero count is fixed |

```python
# Dense-Fixed: regular dimension
I = dense_fixed(m, "int32")

# Sparse-Variable: CSR-like
J = sparse_variable(I, (n, nnz), (j_indptr, j_indices), "int32")

# Sparse-Fixed: ELL-like (fixed nnz per row)
J2 = sparse_fixed(I, (n, nnz_cols), indices, "int32")
```

**Axis Dependencies**: Variable axes depend on parent axis (e.g., `J` depends on `I` for CSR).

### 2.2 Sparse Buffers

```python
# Sparse buffer declaration
A = match_sparse_buffer(a, (I, J), "float32")
B = match_sparse_buffer(b, (J_, K), "float32")
C = match_sparse_buffer(c, (I, K), "float32")
```

**Key Design**: Axes store auxiliary data (indptr, indices), buffers store only values. Two buffers can share auxiliary data if they have the same sparse layout.

### 2.3 Sparse Iterations

```python
# SpMM: C[i,k] = sum_j(A[i,j] * B[j,k])
with sp_iter([I, J, K], "SRS", "spmm") as [i, j, k]:
    with init():
        C[i, k] = 0.0
    C[i, k] = C[i, k] + A[i, j] * B[j, k]
```

- `"SRS"`: S=Spatial, R=Reduction
- Supports affine indices: `A[i * m + j, k]`
- Supports indirect access: `B[eid[i], j * n + k]`
- Supports nested sparse iterations

## 3. Stage I: Coordinate Space

### 3.1 Format Decomposition

Transform computation from single format to composable formats:

```python
# Original: CSR format
with sp_iter([I, J, K], "SRS", "spmm") as [i, j, k]:
    C[i, k] = C[i, k] + A[i, j] * B[j, k]

# After decompose to BSR(2) + ELL(2):
# Generated: Copy to BSR
with sp_iter([IO, II, JO, JI], "SSSS", "copy_bsr") as [io, ii, jo, ji]:
    A_bsr[io, jo, ii, ji] = A[io * 2 + ii, jo * 2 + ji]

# Generated: SpMM on BSR
with sp_iter([IO, II, JO, JI, K], "SSSSR", "spmm_bsr") as [io, ii, jo, ji, k]:
    C[io * 2 + ii, k] = C[io * 2 + ii, k] + A_bsr[io, jo, ii, ji] * B[jo * 2 + ji, k]

# Generated: SpMM on ELL (for remainder)
with sp_iter([I2, J2, K], "SRS", "spmm_ell") as [i, j, k]:
    C[i, k] = C[i, k] + A_ell[i, j] * B[j, k]
```

### 3.2 Stage I Schedules

| Primitive | Effect | Example |
|-----------|--------|---------|
| `sparse_reorder` | Change sparse axis order | `sparse_reorder([K, I, J])` |
| `sparse_fuse` | Fuse iterators into one | `sparse_fuse([I, J])` for SDDMM |

## 4. Stage II: Position Space

### 4.1 Sparse Iteration Lowering

Four steps to transform Stage I → Stage II:

1. **Auxiliary Buffer Materialization**: Create buffers for indptr, indices
2. **Nested Loop Generation**: One loop per axis in sparse iteration
3. **Coordinate Translation**: Rewrite indices from coordinate to position space
4. **Read/Write Region Analysis**: Collect buffer access information

```python
# Stage I (coordinate space)
with sp_iter([I, J, K], "SRS", "spmm") as [i, j, k]:
    C[i, k] = C[i, k] + A[i, j] * B[j, k]

# Stage II (position space)
for i in range(m):
    with block("spmm0"):
        for j in range(0, J_indptr[i + 1] - J_indptr[i]):
            for k in range(feat_size):
                with block("spmm1"):
                    C[i, k] = C[i, k] + A[i, j] * B[J_indices[i, j], k]
                    #                              ↑ coordinate translation
```

### 4.2 Stage II Schedules

Reuse TVM/TensorIR schedule primitives:

| Category | Primitives |
|----------|-----------|
| Loop manipulation | `fuse`, `reorder`, `split` |
| Memory hierarchy | `cache_read`, `cache_write` |
| Parallelization | `bind` to threads |
| Vectorization | `vectorize`, `tensorize` |

## 5. Stage III: Loop-Level IR

### 5.1 Sparse Buffer Lowering

Remove all SparseTIR constructs, flatten buffers:

```python
# Stage II
C[i, k] = C[i, k] + A[i, j] * B[J_indices[i, j], k]

# Stage III (flattened)
C[i * feat_size + k] = C[i * feat_size + k] +
    A[J_indptr[i] + j] * B[J_indices[J_indptr[i] + j] * feat_size + k]
```

## 6. Composable Formats for Deep Learning

### 6.1 hyb(c, k) Format for SpMM

Parameterized composable format for load balancing:

```
┌─────────────────────────────────────────────────────────────────┐
│ hyb(c, k) Format:                                                │
│   1. Partition columns by factor c                               │
│   2. For each partition, bucket rows by length                   │
│      - Bucket i: rows with 2^(i-1) < length ≤ 2^i               │
│   3. Pad to ELL format within each bucket                        │
│   4. Map each 2^(k-i) rows to one thread block                   │
└─────────────────────────────────────────────────────────────────┘
```

**Benefits**:
- Compile-time load balancing (no runtime overhead)
- Better cache locality (column partitioning)
- 1.2-2.3x speedup over cuSPARSE

### 6.2 SR-BCRS Format for Pruned Transformers

For unstructured sparse matrices:

```
┌─────────────────────────────────────────────────────────────────┐
│ SR-BCRS(t, g) Format:                                            │
│   1. Divide into t×1 tiles, omit all-zero tiles                  │
│   2. Group non-zero tiles by factor g within rows                │
│   3. Pad tailing groups with zero tiles                          │
│   - Lower bound density: 1/t (vs 1/b² for BSR(b))               │
└─────────────────────────────────────────────────────────────────┘
```

### 6.3 RGMS (Relational Gather-Matmul-Scatter)

For heterogeneous graphs (RGCN) and sparse convolution:

```
Y[i,l] = Σ_r Σ_j Σ_k A[r,i,j] * X[j,k] * W[r,k,l]
```

**Schedule**:
1. For each relation r: pin W^r in SRAM
2. Gather related rows of X from HBM to SRAM
3. Compute with Tensor Cores
4. Scatter results to Y

## 7. Key Insights for PTO Workload-Schedule Programming (PTO-WSP)

### 7.1 Three-Stage Analogy

SparseTIR's three-stage IR maps to task graph compilation:

| SparseTIR Stage | v7 Analogy | Description |
|-----------------|------------|-------------|
| Stage I: Coordinate Space | **Workload Specification** | High-level task composition |
| Stage II: Position Space | **Schedule Application** | Task dispatch/issue patterns |
| Stage III: Loop-Level | **AICPU Binary** | JIT-compiled execution code |

### 7.2 Composable Task Formats

Just as SparseTIR decomposes sparse matrices into composable formats, we can decompose task graphs:

```cpp
// Composable task formats for MoE
TaskFormat expert_format = TaskFormat::hyb(
    column_partition = 4,    // Partition by expert groups
    bucket_factor = 3        // Bucket by task count
);

// Decompose workload
Workload moe_decomposed = moe_workload.decompose(expert_format);
// Generates: high-load tasks → one format
//            medium-load tasks → another format
//            low-load tasks → ELL-like format
```

### 7.3 Axis-Based Task Space

Adopt SparseTIR's axis abstraction for task iteration spaces:

```cpp
// Dense-fixed axis: batch dimension
Axis batch = Axis::dense_fixed(batch_size);

// Sparse-variable axis: MoE expert assignment
Axis expert = Axis::sparse_variable(batch, max_experts, expert_indptr, expert_indices);

// Task iteration
Workload moe = task_iter({batch, expert}, [](auto b, auto e) {
    return task(expert_kernel[e], {tokens[b], weights[e]});
});
```

### 7.4 Two-Space Model

Separate **coordinate space** (logical) from **position space** (physical):

```cpp
// Coordinate space: logical task specification
Workload attn = parallel_for(batch, b =>
    parallel_for(head, h =>
        task(attention_kernel, {b, h, seq_lens[b]})
    )
);

// Position space: physical task enumeration
// After lowering:
for (int b_pos = 0; b_pos < batch; b_pos++) {
    for (int h_pos = 0; h_pos < head; h_pos++) {
        int b = b_pos;  // coordinate = position for dense
        int h = h_pos;
        issue_task(attention_kernel, {b, h, seq_lens[b]});
    }
}

// For sparse (MoE):
for (int b_pos = 0; b_pos < batch; b_pos++) {
    for (int e_pos = expert_indptr[b]; e_pos < expert_indptr[b+1]; e_pos++) {
        int e = expert_indices[e_pos];  // coordinate ≠ position
        issue_task(expert_kernel[e], {tokens[b], weights[e]});
    }
}
```

### 7.5 Format Decomposition as Schedule

Make format decomposition a schedule primitive:

```cpp
Schedule sched = workload
    .decompose(TaskFormat::hyb(4, 3))  // Decompose task graph
    .dispatch(load_balanced(4))         // Dispatch to 4 AICPUs
    .issue(affinity_by(bucket));        // Same bucket → same core
```

### 7.6 Sparse Buffer Sharing

Allow multiple workloads to share auxiliary structures:

```cpp
// Shared MoE routing structure
MoERouting routing = MoERouting::create(routing_indptr, routing_indices);

// Multiple workloads share routing
Workload ffn1 = moe_task_iter(routing, expert_ffn1);
Workload ffn2 = moe_task_iter(routing, expert_ffn2);

// Fuse workloads that share structure
Workload fused = ffn1.fuse_with(ffn2, shared_routing=true);
```

## 8. Specific Improvements for v7

### 8.1 Add Axis-Based Task Specification

```cpp
// Axis definitions for task spaces
namespace Axis {
    // Dense-fixed: regular dimension
    template<int N>
    struct DenseFixed { static constexpr int size = N; };

    // Dense-variable: ragged dimension
    struct DenseVariable {
        int* indptr;  // Points to start of each row
    };

    // Sparse-variable: CSR-like
    struct SparseVariable {
        int* indptr;
        int* indices;
    };

    // Sparse-fixed: ELL-like
    template<int NNZ>
    struct SparseFixed {
        int* indices;
    };
}

// Task iteration with axes
Workload w = task_iter(
    {Axis::DenseFixed<BATCH>{},
     Axis::SparseVariable{routing_indptr, routing_indices}},
    "SR",  // Spatial, Reduction
    [](auto batch, auto expert) {
        return task(expert_kernels[expert], {batch, expert});
    }
);
```

### 8.2 Add Format Decomposition

```cpp
// Format decomposition rules
struct FormatDecomposeRule {
    string name;
    AxisTransform axis_transform;  // Old axes → New axes
    IndexMap forward_map;          // Old indices → New indices
    IndexMap inverse_map;          // New indices → Old indices
};

// Built-in rules
namespace DecomposeRules {
    auto BSR(int block_size);      // Block Sparse Row
    auto ELL(int nnz_per_row);     // ELLPACK
    auto Hyb(int col_part, int bucket_factor);  // Hybrid
}

// Apply decomposition
Workload decomposed = workload.decompose({
    DecomposeRules::BSR(4),
    DecomposeRules::ELL(2)
});
```

### 8.3 Add Coordinate/Position Space Lowering

```cpp
// Three-stage compilation
Workload workload = ...;  // Stage I: Coordinate space

// Stage I → Stage II: Sparse iteration lowering
PositionWorkload pos_workload = workload.lower_to_position_space();

// Stage II schedules
Schedule sched = pos_workload
    .fuse(batch, head)
    .bind(fused, "aicpu")
    .cache_read(weights, UB)
    .vectorize(feature, 16);

// Stage II → Stage III: Compile to AICPU binary
CompiledProgram prog = Compiler::compile(sched);
```

### 8.4 Add Task-Level Load Balancing

```cpp
// Bucketing for load balance (inspired by hyb format)
Schedule sched = workload
    .bucket_by(task_size, {1, 2, 4, 8, 16})  // Bucket tasks by size
    .dispatch(bucket_aware_round_robin(4))   // Load-balanced dispatch
    .issue(bucket_affinity());               // Same bucket → same core
```

## 9. Summary

| SparseTIR Concept | v7 Application |
|-------------------|----------------|
| Axes (dense/sparse × fixed/variable) | Task space specification |
| Sparse buffers | Shared routing structures |
| Sparse iterations | Task iteration primitives |
| Format decomposition | Task graph decomposition |
| Coordinate/position space | Logical/physical task enumeration |
| Three-stage lowering | Workload → Schedule → AICPU binary |
| Composable formats | Composable task formats (hyb, etc.) |
| Stage I schedules | Workload transformations |
| Stage II schedules | Schedule primitives |
| Tensorization | Task-to-kernel binding |

## References

- [SparseTIR Paper (ASPLOS 2023)](https://doi.org/10.1145/3582016.3582047)
- [SparseTIR Code](https://github.com/uwsampl/sparsetir)
- [TVM Research Note](08_tvm.md)
- [TACO Format Abstraction](https://doi.org/10.1145/3276493)

---
*Version: 1.0*
*Last Updated: 2025-01-17*
