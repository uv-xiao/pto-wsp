# Research Note 3: Programming Language Design for Domain-Specific Accelerators

## Overview

This note surveys programming language design principles from compiler research, focusing on abstractions that separate algorithm from schedule. These principles inform how PTO-ISA can be extended to support dynamic workloads while maintaining clean abstractions.

## 1. Halide: Algorithm-Schedule Separation

### 1.1 Core Principle

Halide's fundamental insight: **decouple WHAT to compute from HOW to compute it**.

```cpp
// Algorithm: WHAT to compute
Func blur(Func input) {
    Func blur_x, blur_y;
    blur_x(x, y) = (input(x-1, y) + input(x, y) + input(x+1, y)) / 3;
    blur_y(x, y) = (blur_x(x, y-1) + blur_x(x, y) + blur_x(x, y+1)) / 3;
    return blur_y;
}

// Schedule: HOW to compute (can be changed independently)
blur_y.tile(x, y, xi, yi, 256, 32)
      .vectorize(xi, 8)
      .parallel(y);
blur_x.compute_at(blur_y, x)
      .vectorize(x, 8);
```

### 1.2 Schedule Primitives

| Primitive | Effect |
|-----------|--------|
| `split(x, xo, xi, factor)` | Split loop into outer/inner |
| `fuse(x, y, xy)` | Fuse two loops |
| `reorder(x, y, z)` | Change loop nesting order |
| `parallel(x)` | Parallelize loop |
| `vectorize(x, width)` | Vectorize loop |
| `unroll(x)` | Unroll loop |
| `compute_at(f, x)` | Compute producer at consumer's loop |
| `store_at(f, x)` | Store producer at consumer's loop |

### 1.3 Key Design Properties

1. **Composability**: Schedules compose—applying one doesn't prevent others
2. **Correctness preservation**: Any valid schedule produces correct results
3. **Hardware independence**: Same algorithm, different schedules for CPU/GPU/DSP
4. **Searchability**: Schedule space can be automatically explored

### 1.4 Relevance to PTO-ISA

PTO-ISA partially embeds scheduling in the algorithm:
```cpp
// Current: Tiling is part of algorithm
TileData src0Tile(kTRows_, kTCols_);  // Tile shape specified
TLOAD(src0Tile, src0Global);
```

**Potential separation:**
```cpp
// Algorithm (what)
TLOAD(dst, src);
TMATMUL(c, a, b);

// Schedule (how) - separate
SCHEDULE(matmul_sched) {
    TILE(M, 128);
    TILE(K, 64);
    REORDER(M_outer, K_outer, M_inner, K_inner);
    DOUBLE_BUFFER(K_outer);
}
```

## 2. TVM/TensorIR: Block-Based Tensor Programs

### 2.1 Core Abstraction: Blocks

TensorIR introduces **blocks** as isolation boundaries:

```python
@T.prim_func
def matmul(A: T.Buffer((M, K), "float32"),
           B: T.Buffer((K, N), "float32"),
           C: T.Buffer((M, N), "float32")):
    for i, j, k in T.grid(M, N, K):
        with T.block("matmul"):
            vi, vj, vk = T.axis.remap("SSR", [i, j, k])
            with T.init():
                C[vi, vj] = 0.0
            C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]
```

### 2.2 Block Properties

| Property | Description |
|----------|-------------|
| **Spatial axes (S)** | Independent iteration (parallelizable) |
| **Reduction axes (R)** | Dependent iteration (requires synchronization) |
| **Init region** | Initialization before reduction |
| **Body region** | Main computation |
| **Read/Write buffers** | Explicit data dependencies |

### 2.3 Schedule Primitives in TensorIR

```python
sch = tvm.tir.Schedule(matmul)
block = sch.get_block("matmul")

# Split loops
i, j, k = sch.get_loops(block)
i_outer, i_inner = sch.split(i, factors=[None, 32])
j_outer, j_inner = sch.split(j, factors=[None, 32])

# Reorder
sch.reorder(i_outer, j_outer, k, i_inner, j_inner)

# Bind to hardware
sch.bind(i_outer, "blockIdx.x")
sch.bind(j_outer, "blockIdx.y")

# Cache read/write
A_shared = sch.cache_read(block, 0, "shared")
sch.compute_at(A_shared, k)
```

### 2.4 Relevance to PTO-ISA

TensorIR's block concept aligns with PTO-ISA's tiles:

| TensorIR | PTO-ISA |
|----------|---------|
| Block | Tile operation |
| Spatial axis | Independent tile dimension |
| Reduction axis | Accumulation dimension |
| Cache read | TLOAD to buffer |
| Cache write | TSTORE from buffer |

**Key difference**: TensorIR supports **dynamic axis bounds** via symbolic shapes.

## 3. Polyhedral Model Basics

### 3.1 Core Concepts

The polyhedral model represents loop nests as:
1. **Iteration domain**: Set of integer points (polyhedron)
2. **Access functions**: Affine mappings from iterations to array elements
3. **Schedule**: Affine mapping from iterations to time

### 3.2 Example

```cpp
for (i = 0; i < N; i++)
    for (j = 0; j < M; j++)
        A[i][j] = B[i-1][j] + B[i][j-1];

// Iteration domain: {(i,j) : 0 ≤ i < N ∧ 0 ≤ j < M}
// Access: A[i,j], B[i-1,j], B[i,j-1]
// Dependencies: (i,j) depends on (i-1,j) and (i,j-1)
```

### 3.3 Transformations

| Transformation | Polyhedral | Effect |
|----------------|------------|--------|
| Loop tiling | Affine schedule change | Cache blocking |
| Loop skewing | Shear transformation | Expose parallelism |
| Loop fusion | Domain union | Reduce memory traffic |
| Loop interchange | Coordinate swap | Change access pattern |

### 3.4 Limitations

- Only handles **static affine** bounds and accesses
- Struggles with **data-dependent** control flow
- Compilation time grows with problem size

### 3.5 Relevance to PTO-ISA

Polyhedral analysis could optimize static PTO-ISA programs, but:
- Dynamic LLM workloads have **non-affine** bounds (seq_len varies)
- Need **hybrid approach**: polyhedral for static parts, runtime for dynamic

## 4. Design Patterns for Dynamic Workloads

### 4.1 Pattern: Symbolic Shapes

TVM's approach: Use **symbolic integers** for dynamic dimensions:

```python
M = T.var("int32")  # Symbolic, known at runtime
N = 1024            # Static, known at compile time

A = T.alloc_buffer([M, N])  # Mixed static/dynamic
```

### 4.2 Pattern: Multi-Version Compilation

Compile **multiple versions** for different shape ranges:

```cpp
// Compile-time: Generate variants
kernel_v1 = compile(seq_len <= 1024);
kernel_v2 = compile(seq_len <= 4096);
kernel_v3 = compile(seq_len <= 16384);

// Runtime: Select variant
kernel = select_kernel(actual_seq_len);
```

### 4.3 Pattern: Descriptor-Guided Execution

Separate **static kernel** from **dynamic parameters**:

```cpp
// Kernel: Static structure
void attention_kernel(WorkDescriptor* desc) {
    // Bounds come from descriptor, not control flow
    for (pos = desc->start; pos < desc->end; pos++) {
        // ...
    }
}

// Dispatch: Dynamic descriptor generation
for (req : requests) {
    emit_descriptor({.start = 0, .end = req.seq_len});
}
```

## 5. Synthesis: Design Principles for PTO-ISA Extension

### 5.1 Principle 1: Separate Algorithm from Schedule

```cpp
// Algorithm (what) - declarative
auto attn = MatMul(Q, K.T()) >> Softmax() >> MatMul(V);

// Schedule (how) - separate concern
Schedule sched;
sched.tile(seq_dim, 1024);
sched.double_buffer(kv_load);
sched.parallel(batch_dim);
```

### 5.2 Principle 2: Explicit Iteration Spaces

```cpp
// Define iteration space with mixed static/dynamic
IterationSpace space;
space.add_axis("batch", 0, batch_size);           // Static
space.add_axis("seq", 0, DYNAMIC);                // Dynamic
space.add_axis("head", 0, num_heads);             // Static
```

### 5.3 Principle 3: Bounds as Data, Not Control

```cpp
// Not this: Control flow determines bounds
if (seq_len <= 1024) { kernel_1k(); }
else if (seq_len <= 4096) { kernel_4k(); }

// This: Bounds in descriptor
struct Descriptor {
    uint32_t seq_start;
    uint32_t seq_end;   // Actual bound
    uint32_t tier;      // Which kernel variant
};
```

### 5.4 Principle 4: Composable Schedule Primitives

```cpp
// Each primitive is independent
schedule.tile(axis, size);        // Doesn't prevent others
schedule.parallel(axis);          // Composes with tile
schedule.prefetch(buffer, depth); // Composes with parallel
```

### 5.5 Principle 5: Hardware-Independent Algorithm

```cpp
// Same algorithm works on:
// - CPU simulation (for debugging)
// - A2/A3 NPU (current target)
// - A5 NPU (future target)
// - Different tile sizes per platform

Algorithm alg = DefineAttention(...);
alg.compile(Target::CPU);   // Different schedule
alg.compile(Target::A3);    // Different schedule
alg.compile(Target::A5);    // Different schedule
```

## 6. Key Takeaways

### 6.1 What to Adopt

| Concept | Source | Application to PTO-ISA |
|---------|--------|------------------------|
| Algorithm-Schedule separation | Halide | Decouple tile shapes from operations |
| Block abstraction | TensorIR | Align with existing Tile concept |
| Symbolic shapes | TVM | Support dynamic dimensions |
| Multi-version compilation | Common | Tier-based kernel selection |
| Descriptor-guided execution | FlashInfer | Bounds as data, not control |

### 6.2 Design Space

```
        Static ◄────────────────────────────► Dynamic

Halide: ████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░
        (Static shapes, static schedule)

TVM:    ████████████████████░░░░░░░░░░░░░░░░░░░
        (Symbolic shapes, static schedule)

PTO-ISA:████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░
Current (Static tiles, compile-time shapes)

Target: ████████████████████████████░░░░░░░░░░░
        (Dynamic bounds, descriptor-guided)
```

## References

- [Halide: Decoupling Algorithms from Schedules (CACM 2018)](https://cacm.acm.org/magazines/2018/1/223877-halide/)
- [TensorIR: An Abstraction for Automatic Tensorized Program Optimization](https://arxiv.org/pdf/2207.04296)
- [TVM Tensor Program Abstraction Documentation](https://tvm.apache.org/docs/deep_dive/tensor_ir/abstraction.html)
- [Learning to Optimize Halide with Tree Search](https://halide-lang.org/papers/halide_autoscheduler_2019.pdf)
- [Meta Schedule RFC](https://github.com/apache/tvm-rfcs/blob/main/rfcs/0005-meta-schedule-autotensorir.md)

---
*Note Version: 1.0*
*Last Updated: 2024-01-13*
