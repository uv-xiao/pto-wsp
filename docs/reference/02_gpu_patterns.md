# Research Note 2: GPU Kernel Acceleration Patterns

## Overview

This note surveys key patterns and abstractions from modern GPU programming frameworks (CUDA, Triton, CUTLASS/CuTe) to inform PTO-ISA runtime extension design. The goal is to identify transferable principles rather than CUDA-specific techniques.

## 1. Memory Hierarchy Patterns

### 1.1 The Memory Pyramid

All GPU architectures share a similar memory hierarchy pattern:

```
                    ┌─────────────┐
                    │  Registers  │  Fastest, thread-local
                    │  (~256 KB)  │
                    ├─────────────┤
                    │   L1/Shared │  Block-shared, programmer-managed
                    │   (64-228KB)│
                    ├─────────────┤
                    │     L2      │  Device-wide, cache-managed
                    │   (6-50 MB) │
                    ├─────────────┤
                    │   Global    │  Slowest, largest
                    │  (16-80 GB) │
                    └─────────────┘
```

### 1.2 Key Optimization Patterns

| Pattern | Description | Benefit |
|---------|-------------|---------|
| **Coalesced Access** | Adjacent threads access adjacent memory | Maximize bandwidth utilization |
| **Data Reuse** | Keep data in fast memory across operations | Reduce memory traffic |
| **Prefetching** | Load data before it's needed | Hide memory latency |
| **Double Buffering** | Overlap compute and memory | Pipeline execution |

**Ascend Mapping:**
- Registers → UB (Unified Buffer)
- Shared Memory → L1 Buffer
- Global Memory → HBM/DDR

### 1.3 Memory Access Pattern Example

```cpp
// Bad: Strided access, poor coalescing
for (int i = 0; i < N; i++) {
    out[threadIdx.x * N + i] = in[threadIdx.x * N + i];
}

// Good: Coalesced access
for (int i = 0; i < N; i++) {
    out[i * blockDim.x + threadIdx.x] = in[i * blockDim.x + threadIdx.x];
}
```

## 2. Triton's Block-Level Abstraction

### 2.1 Core Idea: Blocks, Not Threads

Triton's key insight: **program at the block level, not the thread level**.

```python
# Triton kernel - operates on blocks (tiles) of data
@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Each program instance processes a BLOCK of elements
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load BLOCKS of data
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    # Compute on BLOCKS
    out = x + y

    # Store BLOCK result
    tl.store(out_ptr + offsets, out, mask=mask)
```

### 2.2 What Triton Abstracts Away

| Concern | CUDA | Triton |
|---------|------|--------|
| Thread indexing | Manual | Automatic |
| Memory coalescing | Manual | Automatic |
| Shared memory | Manual | Automatic |
| Bank conflicts | Manual | Automatic |
| Tensor core scheduling | Manual | Automatic |

### 2.3 Relevance to PTO-ISA

PTO-ISA already operates at the **tile level**, similar to Triton's blocks:
- `Tile<Type, Rows, Cols>` is the fundamental unit
- `TLOAD`, `TSTORE` operate on tiles
- Memory management is implicit

**Gap**: PTO-ISA lacks Triton's **dynamic masking** and **variable-size block** support.

## 3. CUTLASS/CuTe Layout Algebra

### 3.1 Core Concept: Hierarchical Layouts

CuTe represents data layouts as **hierarchical shape-stride pairs**:

```cpp
// Row-major 4x8 matrix
Layout layout = make_layout(Shape<_4, _8>{}, Stride<_8, _1>{});

// Equivalent to: idx = i * 8 + j

// Hierarchical: 2x(2x4) with nested strides
Layout hier = make_layout(
    Shape<_2, Shape<_2, _4>>{},
    Stride<_16, Stride<_8, _1>>{}
);
```

### 3.2 Layout Composition

CuTe's power comes from **composing** layouts:

```cpp
// Partition a layout across threads
auto thr_layout = make_layout(Shape<_4, _8>{});  // 32 threads
auto partitioned = logical_product(data_layout, thr_layout);

// Tile a layout
auto tiled = zipped_divide(data_layout, tile_shape);
```

### 3.3 Relevance to PTO-ISA

PTO-ISA has simpler layout handling:
- Fixed layouts: `BLayout::RowMajor`, `BLayout::ColMajor`
- Fractal formats for tensor cores

**Potential Enhancement**: Adopt CuTe-style **layout algebra** for flexible tiling:
```cpp
// Hypothetical PTO-ISA extension
using DataLayout = Layout<Shape<M, K>, Stride<K, 1>>;
using TileLayout = Layout<Shape<TM, TK>>;
using PartitionedLayout = TiledLayout<DataLayout, TileLayout>;
```

## 4. Kernel Fusion Patterns

### 4.1 Why Fusion Matters

```
Unfused:                          Fused:
┌────────┐                        ┌────────────────┐
│ MatMul │ → GM                   │                │
└────────┘                        │  MatMul        │
    ↓                             │    ↓ (registers)
┌────────┐                        │  Softmax       │
│Softmax │ → GM                   │    ↓ (registers)
└────────┘                        │  MatMul        │
    ↓                             │                │
┌────────┐                        └────────────────┘
│ MatMul │ → GM                          ↓
└────────┘                              GM

Memory: 3 GM round-trips           Memory: 1 GM round-trip
```

### 4.2 Fusion Strategies

| Strategy | Description | Example |
|----------|-------------|---------|
| **Vertical** | Chain of ops on same data | MatMul → BiasAdd → ReLU |
| **Horizontal** | Parallel ops on different data | Multi-head attention |
| **Block** | Ops tiled together | Flash Attention |

### 4.3 FlashAttention as Fusion Example

```
Traditional Attention:
Q×K^T → Softmax → ×V
  ↓        ↓      ↓
 (N²)     (N²)   (N²)  Total: 3N² memory

FlashAttention (Block Fusion):
for each Q_block:
    for each KV_block:
        local_attn = Q_block × K_block^T  # In registers
        update_softmax_state()             # In registers
        accumulate(local_attn × V_block)   # In registers
    write output_block                     # Single write

Total: O(N) memory
```

## 5. Work Distribution Patterns

### 5.1 Static vs Dynamic Work Assignment

| Pattern | Pros | Cons |
|---------|------|------|
| **Static Grid** | Simple, predictable | Can't handle variable sizes |
| **Persistent Threads** | Flexible | Complex synchronization |
| **Index-Based** | Handles variable sizes, simple | Requires planning phase |

### 5.2 FlashInfer's Approach (Index-Based)

```cpp
// Planning (CPU)
for (req = 0; req < batch; req++) {
    for (chunk = 0; chunk < ceil(kv_len[req] / chunk_size); chunk++) {
        request_indices[block_id] = req;
        chunk_indices[block_id] = chunk;
        block_id++;
    }
}

// Execution (GPU)
__global__ void kernel() {
    int req = request_indices[blockIdx.x];    // O(1) lookup
    int chunk = chunk_indices[blockIdx.x];    // O(1) lookup
    // Process...
}
```

### 5.3 Relevance to PTO-ISA

Current PTO-ISA assumes **static shapes at compile time**. For dynamic workloads:

**Option 1: Index Array Approach (FlashInfer-style)**
- Pros: Proven, efficient
- Cons: Requires host planning, array management

**Option 2: Descriptor-Based**
- Pros: Type-safe, structured
- Cons: More complex dispatch

**Option 3: Per-Request Kernel Launch**
- Pros: Simple
- Cons: Launch overhead, poor batching

## 6. Parallel Reduction Patterns

### 6.1 Warp-Level Primitives

```cpp
// Warp shuffle for reduction
float sum = value;
for (int offset = 16; offset > 0; offset /= 2) {
    sum += __shfl_xor_sync(0xFFFFFFFF, sum, offset);
}
// All threads in warp have same sum
```

### 6.2 Block-Level Reduction

```cpp
__shared__ float shared[32];  // One per warp

// Warp reduction first
float warp_sum = warpReduce(value);

// Store warp results
if (lane_id == 0) shared[warp_id] = warp_sum;
__syncthreads();

// Final reduction in first warp
if (warp_id == 0) {
    float val = (lane_id < num_warps) ? shared[lane_id] : 0;
    block_sum = warpReduce(val);
}
```

### 6.3 Ascend Equivalent

PTO-ISA has `TROWSUM`, `TCOLSUM`, `TROWMAX` etc. for tile-level reductions.

**Gap**: No warp-equivalent cross-thread reduction in PTO-ISA (handled by hardware).

## 7. Key Takeaways for PTO-ISA Extension

### 7.1 Patterns to Adopt

1. **Index-based work distribution** for variable-length support
2. **Tile-level programming** (already present, needs dynamic bounds)
3. **Layout composition** for flexible tiling
4. **Multi-stage buffering** for memory latency hiding

### 7.2 Abstractions to Consider

| Abstraction | CUDA/Triton | PTO-ISA Extension |
|-------------|-------------|-------------------|
| Work index arrays | `blockIdx.x` lookup | Descriptor buffer |
| Dynamic masking | `tl.load(..., mask=)` | Bounds in descriptor |
| Layout algebra | CuTe layouts | Shape/Stride composition |
| State for fusion | `state_t` struct | Tile-based accumulator |

### 7.3 What NOT to Copy

- Thread-level programming (PTO-ISA is already tile-level)
- Warp shuffle (hardware handles this on Ascend)
- Shared memory management (L1/UB management differs)

## References

- [NVIDIA CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [Simon Boehm's CUDA MatMul Optimization](https://siboehm.com/articles/22/CUDA-MMM)
- [OpenAI Triton Introduction](https://openai.com/index/triton/)
- [CUTLASS/CuTe Documentation](https://docs.nvidia.com/cutlass/)
- [Red Hat: Democratizing AI Accelerators with Triton](https://next.redhat.com/2024/11/07/democratizing-ai-accelerators-and-gpu-kernel-programming-using-triton/)

---
*Note Version: 1.0*
*Last Updated: 2024-01-13*
