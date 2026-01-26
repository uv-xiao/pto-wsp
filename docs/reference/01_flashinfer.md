# Research Note 1: FlashInfer Architecture Deep Dive

## Overview

FlashInfer is a high-performance library for LLM inference attention, notable for its elegant handling of variable-length sequences through a **Plan-Run** execution model. This note extracts key architectural insights relevant to PTO-ISA runtime extension design.

**Source**: Analysis of FlashInfer source code (scheduler.cuh, decode.cuh, page.cuh, state.cuh)

## 1. Core Design Principle: Plan-Run Separation

### 1.1 The Two-Phase Model

FlashInfer cleanly separates work **planning** (CPU) from **execution** (GPU):

```
┌─────────────────────────────────────────────────────────────────────┐
│ PLAN PHASE (CPU, Python/C++)                                        │
│                                                                     │
│ Input: Variable metadata                                            │
│   - kv_indptr[batch_size+1]  (CSR format page counts)              │
│   - qo_indptr[batch_size+1]  (CSR format query lengths)            │
│                                                                     │
│ Operations:                                                         │
│   1. Extract lengths: kv_len[i] = indptr[i+1] - indptr[i]          │
│   2. Binary search optimal chunk_size                               │
│   3. Flatten work: (request, chunk) → linear block index            │
│                                                                     │
│ Output: Index arrays (device memory)                                │
│   - request_indices[total_blocks]                                   │
│   - kv_tile_indices[total_blocks]                                   │
│   - kv_chunk_size (single value)                                    │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│ RUN PHASE (GPU, CUDA)                                               │
│                                                                     │
│ Each CUDA block independently:                                      │
│   1. Read assignment: batch = request_indices[blockIdx.x]          │
│   2. Read chunk: chunk = kv_tile_indices[blockIdx.x]               │
│   3. Compute bounds: start = chunk * size, end = min(..., kv_len)  │
│   4. Execute attention for [start, end)                            │
│   5. Store partial result (if split) or final result               │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.2 Key Insight: Index-Based Indirection

The critical insight is that **variable-length handling is done through index arrays, not control flow**:

| Approach | Code Complexity | Runtime Overhead | Parallelism |
|----------|-----------------|------------------|-------------|
| Per-request kernel launch | Low | High (launch overhead) | Limited |
| Dynamic control flow in kernel | High | Medium (branch divergence) | Limited |
| **Index-based (FlashInfer)** | Medium | **Low** | **Maximum** |

The index arrays encode:
- **Which request** each block processes
- **Which chunk** of that request
- Bounds checking happens via **single comparison** in kernel

## 2. Scheduler Architecture

### 2.1 Binary Search for Chunk Size

FlashInfer uses binary search to find the optimal `kv_chunk_size`:

```cpp
// Goal: Find smallest chunk_size such that total_blocks ≤ max_grid_size
int64_t low = min_kv_chunk_size;
int64_t high = max_kv_len;

while (low < high) {
    int64_t mid = (low + high) / 2;
    int64_t total_blocks = 0;

    for (uint32_t i = 0; i < batch_size; ++i) {
        total_blocks += ceil_div(qo_len[i], qo_chunk) * ceil_div(kv_len[i], mid);
    }

    if (total_blocks > max_grid_size) {
        low = mid + 1;  // Need larger chunks (fewer blocks)
    } else {
        high = mid;     // Can try smaller chunks (more parallelism)
    }
}
```

**Design Choice**: This trades off parallelism (more blocks) vs overhead (merge cost for split KV).

### 2.2 Work Flattening

The 2D work space (request × chunk) is flattened to 1D block indices:

```cpp
uint32_t block_idx = 0;
for (uint32_t req = 0; req < batch_size; ++req) {
    uint32_t num_chunks = ceil_div(kv_len[req], chunk_size);
    for (uint32_t chunk = 0; chunk < num_chunks; ++chunk) {
        request_indices[block_idx] = req;
        kv_tile_indices[block_idx] = chunk;
        block_idx++;
    }
}
```

**Key Property**: Each block's assignment is O(1) to determine at runtime (just array lookup).

## 3. Kernel Design Patterns

### 3.1 Online Softmax with State Merging

FlashInfer uses the online softmax algorithm that allows processing in chunks:

```cpp
template <size_t vec_size>
struct state_t {
    vec_t<float, vec_size> o;  // Accumulated output
    float m;                    // Running max
    float d;                    // Running sum of exp

    void merge(const state_t& other) {
        float m_new = max(m, other.m);
        float scale_old = exp2((m - m_new) * LOG2E);
        float scale_new = exp2((other.m - m_new) * LOG2E);

        o = o * scale_old + other.o * scale_new;
        d = d * scale_old + other.d * scale_new;
        m = m_new;
    }
};
```

**Key Property**: Any partition of KV sequence can be processed independently and merged.

### 3.2 Thread Block Organization

```cpp
// Typical decode kernel block dimensions
dim3 block(bdx=32, bdy=8, bdz=4);
// bdx: Covers head_dim (32 threads × 4 vec_size = 128)
// bdy: GQA group size (8 QO heads per KV head)
// bdz: KV parallelism within block (4 tiles processed in parallel)

// Grid dimensions
dim3 grid(padded_batch_size, num_kv_heads);
```

### 3.3 Memory Access Patterns

```
Global Memory (Paged KV Cache)
    │
    │  cp.async.cg (predicated, L2 prefetch)
    ▼
Shared Memory (Multi-stage pipeline)
    │
    │  ldmatrix / vectorized load
    ▼
Registers (state_t, q_vec, k_vec, v_vec)
    │
    │  FMA, warp shuffle
    ▼
Registers (accumulated output)
    │
    │  cast_store
    ▼
Global Memory (Output)
```

## 4. Paged KV Cache

### 4.1 Data Structure

```cpp
struct paged_kv_t {
    DType* k_data;     // [max_pages, num_heads, page_size, head_dim]
    DType* v_data;     // Same layout
    IdType* indices;   // [total_pages] Logical → Physical mapping
    IdType* indptr;    // [batch_size+1] CSR format
    IdType* last_page_len;  // [batch_size] Tokens in last page

    uint32_t get_length(uint32_t batch_idx) {
        uint32_t num_pages = indptr[batch_idx+1] - indptr[batch_idx];
        return (num_pages - 1) * page_size + last_page_len[batch_idx];
    }
};
```

### 4.2 Address Translation

Token position → (page_index, offset_in_page) via fast integer division:

```cpp
uint32_t page_iter = position / page_size;
uint32_t page_offset = position % page_size;
uint32_t physical_page = indices[indptr[batch_idx] + page_iter];
```

## 5. Key Takeaways for PTO-ISA

### 5.1 What to Adopt

| FlashInfer Pattern | PTO-ISA Adaptation |
|--------------------|--------------------|
| Plan-Run separation | Host planning + AICore execution |
| Index-based work assignment | Descriptor arrays in GM |
| Binary search chunk sizing | Similar algorithm, different constraints |
| State merging for split KV | Same algorithm, different data types |
| Paged KV cache | Similar structure with Ascend memory qualifiers |

### 5.2 What's Different

| FlashInfer (CUDA) | PTO-ISA (Ascend) |
|-------------------|------------------|
| CPU planning | Could be Host or AICPU |
| CUDA blocks | AICore task dispatch |
| Shared memory | L1/UB buffer |
| Warp shuffle | Explicit data movement |
| cp.async | DMA operations |

### 5.3 Critical Questions

1. **Where should planning run?**
   - Host (like FlashInfer) - simplest, but 3μs latency per dispatch
   - AICPU - avoids round-trip, but single-threaded

2. **How to express index arrays?**
   - Global memory arrays (like FlashInfer)
   - Structured descriptors (more type-safe)

3. **How to handle KV split merge?**
   - Separate merge kernel (like FlashInfer)
   - In-kernel reduction
   - Hierarchical merge

## 6. Code References

| Component | FlashInfer File | Key Functions |
|-----------|-----------------|---------------|
| Work distribution | scheduler.cuh:494-613 | `PrefillSplitQOKVIndptr` |
| Binary search | scheduler.cuh:101-130 | `PrefillBinarySearchKVChunkSize` |
| Decode kernel | decode.cuh:393-608 | `BatchDecodeWithPagedKVCacheDevice` |
| State management | state.cuh:28-79 | `state_t::merge` |
| Paged KV | page.cuh:37-200 | `paged_kv_t` |

---
*Note Version: 1.0*
*Last Updated: 2024-01-13*
