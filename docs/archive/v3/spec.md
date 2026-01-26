# PTO Workload-Schedule Programming (PTO-WSP) framework Specification v3: Plan-Descriptor-Execute API

## 1. Overview

This specification defines the concrete API for the Plan-Descriptor-Execute programming model described in `analysis-v3.md`. The API is designed to:

1. **Be elegant**: Minimal concepts, maximum expressiveness
2. **Be practical**: Maps directly to CANN/AICPU capabilities
3. **Enable FlashInfer-style patterns**: CPU planning, parallel execution
4. **Support Human-In-The-Loop**: Declarative scheduling control

## 2. Core API Components

### 2.1 API Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         PTO Workload-Schedule Programming (PTO-WSP) framework API                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐             │
│  │ Iteration Space │  │  Kernel Tiers   │  │   Schedule      │             │
│  │   Definition    │  │   Definition    │  │   Definition    │             │
│  │                 │  │                 │  │                 │             │
│  │ PTO_AXIS        │  │ PTO_TIER_DEF    │  │ PTO_SCHEDULE    │             │
│  │ PTO_TILE        │  │ PTO_TIER        │  │ PTO_AFFINITY    │             │
│  │ PTO_SPACE       │  │ PTO_DEFAULT     │  │ PTO_ORDER       │             │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘             │
│           │                    │                    │                       │
│           └────────────────────┼────────────────────┘                       │
│                                ▼                                            │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    Descriptor Generator                              │   │
│  │                                                                      │   │
│  │  PTO_DESCRIPTOR_GEN { PTO_FOR_EACH ... PTO_EMIT_DESCRIPTOR }        │   │
│  └──────────────────────────────────┬──────────────────────────────────┘   │
│                                     │                                       │
│                                     ▼                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    Kernel Implementation                             │   │
│  │                                                                      │   │
│  │  PTO_KERNEL_IMPL { PTO_GET_DESCRIPTOR(); ... TLOAD/TMATMUL ... }    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Component Responsibilities

| Component | When | Where | What |
|-----------|------|-------|------|
| Iteration Space | Compile | Host | Defines work dimensions and tiling |
| Kernel Tiers | Compile | Host | Pre-compiled kernel variants |
| Schedule | Compile | Host | Scheduling policy constraints |
| Descriptor Generator | Runtime | AICPU T0 | Lightweight work index generation |
| Kernel Implementation | Runtime | AICore | Actual computation |

## 3. Iteration Space API

### 3.1 PTO_SPACE

Defines the iteration space for a computation.

```cpp
PTO_SPACE(name) {
    // Axis definitions
    // Tiling specifications
    // Constraints
}
```

**Example:**
```cpp
PTO_SPACE(FlashAttentionDecode) {
    PTO_AXIS(batch, 0, batch_size)
    PTO_AXIS(kv_chunk, 0, PTO_DYNAMIC)
    PTO_AXIS(head, 0, num_kv_heads)

    PTO_TILE(kv_chunk, 1024)
}
```

### 3.2 PTO_AXIS

Defines a dimension in the iteration space.

```cpp
PTO_AXIS(name, start, end)
PTO_AXIS(name, start, PTO_DYNAMIC)  // Dynamic extent
```

**Semantics:**
- `name`: Identifier for this axis
- `start`: Start index (typically 0)
- `end`: End index (exclusive), or `PTO_DYNAMIC` for runtime-determined

### 3.3 PTO_TILE

Specifies tiling for an axis.

```cpp
PTO_TILE(axis, tile_size)
PTO_TILE(axis, tile_size, PTO_PAD_LAST)  // Pad last tile
PTO_TILE(axis, PTO_AUTO)                 // Auto-tune tile size
```

**Semantics:**
- Splits `axis` into `ceil(extent / tile_size)` iterations
- Each iteration processes `min(tile_size, remaining)` elements

## 4. Kernel Tier API

### 4.1 PTO_TIER_DEF

Defines a set of kernel variants for different input ranges.

```cpp
PTO_TIER_DEF(name) {
    PTO_TIER(condition, kernel_variant)
    ...
    PTO_DEFAULT(kernel_variant)
}
```

**Example:**
```cpp
PTO_TIER_DEF(DecodeAttentionKernel) {
    PTO_TIER(kv_len <= 2048,  decode_2k)
    PTO_TIER(kv_len <= 8192,  decode_8k)
    PTO_TIER(kv_len <= 65536, decode_64k)
    PTO_DEFAULT(decode_128k)
}
```

**Semantics:**
- Conditions are evaluated top-to-bottom
- First matching tier is selected
- `PTO_DEFAULT` handles all remaining cases
- Each kernel is **pre-compiled** at host compile time

### 4.2 Tier Selection at Runtime

```cpp
// In descriptor generator
uint32_t tier = PTO_SELECT_TIER(DecodeAttentionKernel, kv_len);
```

Returns the tier index (0, 1, 2, ...) for the given value.

## 5. Schedule API

### 5.1 PTO_SCHEDULE

Defines scheduling constraints as declarative policy.

```cpp
PTO_SCHEDULE(name) {
    // Affinity constraints
    // Ordering constraints
    // Prefetch hints
}
```

**Example:**
```cpp
PTO_SCHEDULE(BatchLocalitySchedule) {
    PTO_AFFINITY(batch, PTO_CORE_GROUP)
    PTO_ORDER(batch, PTO_SEQUENTIAL)
    PTO_ORDER(kv_chunk, PTO_PARALLEL)
    PTO_PREFETCH(batch, 1)
}
```

### 5.2 PTO_AFFINITY

Specifies core placement preferences.

```cpp
PTO_AFFINITY(axis, policy)
```

**Policies:**
- `PTO_CORE_GROUP`: Keep work with same axis value on same core group
- `PTO_CORE_LOCAL`: Strong locality, same axis → same core
- `PTO_SPREAD`: Spread across all available cores

### 5.3 PTO_ORDER

Specifies execution ordering constraints.

```cpp
PTO_ORDER(axis, ordering)
PTO_ORDER(axis, PTO_INTERLEAVED(group_size))
```

**Orderings:**
- `PTO_SEQUENTIAL`: Process axis values in order (0, 1, 2, ...)
- `PTO_PARALLEL`: No ordering constraint, maximize parallelism
- `PTO_INTERLEAVED(n)`: Process n items from each axis value before advancing

**Example:**
```cpp
PTO_ORDER(batch, PTO_INTERLEAVED(4))
// Processing order: B0C0, B1C0, B2C0, B3C0, B0C1, B1C1, ...
```

### 5.4 PTO_PREFETCH

Hints for data prefetching.

```cpp
PTO_PREFETCH(axis, depth)
```

**Semantics:**
- Prefetch data for `axis + depth` while processing `axis`
- Depth controls how many iterations to look ahead

## 6. Descriptor API

### 6.1 WorkDescriptor Structure

The fundamental data structure for lightweight work specification.

```cpp
struct PTO_WorkDescriptor {
    uint32_t tier;              // Kernel tier index
    uint32_t indices[N];        // Iteration space indices
    uint32_t bounds[M];         // Dynamic bounds (actual lengths)
    uint32_t workspace_offset;  // Pre-allocated workspace position
};

// Example for FlashAttention
struct FADecodeDescriptor {
    uint32_t tier;              // 0=2k, 1=8k, 2=64k, 3=128k
    uint32_t batch_idx;
    uint32_t chunk_idx;
    uint32_t head_idx;
    uint32_t chunk_start;       // Start position in KV sequence
    uint32_t chunk_end;         // End position (actual length)
    uint32_t workspace_offset;
};
```

**Design Principles:**
- Fixed size, known at compile time
- Only indices and bounds, no pointers
- ~20-40 bytes, vs 1000+ bytes for full task instantiation

### 6.2 PTO_DESCRIPTOR_GEN

Defines the descriptor generation logic that runs on AICPU control thread.

```cpp
PTO_DESCRIPTOR_GEN(space_name) {
    // Setup code
    // Loop over iteration space
    // Emit descriptors
}
```

**Example:**
```cpp
PTO_DESCRIPTOR_GEN(FlashAttentionDecode) {
    // Read input metadata
    const uint32_t* kv_lens = PTO_GET_INPUT_LENS(kv_cache);
    const uint32_t batch_size = PTO_GET_DIM(batch);

    // Binary search for optimal chunk size (FlashInfer pattern)
    uint32_t chunk_size = BinarySearchChunkSize(kv_lens, batch_size, max_blocks);

    // Generate descriptors
    PTO_FOR_EACH(batch_idx, 0, batch_size) {
        uint32_t kv_len = kv_lens[batch_idx];
        uint32_t num_chunks = PTO_CEIL_DIV(kv_len, chunk_size);
        uint32_t tier = PTO_SELECT_TIER(DecodeAttentionKernel, kv_len);

        PTO_FOR_EACH(chunk_idx, 0, num_chunks) {
            PTO_FOR_EACH(head_idx, 0, num_kv_heads) {
                PTO_EMIT_DESCRIPTOR({
                    .tier = tier,
                    .batch_idx = batch_idx,
                    .chunk_idx = chunk_idx,
                    .head_idx = head_idx,
                    .chunk_start = chunk_idx * chunk_size,
                    .chunk_end = PTO_MIN((chunk_idx + 1) * chunk_size, kv_len),
                    .workspace_offset = PTO_ALLOC_WORKSPACE(tier, batch_idx, chunk_idx),
                });
            }
        }
    }
}
```

### 6.3 PTO_FOR_EACH

Loop construct for descriptor generation.

```cpp
PTO_FOR_EACH(var, start, end) {
    // Body
}
```

**Semantics:**
- Executes body for each value in `[start, end)`
- Variables are local to the loop
- Nested loops are allowed

### 6.4 PTO_EMIT_DESCRIPTOR

Emits a work descriptor to the dispatch queue.

```cpp
PTO_EMIT_DESCRIPTOR(descriptor_initializer)
```

**Semantics:**
- Creates a WorkDescriptor with given fields
- Enqueues to dispatch buffer
- O(1) operation - just memory copy

## 7. Kernel API

### 7.1 PTO_KERNEL_IMPL

Defines a kernel implementation for a specific tier.

```cpp
PTO_KERNEL_IMPL(kernel_name) {
    // Get descriptor
    auto desc = PTO_GET_DESCRIPTOR();

    // Kernel computation using PTO-ISA
    // ...
}
```

**Example:**
```cpp
PTO_KERNEL_IMPL(decode_8k) {
    auto desc = PTO_GET_DESCRIPTOR();

    // Load Q tile (once per invocation)
    VecTile<half, 1, HEAD_DIM> q_tile;
    TLOAD(q_tile, q, {desc.batch_idx, desc.head_idx, 0});

    // Initialize accumulator
    AccumulationTile<float, 1, HEAD_DIM> acc_tile;
    TASSIGN(acc_tile, 0.0f);

    float m = -INFINITY;
    float d = 0.0f;

    // Process KV range from descriptor
    for (uint32_t pos = desc.chunk_start; pos < desc.chunk_end; pos += KV_TILE) {
        // Load K tile
        RightTile<half, KV_TILE, HEAD_DIM> k_tile;
        TLOAD(k_tile, k_cache, {desc.batch_idx, pos, desc.head_idx});

        // Compute QK
        VecTile<float, 1, KV_TILE> qk_tile;
        TMATMUL(qk_tile, q_tile, k_tile);

        // Online softmax
        float m_new = TROWMAX(qk_tile);
        m_new = PTO_MAX(m, m_new);

        // Rescale and accumulate
        float scale = PTO_EXP2(m - m_new);
        TMULS(acc_tile, acc_tile, scale);
        d = d * scale;

        // Add new contribution
        TSUB(qk_tile, qk_tile, m_new);
        TEXP2(qk_tile, qk_tile);
        float d_new = TROWSUM(qk_tile);
        d = d + d_new;

        // Load V and accumulate
        LeftTile<half, KV_TILE, HEAD_DIM> v_tile;
        TLOAD(v_tile, v_cache, {desc.batch_idx, pos, desc.head_idx});
        TMATMUL_ACC(acc_tile, qk_tile, v_tile);

        m = m_new;
    }

    // Normalize output
    TDIVS(acc_tile, acc_tile, d);

    // Store result (or partial result if split)
    if (desc.chunk_idx == 0 && desc.chunk_end == kv_len) {
        // Single chunk, store final
        TSTORE(output, acc_tile, {desc.batch_idx, desc.head_idx});
    } else {
        // Multiple chunks, store partial for merging
        TSTORE(partial_out, acc_tile, {desc.batch_idx, desc.chunk_idx, desc.head_idx});
        TSTORE(partial_md, {m, d}, {desc.batch_idx, desc.chunk_idx, desc.head_idx});
    }
}
```

### 7.2 PTO_GET_DESCRIPTOR

Retrieves the work descriptor for the current invocation.

```cpp
auto desc = PTO_GET_DESCRIPTOR();
```

**Returns:** Reference to the WorkDescriptor assigned to this core.

## 8. Complete FlashInfer-Equivalent Example

### 8.1 Decode Attention (Complete)

```cpp
// ============================================================================
// FILE: decode_attention.pto
// FlashInfer-equivalent decode attention with variable sequence lengths
// ============================================================================

#include <pto/pto-inst.hpp>
#include <pto/runtime/descriptor.hpp>
#include <pto/runtime/schedule.hpp>

// ----------------------------------------------------------------------------
// 1. ITERATION SPACE DEFINITION
// ----------------------------------------------------------------------------

PTO_SPACE(DecodeAttention) {
    PTO_AXIS(batch, 0, batch_size)
    PTO_AXIS(kv_chunk, 0, PTO_DYNAMIC)      // Determined by actual kv_len
    PTO_AXIS(head, 0, num_kv_heads)

    PTO_TILE(kv_chunk, PTO_AUTO)            // Auto-tune chunk size
}

// ----------------------------------------------------------------------------
// 2. KERNEL TIERS
// ----------------------------------------------------------------------------

PTO_TIER_DEF(DecodeKernel) {
    PTO_TIER(kv_len <= 2048,  decode_kernel_2k,  workspace_2k)
    PTO_TIER(kv_len <= 8192,  decode_kernel_8k,  workspace_8k)
    PTO_TIER(kv_len <= 65536, decode_kernel_64k, workspace_64k)
    PTO_DEFAULT(decode_kernel_128k, workspace_128k)
}

// ----------------------------------------------------------------------------
// 3. SCHEDULING STRATEGY
// ----------------------------------------------------------------------------

PTO_SCHEDULE(DecodeSchedule) {
    // Locality: Same batch stays on same core group for L2 reuse
    PTO_AFFINITY(batch, PTO_CORE_GROUP)

    // Ordering: Interleave batches to hide latency, chunks are parallel
    PTO_ORDER(batch, PTO_INTERLEAVED(4))
    PTO_ORDER(kv_chunk, PTO_PARALLEL)
    PTO_ORDER(head, PTO_PARALLEL)

    // Prefetch next batch's Q while computing current
    PTO_PREFETCH(batch, 1)
}

// ----------------------------------------------------------------------------
// 4. DESCRIPTOR STRUCTURE
// ----------------------------------------------------------------------------

struct DecodeDescriptor {
    uint32_t tier;
    uint32_t batch_idx;
    uint32_t chunk_idx;
    uint32_t head_idx;
    uint32_t chunk_start;
    uint32_t chunk_end;
    uint32_t num_chunks;        // For merge decision
    uint32_t workspace_offset;
};

// ----------------------------------------------------------------------------
// 5. DESCRIPTOR GENERATOR (runs on AICPU control thread)
// ----------------------------------------------------------------------------

PTO_DESCRIPTOR_GEN(DecodeAttention) {
    // Get input metadata
    auto kv_lens = PTO_GET_INPUT_META(kv_cache, lengths);  // [batch_size]
    uint32_t batch_size = PTO_GET_BATCH_SIZE();
    uint32_t num_heads = PTO_GET_PARAM(num_kv_heads);

    // Binary search for optimal chunk size (FlashInfer pattern)
    uint32_t max_blocks = PTO_GET_MAX_BLOCKS();
    uint32_t chunk_size = BinarySearchChunkSize(kv_lens, batch_size, num_heads, max_blocks);

    // Apply schedule
    PTO_APPLY_SCHEDULE(DecodeSchedule);

    // Generate descriptors following schedule order
    PTO_SCHEDULED_FOR(batch_idx, kv_chunk_idx, head_idx) {
        uint32_t kv_len = kv_lens[batch_idx];
        uint32_t num_chunks = PTO_CEIL_DIV(kv_len, chunk_size);

        // Skip if chunk doesn't exist for this batch
        if (kv_chunk_idx >= num_chunks) continue;

        uint32_t tier = PTO_SELECT_TIER(DecodeKernel, kv_len);

        PTO_EMIT_DESCRIPTOR(DecodeDescriptor{
            .tier = tier,
            .batch_idx = batch_idx,
            .chunk_idx = kv_chunk_idx,
            .head_idx = head_idx,
            .chunk_start = kv_chunk_idx * chunk_size,
            .chunk_end = PTO_MIN((kv_chunk_idx + 1) * chunk_size, kv_len),
            .num_chunks = num_chunks,
            .workspace_offset = PTO_ALLOC_WS(tier, batch_idx, kv_chunk_idx, head_idx),
        });
    }
}

// Binary search helper (same as FlashInfer)
uint32_t BinarySearchChunkSize(
    const uint32_t* kv_lens, uint32_t batch_size,
    uint32_t num_heads, uint32_t max_blocks
) {
    uint32_t max_kv = *std::max_element(kv_lens, kv_lens + batch_size);
    uint32_t low = 256, high = max_kv;

    while (low < high) {
        uint32_t mid = (low + high) / 2;
        uint64_t total_blocks = 0;
        for (uint32_t b = 0; b < batch_size; ++b) {
            total_blocks += PTO_CEIL_DIV(kv_lens[b], mid) * num_heads;
        }

        if (total_blocks > max_blocks) {
            low = mid + 1;  // Need larger chunks
        } else {
            high = mid;     // Can use smaller chunks
        }
    }
    return low;
}

// ----------------------------------------------------------------------------
// 6. KERNEL IMPLEMENTATIONS
// ----------------------------------------------------------------------------

// 2K tier - optimized for short sequences
PTO_KERNEL_IMPL(decode_kernel_2k) {
    auto desc = PTO_GET_DESCRIPTOR<DecodeDescriptor>();
    // ... implementation with 2K-specific optimizations ...
}

// 8K tier - balanced implementation
PTO_KERNEL_IMPL(decode_kernel_8k) {
    auto desc = PTO_GET_DESCRIPTOR<DecodeDescriptor>();

    // Load Q (once)
    VecTile<half, 1, HEAD_DIM> q_tile;
    TLOAD(q_tile, q, {desc.batch_idx, desc.head_idx * GQA_RATIO, 0});

    // Online softmax state
    AccTile<float, 1, HEAD_DIM> o_tile;
    TASSIGN(o_tile, 0.0f);
    float m = -INFINITY, d = 0.0f;

    // Process assigned chunk
    for (uint32_t pos = desc.chunk_start; pos < desc.chunk_end; pos += KV_TILE) {
        uint32_t actual_tile = PTO_MIN(KV_TILE, desc.chunk_end - pos);

        // Load K, V tiles
        RightTile<half, KV_TILE, HEAD_DIM> k_tile;
        LeftTile<half, KV_TILE, HEAD_DIM> v_tile;
        TLOAD(k_tile, k_cache, {desc.batch_idx, pos, desc.head_idx});
        TLOAD(v_tile, v_cache, {desc.batch_idx, pos, desc.head_idx});

        // QK^T
        VecTile<float, 1, KV_TILE> scores;
        TMATMUL(scores, q_tile, k_tile);
        TMULS(scores, scores, 1.0f / sqrt(HEAD_DIM));

        // Mask invalid positions (if last tile is partial)
        if (actual_tile < KV_TILE) {
            TMASK(scores, actual_tile, -INFINITY);
        }

        // Online softmax update
        float m_new = PTO_MAX(m, TROWMAX(scores));
        float scale = PTO_EXP2((m - m_new) * LOG2E);
        TMULS(o_tile, o_tile, scale);
        d *= scale;

        TSUBS(scores, scores, m_new);
        TEXP(scores, scores);
        d += TROWSUM(scores);

        // Accumulate P*V
        TMATMUL_ACC(o_tile, scores, v_tile);
        m = m_new;
    }

    // Store result
    if (desc.num_chunks == 1) {
        // Single chunk: normalize and store final output
        TDIVS(o_tile, o_tile, d);
        TSTORE(output, o_tile, {desc.batch_idx, desc.head_idx * GQA_RATIO});
    } else {
        // Multiple chunks: store partial for later merge
        TSTORE(partial_o, o_tile, {desc.batch_idx, desc.chunk_idx, desc.head_idx});
        PTO_STORE_SCALAR(partial_m, m, {desc.batch_idx, desc.chunk_idx, desc.head_idx});
        PTO_STORE_SCALAR(partial_d, d, {desc.batch_idx, desc.chunk_idx, desc.head_idx});
    }
}

// 64K and 128K tiers similar with different optimizations...

// ----------------------------------------------------------------------------
// 7. MERGE KERNEL (for split-KV case)
// ----------------------------------------------------------------------------

PTO_KERNEL_IMPL(merge_partial_outputs) {
    auto desc = PTO_GET_DESCRIPTOR<MergeDescriptor>();

    // Load all partial results for this (batch, head)
    AccTile<float, 1, HEAD_DIM> final_o;
    TASSIGN(final_o, 0.0f);
    float final_m = -INFINITY, final_d = 0.0f;

    for (uint32_t c = 0; c < desc.num_chunks; ++c) {
        AccTile<float, 1, HEAD_DIM> partial_o;
        TLOAD(partial_o, partial_out, {desc.batch_idx, c, desc.head_idx});
        float m_c = PTO_LOAD_SCALAR(partial_m, {desc.batch_idx, c, desc.head_idx});
        float d_c = PTO_LOAD_SCALAR(partial_d, {desc.batch_idx, c, desc.head_idx});

        // Online merge (same as FlashInfer)
        float m_new = PTO_MAX(final_m, m_c);
        float scale_old = PTO_EXP2((final_m - m_new) * LOG2E);
        float scale_new = PTO_EXP2((m_c - m_new) * LOG2E);

        TMULS(final_o, final_o, scale_old);
        AccTile<float, 1, HEAD_DIM> scaled_partial;
        TMULS(scaled_partial, partial_o, scale_new);
        TADD(final_o, final_o, scaled_partial);

        final_d = final_d * scale_old + d_c * scale_new;
        final_m = m_new;
    }

    // Normalize and store
    TDIVS(final_o, final_o, final_d);
    TSTORE(output, final_o, {desc.batch_idx, desc.head_idx * GQA_RATIO});
}

// ----------------------------------------------------------------------------
// 8. HOST ENTRY POINT
// ----------------------------------------------------------------------------

void decode_attention_launch(
    Tensor q,           // [batch, num_qo_heads, 1, head_dim]
    PagedKVCache kv,    // Paged KV cache
    Tensor output,      // [batch, num_qo_heads, 1, head_dim]
    uint32_t* kv_lens   // [batch] actual KV lengths
) {
    // Compile-time: Already have multi-tier kernels and schedule

    // Runtime: Generate descriptors and execute
    PTO_LAUNCH(DecodeAttention, {
        .q = q,
        .kv_cache = kv,
        .output = output,
        .kv_lens = kv_lens,
    });
}
```

## 9. Compilation and Runtime Flow

### 9.1 Compile Time (Host)

```
Source (.pto) ──► PTO Compiler ──┬──► Tier Kernels (N binaries)
                                 ├──► Schedule Policy (dispatch rules)
                                 └──► Descriptor Gen Code (AICPU binary)
```

### 9.2 Runtime (Device)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ AICPU Thread 0 (Control)                                                    │
│   1. Read input metadata (kv_lens, batch_size)                              │
│   2. Binary search optimal chunk_size                                       │
│   3. Generate descriptors (O(total_work) but O(1) per descriptor)           │
│   4. Signal dispatch threads                                                │
├─────────────────────────────────────────────────────────────────────────────┤
│ AICPU Threads 1-N (Dispatch) [PARALLEL]                                     │
│   1. Read descriptor from queue                                             │
│   2. Select kernel binary based on tier                                     │
│   3. Bind parameters from descriptor                                        │
│   4. Dispatch to AICore                                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│ AICore 0-23 / AIVector 0-47 [PARALLEL]                                      │
│   1. Execute kernel with descriptor parameters                              │
│   2. Bounds checking via descriptor fields                                  │
│   3. Store results (or partial results for merge)                           │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 9.3 Key Performance Properties

| Operation | Complexity | Location | Parallelism |
|-----------|------------|----------|-------------|
| Descriptor gen | O(1) per work | AICPU T0 | Sequential |
| Task instantiation | O(1) per work | AICPU T1-N | N-parallel |
| Kernel execution | O(chunk_size) | AICore | 24+48 parallel |
| Total work | O(batch × chunks × heads) | Distributed | Fully parallel |

## 10. Comparison Summary

| Feature | FlashInfer (CUDA) | PyPTO (Current) | PTO-v3 (Proposed) |
|---------|-------------------|-----------------|-------------------|
| Planning | CPU arrays | AICPU runtime | AICPU descriptors |
| Dispatch | CUDA grid | Single-threaded | Multi-threaded |
| Variable len | Bounds check | Per-shape compile | Descriptor bounds |
| Scheduling | Implicit | Automatic | Declarative DSL |
| User control | None | Limited | Full schedule API |
| Kernel tiers | Manual | Manual | Built-in |

## 11. Future Extensions

### 11.1 MoE Support

```cpp
PTO_SPACE(MoEForward) {
    PTO_AXIS(expert, 0, PTO_DYNAMIC)      // Only activated experts
    PTO_AXIS(token_group, 0, PTO_DYNAMIC) // Tokens routed to expert
}

PTO_DESCRIPTOR_GEN(MoEForward) {
    auto routing = PTO_COMPUTE(gating, gate_logits);
    for (auto [expert, tokens] : routing.activated()) {
        for (auto group : tokens.chunks(GROUP_SIZE)) {
            PTO_EMIT_DESCRIPTOR({.expert=expert, .tokens=group});
        }
    }
}
```

### 11.2 Cross-Graph Dependencies

```cpp
PTO_GRAPH(prefill_decode_pipeline) {
    PTO_STAGE(prefill, PrefillAttention)
    PTO_STAGE(decode, DecodeAttention)

    PTO_DEPENDENCY(decode.kv_cache, prefill.kv_update)
}
```

### 11.3 Profile-Guided Optimization

```cpp
PTO_SCHEDULE(AdaptiveSchedule) {
    PTO_PROFILE_GUIDED(batch, {
        .cold_start = PTO_INTERLEAVED(2),
        .steady_state = PTO_SEQUENTIAL,
    })
}
```

---

*Specification Version: 3.0*
*Compatible with: PTO-ISA 1.0, CANN 8.0+*
