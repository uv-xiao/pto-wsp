# PTO Workload-Schedule Programming (PTO-WSP) framework Specification v2

**Focus:** Practical interfaces for dynamic LLM workloads

## 1. Design Goals

Based on the problem analysis, this specification addresses:

1. **Kernel Interface**: How to declare and invoke hand-written kernels
2. **Root Function Interface**: How to write custom task generation logic
3. **Scheduling Control**: How to specify execution strategies
4. **Dependency Management**: How to handle cross-graph dependencies
5. **Multi-Tier Support**: How to handle discrete dynamism efficiently

## 2. Kernel Declaration Interface

### 2.1 Kernel Signature

```cpp
// File: my_kernel.pto.hpp

#include <pto/runtime/kernel_decl.hpp>

// Declare kernel interface (separate from implementation)
PTO_KERNEL_DECL(FlashAttentionDecode) {
    //=== Input Tensors ===
    PTO_INPUT(q,        half,   [batch, heads, 1, head_dim])      // Query (single token)
    PTO_INPUT(k_cache,  half,   [batch, heads, max_seq, head_dim]) // Key cache
    PTO_INPUT(v_cache,  half,   [batch, heads, max_seq, head_dim]) // Value cache

    //=== Output Tensors ===
    PTO_OUTPUT(output,  half,   [batch, heads, 1, head_dim])

    //=== Runtime Parameters (vary per invocation) ===
    PTO_PARAM(actual_seq_len,   uint32_t)   // Actual sequence length (≤ max_seq)
    PTO_PARAM(batch_idx,        uint32_t)   // Which batch element
    PTO_PARAM(head_idx,         uint32_t)   // Which attention head

    //=== Workspace ===
    PTO_WORKSPACE_SIZE([](auto params) {
        // Workspace = softmax temp + accumulator
        return params.actual_seq_len * sizeof(float) + 128 * sizeof(float);
    })

    //=== Scheduling Hints ===
    PTO_CORE_TYPE(AIC_AIV)                  // Needs both cube and vector
    PTO_BLOCK_DIM(1)                        // Single core per invocation
    PTO_COMPUTE_BOUND(params.actual_seq_len * params.head_dim)
};
```

### 2.2 Kernel Implementation

```cpp
// File: my_kernel.pto.cpp

#include "my_kernel.pto.hpp"
#include <pto/pto-inst.hpp>

PTO_KERNEL_IMPL(FlashAttentionDecode) {
    // Access declared inputs/outputs/params
    auto q = PTO_GET_INPUT(q);
    auto k_cache = PTO_GET_INPUT(k_cache);
    auto v_cache = PTO_GET_INPUT(v_cache);
    auto output = PTO_GET_OUTPUT(output);

    auto actual_seq = PTO_GET_PARAM(actual_seq_len);
    auto batch = PTO_GET_PARAM(batch_idx);
    auto head = PTO_GET_PARAM(head_idx);

    // Get workspace pointer
    auto workspace = PTO_GET_WORKSPACE();

    // Standard PTO-ISA tile operations
    MatTile<half, 1, 128> q_tile;
    MatTile<half, 128, 128> k_tile;
    AccumulationTile<float, 1, 128> acc_tile;

    // Load query (single token)
    TLOAD(q_tile, q, {batch, head, 0, 0});

    // Iterate over actual sequence length
    for (uint32_t s = 0; s < actual_seq; s += 128) {
        uint32_t tile_len = min(128u, actual_seq - s);

        // Load key tile
        TLOAD(k_tile, k_cache, {batch, head, s, 0});

        // QK matmul
        TMATMUL_ACC(acc_tile, q_tile, k_tile, acc_tile);
    }

    // ... softmax, PV matmul, store output ...
}
```

### 2.3 Kernel Registration

```cpp
// Compile-time registration
PTO_REGISTER_KERNEL(FlashAttentionDecode, "flash_attention_decode_v1");

// Runtime lookup
auto kernel = PTO_LOOKUP_KERNEL("flash_attention_decode_v1");
```

## 3. Root Function (Task Generator) Interface

### 3.1 Root Function Declaration

```cpp
// File: deepseek_attention_root.pto.hpp

#include <pto/runtime/root_function.hpp>

PTO_ROOT_FUNCTION_DECL(DeepSeekAttentionRoot) {
    //=== Global Inputs (shared across all tasks) ===
    PTO_GLOBAL_INPUT(q_proj,    half,   [batch, seq, hidden])
    PTO_GLOBAL_INPUT(kv_cache,  half,   [batch, heads, max_seq, head_dim])
    PTO_GLOBAL_OUTPUT(output,   half,   [batch, seq, hidden])

    //=== Runtime Configuration ===
    PTO_CONFIG(batch_size,      uint32_t)
    PTO_CONFIG(seq_lens,        uint32_t*,  count=batch_size)  // Per-batch lengths
    PTO_CONFIG(kv_cache_lens,   uint32_t*,  count=batch_size)  // Per-batch KV lengths

    //=== Kernels Used ===
    PTO_USES_KERNEL(FlashAttentionDecode)
    PTO_USES_KERNEL(LightningIndexer_2K)
    PTO_USES_KERNEL(LightningIndexer_8K)
    PTO_USES_KERNEL(LightningIndexer_64K)
};
```

### 3.2 Root Function Implementation

```cpp
// File: deepseek_attention_root.pto.cpp

#include "deepseek_attention_root.pto.hpp"

PTO_ROOT_FUNCTION_IMPL(DeepSeekAttentionRoot) {
    auto batch_size = PTO_GET_CONFIG(batch_size);
    auto seq_lens = PTO_GET_CONFIG(seq_lens);
    auto kv_lens = PTO_GET_CONFIG(kv_cache_lens);

    //=== Phase 1: Generate Task Descriptors ===
    // This runs on AICPU control thread
    // Descriptors are lightweight - just indices and offsets

    PTO_PARALLEL_FOR(b, 0, batch_size) {
        // Batches are independent - descriptors generated in parallel
        auto seq_len = seq_lens[b];
        auto kv_len = kv_lens[b];

        PTO_FOR(pos, 0, seq_len) {
            auto eff_seq = kv_len + pos;

            //=== Tier Selection (O(log N) instead of O(N)) ===
            KernelRef indexer_kernel;
            if (eff_seq <= 2048) {
                indexer_kernel = PTO_KERNEL_REF(LightningIndexer_2K);
            } else if (eff_seq <= 8192) {
                indexer_kernel = PTO_KERNEL_REF(LightningIndexer_8K);
            } else {
                indexer_kernel = PTO_KERNEL_REF(LightningIndexer_64K);
            }

            //=== Emit Task Descriptors ===

            // Task 1: Lightning Indexer (TopK selection)
            auto indexer_task = PTO_EMIT_TASK(indexer_kernel, {
                .batch_idx = b,
                .position = pos,
                .actual_seq = eff_seq,
            });

            // Task 2: Flash Attention (depends on indexer)
            auto fa_task = PTO_EMIT_TASK(FlashAttentionDecode, {
                .batch_idx = b,
                .head_idx = ALL_HEADS,  // Broadcast to all heads
                .actual_seq_len = eff_seq,
            });

            // Explicit dependency
            PTO_DEPENDS(fa_task, indexer_task);
        }
    }

    //=== Phase 2: Apply Scheduling Strategy ===
    PTO_APPLY_SCHEDULE(PTO_GET_SCHEDULE());
}
```

## 4. Scheduling Strategy DSL

### 4.1 Strategy Definition

```cpp
// File: attention_schedules.pto.hpp

#include <pto/runtime/schedule.hpp>

//=== Strategy 1: Default (Task-Major, causes cold-start per batch) ===
PTO_SCHEDULE_DEF(DefaultSchedule) {
    // For each task type, then for each batch
    PTO_ORDER_BY(task_type, SEQUENTIAL);
    PTO_ORDER_BY(batch_idx, SEQUENTIAL);
}

//=== Strategy 2: Batch-Locality (10% faster) ===
PTO_SCHEDULE_DEF(BatchLocalitySchedule) {
    // Complete each batch before moving to next
    PTO_ORDER_BY(batch_idx, SEQUENTIAL);
    PTO_ORDER_BY(task_type, SEQUENTIAL);

    // Keep same-batch tasks on same core group for L2 locality
    PTO_AFFINITY(batch_idx, CORE_GROUP);
}

//=== Strategy 3: Interleaved (Hide Latency) ===
PTO_SCHEDULE_DEF(InterleavedSchedule) {
    // Interleave 4 batches to hide memory latency
    PTO_INTERLEAVE(batch_idx, GROUP_SIZE=4);

    // Within each group, tasks are sequential
    PTO_ORDER_BY(task_type, SEQUENTIAL);

    // Prefetch next group while computing current
    PTO_PREFETCH(batch_group + 1, WHILE=batch_group);
}

//=== Strategy 4: Pipeline (Maximize Throughput) ===
PTO_SCHEDULE_DEF(PipelineSchedule) {
    // Overlap indexer and attention across batches
    PTO_PIPELINE {
        STAGE(indexer, CORES=8);
        STAGE(attention, CORES=16);
    }

    // Different batches can be in different pipeline stages
    PTO_CONCURRENT_BATCHES(MAX=4);
}
```

### 4.2 Strategy Selection

```cpp
// At runtime, select strategy based on workload
auto schedule = SelectSchedule(batch_size, avg_seq_len);

if (batch_size >= 8 && avg_seq_len > 4096) {
    schedule = PTO_SCHEDULE(PipelineSchedule);
} else if (batch_size >= 4) {
    schedule = PTO_SCHEDULE(BatchLocalitySchedule);
} else {
    schedule = PTO_SCHEDULE(DefaultSchedule);
}

// Execute with selected schedule
PTO_EXECUTE(DeepSeekAttentionRoot, inputs, outputs, schedule);
```

## 5. Multi-Graph Concurrent Execution

### 5.1 Graph Declaration with External Dependencies

```cpp
// Graph A: MLA Prolog
PTO_GRAPH_DEF(MLAPrologGraph) {
    // Internal tasks
    PTO_TASK(q_proj, ...)
    PTO_TASK(k_proj, ...)
    PTO_TASK(v_proj, ...)

    // What this graph provides (for other graphs to consume)
    PTO_PROVIDES(q_tensor, AFTER=q_proj)
    PTO_PROVIDES(kv_cache, AFTER=k_proj, v_proj)
}

// Graph B: Lightning Indexer
PTO_GRAPH_DEF(LightningIndexerGraph) {
    // What this graph requires (from other graphs)
    PTO_REQUIRES(q_tensor, FROM=MLAPrologGraph)

    // Internal tasks
    PTO_TASK(qk_dot, ...)
    PTO_TASK(topk, DEPENDS=qk_dot)

    // What this graph provides
    PTO_PROVIDES(topk_indices, AFTER=topk)
}

// Graph C: Sparse Flash Attention
PTO_GRAPH_DEF(SparseFlashAttnGraph) {
    // Multiple requirements
    PTO_REQUIRES(q_tensor, FROM=MLAPrologGraph)
    PTO_REQUIRES(kv_cache, FROM=MLAPrologGraph)
    PTO_REQUIRES(topk_indices, FROM=LightningIndexerGraph)

    // Internal tasks
    PTO_TASK(gather_kv, ...)
    PTO_TASK(attention, DEPENDS=gather_kv)
    PTO_TASK(output, DEPENDS=attention)
}
```

### 5.2 Concurrent Execution

```cpp
// Execute graphs concurrently where dependencies allow
PTO_CONCURRENT_EXECUTE {
    // MLA and IndexerProlog are independent - run in parallel
    PTO_LAUNCH(MLAPrologGraph, args_mla);
    PTO_LAUNCH(IndexerPrologGraph, args_idx);  // Separate from LightningIndexer

    // LightningIndexer waits for IndexerProlog (automatic)
    PTO_LAUNCH(LightningIndexerGraph, args_li);

    // SparseFA waits for MLA and LightningIndexer (automatic)
    PTO_LAUNCH(SparseFlashAttnGraph, args_fa);
}

// Execution timeline (automatic based on dependencies):
//
// Time →
//
// MLA Prolog:        [==========]
// Indexer Prolog:    [======]
// Lightning Indexer:         [========]
// Sparse FA:                           [============]
//                                 ↑
//                    Waits for both MLA and LI
```

### 5.3 Version-Based Tensor Tracking

```cpp
// For WAR/WAW hazards
PTO_MEMORY_ORDER {
    // Write-After-Read: Ensure reads complete before write
    PTO_WAR(tensor_x, {
        READS = {graph_a.task_1, graph_a.task_2},
        WRITE = graph_b.task_3,
    });

    // Write-After-Write: Ensure writes are ordered
    PTO_WAW(tensor_x, {
        WRITE_1 = graph_a.task_3,
        WRITE_2 = graph_b.task_5,
    });

    // Read-After-Write: Automatic from REQUIRES/PROVIDES
}
```

## 6. Tiered Compilation

### 6.1 Tier Definition

```cpp
// Define tiers for a kernel
PTO_TIER_DEF(LightningIndexer) {
    // Each tier is a pre-compiled, calendar-scheduled kernel
    PTO_TIER(2K,   SEQ <= 2048,   LightningIndexer_2K,   workspace_2k)
    PTO_TIER(8K,   SEQ <= 8192,   LightningIndexer_8K,   workspace_8k)
    PTO_TIER(64K,  SEQ <= 65536,  LightningIndexer_64K,  workspace_64k)
    PTO_TIER(128K, DEFAULT,       LightningIndexer_128K, workspace_128k)
}

// Compiler generates optimized version for each tier
// Each tier uses calendar scheduling internally
```

### 6.2 Runtime Tier Selection

```cpp
// Fast tier lookup (O(log N) comparisons)
auto tier = PTO_SELECT_TIER(LightningIndexer, actual_seq_len);

// Emit task using selected tier's kernel
PTO_EMIT_TASK(tier.kernel, {
    .workspace = tier.workspace_offset(batch, pos),
    // ... other params
});
```

## 7. Cost Model Integration

### 7.1 Cost Model Interface

```cpp
// Cost model for scheduling decisions
PTO_COST_MODEL(FlashAttentionDecode) {
    // Compute cycles
    PTO_COMPUTE_COST([](auto params) {
        return params.actual_seq_len * params.head_dim * CUBE_CYCLES_PER_FLOP;
    });

    // Memory bandwidth
    PTO_MEMORY_COST([](auto params) {
        auto read_bytes = params.actual_seq_len * params.head_dim * 2;  // KV cache
        auto write_bytes = params.head_dim * 2;  // Output
        return (read_bytes + write_bytes) / MEMORY_BANDWIDTH;
    });

    // Estimated execution time
    PTO_TOTAL_COST([](auto compute, auto memory) {
        return max(compute, memory);  // Whichever dominates
    });
}
```

### 7.2 Cost-Based Scheduling

```cpp
// Scheduler uses cost model for decisions
PTO_SCHEDULE_DEF(CostAwareSchedule) {
    // Balance load across cores using cost estimates
    PTO_LOAD_BALANCE(BY=estimated_cost);

    // Overlap memory-bound and compute-bound tasks
    PTO_OVERLAP(memory_bound_tasks, compute_bound_tasks);
}
```

## 8. Complete Example: DeepSeek V3.2 Attention

```cpp
// deepseek_v32_attention.pto.cpp

#include <pto/runtime/all.hpp>

//=== Kernel Declarations (see above for full declarations) ===
PTO_KERNEL_DECL(MLAProlog) { ... }
PTO_KERNEL_DECL(IndexerProlog) { ... }
PTO_KERNEL_DECL(LightningIndexer_2K) { ... }
PTO_KERNEL_DECL(LightningIndexer_8K) { ... }
PTO_KERNEL_DECL(LightningIndexer_64K) { ... }
PTO_KERNEL_DECL(SparseFlashAttention) { ... }

//=== Tiers ===
PTO_TIER_DEF(LightningIndexer) {
    PTO_TIER(2K,   SEQ <= 2048,   LightningIndexer_2K)
    PTO_TIER(8K,   SEQ <= 8192,   LightningIndexer_8K)
    PTO_TIER(64K,  DEFAULT,       LightningIndexer_64K)
}

//=== Scheduling Strategy ===
PTO_SCHEDULE_DEF(DeepSeekSchedule) {
    // Batch-local for L2 reuse of KV cache
    PTO_AFFINITY(batch_idx, CORE_GROUP);
    PTO_ORDER_BY(batch_idx, SEQUENTIAL);

    // Within batch: pipeline indexer and attention
    PTO_PIPELINE {
        STAGE(indexer, CORES=8);
        STAGE(attention, CORES=16);
    }
}

//=== Root Function ===
PTO_ROOT_FUNCTION_IMPL(DeepSeekAttention) {
    auto batch_size = PTO_GET_CONFIG(batch_size);
    auto seq_lens = PTO_GET_CONFIG(seq_lens);
    auto kv_lens = PTO_GET_CONFIG(kv_cache_lens);

    // Phase 1: Generate all task descriptors
    PTO_PARALLEL_FOR(b, 0, batch_size) {
        auto seq_len = seq_lens[b];
        auto kv_len = kv_lens[b];

        // MLA Prolog (independent, can run first)
        auto mla_task = PTO_EMIT_TASK(MLAProlog, {
            .batch_idx = b,
            .seq_len = seq_len,
        });

        // Indexer Prolog (independent, parallel with MLA)
        auto idx_prolog_task = PTO_EMIT_TASK(IndexerProlog, {
            .batch_idx = b,
            .seq_len = seq_len,
        });

        PTO_FOR(pos, 0, seq_len) {
            auto eff_seq = kv_len + pos;

            // Select indexer tier
            auto indexer_tier = PTO_SELECT_TIER(LightningIndexer, eff_seq);

            // Lightning Indexer (depends on IndexerProlog)
            auto li_task = PTO_EMIT_TASK(indexer_tier.kernel, {
                .batch_idx = b,
                .position = pos,
                .actual_seq = eff_seq,
            });
            PTO_DEPENDS(li_task, idx_prolog_task);

            // Sparse Flash Attention (depends on MLA and LI)
            auto fa_task = PTO_EMIT_TASK(SparseFlashAttention, {
                .batch_idx = b,
                .position = pos,
                .actual_seq = eff_seq,
            });
            PTO_DEPENDS(fa_task, mla_task);
            PTO_DEPENDS(fa_task, li_task);
        }
    }

    // Phase 2: Execute with schedule
    PTO_APPLY_SCHEDULE(DeepSeekSchedule);
}

//=== Entry Point ===
void RunDeepSeekAttention(
    void* input,
    void* output,
    uint32_t batch_size,
    uint32_t* seq_lens,
    uint32_t* kv_lens
) {
    PTO_EXECUTE(DeepSeekAttention, {
        .input = input,
        .output = output,
        .batch_size = batch_size,
        .seq_lens = seq_lens,
        .kv_cache_lens = kv_lens,
    }, DeepSeekSchedule);
}
```

## 9. Implementation Status

| Feature | Status | Notes |
|---------|--------|-------|
| Kernel Declaration | Proposed | Clean interface for hand-written kernels |
| Root Function | Proposed | Task generation logic |
| Scheduling DSL | Proposed | Human-In-The-Loop control |
| Tier Compilation | Proposed | O(log N) dynamism handling |
| Concurrent Graphs | Proposed | Explicit dependency protocol |
| Cost Model | Proposed | Scheduling optimization |

## 10. Migration Path

For existing PyPTO code:

1. **Phase 1**: Wrap existing kernels with `PTO_KERNEL_DECL`
2. **Phase 2**: Convert control flow to `PTO_ROOT_FUNCTION`
3. **Phase 3**: Add scheduling strategies
4. **Phase 4**: Enable concurrent graph execution

The interfaces are designed to be incrementally adoptable.
