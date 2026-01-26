# PTO Workload-Schedule Programming (PTO-WSP) framework Analysis v3: A Unified Programming Model

## 1. Executive Summary

This document presents a **unified conceptual framework** for PTO-ISA's runtime extension, derived from deep analysis of FlashInfer's Plan-Run pattern, PyPTO's MPMD runtime, and Ascend's hardware model. Rather than addressing individual requirements with piecemeal solutions, we extract a **core abstraction** that elegantly handles all dynamic LLM patterns.

**The Key Insight:**

> The fundamental problem is not "static vs dynamic scheduling" but rather **"separating work specification from work instantiation."**

This insight leads to a **Plan-Descriptor-Execute** programming model that:
1. Preserves calendar scheduling efficiency for the static parts
2. Handles dynamic shapes through lightweight runtime selection
3. Enables Human-In-The-Loop scheduling without kernel recompilation

## 2. The Core Problem: A Unified View

### 2.1 What FlashInfer Teaches Us

FlashInfer's elegant handling of variable sequence lengths reveals a crucial pattern:

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                     FlashInfer's Plan-Run Pattern                            │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  PLAN PHASE (CPU):                                                           │
│    Input:  Variable lengths [512, 2048, 8192, 32768]                         │
│    Output: Work indices     [request_idx[], chunk_idx[]]                     │
│                                                                              │
│    Key Operations:                                                           │
│    1. Extract actual lengths from CSR format                                 │
│    2. Binary search optimal chunk_size given max_grid_size                   │
│    3. Flatten 2D work space (request × chunk) to 1D block indices            │
│                                                                              │
│  RUN PHASE (GPU):                                                            │
│    block_idx → (request_indices[block_idx], chunk_indices[block_idx])        │
│    Each block independently processes its assigned chunk                     │
│    Bounds checking via: chunk_end = min(chunk_start + size, actual_len)      │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

**What makes this elegant:**
- **No kernel code change** for different sequence lengths
- **No wasted computation** on padding
- **Maximum parallelism** via work flattening
- **Efficient bounds handling** via single comparison

### 2.2 What PyPTO Shows Us

PyPTO's MPMD runtime reveals the challenges of dynamic execution:

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                     PyPTO's Current Model                                    │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  CONTROL FLOW (Single AICPU Thread):                                         │
│    for b in batch:                                                           │
│      for i in seq_len[b]:           ← Sequential task generation             │
│        generate_task(b, i)          ← Instantiation at runtime               │
│        resolve_dependencies()       ← Overhead per task                      │
│        dispatch()                                                            │
│                                                                              │
│  DISPATCH (AICPU Manager Threads):                                           │
│    poll_ready_queue()               ← Wait for control thread                │
│    send_to_aicore()                                                          │
│                                                                              │
│  Bottleneck: Control thread serializes ALL task generation                   │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

**The fundamental limitation:**
- Task **specification** and **instantiation** are conflated
- Control flow must run sequentially to generate tasks
- No way to parallelize task preparation

### 2.3 The Unified Abstraction

Comparing FlashInfer and PyPTO reveals the core insight:

| Aspect | FlashInfer | PyPTO | Problem |
|--------|------------|-------|---------|
| Work specification | Index arrays (plan) | Control flow code | Conflated |
| Work instantiation | Block reads index | AICPU generates task | Serial |
| Shape handling | Bounds check | Per-shape compile | Inflexible |
| Scheduling control | Implicit (grid) | None (runtime decides) | No user control |

**The Core Abstraction:**

> **Separate WHAT to compute from HOW to schedule from WHEN to instantiate**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Proposed: Plan-Descriptor-Execute Model                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ PLAN PHASE (Host, Compile Time)                                      │   │
│  │   - Compile multi-tier kernel variants                               │   │
│  │   - Define iteration space template                                  │   │
│  │   - Specify scheduling strategy as declarative constraints           │   │
│  └──────────────────────────────────┬──────────────────────────────────┘   │
│                                     │                                       │
│                                     ▼                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ DESCRIPTOR PHASE (AICPU Control, Runtime)                            │   │
│  │   - Read actual input shapes                                         │   │
│  │   - Select tier for each work item                                   │   │
│  │   - Generate lightweight descriptors (not full task instantiation)   │   │
│  │   - Apply scheduling constraints                                     │   │
│  └──────────────────────────────────┬──────────────────────────────────┘   │
│                                     │                                       │
│                                     ▼                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ EXECUTE PHASE (AICPU Dispatch + AICore, Runtime)                     │   │
│  │   - Dispatch threads instantiate from descriptors IN PARALLEL        │   │
│  │   - AICore executes kernels                                          │   │
│  │   - Bounds checking via descriptor parameters                        │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 3. The Programming Model

### 3.1 Core Concepts

The programming model is built on three orthogonal concepts:

**1. Iteration Space** - WHAT to compute
```
Iteration Space = { (batch, seq_pos, head, ...) | constraints }
```

**2. Scheduling Strategy** - HOW to execute
```
Schedule = mapping from iteration space to (core, time, dependencies)
```

**3. Tier Selection** - WHEN to decide
```
Tier = pre-compiled kernel variant for specific shape range
```

### 3.2 The Loop-Centric Model

Inspired by Halide and TVM, we express computation as **loops over iteration spaces**:

```cpp
// Conceptual representation
ITERATION_SPACE(flash_attention) {
    AXIS(batch, 0, batch_size)       // Batch dimension
    AXIS(seq_chunk, 0, DYNAMIC)      // Dynamic based on actual seq_len
    AXIS(head, 0, num_heads)         // Head dimension

    TILE(seq_chunk, chunk_size=1024) // Tile the dynamic axis

    KERNEL(fa_kernel, batch, seq_chunk, head)
}
```

The key insight: **loops define the iteration space**, but **scheduling decides execution order**.

### 3.3 The Descriptor Abstraction

Instead of generating full tasks at runtime, generate lightweight **descriptors**:

```cpp
struct WorkDescriptor {
    uint32_t kernel_tier;      // Which pre-compiled tier
    uint32_t batch_idx;        // Batch index
    uint32_t chunk_start;      // Start position in sequence
    uint32_t chunk_end;        // End position (actual length)
    uint32_t workspace_offset; // Pre-allocated workspace
};

// Size: 20 bytes vs ~1000+ bytes for full task instantiation
```

**Descriptor generation is O(1) per work item** - just fill in indices.
**Task instantiation happens in parallel** across dispatch threads.

### 3.4 The Schedule Language

Users express scheduling preferences as **declarative constraints**, not imperative code:

```cpp
SCHEDULE(attention_schedule) {
    // Locality: keep same-batch work on same core group
    AFFINITY(batch_idx → core_group)

    // Ordering: complete batch before starting next
    ORDER(batch_idx: SEQUENTIAL, chunk_idx: PARALLEL)

    // Prefetching: hint for next batch while computing current
    PREFETCH(batch_idx + 1, depth=2)

    // Tiling: control work granularity
    TILE(chunk_size=1024, reuse_l1=32)
}
```

The schedule is **compiled to a dispatch policy**, not interpreted at runtime.

## 4. Mapping to FlashInfer Example

### 4.1 FlashInfer's Decode Attention (Reference)

```cpp
// FlashInfer CPU Plan
auto [split_kv, chunk_size] = BinarySearchKVChunkSize(
    max_grid_size, kv_len_arr, qo_len_arr);

for (request_idx : batch_size) {
    for (chunk_idx : ceil_div(kv_len[request_idx], chunk_size)) {
        request_indices.push_back(request_idx);
        kv_tile_indices.push_back(chunk_idx);
    }
}

// FlashInfer GPU Run
__global__ void decode_kernel(Params params) {
    uint32_t batch_idx = params.request_indices[blockIdx.x];
    uint32_t chunk_idx = params.kv_tile_indices[blockIdx.x];
    uint32_t kv_len = paged_kv.get_length(batch_idx);

    uint32_t chunk_start = chunk_idx * chunk_size;
    uint32_t chunk_end = min(chunk_start + chunk_size, kv_len);

    // Process [chunk_start, chunk_end)
}
```

### 4.2 Equivalent in Proposed Model

```cpp
// === PLAN PHASE (Host, Compile Time) ===

// Define iteration space
PTO_ITERATION_SPACE(DecodeAttention) {
    PTO_AXIS(batch, 0, batch_size)
    PTO_AXIS(kv_chunk, 0, DYNAMIC)  // Determined at runtime
    PTO_AXIS(head, 0, num_kv_heads)

    PTO_TILE(kv_chunk, chunk_size)  // chunk_size from binary search
}

// Define kernel tiers
PTO_TIER_DEF(DecodeKernel) {
    PTO_TIER(kv_len <= 2048,  decode_kernel_2k)
    PTO_TIER(kv_len <= 8192,  decode_kernel_8k)
    PTO_TIER(kv_len <= 65536, decode_kernel_64k)
    PTO_DEFAULT(decode_kernel_128k)
}

// Define schedule
PTO_SCHEDULE_DEF(DecodeSchedule) {
    PTO_AFFINITY(batch, CORE_LOCAL)
    PTO_ORDER(batch: INTERLEAVED(4), kv_chunk: SEQUENTIAL)
}

// === DESCRIPTOR PHASE (AICPU Control, Runtime) ===

PTO_DESCRIPTOR_GEN(DecodeAttention) {
    // Binary search for optimal chunk_size (like FlashInfer)
    uint32_t chunk_size = BinarySearchChunkSize(kv_lens, max_blocks);

    // Generate descriptors (not full tasks)
    PTO_FOR_EACH(batch_idx, 0, batch_size) {
        uint32_t kv_len = kv_lens[batch_idx];
        uint32_t num_chunks = ceil_div(kv_len, chunk_size);

        PTO_FOR_EACH(chunk_idx, 0, num_chunks) {
            PTO_EMIT_DESCRIPTOR({
                .tier = SelectTier(kv_len),
                .batch_idx = batch_idx,
                .chunk_start = chunk_idx * chunk_size,
                .chunk_end = min((chunk_idx + 1) * chunk_size, kv_len),
            });
        }
    }
}

// === EXECUTE PHASE (Kernel, same as before) ===

PTO_KERNEL_IMPL(decode_kernel_8k) {
    // Read descriptor
    auto desc = PTO_GET_DESCRIPTOR();

    // Same kernel code, bounds from descriptor
    for (uint32_t pos = desc.chunk_start; pos < desc.chunk_end; pos += TILE) {
        TLOAD(k_tile, k_cache, {desc.batch_idx, pos});
        TMATMUL_ACC(score, q_tile, k_tile);
    }
    // Online softmax merge if split
}
```

### 4.3 Comparison

| Aspect | FlashInfer (CUDA) | Proposed (Ascend) |
|--------|-------------------|-------------------|
| Plan location | CPU (Python/C++) | Host (compile) + AICPU (runtime) |
| Work indices | GPU arrays | Descriptor buffer |
| Dispatch | CUDA grid | AICPU dispatch threads |
| Bounds | In-kernel check | Descriptor fields |
| Schedule control | Implicit | Declarative DSL |
| Multi-tier | Not needed | Built-in tier selection |

## 5. Handling Dynamic LLM Patterns

### 5.1 KV Cache Variable Length

```cpp
// Iteration space naturally handles variable lengths
PTO_ITERATION_SPACE(PagedAttention) {
    PTO_AXIS(batch, 0, batch_size)
    PTO_AXIS(page, 0, DYNAMIC)  // num_pages[batch] varies
}

// Descriptor generation reads actual lengths
PTO_DESCRIPTOR_GEN(PagedAttention) {
    for (b : batch_size) {
        uint32_t num_pages = page_table.get_page_count(b);
        for (p : num_pages) {
            PTO_EMIT_DESCRIPTOR({.batch=b, .page=p, .is_last=(p==num_pages-1)});
        }
    }
}
```

### 5.2 Lightning Indexer TopK Paths

```cpp
// Tiers handle discrete path selection
PTO_TIER_DEF(LightningIndexer) {
    PTO_TIER(eff_seq <= 2048,  indexer_2k)
    PTO_TIER(eff_seq <= 8192,  indexer_8k)
    PTO_TIER(eff_seq <= 65536, indexer_64k)
    PTO_DEFAULT(indexer_128k)
}

// Descriptor generation selects tier per position
PTO_DESCRIPTOR_GEN(LightningIndexer) {
    for (b : batch_size) {
        uint32_t cur_seq = seq_lens[b];
        for (s1 : S1) {
            uint32_t eff_seq = cur_seq - (S1 - s1 - 1);  // Causal offset
            uint32_t tier = SelectTier(eff_seq);
            PTO_EMIT_DESCRIPTOR({.batch=b, .pos=s1, .eff_seq=eff_seq, .tier=tier});
        }
    }
}
```

### 5.3 MoE Dynamic Routing

```cpp
// Iteration space over activated experts
PTO_ITERATION_SPACE(MoEForward) {
    PTO_AXIS(expert, 0, DYNAMIC)  // Only activated experts
    PTO_AXIS(token_group, 0, DYNAMIC)  // Tokens routed to this expert
}

// Descriptor generation after gating
PTO_DESCRIPTOR_GEN(MoEForward) {
    // Run gating first
    auto routing = compute_top_k_routing(gate_logits, k=2);

    // Generate descriptors only for activated experts
    for (expert_id : activated_experts) {
        auto tokens = routing.get_tokens_for_expert(expert_id);
        uint32_t num_groups = ceil_div(tokens.size(), GROUP_SIZE);
        for (g : num_groups) {
            PTO_EMIT_DESCRIPTOR({
                .expert=expert_id,
                .group=g,
                .token_indices=tokens.slice(g*GROUP_SIZE, GROUP_SIZE)
            });
        }
    }
}
```

## 6. Hardware Mapping

### 6.1 Ascend Hardware Model

```
1A (Host CPU)
└── 4B (AICPU)
    ├── Thread 0: Control (descriptor generation)
    └── Threads 1-N: Dispatch (parallel instantiation)
        ├── 24C (Cube Cores)
        └── 48D (Vector Cores)

Latencies:
  A → B: 3μs
  B → C/D: ~0μs
```

### 6.2 Phase Mapping

| Phase | Location | Latency | Parallelism |
|-------|----------|---------|-------------|
| Plan (compile) | Host (A) | N/A | N/A |
| Descriptor gen | AICPU T0 (B) | 3μs to start | Single-threaded |
| Instantiation | AICPU T1-N (B) | ~0μs | N-way parallel |
| Execution | AICore (C/D) | ~0μs | 24C + 48D cores |

### 6.3 Why This Works

1. **Descriptor generation is fast**: O(1) per work item, just index arithmetic
2. **Instantiation parallelizes**: N dispatch threads work independently
3. **No control flow bottleneck**: Descriptors are data, not code
4. **Scheduling is declarative**: Compiled policy, not runtime interpretation

## 7. Comparison with v1/v2 Proposals

| Aspect | v1 (Static Graph) | v2 (Piecemeal) | v3 (Unified) |
|--------|-------------------|----------------|--------------|
| Philosophy | Force static | Patch dynamic | Separate concerns |
| Abstraction | Task graph | Multiple primitives | Iteration space + schedule |
| Scheduling | Graph-level | Strategy hints | Declarative DSL |
| Dynamic handling | Multi-tier compile | Various mechanisms | Descriptor + tier |
| User control | Limited | Per-mechanism | Unified schedule language |
| Implementation | Complex | Scattered | Cohesive |

## 8. Summary

The v3 design extracts a **single unifying principle** from the analysis:

> **"Work specification and work instantiation are orthogonal"**

This leads to the **Plan-Descriptor-Execute** model:

1. **Plan** (compile time): Define iteration space, kernels, and schedule
2. **Descriptor** (AICPU control): Generate lightweight work indices
3. **Execute** (parallel): Instantiate and run from descriptors

The model provides:
- **Elegance**: One abstraction handles all dynamic patterns
- **Efficiency**: Descriptor generation is O(1), instantiation is parallel
- **Control**: Schedule DSL enables Human-In-The-Loop optimization
- **Compatibility**: Maps cleanly to CANN/AICPU capabilities

The next document (runtime-extension-spec-v3.md) will provide the concrete API specification.
