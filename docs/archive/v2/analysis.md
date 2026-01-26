# PTO Workload-Schedule Programming (PTO-WSP) framework Analysis v2: Addressing Dynamic LLM Workloads

## Executive Summary

This document revises the original runtime extension proposal based on deeper analysis of LLM workload dynamics. The key insight is that **LLM inference is fundamentally dynamic**, and any runtime solution must embrace this rather than trying to force static graph patterns.

## 1. The Fundamental Tension

### 1.1 Calendar Scheduling (Static)

Calendar scheduling pre-computes everything at compile time:

```
Compile Time:
  - All task identities, order, and timing
  - All memory addresses and layouts
  - All synchronization points

Runtime:
  - Just "play the calendar" - no decisions needed
  - Maximum throughput, minimum overhead
```

**Advantages:**
- Near-optimal resource utilization
- No runtime scheduling overhead
- Enables aggressive compiler optimizations

**Fatal Flaw for LLMs:**
- Cannot handle variable sequence lengths
- Cannot handle growing KV cache
- Cannot handle MoE dynamic routing
- Must pad to max length → massive waste

### 1.2 Dynamic Execution (Current PyPTO)

Dynamic execution makes decisions at runtime:

```
Runtime:
  - Determine actual shapes from input
  - Generate tasks based on runtime values
  - Resolve dependencies dynamically
  - Allocate memory on-demand
```

**Advantages:**
- Handles all dynamic patterns
- No wasted computation on padding

**Problems:**
- Single-threaded task generation (AICPU bottleneck)
- Stitching overhead vs parallelism tradeoff
- No Human-In-The-Loop control
- Difficulty managing cross-graph dependencies

### 1.3 The Real Goal

Neither pure static nor pure dynamic is sufficient. We need:

> **Structured Dynamism**: Express dynamic patterns in a structured way that
> enables compile-time optimization of the static parts while handling
> runtime-determined values efficiently.

## 2. Analysis of Dynamic Patterns in LLMs

### 2.1 KV Cache Length Variation

```python
# Each batch element has different history length
actSeqKey = [512, 2048, 8192, 32768]  # 64x variation!

# During generation, cache grows each step
for step in range(max_new_tokens):
    k_cache = concat([k_cache, k_new], dim=1)  # Dynamic growth
```

**Characterization:**
- Shape is **runtime-determined** but **bounded**
- Growth pattern is **predictable** (linear, one token at a time)
- Different batches are **independent**

### 2.2 Lightning Indexer TopK Paths

```python
for s1Idx in range(S1):
    effSeq = curSeq - (S1 - s1Idx - 1)  # Runtime-dependent
    actBlock = (effSeq + blockSize - 1) // blockSize  # Dynamic loop bound

    # Path selection based on runtime value
    if effSeq <= 2048:
        # 2K path
    elif effSeq <= 8192:
        # 8K path
    elif effSeq <= 65536:
        # 64K path
    else:
        # 128K path
```

**Characterization:**
- Loop bounds are **runtime-dependent**
- Control flow depends on **runtime values**
- But paths are **discrete** (2K/8K/64K/128K tiers)

### 2.3 MoE Dynamic Routing

```python
# Gate decides which experts to activate - completely runtime-dependent
gate_probs, selected_experts = top_k(gate_logits, k=top_k)

# Scatter tokens to selected experts
for expert_id, tokens in expert_to_tokens.items():
    # Only activated experts compute
```

**Characterization:**
- Expert selection is **data-dependent** (unpredictable)
- Token-to-expert mapping is **sparse** and **irregular**
- Load balancing is **dynamic**

### 2.4 Pattern Classification

| Pattern | Bounded? | Predictable? | Independent? | Strategy |
|---------|----------|--------------|--------------|----------|
| KV Cache growth | Yes (max_len) | Yes (linear) | Yes (batch) | Multi-tier compile |
| Seq len variation | Yes (tiers) | Partially | Yes (batch) | Multi-tier compile |
| TopK path selection | Yes (tiers) | No | Yes (position) | Tier dispatch |
| MoE routing | Yes (experts) | No | Partially | Dynamic gather/scatter |

## 3. Revised Architecture

### 3.1 Three-Level Execution Model

```
┌─────────────────────────────────────────────────────────────────────┐
│ Level 1: Host Machine                                                │
│   - Multi-tier graph compilation (static)                           │
│   - Tier selection based on input characteristics                   │
│   - High-level scheduling strategy specification                    │
├─────────────────────────────────────────────────────────────────────┤
│ Level 2: Control Machine (AICPU Thread 0)                           │
│   - Control flow execution (loops, conditionals)                    │
│   - Task generation with runtime values                             │
│   - Tier instantiation and parameter binding                        │
│   - Scheduling strategy execution                                   │
├─────────────────────────────────────────────────────────────────────┤
│ Level 3: Dispatch Machines (AICPU Threads 1-N)                      │
│   - Task queue management                                           │
│   - Dependency resolution                                           │
│   - Core dispatch and completion tracking                           │
├─────────────────────────────────────────────────────────────────────┤
│ Level 4: Compute Machines (AICore/AIVector)                         │
│   - Kernel execution                                                │
│   - Tile operations (existing PTO-ISA)                              │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.2 Key Design Principles

1. **Multi-Tier Compilation**: Convert O(N) dynamism to O(log N) by pre-compiling for discrete tiers (2K, 8K, 64K, 128K)

2. **Parallel Task Generation**: Allow batch-level parallelism in task generation despite single control thread

3. **Structured Scheduling Hints**: Let users specify scheduling strategies that the runtime executes

4. **Clean Kernel Interface**: Enable hand-written kernels with explicit parameter contracts

5. **Shared Cost Model**: Unify machine model and cost model for accurate scheduling decisions

## 4. Proposed Runtime Primitives

### 4.1 Multi-Tier Graph Definition

```cpp
// Define a tiered computation - compiler generates optimized version for each tier
TIER_BEGIN(flash_attention, SEQ_LEN)
    TIER_CASE(SEQ_LEN <= 2048,  fa_kernel_2k,  workspace_2k)
    TIER_CASE(SEQ_LEN <= 8192,  fa_kernel_8k,  workspace_8k)
    TIER_CASE(SEQ_LEN <= 65536, fa_kernel_64k, workspace_64k)
    TIER_DEFAULT(fa_kernel_128k, workspace_128k)
TIER_END(flash_attention)

// At runtime, select tier based on actual value
TIER_DISPATCH(flash_attention, actual_seq_len, args...)
```

**Benefits:**
- Each tier is calendar-scheduled internally
- O(N) → O(log N) dynamism
- Human can optimize each tier independently

### 4.2 Parallel Task Generation (Batch-Parallel Loops)

```cpp
// Express batch-level parallelism explicitly
PARALLEL_FOR(b, 0, batch_size) {
    // Tasks for different batches can be generated in parallel
    // even though control flow is single-threaded

    FOR(i, 0, seq_len[b]) {  // Sequential within batch
        EMIT_TASK(kernel, batch=b, pos=i, ...)
    }
}
```

**Implementation Strategy:**
- Control thread generates task **descriptors** for all batches first
- Dispatch threads instantiate and execute in parallel
- Avoids serial task generation bottleneck

### 4.3 Scheduling Strategy Specification

```cpp
// User-specified scheduling strategy (Human-In-The-Loop)
SCHEDULE_STRATEGY(attention_schedule) {
    // Option 1: Batch-first (current default - causes cold-start per batch)
    // FOR(task, tasks) FOR(batch, batches) dispatch(batch, task)

    // Option 2: Task-first within batch (better locality)
    FOR(batch, batches) {
        FOR(task, tasks_per_batch) {
            dispatch(batch, task)
        }
    }

    // Option 3: Interleaved (hide latency)
    INTERLEAVE(batch_group_size=4) {
        FOR(batch, batches) FOR(task, tasks) dispatch(batch, task)
    }
}

// Apply strategy to execution
EXECUTE_WITH_STRATEGY(attention_graph, attention_schedule)
```

**Benefits:**
- Users can experiment with scheduling without recompiling kernels
- Runtime executes user-specified order
- Enables the 10% improvement from batch ordering

### 4.4 Explicit Dependency Specification

```cpp
// For stitched graphs, specify dependencies explicitly
GRAPH_BEGIN(combined_attention)
    TASK(qk_matmul, id=1)
    TASK(softmax, id=2, depends={1})
    TASK(pv_matmul, id=3, depends={2})

    // Cross-graph dependencies (for running multiple graphs)
    EXTERNAL_DEP(prev_graph.output, this_graph.input)
GRAPH_END(combined_attention)

// Allow concurrent execution of independent graphs
CONCURRENT_GRAPHS(graph_a, graph_b) {
    // Runtime manages dependencies between concurrent graphs
}
```

### 4.5 Clean Kernel Interface

```cpp
// Kernel declaration with explicit interface
KERNEL_DECL(flash_attention_kernel) {
    // Inputs
    INPUT(q, GlobalTensor<half>, shape=[batch, heads, seq_q, head_dim])
    INPUT(k, GlobalTensor<half>, shape=[batch, heads, seq_k, head_dim])
    INPUT(v, GlobalTensor<half>, shape=[batch, heads, seq_k, head_dim])

    // Outputs
    OUTPUT(o, GlobalTensor<half>, shape=[batch, heads, seq_q, head_dim])

    // Runtime parameters (can vary per invocation)
    PARAM(actual_seq_k, uint32_t)  // Actual sequence length (≤ seq_k)

    // Workspace requirement (for compiler to allocate)
    WORKSPACE(workspace_size_fn(seq_q, seq_k, head_dim))

    // Tiling hint (for scheduler)
    TILE_HINT(cube_bound, seq_q * seq_k * head_dim / tile_size)
}

// Kernel implementation
KERNEL_IMPL(flash_attention_kernel) {
    // Access parameters
    auto q = GET_INPUT(q);
    auto actual_len = GET_PARAM(actual_seq_k);

    // Existing PTO-ISA tile operations
    TLOAD(q_tile, q, offset, shape);
    TMATMUL(qk_tile, q_tile, k_tile);
    // ...
}
```

### 4.6 Root Function (Task Generator) Interface

```cpp
// User-writable root function for custom task generation
ROOT_FUNCTION(deepseek_attention_root) {
    // Runtime parameters
    PARAM(batch_size, uint32_t)
    PARAM(seq_lens, uint32_t*, count=batch_size)  // Per-batch seq lens
    PARAM(kv_cache_lens, uint32_t*, count=batch_size)

    // Task generation logic (runs on AICPU)
    FOR_PARALLEL(b, 0, batch_size) {
        auto seq_len = seq_lens[b];
        auto kv_len = kv_cache_lens[b];

        // Select tier based on runtime value
        auto tier = SELECT_TIER(flash_attention, kv_len);

        // Generate tasks for this batch
        FOR(pos, 0, seq_len) {
            EMIT_TASK(tier.kernel, {
                .batch_idx = b,
                .position = pos,
                .actual_seq = kv_len + pos,
                .workspace = tier.workspace_offset(b, pos),
            });
        }
    }
}
```

## 5. Addressing Key Problems

### 5.1 Single-Threaded Control Flow → Parallel Task Descriptors

**Problem:** Control flow runs on single AICPU, serializing task generation.

**Solution:** Separate task **description** from task **instantiation**:

```
Phase 1 (Control Thread):
  - Generate task DESCRIPTORS for all batches
  - Descriptors are lightweight (kernel_id, param_offsets, deps)
  - O(batch * tasks) descriptor generation, but minimal work per descriptor

Phase 2 (Dispatch Threads, Parallel):
  - Each dispatch thread takes descriptors for assigned batches
  - Instantiate full task (bind params, resolve addresses)
  - Dispatch to cores

Result:
  - Batch parallelism preserved despite single control thread
  - Task generation overhead amortized across dispatch threads
```

### 5.2 Serial Execution Queue → Concurrent Graph Execution

**Problem:** Can't run multiple graphs concurrently due to dependency management complexity.

**Solution:** Explicit dependency protocol:

```cpp
// Each graph declares external dependencies
GRAPH(graph_a) {
    PROVIDES(tensor_x, written_at=task_5)
    REQUIRES(tensor_y, from=external)
}

GRAPH(graph_b) {
    REQUIRES(tensor_x, from=graph_a)
    PROVIDES(tensor_z, written_at=task_3)
}

// Runtime maintains dependency matrix
// [graph_a.task_5] → [graph_b.task_using_x]

// Execution:
START_CONCURRENT(graph_a, graph_b)
// graph_b tasks wait for graph_a.tensor_x automatically
```

### 5.3 No Human-In-The-Loop → Scheduling Strategy DSL

**Problem:** Users know better scheduling but can't express it.

**Solution:** Declarative scheduling strategies:

```cpp
// Strategy DSL - compiled to scheduling instructions
STRATEGY(batch_locality_first) {
    // Hint: Keep same-batch tasks on same core for L2 locality
    AFFINITY(batch_idx, core_group)

    // Hint: Complete one batch before starting another
    ORDERING(batch_idx, SEQUENTIAL)
    ORDERING(task_idx_within_batch, SEQUENTIAL)
}

STRATEGY(latency_hiding) {
    // Hint: Interleave batches to hide memory latency
    ORDERING(batch_idx, INTERLEAVED, group_size=4)

    // Hint: Prefetch next batch while computing current
    PREFETCH(batch_idx + 1, while_computing=batch_idx)
}

// User selects strategy
APPLY_STRATEGY(attention_graph, batch_locality_first)  // 10% faster!
```

### 5.4 Stitching Overhead → Tiered Stitching

**Problem:** More stitching = better steady-state but worse cold-start.

**Solution:** Adaptive stitching with warmup:

```cpp
STITCH_POLICY(adaptive) {
    // Cold-start: Small stitch groups for fast first response
    WARMUP_PHASE(iterations=2, stitch_size=SMALL)

    // Steady-state: Large stitch groups for throughput
    STEADY_PHASE(stitch_size=LARGE)

    // Dynamic adjustment based on queue depth
    IF(queue_depth > threshold) {
        INCREASE_STITCH_SIZE()
    }
}
```

### 5.5 Dependency Management for Running Graphs

**Problem:** Managing dependencies between executing and new graphs is hard.

**Solution:** Version-based tensor tracking:

```cpp
// Each tensor write creates a new version
tensor_x.v1 = graph_a.task_3 writes
tensor_x.v2 = graph_b.task_5 writes

// Dependencies reference specific versions
graph_c.task_1 READS tensor_x.v2
// Runtime ensures graph_b.task_5 completes before graph_c.task_1

// For Read-After-Write, Write-After-Write:
MEMORY_ORDER(tensor_x) {
    WRITE(graph_a.task_3) → READ(graph_c.task_1)  // RAW
    WRITE(graph_a.task_3) → WRITE(graph_b.task_5) // WAW
}
```

## 6. Hardware Model Alignment

### 6.1 Hardware Hierarchy

```
1A (Host CPU)
  └── 1B (AICPU Control) ── 3μs latency
        └── 4B (AICPU Dispatch) ── 0μs latency to C/D
              ├── 24C (Cube Cores)
              └── 48D (Vector Cores)
```

### 6.2 Latency-Aware Design

| Operation | Latency | Design Implication |
|-----------|---------|-------------------|
| A → B (task submit) | 3μs | Batch submissions, minimize round-trips |
| A → C/D (direct) | 3μs | Avoid host-device sync in hot path |
| B → C/D (dispatch) | ~0μs | Dispatch threads critical for parallelism |
| C/D kernel | Variable | Overlap with task generation |

### 6.3 Optimized Execution Flow

```
Time →

Host (A):   [Compile Tiers] ──► [Submit Graph] ──────────────────► [Wait]
                                      │
                                      ▼ 3μs
Control (B): ──────────────► [Gen Descriptors] ──► [Signal Dispatch]
                                                          │
                                                          ▼ ~0
Dispatch (B): ─────────────────────────────────► [Parallel Instantiate]
                                                          │
                                                          ▼ ~0
Cores (C/D): ────────────────────────────────────────► [Execute]
```

## 7. Comparison with Original Proposal

| Aspect | Original (v1) | Revised (v2) |
|--------|---------------|--------------|
| Core assumption | Static task graphs work | Must embrace dynamism |
| Scheduling | Graph-level dependencies | User-specified strategies |
| Task generation | Single-pass | Descriptor + Instantiation |
| Multi-graph | Sequential | Concurrent with explicit deps |
| Optimization | Compile-time fusion | Multi-tier + runtime selection |
| Human control | Hints only | Full scheduling DSL |
| Kernel interface | Implicit | Explicit declaration |

## 8. Implementation Roadmap

### Phase 1: Foundation (1-2 months)
- Clean kernel declaration interface
- Root function interface for hand-written task generators
- Basic tier dispatch mechanism

### Phase 2: Parallel Execution (2-3 months)
- Descriptor-based task generation
- Parallel dispatch threads
- Concurrent graph execution with explicit dependencies

### Phase 3: Scheduling Control (2-3 months)
- Scheduling strategy DSL
- Human-In-The-Loop strategy selection
- Adaptive stitching policies

### Phase 4: Optimization (Ongoing)
- Cost model integration
- Automatic tier selection
- Profile-guided scheduling

## 9. Open Questions

1. **Descriptor format**: How lightweight can task descriptors be while remaining useful?

2. **Dependency protocol**: How to handle WAR/WAW without explicit user annotation?

3. **Strategy compilation**: Should strategies be interpreted or compiled?

4. **Multi-device extension**: How does this model extend to multi-chip?

5. **Backward compatibility**: How to integrate with existing PyPTO graphs?

## 10. Conclusion

The original proposal (v1) tried to apply static graph patterns to fundamentally dynamic workloads. This revised proposal (v2) embraces the dynamic nature of LLM inference while providing structured mechanisms to recover optimization opportunities:

1. **Multi-tier compilation** converts continuous dynamism to discrete choices
2. **Parallel task generation** overcomes single-threaded control bottleneck
3. **Scheduling DSL** enables Human-In-The-Loop optimization
4. **Concurrent graphs** with explicit dependencies enable better resource utilization
5. **Clean interfaces** enable hand-written kernels and root functions

The key insight is that **structure enables optimization**. By providing structured ways to express dynamic patterns, we can optimize the static parts while handling the dynamic parts efficiently.
