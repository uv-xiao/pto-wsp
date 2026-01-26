# PTO Workload-Schedule Programming (PTO-WSP): Conceptual Analysis v4

## Executive Summary

This document presents a unified conceptual framework for extending PTO-ISA to support dynamic LLM workloads. Drawing from five research areas—FlashInfer architecture, GPU acceleration patterns, programming language design, PTO-ISA internals, and Ascend hardware—we identify three fundamental design principles:

1. **Separate work specification from execution**: Define work structure at planning time, instantiate at runtime
2. **Support both SPMD and MPMD programming models**: Enable single-program (all cores same kernel) and multi-program (different kernels) execution
3. **Leverage AICPU for zero-latency dispatch**: Unlike GPU, Ascend's AICPU can dispatch to AICores with ~0μs latency

The extension enables FlashInfer-equivalent execution on Ascend NPU while exploiting Ascend's unique hardware advantages.

## 1. The Fundamental Problem

### 1.1 The Static-Dynamic Tension

Modern LLM inference exhibits four classes of dynamism:

| Class | Example | Predictability | Current PTO-ISA Support |
|-------|---------|----------------|-------------------------|
| **D1: Bounded** | Batch size (1-64) | High | Partial (via multi-tier) |
| **D2: Range** | Sequence length (1-128K) | Medium | None (compile-time loops) |
| **D3: Tiered** | KV length tiers | High | None (single kernel) |
| **D4: Data-Dependent** | MoE expert routing | Low | None |

PTO-ISA excels at static workloads where shapes are known at compile time. The challenge is extending this model to handle D1-D3 patterns efficiently, while providing a path for D4.

### 1.2 Why Existing Approaches Fall Short

**Static Compilation (Calendar Scheduling)**:
- Maximum throughput for fixed shapes
- Cannot handle variable sequence lengths without padding
- Padding waste: up to 64× for batched decode with varying KV lengths

**Dynamic Control Flow (PyPTO)**:
- Handles any shape variation
- Single-threaded AICPU control creates ~3μs overhead per kernel
- For 1000 work units: 3ms overhead (unacceptable for inference latency)

**Per-Shape JIT Compilation**:
- Flexibility of dynamic approach
- Compilation latency makes it unsuitable for real-time inference
- Memory pressure from many compiled variants

### 1.3 The Key Insight

The fundamental tension is not "static vs dynamic" but rather **when decisions are made**:

```
Decision          Static Approach    Dynamic Approach    Ideal
──────────────────────────────────────────────────────────────
Iteration space   Compile time       Never (implicit)    Compile time
Work partitioning Compile time       Runtime (AICPU)     Compile time
Work assignment   Compile time       Runtime (AICPU)     Runtime (Host)
Bounds checking   Compile time       Runtime (control)   Runtime (Data)
Kernel variant    Compile time       N/A                 Runtime (Select)
```

The "Ideal" column represents the Plan-Descriptor-Execute model: define the structure at compile time, but defer the actual values to runtime via lightweight descriptors.

## 2. Design Principles

### 2.1 Principle 1: Support Both SPMD and MPMD

The extension must support two programming models:

**SPMD (Single Program, Multiple Data)**:
- All AICores execute the same kernel
- Each core processes different data (identified by `get_block_idx()`)
- Current PTO-ISA model; most efficient for homogeneous workloads

```cpp
// SPMD: All cores run same kernel, different data
__global__ void attention_kernel(__gm__ WorkDescriptor* descs, ...) {
    WorkDescriptor desc = descs[get_block_idx()];
    // All cores execute same code path
}
```

**MPMD (Multiple Program, Multiple Data)**:
- Different AICores can execute different kernels
- Required for: multi-tier kernels, producer-consumer pipelines, MoE expert routing
- Enables heterogeneous workload distribution

```cpp
// MPMD: Different cores run different kernels based on descriptor
__global__ void dispatch_kernel(__gm__ WorkDescriptor* descs, ...) {
    WorkDescriptor desc = descs[get_block_idx()];
    switch (desc.tier) {
        case 0: kernel_tier0(desc, ...); break;
        case 1: kernel_tier1(desc, ...); break;
        case 2: kernel_tier2(desc, ...); break;
    }
}
```

**Why Both Matter**:
| Pattern | SPMD | MPMD |
|---------|------|------|
| Uniform batch (same seq_len) | ✅ Optimal | Unnecessary overhead |
| Variable seq_len (multi-tier) | ❌ Suboptimal | ✅ Per-tier optimization |
| MoE expert routing | ❌ Cannot express | ✅ Different experts |
| Pipeline stages | ❌ Cannot express | ✅ Producer/consumer |

### 2.2 Principle 2: Bounds as Data, Not Control

FlashInfer's critical insight: encode work assignments in arrays, not control flow.

```cpp
// Anti-pattern: Control flow determines bounds
for (int req = 0; req < batch; req++) {
    for (int chunk = 0; chunk < seq_lens[req] / chunk_size; chunk++) {
        launch_kernel(req, chunk, ...);  // Serial launches!
    }
}

// Pattern: Bounds in data (descriptors)
WorkDescriptor descs[total_work];
generate_descriptors(descs, seq_lens, batch, chunk_size);  // O(N), once
launch_all(descs, total_work);  // Single parallel launch
```

**Rationale**: Moving bounds from control flow to data enables parallel work assignment and eliminates AICPU serialization.

### 2.3 Principle 3: Plan Once, Execute Many (AICPU-Based)

Separate the three phases, with descriptor generation on **AICPU** (not host):

1. **Plan (Compile Time)**: Define iteration space structure, compile kernel tiers
2. **Describe (Runtime, AICPU)**: Generate lightweight descriptors with actual bounds
3. **Execute (Runtime, AICore)**: Each AICore reads its descriptor, executes

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│     PLAN        │     │    DESCRIBE     │     │    EXECUTE      │
│                 │     │  (AICPU Threads)│     │    (AICore)     │
│ • Space shape   │────►│ • Binary search │────►│ • Read desc[i]  │
│ • Tier bounds   │     │ • Gen descriptors│    │ • Select tier   │
│ • Kernel code   │     │ • Write to GM   │     │ • Run kernel    │
│                 │     │                 │     │                 │
│ Once per kernel │     │ Once per batch  │     │ Per AICore      │
│                 │     │ (0μs dispatch)  │     │                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

**Critical Hardware Insight** (from Ascend architecture):
- Host → AICPU: **~3μs latency** (cross-domain communication)
- **AICPU → AICore: ~0μs latency** (register-based dispatch, ALL AICPU threads)
- The bottleneck is not dispatch latency, but **task generation throughput**

**Rationale**: Unlike FlashInfer (which uses CPU planning because GPU lacks an equivalent to AICPU), Ascend's AICPU can dispatch to AICores with **zero latency**. This makes AICPU-based planning superior to host-based planning for latency-sensitive workloads. The key is to use **multiple AICPU threads** for parallel descriptor generation to overcome the single-thread task generation bottleneck.

### 2.4 Principle 4: Compile-Time Polymorphism for Tiers

Different sequence length ranges benefit from different tile sizes:

| Tier | Sequence Range | Optimal Tile | Rationale |
|------|----------------|--------------|-----------|
| 0 | 1-1024 | 256 | Fits in L1, minimal memory traffic |
| 1 | 1025-4096 | 512 | Balance compute/memory |
| 2 | 4097-16384 | 1024 | Maximize reuse |
| 3 | 16385-131072 | 2048 | Memory bound, large prefetch |

Compile all tiers, select at runtime:

```cpp
template <int Tier> void kernel_impl(...);  // Tier-specific optimizations

void dispatch(WorkDescriptor desc, ...) {
    static constexpr auto table = {&kernel_impl<0>, &kernel_impl<1>, ...};
    table[desc.tier](desc, ...);
}
```

**Rationale**: Compile-time polymorphism provides maximum optimization per tier without runtime code generation.

## 3. Conceptual Model

### 3.1 The Descriptor as Universal Interface

A `WorkDescriptor` is a compact, fixed-size struct that fully specifies a work unit:

```
┌─────────────────────────────────────────────────────────┐
│                    WorkDescriptor                        │
├─────────────────────────────────────────────────────────┤
│  work_id: 4 bytes   │ Unique identifier                 │
│  tier: 1 byte       │ Kernel variant to use             │
│  flags: 1 byte      │ First/last chunk, init, etc.      │
│  reserved: 2 bytes  │ Alignment padding                 │
│  params[4]: 16 bytes│ Pattern-specific (req, head, etc.)│
├─────────────────────────────────────────────────────────┤
│  Total: 24 bytes, 8-byte aligned                        │
└─────────────────────────────────────────────────────────┘
```

**Design Decisions**:
- **Fixed size**: Enables array allocation without complex metadata
- **Generic params**: Pattern-specific accessors provide type safety without bloat
- **Flags field**: Eliminates conditional logic for chunk boundaries
- **Tier field**: Single-byte limits to 256 tiers (far more than needed)

### 3.2 The Iteration Space as Template

An `IterationSpace` defines the logical work structure:

```cpp
template <typename... Dims>
class IterationSpace {
    // Compile-time: dimension count, static sizes
    // Runtime: dynamic dimension sizes
    // Operations: total_work(), index_to_coords()
};
```

**Key Properties**:
- **Mixed static/dynamic**: `IterationSpace<Dim<32>, Dim<DYNAMIC>, Dim<128>>`
- **Variadic**: Not limited to 3D (attention) patterns
- **Composable**: Can nest or combine spaces

### 3.3 The Planner as Algorithm

A `WorkPlanner` encapsulates the planning algorithm:

```cpp
template <typename Space, typename TierConfig, typename PatternParams>
class WorkPlanner {
    int plan_chunk_size(...);      // Binary search (FlashInfer pattern)
    PlanResult generate(...);       // Produce descriptors
};
```

**Algorithm**:
1. Binary search for chunk size that yields ≤ max_work_units
2. Iterate over space, generating one descriptor per work unit
3. Set tier based on sequence length
4. Set flags for first/last chunks

### 3.4 The Dispatch as Table Lookup

Kernel dispatch uses a static table to avoid branch misprediction:

```cpp
constexpr KernelFn dispatch_table[] = {
    &kernel_impl<0>,
    &kernel_impl<1>,
    &kernel_impl<2>,
    &kernel_impl<3>,
};

// O(1) dispatch, no switch
dispatch_table[desc.tier](desc, ...);
```

**Rationale**: Static table eliminates branches in hot path, maximizes instruction cache efficiency.

## 4. Relationship to Existing Work

### 4.1 FlashInfer Correspondence

| FlashInfer Concept | PTO-ISA Extension | Notes |
|--------------------|-------------------|-------|
| `request_indices[]` | `WorkDescriptor.params[0]` | Request assignment |
| `kv_tile_indices[]` | `WorkDescriptor.params[2,3]` | Chunk bounds |
| `PrefillSplitQOKVIndptr` | `WorkPlanner.generate()` | Planning algorithm |
| `BinarySearchKVChunkSize` | `WorkPlanner.plan_chunk_size()` | Chunk optimization |
| `state_t::merge()` | Existing FA macros | Online softmax |

### 4.2 Halide/TVM Correspondence

| PL Concept | PTO-ISA Extension | Notes |
|------------|-------------------|-------|
| Algorithm-Schedule separation | Space + Planner | What vs How |
| Symbolic shapes | `Dim<DYNAMIC>` | Runtime bounds |
| Loop tiling | Chunk generation | Work partitioning |
| Multi-version | Tier compilation | Size-optimized variants |

### 4.3 PTO-ISA Internal Correspondence

| Existing Feature | Extension Feature | Notes |
|------------------|-------------------|-------|
| `Shape<DYNAMIC, ...>` | `Dim<DYNAMIC>` | Same pattern |
| `GlobalTensor` stride | `WorkDescriptor` params | Runtime metadata |
| `MAP_INSTR_IMPL` | Tier dispatch table | Platform abstraction |
| `Tile::SetValidRegion` | Descriptor bounds | Runtime sizing |

## 5. Performance Model

### 5.1 Ascend Hardware Latency Model

Understanding the hardware is critical for correct architecture:

```
1A (Host CPU)
  └── 16B (AICPU) ── ~3μs latency from Host
        └── All AICPU Threads ── ~0μs latency to AICores (register-based)
              ├── 24C (Cube Cores)
              └── 48D (Vector Cores)
```

**Key Insight**: ALL AICPU → AICore dispatch has **~0μs latency** (register-based signaling). The bottleneck is not dispatch latency but **task generation throughput**—a single thread can only generate tasks serially.

### 5.2 Overhead Analysis (AICPU-Based)

| Component | Cost | Amortization |
|-----------|------|--------------|
| **Plan (AICPU Manager)** | ~50μs for 10K work units | Per batch |
| **Descriptor generation** | O(N/P) with P threads | Parallelizable |
| **AICPU→AICore dispatch** | **0μs** (register-based) | Per work unit |
| **Read descriptor** | ~100 cycles per AICore | Per work unit |
| **Tier dispatch** | ~10 cycles per AICore | Per work unit |
| **Kernel execution** | 10K-100K cycles | Per work unit |

### 5.3 Comparison with Alternatives

| Approach | Dispatch Latency | Task Generation | Flexibility |
|----------|------------------|-----------------|-------------|
| Static compilation | 0μs | Compile-time | None |
| Per-shape JIT | 0μs (after compile) | 10-100ms | Full |
| Host-based planning | ~3μs (Host→AICPU) | Parallel on host | D1-D3 |
| AICPU (single thread) | **~0μs** | Serial | D1-D3 |
| **AICPU (multi-thread)** | **~0μs** | **Parallel** | **D1-D3** |

**Why AICPU-based planning is preferred for latency-sensitive inference**:
1. Zero dispatch latency (AICPU→AICore is register-based)
2. No H2D copy for descriptors (already in device memory)
3. Can parallelize task generation across multiple AICPU threads

### 5.4 Scalability

The AICPU-based descriptor approach scales linearly:
- **Work units**: O(batch × heads × chunks)
- **Planning**: O(batch × heads × chunks / P) with P AICPU threads
- **Dispatch**: O(1) per AICore (register-based, 0μs)
- **Execution**: O(1) per AICore (descriptor lookup)

## 6. Limitations and Future Work

### 6.1 Current Limitations

1. **D4 (Data-Dependent)**: MoE routing requires runtime expert selection, not fully addressed
2. **Cross-Device**: Multi-NPU descriptor distribution not specified
3. **Memory Management**: Descriptor buffer allocation strategy left to implementation
4. **Cost Model**: No automatic tier selection based on hardware characteristics

### 6.2 Future Extensions

1. **Adaptive Tiers**: Runtime profiling to adjust tier boundaries
2. **Hierarchical Spaces**: Nested iteration spaces for complex patterns
3. **Schedule DSL**: Declarative constraints (affinity, ordering, prefetch)
4. **MoE Support**: Sparse descriptor generation for expert routing

## 7. Conclusion

The PTO Workload-Schedule Programming (PTO-WSP) addresses the static-dynamic tension through three core design principles:

1. **SPMD + MPMD Support**: Enable both single-program (all cores same kernel) and multi-program (different kernels) execution for maximum flexibility
2. **Descriptor-Based Dispatch**: Work specification via data arrays, not control flow, enabling parallel execution
3. **AICPU-Based Planning**: Leverage Ascend's ~0μs AICPU→AICore dispatch for latency-sensitive inference

**Key Architectural Insights**:

| Aspect | GPU (FlashInfer) | Ascend (Extension) |
|--------|------------------|-------------------|
| Planning location | CPU (no AICPU equivalent) | AICPU (preferred) |
| Dispatch latency | CUDA launch (~μs) | AICPU→AICore (~0μs) |
| Programming model | SPMD only | **SPMD + MPMD** |

The extension exploits Ascend's unique hardware advantage: ALL AICPU→AICore dispatch has ~0μs latency (register-based). This means:
- Descriptor generation can happen on AICPU with zero dispatch overhead
- Multiple AICPU threads can parallelize task generation
- No H2D copy overhead for descriptors

Implementation choices (like reusing `DYNAMIC` constant) are secondary to these core principles.

---
*Analysis Version: 4.2*
*Last Updated: 2024-01-14*
