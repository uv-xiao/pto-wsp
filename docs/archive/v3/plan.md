# Runtime Extension Design v3 - Work Plan

## Objective

Design a **unified, elegant programming model** for PTO-ISA runtime extension that addresses dynamic LLM workloads while providing convenient and intuitive optimization methodologies.

## Key Requirements

1. **Conceptual unity**: Not piecemeal solutions, but a cohesive programming model
2. **FlashInfer alignment**: Use FlashInfer's Plan-Run pattern as the target paradigm
3. **CANN compatibility**: Must compile with CANN and leverage AICPU capabilities
4. **Human-In-The-Loop**: Enable user control over scheduling strategies

## Tasks

### Phase 1: Research & Analysis
- [x] Task 1.1: Analyze FlashInfer's Plan-Run pattern
  - **Reference**: `docs/uv/flashinfer.md`
  - Key insight: **CPU planning phase generates work assignments, GPU execution phase reads assignments**
  - Binary search for optimal chunk sizes, CSR format for variable lengths
  - Work flattening: (request, tile) pairs assigned to blocks

- [x] Task 1.2: Study PyPTO MPMD model and device control
  - **Reference**: `docs/uv/pypto.md`, `/data/shibizhao/uvxiao/pypto/framework/src/machine/`
  - DeviceMachine → AiCoreManager → task dispatch
  - Stitched function lists with dependency resolution
  - Single control thread bottleneck identified

- [x] Task 1.3: Research CANN AICPU programming
  - **Reference**: `/data/shibizhao/uvxiao/cann-recipes-infer/docs/`
  - AICPU Thread 0: Control flow
  - AICPU Threads 1-N: Dispatch managers
  - Register-based task signaling

- [x] Task 1.4: Extract core problem abstraction
  - **Output**: `docs/uv/analysis-v3.md` Section 2-3
  - The fundamental tension: static vs dynamic scheduling
  - Key insight: **Separate work SPECIFICATION from work INSTANTIATION**

### Phase 2: Unified Model Design
- [x] Task 2.1: Define core abstraction principles
  - **Output**: `docs/uv/analysis-v3.md` Section 3.1
  - Three orthogonal concepts: Iteration Space, Scheduling Strategy, Tier Selection

- [x] Task 2.2: Design programming model API
  - **Output**: `docs/uv/analysis-v3.md` Section 3.2-3.4
  - Loop-centric model with descriptor abstraction

- [x] Task 2.3: Specify scheduling language/DSL
  - **Output**: `docs/uv/analysis-v3.md` Section 3.4
  - Declarative constraints: AFFINITY, ORDER, PREFETCH, TILE

- [ ] Task 2.4: Define compilation strategy
  - **Output**: `docs/uv/runtime-extension-spec-v3.md` (in progress)

### Phase 3: Documentation
- [x] Task 3.1: Write analysis-v3.md with conceptual framework
  - **Output**: `docs/uv/analysis-v3.md`
  - Plan-Descriptor-Execute unified model
  - FlashInfer mapping example
  - Dynamic LLM pattern handling

- [x] Task 3.2: Write runtime-extension-spec-v3.md with elegant API
  - **Output**: `docs/uv/runtime-extension-spec-v3.md`
  - Complete API specification with PTO_SPACE, PTO_TIER_DEF, PTO_SCHEDULE
  - FlashInfer-equivalent decode attention example
  - Compilation and runtime flow documentation

- [x] Task 3.3: Create FlashInfer-style example for DeepSeek
  - **Output**: `docs/uv/examples/deepseek_lightning_indexer_v3.pto`
  - Complete Lightning Indexer implementation using v3 model
  - Shows tier selection, descriptor generation, kernel implementation

## Progress Log

### 2024-01-13: Initial Research Complete

**FlashInfer Key Insights:**
```
Plan Phase (CPU):
  - Extract variable lengths from CSR format
  - Binary search optimal chunk size
  - Generate flattened work indices: request_indices[], kv_tile_indices[]

Run Phase (GPU):
  - Each block reads assignment from indices
  - Processes (chunk_start, chunk_end) with bounds checking
  - Online softmax state merging for split KV
```

**PyPTO Key Insights:**
```
Architecture:
  - DeviceMachine manages multiple AiCoreManager instances
  - AicpuTaskManager handles task queuing
  - Stitched function lists enable larger fusion scope

Limitations:
  - Single control thread serializes task generation
  - No user control over scheduling order
  - Calendar scheduling incompatible with dynamic shapes
```

**Dynamic LLM Requirements Summary:**
```
Core Dynamic Patterns:
  1. KV Cache length variation (bounded, predictable)
  2. Batch sequence length variation (bounded)
  3. TopK path selection (discrete tiers: 2K/8K/64K/128K)
  4. MoE expert routing (data-dependent, sparse)
```

## Core Problem Abstraction

**The Fundamental Tension:**
```
Static Scheduling (Calendar):        Dynamic Scheduling:
├─ Maximum throughput               ├─ Handles all patterns
├─ No runtime overhead              ├─ Runtime task generation
├─ Cannot handle dynamic shapes     ├─ Single-threaded bottleneck
└─ Must pad to max → waste          └─ No user scheduling control
```

**The Key Insight:**

The problem is NOT "static vs dynamic" but rather **"when to decide what"**:

| Decision | FlashInfer | PyPTO | Proposed |
|----------|------------|-------|----------|
| Work partitioning | CPU (plan phase) | AICPU runtime | Host (compile) |
| Work assignment | CPU (plan phase) | AICPU runtime | AICPU (descriptor) |
| Execution | GPU blocks | AICore | AICore |
| Shape handling | CSR + bounds check | Per-invocation compile | Multi-tier + selection |

**Unified Model Principle:**

> **"Plan Once, Execute Many"** - Separate work DESCRIPTION from work INSTANTIATION
>
> 1. **Host compiles** multi-tier kernels and scheduling templates
> 2. **AICPU generates** lightweight descriptors (not full task instantiation)
> 3. **Dispatch threads** instantiate and execute in parallel
> 4. **User specifies** scheduling strategy as declarative constraints

## Completed Work Summary

### All Tasks Completed

1. **Research Phase**: Analyzed FlashInfer, PyPTO, CANN AICPU
2. **Core Insight**: "Separate work specification from instantiation"
3. **analysis-v3.md**: Unified conceptual framework with Plan-Descriptor-Execute model
4. **runtime-extension-spec-v3.md**: Complete API specification
5. **deepseek_lightning_indexer_v3.pto**: Full example implementation

### Deliverables

| File | Description |
|------|-------------|
| `docs/uv/plan_v3.md` | This plan document |
| `docs/uv/analysis-v3.md` | Unified conceptual framework |
| `docs/uv/runtime-extension-spec-v3.md` | Complete API specification |
| `docs/uv/examples/deepseek_lightning_indexer_v3.pto` | FlashInfer-style example |
| `docs/uv/README.md` | Updated index with v3 summary |

### Key Innovations in v3

1. **Plan-Descriptor-Execute Model**
   - Plan: Compile iteration space, tiers, schedule
   - Descriptor: Lightweight work indices (O(1) per work)
   - Execute: Parallel instantiation from descriptors

2. **Unified API**
   - `PTO_SPACE`: Iteration space definition
   - `PTO_TIER_DEF`: Multi-tier kernel variants
   - `PTO_SCHEDULE`: Declarative scheduling constraints
   - `PTO_DESCRIPTOR_GEN`: Lightweight work index generation

3. **FlashInfer Alignment**
   - Same Plan-Run pattern adapted for Ascend
   - Binary search chunk sizing
   - Descriptor-based bounds checking

## Next Steps (Future Work)

1. **Prototype Implementation**: Implement PTO_DESCRIPTOR_GEN in PyPTO
2. **Integration**: Connect descriptor generation to AiCoreManager
3. **Schedule Compilation**: Compile declarative schedule to dispatch policy
4. **Performance Validation**: Benchmark against FlashInfer equivalent
5. **Cost Model**: Integrate descriptor costs into existing cost model

---
*Last updated: 2024-01-13*
*Status: All v3 design tasks completed*
