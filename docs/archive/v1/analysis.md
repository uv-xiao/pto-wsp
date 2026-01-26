# PTO Workload-Schedule Programming (PTO-WSP) Analysis

## Executive Summary

This document analyzes the design space for a "runtime" extension to PTO-ISA that enables system-level scheduling (SPMD/MPMD) and multi-kernel composition. The goal is to bridge the gap between:
- **Current pto-isa**: AICore-level tile operations (compute-focused)
- **pypto runtime**: AICPU-level scheduling and control flow (orchestration-focused)

## 1. Current Abstraction Levels

### 1.1 PTO-ISA Abstraction (AICore Level)

PTO-ISA currently operates at the **AICore execution level**:

```
┌─────────────────────────────────────────────────────────────┐
│ PTO-ISA Current Scope                                        │
├─────────────────────────────────────────────────────────────┤
│ • Tile operations: TLOAD, TSTORE, TMATMUL, TADD, etc.       │
│ • On-chip memory: L0A/L0B/L0C, L1, UB (Unified Buffer)      │
│ • Synchronization: Events within a single kernel            │
│ • Execution model: Single kernel, SPMD across blockDim      │
│ • Control: Scalar unit drives Vector/Cube/MTE pipelines     │
└─────────────────────────────────────────────────────────────┘
```

**Key characteristics:**
- **Tile-granular**: Operations on 2D tiles (MatTile, VecTile, etc.)
- **Pipeline-aware**: Explicit events for Vector/Cube/MTE synchronization
- **Single-kernel scope**: One entry point, one blockDim, no inter-kernel deps
- **Memory hierarchy**: GM → L2 → L1 → L0 (hardware-managed caching)

### 1.2 Hardware Constraints

From Ascend documentation:

| Resource | Constraint | Impact |
|----------|------------|--------|
| ICache | 32KB | Limits kernel binary size |
| DCache | 32KB | Limits per-core data footprint |
| L0C Buffer | 128KB | Matrix accumulation capacity |
| L1 Buffer | Platform-dependent | Tile staging capacity |
| UB | Platform-dependent | Vector compute I/O |
| Instruction Queue | Depth 32 (MTE1/MMAD) | Async operation limit |

**Critical constraint**: ICache (32KB) fundamentally limits single-kernel complexity. A "mega-kernel" approach for whole-model execution is impractical.

### 1.3 pypto Runtime Abstraction (AICPU Level)

pypto runtime operates at the **AICPU orchestration level**:

```
┌─────────────────────────────────────────────────────────────┐
│ pypto Runtime Scope (Above PTO-ISA)                          │
├─────────────────────────────────────────────────────────────┤
│ • Kernel DAG: Multiple kernels with dependencies            │
│ • Task scheduling: AICPU threads dispatch to AICore         │
│ • Dependency resolution: Dataflow-driven execution          │
│ • Memory management: Workspace allocation, caching          │
│ • Control flow: Dynamic graphs, conditional execution       │
│ • Multi-stream: Parallel kernel execution paths             │
└─────────────────────────────────────────────────────────────┘
```

**Key components:**
- **DeviceCtrlMachine**: Interprets control flow binary, generates tasks
- **AiCoreManager**: Dispatches kernels to available cores
- **CoreFunctionTopo**: Dependency graph (readyCount, depNum)
- **ReadyCoreFunctionQueue**: Lock-free task queues

## 2. Gap Analysis

### 2.1 What pypto Generates Beyond PTO-ISA

```
┌─────────────────────────────────────────────────────────────┐
│ Beyond PTO-ISA (pypto generates this)                        │
├─────────────────────────────────────────────────────────────┤
│ 1. Kernel Binary + Metadata                                  │
│    • CoreFunctionWsAddr: binary addr, entry point, topo     │
│    • Workspace layout and size requirements                 │
│                                                             │
│ 2. Dependency Graph                                          │
│    • CoreFunctionTopo: per-kernel dependency info           │
│    • readyCount: remaining dependencies before execution    │
│    • Successor list for dependency resolution               │
│                                                             │
│ 3. Control Flow Binary                                       │
│    • Dynamic task generation logic                          │
│    • Conditional execution paths                            │
│    • Loop unrolling and iteration management                │
│                                                             │
│ 4. Device Arguments                                          │
│    • Tensor descriptors and addresses                       │
│    • Tiling parameters and configuration                    │
│    • Core allocation (nrAic, nrAiv, blockDim)               │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Gap: No Standard Representation

Currently, there's no standard ISA-level representation for:
1. **Inter-kernel dependencies**: How kernel A's output feeds kernel B
2. **Task graph structure**: DAG of kernels with dataflow edges
3. **Dynamic control flow**: Conditionals, loops at kernel granularity
4. **Core allocation**: How to partition cores across kernel types

## 3. Key Design Questions

### 3.1 Do We Need a New Abstraction Level?

**Analysis:**

| Approach | Pros | Cons |
|----------|------|------|
| **No new level** (mega-kernel) | Simpler ISA | ICache limit (32KB), unmaintainable |
| **Compiler-only** (pypto status quo) | Works today | Proprietary, not portable |
| **ISA extension** | Portable, standardized | Design complexity |

**Conclusion**: A new abstraction level is needed. The ICache constraint (32KB) makes mega-kernels impractical for complex operators like DeepSeek V3.2's MLA+FA+Indexer pipeline.

### 3.2 What Should the New Level Represent?

The new level should correspond to **AICPU/Device-CPU scheduling**:

```
┌─────────────────────────────────────────────────────────────┐
│ Proposed: PTO Workload-Schedule Programming (PTO-WSP) framework                              │
├─────────────────────────────────────────────────────────────┤
│ Level 0: Host Machine (compilation, graph submission)        │
│ Level 1: Device Machine (task graph execution) ← NEW         │
│ Level 2: Core Machine (tile operations) ← Existing PTO-ISA   │
└─────────────────────────────────────────────────────────────┘
```

### 3.3 Can We Use Existing PTO-ISA for Large Kernels?

**No, for several reasons:**

1. **ICache (32KB)**: Complex kernels exceed binary size limits
2. **Register pressure**: Many live tiles exhaust L0/UB capacity
3. **Compilation time**: JIT compilation of mega-kernels is slow
4. **Debuggability**: Monolithic kernels are hard to profile/optimize

**Evidence from DeepSeek V3.2:**
- MLA Prolog: ~2-3KB binary
- Flash Attention: ~5-8KB binary
- Lightning Indexer: ~3-5KB binary
- Combined (if possible): Would exceed ICache

## 4. Proposed Runtime Extension ISA

### 4.1 Design Principles

1. **Minimal but complete**: Only add what's needed for kernel composition
2. **Performance-first**: Enable all pypto optimizations at ISA level
3. **Portable**: Work across A2/A3/A5 and future platforms
4. **Composable**: Allow nesting and recursive graphs

### 4.2 New Abstractions

#### 4.2.1 Task Graph (`TG_*` Instructions)

```
TG_BEGIN(graph_id)              // Begin task graph definition
TG_END(graph_id)                // End task graph definition

TG_KERNEL(kernel_id, entry, blockDim, ...)  // Define a kernel node
TG_DEPEND(src_kernel, dst_kernel, type)     // Define dependency edge
TG_LAUNCH(graph_id, stream)                  // Submit graph for execution
TG_SYNC(graph_id)                            // Wait for graph completion
```

#### 4.2.2 Memory Coordination (`MC_*` Instructions)

```
MC_ALLOC(tensor_id, size, alignment)   // Allocate workspace
MC_FREE(tensor_id)                     // Release workspace
MC_BIND(kernel_id, arg_idx, tensor_id) // Bind tensor to kernel arg
MC_FENCE(type)                         // Memory fence (GM, L2, etc.)
```

#### 4.2.3 Control Flow (`CF_*` Instructions)

```
CF_IF(condition)       // Conditional execution start
CF_ELSE                // Else branch
CF_ENDIF               // End conditional
CF_LOOP(count)         // Loop start with iteration count
CF_ENDLOOP             // Loop end
CF_BREAK               // Early loop exit
```

#### 4.2.4 Core Allocation (`CA_*` Instructions)

```
CA_REQUEST(kernel_id, aic_count, aiv_count)  // Request core allocation
CA_RELEASE(kernel_id)                         // Release cores
CA_BARRIER(kernel_set)                        // Sync across kernels
```

### 4.3 Example: DeepSeek Indexer Attention

```
// Task Graph Definition
TG_BEGIN(deepseek_attn)

  // Define kernels
  TG_KERNEL(mla_prolog, mla_entry, 24, ...)
  TG_KERNEL(indexer_prolog, indexer_entry, 24, ...)
  TG_KERNEL(lightning_indexer, li_entry, 24, ...)
  TG_KERNEL(sparse_fa, fa_entry, 24, ...)

  // Define dependencies (dataflow edges)
  TG_DEPEND(mla_prolog, sparse_fa, AFTER)       // q,k,v → attention
  TG_DEPEND(indexer_prolog, lightning_indexer, AFTER)
  TG_DEPEND(lightning_indexer, sparse_fa, AFTER) // indices → gather

  // Memory bindings
  MC_ALLOC(workspace, 128*1024*1024, 512)
  MC_BIND(mla_prolog, 0, input_hidden)
  MC_BIND(mla_prolog, 1, q_out)
  MC_BIND(sparse_fa, 0, q_out)
  // ... more bindings

TG_END(deepseek_attn)

// Execute
TG_LAUNCH(deepseek_attn, compute_stream)
TG_SYNC(deepseek_attn)
```

### 4.4 Relationship to Existing PTO-ISA

```
┌─────────────────────────────────────────────────────────────┐
│ Runtime Extension (TG_*, MC_*, CF_*, CA_*)                   │
│   Operates at: Device Machine level (AICPU)                  │
│   Controls: Kernel scheduling, memory, control flow          │
├─────────────────────────────────────────────────────────────┤
│ Existing PTO-ISA (T*, M*)                                    │
│   Operates at: Core Machine level (AICore)                   │
│   Controls: Tile operations, pipelines, events               │
└─────────────────────────────────────────────────────────────┘

Interface: TG_KERNEL creates a reference to a PTO-ISA kernel
           The kernel body uses existing tile instructions
```

## 5. Optimization Opportunities

### 5.1 Enabled by Runtime Extension

| Optimization | Description | Impact |
|--------------|-------------|--------|
| **Kernel fusion** | Combine small kernels | Reduce launch overhead |
| **Memory reuse** | Share workspace across kernels | Reduce GM traffic |
| **Pipeline overlap** | Execute independent kernels in parallel | Hide latency |
| **Dynamic shapes** | Conditional kernel selection | Handle variable batch/seq |
| **L2 cache locality** | Co-schedule kernels using same data | Improve bandwidth |

### 5.2 DeepSeek V3.2 Specific Optimizations

```
Without Runtime Extension:
  MLA Prolog → sync → Indexer Prolog → sync → Lightning Indexer → sync → Sparse FA
  Total syncs: 3 kernel boundaries

With Runtime Extension:
  ┌─────────────────┐     ┌─────────────────────┐
  │ MLA Prolog      │────►│ Sparse FA           │
  └────────┬────────┘     └───────────▲─────────┘
           │                          │
  ┌────────▼────────┐     ┌───────────┴─────────┐
  │ Indexer Prolog  │────►│ Lightning Indexer   │
  └─────────────────┘     └─────────────────────┘

  Benefits:
  - MLA Prolog and Indexer Prolog execute in parallel (independent)
  - Sparse FA waits only for actual dependencies
  - Memory can be pre-allocated for entire graph
```

### 5.3 Performance Model

Expected improvements from runtime extension:

| Scenario | Without Extension | With Extension | Improvement |
|----------|-------------------|----------------|-------------|
| 4-kernel pipeline | 4 syncs | 2 syncs | 2x overlap |
| Shared workspace | 4 × alloc | 1 × alloc | Memory reuse |
| Parallel kernels | Sequential | Concurrent | Latency hiding |

## 6. Implementation Considerations

### 6.1 Compilation Strategy

```
Source (Python/C++)
    ↓
PTO-Auto/Manual (per-kernel)
    ↓
Kernel binaries + metadata
    ↓
Runtime Extension (graph definition)
    ↓
Device Program (combined)
    ↓
Execution (AICPU interprets, AICore executes)
```

### 6.2 Hardware Support Required

| Feature | A2/A3 Support | A5 Support | Notes |
|---------|---------------|------------|-------|
| AICPU scheduling | Yes | Yes | Existing |
| Dependency resolution | Yes | Yes | Existing |
| Multi-stream | Yes | Yes | Existing |
| L2 cache hints | Partial | Yes | New for extension |
| Dynamic control flow | Yes | Yes | Software-based |

### 6.3 Backward Compatibility

The extension is **additive**:
- Existing PTO-ISA kernels work unchanged
- Runtime extension wraps existing kernels
- Gradual adoption: start with simple graphs

## 7. Alternative Approaches Considered

### 7.1 Mega-Kernel (Rejected)

**Why rejected:**
- ICache limit (32KB) makes large kernels impractical
- Compilation time explosion
- Debugging nightmare
- No dynamic shape support

### 7.2 Host-Only Scheduling (Current pypto)

**Why insufficient:**
- Host-device round-trips add latency
- No standard representation
- Framework-specific (pypto only)

### 7.3 Pure Dataflow (MLIR-style)

**Why not chosen:**
- Too abstract for performance-critical code
- Loses tile-level control
- Compilation complexity

## 8. Recommendations

### 8.1 Short-Term (3-6 months)

1. **Document pypto's runtime model** formally
2. **Prototype TG_* instructions** in CPU simulator
3. **Create DeepSeek V3.2 example** using proposed extension

### 8.2 Medium-Term (6-12 months)

1. **Implement TG_*/MC_* for NPU** (A3 first)
2. **Add CF_* control flow** support
3. **Benchmark against native pypto**

### 8.3 Long-Term (12+ months)

1. **Standardize across platforms** (A5, future chips)
2. **Enable third-party adoption** (TileLang, etc.)
3. **Hardware acceleration** for graph dispatch

## 9. Conclusion

A runtime extension to PTO-ISA is necessary to enable portable, high-performance multi-kernel composition. The proposed design:

1. **Adds a Device Machine layer** (TG_*, MC_*, CF_*, CA_* instructions)
2. **Preserves existing Core Machine** (T*, M* instructions unchanged)
3. **Enables key optimizations** (fusion, overlap, memory reuse)
4. **Supports complex operators** like DeepSeek V3.2's attention pipeline

The extension fills the gap between tile-level operations and system-level scheduling, providing a standard representation for what pypto currently generates as proprietary control flow.

## Appendix A: Glossary

| Term | Definition |
|------|------------|
| AICore | Matrix/Vector compute unit on Ascend |
| AICPU | ARM64 control processor on Ascend device |
| Tile | 2D on-chip buffer, unit of PTO operations |
| Task Graph | DAG of kernels with dataflow dependencies |
| blockDim | Number of logical cores for kernel execution |
| L0A/L0B/L0C | Cube input/output buffers |
| UB | Unified Buffer for vector operations |
| GM | Global Memory (device DRAM) |

## Appendix B: References

1. PTO-ISA Documentation: `docs/PTOISA.md`
2. PTO Programming Model: `docs/coding/ProgrammingModel.md`
3. PTO Machine Model: `docs/machine/abstract-machine.md`
4. pypto Runtime: `pypto/docs/compile_exec/05_runtime.md`
5. Ascend C Programming Guide: hiascend.com
6. DeepSeek V3.2 Implementation: `pypto/examples/models/deepseek_v32_exp/`
