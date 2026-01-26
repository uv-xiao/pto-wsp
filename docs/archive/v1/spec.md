# PTO Workload-Schedule Programming (PTO-WSP) framework ISA Specification

**Version:** 0.1 (Draft)
**Status:** Proposal

## 1. Overview

The PTO Workload-Schedule Programming (PTO-WSP) framework provides instructions for multi-kernel composition, task scheduling, and device-level control flow. It operates at the **Device Machine** level, above the existing **Core Machine** (tile operations).

```
┌─────────────────────────────────────────────────────────────┐
│ Host Machine                                                 │
│   Compilation, graph submission, host-device sync           │
├─────────────────────────────────────────────────────────────┤
│ Device Machine ← Runtime Extension                           │
│   Task graphs, dependencies, control flow, memory mgmt       │
├─────────────────────────────────────────────────────────────┤
│ Core Machine ← Existing PTO-ISA                              │
│   Tile operations, pipelines, events                        │
└─────────────────────────────────────────────────────────────┘
```

## 2. Instruction Categories

| Category | Prefix | Purpose |
|----------|--------|---------|
| Task Graph | `TG_` | Define and execute kernel DAGs |
| Memory Coordination | `MC_` | Allocate and bind workspace |
| Control Flow | `CF_` | Conditionals, loops at kernel granularity |
| Core Allocation | `CA_` | Partition cores across kernel types |
| Synchronization | `SY_` | Cross-kernel barriers and signals |

## 3. Task Graph Instructions

### 3.1 TG_BEGIN

**Syntax:**
```
TG_BEGIN(graph_id: uint32)
```

**Description:**
Begin definition of a task graph. All subsequent TG_KERNEL and TG_DEPEND instructions until TG_END are part of this graph.

**Parameters:**
- `graph_id`: Unique identifier for this graph (0 to MAX_GRAPH_ID)

**Constraints:**
- Cannot nest TG_BEGIN/TG_END pairs
- graph_id must be unique within a compilation unit

---

### 3.2 TG_END

**Syntax:**
```
TG_END(graph_id: uint32)
```

**Description:**
End definition of a task graph. After TG_END, the graph is finalized and ready for execution.

**Parameters:**
- `graph_id`: Must match the corresponding TG_BEGIN

---

### 3.3 TG_KERNEL

**Syntax:**
```
TG_KERNEL(kernel_id: uint32,
          entry_symbol: symbol,
          blockDim: uint32,
          core_type: CoreType,
          workspace_size: uint64)
```

**Description:**
Define a kernel node within the current task graph.

**Parameters:**
- `kernel_id`: Unique identifier within this graph
- `entry_symbol`: Symbol reference to kernel entry point
- `blockDim`: Number of logical cores (1 to 65535)
- `core_type`: One of {AIC, AIV, AIC_AIV, AICPU}
- `workspace_size`: Per-invocation workspace requirement (bytes)

**Core Types:**
```
enum CoreType {
    AIC = 0,      // Cube cores only
    AIV = 1,      // Vector cores only
    AIC_AIV = 2,  // Mixed (e.g., FA with cube and vector stages)
    AICPU = 3     // AICPU execution (rare)
};
```

---

### 3.4 TG_DEPEND

**Syntax:**
```
TG_DEPEND(src_kernel: uint32,
          dst_kernel: uint32,
          dep_type: DependencyType,
          tensor_id: uint32 = 0)
```

**Description:**
Define a dependency edge from src_kernel to dst_kernel.

**Parameters:**
- `src_kernel`: Producer kernel ID
- `dst_kernel`: Consumer kernel ID
- `dep_type`: Type of dependency
- `tensor_id`: Optional tensor that creates the dependency

**Dependency Types:**
```
enum DependencyType {
    AFTER = 0,       // dst starts after src completes (default)
    DATA = 1,        // dst depends on tensor produced by src
    BARRIER = 2,     // Full sync point (all-to-all)
    SOFT = 3         // Hint only (scheduler may ignore)
};
```

**Semantics:**
- `AFTER`: dst_kernel.readyCount decrements when src_kernel completes
- `DATA`: Additionally ensures tensor is visible to dst_kernel
- `BARRIER`: All kernels before barrier complete before any after
- `SOFT`: Advisory dependency (e.g., for cache locality)

---

### 3.5 TG_LAUNCH

**Syntax:**
```
TG_LAUNCH(graph_id: uint32, stream: StreamHandle)
```

**Description:**
Submit a task graph for execution on the specified stream.

**Parameters:**
- `graph_id`: Graph to execute
- `stream`: Device stream handle (from runtime)

**Behavior:**
- Non-blocking: returns immediately
- Graph execution follows dependency order
- Multiple graphs can execute concurrently on different streams

---

### 3.6 TG_SYNC

**Syntax:**
```
TG_SYNC(graph_id: uint32)
```

**Description:**
Wait for a task graph to complete.

**Parameters:**
- `graph_id`: Graph to wait for

**Behavior:**
- Blocking: returns when all kernels in graph have completed

---

### 3.7 TG_INVOKE

**Syntax:**
```
TG_INVOKE(graph_id: uint32, iteration: uint32)
```

**Description:**
For iterative graphs, invoke a single iteration.

**Parameters:**
- `graph_id`: Graph to invoke
- `iteration`: Iteration index (for dynamic bindings)

**Use case:**
- Training loops where same graph structure executes repeatedly
- Each iteration may have different tensor bindings

## 4. Memory Coordination Instructions

### 4.1 MC_ALLOC

**Syntax:**
```
MC_ALLOC(tensor_id: uint32,
         size: uint64,
         alignment: uint32 = 512,
         location: MemLocation = GM)
```

**Description:**
Allocate workspace memory for use by kernels in the graph.

**Parameters:**
- `tensor_id`: Unique identifier for this allocation
- `size`: Size in bytes
- `alignment`: Alignment requirement (default 512 for GM)
- `location`: Memory location hint

**Memory Locations:**
```
enum MemLocation {
    GM = 0,       // Global memory (default)
    L2 = 1,       // L2 cache resident (hint)
    HBM = 2,      // High-bandwidth memory (if available)
    SHARED = 3    // Cross-kernel shared buffer
};
```

---

### 4.2 MC_FREE

**Syntax:**
```
MC_FREE(tensor_id: uint32)
```

**Description:**
Release previously allocated workspace.

**Parameters:**
- `tensor_id`: Allocation to release

**Constraints:**
- Must not be in use by any executing kernel

---

### 4.3 MC_BIND

**Syntax:**
```
MC_BIND(kernel_id: uint32,
        arg_index: uint32,
        tensor_id: uint32,
        offset: uint64 = 0,
        size: uint64 = 0)
```

**Description:**
Bind a tensor to a kernel argument.

**Parameters:**
- `kernel_id`: Target kernel
- `arg_index`: Kernel argument index
- `tensor_id`: Source tensor
- `offset`: Byte offset into tensor (default 0)
- `size`: Size to bind (default 0 = entire tensor)

---

### 4.4 MC_COPY

**Syntax:**
```
MC_COPY(src_tensor: uint32,
        dst_tensor: uint32,
        size: uint64,
        src_offset: uint64 = 0,
        dst_offset: uint64 = 0)
```

**Description:**
Asynchronous copy between tensors.

**Parameters:**
- `src_tensor`: Source tensor ID
- `dst_tensor`: Destination tensor ID
- `size`: Bytes to copy
- `src_offset`, `dst_offset`: Byte offsets

---

### 4.5 MC_FENCE

**Syntax:**
```
MC_FENCE(scope: FenceScope)
```

**Description:**
Memory fence ensuring visibility.

**Scopes:**
```
enum FenceScope {
    DEVICE = 0,   // Visible to all cores on device
    L2 = 1,       // Flush L2 cache
    GM = 2        // Full memory barrier
};
```

## 5. Control Flow Instructions

### 5.1 CF_IF / CF_ELSE / CF_ENDIF

**Syntax:**
```
CF_IF(condition: ScalarRef)
  // kernels executed if condition is true
CF_ELSE
  // kernels executed if condition is false
CF_ENDIF
```

**Description:**
Conditional execution at kernel granularity.

**Parameters:**
- `condition`: Reference to a scalar value (0 = false, non-zero = true)

**Semantics:**
- Evaluated at graph execution time
- Affects which kernels are enqueued

---

### 5.2 CF_LOOP / CF_ENDLOOP

**Syntax:**
```
CF_LOOP(count: ScalarRef, iterator: uint32)
  // kernels executed count times
CF_ENDLOOP
```

**Description:**
Loop at kernel granularity.

**Parameters:**
- `count`: Number of iterations
- `iterator`: Loop variable ID (accessible in kernel bindings)

---

### 5.3 CF_BREAK

**Syntax:**
```
CF_BREAK(condition: ScalarRef)
```

**Description:**
Early loop exit when condition is true.

---

### 5.4 CF_WHILE / CF_ENDWHILE

**Syntax:**
```
CF_WHILE(condition: ScalarRef)
  // kernels executed while condition is true
CF_ENDWHILE
```

**Description:**
While loop at kernel granularity. Useful for convergence-based algorithms.

## 6. Core Allocation Instructions

### 6.1 CA_REQUEST

**Syntax:**
```
CA_REQUEST(kernel_id: uint32,
           aic_count: uint32,
           aiv_count: uint32,
           aicpu_count: uint32 = 0)
```

**Description:**
Request specific core allocation for a kernel.

**Parameters:**
- `kernel_id`: Target kernel
- `aic_count`: Number of AIC (cube) cores requested
- `aiv_count`: Number of AIV (vector) cores requested
- `aicpu_count`: Number of AICPU threads requested

**Constraints:**
- Total cores cannot exceed device capacity
- Allocation is advisory; runtime may adjust

---

### 6.2 CA_AFFINITY

**Syntax:**
```
CA_AFFINITY(kernel_set: uint32[],
            affinity_type: AffinityType)
```

**Description:**
Hint that kernels should share cores for cache locality.

**Affinity Types:**
```
enum AffinityType {
    COLOCATE = 0,     // Run on same cores
    SEPARATE = 1,     // Run on different cores
    L2_SHARE = 2      // Share L2 cache partition
};
```

---

### 6.3 CA_PARTITION

**Syntax:**
```
CA_PARTITION(partition_id: uint32,
             aic_start: uint32,
             aic_count: uint32,
             aiv_start: uint32,
             aiv_count: uint32)
```

**Description:**
Define a core partition for kernel execution.

**Use case:**
- Multi-tenant scenarios
- Performance isolation

## 7. Synchronization Instructions

### 7.1 SY_SIGNAL

**Syntax:**
```
SY_SIGNAL(event_id: uint32, src_kernel: uint32)
```

**Description:**
Signal an event when kernel completes.

---

### 7.2 SY_WAIT

**Syntax:**
```
SY_WAIT(event_id: uint32, dst_kernel: uint32)
```

**Description:**
Wait for an event before kernel starts.

---

### 7.3 SY_BARRIER

**Syntax:**
```
SY_BARRIER(kernel_set: uint32[], barrier_id: uint32)
```

**Description:**
All kernels in set wait for each other.

---

### 7.4 SY_NOTIFY

**Syntax:**
```
SY_NOTIFY(device_id: uint32, notify_id: uint32)
```

**Description:**
Cross-device notification (for multi-chip).

---

### 7.5 SY_WAIT_NOTIFY

**Syntax:**
```
SY_WAIT_NOTIFY(device_id: uint32, notify_id: uint32)
```

**Description:**
Wait for cross-device notification.

## 8. Example: DeepSeek V3.2 Indexer Attention

```c
// Define the task graph
TG_BEGIN(DEEPSEEK_ATTN)

// Define kernels
TG_KERNEL(MLA_PROLOG, mla_prolog_entry, 24, AIC_AIV, 64*1024)
TG_KERNEL(INDEXER_PROLOG, indexer_prolog_entry, 24, AIC_AIV, 32*1024)
TG_KERNEL(LIGHTNING_INDEXER, lightning_indexer_entry, 24, AIC, 128*1024)
TG_KERNEL(SPARSE_FA, sparse_fa_entry, 24, AIC_AIV, 256*1024)

// Dependencies (enables parallelism)
// MLA_PROLOG and INDEXER_PROLOG can run in parallel
TG_DEPEND(MLA_PROLOG, SPARSE_FA, DATA, Q_TENSOR)      // Q,K,V → FA
TG_DEPEND(INDEXER_PROLOG, LIGHTNING_INDEXER, DATA, QK_SCORES)
TG_DEPEND(LIGHTNING_INDEXER, SPARSE_FA, DATA, TOPK_INDICES)

// Allocate workspace
MC_ALLOC(WORKSPACE, 512*1024*1024, 512, GM)
MC_ALLOC(Q_TENSOR, 1024*128*2, 512, GM)  // Query tensor
MC_ALLOC(KV_CACHE, 4096*128*2, 512, GM)  // KV cache
MC_ALLOC(TOPK_INDICES, 1024*256*4, 512, GM)

// Bind tensors to kernels
MC_BIND(MLA_PROLOG, 0, INPUT_HIDDEN)   // Input
MC_BIND(MLA_PROLOG, 1, Q_TENSOR)       // Output: Q
MC_BIND(MLA_PROLOG, 2, KV_CACHE)       // Output: KV

MC_BIND(INDEXER_PROLOG, 0, INPUT_HIDDEN)
MC_BIND(INDEXER_PROLOG, 1, QK_SCORES)

MC_BIND(LIGHTNING_INDEXER, 0, QK_SCORES)
MC_BIND(LIGHTNING_INDEXER, 1, TOPK_INDICES)

MC_BIND(SPARSE_FA, 0, Q_TENSOR)
MC_BIND(SPARSE_FA, 1, KV_CACHE)
MC_BIND(SPARSE_FA, 2, TOPK_INDICES)
MC_BIND(SPARSE_FA, 3, OUTPUT)

// Core allocation hints
CA_REQUEST(MLA_PROLOG, 12, 24, 0)       // Half the cores
CA_REQUEST(INDEXER_PROLOG, 12, 24, 0)   // Other half (parallel)
CA_AFFINITY({MLA_PROLOG, SPARSE_FA}, L2_SHARE)  // Share L2 for Q,KV

TG_END(DEEPSEEK_ATTN)

// Execute
TG_LAUNCH(DEEPSEEK_ATTN, compute_stream)
TG_SYNC(DEEPSEEK_ATTN)
```

## 9. Execution Model

### 9.1 Dependency Resolution

```
For each kernel K in topological order:
    K.readyCount = number of incoming DATA/AFTER edges
    when a predecessor completes:
        K.readyCount--
        if K.readyCount == 0:
            enqueue K to ready queue

AiCoreManager threads:
    while ready queue not empty:
        dequeue kernel K
        dispatch K to available cores
        on completion: resolve dependents
```

### 9.2 Control Flow Evaluation

```
CF_IF/CF_ELSE/CF_ENDIF:
    evaluated at TG_LAUNCH time
    false branches: kernels not enqueued

CF_LOOP/CF_ENDLOOP:
    unrolled at TG_LAUNCH time (if static count)
    or dynamic iteration with iteration binding
```

### 9.3 Memory Visibility

```
TG_DEPEND(A, B, DATA, T):
    ensures T written by A is visible to B
    implicit MC_FENCE(DEVICE) between A and B for tensor T

TG_DEPEND(A, B, AFTER):
    only ordering, no memory visibility guarantee
    use MC_FENCE explicitly if needed
```

## 10. Implementation Notes

### 10.1 Compilation Pipeline

```
Source (Python/C++)
    │
    ▼
PTO-Auto/Manual Compiler (per kernel)
    │
    ▼
Kernel Binaries + Metadata
    │
    ▼
Runtime Extension Compiler
    │
    ▼
Task Graph Binary
    │
    ▼
Device Program (kernels + graph)
```

### 10.2 Binary Format

```
TaskGraphBinary {
    Header {
        magic: uint32      // "PTOG"
        version: uint32
        kernel_count: uint32
        depend_count: uint32
        alloc_count: uint32
    }
    KernelTable {
        [kernel_id, entry_offset, blockDim, core_type, ws_size] × N
    }
    DependencyTable {
        [src, dst, type, tensor_id] × M
    }
    AllocationTable {
        [tensor_id, size, alignment, location] × K
    }
    BindingTable {
        [kernel_id, arg_idx, tensor_id, offset, size] × L
    }
}
```

### 10.3 Runtime Integration

The runtime extension instructions are interpreted by AICPU's DeviceCtrlMachine:

```cpp
class RuntimeExtensionInterpreter {
    void Execute(TaskGraphBinary* graph) {
        // 1. Parse graph structure
        ParseGraph(graph);

        // 2. Allocate workspace
        for (auto& alloc : graph->allocations) {
            MC_ALLOC_impl(alloc);
        }

        // 3. Bind tensors
        for (auto& bind : graph->bindings) {
            MC_BIND_impl(bind);
        }

        // 4. Initialize ready counts
        for (auto& kernel : graph->kernels) {
            kernel.readyCount = IncomingEdges(kernel);
            if (kernel.readyCount == 0) {
                readyQueue.push(kernel);
            }
        }

        // 5. Execute until all complete
        while (!AllComplete()) {
            auto kernel = readyQueue.pop();
            DispatchKernel(kernel);
            // On completion callback resolves dependents
        }
    }
};
```

## 11. Platform Support Matrix

| Instruction | A2 | A3 | A5 | CPU Sim |
|-------------|----|----|----|---------|
| TG_BEGIN/END | Yes | Yes | Yes | Yes |
| TG_KERNEL | Yes | Yes | Yes | Yes |
| TG_DEPEND | Yes | Yes | Yes | Yes |
| TG_LAUNCH/SYNC | Yes | Yes | Yes | Yes |
| MC_ALLOC/FREE | Yes | Yes | Yes | Yes |
| MC_BIND | Yes | Yes | Yes | Yes |
| MC_FENCE | Partial | Yes | Yes | N/A |
| CF_IF/ELSE | Yes | Yes | Yes | Yes |
| CF_LOOP | Yes | Yes | Yes | Yes |
| CA_REQUEST | Hint | Hint | Hint | N/A |
| CA_AFFINITY | Hint | Hint | Yes | N/A |
| SY_SIGNAL/WAIT | Yes | Yes | Yes | Yes |
| SY_NOTIFY (multi-chip) | No | No | Yes | N/A |

## 12. Future Extensions

### 12.1 Graph Fusion

```
TG_FUSE(graph1, graph2, fused_graph)
```
Combine two graphs, merging compatible kernels.

### 12.2 Persistent Kernels

```
TG_PERSISTENT(kernel_id, duration)
```
Keep kernel resident for repeated invocation.

### 12.3 Stream Priorities

```
TG_SET_PRIORITY(graph_id, priority)
```
Set execution priority for preemption support.

### 12.4 Checkpointing

```
TG_CHECKPOINT(graph_id, checkpoint_id)
TG_RESTORE(graph_id, checkpoint_id)
```
Save/restore graph execution state.

## 13. Revision History

| Version | Date | Changes |
|---------|------|---------|
| 0.1 | 2025-01 | Initial draft |
