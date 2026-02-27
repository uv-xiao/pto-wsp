# Research Note 5: Ascend Hardware Model and AICPU Capabilities

## Overview

This note documents the Ascend NPU hardware architecture and AICPU programming model, focusing on aspects relevant to runtime extension design for dynamic LLM workloads. The analysis draws from PTO-ISA documentation and PyPTO runtime implementation.

**Sources**: `docs/machine/abstract-machine.md`, `docs/coding/ProgrammingModel.md`, `docs/uv/pypto.md`

## 1. Hardware Architecture

### 1.1 Three-Layer Machine Model

```
┌─────────────────────────────────────────────────────────────────────┐
│ PTO Host Machine                                                     │
│   - Compilation and JIT caching                                      │
│   - Graph scheduling and partitioning                                │
│   - Memory allocation (HBM, L2)                                      │
│   - Work submission to Device Machine                                │
└─────────────────────────────────┬───────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│ PTO Device Machine                                                   │
│   - AICPU: Control flow and scheduling                              │
│   - AICore/AIVector scheduling                                       │
│   - Global memory dependency tracking                                │
└─────────────────────────────────┬───────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│ PTO Core Machine (AICore/AIVector)                                   │
│   - Scalar Unit (control flow, address calc)                         │
│   - Vector Pipeline (elementwise ops)                                │
│   - Matrix/Cube Pipeline (GEMM)                                      │
│   - Memory Pipeline (DMA, layout transforms)                         │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.2 AICore Architecture

Each AICore contains multiple functional units:

| Unit | Function | PTO Instructions |
|------|----------|------------------|
| **Scalar Unit (SU)** | Control flow, address calc, event ops | Loop control, branches |
| **Vector Pipeline** | Elementwise tile ops | `TADD`, `TMUL`, `TEXP`, `TROWMAX`, ... |
| **Cube Pipeline** | Matrix multiply | `TMATMUL`, `TMATMUL_ACC`, `TMATMUL_BIAS` |
| **MTE (Memory Transfer Engine)** | Global memory movement | `TLOAD`, `TSTORE` |
| **MTE1** | L1 ↔ L0 transfer | `TEXTRACT` |
| **MTE2** | GM ↔ L1 transfer | Part of `TLOAD` |

### 1.3 Memory Hierarchy

```
┌─────────────────────────────────────────────────────────────────┐
│                        Global Memory (GM)                        │
│                        HBM: 32-96 GB                             │
│                        Bandwidth: 1-2 TB/s                       │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                          L2 Cache                                │
│                          6-50 MB (shared)                        │
│                          Hardware managed                        │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Per-Core On-Chip Memory                      │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │   L1 Buffer     │  │ Unified Buffer  │  │  L0 Buffers     │  │
│  │   (__cbuf__)    │  │   (__ubuf__)    │  │ L0A,L0B,L0C     │  │
│  │   MatTile       │  │   VecTile       │  │ Cube I/O        │  │
│  │   64-128 KB     │  │   192-256 KB    │  │ ~32 KB each     │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### 1.4 Platform Variants

| Platform | Codename | AICore Count | L1/UB Size | Cube Throughput |
|----------|----------|--------------|------------|-----------------|
| A2/910B | Atlas | ~25 | 64/192 KB | ~256 TFLOPS FP16 |
| A3/910C | - | ~50 | 128/256 KB | ~512 TFLOPS FP16 |
| A5/950 | - | ~75 | TBD | ~1000 TFLOPS FP16 |

**Key Difference**: A2/A3 use `MEMORY_BASE` (UB-centric), A5 uses `REGISTER_BASE` (register-centric) programming model.

## 2. AICPU Architecture

### 2.1 Thread Model

AICPU provides multiple CPU cores for control and scheduling:

```
┌──────────────────────────────────────────────────────────────────┐
│                         AICPU Threads                             │
│                                                                   │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │ Thread 0: Control Flow Thread                               │  │
│  │   - GELaunch() orchestration                                │  │
│  │   - Graph-level scheduling                                   │  │
│  │   - Dependency tracking                                      │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                   │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │ Threads 1-N: Kernel Manager Threads                         │  │
│  │   - Each manages ~5 AICores                                  │  │
│  │   - DispatchAiCoreTask() to hardware                        │  │
│  │   - Poll for kernel completion                               │  │
│  │   - Resolve dependencies for finished tasks                  │  │
│  └────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
```

### 2.2 Thread Assignment Algorithm

```cpp
// From PyPTO: Calculate number of scheduler threads
int CalcSchAicpuNumByBlockDim(uint32_t blockDim, uint32_t aiCpuNum) {
    uint32_t maxScheCore = aiCpuNum - 2;  // Reserve 2 for control
    if (blockDim > (MAX_SCHEDULE_AICPU_NUM - 1) * MAX_MNG_AICORE_AVG_NUM) {
        return maxScheCore;
    }
    // ~5 AICores per scheduler thread
    return blockDim / MAX_MNG_AICORE_AVG_NUM + 1;
}
```

### 2.3 Task Dispatch Mechanism

```cpp
// AiCoreManager task dispatch
class AiCoreManager {
    // Per-core tracking
    std::array<uint64_t, MAX_AICORE_NUM> runningIds_;   // Currently executing
    std::array<uint64_t, MAX_AICORE_NUM> pendingIds_;   // Queued

    // Ready queues (separate for AIC and AIV)
    StaticReadyCoreFunctionQueue* readyAicCoreFunctionQue_;
    StaticReadyCoreFunctionQueue* readyAivCoreFunctionQue_;

    // Register-based task signaling (fast path)
    void SendTaskToAiCore(CoreType type, int coreIdx, uint64_t task) {
        WriteReg32(coreIdx, offset, taskId);
    }

    // Completion polling
    uint64_t GetFinishedTask(int coreIdx) {
        return *(finishRegQueues_[GetPhyIdByBlockId(coreIdx)]);
    }

    // Dependency resolution
    void ResolveDep(uint64_t finishId);
    void ResolveDepForAllAiCore(CoreType type, ...);
};
```

### 2.4 Task Data Structures

```cpp
// Task metadata
struct DeviceTask {
    uint32_t coreFunctionCnt;                    // Total kernels
    uint64_t coreFunctionReadyStateAddr;         // Dependency tracking
    StaticReadyCoreFunctionQueue* readyAicQue;   // AIC kernel queue
    StaticReadyCoreFunctionQueue* readyAivQue;   // AIV kernel queue

    struct CoreFuncData {
        CoreFunctionWsAddr* coreFuncWsAddr;      // Workspace addresses
        uint64_t stackWorkSpaceAddr;
        uint64_t stackWorkSpaceSize;
    } coreFuncData;
};

// Per-kernel metadata
struct CoreFunctionWsAddr {
    uint64_t functionBinAddr;    // Binary location
    uint64_t invokeEntryAddr;    // Entry point
    uint64_t topoAddr;           // Dependency graph
};

// Dependency tracking
struct CoreFunctionTopo {
    uint32_t coreType;           // AIC or AIV
    uint32_t psgId;              // Unique ID
    int32_t readyCount;          // Remaining dependencies
    uint32_t depNum;             // Number of dependents
};
```

## 3. Execution Model

### 3.1 SPMD Execution

Standard PTO model where all cores run the same kernel:

```cpp
// Each core selects its work via block index
auto cid = get_block_idx();
auto vid = get_block_idx() * get_subblockdim() + get_subblockid();

// Work partition example
uint32_t mIterIdx = get_block_idx() % mIter;
uint32_t nIterIdx = get_block_idx() / mIter;
```

**Limitation**: Work assignment is static—`get_block_idx()` maps directly to core.

### 3.2 MPMD Execution

Multiple programs dispatched to different cores:

```cpp
// Task ID-based dispatch
__global__ __aicore__ void KernelMPMD(__gm__ float* out,
                                     __gm__ const float* in,
                                     uint32_t task_id) {
    switch (task_id) {
        case 0: return ProducerStage(out, in);
        case 1: return ConsumerStage(out, in);
        default: return;
    }
}
```

**Current Implementation**: The `task_id` mechanism exists but is platform/runtime dependent.

### 3.3 Execution Timeline

```
┌──────────────────────────────────────────────────────────────────┐
│ Time →                                                           │
├──────────────────────────────────────────────────────────────────┤
│ Host:  compile() ──► launch() ──► sync()                        │
│                          │           │                           │
│                          ▼           ▼                           │
│ AICPU: ──────────────► Init ► Exec ► Complete                   │
│                               │                                  │
│                               ▼                                  │
│ AICore: ─────────────────── Kernel 0 ► Kernel 1 ► ... ► Done   │
│                              ├────────────────────────────────┤  │
│                              Dependency-driven scheduling        │
└──────────────────────────────────────────────────────────────────┘
```

### 3.4 Dataflow-Driven Scheduling

```
Host Process
    │
    ├──► AICPU Thread 0 (Control Flow)
    │    └─ GELaunch() → Orchestrates execution
    │
    ├──► AICPU Threads 1-N (Kernel Managers)
    │    └─ aicoreManager_[i]->Run()
    │        └─ DispatchAiCoreTask()
    │
    └──► AICore/AIVector Hardware (25-75 cores)
         └─ Execute kernels (dependency-driven)
```

**Key Property**: Tasks launch immediately when dependencies resolve—no artificial barriers.

## 4. Synchronization Mechanisms

### 4.1 Event-Based Pipeline Sync

```cpp
// Pipeline identifiers
enum pipe_t {
    PIPE_V,      // Vector pipeline
    PIPE_M,      // Matrix/Cube pipeline
    PIPE_MTE1,   // L1 ↔ L0 transfer
    PIPE_MTE2,   // GM ↔ L1 transfer
    PIPE_FIX,    // Fixed-function
};

// Cross-pipeline synchronization
template <pipe_t srcPipe, pipe_t dstPipe>
void SetFlag(uint32_t id) {
    set_flag(srcPipe, dstPipe, static_cast<event_t>(id));
}

template <pipe_t srcPipe, pipe_t dstPipe>
void WaitFlag(uint32_t id) {
    wait_flag(srcPipe, dstPipe, static_cast<event_t>(id));
}
```

### 4.2 Global Memory Dependencies

```cpp
// Device-level ordering via TLOAD/TSTORE
// - TSTORE produces data
// - TLOAD consumes data
// - Runtime ensures ordering via dependency tracking
```

### 4.3 Stream Synchronization

```cpp
int DeviceSynchronize(DeviceStream aicpuStream, DeviceStream aicoreStream) {
    rtStreamSynchronize(aicpuStream);   // Wait for AICPU
    rtStreamSynchronize(aicoreStream);  // Wait for AICore
    return 0;
}
```

## 5. Capabilities and Limitations for Dynamic Workloads

### 5.1 Current Capabilities

| Capability | Status | Notes |
|------------|--------|-------|
| Multi-threaded AICPU | ✅ | 1 control + N managers |
| Register-based task dispatch | ✅ | Low latency |
| Dependency-driven scheduling | ✅ | No artificial barriers |
| MPMD task dispatch | ✅ | Via task_id parameter |
| Dataflow execution | ✅ | Automatic dependency tracking |

### 5.2 Limitations

| Limitation | Impact | Workaround |
|------------|--------|------------|
| Single control thread | Serialized task generation | Move planning to host or parallelize descriptor generation |
| Static work assignment | Can't handle variable batch/seq | Index-based dispatch |
| Compile-time loop bounds | No dynamic iteration | Descriptor-based bounds |
| No descriptor infrastructure | Each work needs explicit dispatch | Add descriptor buffer support |

### 5.3 AICPU Performance Characteristics

From PyPTO analysis:

| Operation | Typical Latency | Notes |
|-----------|-----------------|-------|
| **AICPU → AICore dispatch** | **~0 μs** | Register-based, all AICPU threads |
| Host → AICPU | ~3 μs | Cross-domain communication |
| Task completion polling | ~100 ns | Register read |
| Dependency resolution | ~1 μs | Per-task |

**Key Insight**: ALL AICPU offloading to AICores is essentially zero-latency (register-based signaling). The bottleneck is not dispatch latency but **task generation throughput** on AICPU—a single control thread can only generate tasks serially.

## 6. Implications for Runtime Extension

### 6.1 Where to Run Planning

| Option | Pros | Cons |
|--------|------|------|
| **Host** | Full CPU power, parallelizable | ~3μs latency Host→AICPU |
| **AICPU (single thread)** | 0μs dispatch to AICore | Serial task generation |
| **AICPU (multi-thread)** | 0μs dispatch, parallel generation | Requires coordination |

**Key Insight**: AICPU→AICore dispatch is always ~0μs (register-based). The real tradeoff is:
- **Host**: More compute power, but 3μs Host→AICPU latency
- **AICPU**: 0μs to AICore, but limited parallelism for task generation

**Recommendation**: For latency-sensitive inference, prefer **AICPU-based planning** with parallel task generation across multiple AICPU threads. For throughput-oriented batch processing, host planning may be acceptable.

### 6.2 Descriptor-Based Dispatch

Proposed extension to leverage existing infrastructure:

```cpp
// Planning phase (Host)
for (req = 0; req < batch; req++) {
    for (chunk = 0; chunk < ceil(kv_len[req] / chunk_size); chunk++) {
        descriptors[work_idx++] = {
            .request_idx = req,
            .chunk_idx = chunk,
            .kv_start = chunk * chunk_size,
            .kv_end = min((chunk + 1) * chunk_size, kv_len[req]),
            .tier = SelectTier(kv_len[req])
        };
    }
}

// Execution phase (AICore)
__global__ void kernel(__gm__ WorkDescriptor* descs, ...) {
    WorkDescriptor desc = descs[get_block_idx()];  // O(1) lookup
    // Process based on descriptor...
}
```

### 6.3 Multi-Tier Kernel Selection

Use descriptor's `tier` field for runtime kernel selection:

```cpp
void dispatch_kernel(WorkDescriptor* desc, ...) {
    switch (desc->tier) {
        case 0: decode_kernel_1k(desc, ...); break;
        case 1: decode_kernel_4k(desc, ...); break;
        case 2: decode_kernel_16k(desc, ...); break;
    }
}
```

## 7. Key Takeaways

### 7.1 Hardware Strengths

- **Multi-core parallelism**: 25-75 AICores available
- **Fast task dispatch**: Register-based signaling
- **Dataflow scheduling**: Automatic dependency tracking
- **Flexible memory**: Hierarchical L2/L1/UB/L0

### 7.2 Architecture Constraints

- **AICPU is a bottleneck**: Single control thread serializes task generation
- **Static work assignment**: `get_block_idx()` = core index (no indirection)
- **Compile-time loops**: Bounds must be known at compile time

### 7.3 Design Principles for Extension

1. **Descriptor buffers**: Pass work specs through GM, not control flow
2. **Index-based dispatch**: Block → work mapping via descriptor lookup
3. **Multi-tier compilation**: Generate kernel variants, select at runtime
4. **AICPU-based generation**: Prefer AICPU for task generation (0μs dispatch to AICore)
5. **Support both SPMD and MPMD**: Single program (all cores run same kernel) and multiple programs (different kernels on different cores)

### 7.4 Comparison with GPU (FlashInfer)

| Aspect | FlashInfer (CUDA) | Ascend (Current) | Ascend (Extension) |
|--------|-------------------|------------------|-------------------|
| Planning | CPU (no AICPU equivalent) | N/A | AICPU (preferred) or Host |
| Work assignment | `request_indices[]` | `get_block_idx()` | Descriptor buffer |
| Dispatch latency | CUDA launch (~μs) | **AICPU→AICore: ~0μs** | Same |
| Loop bounds | Runtime | Compile-time | Descriptor |
| Programming model | SPMD | SPMD | **SPMD + MPMD** |

**Key Difference**: Ascend has AICPU which GPU lacks. This enables 0μs dispatch and on-device planning, which is an advantage over FlashInfer's CPU-based approach.

---
*Note Version: 1.1*
*Last Updated: 2024-01-14*
