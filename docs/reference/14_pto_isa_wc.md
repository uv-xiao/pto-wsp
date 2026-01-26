# Research Report 14: pto-isa-wc Analysis (NPU Backend)

## Overview

**Source**: `references/pto-isa-wc/`
**Purpose**: Ascend NPU (910B) backend implementation, device runtime, AICPU-AICore coordination
**v9 Relevance**: Reference for NPU backend architecture and device-side task scheduling

---

## 1. Relationship to pto-isa-lh

pto-isa-wc is a variant of pto-isa-lh with additional NPU-specific components:

| Component | pto-isa-lh | pto-isa-wc |
|-----------|------------|------------|
| Python DSL | `pto_compile.py` | Same (minor diffs) |
| ISA Definition | `pto_isa_definition.py` | Same (minor diffs) |
| CPU Runtime | `pto_runtime.h/c` | Same (minor diffs) |
| **NPU Runtime** | Not present | `runtime/` directory |
| **Ascend Kernels** | Not present | `examples/output_ascend910b/` |

Key difference: pto-isa-wc includes a **complete device-side runtime** for Ascend NPU execution.

---

## 2. NPU Runtime Architecture

### 2.1 Directory Structure

```
runtime/
├── graph/                  # Task dependency graph (portable)
│   ├── graph.h/cpp             # Graph class (same as CPU)
│   ├── handshake.h             # AICPU-AICore communication protocol
│   └── kernel_args.h           # Kernel argument structures
├── host/                   # Host-side device management
│   ├── devicerunner.h/cpp      # Device execution interface
│   └── memoryallocator.h/cpp   # Device memory management
├── aicpu/                  # AICPU scheduler kernels
│   ├── graph_executor.cpp      # Multi-threaded task scheduler
│   ├── device_log.h/cpp        # Device-side logging
│   └── CMakeLists.txt
├── aicore/                 # AICore compute kernels
│   └── kernel.cpp              # Task execution dispatcher
├── python/                 # Python bindings
│   ├── bindings.cpp            # pybind11 bindings
│   └── graphbuilder.py         # Python example
└── graphbuilder.cpp        # C++ example
```

### 2.2 Execution Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           HOST (ARM64)                                   │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │  1. Build task graph (Graph class)                                │  │
│  │  2. Allocate device memory (DeviceRunner)                         │  │
│  │  3. Copy inputs to device                                         │  │
│  │  4. DeviceRunner::Run(graph)                                      │  │
│  │     ├── Copy graph to device memory                               │  │
│  │     ├── Launch AICPU init kernel (handshake)                      │  │
│  │     ├── Launch AICPU main kernel (scheduler)                      │  │
│  │     ├── Launch AICore kernels (workers)                           │  │
│  │     └── Synchronize streams                                       │  │
│  │  5. Copy results back                                             │  │
│  └──────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        ASCEND NPU (910B)                                │
│  ┌────────────────────┐          ┌────────────────────────────────┐   │
│  │       AICPU        │          │           AICore Array          │   │
│  │  (Task Scheduler)  │◄────────►│      (Compute Workers)          │   │
│  │                    │Handshake │                                  │   │
│  │  • Load graph      │  Buffer  │  Core 0: Wait for task           │   │
│  │  • Find ready tasks│          │  Core 1: Execute kernel          │   │
│  │  • Dispatch to     │          │  Core 2: Signal completion       │   │
│  │    idle AICores    │          │  ...                             │   │
│  │  • Update fanin    │          │  Core N-1: Wait for task         │   │
│  │  • Repeat          │          │                                  │   │
│  └────────────────────┘          └────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Handshake Protocol

### 3.1 Data Structure

```c
struct Handshake {
    volatile uint32_t aicpu_ready;   // AICPU → AICore: scheduler ready
    volatile uint32_t aicore_done;   // AICore → AICPU: core acknowledged
    volatile uint64_t task;          // AICPU → AICore: task pointer
    volatile int32_t  task_status;   // 1 = task assigned, 0 = complete
    volatile int32_t  control;       // 1 = shutdown signal
};
```

### 3.2 Communication Flow

```
                AICPU                                AICore
                  │                                     │
    ┌─────────────┴─────────────┐      ┌───────────────┴────────────────┐
    │                           │      │                                │
    │  1. aicpu_ready = 1       │ ──►  │  Wait for aicpu_ready == 1     │
    │                           │      │                                │
    │  Wait for aicore_done     │ ◄──  │  2. aicore_done = 1            │
    │                           │      │                                │
    │  3. task = &task_ptr      │ ──►  │  Wait for task_status == 1     │
    │     task_status = 1       │      │                                │
    │                           │      │  4. Execute task               │
    │  Wait for task_status = 0 │ ◄──  │     task_status = 0            │
    │                           │      │                                │
    │  5. Update fanin for deps │      │  Wait for next task            │
    │     Dispatch next task    │      │                                │
    │                           │      │                                │
    │  N. control = 1 (shutdown)│ ──►  │  Exit when control == 1        │
    └───────────────────────────┘      └────────────────────────────────┘
```

---

## 4. AICPU Graph Executor

### 4.1 Multi-Threaded Scheduler

```cpp
// Hardware configuration
constexpr int MAX_AICPU_THREADS = 4;      // Max scheduler threads
constexpr int MAX_AIC_PER_THREAD = 24;    // AICore Cube units per thread
constexpr int MAX_AIV_PER_THREAD = 48;    // AICore Vector units per thread

// Each AICPU thread manages:
// - 1 AIC (Cube unit for matmul)
// - 2 AIV (Vector units for elementwise ops)

struct GraphExecutorManager {
    std::atomic<int> threadIdx_{0};
    std::atomic<int> ready_count_{0};
    std::mutex ready_queue_mutex_;
    int ready_queue_[GRAPH_MAX_TASKS];
    // ...
};
```

### 4.2 Core Assignment Strategy

```
Thread 0: AIC[0], AIV[24], AIV[25]   # Cube 0 + Vector 0-1
Thread 1: AIC[1], AIV[26], AIV[27]   # Cube 1 + Vector 2-3
Thread 2: AIC[2], AIV[28], AIV[29]   # Cube 2 + Vector 4-5
Thread 3: AIC[3], AIV[30], AIV[31]   # Cube 3 + Vector 6-7
...

Total: 24 AIC + 48 AIV = 72 compute cores
```

**Mapping to v8/v9**: This directly corresponds to our `is_cube` flag for distinguishing matmul vs vector tasks.

---

## 5. Ascend C InCore Functions

### 5.1 Generated Code Pattern

```cpp
// Auto-generated Ascend C code from PTO ISA Compiler
#include "kernel_operator.h"
using namespace AscendC;

class rowmaxInCore {
public:
    __aicore__ inline rowmaxInCore() {}

    __aicore__ inline void Init(GM_ADDR input, GM_ADDR output) {
        inputGm.SetGlobalBuffer((__gm__ float*)input);
        outputGm.SetGlobalBuffer((__gm__ float*)output);
        pipe.InitBuffer(inQueueX, 1, 288);
        pipe.InitBuffer(outQueueY, 1, 288);
    }

    __aicore__ inline void Process() {
        CopyIn();
        Compute();
        CopyOut();
    }

private:
    __aicore__ inline void CopyIn() {
        LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
        DataCopy(xLocal, inputGm, 72);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute() {
        LocalTensor<float> xLocal = inQueueX.DeQue<float>();
        LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();
        ReduceMax(result, x, 8);  // TROWMAX
        outQueueY.EnQue(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut() {
        LocalTensor<float> yLocal = outQueueY.DeQue<float>();
        DataCopy(outputGm, yLocal, 72);
        outQueueY.FreeTensor(yLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, 1> inQueueX;
    TQue<QuePosition::VECOUT, 1> outQueueY;
    GlobalTensor<float> inputGm;
    GlobalTensor<float> outputGm;
};
```

### 5.2 Key Ascend C Patterns

| Pattern | Description | v9 Backend Requirement |
|---------|-------------|------------------------|
| `TPipe` | Hardware pipeline management | Abstract in backend interface |
| `TQue` | Double-buffering queues | Implicit in backend |
| `LocalTensor` | On-chip tile storage | Map to our Tile type |
| `GlobalTensor` | Global memory reference | Map to our MemRef type |
| `DataCopy` | DMA transfer | TLOAD/TSTORE codegen |
| Vector ops | `Add`, `Mul`, `Exp`, etc. | Tile instruction codegen |
| `ReduceMax/Sum` | Reduction operations | TROWMAX/TROWSUM codegen |

### 5.3 Single-Core vs SPMD

```cpp
// Callable InCore function (single-core, task-scheduled)
__aicore__ inline void rowmax(
    GM_ADDR input, int32_t in_row_off, int32_t in_col_off,
    GM_ADDR output, int32_t out_row_off, int32_t out_col_off,
    int32_t tile_rows, int32_t tile_cols)
{
    int32_t in_offset = (in_row_off * tile_cols + in_col_off) * sizeof(float);
    int32_t out_offset = (out_row_off * tile_cols + out_col_off) * sizeof(float);

    rowmaxInCore op;
    op.Init((GM_ADDR)((uint8_t*)input + in_offset),
            (GM_ADDR)((uint8_t*)output + out_offset));
    op.Process();
}

// SPMD Kernel (multi-core, for standalone testing)
#ifdef PTO_GENERATE_SPMD_KERNEL
extern "C" __global__ __aicore__ void rowmax_kernel(GM_ADDR input, GM_ADDR output) {
    rowmaxInCore op;
    op.Init(input, output);
    op.Process();
}
#endif
```

**Critical Insight**: InCore functions are **single-core** (SPSD), scheduled as tasks by the runtime. SPMD kernels are only for standalone testing.

---

## 6. Python Integration

### 6.1 Python API

```python
import numpy as np
import pto_runtime

# Initialize device
runner = pto_runtime.DeviceRunner.get()
runner.init(device_id=9, cores=3,
            aicpu_kernel="./aicpu/libaicpu_graph_kernel.so",
            aicore_kernel="./aicore/kernel.o")

# Allocate tensors
dev_a = runner.allocate_tensor(128 * 128 * 4)
dev_b = runner.allocate_tensor(128 * 128 * 4)
dev_c = runner.allocate_tensor(128 * 128 * 4)

# Copy data to device
a = np.full((128, 128), 2.0, dtype=np.float32)
runner.copy_to_device(dev_a, a)

# Build and run graph
graph = pto_runtime.Graph()
t0 = graph.add_task([dev_a, dev_b, dev_c, 128*128], func_id=0)
runner.run(graph)

# Get results
result = np.zeros((128, 128), dtype=np.float32)
runner.copy_from_device(result, dev_c)

# Cleanup
runner.free_tensor(dev_a)
runner.finalize()
```

### 6.2 pybind11 Bindings

```cpp
// python/bindings.cpp
PYBIND11_MODULE(pto_runtime, m) {
    py::class_<DeviceRunner>(m, "DeviceRunner")
        .def_static("get", &DeviceRunner::get)
        .def("init", &DeviceRunner::Init)
        .def("run", &DeviceRunner::Run)
        .def("allocate_tensor", &DeviceRunner::AllocateTensor)
        .def("copy_to_device", [](DeviceRunner& self, uint64_t dev_ptr, py::array_t<float> arr) {
            // NumPy array to device copy
        })
        .def("copy_from_device", [](DeviceRunner& self, py::array_t<float> arr, uint64_t dev_ptr) {
            // Device to NumPy array copy
        });

    py::class_<Graph>(m, "Graph")
        .def(py::init<>())
        .def("add_task", &Graph::add_task)
        .def("add_successor", &Graph::add_successor);
}
```

---

## 7. Generated Kernel Examples

### 7.1 Available Primitives

Located in `examples/output_ascend910b/`:

| Category | Examples | Count |
|----------|----------|-------|
| **aten_primitives** | `aten_relu`, `aten_mm`, `aten_sigmoid`, `prims_add`, `prims_exp` | 27 |
| **fused_softmax** | `rowmax`, `rowsum`, `elem_exp`, `rowexpandsub`, `rowexpanddiv` | 6 |
| **llama7b** | `flash_attn_*`, `linear_tile`, `rmsnorm_tile`, `tile_matmul` | 16 |
| **flex_attention** | `attention_alibi`, `attention_causal_mask`, `soft_capping_attention` | 16 |
| **torch_functional** | `F_softmax`, `F_layer_norm`, `F_gelu`, `F_cross_entropy` | 30 |

### 7.2 Buffer Analysis Header

Each generated kernel includes buffer analysis:

```cpp
// ======================================================================
// TILE BUFFER ANALYSIS: rowmax
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     2
//   Total capacity (no reuse): 288 bytes (0.3 KB)
//   Total capacity (w/ reuse): 288 bytes (0.3 KB)
//   Reuse savings:            0 bytes (0.0%)
//
// TILE DETAILS:
//   Name        Shape    Type   Bytes    Liveness [write,read]   Reuse
//   -------------------------------------------------------------------
//   result      8x1      f32       32   [  1,   2]           -
//   x           8x8      f32      256   [  0,   1]           -
```

---

## 8. Mapping to v8/v9 Concepts

| pto-isa-wc | v8 Runtime Extension | v9 Target |
|------------|---------------------|-----------|
| `Graph` class | Compiled Schedule | IR + Compiled Schedule |
| `Task` struct | Task entry | Task IR node |
| `fanin/fanout` | Dependency edges | Same (preserved) |
| `DeviceRunner` | - (CPU only) | Backend interface |
| `Handshake` protocol | - | NPU backend internal |
| AICPU scheduler | - | NPU backend internal |
| AICore InCore function | Task execution | Task codegen |
| `is_cube` (AIC vs AIV) | `is_cube` flag | Task property |

---

## 9. Key Takeaways for v9

### 9.1 What v9 NPU Backend Needs

1. **Host Interface (DeviceRunner)**
   - Device initialization
   - Memory allocation/transfer
   - Graph submission
   - Synchronization

2. **Device Runtime (AICPU)**
   - Task graph loading
   - Dependency tracking
   - Core assignment (AIC vs AIV)
   - Handshake protocol

3. **Kernel Generation**
   - Ascend C InCore pattern (class + Init + Process)
   - TQue/TPipe for double buffering
   - Single-core execution (SPSD mode)

4. **Integration Pattern**
   - Python builds Workload + Schedule
   - Compiler generates IR
   - Backend generates Ascend C code
   - Runtime executes on device

### 9.2 Abstraction Opportunities

| pto-isa-wc Specific | v9 Abstraction |
|---------------------|----------------|
| Ascend C / CANN APIs | `Backend::generate_kernel()` |
| AICPU handshake | `Backend::device_scheduler()` |
| DeviceRunner | `Backend::device_interface()` |
| AIC/AIV assignment | `schedule.dispatch(is_cube=...)` |

### 9.3 Differences from CPU Simulation

| Aspect | CPU Sim | NPU Backend |
|--------|---------|-------------|
| Task execution | Function pointer call | AICore kernel launch |
| Scheduling | Thread pool | AICPU scheduler |
| Memory | Shared RAM | Device memory + DMA |
| Synchronization | pthread | Handshake protocol |
| Parallelism | N threads | 72 cores (24 AIC + 48 AIV) |

---

## 10. Open Questions for v9

1. **Backend Interface Design**: How to abstract DeviceRunner for multiple backends (NPU, AIE)?
2. **Handshake Abstraction**: Should handshake be part of backend interface or internal?
3. **Memory Management**: How to express device memory in Python DSL?
4. **Kernel Registry**: How to map func_name → kernel_ptr on device?

---

## 11. Recent Updates (January 2026)

### 11.1 Shared Updates with pto-isa-lh

The following updates from pto-isa-lh also apply to pto-isa-wc (shared codebase):

1. **Tile Shape Fix**: Orchestration code now correctly uses actual tile dimensions (32×128, 64×128) instead of hardcoded (8,8)

2. **Recursive Binary Expansion**: Nested loops (e.g., Flash Attention's Q×KV) get 75% task reduction via N² → (N/2)² optimization

3. **Platform-Independent Orchestration**: Same C code for orchestration across ARM64/CUDA/Ascend; only InCore functions differ

4. **SRAM Constraint**: Maximum tile size is 64 rows due to score matrix S = tile_rows² being the bottleneck in 256KB SRAM

### 11.2 A2A3 Codegen Enhancement

`pto_codegen_ascend_a2a3_sim.py` now generates:
- Actual Ascend instructions for InCore functions
- Instruction code as string constants for core simulator parsing
- Registration functions for each InCore function
- Orchestration code with task submission

### 11.3 Dual-Queue Mode for NPU Simulation

The NPU simulation now correctly routes tasks to the appropriate core type:

| Core Type | Count | Function Types |
|-----------|-------|----------------|
| Vector (AIV) | 48 | rmsnorm, rope, softmax, swiglu |
| Cube (AIC) | 24 | matmul, attention_score, attention_output |

**Codegen fix**: `pto_task_alloc()` parameter order corrected:
```c
// Before (incorrect)
pto_task_alloc(rt, "func_name", NULL, is_cube, 0, 0);

// After (correct)
pto_task_alloc(rt, "func_name", NULL, 0, 0, is_cube);
```

### 11.4 Enhanced Trace Export

Chrome tracing now includes:
- Separate process groups for Vector Workers (pid=0) and Cube Workers (pid=1)
- Thread naming (`Vector-0`, `Cube-0`, etc.)
- Worker count in process names (`Vector Workers (48)`, `Cube Workers (24)`)

### 11.5 Auto-Detection of Cube Operations

Operations using `.matmul()` in the builder automatically set `is_cube = True`, ensuring correct core assignment at runtime.

---

## References

- `references/pto-isa-wc/README.md` - Comprehensive documentation (Chinese)
- `references/pto-isa-wc/runtime/README.md` - Runtime architecture
- `references/pto-isa-wc/runtime/graph/graph.h` - Task graph data structure
- `references/pto-isa-wc/runtime/graph/handshake.h` - AICPU-AICore protocol
- `references/pto-isa-wc/runtime/aicpu/graph_executor.cpp` - Multi-threaded scheduler
- `references/pto-isa-wc/examples/output_ascend910b/` - Generated Ascend C kernels
- `references/pto-isa-wc/new_updates.md` - January 20, 2026 updates
- `references/pto-isa-wc/pto_compile.py` - Python builder with auto is_cube detection
