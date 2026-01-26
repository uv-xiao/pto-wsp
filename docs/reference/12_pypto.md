# Runtime: Execution on Hardware

## Overview

The PyPTO runtime system bridges compiled kernels to hardware execution on Ascend NPU. It implements a three-layer architecture: Python API, Host C++ runtime, and Device-side execution.

## Runtime Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ Python Layer (runtime.py)                                   │
│   @pypto.jit decorated function                             │
│   ├── compile() → Backend compilation                       │
│   └── __call__() → Dispatch execution                       │
└─────────────────────────┬───────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────┐
│ Host C++ Layer (machine/)                                   │
│   ├── ExportedOperator (compiled kernel handle)             │
│   ├── DeviceLauncher (kernel launching)                     │
│   └── RuntimeAgent (memory, streams)                        │
└─────────────────────────┬───────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────┐
│ Device Layer (device/)                                      │
│   ├── DeviceCtrlMachine (control flow)                      │
│   ├── AiCoreManager (kernel scheduling)                     │
│   └── AICore/AIVector Hardware                              │
└─────────────────────────────────────────────────────────────┘
```

## Python Runtime Interface

**Location:** `python/pypto/runtime.py`

### JitCallableWrapper Execution

```python
class JitCallableWrapper:
    def __call__(self, *args, **kwargs):
        # 1. Validate inputs (all tensors, same device, contiguous)
        self._validate_inputs(args)

        # 2. Compile if needed (lazy compilation)
        self._compile_if_needed(concrete_input_shapes)

        # 3. Allocate outputs
        outputs = self._allocate_outputs(device)

        # 4. Convert torch tensors to pypto tensors
        pto_tensors = [pypto.from_torch(t) for t in args + outputs]

        # 5. Dispatch to device
        self._dispatch_with_run_mode(pto_tensors, device)

        return outputs
```

### Execution Modes

```python
def _dispatch_with_run_mode(self, tensors, device):
    run_mode = self._runtime_options.get("run_mode", RunMode.NPU)

    if run_mode == RunMode.NPU:
        self._run_with_npu(in_tensors, out_tensors, device)
    else:
        self._run_with_cpu(in_tensors, out_tensors)  # Simulation
```

## C++ Backend Interface

**Location:** `python/src/bindings/runtime.cpp`

### Compilation Phase

```cpp
// Initialize backend
pypto_impl.DeviceInit()

// Start operator definition
handler = pypto_impl.OperatorBegin()

// ... Parser generates IR ...

// Finalize compilation
pypto_impl.OperatorEnd(handler)
```

### Execution Phase (NPU)

```cpp
// Get workspace size
workspace_size = pypto_impl.GetWorkSpaceSize(handler, in_data, out_data)

// Execute kernel
error_msg = pypto_impl.OperatorDeviceRunOnceDataFromDevice(
    handler,
    in_data + out_data,  // Tensor list
    [],                   // Extra args
    torch.npu.current_stream().npu_stream,
    workspace_ptr
)
```

### Execution Phase (Simulation)

```cpp
pypto_impl.DeviceRunOnceDataFromHost(tensor_data, [])
```

## Host-Side Runtime

**Location:** `framework/src/machine/runtime/`

### RuntimeAgent

Singleton managing device resources:

```cpp
class RuntimeAgent {
    // Memory management
    HugePageManager hugePages_;      // 1GB and 2MB pages
    L2CacheManager l2Cache_;         // L2 cache offsets

    // Stream management
    DeviceStream computeStream_;     // AICore compute
    DeviceStream controlStream_;     // Control flow
};
```

### Device Launcher

**Location:** `framework/src/machine/runtime/device_launcher_binding.h`

```cpp
int ExportedOperatorDeviceLaunchOnceWithDeviceTensorData(
    ExportedOperator* op,
    const std::vector<DeviceTensorData>& inputList,
    const std::vector<DeviceTensorData>& outputList,
    DeviceStream aicpuStream,
    DeviceStream aicoreStream,
    bool streamSynchronize,
    const DeviceLauncherConfig& config
) {
    // 1. Prepare device program arguments
    PrepareDevProgArgs(op, inputList, outputList);

    // 2. Fill kernel metadata and tensor lists
    FillKernelMeta(op, config);

    // 3. Launch kernels to AICPU
    LaunchKernelToDevice(op, aicpuStream, aicoreStream);

    // 4. Synchronize if requested
    if (streamSynchronize) {
        DeviceSynchronize(aicpuStream, aicoreStream);
    }
}
```

## Device-Side Execution

**Location:** `framework/src/machine/device/`

### Device Task Structure

```cpp
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
```

### DeviceCtrlMachine

Controls device execution flow:

```cpp
class DeviceCtrlMachine {
    int InitDyn(AstKernelArgs* kargs);   // Initialize execution
    int ExecDyn(AstKernelArgs* args);    // Execute control flow
    int PushTask(int type, DynDeviceTask* task, DeviceExecuteContext* ctx);
    int SyncTask(int idx);               // Wait for completion
};
```

### Execution Flow

```
AICPU Control Flow Thread (Thread 0)
    │
    ├── InitDyn()
    │   ├── Initialize DeviceExecuteContext
    │   ├── Setup control flow binary entry
    │   └── Decode input/output tensor lists
    │
    ├── ExecDyn()
    │   └── ctx.GELaunch()  // Execute dynamic graph
    │       └── For each kernel:
    │           └── PushTask(DynDeviceTask, context)
    │
    └── SyncTask()
        └── Wait for all AICore managers

Parallel: AICPU Manager Threads (1-N)
    │
    ├── Run(threadIdx, devArgs)
    │   ├── Initialize core assignment
    │   └── Enter kernel polling loop
    │
    ├── RunTask(taskCtrl)
    │   └── DispatchAiCoreTask()
    │       ├── TryBatchSendTask()
    │       └── BatchSendTask()
    │
    └── Handle Task Completion
        ├── ResolveDepForAllAiCore()
        └── Send next-ready kernels
```

## MPMD Scheduling

**Multi-Process Multi-Device Execution:**

```
Host Process
    │
    ├──► AICPU Thread 0 (Control Flow)
    │    └─ GELaunch() → Orchestrates execution
    │
    ├──► AICPU Thread 1-N (Kernel Managers)
    │    └─ aicoreManager_[i]->Run()
    │        └─ DispatchAiCoreTask()
    │
    └──► AICore/AIVector Hardware (25-75 cores)
         └─ Execute kernels
```

### Thread Assignment

```cpp
int CalcSchAicpuNumByBlockDim(uint32_t blockDim, uint32_t aiCpuNum) {
    uint32_t maxScheCore = aiCpuNum - 2;  // Reserve 2 for control
    if (blockDim > (MAX_SCHEDULE_AICPU_NUM - 1) * MAX_MNG_AICORE_AVG_NUM) {
        return maxScheCore;
    }
    return blockDim / MAX_MNG_AICORE_AVG_NUM + 1;  // ~5 cores per scheduler
}
```

## Task Dependency Resolution

**Location:** `framework/src/machine/device/dynamic/`

```cpp
class AiCoreManager {
    // Task submission
    std::array<uint64_t, MAX_AICORE_NUM> runningIds_;   // Executing
    std::array<uint64_t, MAX_AICORE_NUM> pendingIds_;   // Queued

    // Ready queues
    StaticReadyCoreFunctionQueue* readyAicCoreFunctionQue_;
    StaticReadyCoreFunctionQueue* readyAivCoreFunctionQue_;

    // Dependency resolution
    void ResolveDep(uint64_t finishId);
    void ResolveDepForAllAiCore(CoreType type, ...);
};
```

**Dependency Flow:**
1. Monitor kernel completion via register polling
2. Extract finished task ID from AICore register
3. Look up dependencies in CoreFunctionReadyState
4. Mark dependent tasks as ready
5. Dispatch ready tasks to available cores

## Kernel Invocation Data

```cpp
struct CoreFunctionWsAddr {
    uint64_t functionBinAddr;    // Binary location
    uint64_t invokeEntryAddr;    // Entry point
    uint64_t topoAddr;           // Dependency graph
};

struct CoreFunctionTopo {
    uint32_t coreType;           // AIC or AIV
    uint32_t psgId;              // Unique ID
    int32_t readyCount;          // Remaining dependencies
    uint32_t depNum;             // Number of dependents
};
```

## Stream Synchronization

```cpp
int DeviceSynchronize(DeviceStream aicpuStream,
                      DeviceStream aicoreStream) {
    rtStreamSynchronize(aicpuStream);   // Wait for AICPU
    rtStreamSynchronize(aicoreStream);  // Wait for AICore
    return 0;
}
```

## Tensor Data Representation

```cpp
class DeviceTensorData {
    DataType dtype_;
    void* addr_;                    // Device address
    std::vector<int64_t> shape_;

    int64_t GetDataSize() const {
        return accumulate(shape_) * BytesOf(dtype_);
    }
};
```

## Caching

### Python-Level Caching

```python
class JitCallableWrapper:
    _handler_cache: dict[tuple, uintptr_t]  # Shape signature → compiled op

    def _hit_cache(self, input_hash):
        return input_hash in self._handler_cache
```

### Device Program Caching

```cpp
struct DevAscendProgram {
    DeviceArgs devArgs;
    std::vector<CoreFunctionWsAddr> coreFuncData;
    uint64_t controlFlowBinaryAddr;  // Cached control flow
    MemoryBudget memBudget;
    ControlFlowCache controlFlowCache;
};
```

## Execution Timeline

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
│                              ├────────────────────────────┤      │
│                              Dependency-driven scheduling        │
└──────────────────────────────────────────────────────────────────┘
```

## Performance Optimizations

1. **Register-Based Task Signaling**
   ```cpp
   // Fast path: Direct register writes
   void SendTaskToAiCore(CoreType type, int coreIdx, uint64_t task) {
       WriteReg32(coreIdx, offset, taskId);
   }

   // Polling for completion
   uint64_t GetFinishedTask(int coreIdx) {
       return *(finishRegQueues_[GetPhyIdByBlockId(coreIdx)]);
   }
   ```

2. **Dynamic Workspace Sizing**
   ```cpp
   uint64_t totalWorkspace =
       aicoreCnt * workSpaceStackSize +    // Per-core stack
       invokeParaWorkSpaceSize;             // Shared params
   ```

3. **Parallel Kernel Compilation**
   - Thread pool for concurrent compilation of multiple kernels

4. **Dataflow-Driven Scheduling**
   - Tasks launch immediately when dependencies resolve
   - No artificial barriers between independent operations
