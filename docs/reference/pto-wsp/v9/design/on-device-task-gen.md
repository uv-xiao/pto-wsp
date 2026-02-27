# On-Device Task Generation Design for PTO-WSP v9

> **Status (as-built):** PTO‑RT v9 currently emits an **AICPU expander C++ translation unit** during NPU codegen
> (`target="ascend_npu"`). This document describes a **bytecode interpreter** approach that is a design exploration /
> future direction unless explicitly implemented and validated in this repo.

## Executive Summary

This document describes the on-device task generation architecture for PTO-WSP v9, which represents a fundamental shift from the host-side task expansion approach used in pto-isa-lh/wc.

**Key Insight**: Workload is a **program** that generates tasks, not a **list** of tasks.

Two viable approaches exist:

1) **As-built (v9 today): generated expander**
- Compile workload IR to a compact plan + **generated AICPU expander C++**.
- Upload plan + runtime symbols; AICPU expansion runs on-device.

2) **Design exploration (this document): bytecode interpreter**
- Compile workload IR to compact bytecode
- Transfer bytecode to each AICPU (~4KB vs ~400MB)
- Each AICPU interprets bytecode and generates only its own tasks locally
- Task generation is pipelined with task execution

## 1. Problem Analysis

### 1.1 Current Approach in pto-isa-lh/wc (Host-Side Expansion)

From `pto_runtime_common.h`:
```c
typedef struct {
    int32_t      task_id;
    const char*  func_name;
    void*        func_ptr;
    TaskArg      args[PTO_MAX_ARGS];
    int32_t      fanin;
    int32_t      fanout[PTO_MAX_FANOUT];
    // ...
} PendingTask;
```

**The Problem:**
1. Host CPU expands workload into **O(N)** individual `PendingTask` structures
2. All tasks must be **transferred to device** before execution begins
3. For LLaMA 7B at 16K seqlen: 200,704 tasks per layer = 200K+ task structures
4. With `PTO_MAX_FANOUT=512` and `PTO_MAX_ARGS=16`, each task is ~2KB
5. Total transfer: **~400MB** of task graph data per forward pass

### 1.2 Our Solution

A `parallel_for(batch=4, heads=8)` doesn't need 32 task structures - it needs:
- Loop structure: 2 nested `ParallelForNode` (~100 bytes)
- Loop bounds: batch=4, heads=8 (8 bytes)
- Kernel reference: 1 `TaskNode` template (~50 bytes)

**Total: ~160 bytes vs 32 * 2KB = 64KB** (400x reduction)

---

## 2. Architecture Overview

```
┌───────────────────────────────────────────────────────────────────────┐
│                            HOST CPU                                    │
│  ┌────────────────────────────────────────────────────────────────┐   │
│  │  Python Frontend                                                │   │
│  │  @workload                                                      │   │
│  │  def attention(batch, heads):                                   │   │
│  │      for b, h in P(batch, heads):                               │   │
│  │          attn[b,h](Q[b,h], K[b], V[b], O[b,h])                  │   │
│  └────────────────────────────────────────────────────────────────┘   │
│                              │                                         │
│                              ▼                                         │
│  ┌────────────────────────────────────────────────────────────────┐   │
│  │  Compile: WorkloadIR + ScheduleIR → WorkloadBytecode            │   │
│  │  - Serialize IR to compact bytecode                             │   │
│  │  - Embed dispatch config (which AICPU owns which tasks)         │   │
│  │  - O(IR size) not O(tasks)                                      │   │
│  └────────────────────────────────────────────────────────────────┘   │
│                              │                                         │
│              Transfer: ~4KB bytecode (not 400MB task list)            │
└──────────────────────────────┼────────────────────────────────────────┘
                               ▼
┌───────────────────────────────────────────────────────────────────────┐
│                         ASCEND NPU (910B)                              │
│  ┌────────────────────┐  ┌────────────────────┐  ┌─────────────────┐  │
│  │     AICPU 0        │  │     AICPU 1        │  │    AICPU N      │  │
│  │  ┌──────────────┐  │  │  ┌──────────────┐  │  │  ┌───────────┐  │  │
│  │  │  Workload    │  │  │  │  Workload    │  │  │  │ Workload  │  │  │
│  │  │  Interpreter │  │  │  │  Interpreter │  │  │  │Interpreter│  │  │
│  │  └──────────────┘  │  │  └──────────────┘  │  │  └───────────┘  │  │
│  │        │           │  │        │           │  │       │        │  │
│  │        ▼           │  │        ▼           │  │       ▼        │  │
│  │  ┌──────────────┐  │  │  ┌──────────────┐  │  │  ┌───────────┐  │  │
│  │  │ Local Task   │  │  │  │ Local Task   │  │  │  │Local Task │  │  │
│  │  │ Queue        │  │  │  │ Queue        │  │  │  │Queue      │  │  │
│  │  │ (only b=0,2) │  │  │  │ (only b=1,3) │  │  │  │(only b=N) │  │  │
│  │  └──────────────┘  │  │  └──────────────┘  │  │  └───────────┘  │  │
│  └────────────────────┘  └────────────────────┘  └─────────────────┘  │
│                                                                        │
│  ┌────────────────────────────────────────────────────────────────┐   │
│  │                      AICore Array (72 cores)                    │   │
│  │   Cube 0-23 (matmul)        Vector 24-71 (elementwise)         │   │
│  └────────────────────────────────────────────────────────────────┘   │
└───────────────────────────────────────────────────────────────────────┘
```

---

## 3. AICPU-Side Workload Interpreter Design

### 3.1 Bytecode Format

```cpp
// Bytecode opcodes
enum class WLOpcode : uint8_t {
    // Control
    HALT            = 0x00,
    NOP             = 0x01,

    // Workload structure
    PARALLEL_FOR    = 0x10,  // axis_id, body_offset
    FOR_EACH        = 0x11,  // axis_id, body_offset
    SELECT          = 0x12,  // sparse_axis_id, body_offset
    COND            = 0x13,  // pred_expr, then_offset, else_offset
    COMBINE         = 0x14,  // num_branches, offsets...
    SEQUENTIAL      = 0x15,  // num_steps, offsets...

    // Task generation
    TASK            = 0x20,  // kernel_id, num_params, num_io
    PARAM_CONST     = 0x21,  // value (32-bit)
    PARAM_LOOPVAR   = 0x22,  // loop_depth (which enclosing loop)
    IO_INPUT        = 0x23,  // tensor_id, offset_expr...
    IO_OUTPUT       = 0x24,  // tensor_id, offset_expr...

    // Axis definition
    AXIS_DENSE      = 0x30,  // extent (32-bit)
    AXIS_DENSE_DYN  = 0x31,  // runtime_param_slot
    AXIS_RAGGED     = 0x32,  // outer_extent, lengths_ptr
    AXIS_SPARSE     = 0x33,  // outer_extent, indptr_ptr, indices_ptr

    // Dispatch filter
    DISPATCH_FILTER = 0x40,  // predicate_type, params...
};

// Bytecode instruction layout (fixed 8-byte aligned)
struct WLInstruction {
    WLOpcode opcode;
    uint8_t  flags;       // Reserved
    uint16_t operand1;    // First operand (opcode-specific)
    uint32_t operand2;    // Second operand (opcode-specific)
};

// Compact bytecode stream structure
struct WorkloadBytecode {
    uint32_t magic;           // 0x50544F57 = "PTOW"
    uint32_t version;         // Bytecode version
    uint32_t num_instructions;
    uint32_t num_axes;
    uint32_t num_kernels;
    uint32_t num_tensors;
    // Followed by:
    // - WLInstruction instructions[num_instructions]
    // - AxisDescriptor axes[num_axes]
    // - KernelDescriptor kernels[num_kernels]
    // - TensorDescriptor tensors[num_tensors]
};
```

### 3.2 Interpreter State Machine

```cpp
// AICPU-side interpreter context
struct WorkloadInterpreter {
    // Bytecode pointer (device memory)
    const WorkloadBytecode* bytecode;

    // Current execution state
    uint32_t pc;                          // Program counter
    int32_t loop_stack[MAX_LOOP_DEPTH];   // Current loop indices
    int32_t loop_bound[MAX_LOOP_DEPTH];   // Loop upper bounds
    uint8_t loop_depth;                   // Current nesting depth

    // Dispatch state
    DispatchConfig dispatch;
    uint16_t my_aicpu_id;

    // Local task queue
    LocalTaskQueue local_queue;

    // Methods
    void interpret();                      // Main loop
    bool evaluate_dispatch_predicate();    // Check if current task is mine
    void emit_task(const WLInstruction* task_instr);
};
```

### 3.3 Interpretation Algorithm

```cpp
void WorkloadInterpreter::interpret() {
    while (pc < bytecode->num_instructions) {
        const WLInstruction* instr = &bytecode->instructions[pc];

        switch (instr->opcode) {
        case WLOpcode::PARALLEL_FOR: {
            int32_t extent = evaluate_axis_extent(instr->operand1);

            // Each AICPU iterates ALL indices but filters on dispatch
            for (int32_t i = 0; i < extent; i++) {
                push_loop(extent);
                loop_stack[loop_depth - 1] = i;

                uint32_t saved_pc = pc;
                pc = instr->operand2;  // Jump to body
                interpret();           // Recursive call
                pc = saved_pc;

                pop_loop();
            }
            pc++;
            break;
        }

        case WLOpcode::TASK: {
            // Check dispatch predicate: does this task belong to me?
            if (evaluate_dispatch_predicate()) {
                emit_task(instr);
            }
            pc++;
            break;
        }

        case WLOpcode::HALT:
            return;
        }
    }
}
```

---

## 4. Dispatch Config Format Design

### 4.1 Dispatch Config Structure

```cpp
struct DispatchConfig {
    uint16_t num_aicpus;
    uint16_t my_aicpu_id;

    enum class PolicyType : uint8_t {
        ROUND_ROBIN,      // task_id % num_aicpus == my_aicpu_id
        AFFINITY,         // Based on axis value
        THRESHOLD,        // Multi-level based on axis extents
        STATIC_PARTITION, // Explicit task ranges
    } policy_type;

    union {
        struct {
            uint8_t affinity_axis_depth;  // Which loop axis to use
        } affinity;

        struct {
            uint16_t num_levels;
            uint32_t thresholds[8];
            uint8_t  policy_per_level[8];
        } threshold;

        struct {
            uint32_t task_range_start;
            uint32_t task_range_end;
        } static_partition;
    };
};
```

### 4.2 Schedule Directive Lowering

```python
# Python API
attention.dispatch(DispatchPolicy.affinity(lambda t: t.batch))

# Lowers to DispatchConfig:
DispatchConfig {
    policy_type: AFFINITY,
    affinity: {
        affinity_axis_depth: 0  # Outermost loop = batch
    }
}
```

---

## 5. Local Task Queue Management

### 5.1 Per-AICPU Task Queue

```cpp
struct LocalTask {
    uint16_t id;              // Local task ID
    uint16_t kernel_id;       // Which kernel to execute
    uint8_t  pool;            // ExecPool: Vector (0) or Cube (1)
    std::atomic<int16_t> fanin;
    uint16_t fanout_start;
    uint16_t fanout_count;
    uint32_t params[MAX_LOCAL_PARAMS];
};

struct LocalTaskQueue {
    LocalTask tasks[LOCAL_WINDOW_SIZE];  // e.g., 1024 tasks
    uint32_t head;
    uint32_t tail;

    // Ready queues (dual-queue for cube/vector)
    struct ReadyQueue {
        uint16_t queue[READY_QUEUE_SIZE];
        std::atomic<uint16_t> head;
        std::atomic<uint16_t> tail;
    } ready_vector, ready_cube;
};
```

### 5.2 Pipelined Generation and Execution

```cpp
void aicpu_main(WorkloadBytecode* bytecode, DispatchConfig* dispatch) {
    WorkloadInterpreter interp;
    interp.init(bytecode, dispatch);

    // Start interpreter in background (generates tasks)
    std::thread generator([&interp]() {
        interp.interpret();
    });

    // Dispatch loop (consumes tasks) - pipelined with generation
    while (!interp.is_complete() || !local_queue.all_complete()) {
        poll_aicore_completion();
        dispatch_ready_tasks();
    }

    generator.join();
}
```

---

## 6. Inter-AICPU Synchronization

### 6.1 Cross-AICPU Dependencies

When a task on AICPU-0 produces output consumed by AICPU-1:

```cpp
struct GlobalTensorMap {
    // Hash table in shared device memory
    struct Entry {
        TensorRegion2D region;
        uint16_t producer_aicpu;
        uint16_t producer_local_task;
        std::atomic<uint8_t> is_complete;
    };
    Entry entries[GLOBAL_MAP_SIZE];
};
```

### 6.2 Completion Notification Protocol

```cpp
struct CompletionNotifyBuffer {
    struct Notification {
        uint16_t task_id;
        uint16_t sequence;
    };
    Notification ring[NOTIFY_RING_SIZE];
    std::atomic<uint32_t> write_idx;
    std::atomic<uint32_t> read_idx;
};
```

### 6.3 Locality-Aware Dispatch

```cpp
// Minimize cross-AICPU dependencies via dispatch policy
// E.g., for attention: batch dimension partitioning keeps
// Q[b], K[b], V[b], O[b] on same AICPU
DispatchConfig {
    policy_type: AFFINITY,
    affinity: { affinity_axis_depth: 0 }  // Batch axis
}
```

---

## 7. Comparison: On-Device Generation vs pto-isa-lh/wc

### 7.1 Quantitative Comparison

| Metric | pto-isa-lh/wc (Host Expansion) | PTO-WSP v9 (On-Device Gen) |
|--------|-------------------------------|---------------------------|
| **Data Transfer** | O(tasks) * sizeof(PendingTask) | O(IR size) + O(config) |
| | ~200K * 2KB = 400MB | ~4KB bytecode |
| **Host Memory** | O(tasks) task structures | O(1) bytecode |
| **Startup Latency** | Build + Transfer + Execute | Transfer + Execute (pipelined) |
| **Scalability** | Limited by host memory | Limited by device cores |
| **Dynamic Shapes** | Rebuild entire graph | Same bytecode, different bounds |

### 7.2 Example: LLaMA 7B Attention (16K seq)

**pto-isa-lh approach:**
```
Host: Expand parallel_for(batch=4, heads=32, q_tiles=512, kv_tiles=512)
      → 4 * 32 * 512 * 512 = 33.5M task enumerations
      → Filter down to ~200K tasks for this AICPU
      → Transfer 200K * 2KB = 400MB
```

**PTO-WSP v9 approach:**
```
Host: Compile workload IR → ~4KB bytecode
      → Transfer bytecode + dispatch config (~8KB total)
Device: Each AICPU interprets bytecode
        → Generates only its own ~50K tasks locally
        → Pipelined with execution
```

### 7.3 Feature Comparison

| Feature | pto-isa-lh/wc | PTO-WSP v9 On-Device |
|---------|---------------|---------------------|
| **Workload representation** | Flat task list | Structured bytecode |
| **Task generation location** | Host CPU | AICPU |
| **Dependency tracking** | Host TensorMap | Local + Global TensorMap |
| **Dispatch policy** | Implicit | Explicit, configurable |
| **Dynamic shapes** | Rebuild graph | Reinterpret bytecode |
| **CSP support** | Not native | First-class |
| **Dual-queue (AIC/AIV)** | Supported | Supported |

---

## 8. Implementation Plan

### Phase 1: Host Simulation
- Implement bytecode format and interpreter in pure C++
- Validate against pto-isa-lh task graph outputs
- Unit test all opcodes

### Phase 2: Single-AICPU Deployment
- Port interpreter to AICPU
- Local task queue management
- Integrate with existing handshake protocol

### Phase 3: Multi-AICPU with Cross-Deps
- Implement global tensor map
- Completion notification protocol
- Locality-aware dispatch policies

### Phase 4: Optimization
- Binary search in dispatch predicates
- Prefetch bytecode instructions
- Cache axis extents

---

## 9. Integration with Backend Architecture

```cpp
// In include/pto/wsp/backend/npu/on_device_gen.hpp
namespace pto::wsp::backend::npu {

class OnDeviceGenerator {
public:
    // Compile IR to bytecode
    WorkloadBytecode compile(const ir::WorkloadDef& workload,
                             const ir::ScheduleDef& schedule);

    // Generate dispatch config from schedule
    DispatchConfig compile_dispatch(const ir::ScheduleDef& schedule,
                                    int num_aicpus);
};

}
```

---

*Version: 1.0*
*Last Updated: 2026-01-25*
