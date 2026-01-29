# PTO-RT Concurrency Mechanisms

This document describes the concurrency features available in each backend.

## 1. Overview

PTO-RT provides concurrent execution through multiple mechanisms:

| Mechanism | CPU Sim | Ascend NPU |
|-----------|---------|------------|
| Thread Pool | Yes | N/A |
| Streams | Yes (simulated) | Yes (hardware) |
| Task Graph | Yes | Yes |
| CSP Channels | Yes (threading) | Yes (AICPU) |

## 2. CPU Simulation Backend

### 2.1 Thread Pool Executor

The CPU backend uses a thread pool for parallel task execution:

```cpp
// From include/pto/rt/backend/cpu_sim.hpp
class CPUSimProgram : public Program {
    ThreadPool thread_pool_;        // Worker threads
    std::queue<TaskNode*> ready_;   // Ready-to-execute tasks
    std::atomic<size_t> completed_; // Completion counter
};
```

Configuration via `CompileOptions`:
- `num_threads`: Number of worker threads (default: hardware_concurrency)
- `num_streams`: Logical streams for task grouping
- `enable_profiling`: Enable timing measurements

### 2.2 Simulated Streams

Streams are simulated via task affinity:
- Tasks assigned to same stream execute on same thread when possible
- Enables testing of stream-based scheduling without hardware

### 2.3 Task Graph Execution

The CPU backend builds a task graph with dependency tracking:

```cpp
struct TaskNode {
    std::vector<TaskNode*> fanin;   // Dependencies
    std::vector<TaskNode*> fanout;  // Dependents
    std::atomic<int> pending;       // fanin count remaining
};

// Task becomes ready when pending reaches 0
void complete(TaskNode* task) {
    for (auto* dep : task->fanout) {
        if (--dep->pending == 0) {
            ready_queue_.push(dep);
        }
    }
}
```

## 3. Ascend NPU Backend

### 3.1 Hardware Streams

Ascend NPU provides hardware command queues (streams):
- Each stream executes tasks sequentially
- Tasks in different streams can execute concurrently
- Synchronization via stream events

```cpp
// From pto-isa-lh patterns
aclrtStream stream;
aclrtCreateStream(&stream);
aclrtSynchronizeStream(stream);
```

### 3.2 AICPU Task Dispatch

Task dispatch to AI Cores via AICPU:
- Dispatcher thread receives task descriptors
- Round-robin or affinity-based dispatch to AI Cores
- Work-stealing for load balancing (optional)

### 3.3 Multi-Core Execution

Ascend NPU features for parallel execution:
- **AI Cores**: Parallel compute units (e.g., 32 cores on 910B)
- **AICPUs**: Host-side processors for task management
- **HCCL**: Collective communication for multi-chip

## 4. pto-isa-lh Concurrency Patterns

The pto-isa-lh project provides tested concurrency patterns:

### 4.1 Tile Synchronization

```cpp
// Double buffering pattern
namespace pto::sync {
    void pingpong_tiles(
        Tile* tiles[2],
        std::function<void(Tile*)> compute,
        std::function<void(Tile*)> copy
    );
}
```

### 4.2 Event-Based Synchronization

```cpp
// Producer-consumer with events
aclrtEvent event;
aclrtCreateEvent(&event);
aclrtRecordEvent(event, stream);
aclrtStreamWaitEvent(other_stream, event);
```

## 5. pto-isa-wc Workgroup Coordination

The pto-isa-wc project provides workgroup-level coordination:

### 5.1 Block Synchronization

```cpp
// Barrier within tile compute
__syncthreads();  // CUDA-style
aicore_sync();    // Ascend equivalent
```

### 5.2 Shared Memory Coordination

- UB (Unified Buffer) shared across vector/cube units
- L1/L2 coherency for multi-core access

## 6. Schedule Configuration

Concurrency is configured via schedule combinators:

```python
program = (workload
    .dispatch(DispatchPolicy.round_robin(4))  # 4 executors
    .streams(2)                                # 2 streams
    .stream_by(lambda t: t.get("head") % 2)   # Stream assignment
    .compile())
```

### 6.1 Dispatch Policies

| Policy | Description |
|--------|-------------|
| `round_robin(n)` | Distribute evenly across n executors |
| `affinity(fn)` | Group tasks by key (e.g., batch index) |
| `work_stealing` | Dynamic load balancing |

### 6.2 Stream Policies

| Policy | Description |
|--------|-------------|
| `streams(n)` | Number of concurrent streams |
| `stream_by(fn)` | Task-to-stream assignment |

## 7. CSP Channel Primitives

Channel-based concurrency for pipeline parallelism:

```python
from pto_wsp import Channel, process, send, consume

# Producer-consumer pattern
data_ch = Channel(capacity=32)

producer = (process("producer")
    .produces(data_ch)
    .body(for_each(items, lambda i: send(data_ch, load(i)))))

consumer = (process("consumer")
    .consumes(data_ch)
    .body(consume(data_ch, lambda t: compute(t))))
```

### 7.1 Channel Implementation

| Backend | Implementation |
|---------|---------------|
| CPU Sim | Threading + condition variables |
| Ascend NPU | AICPU message queues |

## 8. Best Practices

1. **Batch by locality**: Group tasks accessing same memory
2. **Stream by independence**: Separate independent work into streams
3. **Double buffer**: Overlap compute and memory transfer
4. **Minimize synchronization**: Reduce cross-stream dependencies
