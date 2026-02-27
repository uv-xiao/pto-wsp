# Research Note 7: CUDA Streams, Events, and Work Stealing

## Overview

This note explores **CUDA streams and events** for dependency management and **work stealing** for dynamic load balancing. These concepts are critical for understanding how to design efficient task schedulers for dynamic GPU workloads.

**Sources**:
- [NVIDIA CUDA Programming Guide - Asynchronous Execution](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html)
- [Whippletree: Task-based Scheduling](https://dl.acm.org/doi/10.1145/2661229.2661250)
- [Dynamic Task Parallelism with GPU Work-Stealing](https://link.springer.com/chapter/10.1007/978-3-642-36036-7_14)

## 1. CUDA Streams: Work Queues

### 1.1 Core Concept

A CUDA stream is a **sequence of operations** (kernels, memory copies) that execute in order:

```
┌─────────────────────────────────────────────────────────────────────┐
│ STREAM = Ordered Work Queue                                         │
│                                                                     │
│ Stream A: [Kernel1] → [Kernel2] → [MemCopy1] → [Kernel3]          │
│           Sequential execution within stream                        │
│                                                                     │
│ Stream B: [Kernel4] → [Kernel5] → [MemCopy2]                       │
│           Can run concurrently with Stream A                        │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.2 Key Properties

| Property | Description |
|----------|-------------|
| **Asynchronous** | Kernel launches return immediately to host |
| **Sequential ordering** | Operations within a stream cannot "leap-frog" |
| **Concurrent streams** | Different streams can execute concurrently |
| **Resource-limited** | Concurrency depends on GPU resource availability |

### 1.3 API

```cpp
// Stream creation
cudaStream_t stream;
cudaStreamCreate(&stream);

// Kernel launch into stream
myKernel<<<grid, block, shared_mem, stream>>>(...);

// Memory copy in stream
cudaMemcpyAsync(dst, src, size, kind, stream);

// Cleanup
cudaStreamDestroy(stream);
```

### 1.4 Benefits of Multiple Streams

From NVIDIA documentation:
> "Leveraging three or more parallel channels can reduce host-to-device and device-to-host latency by **40-60%** on NVIDIA Tesla-class hardware."

```
Single Stream:
[H2D Copy]────────────────[Compute]────────────────[D2H Copy]

Three Streams (overlapped):
Stream 0: [H2D]────[Compute]────[D2H]
Stream 1:     [H2D]────[Compute]────[D2H]
Stream 2:         [H2D]────[Compute]────[D2H]
          ^^^^ Copies overlap with compute ^^^^
```

### 1.5 Practical Limits

> "Performance often degrades when the number of parallel jobs exceeds the hardware's ability to simultaneously process them. On many cards, more than **8 concurrent execution lines** offers no benefit and can introduce additional scheduling overhead."

## 2. CUDA Events: Synchronization Points

### 2.1 Core Concept

Events are **markers** within streams that track progress and enable synchronization:

```
┌─────────────────────────────────────────────────────────────────────┐
│ EVENT = Point-in-time marker                                        │
│                                                                     │
│ Stream: [Op1] → [Op2] → ◆Event → [Op3] → [Op4]                    │
│                         │                                           │
│                         └─ Can query/wait for this point            │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 Key Functions

| Function | Purpose |
|----------|---------|
| `cudaEventCreate()` | Create event |
| `cudaEventRecord(event, stream)` | Insert marker at current position |
| `cudaEventSynchronize(event)` | Host blocks until event completes |
| `cudaEventQuery(event)` | Non-blocking check if event done |
| `cudaEventElapsedTime()` | Measure time between events |

### 2.3 Common Patterns

**Timing kernels:**
```cpp
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaEventRecord(start, stream);
myKernel<<<...>>>(...)
cudaEventRecord(stop, stream);

cudaEventSynchronize(stop);
float ms;
cudaEventElapsedTime(&ms, start, stop);
```

**Fine-grained synchronization:**
```cpp
// Only wait for specific operation, not entire stream
cudaEventRecord(checkpoint, streamA);
// ... other work on host ...
cudaEventSynchronize(checkpoint);  // Wait for just this point
```

## 3. Inter-Stream Dependencies: cudaStreamWaitEvent

### 3.1 Core Concept

`cudaStreamWaitEvent(stream, event)` makes all future operations in `stream` wait until `event` completes:

```
┌─────────────────────────────────────────────────────────────────────┐
│ DEPENDENCY GRAPH with cudaStreamWaitEvent                           │
│                                                                     │
│ Stream A: [Producer] → ◆Event_A                                    │
│                           │                                         │
│                           └──────────────────┐                      │
│                                              ▼                      │
│ Stream B:                          [Wait for Event_A] → [Consumer]  │
│                                                                     │
│ Stream B operations after Wait won't start until Event_A completes  │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.2 DAG Construction

Multiple dependencies can create directed acyclic graphs:

```cpp
// Producer in Stream A
producerKernel<<<..., streamA>>>(...);
cudaEventRecord(eventA, streamA);

// Producer in Stream B
producerKernel2<<<..., streamB>>>(...);
cudaEventRecord(eventB, streamB);

// Consumer waits for both producers
cudaStreamWaitEvent(streamC, eventA, 0);
cudaStreamWaitEvent(streamC, eventB, 0);
consumerKernel<<<..., streamC>>>(...);  // Runs after both complete
```

### 3.3 Key Insight for PTO-ISA

This pattern enables **explicit task dependencies** without blocking the host:

```
Traditional (blocking):
Host: Launch A → Sync → Launch B → Sync → Launch C
               ^block^         ^block^

Event-based (non-blocking):
Host: Launch A, Record A → Launch B, Wait A → Launch C, Wait B → [continue]
      (returns immediately)
```

## 4. Host Synchronization

### 4.1 Synchronization Hierarchy

| Function | Scope | Use Case |
|----------|-------|----------|
| `cudaDeviceSynchronize()` | All streams | End of computation |
| `cudaStreamSynchronize(stream)` | Single stream | Need results from stream |
| `cudaEventSynchronize(event)` | Single point | Need results from operation |

### 4.2 Non-blocking Alternatives

```cpp
// Query without blocking
cudaError_t status = cudaStreamQuery(stream);
if (status == cudaSuccess) {
    // Stream is idle
} else if (status == cudaErrorNotReady) {
    // Stream still has work
}

// Same for events
cudaError_t status = cudaEventQuery(event);
```

### 4.3 Best Practice

> "Delay synchronization as long as possible and issue independent operations before dependent ones to maximize concurrency opportunities."

## 5. Implicit vs Explicit Synchronization

### 5.1 The Default Stream Problem

The **default (NULL) stream** implicitly synchronizes with all other blocking streams:

```cpp
// Legacy behavior - implicit barriers!
kernel1<<<..., stream1>>>(...);
kernel2<<<..., 0>>>(...);           // NULL stream - creates barrier!
kernel3<<<..., stream1>>>(...);     // Cannot overlap with kernel1
```

### 5.2 Solution: Per-Thread Default Streams

Compile with `--default-stream per-thread` to avoid implicit synchronization:

```bash
nvcc --default-stream per-thread myprogram.cu
```

### 5.3 Recommendation

Always use **explicit streams** for production code to have full control over synchronization.

## 6. GPU Work Stealing

### 6.1 The Problem

CUDA's standard execution model assigns work statically at kernel launch. For irregular or dynamic workloads, this leads to **load imbalance**:

```
Static Assignment (poor load balancing):
SM0: [Long Task]──────────────────────────[idle]
SM1: [Short]──[idle]────────────────────────────
SM2: [Short]──[Short]──[idle]───────────────────
     ^^^^^^^^ Wasted GPU cycles ^^^^^^^^^
```

### 6.2 Work Stealing Solution

Implement a **shared work queue** that SMs can pull from dynamically:

```
Work Stealing (better load balancing):
Shared Queue: [T1][T2][T3][T4][T5][T6][T7][T8][T9]...
                │    │    │
SM0:            └────┴────┴─[T1][T4][T7]...
SM1:                        [T2][T5][T8]...
SM2:                        [T3][T6][T9]...
     ^^^^ All SMs stay busy ^^^^
```

### 6.3 Persistent Threads Pattern

From [Whippletree](https://dl.acm.org/doi/10.1145/2661229.2661250):

> "Worker-blocks are launched to exactly fill up all multiprocessors. These worker-blocks execute a loop drawing tasks from work queues."

```cpp
// Persistent thread pattern (conceptual)
__global__ void persistentKernel(TaskQueue* queue) {
    while (true) {
        Task* task = queue->dequeue();  // Atomic fetch from shared queue
        if (task == nullptr) break;     // No more work

        executeTask(task);              // Process task
    }
}
```

### 6.4 Implementation Challenges

| Challenge | Solution |
|-----------|----------|
| **Queue contention** | Partition queues, local stealing |
| **Task granularity** | Balance overhead vs. load balancing |
| **Termination detection** | Distributed counting, barriers |
| **Memory consistency** | Careful use of atomics, fences |

### 6.5 Performance Results

From [Rice University GPU Work-Stealing](https://link.springer.com/chapter/10.1007/978-3-642-36036-7_14):

> "Experimental results show that fine-grained task solution can utilize the hardware more efficiently than the CUDA scheduler for unbalanced workloads."

From Megakernels (Research Note 6):
> "For large batch sizes (4096-8192), global work queue provides **14.2% improvement** due to better load balancing across variable-length sequences."

## 7. Task-Based Programming Models

### 7.1 Whippletree Model

[Whippletree](https://dl.acm.org/doi/10.1145/2661229.2661250) introduces task-based parallelism on GPU:

```
┌─────────────────────────────────────────────────────────────────────┐
│ WHIPPLETREE EXECUTION MODEL                                         │
│                                                                     │
│ Persistent Worker Blocks (fill all SMs)                            │
│ ┌─────────────────────────────────────────────────────────────────┐ │
│ │ while (work_available) {                                        │ │
│ │     Task* t = queue.dequeue();   // Get task from shared queue │ │
│ │     switch (t->type) {           // Branch to procedure         │ │
│ │         case PROC_A: runA(t); break;                           │ │
│ │         case PROC_B: runB(t); break;                           │ │
│ │     }                                                           │ │
│ │     // Tasks can generate new tasks → push to queue            │ │
│ │ }                                                               │ │
│ └─────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

### 7.2 Benefits

| Benefit | Description |
|---------|-------------|
| **Dynamic work generation** | Tasks can spawn new tasks |
| **Load balancing** | Work queues distribute work automatically |
| **Pipeline support** | Different task types run concurrently |
| **Irregular parallelism** | Handles variable-size work items |

### 7.3 Relation to Megakernels

Megakernels (Research Note 6) uses a similar persistent kernel approach but with:
- Warp specialization (loader/consumer/storer)
- Instruction-based work units
- Optional global work queue for dynamic scheduling

## 8. Key Takeaways for PTO-ISA

### 8.1 Patterns to Adopt

| CUDA/GPU Pattern | PTO-ISA Adaptation |
|------------------|-------------------|
| **Streams** | Task queues per AICPU thread or AICore |
| **Events** | Completion signals / task dependencies |
| **cudaStreamWaitEvent** | Task dependency edges in DAG |
| **Work stealing** | Shared task queue for dynamic dispatch |
| **Persistent threads** | Persistent AICPU scheduler |

### 8.2 Dependency Management Options

1. **Implicit (sequential)**: Tasks in same "stream" execute in order
2. **Explicit events**: Tasks signal completion, others wait
3. **DAG-based**: Full task graph with explicit edges
4. **Barrier-based**: Bulk synchronization points

### 8.3 Design Questions

1. **Stream granularity**: One stream per AICore? Per task type? Per batch?

2. **Event implementation**: Hardware semaphores? Memory counters? AICPU coordination?

3. **Work stealing granularity**:
   - Individual tasks (high overhead, best balance)
   - Task batches (lower overhead, coarser balance)
   - Hybrid (steal batches, execute individually)

4. **Termination**: How to know all work is done?
   - Centralized counter
   - Distributed voting
   - Explicit "done" tasks

### 8.4 Comparison: Static vs Dynamic Scheduling

| Aspect | Static (Round-Robin) | Dynamic (Work Stealing) |
|--------|---------------------|------------------------|
| **Load balance** | Poor for irregular | Good |
| **Overhead** | Minimal | Atomic operations |
| **Predictability** | High | Lower |
| **Best for** | Uniform workloads | Variable workloads |
| **Megakernels data** | Better at small batch | +14.2% at large batch |

### 8.5 Recommended Approach

Based on research, a **hybrid approach** is recommended:

1. **Default**: Static scheduling for predictable workloads
2. **Optional**: Work stealing when dynamism is high (MoE, variable sequences)
3. **Events**: For explicit task dependencies
4. **Barriers**: For bulk synchronization (layer boundaries)

```
┌─────────────────────────────────────────────────────────────────────┐
│ HYBRID SCHEDULING FOR PTO-ISA                                       │
│                                                                     │
│ ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐   │
│ │ Static Queues   │   │ Dynamic Queue   │   │ Dependency DAG  │   │
│ │ (per AICore)    │   │ (shared, opt.)  │   │ (events)        │   │
│ └────────┬────────┘   └────────┬────────┘   └────────┬────────┘   │
│          │                     │                     │             │
│          └─────────────────────┼─────────────────────┘             │
│                                ▼                                   │
│                    ┌─────────────────────┐                         │
│                    │ AICPU Dispatcher    │                         │
│                    │ (orchestrates)      │                         │
│                    └─────────────────────┘                         │
└─────────────────────────────────────────────────────────────────────┘
```

## 9. References

### Academic Papers
- [Whippletree: Task-based Scheduling of Dynamic Workloads on the GPU](https://dl.acm.org/doi/10.1145/2661229.2661250)
- [Dynamic Task Parallelism with a GPU Work-Stealing Runtime System](https://link.springer.com/chapter/10.1007/978-3-642-36036-7_14)
- [Softshell: Dynamic Scheduling on GPUs](https://dl.acm.org/doi/10.1145/2366145.2366180)

### Documentation
- [NVIDIA CUDA Programming Guide - Asynchronous Execution](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html)
- [CUDA Runtime API - Streams](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html)

### Implementations
- [Whippletree GitHub](https://github.com/apc-llc/whippletree)
- Megakernels (see Research Note 6)

---
*Note Version: 1.0*
*Last Updated: 2025-01-16*
