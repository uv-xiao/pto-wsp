# PTO Workload-Schedule Programming (PTO-WSP): Conceptual Analysis v5

## Executive Summary

This document presents a conceptual framework for extending PTO-ISA to support dynamic LLM workloads on AICPU. The design prioritizes **flexibility** and **human-in-the-loop** optimization, drawing from three key research areas:

1. **Megakernels**: Instruction-interpreter model, inter-task pipelining, work stealing
2. **CUDA Streams/Events**: Flexible dependency management without strict ordering
3. **FlashInfer**: Index-based work assignment, tiered kernel dispatch

**Core Design Principles**:

1. **Task = Kernel Call**: One task corresponds to one PTO-ISA kernel invocation
2. **Dispatch vs Issue**: Dispatch = AICPU receives tasksets; Issue = AICPU launches to AICores
3. **Flexible Dependencies**: Tasks can overlap within queues; explicit events for synchronization
4. **Dynamic Task Generation**: Tasks can spawn new tasks (for MoE routing)
5. **AICPU-Centric**: All scheduling, dispatch, and coordination happens on AICPU

---

## 1. Problem Recap

### 1.1 The Core Challenges (from requirements.md)

| Problem | Description | Impact |
|---------|-------------|--------|
| **Single-threaded task generation** | Control flow runs serially on AICPU | Poor parallelism |
| **Serial execution queues** | Tasks wait for predecessors in queue | Cannot overlap independent work |
| **No Human-In-The-Loop** | Cannot customize scheduling strategies | Suboptimal performance |
| **Cold start overhead** | Stitching helps but has its own overhead | First tasks execute poorly |

### 1.2 Hardware Model

```
1A (Host) ──3μs──> 16B (AICPU) ──0μs──> 24C (Cube) + 48D (Vector)
                        │
                        └── Multiple threads, multiple streams supported
```

**Key Insight**: AICPU→AICore dispatch is **~0μs** (register-based). The bottleneck is **task generation throughput**, not dispatch latency.

### 1.3 Dynamic Patterns to Support

| Pattern | Example | Requirement |
|---------|---------|-------------|
| **Variable KV length** | 512, 2048, 8192, 32768 per batch | Tiered kernels |
| **TopK Attention** | DeepSeek Lightning Indexer | Data-dependent paths |
| **MoE Routing** | Dynamic expert selection | Task spawning |
| **Batch Stitching** | Parallel independent batches | Overlapped execution |

---

## 2. Core Concepts

### 2.1 Task: The Unit of Work

A **Task** is a single invocation of an existing PTO-ISA kernel:

```
┌─────────────────────────────────────────────────────────────────────┐
│ TASK = One Kernel Invocation                                        │
│                                                                     │
│ ┌─────────────────┐                                                │
│ │ kernel_fn       │ ← Which PTO-ISA kernel to run                  │
│ │ params          │ ← Runtime parameters (batch_idx, seq_len, etc.)│
│ │ resources       │ ← Tensors, buffers used                        │
│ │ dependencies    │ ← Events to wait for                           │
│ │ signal          │ ← Event to signal on completion                │
│ └─────────────────┘                                                │
└─────────────────────────────────────────────────────────────────────┘
```

**Design Choice**: Task granularity = one kernel call. This matches PTO-ISA's existing kernel model and gives programmers direct control.

### 2.2 Dispatch vs Issue

Critical terminology distinction:

| Term | Level | Description |
|------|-------|-------------|
| **Dispatch** | Host → AICPU | Host sends a **TaskSet** (collection of tasks) to AICPU |
| **Issue** | AICPU → AICore | AICPU launches individual **Tasks** to AICores |

```
Host                    AICPU                   AICores
  │                       │                       │
  │ ──dispatch(taskset)─► │                       │
  │     (~3μs latency)    │                       │
  │                       │ ──issue(task0)──────► │ Core0
  │                       │ ──issue(task1)──────► │ Core1
  │                       │ ──issue(task2)──────► │ Core2
  │                       │     (~0μs each)       │
  │                       │                       │
```

**Implication**: Minimize dispatches (expensive), maximize tasks per dispatch.

### 2.3 TaskSet: Collection of Tasks

A **TaskSet** is dispatched from Host to AICPU as a unit:

```cpp
struct TaskSet {
    Task* tasks;           // Array of tasks
    int32_t task_count;    // Number of tasks
    IssuePolicy policy;    // How to issue tasks (static, dynamic, etc.)
    Event* completion;     // Optional: signal when all tasks complete
};
```

**Key Properties**:
- Tasks within a TaskSet can have dependencies on each other
- Tasks within a TaskSet can have dependencies on tasks in previously dispatched TaskSets
- A TaskSet can specify its issuing policy

### 2.4 Event: Synchronization Primitive

**Events** enable flexible dependency management (inspired by CUDA events):

```cpp
struct Event {
    // Opaque handle - implementation managed by runtime
};

// Usage patterns:
Event e1 = create_event();

// Task signals event on completion
task_a.signal = e1;

// Task waits for event before starting
task_b.wait_for(e1);
```

**Key Properties**:
- Events are **explicit** - programmer decides what depends on what
- Events can cross TaskSet boundaries
- Multiple tasks can wait on the same event
- A task can wait on multiple events

### 2.5 Dependency Model

Unlike CUDA streams (strict ordering within stream), our model allows **overlap within the same queue**:

```
CUDA Streams (strict ordering):
Queue A: [Task1] ─────► [Task2] ─────► [Task3]
         Must finish    Must finish

Our Model (flexible ordering with events):
Queue A: [Task1]   [Task2]   [Task3]
              │        │
              ▼        │
         [Event1]      │ (no dependency - can overlap)
              │        │
              └────────┴─► [Task4] waits for Event1
```

**Rationale**: Independent tasks (e.g., different batches) should overlap without artificial barriers.

---

## 3. Issuing Strategies

The extension supports multiple **issue policies**, all selectable by the programmer:

### 3.1 Static Issuing (Round-Robin)

Pre-assign tasks to AICores statically:

```
Tasks: [T0, T1, T2, T3, T4, T5, T6, T7]
AICores: [C0, C1, C2, C3]

Assignment:
  C0: T0, T4
  C1: T1, T5
  C2: T2, T6
  C3: T3, T7
```

**Use Case**: Uniform workloads where all tasks have similar execution time.

### 3.2 Dynamic Issuing (Work Stealing)

AICores pull tasks from a shared queue:

```
Shared Queue: [T0][T1][T2][T3][T4][T5][T6][T7]
                │
C0: pop() → T0 ─┘
C1: pop() → T1
C2: pop() → T2 ──── (faster AICore gets more tasks)
C2: pop() → T3
C3: pop() → T4
C2: pop() → T5
...
```

**Use Case**: Variable-length workloads (different seq_len per batch).

**Stealing Granularity**: Single task (as specified in design questions).

### 3.3 Hybrid Issuing

Combine static assignment with dynamic fallback:

```
Primary Assignment: Static (for predictable work)
Fallback: Work stealing (when imbalance detected)
```

**Use Case**: Mostly uniform with occasional outliers.

### 3.4 Adaptive Issuing

Runtime selects strategy based on workload characteristics:

```cpp
IssuePolicy adaptive_select(TaskSet& ts) {
    float variance = compute_task_variance(ts);
    if (variance < THRESHOLD)
        return STATIC;
    else
        return DYNAMIC;
}
```

**Use Case**: Automated optimization when workload patterns are unknown.

---

## 4. Dynamic Task Generation (MoE Support)

For MoE and other data-dependent patterns, tasks can **spawn new tasks**:

### 4.1 The Pattern

```cpp
// Router task determines which experts to activate
void moe_router_task(RouterParams params, TaskSpawner* spawner) {
    // Compute routing decisions
    int* selected_experts = compute_routing(params.input, params.top_k);

    // Dynamically spawn expert tasks based on routing result
    for (int i = 0; i < params.batch_size * params.top_k; i++) {
        int expert_id = selected_experts[i];
        int token_idx = i / params.top_k;

        // Spawn a task for this expert-token pair
        spawner->spawn(expert_kernels[expert_id], {
            .token_idx = token_idx,
            .expert_id = expert_id,
            .weight = routing_weights[i]
        });
    }
}
```

### 4.2 TaskSpawner Interface

```cpp
struct TaskSpawner {
    // Spawn a new task (added to current TaskSet's execution)
    void spawn(KernelFn fn, TaskParams params);

    // Spawn with explicit dependencies
    void spawn(KernelFn fn, TaskParams params, Event* wait_for);

    // Spawn and get event for the new task
    Event* spawn_with_signal(KernelFn fn, TaskParams params);
};
```

### 4.3 Execution Model

Spawned tasks are added to the **current issuing context**:

```
Initial TaskSet: [Router_B0, Router_B1, Router_B2]

After Router_B0 completes (spawns 2 expert tasks):
  Active: [Router_B1, Router_B2, Expert_B0_E3, Expert_B0_E7]

After Router_B1 completes (spawns 2 expert tasks):
  Active: [Router_B2, Expert_B0_E3, Expert_B0_E7, Expert_B1_E1, Expert_B1_E5]
```

**Key**: Spawned tasks participate in the same issuing policy (static/dynamic/etc.).

---

## 5. Stitching

### 5.1 What is Stitching

**Stitching** = Combining tasks that have no dependencies to enable overlapped execution:

```
Without Stitching:
Batch 0: [Prefill0] ─► [Decode0] ─► done
Batch 1:                           [Prefill1] ─► [Decode1] ─► done
         ^^^^^^^^^^^^ wasted time ^^^^^^^^^^^^

With Stitching:
Batch 0: [Prefill0] ─► [Decode0] ─► done
Batch 1: [Prefill1] ─► [Decode1] ─► done
         ^^^^ overlapped execution ^^^^
```

### 5.2 Types of Stitching

| Type | Description | Example |
|------|-------------|---------|
| **Graph Stitching** | Combine multiple task graphs | Merge attention + MLP subgraphs |
| **Batch Stitching** | Interleave independent batches | Run B0 and B1 tasks together |

### 5.3 Programmer Control

Stitching is **explicit** - programmer decides stitch points:

```cpp
// Create tasks for two independent batches
TaskSet batch0_tasks = create_attention_tasks(batch_0_data);
TaskSet batch1_tasks = create_attention_tasks(batch_1_data);

// Stitch them together for overlapped execution
TaskSet stitched = stitch(batch0_tasks, batch1_tasks);

// Dispatch as single unit
dispatch(stitched);
```

**Rationale**: Automatic stitching is complex and may not match programmer intent. Human-in-the-loop stitching gives full control.

---

## 6. Memory and Resources

### 6.1 Tensors as First-Class Citizens

PTO-ISA already has first-class tensor types (`GlobalTensor`, `LocalTensor`, etc.). The extension builds on this:

```cpp
// Tensors are passed to tasks by reference
GlobalTensor<half> kv_cache = ...;
GlobalTensor<half> output = ...;

Task t = {
    .kernel_fn = attention_decode,
    .params = { batch_idx, seq_len, head_idx },
    .inputs = { query, kv_cache },
    .outputs = { output }
};
```

### 6.2 Buffer Allocation

**Primary**: User allocates and manages buffers.

```cpp
// User allocates
GlobalTensor<half> workspace = allocate<half>(workspace_size);

// User passes to tasks
task.workspace = workspace;

// User deallocates when done
deallocate(workspace);
```

**Optional**: User can specify a buffer pool for runtime allocation:

```cpp
BufferPool pool = create_pool(pool_size);
task.workspace = pool.alloc(size);  // Runtime allocates from pool
// ...
pool.free(task.workspace);          // Return to pool
```

---

## 7. Programming Interface

### 7.1 Function-Based API

The API uses simple function calls (not macros, not builders):

```cpp
// Create a task
Task create_task(KernelFn fn, TaskParams params);

// Set dependencies
void wait_for(Task& task, Event* event);
void signal_on_complete(Task& task, Event* event);

// Create a taskset
TaskSet create_taskset(Task* tasks, int count, IssuePolicy policy);

// Stitch tasksets
TaskSet stitch(TaskSet a, TaskSet b);

// Dispatch to AICPU
void dispatch(TaskSet& ts);

// Events
Event* create_event();
void destroy_event(Event* e);
```

### 7.2 Example: FlashInfer-style Decode Attention

```cpp
void dispatch_batched_decode_attention(
    GlobalTensor<half>& query,
    GlobalTensor<half>& kv_cache,
    GlobalTensor<half>& output,
    int* seq_lens,     // Per-batch sequence lengths
    int batch_size,
    int num_heads
) {
    // 1. Determine task count and create tasks
    std::vector<Task> tasks;
    for (int b = 0; b < batch_size; b++) {
        int seq_len = seq_lens[b];
        int tier = select_tier(seq_len);  // Choose kernel variant

        for (int h = 0; h < num_heads; h++) {
            Task t = create_task(
                decode_kernels[tier],  // Tier-specific kernel
                { .batch_idx = b, .head_idx = h, .seq_len = seq_len }
            );
            t.inputs = { query, kv_cache };
            t.outputs = { output };
            tasks.push_back(t);
        }
    }

    // 2. Create taskset with dynamic issuing (variable seq_len)
    TaskSet ts = create_taskset(tasks.data(), tasks.size(), ISSUE_DYNAMIC);

    // 3. Dispatch
    dispatch(ts);
}
```

### 7.3 Example: MoE with Dynamic Routing

```cpp
void dispatch_moe_layer(
    GlobalTensor<half>& input,
    GlobalTensor<half>& output,
    MoEWeights& weights,
    int batch_size,
    int num_experts,
    int top_k
) {
    // 1. Create router tasks
    std::vector<Task> router_tasks;
    for (int b = 0; b < batch_size; b++) {
        Task t = create_task(moe_router_kernel, {
            .batch_idx = b,
            .num_experts = num_experts,
            .top_k = top_k
        });
        t.spawner = &global_spawner;  // Enable task spawning
        router_tasks.push_back(t);
    }

    // 2. Dispatch routers (they will spawn expert tasks dynamically)
    TaskSet router_ts = create_taskset(router_tasks.data(), router_tasks.size(), ISSUE_STATIC);

    // 3. Create completion event
    Event* moe_complete = create_event();
    router_ts.completion = moe_complete;

    // 4. Dispatch
    dispatch(router_ts);

    // 5. Wait for all expert tasks to complete (including spawned ones)
    wait(moe_complete);
}
```

---

## 8. Relationship to Existing Work

### 8.1 Mapping to Megakernels Concepts

| Megakernels | PTO-ISA Extension | Notes |
|-------------|-------------------|-------|
| Instruction | Task | One kernel invocation |
| Instruction tensor | TaskSet | Collection dispatched together |
| Controller warp | AICPU scheduler | Orchestrates issue |
| Worker warps | AICores | Execute tasks |
| Global work queue | ISSUE_DYNAMIC | Work stealing |
| Wave interleaving | Stitching | Overlap compute/memory |
| Barrier counters | Events | Synchronization |

### 8.2 Mapping to CUDA Concepts

| CUDA | PTO-ISA Extension | Notes |
|------|-------------------|-------|
| Stream | (not directly) | Tasks use events instead |
| Event | Event | Same concept |
| cudaStreamWaitEvent | wait_for() | Dependency edge |
| Kernel launch | Task + dispatch | Two-level dispatch |
| cudaDeviceSynchronize | wait(completion_event) | Explicit sync |

### 8.3 Mapping to FlashInfer Concepts

| FlashInfer | PTO-ISA Extension | Notes |
|------------|-------------------|-------|
| Plan phase | TaskSet creation | Build task list |
| request_indices | Task.params.batch_idx | Work assignment |
| kv_tile_indices | Task.params.chunk_idx | Chunking |
| Kernel tiers | decode_kernels[tier] | Size-optimized variants |

---

## 9. Implementation Considerations

### 9.1 AICPU State

AICPU maintains persistent state for dependency tracking:

```cpp
struct AICPUSchedulerState {
    // Event tracking
    EventTable event_table;

    // Active tasks
    TaskQueue pending_tasks;
    TaskQueue ready_tasks;  // Dependencies satisfied

    // Issuing state
    IssueContext current_context;

    // Spawned task accumulator
    SpawnedTaskBuffer spawn_buffer;
};
```

### 9.2 Multi-Threading on AICPU

Multiple AICPU threads can parallelize task processing:

```
AICPU Thread 0: Scheduler (manages dependencies, maintains ready queue)
AICPU Thread 1: Issuer (issues ready tasks to AICores)
AICPU Thread 2: Event handler (processes task completions)
```

**Note**: This internal threading is implementation detail, not exposed to programmer.

### 9.3 Compatibility with PTO-ISA

The extension is **drop-in compatible**:

- Existing PTO-ISA kernels work unchanged
- Tasks call existing kernels via function pointers
- No modifications to kernel interfaces
- New functionality is additive

---

## 10. Summary

### 10.1 Key Design Decisions

| Aspect | Decision | Rationale |
|--------|----------|-----------|
| Task granularity | One kernel call | Matches PTO-ISA model |
| Dependencies | Explicit events | Maximum flexibility |
| Dispatch/Issue | Two-level | Minimize Host-AICPU trips |
| Task spawning | Supported | MoE requires it |
| Stitching | Explicit | Human-in-the-loop control |
| Issue policies | All supported | Workload-dependent |
| Memory | User-managed | Predictable behavior |
| API style | Function-based | Simple, composable |

### 10.2 What This Enables

1. **Variable-length batches**: Different seq_len per batch with tiered kernels
2. **MoE routing**: Dynamic expert selection via task spawning
3. **Batch stitching**: Overlap independent batches
4. **Custom scheduling**: Programmer chooses issue policy
5. **Fine-grained sync**: Events enable precise dependency control

### 10.3 What This Does NOT Do

- **Automatic optimization**: Programmer decides stitching, tiers, etc.
- **Implicit dependencies**: All dependencies are explicit via events
- **Memory management**: User allocates and manages buffers
- **Cross-device**: Multi-NPU not covered (future work)

---

*Analysis Version: 5.0*
*Last Updated: 2025-01-16*
