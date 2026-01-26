# PTO Workload-Schedule Programming (PTO-WSP): Conceptual Analysis v6

## Executive Summary

This document presents a **programmable runtime model** for PTO-ISA, enabling users to write **Runtime Programs** that execute on AICPU to control dynamic LLM workloads. The key insight is that dispatch and issue strategies cannot be fixed enums—they must be **user-programmable logic**.

**Core Principle**: The extension provides primitives for users to **program** the AICPU runtime behavior, not just configure it.

---

## 1. Problem Re-Statement

### 1.1 What Must Be Programmed

From requirements.md, the runtime must let users express:

```
1. 对于单个kernel而言，怎么定调用接口，怎么给它准备参数。
2. 对于多个kernel而言，怎么表达图（即kernel的并行、依赖关系），怎么分配内存，怎么调度。
```

This requires **programmability** at multiple levels:

| Level | What to Program | Example |
|-------|-----------------|---------|
| **Task Definition** | How to create a task from a kernel | `task = make_task(kernel, params)` |
| **Task Generation** | Control flow for creating tasks | `for b in batch: for h in heads: ...` |
| **Dependency Logic** | When tasks can run | `task_b waits for task_a if same_buffer` |
| **Issue Strategy** | Which AICore gets which task | `core = hash(task.batch_idx) % num_cores` |
| **Synchronization** | How to coordinate execution | `wait_until(all_experts_done)` |

### 1.2 Why Enums Are Insufficient

v5 used:
```cpp
enum class IssuePolicy { STATIC, DYNAMIC, HYBRID, ADAPTIVE };
```

But real scheduling needs are:
```cpp
// "Schedule same-batch tasks together to avoid cold start"
for (core = 0; core < num_cores; core++) {
    batch_idx = core;
    for (task in tasks_of_batch[batch_idx]) {
        issue(task, core);
    }
}

// vs "Interleave batches for better overlap"
for (task_idx = 0; task_idx < total_tasks; task_idx++) {
    core = task_idx % num_cores;
    issue(tasks[task_idx], core);
}
```

These are **algorithms**, not enum values. Users must be able to write them.

### 1.3 The AICPU Runtime Program

The extension enables writing a **Runtime Program** that:

```
┌─────────────────────────────────────────────────────────────────────┐
│ RUNTIME PROGRAM (runs on AICPU)                                     │
│                                                                     │
│ 1. RECEIVES work requests from Host                                │
│ 2. GENERATES tasks based on runtime data (seq_lens, routing, etc.) │
│ 3. MANAGES dependencies between tasks                               │
│ 4. ISSUES tasks to AICores with user-defined strategy              │
│ 5. HANDLES completions and spawned tasks                           │
│ 6. SYNCHRONIZES as needed                                          │
│                                                                     │
│ This is a LONG-LIVED PROGRAM with PERSISTENT STATE                 │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. Core Model: Runtime Program

### 2.1 Structure

A Runtime Program consists of:

```
┌─────────────────────────────────────────────────────────────────────┐
│ RUNTIME PROGRAM                                                     │
│                                                                     │
│ ┌─────────────────────────────────────────────────────────────────┐ │
│ │ STATE                                                           │ │
│ │ - Task queues (pending, ready, running)                        │ │
│ │ - Event table (for synchronization)                            │ │
│ │ - AICore status (which core is busy/free)                      │ │
│ │ - User-defined state (counters, buffers, etc.)                 │ │
│ └─────────────────────────────────────────────────────────────────┘ │
│                                                                     │
│ ┌─────────────────────────────────────────────────────────────────┐ │
│ │ HANDLERS (user-programmable)                                    │ │
│ │                                                                 │ │
│ │ on_dispatch(work_request):                                      │ │
│ │     # Generate tasks from work request                          │ │
│ │     # Add to pending queue                                      │ │
│ │                                                                 │ │
│ │ on_ready(task):                                                 │ │
│ │     # Task's dependencies satisfied                             │ │
│ │     # Decide which AICore to issue to                           │ │
│ │                                                                 │ │
│ │ on_complete(task, core):                                        │ │
│ │     # Task finished on core                                     │ │
│ │     # Signal events, spawn new tasks, etc.                      │ │
│ │                                                                 │ │
│ │ on_spawn(parent_task, new_task):                                │ │
│ │     # Dynamic task generation (MoE)                             │ │
│ │     # Add to pending queue with dependencies                    │ │
│ └─────────────────────────────────────────────────────────────────┘ │
│                                                                     │
│ ┌─────────────────────────────────────────────────────────────────┐ │
│ │ MAIN LOOP (runtime-provided, calls handlers)                    │ │
│ │                                                                 │ │
│ │ while (running) {                                               │ │
│ │     if (has_dispatch()) call on_dispatch(...)                   │ │
│ │     for (task in newly_ready()) call on_ready(task)             │ │
│ │     for (completion in completions()) call on_complete(...)     │ │
│ │     for (spawn in spawns()) call on_spawn(...)                  │ │
│ │ }                                                               │ │
│ └─────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 Event-Driven vs Imperative

Two programming styles are possible:

**Event-Driven** (handlers):
```cpp
void on_ready(Task& task, RuntimeContext& ctx) {
    // Pick core based on batch affinity
    int core = task.params.batch_idx % ctx.num_cores;
    ctx.issue(task, core);
}
```

**Imperative** (main loop):
```cpp
void runtime_main(RuntimeContext& ctx) {
    while (ctx.running()) {
        WorkRequest req = ctx.receive();

        // Generate tasks
        for (int b = 0; b < req.batch_size; b++) {
            for (int h = 0; h < req.num_heads; h++) {
                Task t = make_task(attention_kernel, {b, h, req.seq_lens[b]});
                ctx.enqueue(t);
            }
        }

        // Issue with custom strategy
        while (ctx.has_ready_tasks()) {
            Task t = ctx.pop_ready();
            int core = my_scheduling_logic(t, ctx);
            ctx.issue(t, core);
        }

        // Wait for completion
        ctx.sync();
    }
}
```

Both styles should be supported. The imperative style gives maximum control.

---

## 3. Primitives

### 3.1 Task Primitives

```cpp
// Task creation
Task make_task(KernelFn kernel, TaskParams params);
Task make_task(KernelFn kernel, TaskParams params, TaskResources resources);

// Task properties
void set_dependency(Task& task, Event* wait_for);
void set_signal(Task& task, Event* signal_on_complete);
void set_priority(Task& task, int priority);
void set_affinity(Task& task, int preferred_core);
```

### 3.2 Queue Primitives

```cpp
// Task queues (user can create multiple)
TaskQueue create_queue();
void enqueue(TaskQueue& q, Task task);
Task dequeue(TaskQueue& q);
bool empty(TaskQueue& q);
int size(TaskQueue& q);

// Filtering
Task dequeue_if(TaskQueue& q, Predicate pred);
void reorder(TaskQueue& q, Comparator cmp);
```

### 3.3 Issue Primitives

```cpp
// Direct issue to specific core
void issue(Task task, int core_id);

// Issue to any available core
void issue_any(Task task);

// Batch issue
void issue_batch(Task* tasks, int count, int* core_assignments);

// Query core status
bool is_core_busy(int core_id);
int get_free_core();
int count_free_cores();
```

### 3.4 Synchronization Primitives

```cpp
// Events
Event* create_event();
void signal(Event* e);
void wait(Event* e);
bool query(Event* e);
void reset(Event* e);

// Barriers
Barrier* create_barrier(int count);
void arrive(Barrier* b);
void wait_barrier(Barrier* b);

// Completion tracking
void wait_all_issued();
void wait_task(Task& task);
void wait_any(Task* tasks, int count, int* completed_idx);
```

### 3.5 Dynamic Task Generation

```cpp
// Spawn from within a task (for MoE)
void spawn(Task new_task);
void spawn_with_dependency(Task new_task, Event* wait_for);

// Spawn multiple
void spawn_batch(Task* tasks, int count);

// Conditional spawn
void spawn_if(bool condition, Task task);
```

### 3.6 Control Flow Primitives

```cpp
// Receive work from Host
WorkRequest receive();
bool has_pending_request();

// Runtime control
bool running();
void yield();  // Let other AICPU threads run
void sleep_until(Event* e);

// State
void* get_state();
void set_state(void* state);
```

---

## 4. Programming Patterns

### 4.1 Pattern: Static Round-Robin Issue

```cpp
void on_dispatch(WorkRequest& req, RuntimeContext& ctx) {
    int task_idx = 0;
    for (int b = 0; b < req.batch_size; b++) {
        for (int h = 0; h < req.num_heads; h++) {
            Task t = make_task(attention_kernel, {b, h, req.seq_lens[b]});
            int core = task_idx % ctx.num_cores;
            ctx.issue(t, core);
            task_idx++;
        }
    }
}
```

### 4.2 Pattern: Batch-Affinity Issue

```cpp
void on_dispatch(WorkRequest& req, RuntimeContext& ctx) {
    // Each batch assigned to specific cores
    int cores_per_batch = ctx.num_cores / req.batch_size;

    for (int b = 0; b < req.batch_size; b++) {
        int base_core = b * cores_per_batch;
        int task_in_batch = 0;

        for (int h = 0; h < req.num_heads; h++) {
            Task t = make_task(attention_kernel, {b, h, req.seq_lens[b]});
            int core = base_core + (task_in_batch % cores_per_batch);
            ctx.issue(t, core);
            task_in_batch++;
        }
    }
}
```

### 4.3 Pattern: Work Stealing

```cpp
void runtime_main(RuntimeContext& ctx) {
    TaskQueue shared_queue = create_queue();

    while (ctx.running()) {
        // Generate all tasks into shared queue
        WorkRequest req = ctx.receive();
        for (int b = 0; b < req.batch_size; b++) {
            for (int h = 0; h < req.num_heads; h++) {
                Task t = make_task(attention_kernel, {b, h, req.seq_lens[b]});
                enqueue(shared_queue, t);
            }
        }

        // Cores pull from shared queue (work stealing)
        while (!empty(shared_queue)) {
            int core = ctx.get_free_core();
            if (core >= 0) {
                Task t = dequeue(shared_queue);
                ctx.issue(t, core);
            } else {
                ctx.yield();  // Wait for a core to free up
            }
        }

        ctx.wait_all_issued();
    }
}
```

### 4.4 Pattern: Pipeline with Dependencies

```cpp
void on_dispatch(WorkRequest& req, RuntimeContext& ctx) {
    Event* stage1_done[MAX_BATCH];
    Event* stage2_done[MAX_BATCH];

    // Stage 1: All batches
    for (int b = 0; b < req.batch_size; b++) {
        stage1_done[b] = create_event();
        Task t = make_task(stage1_kernel, {b});
        set_signal(t, stage1_done[b]);
        ctx.issue_any(t);
    }

    // Stage 2: Wait for stage 1 per-batch
    for (int b = 0; b < req.batch_size; b++) {
        stage2_done[b] = create_event();
        Task t = make_task(stage2_kernel, {b});
        set_dependency(t, stage1_done[b]);
        set_signal(t, stage2_done[b]);
        ctx.issue_any(t);
    }

    // Stage 3: Wait for stage 2
    for (int b = 0; b < req.batch_size; b++) {
        Task t = make_task(stage3_kernel, {b});
        set_dependency(t, stage2_done[b]);
        ctx.issue_any(t);
    }
}
```

### 4.5 Pattern: MoE Dynamic Routing

```cpp
// Router task spawns expert tasks
void moe_router_kernel_with_spawn(TaskParams params, RuntimeContext& ctx) {
    int batch_idx = params.batch_idx;
    int num_tokens = params.num_tokens;
    int top_k = params.top_k;

    // Compute routing (which experts for which tokens)
    int* routing = compute_routing(...);  // Returns [num_tokens * top_k]

    // Spawn expert tasks dynamically
    for (int t = 0; t < num_tokens; t++) {
        for (int k = 0; k < top_k; k++) {
            int expert_id = routing[t * top_k + k];
            Task expert_task = make_task(
                expert_kernels[expert_id],
                {batch_idx, t, expert_id}
            );
            ctx.spawn(expert_task);
        }
    }
}

void on_dispatch(WorkRequest& req, RuntimeContext& ctx) {
    // Create router tasks that will spawn expert tasks
    for (int b = 0; b < req.batch_size; b++) {
        Task router = make_task(moe_router_kernel_with_spawn, {b, req.num_tokens, req.top_k});
        router.enable_spawn = true;  // Allow this task to spawn
        ctx.issue_any(router);
    }
}

void on_spawn(Task& parent, Task& child, RuntimeContext& ctx) {
    // Expert task spawned by router
    // Issue with affinity to same core group as parent
    int core = parent.core_id;  // Or custom logic
    ctx.issue(child, core);
}
```

### 4.6 Pattern: Stitching Independent Batches

```cpp
void on_dispatch(WorkRequest& req, RuntimeContext& ctx) {
    // Stitch: interleave tasks from different batches for overlap
    std::vector<Task> all_tasks;

    // Generate all tasks
    for (int b = 0; b < req.batch_size; b++) {
        for (int h = 0; h < req.num_heads; h++) {
            Task t = make_task(attention_kernel, {b, h, req.seq_lens[b]});
            t.batch_idx = b;  // Tag for later
            all_tasks.push_back(t);
        }
    }

    // Sort to interleave: B0_H0, B1_H0, B2_H0, B0_H1, B1_H1, ...
    std::sort(all_tasks.begin(), all_tasks.end(), [](Task& a, Task& b) {
        return a.head_idx < b.head_idx ||
               (a.head_idx == b.head_idx && a.batch_idx < b.batch_idx);
    });

    // Issue in interleaved order
    for (int i = 0; i < all_tasks.size(); i++) {
        int core = i % ctx.num_cores;
        ctx.issue(all_tasks[i], core);
    }
}
```

---

## 5. Runtime State Management

### 5.1 Built-in State

The runtime maintains:

```cpp
struct RuntimeState {
    // Task management
    TaskQueue pending;      // Tasks waiting for dependencies
    TaskQueue ready;        // Tasks ready to issue
    Task* running[MAX_CORES];  // Currently running on each core

    // Synchronization
    EventTable events;
    BarrierTable barriers;

    // AICore status
    bool core_busy[MAX_CORES];
    int tasks_issued;
    int tasks_completed;

    // Communication with Host
    WorkRequestQueue from_host;
    CompletionQueue to_host;
};
```

### 5.2 User-Defined State

Users can maintain additional state:

```cpp
struct MySchedulerState {
    int tasks_per_batch[MAX_BATCH];
    int completed_per_batch[MAX_BATCH];
    float load_per_core[MAX_CORES];
    // ... custom state
};

void on_dispatch(WorkRequest& req, RuntimeContext& ctx) {
    MySchedulerState* state = (MySchedulerState*)ctx.get_state();
    // Use and update state...
}
```

---

## 6. Execution Model

### 6.1 Host-AICPU Interaction

```
Host                          AICPU Runtime Program
  │                                    │
  │ ─── dispatch(WorkRequest) ───────► │
  │       (async, ~3μs)                │
  │                                    │ ◄── on_dispatch() runs
  │                                    │ ◄── generates tasks
  │                                    │ ◄── issues to AICores
  │                                    │
  │ ◄── completion notification ────── │
  │       (when all tasks done)        │
  │                                    │
  │ ─── next dispatch ───────────────► │
  │                                    │
```

### 6.2 AICPU-AICore Interaction

```
AICPU Runtime                        AICores
     │                                  │
     │ ─── issue(task, core_0) ───────► │ Core 0 executes
     │ ─── issue(task, core_1) ───────► │ Core 1 executes
     │       (~0μs each)                │
     │                                  │
     │ ◄── completion(core_0) ───────── │
     │ ◄── completion(core_1) ───────── │
     │                                  │
     │ ─── on_complete() runs           │
     │ ─── signal events                │
     │ ─── issue more tasks ──────────► │
     │                                  │
```

### 6.3 Task Lifecycle

```
┌────────┐   enqueue   ┌─────────┐  deps satisfied  ┌───────┐
│ Created │ ──────────► │ Pending │ ───────────────► │ Ready │
└────────┘             └─────────┘                  └───────┘
                                                        │
                                                        │ issue()
                                                        ▼
┌───────────┐  completion  ┌─────────┐
│ Completed │ ◄─────────── │ Running │
└───────────┘              └─────────┘
      │
      │ on_complete()
      ▼
 signal events, spawn tasks, etc.
```

---

## 7. Comparison with Previous Designs

### 7.1 v5 vs v6

| Aspect | v5 | v6 |
|--------|----|----|
| Dispatch strategy | `IssuePolicy` enum | User-written code |
| Issue logic | Fixed round-robin/stealing | Programmable |
| Task generation | TaskSet built externally | Generated in handlers |
| Synchronization | Events as data | Events + programmatic control |
| State | Implicit in TaskSet | Explicit, user-managed |
| MoE support | TaskSpawner interface | spawn() in handlers |

### 7.2 Key Improvements

1. **Full programmability**: Any scheduling strategy expressible
2. **Human-in-the-loop**: Users can tune algorithms, not just parameters
3. **Persistent runtime**: State maintained across dispatches
4. **Control flow**: Loops, conditionals, custom logic in handlers

---

## 8. Relationship to Research

### 8.1 Megakernels

| Megakernels | v6 Runtime |
|-------------|------------|
| Controller warp | Main loop / handlers |
| Instruction fetch | Task dequeue |
| Page allocation | User-managed buffers |
| Work stealing queue | TaskQueue primitives |
| Barrier counters | Event/Barrier primitives |

### 8.2 CUDA Streams/Events

| CUDA | v6 Runtime |
|------|------------|
| Stream | TaskQueue (but more flexible) |
| Event | Event |
| cudaStreamWaitEvent | set_dependency() |
| Kernel launch | issue() |
| cudaDeviceSynchronize | wait_all_issued() |

### 8.3 Whippletree

| Whippletree | v6 Runtime |
|-------------|------------|
| Persistent threads | Long-lived runtime program |
| Work queue | TaskQueue |
| Procedure dispatch | Task with kernel pointer |
| Dynamic task generation | spawn() |

---

## 9. Summary

### 9.1 Core Insight

The runtime extension is not a library with fixed behaviors—it's a **programming model** for writing AICPU schedulers.

### 9.2 Key Components

| Component | Purpose |
|-----------|---------|
| **Task** | Unit of work (one kernel invocation) |
| **TaskQueue** | User-managed task containers |
| **Event/Barrier** | Synchronization primitives |
| **Handlers** | User-programmable logic (on_dispatch, on_ready, etc.) |
| **issue()** | Primitive to launch task on specific AICore |
| **spawn()** | Dynamic task generation |
| **State** | Persistent data across dispatches |

### 9.3 What Users Write

```cpp
// A complete runtime program
RuntimeProgram my_scheduler = {
    .on_dispatch = [](WorkRequest& req, RuntimeContext& ctx) {
        // Custom task generation
        // Custom dependency setup
    },

    .on_ready = [](Task& task, RuntimeContext& ctx) {
        // Custom issue strategy
        int core = my_core_selection(task, ctx);
        ctx.issue(task, core);
    },

    .on_complete = [](Task& task, int core, RuntimeContext& ctx) {
        // Custom completion handling
        // Signal events, spawn tasks, etc.
    },

    .on_spawn = [](Task& parent, Task& child, RuntimeContext& ctx) {
        // Handle dynamically spawned tasks
    },

    .state_size = sizeof(MySchedulerState),
    .init_state = my_state_init,
};

// Deploy to AICPU
deploy_runtime(my_scheduler);
```

---

*Analysis Version: 6.0*
*Last Updated: 2025-01-16*
