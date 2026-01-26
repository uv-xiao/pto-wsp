# PTO Workload-Schedule Programming (PTO-WSP) Specification v6

## 1. Overview

This specification defines a **programmable runtime model** for PTO-ISA. Users write **Runtime Programs** that execute on AICPU to control task generation, scheduling, and synchronization.

### 1.1 Design Principles

1. **Programmability over Configuration**: Strategies are code, not enums
2. **Explicit Control**: Users decide exactly how tasks are issued
3. **Composable Primitives**: Small, orthogonal building blocks
4. **Persistent Execution**: Runtime maintains state across dispatches

---

## 2. Core Types

### 2.1 Task

```cpp
namespace pto::runtime {

// Kernel function signature
using KernelFn = void(*)(void* params, void* inputs, void* outputs);

// Task parameters (up to 8 int32 values)
struct TaskParams {
    int32_t data[8];

    int32_t& operator[](int i) { return data[i]; }
    const int32_t& operator[](int i) const { return data[i]; }

    // Named accessors (optional, for readability)
    int32_t batch_idx() const { return data[0]; }
    int32_t head_idx() const { return data[1]; }
    int32_t seq_len() const { return data[2]; }
    // ... user can define more
};

// Resource bindings
struct TaskResources {
    void** inputs;
    int32_t input_count;
    void** outputs;
    int32_t output_count;
    void* workspace;
};

// Task structure
struct Task {
    KernelFn kernel;
    TaskParams params;
    TaskResources resources;

    // Scheduling hints (optional)
    int32_t priority;        // Higher = more urgent
    int32_t affinity;        // Preferred core (-1 = any)
    bool enable_spawn;       // Can this task spawn new tasks?

    // Internal (set by runtime)
    uint32_t task_id;
    int32_t assigned_core;
};

}  // namespace pto::runtime
```

### 2.2 Event

```cpp
namespace pto::runtime {

struct Event {
    uint32_t id;
    // Internal state managed by runtime
};

}  // namespace pto::runtime
```

### 2.3 Barrier

```cpp
namespace pto::runtime {

struct Barrier {
    uint32_t id;
    int32_t target_count;
    // Internal state managed by runtime
};

}  // namespace pto::runtime
```

### 2.4 TaskQueue

```cpp
namespace pto::runtime {

struct TaskQueue {
    uint32_t id;
    // Internal storage managed by runtime
};

}  // namespace pto::runtime
```

### 2.5 WorkRequest

```cpp
namespace pto::runtime {

// Work request from Host to AICPU
struct WorkRequest {
    void* data;              // Pointer to request-specific data
    int32_t data_size;
    uint32_t request_id;

    // Common fields for LLM inference
    int32_t batch_size;
    int32_t* seq_lens;       // Per-batch sequence lengths
    int32_t num_heads;
    int32_t num_experts;     // For MoE
    int32_t top_k;           // For MoE routing

    // Tensors
    void* query;
    void* kv_cache;
    void* output;
    // ... more as needed
};

}  // namespace pto::runtime
```

### 2.6 RuntimeContext

```cpp
namespace pto::runtime {

struct RuntimeContext {
    // Core count
    int32_t num_cores;

    // User state
    void* user_state;

    // --- Task Primitives ---
    Task make_task(KernelFn kernel, TaskParams params);
    Task make_task(KernelFn kernel, TaskParams params, TaskResources resources);

    // --- Queue Primitives ---
    TaskQueue create_queue();
    void enqueue(TaskQueue& q, Task task);
    Task dequeue(TaskQueue& q);
    bool empty(TaskQueue& q);
    int32_t size(TaskQueue& q);

    // --- Issue Primitives ---
    void issue(Task task, int32_t core_id);
    void issue_any(Task task);
    bool is_core_busy(int32_t core_id);
    int32_t get_free_core();  // Returns -1 if none free
    int32_t count_free_cores();

    // --- Event Primitives ---
    Event* create_event();
    void destroy_event(Event* e);
    void signal(Event* e);
    void wait(Event* e);
    bool query(Event* e);
    void reset(Event* e);

    // --- Barrier Primitives ---
    Barrier* create_barrier(int32_t count);
    void destroy_barrier(Barrier* b);
    void arrive(Barrier* b);
    void wait_barrier(Barrier* b);

    // --- Dependency Primitives ---
    void set_dependency(Task& task, Event* wait_for);
    void set_dependencies(Task& task, Event** events, int32_t count);
    void set_signal(Task& task, Event* signal_on_complete);

    // --- Spawn Primitives ---
    void spawn(Task task);
    void spawn_with_dependency(Task task, Event* wait_for);

    // --- Completion Primitives ---
    void wait_all_issued();
    void wait_task(Task& task);

    // --- Control Primitives ---
    bool running();
    void yield();

    // --- State Primitives ---
    void* get_state();
    void set_state(void* state);
};

}  // namespace pto::runtime
```

---

## 3. Handler Signatures

### 3.1 on_dispatch

Called when Host sends a work request:

```cpp
using DispatchHandler = void(*)(WorkRequest& request, RuntimeContext& ctx);
```

**Responsibilities**:
- Parse work request
- Generate tasks
- Set up dependencies
- Enqueue or issue tasks

### 3.2 on_ready

Called when a task's dependencies are satisfied:

```cpp
using ReadyHandler = void(*)(Task& task, RuntimeContext& ctx);
```

**Responsibilities**:
- Decide which core to issue to
- Apply scheduling policy
- Call `ctx.issue(task, core)`

### 3.3 on_complete

Called when a task finishes execution:

```cpp
using CompleteHandler = void(*)(Task& task, int32_t core_id, RuntimeContext& ctx);
```

**Responsibilities**:
- Signal completion events
- Update state
- Trigger dependent tasks

### 3.4 on_spawn

Called when a task spawns a new task:

```cpp
using SpawnHandler = void(*)(Task& parent, Task& child, RuntimeContext& ctx);
```

**Responsibilities**:
- Set dependencies for child
- Enqueue or issue child
- Track spawned tasks

### 3.5 on_init / on_shutdown

```cpp
using InitHandler = void(*)(RuntimeContext& ctx);
using ShutdownHandler = void(*)(RuntimeContext& ctx);
```

---

## 4. RuntimeProgram

### 4.1 Definition

```cpp
namespace pto::runtime {

struct RuntimeProgram {
    // Handlers (user-provided)
    DispatchHandler on_dispatch;
    ReadyHandler on_ready;         // Optional: default issues to any free core
    CompleteHandler on_complete;   // Optional: default signals events
    SpawnHandler on_spawn;         // Optional: default enqueues child

    InitHandler on_init;           // Optional: initialize state
    ShutdownHandler on_shutdown;   // Optional: cleanup

    // State configuration
    size_t state_size;             // Bytes for user state (0 = none)

    // Options
    bool auto_dependency_tracking; // Track read/write dependencies automatically
    int32_t max_concurrent_tasks;  // Limit on running tasks (0 = unlimited)
};

}  // namespace pto::runtime
```

### 4.2 Deployment

```cpp
namespace pto::runtime {

// Deploy runtime program to AICPU
void deploy(const RuntimeProgram& program);

// Stop runtime
void stop();

// Dispatch work request from Host
void dispatch(WorkRequest& request);

// Wait for all work to complete
void synchronize();

}  // namespace pto::runtime
```

---

## 5. Programming Patterns

### 5.1 Minimal Runtime (Default Behavior)

```cpp
RuntimeProgram minimal = {
    .on_dispatch = [](WorkRequest& req, RuntimeContext& ctx) {
        // Generate tasks
        for (int b = 0; b < req.batch_size; b++) {
            for (int h = 0; h < req.num_heads; h++) {
                Task t = ctx.make_task(attention_kernel, {b, h, req.seq_lens[b]});
                ctx.issue_any(t);  // Issue to any free core
            }
        }
    },
    // Other handlers use defaults
};
```

### 5.2 Round-Robin Static Scheduling

```cpp
RuntimeProgram round_robin = {
    .on_dispatch = [](WorkRequest& req, RuntimeContext& ctx) {
        int task_idx = 0;
        for (int b = 0; b < req.batch_size; b++) {
            for (int h = 0; h < req.num_heads; h++) {
                Task t = ctx.make_task(attention_kernel, {b, h, req.seq_lens[b]});
                int core = task_idx % ctx.num_cores;
                ctx.issue(t, core);
                task_idx++;
            }
        }
    },
};
```

### 5.3 Batch-Affinity Scheduling

```cpp
RuntimeProgram batch_affinity = {
    .on_dispatch = [](WorkRequest& req, RuntimeContext& ctx) {
        // Assign batches to core groups
        int cores_per_batch = ctx.num_cores / req.batch_size;

        for (int b = 0; b < req.batch_size; b++) {
            int base_core = b * cores_per_batch;
            int local_idx = 0;

            for (int h = 0; h < req.num_heads; h++) {
                Task t = ctx.make_task(attention_kernel, {b, h, req.seq_lens[b]});
                int core = base_core + (local_idx % cores_per_batch);
                ctx.issue(t, core);
                local_idx++;
            }
        }
    },
};
```

### 5.4 Work Stealing with Queue

```cpp
struct WorkStealingState {
    TaskQueue shared_queue;
};

RuntimeProgram work_stealing = {
    .on_init = [](RuntimeContext& ctx) {
        auto* state = (WorkStealingState*)ctx.get_state();
        state->shared_queue = ctx.create_queue();
    },

    .on_dispatch = [](WorkRequest& req, RuntimeContext& ctx) {
        auto* state = (WorkStealingState*)ctx.get_state();

        // Enqueue all tasks
        for (int b = 0; b < req.batch_size; b++) {
            for (int h = 0; h < req.num_heads; h++) {
                Task t = ctx.make_task(attention_kernel, {b, h, req.seq_lens[b]});
                ctx.enqueue(state->shared_queue, t);
            }
        }

        // Issue to free cores
        while (!ctx.empty(state->shared_queue)) {
            int core = ctx.get_free_core();
            if (core >= 0) {
                Task t = ctx.dequeue(state->shared_queue);
                ctx.issue(t, core);
            } else {
                ctx.yield();
            }
        }
    },

    .on_complete = [](Task& task, int32_t core, RuntimeContext& ctx) {
        auto* state = (WorkStealingState*)ctx.get_state();

        // Issue next task to the now-free core
        if (!ctx.empty(state->shared_queue)) {
            Task t = ctx.dequeue(state->shared_queue);
            ctx.issue(t, core);
        }
    },

    .state_size = sizeof(WorkStealingState),
};
```

### 5.5 Pipeline with Events

```cpp
RuntimeProgram pipeline = {
    .on_dispatch = [](WorkRequest& req, RuntimeContext& ctx) {
        std::vector<Event*> stage1_events(req.batch_size);
        std::vector<Event*> stage2_events(req.batch_size);

        // Stage 1
        for (int b = 0; b < req.batch_size; b++) {
            stage1_events[b] = ctx.create_event();
            Task t = ctx.make_task(stage1_kernel, {b});
            ctx.set_signal(t, stage1_events[b]);
            ctx.issue_any(t);
        }

        // Stage 2 (depends on stage 1)
        for (int b = 0; b < req.batch_size; b++) {
            stage2_events[b] = ctx.create_event();
            Task t = ctx.make_task(stage2_kernel, {b});
            ctx.set_dependency(t, stage1_events[b]);
            ctx.set_signal(t, stage2_events[b]);
            ctx.issue_any(t);
        }

        // Stage 3 (depends on stage 2)
        for (int b = 0; b < req.batch_size; b++) {
            Task t = ctx.make_task(stage3_kernel, {b});
            ctx.set_dependency(t, stage2_events[b]);
            ctx.issue_any(t);
        }

        // Cleanup events (after all complete)
        ctx.wait_all_issued();
        for (int b = 0; b < req.batch_size; b++) {
            ctx.destroy_event(stage1_events[b]);
            ctx.destroy_event(stage2_events[b]);
        }
    },
};
```

### 5.6 MoE with Dynamic Spawning

```cpp
struct MoEState {
    Event* batch_complete[MAX_BATCH];
    int32_t experts_pending[MAX_BATCH];
};

// Router kernel that spawns expert tasks
void moe_router_kernel(void* params_ptr, void* inputs_ptr, void* outputs_ptr) {
    // This runs on AICore, can spawn via special mechanism
    // (Implementation depends on how spawn is exposed to kernels)
}

RuntimeProgram moe_runtime = {
    .on_init = [](RuntimeContext& ctx) {
        auto* state = (MoEState*)ctx.get_state();
        for (int b = 0; b < MAX_BATCH; b++) {
            state->batch_complete[b] = nullptr;
            state->experts_pending[b] = 0;
        }
    },

    .on_dispatch = [](WorkRequest& req, RuntimeContext& ctx) {
        auto* state = (MoEState*)ctx.get_state();

        // Create completion events for each batch
        for (int b = 0; b < req.batch_size; b++) {
            state->batch_complete[b] = ctx.create_event();
            state->experts_pending[b] = 0;
        }

        // Launch router tasks (they will spawn expert tasks)
        for (int b = 0; b < req.batch_size; b++) {
            Task router = ctx.make_task(moe_router_kernel, {b, req.num_experts, req.top_k});
            router.enable_spawn = true;
            ctx.issue_any(router);
        }
    },

    .on_spawn = [](Task& parent, Task& child, RuntimeContext& ctx) {
        auto* state = (MoEState*)ctx.get_state();
        int batch_idx = parent.params.batch_idx();

        // Track spawned expert
        state->experts_pending[batch_idx]++;

        // Issue expert task
        ctx.issue_any(child);
    },

    .on_complete = [](Task& task, int32_t core, RuntimeContext& ctx) {
        auto* state = (MoEState*)ctx.get_state();

        // Check if this is an expert task
        if (task.kernel == expert_kernel) {
            int batch_idx = task.params.batch_idx();
            state->experts_pending[batch_idx]--;

            // Signal batch complete when all experts done
            if (state->experts_pending[batch_idx] == 0) {
                ctx.signal(state->batch_complete[batch_idx]);
            }
        }
    },

    .state_size = sizeof(MoEState),
};
```

### 5.7 Stitching (Interleaved Batches)

```cpp
RuntimeProgram stitched = {
    .on_dispatch = [](WorkRequest& req, RuntimeContext& ctx) {
        // Collect all tasks
        std::vector<Task> tasks;
        for (int b = 0; b < req.batch_size; b++) {
            for (int h = 0; h < req.num_heads; h++) {
                Task t = ctx.make_task(attention_kernel, {b, h, req.seq_lens[b]});
                tasks.push_back(t);
            }
        }

        // Reorder: interleave by head first, then batch
        // This stitches independent batches together
        std::sort(tasks.begin(), tasks.end(), [](Task& a, Task& b) {
            int a_head = a.params[1], b_head = b.params[1];
            int a_batch = a.params[0], b_batch = b.params[0];
            return (a_head < b_head) || (a_head == b_head && a_batch < b_batch);
        });

        // Issue in stitched order
        for (size_t i = 0; i < tasks.size(); i++) {
            int core = i % ctx.num_cores;
            ctx.issue(tasks[i], core);
        }
    },
};
```

---

## 6. Advanced Features

### 6.1 Tiered Kernels

```cpp
// Register tiered kernels for different sequence lengths
KernelFn attention_tiers[] = {
    attention_tier0,  // seq_len <= 1024
    attention_tier1,  // seq_len <= 4096
    attention_tier2,  // seq_len <= 16384
    attention_tier3,  // seq_len > 16384
};

int32_t tier_bounds[] = { 1024, 4096, 16384, INT32_MAX };

KernelFn select_tier(int32_t seq_len) {
    for (int i = 0; i < 4; i++) {
        if (seq_len <= tier_bounds[i]) return attention_tiers[i];
    }
    return attention_tiers[3];
}

RuntimeProgram tiered = {
    .on_dispatch = [](WorkRequest& req, RuntimeContext& ctx) {
        for (int b = 0; b < req.batch_size; b++) {
            KernelFn kernel = select_tier(req.seq_lens[b]);
            for (int h = 0; h < req.num_heads; h++) {
                Task t = ctx.make_task(kernel, {b, h, req.seq_lens[b]});
                ctx.issue_any(t);
            }
        }
    },
};
```

### 6.2 Priority-Based Scheduling

```cpp
struct PriorityState {
    TaskQueue high_priority;
    TaskQueue low_priority;
};

RuntimeProgram priority_scheduler = {
    .on_init = [](RuntimeContext& ctx) {
        auto* state = (PriorityState*)ctx.get_state();
        state->high_priority = ctx.create_queue();
        state->low_priority = ctx.create_queue();
    },

    .on_dispatch = [](WorkRequest& req, RuntimeContext& ctx) {
        auto* state = (PriorityState*)ctx.get_state();

        for (int b = 0; b < req.batch_size; b++) {
            for (int h = 0; h < req.num_heads; h++) {
                Task t = ctx.make_task(attention_kernel, {b, h, req.seq_lens[b]});

                // Short sequences are high priority (lower latency)
                if (req.seq_lens[b] < 1024) {
                    t.priority = 10;
                    ctx.enqueue(state->high_priority, t);
                } else {
                    t.priority = 1;
                    ctx.enqueue(state->low_priority, t);
                }
            }
        }

        // Issue high priority first
        while (!ctx.empty(state->high_priority)) {
            int core = ctx.get_free_core();
            if (core >= 0) {
                Task t = ctx.dequeue(state->high_priority);
                ctx.issue(t, core);
            } else {
                break;
            }
        }

        // Then low priority
        while (!ctx.empty(state->low_priority)) {
            int core = ctx.get_free_core();
            if (core >= 0) {
                Task t = ctx.dequeue(state->low_priority);
                ctx.issue(t, core);
            } else {
                break;
            }
        }
    },

    .on_complete = [](Task& task, int32_t core, RuntimeContext& ctx) {
        auto* state = (PriorityState*)ctx.get_state();

        // Issue next task to freed core (high priority first)
        if (!ctx.empty(state->high_priority)) {
            Task t = ctx.dequeue(state->high_priority);
            ctx.issue(t, core);
        } else if (!ctx.empty(state->low_priority)) {
            Task t = ctx.dequeue(state->low_priority);
            ctx.issue(t, core);
        }
    },

    .state_size = sizeof(PriorityState),
};
```

---

## 7. Error Handling

```cpp
namespace pto::runtime {

enum class Error : int32_t {
    SUCCESS = 0,
    INVALID_TASK = -1,
    INVALID_CORE = -2,
    QUEUE_FULL = -3,
    EVENT_NOT_FOUND = -4,
    SPAWN_NOT_ALLOWED = -5,
    OUT_OF_MEMORY = -6,
};

// Get last error
Error get_last_error();
const char* error_string(Error e);

// Error callback
using ErrorCallback = void(*)(Error e, const char* msg);
void set_error_callback(ErrorCallback cb);

}  // namespace pto::runtime
```

---

## 8. Integration with PTO-ISA

### 8.1 Kernel Compatibility

Existing PTO-ISA kernels work unchanged:

```cpp
// Existing PTO-ISA kernel
template<int TileSize>
__global__ void my_kernel(__gm__ half* input, __gm__ half* output, int size) {
    // ... existing code
}

// Wrap for runtime
void my_kernel_wrapper(void* params, void* inputs, void* outputs) {
    auto* p = (TaskParams*)params;
    auto** in = (void**)inputs;
    auto** out = (void**)outputs;

    my_kernel<64><<<grid, block>>>(
        (half*)in[0], (half*)out[0], p->data[0]
    );
}
```

### 8.2 Resource Binding

```cpp
// Using PTO-ISA GlobalTensor
GlobalTensor<half> query = ...;
GlobalTensor<half> output = ...;

TaskResources res = {
    .inputs = (void*[]){ query.data() },
    .input_count = 1,
    .outputs = (void*[]){ output.data() },
    .output_count = 1,
};

Task t = ctx.make_task(my_kernel_wrapper, {size}, res);
```

---

## 9. Summary

### 9.1 Key Primitives

| Category | Primitives |
|----------|------------|
| **Task** | `make_task`, `set_dependency`, `set_signal` |
| **Queue** | `create_queue`, `enqueue`, `dequeue`, `empty`, `size` |
| **Issue** | `issue`, `issue_any`, `get_free_core`, `is_core_busy` |
| **Sync** | `create_event`, `signal`, `wait`, `create_barrier`, `arrive` |
| **Spawn** | `spawn`, `spawn_with_dependency` |
| **Control** | `running`, `yield`, `wait_all_issued` |
| **State** | `get_state`, `set_state` |

### 9.2 Handler Model

| Handler | When Called | Purpose |
|---------|-------------|---------|
| `on_init` | Runtime starts | Initialize state |
| `on_dispatch` | Work request arrives | Generate and schedule tasks |
| `on_ready` | Task dependencies met | Decide issue strategy |
| `on_complete` | Task finishes | Handle completion, signal events |
| `on_spawn` | Task spawns child | Handle dynamic generation |
| `on_shutdown` | Runtime stops | Cleanup |

### 9.3 What Users Program

Users write code that:
1. **Generates tasks** from work requests
2. **Manages dependencies** with events/barriers
3. **Decides scheduling** via custom issue logic
4. **Handles dynamics** via spawn handlers
5. **Maintains state** across dispatches

---

*Specification Version: 6.0*
*Last Updated: 2025-01-16*
