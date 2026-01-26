# PTO Workload-Schedule Programming (PTO-WSP) Specification v5

## 1. Overview

This specification defines the **AICPU Runtime Extension** for PTO-ISA, enabling dynamic LLM workloads with flexible task scheduling and dependency management.

### 1.1 Design Goals

| Priority | Goal | Description |
|----------|------|-------------|
| 1 | **Dynamic workloads** | Support variable-length sequences, MoE routing |
| 2 | **Throughput** | Maximize AICore utilization |
| 3 | **Latency** | Minimize dispatch overhead |
| 4 | **Ease of use** | Simple, composable API |
| 5 | **Memory** | Efficient buffer usage |

### 1.2 Scope

- **In scope**: Task creation, dependency management, issuing policies, task spawning
- **Out of scope**: Automatic optimization, memory management, cross-device

---

## 2. Core Types

### 2.1 Task

```cpp
namespace pto::runtime {

// Forward declarations
struct Event;
struct TaskSpawner;

// Kernel function pointer type
using KernelFn = void(*)(void* params, void* inputs, void* outputs);

// Task parameters (flexible, kernel-specific)
struct TaskParams {
    int32_t values[8];  // Up to 8 int32 parameters

    // Convenience accessors
    int32_t& operator[](int i) { return values[i]; }
    const int32_t& operator[](int i) const { return values[i]; }
};

// Resource binding for a task
struct TaskResources {
    void** inputs;       // Array of input tensor pointers
    int32_t input_count;
    void** outputs;      // Array of output tensor pointers
    int32_t output_count;
    void* workspace;     // Optional workspace buffer
};

// A single task (one kernel invocation)
struct Task {
    KernelFn kernel;           // Kernel to execute
    TaskParams params;         // Runtime parameters
    TaskResources resources;   // Tensor bindings

    // Dependencies (optional)
    Event** wait_events;       // Events to wait for before execution
    int32_t wait_count;
    Event* signal_event;       // Event to signal on completion

    // Task spawning (optional, for MoE)
    TaskSpawner* spawner;      // If non-null, task can spawn new tasks
};

}  // namespace pto::runtime
```

### 2.2 Event

```cpp
namespace pto::runtime {

// Opaque event handle
struct Event {
    uint64_t id;  // Internal identifier
};

// Event creation/destruction
Event* create_event();
void destroy_event(Event* event);

// Event operations
void reset_event(Event* event);      // Reset to unsignaled state
bool query_event(Event* event);      // Non-blocking: is event signaled?
void wait_event(Event* event);       // Blocking wait for event

}  // namespace pto::runtime
```

### 2.3 TaskSet

```cpp
namespace pto::runtime {

// Issue policy for tasks in a taskset
enum class IssuePolicy : uint8_t {
    STATIC,     // Round-robin assignment to AICores
    DYNAMIC,    // Work stealing from shared queue
    HYBRID,     // Static with dynamic fallback
    ADAPTIVE    // Runtime selects based on variance
};

// TaskSet: collection of tasks dispatched together
struct TaskSet {
    Task* tasks;              // Array of tasks
    int32_t task_count;       // Number of tasks

    IssuePolicy policy;       // How to issue tasks
    Event* completion;        // Optional: signaled when all tasks complete
};

}  // namespace pto::runtime
```

### 2.4 TaskSpawner

```cpp
namespace pto::runtime {

// Interface for dynamic task spawning (MoE support)
struct TaskSpawner {
    // Spawn a new task into the current execution context
    virtual void spawn(KernelFn kernel, TaskParams params, TaskResources resources) = 0;

    // Spawn with dependency
    virtual void spawn(KernelFn kernel, TaskParams params, TaskResources resources,
                       Event** wait_events, int32_t wait_count) = 0;

    // Spawn and get completion event
    virtual Event* spawn_with_signal(KernelFn kernel, TaskParams params,
                                     TaskResources resources) = 0;

    virtual ~TaskSpawner() = default;
};

}  // namespace pto::runtime
```

---

## 3. Core Functions

### 3.1 Task Creation

```cpp
namespace pto::runtime {

// Create a task with minimal parameters
Task create_task(KernelFn kernel, TaskParams params);

// Create a task with resources
Task create_task(KernelFn kernel, TaskParams params, TaskResources resources);

// Set task dependencies
void task_wait_for(Task& task, Event* event);
void task_wait_for(Task& task, Event** events, int32_t count);

// Set task signal
void task_signal(Task& task, Event* event);

// Enable task spawning
void task_enable_spawning(Task& task, TaskSpawner* spawner);

}  // namespace pto::runtime
```

### 3.2 TaskSet Creation

```cpp
namespace pto::runtime {

// Create taskset from array of tasks
TaskSet create_taskset(Task* tasks, int32_t count, IssuePolicy policy = IssuePolicy::STATIC);

// Create taskset with completion event
TaskSet create_taskset(Task* tasks, int32_t count, IssuePolicy policy, Event* completion);

}  // namespace pto::runtime
```

### 3.3 Stitching

```cpp
namespace pto::runtime {

// Stitch two tasksets together (for overlapped execution)
TaskSet stitch(const TaskSet& a, const TaskSet& b);

// Stitch multiple tasksets
TaskSet stitch(TaskSet* tasksets, int32_t count);

// Stitch with explicit policy override
TaskSet stitch(const TaskSet& a, const TaskSet& b, IssuePolicy policy);

}  // namespace pto::runtime
```

### 3.4 Dispatch

```cpp
namespace pto::runtime {

// Dispatch taskset to AICPU for execution
// Returns immediately (asynchronous)
void dispatch(TaskSet& taskset);

// Dispatch and wait for completion
void dispatch_sync(TaskSet& taskset);

// Dispatch multiple tasksets (for pipelining)
void dispatch_all(TaskSet* tasksets, int32_t count);

}  // namespace pto::runtime
```

### 3.5 Synchronization

```cpp
namespace pto::runtime {

// Wait for event to be signaled
void wait(Event* event);

// Wait for multiple events (all must be signaled)
void wait_all(Event** events, int32_t count);

// Wait for any event (returns index of first signaled)
int32_t wait_any(Event** events, int32_t count);

// Synchronize all pending work
void synchronize();

}  // namespace pto::runtime
```

---

## 4. Helper Types and Functions

### 4.1 Tiered Kernel Registration

```cpp
namespace pto::runtime {

// Register tiered kernels for automatic selection
template<int MaxTiers = 8>
struct TieredKernels {
    KernelFn kernels[MaxTiers];
    int32_t tier_bounds[MaxTiers];  // Upper bound for each tier
    int32_t tier_count;

    // Select kernel based on value (e.g., seq_len)
    KernelFn select(int32_t value) const {
        for (int i = 0; i < tier_count; i++) {
            if (value <= tier_bounds[i]) return kernels[i];
        }
        return kernels[tier_count - 1];  // Last tier for anything larger
    }

    int32_t select_tier(int32_t value) const {
        for (int i = 0; i < tier_count; i++) {
            if (value <= tier_bounds[i]) return i;
        }
        return tier_count - 1;
    }
};

}  // namespace pto::runtime
```

### 4.2 Parameter Builders

```cpp
namespace pto::runtime {

// Convenience builder for TaskParams
struct ParamBuilder {
    TaskParams params{};
    int index = 0;

    ParamBuilder& add(int32_t value) {
        params.values[index++] = value;
        return *this;
    }

    TaskParams build() { return params; }
};

// Usage: auto params = ParamBuilder().add(batch_idx).add(seq_len).add(head_idx).build();

}  // namespace pto::runtime
```

### 4.3 Resource Builders

```cpp
namespace pto::runtime {

// Convenience builder for TaskResources
struct ResourceBuilder {
    std::vector<void*> inputs;
    std::vector<void*> outputs;
    void* workspace = nullptr;

    ResourceBuilder& input(void* tensor) {
        inputs.push_back(tensor);
        return *this;
    }

    ResourceBuilder& output(void* tensor) {
        outputs.push_back(tensor);
        return *this;
    }

    ResourceBuilder& work(void* buffer) {
        workspace = buffer;
        return *this;
    }

    TaskResources build() {
        return {
            inputs.data(), static_cast<int32_t>(inputs.size()),
            outputs.data(), static_cast<int32_t>(outputs.size()),
            workspace
        };
    }
};

}  // namespace pto::runtime
```

---

## 5. Usage Examples

### 5.1 Basic: Uniform Batch Attention

```cpp
#include <pto/runtime/runtime.hpp>

using namespace pto::runtime;

void run_uniform_attention(
    void* query, void* kv_cache, void* output,
    int batch_size, int num_heads, int seq_len
) {
    // All tasks use same kernel (uniform seq_len)
    extern KernelFn attention_decode_kernel;

    std::vector<Task> tasks;
    for (int b = 0; b < batch_size; b++) {
        for (int h = 0; h < num_heads; h++) {
            Task t = create_task(
                attention_decode_kernel,
                ParamBuilder().add(b).add(h).add(seq_len).build(),
                ResourceBuilder().input(query).input(kv_cache).output(output).build()
            );
            tasks.push_back(t);
        }
    }

    // Static issuing (uniform workload)
    TaskSet ts = create_taskset(tasks.data(), tasks.size(), IssuePolicy::STATIC);
    dispatch_sync(ts);
}
```

### 5.2 Variable-Length: Tiered Attention

```cpp
void run_tiered_attention(
    void* query, void* kv_cache, void* output,
    int* seq_lens,  // Per-batch sequence lengths
    int batch_size, int num_heads
) {
    // Register tiered kernels
    TieredKernels<4> kernels = {
        .kernels = { attn_tier0, attn_tier1, attn_tier2, attn_tier3 },
        .tier_bounds = { 1024, 4096, 16384, 131072 },
        .tier_count = 4
    };

    std::vector<Task> tasks;
    for (int b = 0; b < batch_size; b++) {
        int seq_len = seq_lens[b];
        KernelFn kernel = kernels.select(seq_len);

        for (int h = 0; h < num_heads; h++) {
            Task t = create_task(
                kernel,
                ParamBuilder().add(b).add(h).add(seq_len).build(),
                ResourceBuilder().input(query).input(kv_cache).output(output).build()
            );
            tasks.push_back(t);
        }
    }

    // Dynamic issuing (variable workload)
    TaskSet ts = create_taskset(tasks.data(), tasks.size(), IssuePolicy::DYNAMIC);
    dispatch_sync(ts);
}
```

### 5.3 Dependencies: Pipeline Stages

```cpp
void run_pipeline(void* input, void* output, int batch_size) {
    extern KernelFn stage1_kernel, stage2_kernel, stage3_kernel;

    // Create events for stage synchronization
    std::vector<Event*> stage1_done(batch_size);
    std::vector<Event*> stage2_done(batch_size);
    for (int b = 0; b < batch_size; b++) {
        stage1_done[b] = create_event();
        stage2_done[b] = create_event();
    }

    std::vector<Task> all_tasks;

    // Stage 1 tasks (no dependencies)
    for (int b = 0; b < batch_size; b++) {
        Task t = create_task(stage1_kernel, ParamBuilder().add(b).build());
        task_signal(t, stage1_done[b]);
        all_tasks.push_back(t);
    }

    // Stage 2 tasks (wait for stage 1)
    for (int b = 0; b < batch_size; b++) {
        Task t = create_task(stage2_kernel, ParamBuilder().add(b).build());
        task_wait_for(t, stage1_done[b]);
        task_signal(t, stage2_done[b]);
        all_tasks.push_back(t);
    }

    // Stage 3 tasks (wait for stage 2)
    for (int b = 0; b < batch_size; b++) {
        Task t = create_task(stage3_kernel, ParamBuilder().add(b).build());
        task_wait_for(t, stage2_done[b]);
        all_tasks.push_back(t);
    }

    // Dispatch all - independent tasks can overlap across stages
    TaskSet ts = create_taskset(all_tasks.data(), all_tasks.size(), IssuePolicy::DYNAMIC);
    dispatch_sync(ts);

    // Cleanup
    for (int b = 0; b < batch_size; b++) {
        destroy_event(stage1_done[b]);
        destroy_event(stage2_done[b]);
    }
}
```

### 5.4 Stitching: Multiple Batches

```cpp
void run_stitched_batches(
    void** queries,    // queries[batch_group]
    void** kv_caches,
    void** outputs,
    int num_batch_groups,
    int batch_size_per_group,
    int num_heads
) {
    extern KernelFn attention_kernel;

    std::vector<TaskSet> tasksets(num_batch_groups);

    for (int g = 0; g < num_batch_groups; g++) {
        std::vector<Task> tasks;
        for (int b = 0; b < batch_size_per_group; b++) {
            for (int h = 0; h < num_heads; h++) {
                Task t = create_task(
                    attention_kernel,
                    ParamBuilder().add(b).add(h).build(),
                    ResourceBuilder()
                        .input(queries[g])
                        .input(kv_caches[g])
                        .output(outputs[g])
                        .build()
                );
                tasks.push_back(t);
            }
        }
        tasksets[g] = create_taskset(tasks.data(), tasks.size(), IssuePolicy::STATIC);
    }

    // Stitch all batch groups together
    TaskSet stitched = stitch(tasksets.data(), num_batch_groups);

    // Execute with overlap
    dispatch_sync(stitched);
}
```

### 5.5 Dynamic: MoE with Task Spawning

```cpp
// Global spawner (set by runtime)
extern TaskSpawner* global_spawner;

// Expert kernels
extern KernelFn expert_kernels[256];  // Up to 256 experts

// Router kernel - spawns expert tasks dynamically
void moe_router_kernel(void* params_ptr, void* inputs_ptr, void* outputs_ptr) {
    auto& params = *static_cast<TaskParams*>(params_ptr);
    int batch_idx = params[0];
    int num_experts = params[1];
    int top_k = params[2];

    // Compute routing (simplified)
    auto& input = *static_cast<GlobalTensor<half>*>(inputs_ptr);
    int selected[8];  // top_k selected experts
    float weights[8];
    compute_routing(input, batch_idx, num_experts, top_k, selected, weights);

    // Spawn expert tasks based on routing result
    for (int k = 0; k < top_k; k++) {
        int expert_id = selected[k];
        global_spawner->spawn(
            expert_kernels[expert_id],
            ParamBuilder().add(batch_idx).add(expert_id).add(*(int*)&weights[k]).build(),
            ResourceBuilder().input(inputs_ptr).output(outputs_ptr).build()
        );
    }
}

void run_moe_layer(void* input, void* output, int batch_size, int num_experts, int top_k) {
    std::vector<Task> router_tasks;
    for (int b = 0; b < batch_size; b++) {
        Task t = create_task(
            moe_router_kernel,
            ParamBuilder().add(b).add(num_experts).add(top_k).build(),
            ResourceBuilder().input(input).output(output).build()
        );
        task_enable_spawning(t, global_spawner);
        router_tasks.push_back(t);
    }

    // Completion event for all tasks (including spawned)
    Event* moe_done = create_event();
    TaskSet ts = create_taskset(router_tasks.data(), router_tasks.size(),
                                IssuePolicy::DYNAMIC, moe_done);

    dispatch(ts);
    wait(moe_done);
    destroy_event(moe_done);
}
```

---

## 6. Implementation Notes

### 6.1 AICPU Internal Structure

```
┌─────────────────────────────────────────────────────────────────┐
│ AICPU Runtime                                                    │
│                                                                 │
│ ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│ │ Event Table │  │ Task Queues │  │ Issue Controllers       │  │
│ │             │  │             │  │                         │  │
│ │ event_id -> │  │ pending[]   │  │ static_issuer           │  │
│ │ {signaled,  │  │ ready[]     │  │ dynamic_issuer          │  │
│ │  waiters[]} │  │ spawned[]   │  │ hybrid_issuer           │  │
│ └─────────────┘  └─────────────┘  └─────────────────────────┘  │
│                                                                 │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ Scheduler Thread                                             │ │
│ │ 1. Check event table for newly satisfied dependencies      │ │
│ │ 2. Move pending tasks to ready queue                        │ │
│ │ 3. Process spawned tasks                                    │ │
│ └─────────────────────────────────────────────────────────────┘ │
│                                                                 │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ Issuer Thread                                                │ │
│ │ 1. Pull tasks from ready queue                              │ │
│ │ 2. Issue to AICores based on policy                         │ │
│ │ 3. Register completion callbacks                            │ │
│ └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### 6.2 Issue Policy Implementation

**Static (Round-Robin)**:
```cpp
void static_issue(Task* tasks, int count, int num_cores) {
    for (int i = 0; i < count; i++) {
        int core_id = i % num_cores;
        issue_to_core(core_id, tasks[i]);
    }
}
```

**Dynamic (Work Stealing)**:
```cpp
// Shared queue
AtomicQueue<Task> work_queue;

void dynamic_issue(Task* tasks, int count) {
    // Enqueue all tasks
    for (int i = 0; i < count; i++) {
        work_queue.push(tasks[i]);
    }
    // AICores pull from queue atomically
}

// On AICore side:
void aicore_worker() {
    while (Task* t = work_queue.pop()) {
        execute(t);
    }
}
```

### 6.3 Compatibility with PTO-ISA

The extension integrates with existing PTO-ISA:

```cpp
// Existing PTO-ISA kernel
template<int TileM, int TileN>
__global__ void gemm_kernel(
    __gm__ half* A, __gm__ half* B, __gm__ half* C,
    int M, int N, int K
) {
    // Existing kernel code...
}

// Wrapping for runtime extension
void gemm_wrapper(void* params_ptr, void* inputs_ptr, void* outputs_ptr) {
    auto& params = *static_cast<TaskParams*>(params_ptr);
    auto inputs = static_cast<void**>(inputs_ptr);
    auto outputs = static_cast<void**>(outputs_ptr);

    int M = params[0], N = params[1], K = params[2];

    // Call existing kernel
    gemm_kernel<64, 64><<<grid, block>>>(
        static_cast<half*>(inputs[0]),
        static_cast<half*>(inputs[1]),
        static_cast<half*>(outputs[0]),
        M, N, K
    );
}
```

---

## 7. Error Handling

```cpp
namespace pto::runtime {

enum class ErrorCode : int32_t {
    SUCCESS = 0,
    INVALID_TASK = -1,
    INVALID_EVENT = -2,
    INVALID_POLICY = -3,
    QUEUE_FULL = -4,
    TIMEOUT = -5,
    SPAWN_FAILED = -6
};

// Get last error
ErrorCode get_last_error();

// Error to string
const char* error_string(ErrorCode code);

// Set error callback
using ErrorCallback = void(*)(ErrorCode, const char* message);
void set_error_callback(ErrorCallback cb);

}  // namespace pto::runtime
```

---

## 8. Version History

| Version | Date | Changes |
|---------|------|---------|
| 5.0 | 2025-01-16 | Complete redesign based on design questions |
| 4.x | 2024-01-14 | Previous tiered dispatcher design (archived) |

---

*Specification Version: 5.0*
*Last Updated: 2025-01-16*
