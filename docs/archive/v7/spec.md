# PTO Workload-Schedule Programming (PTO-WSP): API Specification (v7)

## 1. Overview

This specification defines the **Workload-Schedule** API for PTO-ISA runtime extension, using a **CSP-based execution model** with **JIT compilation** to AICPU.

### 1.1 Design Goals

1. **Workload-Schedule separation**: Decouple what to compute from how to execute
2. **CSP model**: Processes communicate via channels, no global sync
3. **JIT compilation**: Compile to AICPU binary for efficient execution
4. **Expressiveness**: Support dynamic workloads (MoE, variable lengths)
5. **Human-in-the-loop**: Programmers tune schedules, not black boxes

### 1.2 Namespace

All APIs are under `pto::wsp` namespace:
```cpp
namespace pto::wsp {
    // Workload primitives
    // Schedule primitives
    // CSP primitives
    // Runtime types
}
```

## 2. Core Types

### 2.1 Task

A **Task** is a single kernel invocation with parameters.

```cpp
struct Task {
    KernelId kernel;          // Which kernel to invoke
    ParamBlock params;        // Kernel parameters
    ResourceSet resources;    // Memory resources (tensors)
    TaskId id;                // Unique identifier
    uint32_t priority;        // Scheduling priority
};

// ParamBlock: Type-erased parameter storage
struct ParamBlock {
    void* data;
    size_t size;

    template<typename T>
    T& get(size_t offset);
};

// ResourceSet: Tensors used by this task
struct ResourceSet {
    Tensor* inputs[MAX_INPUTS];
    Tensor* outputs[MAX_OUTPUTS];
    uint32_t num_inputs;
    uint32_t num_outputs;
};
```

### 2.2 Tensor

**Tensor** is a first-class citizen representing memory resources.

```cpp
struct Tensor {
    void* data;               // Pointer to data
    Shape shape;              // Dimensions
    DType dtype;              // Data type
    Layout layout;            // Memory layout
    Location location;        // Where the tensor resides
};

enum class Location : uint8_t {
    GLOBAL,                   // Global memory
    UB,                       // Unified Buffer (AICore local)
    L1,                       // L1 cache
};
```

### 2.3 Kernel Registration

Kernels are registered at compile time:

```cpp
// Register a kernel
REGISTER_KERNEL(attention_kernel, AttentionParams, AttentionResources);

// Kernel parameter structure
struct AttentionParams {
    int batch_idx;
    int head_idx;
    int seq_len;
};

// Kernel resource structure
struct AttentionResources {
    Tensor* q;
    Tensor* k;
    Tensor* v;
    Tensor* output;
};
```

## 3. Workload Specification

### 3.1 Workload Type

A **Workload** is a specification of tasks to generate.

```cpp
class Workload {
public:
    // Iteration primitives
    static Workload for_each(Range range, std::function<Workload(int)> body);
    static Workload parallel_for(Range range, std::function<Workload(int)> body);
    static Workload reduce(Range range, Value init, std::function<Value(Value, int)> body);

    // Control flow primitives
    static Workload cond(Expr pred, Workload then_branch, Workload else_branch);
    static Workload select(Indices indices, std::function<Workload(int)> body);
    static Workload spawn(std::function<Workload()> generator);

    // Task primitives
    static Workload task(KernelId kernel, ParamBlock params, ResourceSet resources);
    static Workload fused(std::initializer_list<Workload> tasks);
    static Workload empty();

    // Composition
    Workload then(Workload next);            // Sequential composition
    Workload parallel(Workload other);       // Parallel composition
    Workload depends_on(Channel<void> dep);  // Dependency

    // Schedule binding
    Schedule schedule(/* schedule primitives */);
};
```

### 3.2 Range Type

**Range** specifies iteration bounds.

```cpp
class Range {
public:
    // Static range (known at compile time)
    static Range static_range(int start, int end, int step = 1);

    // Dynamic range (known at runtime)
    static Range dynamic_range(Expr start, Expr end, Expr step = 1);

    // Named range
    static Range named(const char* name, Expr size);

    int start() const;
    int end() const;
    int step() const;
    bool is_static() const;
};

// Convenience
Range range(int end);                    // [0, end)
Range range(int start, int end);         // [start, end)
Range range(Expr end);                   // Dynamic [0, end)
```

### 3.3 Expression Type

**Expr** represents runtime values.

```cpp
class Expr {
public:
    // Literals
    static Expr constant(int64_t value);
    static Expr constant(double value);

    // Variables (bound at runtime)
    static Expr var(const char* name);

    // Array access
    static Expr load(const char* array, Expr index);

    // Operators
    Expr operator+(Expr other);
    Expr operator-(Expr other);
    Expr operator*(Expr other);
    Expr operator/(Expr other);
    Expr operator<(Expr other);
    Expr operator<=(Expr other);
    Expr operator>(Expr other);
    Expr operator>=(Expr other);
    Expr operator==(Expr other);
    Expr operator!=(Expr other);

    // Functions
    static Expr min(Expr a, Expr b);
    static Expr max(Expr a, Expr b);
    static Expr select(Expr cond, Expr then_val, Expr else_val);
};
```

### 3.4 Workload Examples

**Example 1: Simple Attention**
```cpp
auto seq_lens = Expr::var("seq_lens");
auto batch_size = Expr::var("batch_size");
auto num_heads = Expr::var("num_heads");

Workload attn = Workload::parallel_for(range(batch_size), [&](int b) {
    return Workload::parallel_for(range(num_heads), [&](int h) {
        return Workload::task(
            attention_kernel,
            ParamBlock{b, h, seq_lens[b]},
            ResourceSet{q[b][h], k[b], v[b], out[b][h]}
        );
    });
});
```

**Example 2: Tiered Kernels**
```cpp
Workload tiered_attn = Workload::parallel_for(range(batch_size), [&](int b) {
    auto len = Expr::load("seq_lens", b);
    return Workload::cond(
        len <= 2048,
        Workload::task(attn_2k, {b, len}, resources),
        Workload::cond(
            len <= 8192,
            Workload::task(attn_8k, {b, len}, resources),
            Workload::task(attn_32k, {b, len}, resources)
        )
    );
});
```

**Example 3: MoE with Dynamic Routing**
```cpp
Workload moe = Workload::parallel_for(range(batch_size), [&](int b) {
    return Workload::parallel_for(range(num_tokens), [&](int t) {
        // Router returns indices of selected experts
        auto selected = Expr::load("routing_result", {b, t});
        return Workload::select(selected, [&](int expert_idx) {
            return Workload::task(
                expert_kernels[expert_idx],
                {b, t, expert_idx},
                expert_resources[expert_idx]
            );
        });
    });
});
```

## 4. Schedule Specification

### 4.1 Schedule Type

A **Schedule** specifies how to execute a workload.

```cpp
class Schedule {
public:
    // Dispatch primitives (Host → AICPU)
    Schedule dispatch(DispatchPolicy policy);
    Schedule colocate(Expr axis);
    Schedule replicate(int n);

    // Issue primitives (AICPU → AICore)
    Schedule issue(IssueOrder order);
    Schedule bind(Expr axis, CoreSet cores);
    Schedule steal();

    // Pipeline primitives
    Schedule pipeline(int depth);
    Schedule double_buffer();
    Schedule prefetch(int n);

    // Stitch primitives
    static Schedule stitch(Schedule a, Schedule b);
    static Schedule fuse(Schedule a, Schedule b);
    static Schedule interleave(Schedule a, Schedule b);

    // Compile to AICPU binary
    CompiledProgram compile();
};
```

### 4.2 Dispatch Policies

```cpp
class DispatchPolicy {
public:
    // Static policies
    static DispatchPolicy round_robin(int num_aicpu);
    static DispatchPolicy hash(Expr key);
    static DispatchPolicy affinity(Expr axis);
    static DispatchPolicy range_partition(Range range, int num_aicpu);

    // Dynamic policies
    static DispatchPolicy dynamic();
    static DispatchPolicy load_balanced();

    // Custom policy
    static DispatchPolicy custom(std::function<int(Task&)> selector);
};
```

### 4.3 Issue Orders

```cpp
class IssueOrder {
public:
    // Fixed orders
    static IssueOrder fifo();
    static IssueOrder lifo();

    // Affinity-based
    static IssueOrder batch_affinity();
    static IssueOrder core_affinity(Expr axis);

    // Priority-based
    static IssueOrder priority(std::function<int(Task&)> priority_fn);

    // Custom
    static IssueOrder custom(std::function<Task(TaskQueue&)> selector);
};
```

### 4.4 Schedule Examples

**Example 1: Static Round-Robin**
```cpp
Schedule sched = attn
    .schedule()
    .dispatch(DispatchPolicy::round_robin(4))
    .issue(IssueOrder::fifo());
```

**Example 2: Work-Stealing**
```cpp
Schedule sched = attn
    .schedule()
    .dispatch(DispatchPolicy::affinity(Expr::var("batch")))
    .issue(IssueOrder::batch_affinity())
    .steal()
    .pipeline(2);
```

**Example 3: Stitched Batches**
```cpp
Schedule sched0 = batch0.schedule()
    .dispatch(DispatchPolicy::hash(Expr::var("batch_id")));

Schedule sched1 = batch1.schedule()
    .dispatch(DispatchPolicy::hash(Expr::var("batch_id")));

Schedule stitched = Schedule::stitch(sched0, sched1)
    .issue(IssueOrder::batch_affinity())
    .interleave(compute_tasks, memory_tasks)
    .pipeline(3);
```

**Example 4: MoE with Load Balancing**
```cpp
Schedule moe_sched = moe
    .schedule()
    .dispatch(DispatchPolicy::dynamic())
    .issue(IssueOrder::priority([](Task& t) {
        return -expert_load[t.params.get<int>(2)];  // Prioritize underloaded
    }))
    .colocate(Expr::var("expert_idx"));
```

## 5. CSP Primitives

### 5.1 Process Type

A **Process** is a concurrent unit of execution with local time.

```cpp
class Process {
public:
    // Create process from function
    template<typename F>
    static Process create(F&& body);

    // Process control
    void spawn();                    // Start execution
    void join();                     // Wait for completion
    bool is_alive() const;

    // Local time
    Time tick() const;               // Current local time
    void advance(Duration d);        // Advance time by d
    void wait_until(Time t);         // Wait until time t
    void yield();                    // Yield to other processes
};

// Time types
using Time = uint64_t;               // Nanoseconds
using Duration = uint64_t;           // Nanoseconds
```

### 5.2 Channel Type

A **Channel** is a typed communication link between processes.

```cpp
template<typename T>
class Channel {
public:
    // Create channel
    static Channel create(size_t capacity = 0);  // 0 = unbounded

    // Sender operations
    void send(T value);                          // Block if full
    bool try_send(T value);                      // Non-blocking
    bool try_send_for(T value, Duration timeout);

    // Receiver operations
    T recv();                                    // Block if empty
    std::optional<T> try_recv();                 // Non-blocking
    std::optional<T> try_recv_for(Duration timeout);

    // Peek (non-consuming)
    std::optional<T> peek();
    std::optional<T> peek_next();                // Block until available

    // Channel state
    bool empty() const;
    bool full() const;
    size_t size() const;
    size_t capacity() const;

    // Close channel
    void close();
    bool is_closed() const;
};
```

### 5.3 Select Statement

**Select** waits on multiple channels.

```cpp
// Select builder
class SelectBuilder {
public:
    template<typename T>
    SelectBuilder& recv(Channel<T>& ch, std::function<void(T)> handler);

    template<typename T>
    SelectBuilder& send(Channel<T>& ch, T value, std::function<void()> handler);

    SelectBuilder& timeout(Duration d, std::function<void()> handler);

    SelectBuilder& default_case(std::function<void()> handler);

    void wait();                     // Block until one case fires
};

// Convenience function
SelectBuilder select();

// Usage example
select()
    .recv(task_chan, [&](Task t) { /* handle task */ })
    .recv(complete_chan, [&](TaskId id) { /* handle completion */ })
    .timeout(100_ns, [&]() { /* timeout */ })
    .wait();
```

### 5.4 CSP Process Examples

**Example 1: Task Generator**
```cpp
Process generator = Process::create([&](Channel<Task>& out) {
    for (int b = 0; b < batch_size; b++) {
        for (int h = 0; h < num_heads; h++) {
            Task t = make_task(attn_kernel, {b, h, seq_lens[b]});
            out.send(t);
            time.advance(task_gen_latency);
        }
    }
    out.close();
});
```

**Example 2: Work-Stealing Scheduler**
```cpp
Process scheduler = Process::create([&](
    Channel<Task>& in,
    std::array<Channel<Task>, NUM_CORES>& core_queues
) {
    std::array<TaskQueue, NUM_CORES> local_queues;

    while (true) {
        select()
            .recv(in, [&](Task t) {
                int core = t.params.get<int>(0) % NUM_CORES;  // Batch affinity
                local_queues[core].push(t);
            })
            .recv(work_requests, [&](int core) {
                if (!local_queues[core].empty()) {
                    core_queues[core].send(local_queues[core].pop());
                } else {
                    // Work stealing
                    for (int v = 0; v < NUM_CORES; v++) {
                        if (!local_queues[v].empty()) {
                            core_queues[core].send(local_queues[v].steal());
                            break;
                        }
                    }
                }
            })
            .timeout(100_ns, [&]() {
                time.advance(100_ns);
            })
            .wait();

        if (in.is_closed() && all_empty(local_queues)) break;
    }
});
```

**Example 3: Issuer with Pipelining**
```cpp
Process issuer = Process::create([&](
    Channel<Task>& ready,
    std::array<AICore*, NUM_CORES> cores
) {
    constexpr int PIPELINE_DEPTH = 2;
    std::array<Task, PIPELINE_DEPTH> in_flight;
    int num_in_flight = 0;

    while (true) {
        // Check completions
        for (int i = 0; i < num_in_flight; i++) {
            if (cores[in_flight[i].core]->is_complete()) {
                complete_chan.send(in_flight[i].id);
                std::swap(in_flight[i], in_flight[--num_in_flight]);
            }
        }

        // Issue new tasks
        while (num_in_flight < PIPELINE_DEPTH) {
            auto task_opt = ready.try_recv();
            if (!task_opt) break;

            Task t = *task_opt;
            int core = find_free_core(cores);
            if (core < 0) break;

            cores[core]->issue(t);
            in_flight[num_in_flight++] = t;
            t.core = core;
        }

        time.advance(issue_check_interval);
    }
});
```

## 6. JIT Compilation Interface

### 6.1 CompiledProgram

```cpp
class CompiledProgram {
public:
    // Get compiled binary
    const uint8_t* binary() const;
    size_t binary_size() const;

    // Get metadata
    size_t state_size() const;           // Required state memory
    size_t stack_size() const;           // Required stack
    int num_processes() const;

    // Bind runtime values
    void bind(const char* name, int64_t value);
    void bind(const char* name, void* ptr);
    void bind(const char* name, Tensor* tensor);

    // Execute
    void execute(AICPUContext* ctx);
};
```

### 6.2 Compilation API

```cpp
class Compiler {
public:
    // Compile workload + schedule
    static CompiledProgram compile(Schedule schedule);

    // Compilation options
    struct Options {
        OptLevel opt_level = OptLevel::O2;
        bool enable_profiling = false;
        bool enable_debug = false;
        int max_processes = 16;
        int max_channels = 64;
    };

    static CompiledProgram compile(Schedule schedule, Options opts);

    // Incremental compilation
    static CompiledProgram recompile(
        CompiledProgram& base,
        Schedule new_schedule
    );
};

enum class OptLevel {
    O0,  // No optimization
    O1,  // Basic optimization
    O2,  // Standard optimization
    O3,  // Aggressive optimization
};
```

### 6.3 AICPU Context

```cpp
class AICPUContext {
public:
    // Get AICPU info
    int aicpu_id() const;
    int num_aicpu() const;

    // Get AICore info
    int num_cores() const;
    AICore* core(int idx);

    // Memory allocation
    void* alloc(size_t size);
    void free(void* ptr);

    // Runtime state
    void* state();
    size_t state_size() const;

    // Execution control
    void run(CompiledProgram& prog);
    void stop();
    bool is_running() const;
};
```

## 7. Runtime Library

### 7.1 Built-in Functions

These functions are available in compiled programs:

```cpp
// Task operations
Task make_task(KernelId kernel, ParamBlock params, ResourceSet resources);
void issue_task(Task t, int core_id);
bool is_task_complete(TaskId id);

// Core operations
int get_free_core();
bool is_core_free(int core_id);
void wait_core(int core_id);

// Memory operations
void* alloc_ub(size_t size);
void free_ub(void* ptr);
void dma_load(void* dst, void* src, size_t size);
void dma_store(void* dst, void* src, size_t size);

// Synchronization
void barrier(int count);
void fence();

// Timing
Time now();
void sleep(Duration d);
```

### 7.2 Built-in Schedulers

Pre-built scheduler patterns:

```cpp
namespace schedulers {

// Round-robin static scheduler
CompiledProgram round_robin(Workload w, int num_aicpu);

// Work-stealing scheduler
CompiledProgram work_stealing(Workload w, int num_aicpu);

// Batch-affinity scheduler
CompiledProgram batch_affinity(Workload w, int num_aicpu);

// Pipeline scheduler
CompiledProgram pipeline(Workload w, int depth);

// MoE-optimized scheduler
CompiledProgram moe_scheduler(Workload w, int num_experts);

}
```

## 8. Complete Examples

### 8.1 Attention with Work-Stealing

```cpp
// Define tensors
Tensor* q = ...;
Tensor* k = ...;
Tensor* v = ...;
Tensor* out = ...;
int* seq_lens = ...;

// Define workload
Workload attn = Workload::parallel_for(range(batch_size), [&](int b) {
    return Workload::parallel_for(range(num_heads), [&](int h) {
        return Workload::task(
            attention_kernel,
            ParamBlock{b, h},
            ResourceSet{q, k, v, out}
        );
    });
});

// Define schedule
Schedule sched = attn
    .schedule()
    .dispatch(DispatchPolicy::affinity(Expr::var("batch")))
    .issue(IssueOrder::batch_affinity())
    .steal()
    .pipeline(2);

// Compile
CompiledProgram prog = Compiler::compile(sched);

// Bind runtime values
prog.bind("batch_size", batch_size);
prog.bind("num_heads", num_heads);
prog.bind("seq_lens", seq_lens);
prog.bind("q", q);
prog.bind("k", k);
prog.bind("v", v);
prog.bind("out", out);

// Execute
AICPUContext ctx;
prog.execute(&ctx);
```

### 8.2 MoE with Dynamic Routing

```cpp
// Define router workload (executed first)
Workload router = Workload::parallel_for(range(batch_size), [&](int b) {
    return Workload::parallel_for(range(num_tokens), [&](int t) {
        return Workload::task(
            router_kernel,
            ParamBlock{b, t},
            ResourceSet{input, routing_result}
        );
    });
});

// Define expert workload (uses routing result)
Workload experts = Workload::parallel_for(range(batch_size), [&](int b) {
    return Workload::parallel_for(range(num_tokens), [&](int t) {
        auto selected = Expr::load("routing_result", {b, t});
        return Workload::select(selected, [&](int expert_idx) {
            return Workload::task(
                expert_kernels[expert_idx],
                ParamBlock{b, t, expert_idx},
                expert_resources[expert_idx]
            );
        });
    });
});

// Compose workloads
Workload moe = router.then(experts);

// Schedule with dynamic load balancing
Schedule sched = moe
    .schedule()
    .dispatch(DispatchPolicy::dynamic())
    .issue(IssueOrder::priority([](Task& t) {
        int expert = t.params.get<int>(2);
        return -expert_load[expert];  // Prioritize underloaded experts
    }))
    .colocate(Expr::var("expert_idx"))
    .pipeline(3);

// Compile and execute
CompiledProgram prog = Compiler::compile(sched);
prog.bind("num_experts", num_experts);
prog.bind("expert_load", expert_load);
// ... bind other values ...
prog.execute(&ctx);
```

### 8.3 Stitched Multi-Batch Pipeline

```cpp
// Define workloads for two batches
Workload batch0 = /* attention for batch 0 */;
Workload batch1 = /* attention for batch 1 */;

// Schedule each batch
Schedule sched0 = batch0
    .schedule()
    .dispatch(DispatchPolicy::hash(Expr::constant(0)));

Schedule sched1 = batch1
    .schedule()
    .dispatch(DispatchPolicy::hash(Expr::constant(1)));

// Stitch and interleave
Schedule stitched = Schedule::stitch(sched0, sched1)
    .issue(IssueOrder::batch_affinity())
    .interleave(compute_pool, memory_pool)
    .pipeline(4);

// Compile with stitching optimization
Compiler::Options opts;
opts.opt_level = OptLevel::O3;
CompiledProgram prog = Compiler::compile(stitched, opts);

// Execute
prog.execute(&ctx);
```

### 8.4 Custom CSP Scheduler

```cpp
// Define custom scheduler using CSP primitives directly
Process custom_scheduler = Process::create([&](
    Channel<Task>& in,
    Channel<Task>& out
) {
    // Custom state
    PriorityQueue<Task> pq;
    int pending = 0;

    while (true) {
        select()
            .recv(in, [&](Task t) {
                // Custom priority based on seq_len
                t.priority = 1000 - t.params.get<int>(2);
                pq.push(t);
                pending++;
            })
            .recv(complete, [&](TaskId id) {
                pending--;
            })
            .timeout(10_ns, [&]() {
                // Issue highest priority tasks
                while (!pq.empty() && pending < max_in_flight) {
                    Task t = pq.pop();
                    out.send(t);
                }
            })
            .wait();

        if (in.is_closed() && pq.empty() && pending == 0) break;
    }
});

// Wrap in schedule
Schedule sched = workload
    .schedule()
    .with_custom_scheduler(custom_scheduler);
```

## 9. Error Handling

### 9.1 Compilation Errors

```cpp
enum class CompileError {
    INVALID_WORKLOAD,         // Workload specification error
    INVALID_SCHEDULE,         // Schedule specification error
    TYPE_MISMATCH,            // Type error in expressions
    UNBOUND_VARIABLE,         // Undefined variable
    RESOURCE_OVERFLOW,        // Too many resources
    UNSUPPORTED_FEATURE,      // Feature not supported
};

class CompileException : public std::exception {
public:
    CompileError error() const;
    const char* message() const;
    SourceLocation location() const;
};
```

### 9.2 Runtime Errors

```cpp
enum class RuntimeError {
    DEADLOCK,                 // Deadlock detected
    CHANNEL_CLOSED,           // Send/recv on closed channel
    OUT_OF_MEMORY,            // Memory allocation failed
    CORE_UNAVAILABLE,         // No cores available
    TASK_FAILED,              // Task execution failed
};

class RuntimeException : public std::exception {
public:
    RuntimeError error() const;
    const char* message() const;
    TaskId failed_task() const;
};
```

## 10. Performance Considerations

### 10.1 Compilation Overhead

| Component | Typical Time |
|-----------|--------------|
| Workload parse | < 1ms |
| Schedule optimization | 1-10ms |
| AICPU codegen | 5-20ms |
| Total | < 50ms |

### 10.2 Runtime Overhead

| Operation | Overhead |
|-----------|----------|
| Channel send/recv | ~10ns |
| Task generation | ~50ns |
| Task issue | ~0 (hardware) |
| Process switch | ~100ns |

### 10.3 Memory Usage

| Component | Size |
|-----------|------|
| Compiled program | 1-100KB |
| Process state | 1-4KB per process |
| Channel buffer | Capacity × sizeof(T) |

## 11. Appendix: Grammar

### 11.1 Workload Grammar (EBNF)

```ebnf
workload     = for_each | parallel_for | reduce | cond | select | spawn | task | fused | empty | compose
for_each     = "for_each" "(" range "," lambda ")"
parallel_for = "parallel_for" "(" range "," lambda ")"
reduce       = "reduce" "(" range "," expr "," lambda ")"
cond         = "cond" "(" expr "," workload "," workload ")"
select       = "select" "(" expr "," lambda ")"
spawn        = "spawn" "(" lambda ")"
task         = "task" "(" kernel_id "," param_block "," resource_set ")"
fused        = "fused" "(" workload { "," workload } ")"
empty        = "empty" "(" ")"
compose      = workload ".then" "(" workload ")" | workload ".parallel" "(" workload ")"

range        = "range" "(" expr ["," expr ["," expr]] ")"
lambda       = "[" captures "]" "(" params ")" "{" body "}"
expr         = literal | variable | binary_op | unary_op | load | call
```

### 11.2 Schedule Grammar (EBNF)

```ebnf
schedule     = workload ".schedule" "(" ")" { primitive }
primitive    = dispatch | issue | pipeline | stitch | interleave | ...
dispatch     = ".dispatch" "(" policy ")"
issue        = ".issue" "(" order ")"
pipeline     = ".pipeline" "(" int ")"
stitch       = "Schedule::stitch" "(" schedule "," schedule ")"
interleave   = ".interleave" "(" schedule "," schedule ")"

policy       = "round_robin" "(" int ")" | "hash" "(" expr ")" | "affinity" "(" expr ")" | "dynamic" "(" ")" | ...
order        = "fifo" "(" ")" | "lifo" "(" ")" | "batch_affinity" "(" ")" | "priority" "(" lambda ")" | ...
```

---
*Version: 7.0*
*Last Updated: 2025-01-17*
