# PTO Workload-Schedule Programming (PTO-WSP): API Specification (v8)

## 1. Overview

This specification defines the **typed workload expression** API for PTO-ISA runtime extension, supporting **two parallelism modes** (data-parallel and pipeline-parallel via CSP) with **JIT compilation** to AICPU.

### 1.1 Design Goals

1. **Typed workload expressions**: Workloads are types with axes, tasks, and dependencies
2. **Two parallelism modes**: Data-parallel (`parallel_for`) and pipeline-parallel (CSP)
3. **Declarative primitives**: JIT-friendly, no imperative loops
4. **Structural dependencies**: Inferred from composition, not declared separately
5. **Human-in-the-loop**: Programmers control dispatch and issue via schedules

### 1.2 Namespace

All APIs are under `pto::wsp` namespace:

```cpp
namespace pto::wsp {
    // Type system
    // Workload primitives (data-parallel)
    // CSP primitives (pipeline-parallel)
    // Schedule primitives
    // Stream and event operations
    // Compilation API
}
```

### 1.3 Scope

**What we control:**
- Task generation from workload expressions
- Dispatch: Task → AICPU assignment
- Issue: Task → Stream → AICore ordering
- Timing: When tasks are issued

**What we don't control:**
- Kernel internals (fusion is PTO-ISA level, not runtime)

---

## 2. Type System

### 2.1 Axis Types

Axes describe iteration spaces with density and size attributes.

```cpp
namespace pto::wsp {

// Dense axis with static size
template<int64_t N>
struct Dense {
    static constexpr int64_t size = N;
    using Index = int64_t;
};

// Dense axis with dynamic size
struct DenseDyn {
    int64_t size;
    using Index = int64_t;

    explicit DenseDyn(int64_t n) : size(n) {}
    explicit DenseDyn(int64_t* ptr) : size(*ptr) {}
};

// Ragged axis: variable size per outer element
struct Ragged {
    int64_t outer_size;     // Number of outer elements
    int64_t* lengths;       // Length of each outer element
    using Index = int64_t;

    Ragged(int64_t n, int64_t* lens) : outer_size(n), lengths(lens) {}

    int64_t length(int64_t outer_idx) const { return lengths[outer_idx]; }
    int64_t total() const;  // Sum of all lengths
};

// Sparse axis: CSR format
struct Sparse {
    int64_t outer_size;     // Number of rows
    int64_t* indptr;        // Row pointers [outer_size + 1]
    int64_t* indices;       // Column indices [nnz]
    using Index = int64_t;

    Sparse(int64_t n, int64_t* ptr, int64_t* idx)
        : outer_size(n), indptr(ptr), indices(idx) {}

    int64_t nnz() const { return indptr[outer_size]; }
    int64_t row_nnz(int64_t row) const { return indptr[row+1] - indptr[row]; }
};

// Note: Range has been removed (redundant with DenseDyn)
// Use DenseDyn(n) for [0, n) iteration

}  // namespace pto::wsp
```

### 2.2 Task Type

A **Task** is a kernel invocation with typed parameters and resources.

```cpp
namespace pto::wsp {

// Forward declarations
struct Tensor;
struct KernelId;

// Task: single kernel invocation
struct Task {
    KernelId kernel;          // Which kernel to invoke
    ParamBlock params;        // Kernel parameters (type-erased)
    ResourceSet resources;    // Memory resources
    TaskId id;                // Unique identifier (assigned at enumeration)
    uint32_t priority;        // Optional priority hint

    // Access parameter by type
    template<typename T>
    T& param(size_t offset = 0);

    // Access axis value (for dispatch/issue policies)
    int64_t get(auto axis) const;
};

// ParamBlock: type-erased parameter storage
struct ParamBlock {
    alignas(16) uint8_t data[MAX_PARAM_SIZE];
    size_t size;

    template<typename T>
    T& get(size_t offset = 0) {
        return *reinterpret_cast<T*>(data + offset);
    }
};

// ResourceSet: tensors used by task
struct ResourceSet {
    static constexpr size_t MAX_TENSORS = 16;

    Tensor* tensors[MAX_TENSORS];
    uint8_t num_inputs;
    uint8_t num_outputs;

    // First num_inputs are inputs, rest are outputs
    Tensor* input(size_t i) const { return tensors[i]; }
    Tensor* output(size_t i) const { return tensors[num_inputs + i]; }
};

// TaskId: unique task identifier
struct TaskId {
    uint64_t value;

    bool operator==(const TaskId&) const = default;
    auto operator<=>(const TaskId&) const = default;
};

}  // namespace pto::wsp
```

### 2.3 Tensor Type

```cpp
namespace pto::wsp {

// Shape: up to 8 dimensions
struct Shape {
    static constexpr size_t MAX_DIMS = 8;
    int64_t dims[MAX_DIMS];
    uint8_t ndim;

    int64_t operator[](size_t i) const { return dims[i]; }
    int64_t numel() const;
};

// Data types
enum class DType : uint8_t {
    F16, BF16, F32, F64,
    I8, I16, I32, I64,
    U8, U16, U32, U64,
};

// Memory location (PascalCase to match implementation)
enum class Location : uint8_t {
    Global,     // Global memory (HBM)
    L2,         // L2 cache
    UB,         // Unified Buffer (AICore local)
    L1,         // L1 buffer
};

// Tensor: first-class memory resource
struct Tensor {
    void* data;
    Shape shape;
    DType dtype;
    Location location;

    size_t nbytes() const;

    // Indexing (returns sub-tensor view)
    Tensor operator[](int64_t idx) const;
    Tensor slice(int64_t start, int64_t end) const;
};

}  // namespace pto::wsp
```

### 2.4 Dependency Types

Dependencies are encoded in the workload type.

```cpp
namespace pto::wsp {

// Dependency kinds
struct Independent {};      // All tasks can run in parallel
struct Sequential {};       // Tasks run in order (i depends on i-1)
struct ChannelDep {};       // CSP channel-based dependency
struct Combined {};         // Composed workloads (schedule determines order)
struct None {};             // Single task with no dependency structure

// Note: DAG and Reduction have been removed
// - DAG: Use structural dependencies or ChannelDep instead
// - Reduction: Not implemented; use sequential() for ordered composition

}  // namespace pto::wsp
```

### 2.5 Workload Type

A **Workload** is a typed expression describing task generation.

```cpp
namespace pto::wsp {

// Workload: typed task generation expression
template<typename Axes, typename TaskT, typename Deps>
class Workload {
public:
    using axes_type = Axes;
    using task_type = TaskT;
    using deps_type = Deps;

    // Enumerate all tasks
    std::vector<Task> enumerate() const;

    // Get dependency type
    Deps dependencies() const;

    // Bind to schedule
    auto schedule() const -> Schedule<Workload>;

    // Record event after workload completes
    Event record() const;
};

// Type aliases for common workloads
template<typename A, typename T>
using IndependentWorkload = Workload<A, T, Independent>;

template<typename A, typename T>
using SequentialWorkload = Workload<A, T, Sequential>;

}  // namespace pto::wsp
```

---

## 3. Data-Parallel Primitives

### 3.1 parallel_for

Creates independent tasks over an axis.

```cpp
namespace pto::wsp {

// parallel_for: independent iteration
// Type: (Axis A, (Index → Workload W)) → Workload[A × W.Axes, W.Task, Independent]
template<typename Axis, typename Body>
auto parallel_for(Axis axis, Body body)
    -> Workload</* A × inner axes */, /* task type */, Independent>;

// Example:
auto w = parallel_for(Dense<4>{}, [](int64_t b) {
    return parallel_for(Dense<8>{}, [b](int64_t h) {
        return task(attn_kernel, {b, h}, {Q[b][h], K[b], V[b], O[b][h]});
    });
});
// Type: Workload<Dense<4> × Dense<8>, AttnTask, Independent>

}  // namespace pto::wsp
```

### 3.2 for_each

Creates sequential tasks over an axis.

```cpp
namespace pto::wsp {

// for_each: sequential iteration
// Type: (Axis A, (Index → Workload W)) → Workload[A × W.Axes, W.Task, Sequential]
template<typename Axis, typename Body>
auto for_each(Axis axis, Body body)
    -> Workload</* A × inner axes */, /* task type */, Sequential>;

// Example:
auto scan = for_each(DenseDyn(seq_len), [](int64_t i) {
    return task(scan_kernel, {i}, {in[i], out[i]});
});
// Type: Workload<DenseDyn, ScanTask, Sequential>
// Task[i] depends on Task[i-1]

}  // namespace pto::wsp
```

### 3.3 select

Creates tasks over sparse indices (MoE routing).

```cpp
namespace pto::wsp {

// select: sparse iteration
// Type: (Sparse S, (Index → Workload W)) → Workload[S × W.Axes, W.Task, Independent]
template<typename Body>
auto select(Sparse indices, Body body)
    -> Workload</* Sparse × inner axes */, /* task type */, Independent>;

// Example:
Sparse routing{batch_size, indptr, expert_indices};
auto moe = parallel_for(DenseDyn(batch_size), [&](int64_t b) {
    return select(routing[b], [&](int64_t e) {
        return task(expert_kernels[e], {b, e}, {tokens[b], weights[e], out[b][e]});
    });
});
// Type: Workload<DenseDyn × Sparse, ExpertTask, Independent>

}  // namespace pto::wsp
```

### 3.4 cond

Conditional workload selection.

```cpp
namespace pto::wsp {

// cond: conditional selection
// Type: (bool, Workload T, Workload E) → Workload[T ∪ E]
template<typename Then, typename Else>
auto cond(bool predicate, Then then_w, Else else_w)
    -> Workload</* union axes */, /* union task */, /* union deps */>;

// Runtime condition version
template<typename Pred, typename Then, typename Else>
auto cond(Pred pred, Then then_w, Else else_w);

// Example: tiered kernels
auto tiered = parallel_for(DenseDyn(batch), [&](int64_t b) {
    int len = seq_lens[b];
    return cond(len <= 2048,
        task(attn_2k, {b, len}, resources[b]),
        cond(len <= 8192,
            task(attn_8k, {b, len}, resources[b]),
            task(attn_32k, {b, len}, resources[b])
        )
    );
});

}  // namespace pto::wsp
```

### 3.5 combine

Sequential composition of workloads (both should run; schedule determines timing).

```cpp
namespace pto::wsp {

// combine: sequential composition
// Type: (Workload W1, Workload W2, ...) → Workload[W1.Axes ∪ W2.Axes, W1.Task ∪ W2.Task, Combined]
template<typename... Workloads>
auto combine(Workloads... workloads)
    -> Workload</* union axes */, /* union task */, Combined>;

// Example: Compose transformer layer
auto layer = combine(
    rms_norm_workload,      // First normalization
    attention_workload,     // Attention computation
    residual_add_workload,  // Residual connection
    ffn_workload            // Feed-forward network
);
// Note: combine does NOT imply execution order
// Schedule determines actual issue timing based on dependencies

// Example: Combine data-parallel and sequential workloads
auto mixed = combine(
    parallel_for(batch, [](b) { return task(prep_kernel, {b}); }),
    for_each(seq, [](i) { return task(scan_kernel, {i}); })
);

}  // namespace pto::wsp
```

### 3.6 sequential

Explicit ordering of workloads (B depends on A completing).

```cpp
namespace pto::wsp {

// sequential: explicit ordering (B waits for A)
// Type: (Workload A, Workload B, ...) → Workload[A.Axes ∪ B.Axes, A.Task ∪ B.Task, Sequential]
template<typename... Workloads>
auto sequential(Workloads... workloads)
    -> Workload</* union axes */, /* union task */, Sequential>;

// Example: Explicit ordering for task B after task A
auto ordered = sequential(
    task(task_A, params_A, resources_A),
    task(task_B, params_B, resources_B)  // B runs after A completes
);

// Example: Multi-stage processing with explicit ordering
auto pipeline = sequential(
    parallel_for(batch, [](b) { return task(load_kernel, {b}); }),
    parallel_for(batch, [](b) { return task(compute_kernel, {b}); }),
    parallel_for(batch, [](b) { return task(store_kernel, {b}); })
);
// Each stage waits for the previous stage to complete

// Note: Use sequential() for explicit ordering
// Use combine() when order doesn't matter (schedule determines timing)
// Use CSP channels for pipeline parallelism with double buffering

}  // namespace pto::wsp
```

### 3.7 task

Creates a single task (leaf workload).

```cpp
namespace pto::wsp {

// task: single kernel invocation
// Type: (Kernel K, Params P, Resources R) → Workload[Unit, Task[K,P,R], None]
template<typename Kernel, typename Params, typename Resources>
auto task(Kernel kernel, Params params, Resources resources)
    -> Workload<Unit, Task, None>;

// Helper for parameter pack
template<typename Kernel, typename... Args>
auto task(Kernel kernel, std::tuple<Args...> params, ResourceSet resources);

// Example:
auto t = task(attention_kernel,
    {.batch=0, .head=0, .seq_len=512},
    {Q[0][0], K[0], V[0], O[0][0]}
);

}  // namespace pto::wsp
```

### 3.7 cross

Cartesian product of axes (helper).

```cpp
namespace pto::wsp {

// cross: cartesian product iteration
template<typename... Axes>
auto cross(Axes... axes);

// Example:
auto w = for_each(cross(batch_axis, head_axis), [](int64_t b, int64_t h) {
    return task(kernel, {b, h}, resources);
});

}  // namespace pto::wsp
```

---

## 4. CSP Primitives (Pipeline-Parallel)

**Key Design Principle**: CSP channels carry **Workloads** (including single Tasks), not raw values. This unifies CSP with data-parallel primitives.

### 4.1 Channel

Typed, bounded communication channel for Workloads.

```cpp
namespace pto::wsp {

// Channel: typed bounded buffer (typically for Workloads)
template<typename T, size_t Capacity = 1>
class Channel {
public:
    using value_type = T;  // T is typically Workload or Task
    static constexpr size_t capacity = Capacity;

    // Create named channel
    static Channel create(std::string_view name);

    // Channel state
    bool is_open() const;
    size_t size() const;      // Current elements in buffer
    bool empty() const;
    bool full() const;
    void close();             // Signal no more values
};

// Example: Channel for Task workloads (pipeline depth = 2)
// Use constructor directly (factory functions removed)
Channel<Workload, 2> load_to_compute("l2c");

// Event: unbuffered channel for synchronization
// Event = Channel<Signal, 0>
using Event = Channel<Signal, 0>;

// Create event using constructor directly
Event sync_event("sync");

// Note: Factory functions (channel<T>(), create_event(), create_signal_channel())
// have been removed. Use constructors directly for simplicity.

}  // namespace pto::wsp
```

### 4.2 Process

Concurrent unit of execution with declarative body.

```cpp
namespace pto::wsp {

// ProcessBuilder: fluent API for process construction
template<typename In = void, typename Out = void>
class ProcessBuilder {
public:
    // Declare input channels
    template<typename... Channels>
    auto consumes(Channels&... chs) -> ProcessBuilder</* updated In */, Out>;

    // Declare output channels
    template<typename... Channels>
    auto produces(Channels&... chs) -> ProcessBuilder<In, /* updated Out */>;

    // Set process body (must be declarative - no while loops)
    template<typename Body>
    auto body(Body b) -> Process<In, Out>;
};

// Process: concurrent execution unit
template<typename In, typename Out>
class Process {
public:
    using input_type = In;
    using output_type = Out;

    std::string_view name() const;
};

// Process factory
auto process(std::string_view name) -> ProcessBuilder<>;

// Example: Process that sends Task workloads through channel
Process loader = process("loader")
    .produces(load_to_compute)
    .body(for_each(DenseDyn(num_tiles), [&](int i) {
        // Create Task workload and send through channel
        auto load_task = task(load_kernel, {i}, {input[i], tile_buf});
        return send(load_to_compute, load_task);
    }));

// Note: Process body must use declarative primitives only (JIT-friendly)
// - Use for_each, parallel_for, not for/while loops
// - Use consume, not while(recv())

}  // namespace pto::wsp
```

### 4.3 send

Put workload into channel (blocks if full).

```cpp
namespace pto::wsp {

// send: put workload into channel
// Blocks if channel is full (backpressure)
template<typename T, size_t N>
auto send(Channel<T, N>& ch, T workload) -> void;

// Non-blocking try_send
template<typename T, size_t N>
auto try_send(Channel<T, N>& ch, T workload) -> bool;

// Example: Send Task workload
auto load_task = task(load_kernel, {tile_idx}, resources);
send(load_channel, load_task);  // Task IS a Workload

}  // namespace pto::wsp
```

### 4.4 consume

Declarative iteration over channel (replaces while-recv).

```cpp
namespace pto::wsp {

// consume: declarative channel iteration
// Processes all workloads until channel is closed
// Type: (Channel[Workload], (Workload → Computation)) → Computation
template<typename T, size_t N, typename Body>
auto consume(Channel<T, N>& ch, Body body) -> Computation;

// Example: Process workloads from channel
Process computer = process("computer")
    .consumes(load_to_compute)
    .produces(compute_to_store)
    .body(consume(load_to_compute, [&](Workload load_result) {
        // Receive Task workload, create compute Task, send onwards
        auto compute_task = task(compute_kernel, {load_result}, {tile_buf, result_buf});
        return send(compute_to_store, compute_task);
    }));

// This replaces imperative:
//   while (auto t = recv(ch)) { ... }
// which is not JIT-friendly
//
// consume is JIT-analyzable: the iteration pattern is statically known

}  // namespace pto::wsp
```

### 4.5 connect

Wire processes and channels into a Pipeline with explicit lifecycle control.

```cpp
namespace pto::wsp {

// Pipeline: connected set of processes and channels
class Pipeline {
public:
    void start();       // Start all processes
    void join();        // Wait for all processes to complete
    bool is_running() const;
};

// connect: create Pipeline from processes and channels
template<typename... Processes, typename... Channels>
auto connect(std::tuple<Processes...> procs, std::tuple<Channels...> channels)
    -> Pipeline;

// Variadic version
template<typename... Args>
auto connect(std::initializer_list<Process> procs, std::initializer_list<Channel> channels)
    -> Pipeline;

// Example:
Pipeline mega = connect(
    {loader, computer, storer},
    {load_to_compute, compute_to_store}
);

// Explicit lifecycle control
mega.start();  // Start all processes
mega.join();   // Wait for completion

}  // namespace pto::wsp
```

### 4.6 replicate

Create multiple instances of a process.

```cpp
namespace pto::wsp {

// replicate: create N instances of a process
template<typename P>
auto replicate(P process, size_t count) -> ReplicatedProcess<P>;

// Example: 4 worker processes sharing a queue
Workload work_steal = connect(
    {generator, replicate(worker, 4)},
    {work_queue}
);

}  // namespace pto::wsp
```

---

## 5. Schedule API

### 5.1 Schedule Type

Schedule binds execution strategy to a workload.

```cpp
namespace pto::wsp {

template<typename WorkloadT>
class Schedule {
public:
    using workload_type = WorkloadT;

    // Dispatch phase: Task → AICPU
    auto dispatch(DispatchPolicy policy) -> Schedule&;

    // Issue phase: Task → Stream
    auto issue(IssuePolicy policy) -> Schedule&;

    // Stream configuration
    auto streams(int num_streams) -> Schedule&;
    auto stream_by(auto key_fn) -> Schedule&;

    // Timing control
    auto timing(TimingPolicy policy) -> Schedule&;

    // Compilation
    auto compile() -> Program;
    auto compile_async() -> std::future<Program>;
};

}  // namespace pto::wsp
```

### 5.2 Dispatch Policy

Dispatch determines which AICPU handles each task.

```cpp
namespace pto::wsp {

struct DispatchPolicy {
    // General: user-defined function (most flexible)
    // fn: Task → int (AICPU index)
    static auto dispatch_by(auto fn) -> DispatchPolicy;

    // Built-in policies (sugar for dispatch_by)
    static auto round_robin(int num_aicpus) -> DispatchPolicy;
    static auto affinity(auto axis) -> DispatchPolicy;
    static auto hash(auto key_fn) -> DispatchPolicy;
    static auto range(auto axis, int num_aicpus) -> DispatchPolicy;

    // Dynamic dispatch
    static auto work_steal() -> DispatchPolicy;
};

// Examples:
auto sched = workload.schedule()
    // Round-robin across 4 AICPUs
    .dispatch(DispatchPolicy::round_robin(4))

    // Same batch → same AICPU
    .dispatch(DispatchPolicy::affinity(batch_axis))

    // Custom: long sequences to dedicated AICPUs
    .dispatch(DispatchPolicy::dispatch_by([](Task t) {
        return t.param<int>("seq_len") > 8192 ? 0 : 1;
    }));

}  // namespace pto::wsp
```

### 5.3 Issue Policy

Issue determines task ordering within each AICPU.

```cpp
namespace pto::wsp {

struct IssuePolicy {
    // Stream assignment
    static auto stream_by(auto key_fn) -> IssuePolicy;    // fn: Task → int
    static auto single_stream() -> IssuePolicy;           // All in stream 0
    static auto per_axis(auto axis) -> IssuePolicy;       // Each axis value → stream

    // Ordering
    static auto fifo() -> IssuePolicy;                    // First-in-first-out
    static auto priority(auto priority_fn) -> IssuePolicy;
    // Note: lifo() has been removed (unclear use case)
};

// Examples:
auto sched = workload.schedule()
    // Assign to streams by batch index
    .stream_by([](Task t) { return t.param<int>("batch") % 2; })

    // Issue by priority (lower = higher priority)
    .issue(IssuePolicy::priority([](Task t) {
        return t.param<int>("seq_len");  // Short sequences first
    }));

}  // namespace pto::wsp
```

### 5.4 Timing Policy

Timing controls when tasks are issued.

```cpp
namespace pto::wsp {

struct TimingPolicy {
    // Issue as soon as dependencies satisfied
    static auto immediate() -> TimingPolicy;

    // Batch N tasks before issuing
    static auto batched(int n) -> TimingPolicy;

    // Round-robin across streams
    static auto interleaved(int streams) -> TimingPolicy;

    // Rate limiting
    static auto rate_limit(int tasks_per_ms) -> TimingPolicy;
};

// Example:
auto sched = workload.schedule()
    .streams(4)
    .timing(TimingPolicy::interleaved(4));  // Round-robin issue

}  // namespace pto::wsp
```

---

## 6. Stream and Event API

### 6.1 Stream Type

```cpp
namespace pto::wsp {

// Stream: ordered sequence of task issues
class Stream {
public:
    StreamId id() const;

    // Issue task to this stream
    void issue(Task& task);

    // Record event at current position
    Event record();

    // Wait for event before continuing
    void wait(Event e);

    // Synchronize: wait for all issued tasks to complete
    void synchronize();
};

// Stream ordering
enum class StreamOrdering {
    FIFO,       // First-in-first-out (default)
    Priority,   // By task priority
    Deadline,   // By deadline
};

}  // namespace pto::wsp
```

### 6.2 Event Type (Unified CSP Model)

**Event = Channel<Signal, 0>** - Events are unified with CSP channels using true rendezvous semantics.

```cpp
namespace pto::wsp {

// Signal type for Events (empty marker)
struct Signal {};

// Event: unbuffered rendezvous channel for synchronization
// Event = Channel<Signal, 0> with true rendezvous semantics
using Event = Channel<Signal, 0>;

// Event API maps to Channel operations:
//   record(event)      → send(event, Signal{})  -- blocks until synchronize()
//   synchronize(event) → recv(event)            -- blocks until record()
//   query(event)       → try_recv(event)        -- non-blocking

// Create event using constructor directly
Event sync_event("my_event");

// Event operations (free functions)
void record(Event& e);       // Signal completion (blocks until receiver ready)
void synchronize(Event& e);  // Wait for completion (blocks until sender ready)
bool query(Event& e);        // Non-blocking check (true if sender is waiting)

}  // namespace pto::wsp
```

**Key properties of rendezvous semantics:**
- `record()` (send) blocks until `synchronize()` (recv) is called
- `synchronize()` (recv) blocks until `record()` (send) is called
- Both complete simultaneously (true rendezvous handoff)
- `close()` unblocks any waiting threads

### 6.3 Cross-Stream Synchronization

```cpp
namespace pto::wsp {

// Stream waits for event from another stream
void stream_wait(Stream& s, Event& e);

// Barrier: wait for multiple events
void barrier(std::vector<EventPtr> events);

// Example with unified CSP model:
Stream s0, s1;
s0.issue(producer_task);
EventPtr e = s0.record_event();  // Create event for this stream

// In another thread, signal when producer completes:
record(*e);

// s1 waits using rendezvous:
synchronize(*e);  // Blocks until record() is called
s1.issue(consumer_task);

}  // namespace pto::wsp
```

---

## 7. Compilation API

### 7.1 Program Type

```cpp
namespace pto::wsp {

// Program: compiled schedule ready for execution
class Program {
public:
    // Execute the program
    void execute();
    void execute_async();

    // Wait for completion
    void synchronize();

    // Query status
    bool is_complete() const;

    // Profiling
    Duration elapsed() const;
    ProgramStats stats() const;
};

struct ProgramStats {
    size_t num_tasks;
    size_t num_streams;
    size_t num_aicpus;
    Duration compile_time;
    Duration execute_time;
};

}  // namespace pto::wsp
```

### 7.2 Compilation

```cpp
namespace pto::wsp {

// Compile schedule to program
template<typename WorkloadT>
Program compile(Schedule<WorkloadT>& sched);

// Compilation options
struct CompileOptions {
    bool enable_profiling = false;
    bool enable_debug = false;
    int optimization_level = 2;      // 0-3
    size_t max_code_size = 64 * 1024;  // Max AICPU binary size
};

template<typename WorkloadT>
Program compile(Schedule<WorkloadT>& sched, CompileOptions opts);

// Example:
auto sched = workload.schedule()
    .dispatch(DispatchPolicy::round_robin(4))
    .streams(2)
    .timing(TimingPolicy::immediate());

Program prog = sched.compile();
prog.execute();
prog.synchronize();

}  // namespace pto::wsp
```

### 7.3 JIT Compilation Details

```cpp
namespace pto::wsp {

// JIT compiles workload + schedule to AICPU binary
//
// Compilation phases:
// 1. Workload analysis: extract iteration space, dependencies
// 2. Task enumeration: generate concrete tasks (may be lazy)
// 3. Schedule application: assign dispatch, issue, timing
// 4. Code generation: emit AICPU binary
// 5. Linking: link with kernel symbols

// The compiled program contains:
// - Per-AICPU code for task dispatch loop
// - Stream management logic
// - Dependency tracking
// - Event signaling

}  // namespace pto::wsp
```

---

## 8. Kernel Registration

### 8.1 Kernel Definition

```cpp
namespace pto::wsp {

// Kernel identifier
struct KernelId {
    uint32_t id;
    const char* name;
};

// Register kernel at compile time
#define REGISTER_KERNEL(name, params_type, resources_type) \
    static constexpr KernelId name##_kernel = { __COUNTER__, #name }; \
    using name##_params = params_type; \
    using name##_resources = resources_type;

// Example:
struct AttentionParams {
    int batch_idx;
    int head_idx;
    int seq_len;
};

struct AttentionResources {
    Tensor* q;
    Tensor* k;
    Tensor* v;
    Tensor* output;
};

REGISTER_KERNEL(attention, AttentionParams, AttentionResources);

}  // namespace pto::wsp
```

### 8.2 Kernel Invocation

```cpp
namespace pto::wsp {

// Create task for registered kernel
template<typename K>
auto make_task(K kernel, typename K::params_type params, typename K::resources_type resources)
    -> Task;

// Example:
auto t = make_task(attention_kernel,
    AttentionParams{.batch_idx=0, .head_idx=0, .seq_len=512},
    AttentionResources{.q=Q[0][0], .k=K[0], .v=V[0], .output=O[0][0]}
);

}  // namespace pto::wsp
```

---

## 9. Complete Examples

### 9.1 Attention with Variable Lengths

```cpp
using namespace pto::wsp;

// Input data
int batch_size = 4;
int num_heads = 8;
int64_t seq_lens[] = {512, 2048, 8192, 32768};

Tensor* Q[4][8], *K[4], *V[4], *O[4][8];  // Allocated elsewhere

// Define workload
auto attn = parallel_for(DenseDyn(batch_size), [&](int64_t b) {
    return parallel_for(Dense<8>{}, [&, b](int64_t h) {
        return task(attention_kernel,
            {.batch=b, .head=h, .seq_len=seq_lens[b]},
            {Q[b][h], K[b], V[b], O[b][h]}
        );
    });
});

// Define schedule
auto sched = attn.schedule()
    .dispatch(DispatchPolicy::affinity(/* batch axis */))  // Same batch → same AICPU
    .streams(2)
    .stream_by([](Task t) { return t.param<int>("batch") % 2; })
    .timing(TimingPolicy::immediate());

// Compile and execute
Program prog = sched.compile();
prog.execute();
prog.synchronize();
```

### 9.2 MoE with Sparse Routing

```cpp
using namespace pto::wsp;

// Routing results (CSR format)
int64_t indptr[] = {0, 2, 5, 7, 10};      // 4 tokens
int64_t indices[] = {1, 3, 0, 2, 4, 1, 5, 0, 3, 7};  // Selected experts

Sparse routing(4, indptr, indices);

// Define workload
auto moe = parallel_for(DenseDyn(batch_size), [&](int64_t b) {
    return select(routing[b], [&, b](int64_t e) {
        return task(expert_kernels[e],
            {.batch=b, .expert=e},
            {tokens[b], expert_weights[e], expert_out[b][e]}
        );
    });
});

// Schedule with work stealing for load balance
auto sched = moe.schedule()
    .dispatch(DispatchPolicy::work_steal())
    .streams(8)
    .stream_by([](Task t) { return t.param<int>("expert"); });

Program prog = sched.compile();
prog.execute();
```

### 9.3 Megakernel Pipeline (CSP)

```cpp
using namespace pto::wsp;

// Channels
Channel<Tile> l2c = channel<Tile>("load_to_compute", 2);
Channel<Result> c2s = channel<Result>("compute_to_store", 2);

// Processes
Process loader = process("loader")
    .produces(l2c)
    .body(for_each(DenseDyn(num_tiles), [&](int64_t i) {
        Tile t = task(load_kernel, {i}, {input[i], tile_buf});
        return send(l2c, t);
    }));

Process computer = process("computer")
    .consumes(l2c)
    .produces(c2s)
    .body(consume(l2c, [&](Tile t) {
        Result r = task(compute_kernel, {t}, {tile_buf, result_buf});
        return send(c2s, r);
    }));

Process storer = process("storer")
    .consumes(c2s)
    .body(consume(c2s, [&](Result r) {
        return task(store_kernel, {r}, {result_buf, output});
    }));

// Connect into CSP workload
Workload mega = connect({loader, computer, storer}, {l2c, c2s});

// Schedule
auto sched = mega.schedule()
    .dispatch(DispatchPolicy::round_robin(1));  // Single AICPU

Program prog = sched.compile();
prog.execute();
```

### 9.4 Work-Stealing Scheduler

```cpp
using namespace pto::wsp;

// Shared work queue
Channel<Task> queue = channel<Task>("work_queue", 1024);

// Generator
Process gen = process("generator")
    .produces(queue)
    .body(for_each(cross(DenseDyn(batch), Dense<8>{}), [&](int64_t b, int64_t h) {
        Task t = make_task(attn_kernel, {b, h}, {Q[b][h], K[b], V[b], O[b][h]});
        return send(queue, t);
    }));

// Worker
Process worker = process("worker")
    .consumes(queue)
    .body(consume(queue, [](Task t) {
        return execute(t);
    }));

// 4 workers steal from shared queue
Workload ws = connect({gen, replicate(worker, 4)}, {queue});

Program prog = ws.schedule()
    .dispatch(DispatchPolicy::dispatch_by([](Task t) {
        // Assign to AICPU based on which worker process
        return t.process_id();
    }))
    .compile();

prog.execute();
```

---

## 10. Error Handling

### 10.1 Exception Types

```cpp
namespace pto::wsp {

// Base exception
class RuntimeError : public std::runtime_error {
    using std::runtime_error::runtime_error;
};

// Compilation errors
class CompileError : public RuntimeError {
    using RuntimeError::RuntimeError;
};

// Execution errors
class ExecutionError : public RuntimeError {
    using RuntimeError::RuntimeError;
};

// Resource errors
class ResourceError : public RuntimeError {
    using RuntimeError::RuntimeError;
};

// Channel errors
class ChannelError : public RuntimeError {
    using RuntimeError::RuntimeError;
};

class ChannelClosed : public ChannelError {
    using ChannelError::ChannelError;
};

}  // namespace pto::wsp
```

### 10.2 Error Codes

```cpp
namespace pto::wsp {

enum class ErrorCode {
    OK = 0,

    // Compilation
    INVALID_WORKLOAD,
    INVALID_SCHEDULE,
    CODE_SIZE_EXCEEDED,

    // Execution
    KERNEL_NOT_FOUND,
    RESOURCE_NOT_AVAILABLE,
    TIMEOUT,

    // CSP
    CHANNEL_CLOSED,
    DEADLOCK_DETECTED,
};

}  // namespace pto::wsp
```

---

## 11. Summary

| Category | Primitives |
|----------|------------|
| **Axes** | `Dense<N>`, `DenseDyn`, `Ragged`, `Sparse` |
| **Data-Parallel** | `parallel_for`, `for_each`, `combine`, `sequential`, `select`, `cond`, `task`, `cross` |
| **CSP** | `process`, `Channel`, `send`, `consume`, `connect`, `replicate` |
| **Schedule** | `dispatch`, `issue`, `streams`, `stream_by`, `timing` |
| **Policies** | `dispatch_by`, `round_robin`, `affinity`, `work_steal`, `stream_by`, `fifo`, `priority` |
| **Execution** | `Stream`, `Event`, `Program`, `Pipeline`, `compile`, `execute` |
| **Sync** | `Event = Channel<Signal, 0>`, `record`, `synchronize` |

### Key Design Principles (v8.2)

1. **CSP channels carry Workloads** - Tasks are Workloads, unifying CSP with data-parallel
2. **Event = unbuffered Channel** - `record()/synchronize()` ≈ `send()/recv()` on `Channel<Signal, 0>`
3. **Process bodies must be declarative** - Use `for_each`, `consume`, not `while`/`for` loops
4. **`combine()` for composition** - Combine workloads without implying execution order
5. **`sequential()` for ordering** - Explicit ordering when B must wait for A

### Removed from v8.1 (Redundancy Cleanup)

- `Range` axis type (use `DenseDyn`)
- `DAG` dependency type (use structural deps or `ChannelDep`)
- `Reduction` dependency type (not implemented)
- CSP `select` (not needed in examples)
- `lifo()` issue policy (unclear use case)
- Factory functions (`create_event()`, `channel<T,N>()`) - use constructors

---

*Version: 8.2*
*Last Updated: 2026-01-22*
*Changes: Added `sequential()`, removed redundancies (Range, DAG, CSP select, lifo, factories)*
