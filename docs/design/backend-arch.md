# PTO Workload-Schedule Programming (PTO-WSP) v9: Backend Architecture

## 1. Overview

This document specifies the backend architecture for v9 runtime extension, supporting:

1. **CPU Simulation Backend** - Fast iteration, debugging, reference implementation
2. **Ascend NPU Backend** - Production deployment on 910B/910C/950
3. **AMD AIE Backend** - Spatial architecture (XDNA1/XDNA2)

### 1.1 Backend Interface

All backends implement a common interface:

```cpp
namespace pto::wsp::backend {

class Backend {
public:
    virtual ~Backend() = default;

    // Compile IR to executable program
    virtual std::unique_ptr<Program> compile(
        const ir::WorkloadDef& workload,
        const ir::ScheduleDef& schedule) = 0;

    // Compile CSP pipeline
    virtual std::unique_ptr<Program> compile(
        const ir::PipelineNode& pipeline,
        const ir::ScheduleDef& schedule) = 0;

    // Query capabilities
    virtual bool supports(ir::NodeKind kind) const = 0;
    virtual std::string name() const = 0;
    virtual std::vector<std::string> supported_targets() const = 0;
};

class Program {
public:
    virtual ~Program() = default;

    // Execution
    virtual void execute() = 0;
    virtual void execute_async() = 0;
    virtual void synchronize() = 0;
    virtual bool is_complete() const = 0;

    // Profiling
    virtual double elapsed_ms() const = 0;
    virtual ProgramStats stats() const = 0;
};

struct ProgramStats {
    size_t num_tasks;
    size_t num_streams;
    size_t num_executors;  // AICPUs, threads, tiles
    double compile_time_ms;
    double execute_time_ms;
};

}  // namespace pto::wsp::backend
```

---

## 2. CPU Simulation Backend

### 2.1 Purpose

- Fast iteration during development
- Reference implementation for correctness
- Debugging with full observability
- Cross-platform (Linux, macOS, Windows)

### 2.2 Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         CPU Simulation Backend                               │
│                                                                              │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────────────┐  │
│  │   IR Lowering   │ -> │   Task Graph    │ -> │   Thread Pool Executor  │  │
│  │                 │    │   Construction  │    │                         │  │
│  │ - Enumerate     │    │ - TaskNode      │    │ - Ready queue           │  │
│  │ - Dependency    │    │ - fanin/fanout  │    │ - Worker threads        │  │
│  │   analysis      │    │ - TensorMap     │    │ - Dependency tracking   │  │
│  └─────────────────┘    └─────────────────┘    └─────────────────────────┘  │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                      Kernel Registry                                 │    │
│  │  kernel_name -> std::function<void(TaskParams&)>                    │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.3 Task Graph Runtime

Based on pto-isa-lh design (Report 13):

```cpp
namespace pto::wsp::backend::cpu {

// Task in the graph
struct TaskNode {
    int32_t task_id;
    std::string kernel_name;
    std::vector<std::string> params;
    std::vector<TensorRegion> resources;

    // Dependencies
    std::atomic<int32_t> fanin{0};       // Remaining input deps
    std::vector<int32_t> fanout;          // Downstream task IDs

    // Status
    std::atomic<bool> is_complete{false};
};

// Tensor region for dependency tracking
struct TensorRegion {
    void* data;
    int64_t offset;
    int64_t size;
    bool is_output;

    bool overlaps(const TensorRegion& other) const;
};

// Producer lookup table
class TensorMap {
    struct Entry {
        TensorRegion region;
        int32_t producer_task_id;
    };
    std::unordered_map<void*, std::vector<Entry>> map;

public:
    void register_output(void* tensor, TensorRegion region, int32_t task_id);
    std::optional<int32_t> find_producer(void* tensor, TensorRegion region) const;
};

// Main runtime
class TaskGraphRuntime {
    std::vector<TaskNode> tasks;
    TensorMap producer_map;
    std::vector<int32_t> ready_queue;

    std::mutex queue_mutex;
    std::condition_variable queue_cv;

public:
    // Build graph from IR
    void build(const ir::WorkloadDef& workload);

    // Add task (returns task_id)
    int32_t add_task(const std::string& kernel,
                     const std::vector<std::string>& params,
                     const std::vector<TensorRegion>& resources);

    // Execute all tasks
    void execute(int num_threads);

private:
    void worker_thread();
    void task_complete(int32_t task_id);
};

}
```

### 2.4 Thread Pool Executor

```cpp
namespace pto::wsp::backend::cpu {

class ThreadPoolExecutor {
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> work_queue;
    std::mutex queue_mutex;
    std::condition_variable work_available;
    std::atomic<bool> shutdown{false};
    std::atomic<int32_t> tasks_pending{0};

public:
    explicit ThreadPoolExecutor(int num_threads);
    ~ThreadPoolExecutor();

    void submit(std::function<void()> task);
    void wait_all();

private:
    void worker_loop();
};

}
```

### 2.5 Kernel Registry

```cpp
namespace pto::wsp::backend::cpu {

using KernelFunc = std::function<void(const std::vector<void*>& params,
                                       const std::vector<TensorRegion>& resources)>;

class KernelRegistry {
    std::unordered_map<std::string, KernelFunc> kernels;

public:
    static KernelRegistry& instance();

    void register_kernel(const std::string& name, KernelFunc func);
    KernelFunc get_kernel(const std::string& name) const;
    bool has_kernel(const std::string& name) const;
};

// Registration macro
#define REGISTER_CPU_KERNEL(name, func) \
    static bool _reg_##name = (KernelRegistry::instance().register_kernel(#name, func), true)

}
```

### 2.6 Complete Backend

```cpp
namespace pto::wsp::backend::cpu {

class CPUSimBackend : public Backend {
    int num_threads_;

public:
    explicit CPUSimBackend(int num_threads = std::thread::hardware_concurrency())
        : num_threads_(num_threads) {}

    std::unique_ptr<Program> compile(
        const ir::WorkloadDef& workload,
        const ir::ScheduleDef& schedule) override;

    std::unique_ptr<Program> compile(
        const ir::PipelineNode& pipeline,
        const ir::ScheduleDef& schedule) override;

    bool supports(ir::NodeKind kind) const override { return true; }
    std::string name() const override { return "cpu_sim"; }
    std::vector<std::string> supported_targets() const override { return {"cpu_sim"}; }
};

class CPUSimProgram : public Program {
    TaskGraphRuntime runtime_;
    int num_threads_;
    std::chrono::duration<double, std::milli> elapsed_;

public:
    void execute() override {
        auto start = std::chrono::high_resolution_clock::now();
        runtime_.execute(num_threads_);
        elapsed_ = std::chrono::high_resolution_clock::now() - start;
    }

    void execute_async() override;
    void synchronize() override;
    bool is_complete() const override;
    double elapsed_ms() const override { return elapsed_.count(); }
    ProgramStats stats() const override;
};

}
```

### 2.7 Performance Target

From pto-isa-lh benchmarks (Report 13):

| Metric | Target |
|--------|--------|
| Task throughput | ~5000+ tasks/ms |
| Graph build overhead | <5% of execution time |
| Thread scalability | Linear to 8 threads |

---

## 3. Ascend NPU Backend

### 3.1 Purpose

- Production deployment on Ascend 910B/910C/950
- Leverage AICPU for task orchestration
- Support cube/vector pipelines
- InCore function integration

### 3.2 Architecture

Based on pto-isa-wc design (Report 14):

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Ascend NPU Backend                                 │
│                                                                              │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────────────┐  │
│  │   IR Lowering   │ -> │  AICPU CodeGen  │ -> │    Graph Executor       │  │
│  │                 │    │                 │    │                         │  │
│  │ - Schedule      │    │ - Orchestration │    │ - Multi-threaded        │  │
│  │   application   │    │   function      │    │ - Handshake protocol    │  │
│  │ - Dispatch      │    │ - Task loop     │    │ - AICore coordination   │  │
│  │   routing       │    │ - Dependency    │    │                         │  │
│  └─────────────────┘    └─────────────────┘    └─────────────────────────┘  │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    AICore Kernel Wrappers                            │    │
│  │  kernel_name -> Ascend C wrapper function                           │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.3 Code Generation

#### 3.3.1 AICPU Orchestration Code

```cpp
namespace pto::wsp::backend::npu {

class AICPUCodeGen {
public:
    // Generate orchestration function
    std::string generate(const ir::WorkloadDef& workload,
                         const ir::ScheduleDef& schedule);

    // Generate task dispatch code
    std::string generate_dispatch(const ir::DispatchNode& dispatch);

    // Generate stream management
    std::string generate_stream_mgmt(const ir::StreamNode& stream);
};

}
```

Generated AICPU code structure:

```cpp
// Generated orchestration function
void ${workload_name}_orchestrate(${params}) {
    // Initialize graph
    Graph graph;

    // Generate tasks from workload IR
    ${task_generation_code}

    // Apply dispatch policy
    ${dispatch_code}

    // Execute graph
    GraphExecutor executor(graph);
    executor.execute();
}
```

#### 3.3.2 AICore Kernel Wrappers

```cpp
namespace pto::wsp::backend::npu {

class AICorewrapperGen {
public:
    // Generate wrapper for registered kernel
    std::string generate_wrapper(const std::string& kernel_name,
                                 const KernelSignature& sig);

    // Generate InCore function call
    std::string generate_incore_call(const ir::TaskNode& task);
};

}
```

Generated wrapper structure:

```cpp
// Generated AICore wrapper
extern "C" __aicore__ void ${kernel_name}_wrapper(
    GM_ADDR params_gm,
    GM_ADDR ${resources}
) {
    // Load parameters from GM
    ${param_struct} params;
    DataCopy(params, params_gm, sizeof(params));

    // Call InCore function
    ${kernel_name}_incore(params, ${resources});

    // Signal completion
    ${completion_signal}
}
```

### 3.4 Graph Executor

From pto-isa-wc (Report 14):

```cpp
namespace pto::wsp::backend::npu {

struct Node {
    int32_t node_id;
    std::string kernel_name;
    void* params;
    size_t params_size;
    std::vector<TensorRegion> resources;

    std::vector<int32_t> dependencies;
    std::atomic<int32_t> pending_deps{0};
};

class Graph {
    std::vector<Node> nodes;
    std::vector<std::vector<int32_t>> wait_graph;  // Dependency edges

public:
    int32_t add_node(const std::string& kernel,
                     void* params, size_t params_size,
                     const std::vector<TensorRegion>& resources);

    void add_dependency(int32_t from, int32_t to);
    void finalize();

    const std::vector<Node>& get_nodes() const { return nodes; }
    const std::vector<int32_t>& get_ready_nodes() const;
};

class GraphExecutor {
    Graph& graph;
    std::vector<std::thread> workers;
    std::queue<int32_t> ready_queue;
    std::mutex queue_mutex;

public:
    explicit GraphExecutor(Graph& g, int num_threads = 4);

    void execute();

private:
    void worker_thread();
    void launch_kernel(const Node& node);
    void node_complete(int32_t node_id);
};

}
```

### 3.5 AICPU-AICore Handshake

```cpp
namespace pto::wsp::backend::npu {

// Handshake primitives
void kernel_launch(int kernel_id, void* params, size_t params_size);
void task_wait_for_kernel(int kernel_id);
void signal_kernel_complete(int kernel_id);

// Stream synchronization
class StreamManager {
    std::vector<cudaStream_t> streams;  // Or Ascend equivalent
    std::unordered_map<int32_t, int> task_to_stream;

public:
    void assign_stream(int32_t task_id, int stream_id);
    void synchronize_stream(int stream_id);
    void synchronize_all();
};

}
```

### 3.6 Complete Backend

```cpp
namespace pto::wsp::backend::npu {

class AscendNPUBackend : public Backend {
    AICPUCodeGen aicpu_codegen;
    AICorewrapperGen aicore_codegen;

public:
    std::unique_ptr<Program> compile(
        const ir::WorkloadDef& workload,
        const ir::ScheduleDef& schedule) override;

    std::unique_ptr<Program> compile(
        const ir::PipelineNode& pipeline,
        const ir::ScheduleDef& schedule) override;

    bool supports(ir::NodeKind kind) const override;
    std::string name() const override { return "ascend_npu"; }
    std::vector<std::string> supported_targets() const override {
        return {"ascend_npu", "ascend_910b", "ascend_910c", "ascend_950"};
    }
};

}
```

---

## 4. AMD AIE Backend

### 4.1 Purpose

- Target AMD XDNA1/XDNA2 (Ryzen AI NPU)
- Spatial architecture with tile array
- Stream-based communication
- Layout-aware data distribution

### 4.2 Architecture

Based on allo and dato (Reports 15-16):

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            AMD AIE Backend                                   │
│                                                                              │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────────────┐  │
│  │   IR Lowering   │ -> │  Spatial Mapper │ -> │   MLIR Lowering         │  │
│  │                 │    │                 │    │                         │  │
│  │ - SpatialMap    │    │ - Task → Tile   │    │ - aie dialect           │  │
│  │ - Layout        │    │ - Data layout   │    │ - memref/arith          │  │
│  │ - Stream        │    │ - Stream route  │    │ - func                  │  │
│  └─────────────────┘    └─────────────────┘    └─────────────────────────┘  │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                       Tile Placement                                 │    │
│  │  WorkloadNode → (tile_x, tile_y, compute_kernel)                    │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                       Data Distribution                              │    │
│  │  Tensor → layout_per_tile (based on Shard/Replicate annotations)    │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.3 Spatial Mapper

```cpp
namespace pto::wsp::backend::aie {

struct TilePlacement {
    int tile_x;
    int tile_y;
    std::string kernel_name;
    std::vector<std::string> resources;
};

struct StreamRoute {
    int src_tile_x, src_tile_y;
    int dst_tile_x, dst_tile_y;
    std::string channel_name;
    int bandwidth;
};

class SpatialMapper {
public:
    // Map workload to tile grid
    std::vector<TilePlacement> map_workload(
        const ir::WorkloadDef& workload,
        const ir::SpatialMapNode& spatial_map);

    // Route channels through tile array
    std::vector<StreamRoute> route_channels(
        const ir::PipelineNode& pipeline,
        const std::vector<TilePlacement>& placements);

    // Apply data layout
    void apply_layout(const std::vector<ir::LayoutNode>& layouts,
                      std::vector<TilePlacement>& placements);
};

}
```

### 4.4 Layout Application

From dato paper (Report 16):

```cpp
namespace pto::wsp::backend::aie {

// Layout descriptor for a tensor
struct TensorLayout {
    std::string tensor_name;
    std::vector<ir::LayoutDim> dims;

    // Compute local tile for given global coordinates
    TileCoord get_tile(const std::vector<int64_t>& global_idx,
                       const std::vector<int64_t>& grid) const;

    // Compute local offset within tile
    std::vector<int64_t> get_local_idx(const std::vector<int64_t>& global_idx,
                                        const std::vector<int64_t>& grid) const;
};

class LayoutAnalyzer {
public:
    // Analyze layout requirements for workload
    std::vector<TensorLayout> analyze(const ir::WorkloadDef& workload,
                                       const std::vector<ir::LayoutNode>& layouts);

    // Check layout compatibility for operations
    bool are_compatible(const TensorLayout& a, const TensorLayout& b,
                        ir::NodeKind op) const;
};

}
```

### 4.5 MLIR Lowering

```cpp
namespace pto::wsp::backend::aie {

class MLIRLowering {
    mlir::MLIRContext ctx;
    mlir::OpBuilder builder;

public:
    MLIRLowering();

    // Lower IR to MLIR module
    mlir::OwningOpRef<mlir::ModuleOp> lower(
        const ir::WorkloadDef& workload,
        const ir::ScheduleDef& schedule,
        const std::vector<TilePlacement>& placements);

    // Lower CSP pipeline
    mlir::OwningOpRef<mlir::ModuleOp> lower(
        const ir::PipelineNode& pipeline,
        const std::vector<TilePlacement>& placements,
        const std::vector<StreamRoute>& routes);

private:
    // Lower individual nodes
    mlir::Value lower_task(const ir::TaskNode& task);
    mlir::Value lower_parallel_for(const ir::ParallelForNode& pf);
    mlir::Value lower_channel(const ir::ChannelNode& ch);
    mlir::Value lower_process(const ir::ProcessNode& proc);
};

}
```

### 4.6 Complete Backend

```cpp
namespace pto::wsp::backend::aie {

class AMDAIEBackend : public Backend {
    SpatialMapper mapper;
    MLIRLowering lowering;

public:
    std::unique_ptr<Program> compile(
        const ir::WorkloadDef& workload,
        const ir::ScheduleDef& schedule) override;

    std::unique_ptr<Program> compile(
        const ir::PipelineNode& pipeline,
        const ir::ScheduleDef& schedule) override;

    bool supports(ir::NodeKind kind) const override;
    std::string name() const override { return "amd_aie"; }
    std::vector<std::string> supported_targets() const override {
        return {"amd_aie", "xdna1", "xdna2"};
    }
};

}
```

---

## 5. Backend Selection

### 5.1 Backend Registry

```cpp
namespace pto::wsp::backend {

class BackendRegistry {
    std::unordered_map<std::string, std::unique_ptr<Backend>> backends;

public:
    static BackendRegistry& instance();

    void register_backend(std::unique_ptr<Backend> backend);
    Backend* get_backend(const std::string& name) const;
    std::vector<std::string> available_backends() const;

    // Auto-select based on capabilities
    Backend* select_backend(const ir::Module& module) const;
};

// Registration
static bool register_cpu_sim = []() {
    BackendRegistry::instance().register_backend(
        std::make_unique<cpu::CPUSimBackend>());
    return true;
}();

static bool register_npu = []() {
    BackendRegistry::instance().register_backend(
        std::make_unique<npu::AscendNPUBackend>());
    return true;
}();

static bool register_aie = []() {
    BackendRegistry::instance().register_backend(
        std::make_unique<aie::AMDAIEBackend>());
    return true;
}();

}
```

### 5.2 Compile Options

```cpp
namespace pto::wsp::backend {

struct CompileOptions {
    std::string target = "cpu_sim";  // Default
    bool enable_profiling = false;
    bool enable_debug = false;
    int optimization_level = 2;  // 0-3

    // CPU-specific
    int num_threads = 0;  // 0 = auto

    // NPU-specific
    int num_aicpus = 1;
    int num_streams = 2;

    // AIE-specific
    std::vector<int64_t> grid;  // Tile grid dimensions
};

// Compile with options
std::unique_ptr<Program> compile(const ir::Module& module,
                                  const CompileOptions& options);

}
```

---

## 6. Shared Infrastructure (Code Reuse Strategy)

This section addresses requirement R6 from `docs/task_plan.md`: maximizing code reuse across backends while keeping backend-specific logic cleanly separated.

### 6.1 Design Principles

1. **Two-layer representation**: Separate "built graph" (device-copyable, immutable) from "runtime state" (host-only, mutable)
2. **Portable core**: C-compatible POD types usable by AICPU/AICore builds (no STL/exceptions)
3. **Pluggable components**: Interchangeable implementations for TensorMap, ReadyQueue with common interfaces
4. **Template-based codegen**: Generate thin glue code, link against shared runtime library

### 6.2 Shared Data Structures

#### 6.2.1 Core Types (Portable)

```cpp
namespace pto::wsp::graph {

// Execution domain: where does the task run?
enum class ExecDomain : uint8_t { HostCPU, AscendAICore, AMDAIETile };

// Execution pool: for dual-queue scheduling (vector/cube workers)
enum class ExecPool : uint8_t { Vector = 0, Cube = 1, Any = 255 };

using TaskId   = uint32_t;
using KernelId = uint16_t;
using StreamId = uint16_t;
using TargetId = uint16_t;  // Per-core/per-tile affinity

// Tensor region for dependency tracking (2D tile-based)
struct TensorRegion2D {
    uint64_t base;           // Pointer/handle (device-visible)
    int32_t  row_off, col_off;
    int32_t  rows, cols;
};

// Task I/O descriptor
struct TaskIO {
    TensorRegion2D region;
    uint8_t is_output;       // 0 = input, 1 = output
};

// Plain-old-data task node (device-copyable)
struct TaskNodePod {
    TaskId    id;
    KernelId  kernel;
    ExecDomain domain;
    ExecPool   pool;
    StreamId   stream;
    TargetId   affinity;

    // Dependencies (fanin = atomic counter decremented as deps complete)
    int32_t  fanin;
    uint32_t fanout_begin;   // Index into flat fanout_edges[]
    uint16_t fanout_count;

    // Arguments and I/O
    uint16_t num_u64_args;
    uint16_t num_io;
    // Followed by: uint64_t u64_args[MAX_ARGS]; TaskIO io[MAX_IO];

    // Schedule metadata (extended primitives)
    uint16_t sched_tags;     // Bitfield: barrier, flush_batch, etc.
};

}  // namespace pto::wsp::graph
```

#### 6.2.2 TaskGraph (Two-Layer)

```cpp
namespace pto::wsp::graph {

// Layer 1: TaskGraphStorage (immutable after build, device-copyable)
struct TaskGraphStorage {
    std::vector<TaskNodePod> tasks;
    std::vector<TaskId> fanout_edges;  // Flat adjacency list
    std::vector<uint32_t> task_offsets;  // Optional: for faster iteration
    KernelBundle kernel_table;           // KernelId → symbol/name/ABI
};

// Layer 2: TaskGraphRuntime (host-scheduler state, not device-copied)
struct TaskGraphRuntime {
    TensorMap producer_map;      // Dependency inference during build
    ReadyQueueSet ready;         // Task scheduling
    WindowState window;          // task_window primitive support
    IssueGates gates;            // pipeline_depth primitive support
    DepBatcher batcher;          // batch_deps primitive support
    // Counters, tracing hooks
};

}
```

#### 6.2.3 TensorMap

Two interchangeable implementations:

```cpp
namespace pto::wsp::graph {

struct ProducerRef {
    TaskId producer;
    uint32_t generation;  // For GC/window support
};

class TensorMap {
public:
    // Register output region produced by task
    void insert_output(TensorRegion2D r, ProducerRef p);

    // Find latest producer overlapping region (exact match first)
    bool find_producer(TensorRegion2D r, ProducerRef* out) const;

    // task_window support: garbage collect entries before oldest_live
    void gc_before(TaskId oldest_live_task);
};

// Implementations:
// - FixedTensorMap: hash table + entry pool (AICPU, fast CPU sim)
// - DynamicTensorMap: std::unordered_map (debugging, large workloads)

}
```

#### 6.2.4 ReadyQueueSet

Unified multi-queue interface for dual-queue (vector/cube) dispatch:

```cpp
namespace pto::wsp::graph {

class ReadyQueueSet {
public:
    void push(ExecPool pool, TaskId tid);
    bool try_pop(ExecPool pool, TaskId* tid);   // Per-pool worker
    bool try_pop_any(TaskId* tid);               // Steal mode
};

// Implementations:
// - ReadyQueueMPMC: mutex+cv or lock-free ring (CPU sim, AICPU)
// - ReadyQueueSPSC: per-worker fast path (optional optimization)

}
```

### 6.3 Two-Phase Compilation

Maximize reuse by separating **shared lowering** from **backend-specific emission**:

```cpp
namespace pto::wsp::backend {

// Lowered form (backend-neutral)
struct LoweredPlan {
    graph::TaskGraphStorage graph;
    KernelBundle kernels;
    ScheduleRuntimeConfig sched;  // Lowered extended primitives
};

// Backend interface with two-phase compilation
class Backend {
public:
    virtual ~Backend() = default;
    virtual std::string name() const = 0;
    virtual bool supports(const ir::Module&) const = 0;

    // Phase 1: Shared lowering (can use common implementation)
    virtual LoweredPlan lower(const ir::Module&, const CompileOptions&) = 0;

    // Phase 2: Backend-specific emission/build
    virtual std::unique_ptr<Program> compile(const LoweredPlan&, const CompileOptions&) = 0;
};

}
```

### 6.4 Extended Primitives Lowering

How each extended primitive lowers to backend-specific form:

| Primitive | CPU Sim | Ascend NPU | AMD AIE |
|-----------|---------|------------|---------|
| `dispatch_threshold` | Evaluate metric → route to `ReadyQueueSet(pool)` | Select AIV/AIC pool, per-pool dispatch policy | Select mapping strategy (grid/tiling) |
| `pipeline_depth` | Token gates per scope (global/stream/pool) | `in_flight < depth` counters in AICPU executor | FIFO/buffer depth between tiles |
| `task_window` | Sliding window + TensorMap GC | Codegen-time capacity; `stall` enables pipelined build | Host-side descriptor queue bound |
| `batch_deps` | Local batch map → global TensorMap; flush on barriers | "emit → flush → run" structure | Compile-time pass (explicit edges in MLIR) |

### 6.5 Code Generation Templates

Generate thin glue, link against shared runtime:

**Template set:**
- `templates/common/kernel_ids.h.in` — Single source of truth for `KernelId` enum
- `templates/common/task_abi.h.in` — `TaskNodePod`/`TaskIO` layout and pack/unpack helpers
- `templates/ascend/aicpu_orchestrate.cc.in` — Task emission, schedule config, executor launch
- `templates/ascend/aicore_kernel_dispatch.cc.in` — `switch(kernel_id)` dispatch + ABI unpack

**Pattern: X-macro kernel list (avoids duplicate switches)**
```cpp
// Generated: kernel_list.h
#define KERNEL_LIST(X) \
    X(ATTN_KERNEL, attn_kernel, AttnParams) \
    X(MLP_KERNEL, mlp_kernel, MlpParams)

// Usage in dispatch:
switch (task->kernel) {
#define DISPATCH_CASE(id, name, params) case id: name##_wrapper(task); break;
    KERNEL_LIST(DISPATCH_CASE)
#undef DISPATCH_CASE
}
```

**Pattern: Builder API in generated code**
```cpp
// Generated AICPU orchestration uses builder API:
builder.alloc_task(ATTN_KERNEL, ExecDomain::AscendAICore, ExecPool::Cube, stream, ...);
builder.add_input(tensor_region);
builder.add_output(tensor_region);
builder.submit(tid);
// Hard parts (TensorMap deps, fanout wiring, batching) in shared library
```

### 6.6 Share vs Backend-Specific Summary

| Component | Shared | Backend-Specific |
|-----------|--------|------------------|
| `TaskNodePod`, `TaskGraphStorage` | ✓ | |
| `TensorMap`, `ReadyQueueSet` | ✓ (interface + implementations) | |
| Window/Gate/Batcher utilities | ✓ | |
| IR → LoweredPlan lowering | ✓ | |
| Template engine + common templates | ✓ | |
| Thread pool implementation | | CPU sim |
| Handshake protocol, AICPU/AICore build | | Ascend NPU |
| Spatial mapper, MLIR lowering, XRT integration | | AMD AIE |

### 6.7 Header Organization

```
include/pto/rt/
├── graph/
│   ├── types.hpp         # ExecDomain, ExecPool, TaskNodePod, TensorRegion2D
│   ├── storage.hpp       # TaskGraphStorage
│   ├── runtime.hpp       # TaskGraphRuntime, WindowState, IssueGates, DepBatcher
│   ├── tensor_map.hpp    # TensorMap interface + FixedTensorMap, DynamicTensorMap
│   └── ready_queue.hpp   # ReadyQueueSet interface + implementations
├── backend/
│   ├── backend.hpp       # Backend, Program, CompileOptions, LoweredPlan
│   ├── lowering.hpp      # Shared IR → LoweredPlan utilities
│   └── codegen.hpp       # Template engine utilities
└── backend/cpu/
    └── ...               # CPU-specific
└── backend/npu/
    └── ...               # NPU-specific
└── backend/aie/
    └── ...               # AIE-specific
```

---

## 7. Backend Comparison

| Feature | CPU Sim | Ascend NPU | AMD AIE |
|---------|---------|------------|---------|
| **Primary Use** | Development | Production | Spatial workloads |
| **Task Model** | Thread pool | AICPU + AICore | Tile array |
| **Parallelism** | Thread-level | Task-level | Spatial + pipeline |
| **Dependency** | fanin/fanout | Graph executor | Stream-based |
| **Memory** | Host memory | HBM + UB | Tile-local + streams |
| **Scheduling** | Work-stealing | Dispatch policy | Spatial mapping |
| **CSP Support** | Full | Full | Native (streams) |
| **Spatial Support** | Simulated | - | Native |
| **Layout Support** | Simulated | - | Native |
| **Extended Primitives** | All (runtime) | All (AICPU) | batch_deps compile-time only |

---

## 8. Implementation Phases

### Phase 5.1: Shared Infrastructure (R6)
1. Implement `TaskNodePod`, `TaskGraphStorage` in `include/pto/rt/graph/`
2. Implement `TensorMap` (FixedTensorMap + DynamicTensorMap)
3. Implement `ReadyQueueSet` (MPMC + optional SPSC)
4. Implement window/gate/batcher utilities for extended primitives
5. Implement shared IR → LoweredPlan lowering
6. Unit tests for shared infrastructure

### Phase 5.2: CPU Simulation Backend
1. Implement `ThreadPoolExecutor` with dual-queue support
2. Implement `KernelRegistry`
3. Implement `CPUSimBackend` using shared infrastructure
4. Integration tests
5. Performance benchmarks (target: 5000+ tasks/ms)

### Phase 5.3: Ascend NPU Backend
1. Implement `AICPUCodeGen` using templates
2. Implement `AICorewrapperGen` using X-macro pattern
3. Implement handshake protocol
4. Implement `AscendNPUBackend` using shared infrastructure
5. Integration tests

### Phase 5.4 (P1): AMD AIE Backend
1. Implement `SpatialMapper`
2. Implement `LayoutAnalyzer`
3. Implement `MLIRLowering`
4. Implement `AMDAIEBackend`
5. Integration tests

---

## 9. Summary

### 9.1 Backend Interface (Two-Phase)

```cpp
class Backend {
    // Phase 1: Shared lowering
    virtual LoweredPlan lower(const ir::Module&, const CompileOptions&) = 0;

    // Phase 2: Backend-specific emission
    virtual std::unique_ptr<Program> compile(const LoweredPlan&, const CompileOptions&) = 0;

    virtual bool supports(const ir::Module&) const = 0;
    virtual std::string name() const = 0;
};
```

### 9.2 Backend Implementations

| Backend | Class | Namespace |
|---------|-------|-----------|
| CPU Simulation | `CPUSimBackend` | `pto::wsp::backend::cpu` |
| Ascend NPU | `AscendNPUBackend` | `pto::wsp::backend::npu` |
| AMD AIE | `AMDAIEBackend` | `pto::wsp::backend::aie` |

### 9.3 Shared Infrastructure

| Component | Location | Description |
|-----------|----------|-------------|
| `TaskNodePod` | `graph/types.hpp` | Device-copyable task descriptor |
| `TaskGraphStorage` | `graph/storage.hpp` | Immutable built graph |
| `TensorMap` | `graph/tensor_map.hpp` | Dependency inference + GC |
| `ReadyQueueSet` | `graph/ready_queue.hpp` | Multi-queue task scheduling |
| Window/Gate/Batcher | `graph/runtime.hpp` | Extended primitive support |

### 9.4 Priority

| Backend | Priority | Rationale |
|---------|----------|-----------|
| Shared Infrastructure | P0 | Foundation for all backends |
| CPU Simulation | P0 | Development, debugging |
| Ascend NPU | P0 | Production deployment |
| AMD AIE | P1 | Spatial architecture support |

---

*Version: 9.0*
*Last Updated: 2026-01-24*
