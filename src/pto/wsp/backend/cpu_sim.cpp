// PTO Workload-Schedule Programming (PTO-WSP) framework v9 - CPU Simulation Backend Implementation
// Copyright 2026 PTO-RT Authors
// SPDX-License-Identifier: Apache-2.0

#include "pto/rt/backend/cpu_sim.hpp"

namespace pto::wsp::backend::cpu {

// ============================================================
// KernelRegistry Implementation
// ============================================================

KernelRegistry& KernelRegistry::instance() {
    static KernelRegistry registry;
    return registry;
}

void KernelRegistry::register_kernel(const std::string& name, KernelFunc func) {
    kernels_[name] = std::move(func);
}

void KernelRegistry::register_kernel(graph::KernelId id, KernelFunc func) {
    kernels_by_id_[id] = std::move(func);
}

KernelFunc KernelRegistry::get_kernel(const std::string& name) const {
    auto it = kernels_.find(name);
    return (it != kernels_.end()) ? it->second : nullptr;
}

KernelFunc KernelRegistry::get_kernel(graph::KernelId id) const {
    auto it = kernels_by_id_.find(id);
    return (it != kernels_by_id_.end()) ? it->second : nullptr;
}

bool KernelRegistry::has_kernel(const std::string& name) const {
    return kernels_.find(name) != kernels_.end();
}

// ============================================================
// ThreadPoolExecutor Implementation
// ============================================================

ThreadPoolExecutor::ThreadPoolExecutor(int num_threads)
    : num_threads_(num_threads > 0 ? num_threads
                                    : static_cast<int>(std::thread::hardware_concurrency())) {
    // Workers are started on demand
}

ThreadPoolExecutor::~ThreadPoolExecutor() {
    shutdown();
}

void ThreadPoolExecutor::start() {
    if (running_) return;
    running_ = true;
    shutdown_requested_ = false;

    workers_.reserve(num_threads_);
    for (int i = 0; i < num_threads_; ++i) {
        workers_.emplace_back([this]() { worker_loop(); });
    }
}

void ThreadPoolExecutor::shutdown() {
    if (!running_) return;

    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        shutdown_requested_ = true;
    }
    work_available_.notify_all();

    for (auto& worker : workers_) {
        if (worker.joinable()) {
            worker.join();
        }
    }
    workers_.clear();
    running_ = false;
}

void ThreadPoolExecutor::submit(std::function<void()> task) {
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        work_queue_.push(std::move(task));
        tasks_pending_.fetch_add(1, std::memory_order_release);
    }
    work_available_.notify_one();
}

void ThreadPoolExecutor::wait_all() {
    std::unique_lock<std::mutex> lock(queue_mutex_);
    completion_cv_.wait(lock, [this] {
        return tasks_pending_.load(std::memory_order_acquire) == 0;
    });
}

int ThreadPoolExecutor::num_threads() const { return num_threads_; }

void ThreadPoolExecutor::worker_loop() {
    while (true) {
        std::function<void()> task;

        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            work_available_.wait(lock, [this] {
                return shutdown_requested_ || !work_queue_.empty();
            });

            if (shutdown_requested_ && work_queue_.empty()) {
                return;
            }

            task = std::move(work_queue_.front());
            work_queue_.pop();
        }

        // Execute task outside lock
        task();

        // Signal completion
        if (tasks_pending_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
            completion_cv_.notify_all();
        }
    }
}

// ============================================================
// CPUSimProgram Implementation
// ============================================================

CPUSimProgram::CPUSimProgram(graph::TaskGraphStorage storage, ScheduleRuntimeConfig config, int num_threads)
    : storage_(std::move(storage)),
      config_(std::move(config)),
      runtime_(storage_),
      executor_(num_threads),
      num_threads_(num_threads > 0 ? num_threads
                                   : static_cast<int>(std::thread::hardware_concurrency())) {}

void CPUSimProgram::execute() {
    auto start = std::chrono::high_resolution_clock::now();

    // Initialize runtime
    runtime_.initialize();
    runtime_.window().set_size(config_.window_size);
    runtime_.window().set_mode(config_.window_mode);
    runtime_.gates().set_depth(config_.pipeline_depth);
    runtime_.batcher().set_threshold(config_.batch_threshold);

    // Configure queue type
    if (config_.dual_queue_enabled) {
        runtime_.set_ready_queues(graph::make_dual_queue_set());
    }

    // Start executor
    executor_.start();

    // Initial dispatch of ready tasks
    dispatch_ready_tasks();

    // Wait for completion
    while (!runtime_.all_complete()) {
        std::this_thread::yield();
    }

    executor_.shutdown();

    auto end = std::chrono::high_resolution_clock::now();
    elapsed_ = std::chrono::duration<double, std::milli>(end - start);
    is_complete_ = true;
}

void CPUSimProgram::execute_async() {
    async_thread_ = std::thread([this]() { execute(); });
}

void CPUSimProgram::synchronize() {
    if (async_thread_.joinable()) {
        async_thread_.join();
    }
}

bool CPUSimProgram::is_complete() const {
    return is_complete_.load(std::memory_order_acquire);
}

double CPUSimProgram::elapsed_ms() const {
    return elapsed_.count();
}

ProgramStats CPUSimProgram::stats() const {
    return ProgramStats{
        .num_tasks = storage_.num_tasks(),
        .num_streams = config_.num_streams,
        .num_executors = static_cast<size_t>(num_threads_),
        .compile_time_ms = compile_time_ms_,
        .execute_time_ms = elapsed_.count(),
        .peak_memory_bytes = 0,  // Not tracked
        .total_edges = storage_.num_edges(),
    };
}

std::string CPUSimProgram::dump() const {
    std::ostringstream oss;
    oss << "CPUSimProgram:\n";
    oss << "  Tasks: " << storage_.num_tasks() << "\n";
    oss << "  Edges: " << storage_.num_edges() << "\n";
    oss << "  Threads: " << num_threads_ << "\n";
    oss << "  Window size: " << config_.window_size << "\n";
    oss << "  Pipeline depth: " << config_.pipeline_depth << "\n";
    return oss.str();
}

void CPUSimProgram::set_compile_time(double ms) { compile_time_ms_ = ms; }

void CPUSimProgram::dispatch_ready_tasks() {
    // Dispatch all currently ready tasks
    while (auto tid = runtime_.try_get_ready()) {
        const auto task_id = *tid;

        // BUG-1 FIX: Capture only tid, fetch task inside lambda to avoid dangling reference
        executor_.submit([this, task_id]() {
            const auto& task = storage_.get_task(task_id);
            execute_task(task_id, task);
        });
    }
}

void CPUSimProgram::execute_task(graph::TaskId tid, const graph::TaskNodePod& task) {
    // Look up kernel
    auto kernel = KernelRegistry::instance().get_kernel(task.kernel);

    if (kernel) {
        // Execute the kernel (CPP-3 FIX: Use spans instead of pointer+count)
        kernel(std::span<const uint64_t>(task.args, task.num_u64_args),
               std::span<const graph::TaskIO>(task.io, task.num_io));
    }
    // If no kernel registered, task is a no-op (for testing)

    // Mark complete and propagate
    runtime_.task_complete(tid);

    // Dispatch any newly ready tasks
    dispatch_ready_tasks();
}

// ============================================================
// Workload Lowering
// ============================================================

/// Helper class that walks workload IR and generates concrete tasks.
/// For static Dense axes, it fully expands loops into individual tasks.
/// For dynamic axes (DenseDyn, Ragged, Sparse), it creates placeholder tasks
/// since actual expansion requires runtime values.
class WorkloadLowerer {
public:
    WorkloadLowerer(graph::TaskGraphStorage& storage, const ir::Module& module)
        : storage_(storage), builder_(storage) {
        // Register all kernels from workload definitions
        for (const auto& workload : module.workloads) {
            if (workload.body) {
                register_kernels(workload.body);
            }
        }
    }

    /// Lower all workloads in the module to tasks
    void lower(const ir::Module& module) {
        for (const auto& workload : module.workloads) {
            if (workload.body) {
                // Create empty binding context for top-level
                std::unordered_map<std::string, int64_t> bindings;
                lower_workload(workload.body, bindings);
            }
        }
        builder_.finalize();
    }

private:
    graph::TaskGraphStorage& storage_;
    graph::TaskGraphBuilder builder_;
    std::vector<graph::TaskId> last_tasks_;  // For sequential dependencies

    /// Register all kernel names found in workload tree
    void register_kernels(const ir::IRPtr<ir::WorkloadNode>& node) {
        if (!node) return;

        if (node->kind == ir::NodeKind::Task) {
            auto task = std::static_pointer_cast<const ir::TaskNode>(node);
            graph::KernelInfo info{
                task->kernel_name, task->kernel_name, 0, 0,
                graph::ExecDomain::HostCPU, graph::ExecPool::Any
            };
            storage_.kernel_bundle().register_kernel(info);
        }

        // Recurse into children
        node->forEachChild([this](const ir::IRPtr<ir::IRNode>& child) {
            auto workload = std::dynamic_pointer_cast<const ir::WorkloadNode>(child);
            if (workload) {
                register_kernels(workload);
            }
        });
    }

    /// Lower a workload node with current loop bindings
    void lower_workload(const ir::IRPtr<ir::WorkloadNode>& node,
                        std::unordered_map<std::string, int64_t>& bindings) {
        if (!node) return;

        switch (node->kind) {
            case ir::NodeKind::Task:
                lower_task(std::static_pointer_cast<const ir::TaskNode>(node), bindings);
                break;

            case ir::NodeKind::ParallelFor:
                lower_parallel_for(
                    std::static_pointer_cast<const ir::ParallelForNode>(node), bindings);
                break;

            case ir::NodeKind::ForEach:
                lower_for_each(
                    std::static_pointer_cast<const ir::ForEachNode>(node), bindings);
                break;

            case ir::NodeKind::Combine:
                lower_combine(
                    std::static_pointer_cast<const ir::CombineNode>(node), bindings);
                break;

            case ir::NodeKind::Sequential:
                lower_sequential(
                    std::static_pointer_cast<const ir::SequentialNode>(node), bindings);
                break;

            case ir::NodeKind::Select:
            case ir::NodeKind::Cond:
            case ir::NodeKind::Call:
                // These require more complex handling - skip for now
                break;

            default:
                break;
        }
    }

    /// Lower a task node - creates a concrete task in the graph
    void lower_task(const ir::IRPtr<ir::TaskNode>& task,
                    const std::unordered_map<std::string, int64_t>& bindings) {
        auto kernel_id = storage_.kernel_bundle().find_kernel(task->kernel_name);
        if (!kernel_id) return;

        builder_.begin_task(*kernel_id);

        // Add loop index values as arguments
        for (const auto& param : task->params) {
            auto it = bindings.find(param);
            if (it != bindings.end()) {
                builder_.add_arg(static_cast<uint64_t>(it->second));
            }
        }

        graph::TaskId tid = builder_.submit();
        last_tasks_.push_back(tid);
    }

    /// Lower parallel_for - expand loop into parallel tasks
    void lower_parallel_for(const ir::IRPtr<ir::ParallelForNode>& pf,
                            std::unordered_map<std::string, int64_t>& bindings) {
        // Get loop bounds from axis
        int64_t size = get_axis_size(pf->axis);
        if (size <= 0) return;  // Dynamic axis - can't expand statically

        // Expand loop - all iterations are independent (parallel)
        for (int64_t i = 0; i < size; ++i) {
            bindings[pf->index_var] = i;
            lower_workload(pf->body, bindings);
        }
        bindings.erase(pf->index_var);
    }

    /// Lower for_each - expand loop into sequential tasks
    void lower_for_each(const ir::IRPtr<ir::ForEachNode>& fe,
                        std::unordered_map<std::string, int64_t>& bindings) {
        int64_t size = get_axis_size(fe->axis);
        if (size <= 0) return;

        // Expand loop - iterations are sequential (add dependencies)
        graph::TaskId prev_task = graph::INVALID_TASK_ID;
        for (int64_t i = 0; i < size; ++i) {
            bindings[fe->index_var] = i;
            size_t before = last_tasks_.size();
            lower_workload(fe->body, bindings);
            size_t after = last_tasks_.size();

            // Add dependency from previous iteration's tasks to this iteration's tasks
            if (prev_task != graph::INVALID_TASK_ID && after > before) {
                for (size_t j = before; j < after; ++j) {
                    builder_.add_dependency(prev_task, last_tasks_[j]);
                }
            }
            if (after > before) {
                prev_task = last_tasks_[after - 1];
            }
        }
        bindings.erase(fe->index_var);
    }

    /// Lower combine - all children are independent
    void lower_combine(const ir::IRPtr<ir::CombineNode>& combine,
                       std::unordered_map<std::string, int64_t>& bindings) {
        for (const auto& child : combine->workloads) {
            lower_workload(child, bindings);
        }
    }

    /// Lower sequential - add dependencies between children
    void lower_sequential(const ir::IRPtr<ir::SequentialNode>& seq,
                          std::unordered_map<std::string, int64_t>& bindings) {
        std::vector<graph::TaskId> prev_tasks;

        for (const auto& child : seq->workloads) {
            size_t before = last_tasks_.size();
            lower_workload(child, bindings);
            size_t after = last_tasks_.size();

            // Add dependencies from previous step's tasks to this step's tasks
            if (!prev_tasks.empty() && after > before) {
                for (graph::TaskId prev : prev_tasks) {
                    for (size_t j = before; j < after; ++j) {
                        builder_.add_dependency(prev, last_tasks_[j]);
                    }
                }
            }

            // Update prev_tasks for next iteration
            prev_tasks.clear();
            for (size_t j = before; j < after; ++j) {
                prev_tasks.push_back(last_tasks_[j]);
            }
        }
    }

    /// Get static size from axis, or -1 if dynamic
    int64_t get_axis_size(const ir::IRPtr<ir::AxisNode>& axis) {
        if (!axis) return -1;

        if (axis->kind == ir::NodeKind::DenseAxis) {
            auto dense = std::static_pointer_cast<const ir::DenseAxisNode>(axis);
            return dense->size;
        }

        // Dynamic axes - can't expand statically
        return -1;
    }
};

// ============================================================
// CPUSimBackend Implementation
// ============================================================

CPUSimBackend::CPUSimBackend(int num_threads)
    : num_threads_(num_threads) {}

std::string CPUSimBackend::name() const { return "cpu_sim"; }

std::vector<std::string> CPUSimBackend::supported_targets() const {
    return {"cpu_sim", "cpu"};
}

bool CPUSimBackend::supports(const ir::Module& /*module*/) const {
    return true;  // CPU sim supports everything
}

bool CPUSimBackend::supports(ir::NodeKind /*kind*/) const {
    return true;  // CPU sim supports all node kinds
}

/// Lower IR Module to a LoweredPlan for CPU simulation.
///
/// This function transforms the high-level workload IR into a task graph
/// that can be executed by the CPU simulation runtime. The lowering process:
///
/// 1. Creates a LoweredPlan with schedule configuration from options
/// 2. Registers kernel metadata from workload definitions
/// 3. Expands workload IR (parallel_for, for_each, task) into concrete tasks
///
/// @param module The IR module containing workload and schedule definitions
/// @param options Compile options controlling window size, pipeline depth, etc.
/// @return LoweredPlan ready for compilation into an executable Program
///
/// @note Static Dense axes are fully expanded. Dynamic axes (DenseDyn, Ragged,
///       Sparse) cannot be expanded statically and require runtime values.
LoweredPlan CPUSimBackend::lower(const ir::Module& module,
                                  const CompileOptions& options) {
    LoweredPlan plan;
    plan.workload_name = module.name;

    // Configure schedule runtime
    plan.sched_config.window_size = options.task_window_size;
    plan.sched_config.window_mode = options.task_window_mode;
    plan.sched_config.pipeline_depth = options.pipeline_depth;
    plan.sched_config.gate_scope = options.gate_scope;
    plan.sched_config.batch_threshold = options.batch_deps_threshold;
    plan.sched_config.num_streams = options.num_streams;
    plan.sched_config.dual_queue_enabled = false;  // CPU sim uses single queue by default

    // Use WorkloadLowerer to expand IR into concrete tasks
    WorkloadLowerer lowerer(plan.graph, module);
    lowerer.lower(module);

    return plan;
}

/// Compile a LoweredPlan into an executable CPUSimProgram.
///
/// Creates a CPUSimProgram that wraps the task graph with a thread pool
/// executor. The program can be executed multiple times with different inputs.
///
/// @param plan The lowered plan containing task graph and schedule config
/// @param options Compile options (num_threads used if > 0)
/// @return Executable program that can run on CPU threads
std::unique_ptr<Program> CPUSimBackend::compile(const LoweredPlan& plan,
                                                 const CompileOptions& options) {
    auto start = std::chrono::high_resolution_clock::now();

    int threads = (options.num_threads > 0) ? options.num_threads : num_threads_;

    auto program = std::make_unique<CPUSimProgram>(
        plan.graph, plan.sched_config, threads);

    auto end = std::chrono::high_resolution_clock::now();
    auto compile_time = std::chrono::duration<double, std::milli>(end - start);
    program->set_compile_time(compile_time.count());

    return program;
}

}  // namespace pto::wsp::backend::cpu
