// Copyright 2026 PTO-RT Authors
// SPDX-License-Identifier: Apache-2.0

#pragma once  // API-2 FIX: Consistent include guard style
// (was: #ifndef PTO_WSP_BACKEND_CPU_SIM_HPP)

#include "pto/rt/backend/backend.hpp"
#include "pto/rt/graph/graph.hpp"

#include <thread>
#include <atomic>
#include <functional>
#include <unordered_map>
#include <chrono>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <span>  // CPP-3 FIX: For span-based kernel signature

namespace pto::wsp::backend::cpu {

// ============================================================
// Kernel Function Type
// ============================================================

/// Kernel function signature (CPP-3 FIX: Use span for clarity and safety)
using KernelFunc = std::function<void(
    std::span<const uint64_t> args,
    std::span<const graph::TaskIO> io)>;

// ============================================================
// Kernel Registry
// ============================================================

/// Registry for CPU simulation kernels
class KernelRegistry {
public:
    static KernelRegistry& instance();

    void register_kernel(const std::string& name, KernelFunc func);
    void register_kernel(graph::KernelId id, KernelFunc func);

    KernelFunc get_kernel(const std::string& name) const;
    KernelFunc get_kernel(graph::KernelId id) const;

    bool has_kernel(const std::string& name) const;

private:
    KernelRegistry() = default;
    std::unordered_map<std::string, KernelFunc> kernels_;
    std::unordered_map<graph::KernelId, KernelFunc> kernels_by_id_;
};

/// Registration macro
#define REGISTER_CPU_KERNEL(name, func) \
    namespace { \
        static bool _reg_kernel_##name = []() { \
            ::pto::wsp::backend::cpu::KernelRegistry::instance().register_kernel(#name, func); \
            return true; \
        }(); \
    }

// ============================================================
// Thread Pool Executor
// ============================================================

/// Simple thread pool for task execution
class ThreadPoolExecutor {
public:
    explicit ThreadPoolExecutor(int num_threads = 0);
    ~ThreadPoolExecutor();

    /// Start the thread pool
    void start();

    /// Shutdown the thread pool
    void shutdown();

    /// Submit a task for execution
    void submit(std::function<void()> task);

    /// Wait for all submitted tasks to complete
    void wait_all();

    int num_threads() const;

private:
    void worker_loop();

    int num_threads_;
    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> work_queue_;
    std::mutex queue_mutex_;
    std::condition_variable work_available_;
    std::condition_variable completion_cv_;
    std::atomic<int> tasks_pending_{0};
    bool shutdown_requested_ = false;
    bool running_ = false;
};

// ============================================================
// CPU Simulation Program
// ============================================================

/// Compiled program for CPU simulation
class CPUSimProgram : public Program {
public:
    CPUSimProgram(graph::TaskGraphStorage storage, ScheduleRuntimeConfig config, int num_threads);

    void execute() override;
    void execute_async() override;
    void synchronize() override;
    bool is_complete() const override;
    double elapsed_ms() const override;
    ProgramStats stats() const override;
    std::string dump() const override;

    void set_compile_time(double ms);

private:
    void dispatch_ready_tasks();
    void execute_task(graph::TaskId tid, const graph::TaskNodePod& task);

    graph::TaskGraphStorage storage_;
    ScheduleRuntimeConfig config_;
    graph::TaskGraphRuntime runtime_;
    ThreadPoolExecutor executor_;
    int num_threads_;
    std::thread async_thread_;
    std::chrono::duration<double, std::milli> elapsed_{0};
    std::atomic<bool> is_complete_{false};
    double compile_time_ms_ = 0;
};

// ============================================================
// CPU Simulation Backend
// ============================================================

/// CPU simulation backend
class CPUSimBackend : public Backend {
public:
    explicit CPUSimBackend(int num_threads = 0);

    std::string name() const override;
    std::vector<std::string> supported_targets() const override;
    bool supports(const ir::Module& module) const override;
    bool supports(ir::NodeKind kind) const override;

    LoweredPlan lower(const ir::Module& module, const CompileOptions& options) override;
    std::unique_ptr<Program> compile(const LoweredPlan& plan,
                                      const CompileOptions& options) override;

private:
    int num_threads_;
};

// Register CPU simulation backend
REGISTER_BACKEND(CPUSimBackend)

}  // namespace pto::wsp::backend::cpu

