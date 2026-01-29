// Copyright 2026 PTO-RT Authors
// SPDX-License-Identifier: MIT

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
#include <optional>
#include <memory>

namespace pto::wsp::backend::cpu {

// ============================================================
// CSP Runtime: BoundedChannel
// ============================================================

/// Thread-safe bounded channel for CSP communication.
/// Supports blocking send/recv with proper backpressure.
/// Capacity 0 = rendezvous (synchronous handoff).
template<typename T>
class BoundedChannel {
public:
    explicit BoundedChannel(size_t capacity) : capacity_(capacity) {}

    /// Blocking send - waits if buffer is full
    /// Returns false if channel is closed
    bool send(T value) {
        std::unique_lock<std::mutex> lock(mutex_);

        // Wait for space (or close)
        if (capacity_ == 0) {
            // Rendezvous: wait for receiver
            ++waiting_senders_;
            sender_cv_.notify_one();  // Signal that sender is ready
            not_full_.wait(lock, [this] {
                return closed_ || waiting_receivers_ > 0;
            });
            --waiting_senders_;

            if (closed_) return false;

            // Direct handoff
            rendezvous_value_ = std::move(value);
            has_rendezvous_ = true;
            receiver_cv_.notify_one();

            // Wait for receiver to take it
            not_full_.wait(lock, [this] { return !has_rendezvous_ || closed_; });
            return !closed_;
        } else {
            // Buffered: wait for space
            not_full_.wait(lock, [this] {
                return closed_ || buffer_.size() < capacity_;
            });

            if (closed_) return false;

            buffer_.push(std::move(value));
            not_empty_.notify_one();
            return true;
        }
    }

    /// Blocking receive - waits if buffer is empty
    /// Returns nullopt if channel is closed and empty
    std::optional<T> recv() {
        std::unique_lock<std::mutex> lock(mutex_);

        if (capacity_ == 0) {
            // Rendezvous: wait for sender
            ++waiting_receivers_;
            not_full_.notify_one();  // Signal that receiver is ready
            receiver_cv_.wait(lock, [this] {
                return closed_ || has_rendezvous_;
            });
            --waiting_receivers_;

            if (closed_ && !has_rendezvous_) return std::nullopt;

            T value = std::move(rendezvous_value_);
            has_rendezvous_ = false;
            not_full_.notify_one();  // Signal sender can continue
            return value;
        } else {
            // Buffered: wait for data
            not_empty_.wait(lock, [this] {
                return closed_ || !buffer_.empty();
            });

            if (buffer_.empty()) return std::nullopt;

            T value = std::move(buffer_.front());
            buffer_.pop();
            not_full_.notify_one();
            return value;
        }
    }

    /// Non-blocking send attempt
    bool try_send(T value) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (closed_) return false;

        if (capacity_ == 0) {
            // Rendezvous can't be non-blocking
            return false;
        }

        if (buffer_.size() >= capacity_) return false;

        buffer_.push(std::move(value));
        not_empty_.notify_one();
        return true;
    }

    /// Non-blocking receive attempt
    std::optional<T> try_recv() {
        std::lock_guard<std::mutex> lock(mutex_);

        if (capacity_ == 0) {
            // Rendezvous can't be non-blocking
            return std::nullopt;
        }

        if (buffer_.empty()) return std::nullopt;

        T value = std::move(buffer_.front());
        buffer_.pop();
        not_full_.notify_one();
        return value;
    }

    /// Close the channel - no more sends allowed
    void close() {
        std::lock_guard<std::mutex> lock(mutex_);
        closed_ = true;
        not_empty_.notify_all();
        not_full_.notify_all();
        sender_cv_.notify_all();
        receiver_cv_.notify_all();
    }

    /// Check if channel is closed
    bool is_closed() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return closed_;
    }

    /// Check if channel is empty
    bool empty() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return buffer_.empty();
    }

    /// Get current buffer size
    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return buffer_.size();
    }

    size_t capacity() const { return capacity_; }

private:
    std::queue<T> buffer_;
    size_t capacity_;
    mutable std::mutex mutex_;
    std::condition_variable not_full_;
    std::condition_variable not_empty_;
    std::condition_variable sender_cv_;
    std::condition_variable receiver_cv_;
    bool closed_ = false;

    // For rendezvous (capacity == 0)
    T rendezvous_value_;
    bool has_rendezvous_ = false;
    size_t waiting_senders_ = 0;
    size_t waiting_receivers_ = 0;
};

// ============================================================
// CSP Runtime: ChannelRegistry
// ============================================================

/// Registry for named channels used by CSP processes.
/// Channels carry TaskIds to represent work items flowing through pipeline.
class ChannelRegistry {
public:
    using ChannelPtr = std::shared_ptr<BoundedChannel<graph::TaskId>>;

    /// Create a new channel with given capacity
    void create_channel(const std::string& name, size_t capacity) {
        std::lock_guard<std::mutex> lock(mutex_);
        channels_[name] = std::make_shared<BoundedChannel<graph::TaskId>>(capacity);
    }

    /// Get channel by name (returns nullptr if not found)
    ChannelPtr get_channel(const std::string& name) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = channels_.find(name);
        return (it != channels_.end()) ? it->second : nullptr;
    }

    /// Check if channel exists
    bool has_channel(const std::string& name) const {
        std::lock_guard<std::mutex> lock(mutex_);
        return channels_.find(name) != channels_.end();
    }

    /// Close all channels
    void close_all() {
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto& [name, ch] : channels_) {
            ch->close();
        }
    }

    /// Clear all channels
    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        channels_.clear();
    }

private:
    std::unordered_map<std::string, ChannelPtr> channels_;
    mutable std::mutex mutex_;
};

// ============================================================
// CSP Runtime: Process Info
// ============================================================

/// Information about a CSP process for runtime execution.
struct ProcessInfo {
    std::string name;
    std::vector<std::string> consumes;  // Input channel names
    std::vector<std::string> produces;  // Output channel names
    // Process body is executed as tasks
};

// ============================================================
// CSP Runtime: CSP Executor
// ============================================================

/// Executor for CSP (Communicating Sequential Processes) pipelines.
/// Runs processes concurrently with channel-based communication.
class CSPExecutor {
public:
    explicit CSPExecutor(ChannelRegistry& channels, int num_threads = 0);
    ~CSPExecutor();

    /// Add a producer process that generates work items
    void add_producer(const std::string& name,
                      const std::vector<std::string>& output_channels,
                      std::function<void()> body);

    /// Add a consumer process that reads from channels and processes items
    void add_consumer(const std::string& name,
                      const std::vector<std::string>& input_channels,
                      const std::vector<std::string>& output_channels,
                      std::function<void(graph::TaskId)> body);

    /// Add a terminal consumer (no output channels)
    void add_sink(const std::string& name,
                  const std::vector<std::string>& input_channels,
                  std::function<void(graph::TaskId)> body);

    /// Start all processes
    void start();

    /// Wait for all processes to complete
    void wait();

    /// Signal that no more work will be produced (closes input channels)
    void signal_done();

    /// Check if all processes are complete
    bool is_complete() const;

private:
    ChannelRegistry& channels_;
    int num_threads_;
    std::vector<std::thread> process_threads_;
    std::atomic<bool> shutdown_{false};
    std::atomic<int> active_processes_{0};
    std::mutex mutex_;
    std::condition_variable cv_;
};

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
