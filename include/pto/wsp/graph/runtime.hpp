// Copyright 2026 PTO-RT Authors
// SPDX-License-Identifier: MIT

#pragma once  // API-2 FIX: Consistent include guard style
// (was: #ifndef PTO_WSP_GRAPH_RUNTIME_HPP)

#include "pto/rt/graph/types.hpp"
#include "pto/rt/graph/storage.hpp"
#include "pto/rt/graph/tensor_map.hpp"
#include "pto/rt/graph/ready_queue.hpp"

#include <memory>
#include <functional>
#include <atomic>
#include <mutex>
#include <thread>
#include <vector>
#include <unordered_map>  // INC-1 FIX: for stream_gates_
#include <chrono>         // L12: for TracePolicy timing

namespace pto::wsp::graph {

// ============================================================
// Window State (task_window primitive support)
// ============================================================

/// Window mode for overflow handling
enum class WindowMode {
    Stall,     // Block until window has space
    Abort,     // Fail with error
    Benchmark, // Report stats but continue
};

/// Task window state for sliding window execution
class WindowState {
public:
    explicit WindowState(size_t size = 8192, WindowMode mode = WindowMode::Stall);

    /// Check if window has capacity
    bool has_capacity() const;

    /// Try to enter window (returns false if full)
    bool try_enter();

    /// Enter window (blocks if stall mode)
    bool enter();

    /// Exit window (task completed)
    void exit();

    size_t window_size() const;
    size_t active_count() const;
    size_t overflow_count() const;

    void set_size(size_t size);
    void set_mode(WindowMode mode);

private:
    size_t window_size_;
    WindowMode mode_;
    std::atomic<size_t> active_count_{0};
    std::atomic<size_t> overflow_count_{0};
};

// ============================================================
// Issue Gates (pipeline_depth primitive support)
// ============================================================

/// Gate scope for pipeline depth control
enum class GateScope {
    Global,    // Single gate for all tasks
    PerStream, // Gate per stream
    PerPool,   // Gate per execution pool
};

/// Issue gate for controlling concurrent in-flight tasks
class IssueGate {
public:
    explicit IssueGate(size_t depth = 2);

    /// Try to acquire a slot (non-blocking)
    bool try_acquire();

    /// Acquire a slot (blocking)
    void acquire();

    /// Release a slot
    void release();

    size_t max_depth() const;
    size_t in_flight() const;
    void set_depth(size_t depth);

private:
    size_t max_depth_;
    std::atomic<size_t> in_flight_{0};
};

/// Collection of issue gates by scope
class IssueGates {
public:
    explicit IssueGates(GateScope scope = GateScope::Global, size_t depth = 2);

    /// Get gate for given context
    IssueGate& get_gate(StreamId stream = 0, ExecPool pool = ExecPool::Any);

    void set_depth(size_t depth);

private:
    IssueGate& get_or_create_stream_gate(StreamId stream);
    IssueGate& get_pool_gate(ExecPool pool);

    GateScope scope_;
    size_t default_depth_;
    std::unique_ptr<IssueGate> global_gate_;
    std::unique_ptr<IssueGate> vector_gate_;
    std::unique_ptr<IssueGate> cube_gate_;
    std::unordered_map<StreamId, std::unique_ptr<IssueGate>> stream_gates_;
    std::mutex mutex_;
};

// ============================================================
// Dependency Batcher (batch_deps primitive support)
// ============================================================

/// Batches dependency resolution for better cache locality
class DepBatcher {
public:
    explicit DepBatcher(size_t threshold = 64);

    /// Add a pending dependency to resolve
    void add_pending(TaskId producer, TaskId consumer);

    /// Check if batch should be flushed
    bool needs_flush() const;

    /// Flush pending dependencies and return resolved edges
    std::vector<std::pair<TaskId, TaskId>> flush();

    size_t threshold() const;
    void set_threshold(size_t threshold);

private:
    size_t threshold_;
    std::vector<std::pair<TaskId, TaskId>> pending_;
    std::atomic<bool> needs_flush_{false};
    std::mutex mutex_;
};

// ============================================================
// Start Policy (L12 - Concurrent Mechanism)
// ============================================================

/// When to start task execution
enum class StartMode {
    AfterOrchestration,  // Wait for full orchestration before starting
    Immediate,           // Start as soon as tasks are ready
    Batched,             // Start in batches of N tasks
};

/// Policy for starting task execution (L12 requirement)
class StartPolicy {
public:
    StartPolicy(StartMode mode = StartMode::AfterOrchestration, size_t batch_size = 0)
        : mode_(mode), batch_size_(batch_size), ready_count_(0) {}

    /// Check if execution should start
    bool should_start(size_t total_tasks) const {
        switch (mode_) {
        case StartMode::AfterOrchestration:
            return ready_count_.load(std::memory_order_acquire) > 0;
        case StartMode::Immediate:
            return true;
        case StartMode::Batched:
            return ready_count_.load(std::memory_order_acquire) >= batch_size_;
        }
        return true;
    }

    /// Increment ready count
    void task_ready() { ready_count_.fetch_add(1, std::memory_order_release); }

    /// Decrement ready count
    void task_started() { ready_count_.fetch_sub(1, std::memory_order_release); }

    StartMode mode() const { return mode_; }
    size_t batch_size() const { return batch_size_; }

    /// Configure policy (for runtime reconfiguration)
    void configure(StartMode mode, size_t batch_size = 0) {
        mode_ = mode;
        batch_size_ = batch_size;
    }

private:
    StartMode mode_;
    size_t batch_size_;
    std::atomic<size_t> ready_count_;
};

// ============================================================
// Trace Policy (L12 - Concurrent Mechanism)
// ============================================================

/// What to trace during execution
enum class TraceLevel {
    None,       // No tracing
    Timing,     // Task start/end times
    Full,       // Full execution trace with deps
};

/// Trace event for post-hoc analysis
struct TraceEvent {
    TaskId task_id;
    uint64_t start_ns;
    uint64_t end_ns;
    size_t worker_id;
    ExecPool pool;
};

/// Policy for tracing execution (L12 requirement)
class TracePolicy {
public:
    explicit TracePolicy(TraceLevel level = TraceLevel::None)
        : level_(level) {}

    /// Check if tracing is enabled
    bool is_enabled() const { return level_ != TraceLevel::None; }

    /// Record task start
    void record_start(TaskId tid, size_t worker_id, ExecPool pool) {
        if (level_ == TraceLevel::None) return;
        std::lock_guard<std::mutex> lock(mutex_);
        current_events_[tid] = TraceEvent{tid, now_ns(), 0, worker_id, pool};
    }

    /// Record task end
    void record_end(TaskId tid) {
        if (level_ == TraceLevel::None) return;
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = current_events_.find(tid);
        if (it != current_events_.end()) {
            it->second.end_ns = now_ns();
            completed_events_.push_back(it->second);
            current_events_.erase(it);
        }
    }

    /// Get completed trace events
    const std::vector<TraceEvent>& events() const { return completed_events_; }

    /// Clear trace
    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        completed_events_.clear();
        current_events_.clear();
    }

    TraceLevel level() const { return level_; }

    /// Configure trace level (for runtime reconfiguration)
    void set_level(TraceLevel level) {
        level_ = level;
    }

private:
    static uint64_t now_ns() {
        auto now = std::chrono::high_resolution_clock::now();
        return std::chrono::duration_cast<std::chrono::nanoseconds>(
            now.time_since_epoch()).count();
    }

    TraceLevel level_;
    std::mutex mutex_;
    std::unordered_map<TaskId, TraceEvent> current_events_;
    std::vector<TraceEvent> completed_events_;
};

// ============================================================
// Task Graph Runtime (Host-Scheduler State)
// ============================================================

/// Runtime state for task graph execution
class TaskGraphRuntime {
public:
    explicit TaskGraphRuntime(TaskGraphStorage& storage);

    /// Initialize runtime from storage
    void initialize();

    /// Mark task complete and propagate to dependents
    void task_complete(TaskId tid);

    /// Try to get next ready task
    std::optional<TaskId> try_get_ready(ExecPool pool = ExecPool::Any);

    /// Check if all tasks are complete
    bool all_complete() const;

    // Access to components
    TensorMap& tensor_map();
    ReadyQueueSet& ready_queues();
    WindowState& window();
    IssueGates& gates();
    DepBatcher& batcher();
    StartPolicy& start_policy();
    TracePolicy& trace_policy();

    // Stats
    size_t completed_count() const;

    // Configure runtime
    void set_tensor_map(std::unique_ptr<TensorMap> map);
    void set_ready_queues(std::unique_ptr<ReadyQueueSet> queues);
    void configure_start_policy(StartMode mode, size_t batch_size = 0);
    void configure_trace_policy(TraceLevel level);

private:
    TaskGraphStorage& storage_;
    std::unique_ptr<TensorMap> tensor_map_;
    std::unique_ptr<ReadyQueueSet> ready_queues_;
    WindowState window_;
    IssueGates gates_;
    DepBatcher batcher_;
    StartPolicy start_policy_;
    TracePolicy trace_policy_;
    std::atomic<size_t> completed_count_{0};

    // BUG-2 FIX: Thread-safe fanin counters (separate from immutable storage)
    std::vector<std::atomic<int32_t>> fanin_remaining_;
};

}  // namespace pto::wsp::graph
