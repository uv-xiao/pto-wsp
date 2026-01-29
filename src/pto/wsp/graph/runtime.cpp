// PTO Workload-Schedule Programming (PTO-WSP) framework v9 - Task Graph Runtime Implementation
// Copyright 2026 PTO-RT Authors
// SPDX-License-Identifier: MIT

#include "pto/rt/graph/runtime.hpp"

namespace pto::wsp::graph {

// ============================================================
// WindowState Implementation
// ============================================================

/// WindowState manages a sliding window over active tasks.
/// Used for task_window scheduling to limit concurrent task metadata.
///
/// Three overflow modes:
/// - Stall: Block until window has capacity (default)
/// - Abort: Return false immediately on overflow
/// - Benchmark: Allow overflow but count occurrences

WindowState::WindowState(size_t size, WindowMode mode)
    : window_size_(size), mode_(mode) {}

/// Check if there's capacity in the window without blocking.
bool WindowState::has_capacity() const {
    return active_count_.load(std::memory_order_acquire) < window_size_;
}

/// Try to enter the window (non-blocking).
/// @return true if entered successfully, false if window is full
bool WindowState::try_enter() {
    size_t current = active_count_.load(std::memory_order_acquire);
    while (current < window_size_) {
        if (active_count_.compare_exchange_weak(
                current, current + 1,
                std::memory_order_acq_rel,
                std::memory_order_acquire)) {
            return true;
        }
    }
    return false;
}

bool WindowState::enter() {
    while (!try_enter()) {
        switch (mode_) {
        case WindowMode::Stall:
            std::this_thread::yield();
            break;
        case WindowMode::Abort:
            return false;
        case WindowMode::Benchmark:
            overflow_count_++;
            active_count_.fetch_add(1, std::memory_order_acq_rel);
            return true;
        }
    }
    return true;
}

void WindowState::exit() {
    active_count_.fetch_sub(1, std::memory_order_acq_rel);
}

size_t WindowState::window_size() const { return window_size_; }
size_t WindowState::active_count() const { return active_count_.load(std::memory_order_acquire); }
size_t WindowState::overflow_count() const { return overflow_count_; }

void WindowState::set_size(size_t size) { window_size_ = size; }
void WindowState::set_mode(WindowMode mode) { mode_ = mode; }

// ============================================================
// IssueGate Implementation
// ============================================================

/// IssueGate limits the number of in-flight tasks to control pipeline depth.
/// Used for pipeline_depth scheduling to prevent task queue overflow.
/// This is a counting semaphore with configurable maximum depth.

IssueGate::IssueGate(size_t depth) : max_depth_(depth) {}

/// Try to acquire a slot (non-blocking).
/// @return true if acquired, false if at maximum depth
bool IssueGate::try_acquire() {
    size_t current = in_flight_.load(std::memory_order_acquire);
    while (current < max_depth_) {
        if (in_flight_.compare_exchange_weak(
                current, current + 1,
                std::memory_order_acq_rel,
                std::memory_order_acquire)) {
            return true;
        }
    }
    return false;
}

void IssueGate::acquire() {
    while (!try_acquire()) {
        std::this_thread::yield();
    }
}

void IssueGate::release() {
    in_flight_.fetch_sub(1, std::memory_order_acq_rel);
}

size_t IssueGate::max_depth() const { return max_depth_; }
size_t IssueGate::in_flight() const { return in_flight_.load(std::memory_order_acquire); }
void IssueGate::set_depth(size_t depth) { max_depth_ = depth; }

// ============================================================
// IssueGates Implementation
// ============================================================

/// IssueGates manages multiple IssueGate instances based on gate scope.
/// Three scoping strategies:
/// - Global: Single gate shared by all tasks
/// - PerStream: Separate gate per stream ID (for stream-level backpressure)
/// - PerPool: Separate gates for Vector and Cube execution pools

IssueGates::IssueGates(GateScope scope, size_t depth)
    : scope_(scope), default_depth_(depth),
      global_gate_(std::make_unique<IssueGate>(depth)),
      vector_gate_(std::make_unique<IssueGate>(depth)),
      cube_gate_(std::make_unique<IssueGate>(depth)) {}

IssueGate& IssueGates::get_gate(StreamId stream, ExecPool pool) {
    switch (scope_) {
    case GateScope::Global:
        return *global_gate_;
    case GateScope::PerStream:
        return get_or_create_stream_gate(stream);
    case GateScope::PerPool:
        return get_pool_gate(pool);
    }
    return *global_gate_;
}

void IssueGates::set_depth(size_t depth) {
    default_depth_ = depth;
    global_gate_->set_depth(depth);
    for (auto& [_, gate] : stream_gates_) {
        gate->set_depth(depth);
    }
    vector_gate_->set_depth(depth);
    cube_gate_->set_depth(depth);
}

IssueGate& IssueGates::get_or_create_stream_gate(StreamId stream) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = stream_gates_.find(stream);
    if (it == stream_gates_.end()) {
        auto [inserted_it, success] = stream_gates_.emplace(
            stream, std::make_unique<IssueGate>(default_depth_));
        it = inserted_it;
    }
    return *it->second;
}

IssueGate& IssueGates::get_pool_gate(ExecPool pool) {
    switch (pool) {
    case ExecPool::Vector:
        return *vector_gate_;
    case ExecPool::Cube:
        return *cube_gate_;
    case ExecPool::Any:
    default:
        return *global_gate_;
    }
}

// ============================================================
// DepBatcher Implementation
// ============================================================

/// DepBatcher batches dependency resolutions to amortize overhead.
/// Instead of resolving each dependency individually, it collects
/// producer→consumer pairs and flushes them in batches when threshold
/// is reached. This reduces lock contention in high-task-count scenarios.

DepBatcher::DepBatcher(size_t threshold) : threshold_(threshold) {}

void DepBatcher::add_pending(TaskId producer, TaskId consumer) {
    std::lock_guard<std::mutex> lock(mutex_);
    pending_.push_back({producer, consumer});
    if (pending_.size() >= threshold_) {
        needs_flush_ = true;
    }
}

bool DepBatcher::needs_flush() const {
    return needs_flush_.load(std::memory_order_acquire);
}

std::vector<std::pair<TaskId, TaskId>> DepBatcher::flush() {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<std::pair<TaskId, TaskId>> result;
    result.swap(pending_);
    needs_flush_.store(false, std::memory_order_release);
    return result;
}

size_t DepBatcher::threshold() const { return threshold_; }
void DepBatcher::set_threshold(size_t threshold) { threshold_ = threshold; }

// ============================================================
// TaskGraphRuntime Implementation
// ============================================================

/// TaskGraphRuntime manages execution state for a task graph.
/// It tracks:
/// - Task completion status (remaining dependency counts)
/// - Ready queue for tasks whose dependencies are satisfied
/// - Completion count for termination detection
///
/// Usage pattern:
/// 1. initialize() - populate ready queue with root tasks (fanin=0)
/// 2. try_get_ready() - get next ready task
/// 3. task_complete() - mark task done and propagate to dependents
/// 4. all_complete() - check if graph execution is finished

TaskGraphRuntime::TaskGraphRuntime(TaskGraphStorage& storage)
    : storage_(storage),
      tensor_map_(std::make_unique<DynamicTensorMap>()),
      ready_queues_(std::make_unique<SingleQueueSet>()) {}

void TaskGraphRuntime::initialize() {
    tensor_map_->clear();

    DependencyAnalyzer analyzer(*tensor_map_);

    for (const auto& task : storage_.tasks()) {
        analyzer.register_outputs(task);
    }

    // BUG-2 FIX: Initialize thread-safe fanin counters from storage
    // std::atomic is not copyable, so we resize and then store values
    const size_t num_tasks = storage_.num_tasks();
    fanin_remaining_ = std::vector<std::atomic<int32_t>>(num_tasks);
    for (size_t i = 0; i < num_tasks; ++i) {
        fanin_remaining_[i].store(
            storage_.get_task(static_cast<TaskId>(i)).fanin,
            std::memory_order_relaxed);
    }

    for (TaskId tid : storage_.ready_tasks()) {
        const auto& task = storage_.get_task(tid);
        ready_queues_->push(task.pool, tid);
    }

    completed_count_.store(0, std::memory_order_relaxed);
}

void TaskGraphRuntime::task_complete(TaskId tid) {
    // CPP-4 FIX: Use span instead of allocating vector
    auto fanout = storage_.fanout_span(tid);

    // BUG-2 FIX: Use atomic decrement for thread-safe fanin tracking
    for (TaskId dep_tid : fanout) {
        // Atomically decrement fanin; if it reaches 0, task is ready
        if (fanin_remaining_[dep_tid].fetch_sub(1, std::memory_order_acq_rel) == 1) {
            const auto& task = storage_.get_task(dep_tid);
            ready_queues_->push(task.pool, dep_tid);
        }
    }

    window_.exit();
    completed_count_.fetch_add(1, std::memory_order_release);
}

std::optional<TaskId> TaskGraphRuntime::try_get_ready(ExecPool pool) {
    if (!window_.try_enter()) {
        return std::nullopt;
    }

    auto tid = ready_queues_->try_pop(pool);
    if (!tid) {
        window_.exit();
    }
    return tid;
}

bool TaskGraphRuntime::all_complete() const {
    return completed_count_.load(std::memory_order_acquire) >= storage_.num_tasks();
}

TensorMap& TaskGraphRuntime::tensor_map() { return *tensor_map_; }
ReadyQueueSet& TaskGraphRuntime::ready_queues() { return *ready_queues_; }
WindowState& TaskGraphRuntime::window() { return window_; }
IssueGates& TaskGraphRuntime::gates() { return gates_; }
DepBatcher& TaskGraphRuntime::batcher() { return batcher_; }

size_t TaskGraphRuntime::completed_count() const {
    return completed_count_.load(std::memory_order_acquire);
}

void TaskGraphRuntime::set_tensor_map(std::unique_ptr<TensorMap> map) {
    tensor_map_ = std::move(map);
}

void TaskGraphRuntime::set_ready_queues(std::unique_ptr<ReadyQueueSet> queues) {
    ready_queues_ = std::move(queues);
}

StartPolicy& TaskGraphRuntime::start_policy() { return start_policy_; }
TracePolicy& TaskGraphRuntime::trace_policy() { return trace_policy_; }

void TaskGraphRuntime::configure_start_policy(StartMode mode, size_t batch_size) {
    start_policy_.configure(mode, batch_size);
}

void TaskGraphRuntime::configure_trace_policy(TraceLevel level) {
    trace_policy_.set_level(level);
}

}  // namespace pto::wsp::graph
