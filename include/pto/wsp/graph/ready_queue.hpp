// Copyright 2026 PTO-RT Authors
// SPDX-License-Identifier: MIT

#pragma once  // API-2 FIX: Consistent include guard style
// (was: #ifndef PTO_WSP_GRAPH_READY_QUEUE_HPP)

#include "pto/rt/graph/types.hpp"

#include <atomic>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <vector>
#include <optional>
#include <thread>  // INC-2 FIX: for std::this_thread::yield()

namespace pto::wsp::graph {

// ============================================================
// Ready Queue Interface
// ============================================================

/// Abstract interface for ready task queue
class ReadyQueue {
public:
    virtual ~ReadyQueue() = default;

    /// Push a ready task
    virtual void push(TaskId tid) = 0;

    /// Try to pop a ready task (non-blocking)
    virtual std::optional<TaskId> try_pop() = 0;

    /// Pop a ready task (blocking)
    virtual TaskId pop() = 0;

    /// Check if empty
    virtual bool empty() const = 0;

    /// Get approximate size
    virtual size_t size() const = 0;

    /// Signal shutdown (wake up blocked workers)
    virtual void shutdown() = 0;
};

// ============================================================
// Ready Queue Set (Multi-Queue for Dual-Queue Dispatch)
// ============================================================

/// Multi-queue interface for dual-queue (vector/cube) dispatch
class ReadyQueueSet {
public:
    virtual ~ReadyQueueSet() = default;

    /// Push task to specific pool queue
    virtual void push(ExecPool pool, TaskId tid) = 0;

    /// Try to pop from specific pool queue
    virtual std::optional<TaskId> try_pop(ExecPool pool) = 0;

    /// Try to pop from any queue (work stealing)
    virtual std::optional<TaskId> try_pop_any() = 0;

    /// Pop from specific pool (blocking)
    virtual TaskId pop(ExecPool pool) = 0;

    /// Check if all queues are empty
    virtual bool all_empty() const = 0;

    /// Get total count across all queues
    virtual size_t total_size() const = 0;

    /// Signal shutdown
    virtual void shutdown() = 0;
};

// ============================================================
// MPMC Ready Queue (Mutex + Condition Variable)
// ============================================================

/// Multi-producer multi-consumer ready queue
class ReadyQueueMPMC : public ReadyQueue {
public:
    ReadyQueueMPMC() = default;

    void push(TaskId tid) override {
        std::lock_guard<std::mutex> lock(mutex_);
        queue_.push(tid);
        cv_.notify_one();
    }

    std::optional<TaskId> try_pop() override {
        std::lock_guard<std::mutex> lock(mutex_);
        if (queue_.empty()) {
            return std::nullopt;
        }
        TaskId tid = queue_.front();
        queue_.pop();
        return tid;
    }

    TaskId pop() override {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this] { return !queue_.empty() || shutdown_; });
        if (shutdown_ && queue_.empty()) {
            return INVALID_TASK_ID;
        }
        TaskId tid = queue_.front();
        queue_.pop();
        return tid;
    }

    bool empty() const override {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.empty();
    }

    size_t size() const override {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.size();
    }

    void shutdown() override {
        std::lock_guard<std::mutex> lock(mutex_);
        shutdown_ = true;
        cv_.notify_all();
    }

private:
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::queue<TaskId> queue_;
    bool shutdown_ = false;
};

// ============================================================
// Lock-Free Ring Buffer Queue (SPSC)
// ============================================================

/// Single-producer single-consumer lock-free ring buffer
/// For per-worker fast path optimization
class ReadyQueueSPSC : public ReadyQueue {
public:
    static constexpr size_t DEFAULT_CAPACITY = 4096;

    explicit ReadyQueueSPSC(size_t capacity = DEFAULT_CAPACITY)
        : capacity_(capacity), buffer_(capacity) {
        head_.store(0, std::memory_order_relaxed);
        tail_.store(0, std::memory_order_relaxed);
    }

    void push(TaskId tid) override {
        size_t head = head_.load(std::memory_order_relaxed);
        size_t next_head = (head + 1) % capacity_;

        // Spin if full (should not happen with proper sizing)
        while (next_head == tail_.load(std::memory_order_acquire)) {
            // Queue full - in production, would resize or block
        }

        buffer_[head] = tid;
        head_.store(next_head, std::memory_order_release);
    }

    std::optional<TaskId> try_pop() override {
        size_t tail = tail_.load(std::memory_order_relaxed);
        if (tail == head_.load(std::memory_order_acquire)) {
            return std::nullopt;  // Empty
        }

        TaskId tid = buffer_[tail];
        tail_.store((tail + 1) % capacity_, std::memory_order_release);
        return tid;
    }

    TaskId pop() override {
        // Spin until available (SPSC doesn't need condition variable)
        while (true) {
            auto result = try_pop();
            if (result || shutdown_.load(std::memory_order_acquire)) {
                return result.value_or(INVALID_TASK_ID);
            }
            // Busy wait (could add backoff)
        }
    }

    bool empty() const override {
        return tail_.load(std::memory_order_acquire) ==
               head_.load(std::memory_order_acquire);
    }

    size_t size() const override {
        size_t head = head_.load(std::memory_order_acquire);
        size_t tail = tail_.load(std::memory_order_acquire);
        return (head >= tail) ? (head - tail) : (capacity_ - tail + head);
    }

    void shutdown() override {
        shutdown_.store(true, std::memory_order_release);
    }

private:
    size_t capacity_;
    std::vector<TaskId> buffer_;
    std::atomic<size_t> head_;
    std::atomic<size_t> tail_;
    std::atomic<bool> shutdown_{false};
};

// ============================================================
// Ready Queue Set Implementation (Dual-Queue)
// ============================================================

/// Dual-queue implementation for vector/cube workers
class DualQueueSet : public ReadyQueueSet {
public:
    DualQueueSet()
        : vector_queue_(std::make_unique<ReadyQueueMPMC>()),
          cube_queue_(std::make_unique<ReadyQueueMPMC>()) {}

    void push(ExecPool pool, TaskId tid) override {
        switch (pool) {
        case ExecPool::Vector:
            vector_queue_->push(tid);
            break;
        case ExecPool::Cube:
            cube_queue_->push(tid);
            break;
        case ExecPool::Any:
            // Default to vector queue
            vector_queue_->push(tid);
            break;
        }
    }

    std::optional<TaskId> try_pop(ExecPool pool) override {
        switch (pool) {
        case ExecPool::Vector:
            return vector_queue_->try_pop();
        case ExecPool::Cube:
            return cube_queue_->try_pop();
        case ExecPool::Any:
            return try_pop_any();
        }
        return std::nullopt;
    }

    std::optional<TaskId> try_pop_any() override {
        // Try vector first, then cube
        auto result = vector_queue_->try_pop();
        if (result) return result;
        return cube_queue_->try_pop();
    }

    TaskId pop(ExecPool pool) override {
        switch (pool) {
        case ExecPool::Vector:
            return vector_queue_->pop();
        case ExecPool::Cube:
            return cube_queue_->pop();
        case ExecPool::Any:
            // Blocking pop from any - poll both queues
            while (true) {
                auto result = try_pop_any();
                if (result) return *result;
                // Brief sleep to avoid busy wait
                std::this_thread::yield();
            }
        }
        return INVALID_TASK_ID;
    }

    bool all_empty() const override {
        return vector_queue_->empty() && cube_queue_->empty();
    }

    size_t total_size() const override {
        return vector_queue_->size() + cube_queue_->size();
    }

    void shutdown() override {
        vector_queue_->shutdown();
        cube_queue_->shutdown();
    }

private:
    std::unique_ptr<ReadyQueueMPMC> vector_queue_;
    std::unique_ptr<ReadyQueueMPMC> cube_queue_;
};

// ============================================================
// Single Queue Set (for non-dual-queue backends)
// ============================================================

/// Single queue implementation (for CPU sim without dual-queue)
class SingleQueueSet : public ReadyQueueSet {
public:
    SingleQueueSet() : queue_(std::make_unique<ReadyQueueMPMC>()) {}

    void push(ExecPool /*pool*/, TaskId tid) override {
        queue_->push(tid);
    }

    std::optional<TaskId> try_pop(ExecPool /*pool*/) override {
        return queue_->try_pop();
    }

    std::optional<TaskId> try_pop_any() override {
        return queue_->try_pop();
    }

    TaskId pop(ExecPool /*pool*/) override {
        return queue_->pop();
    }

    bool all_empty() const override {
        return queue_->empty();
    }

    size_t total_size() const override {
        return queue_->size();
    }

    void shutdown() override {
        queue_->shutdown();
    }

private:
    std::unique_ptr<ReadyQueueMPMC> queue_;
};

// ============================================================
// Work-Stealing Queue Set (L12 - Concurrent Mechanism)
// ============================================================

/// Per-worker deque for work-stealing scheduler.
/// Each worker has its own deque; workers steal from others when idle.
class WorkStealingDeque {
public:
    static constexpr size_t DEFAULT_CAPACITY = 1024;

    explicit WorkStealingDeque(size_t capacity = DEFAULT_CAPACITY)
        : capacity_(capacity), buffer_(capacity) {
        top_.store(0, std::memory_order_relaxed);
        bottom_.store(0, std::memory_order_relaxed);
    }

    /// Push to bottom (owner only)
    void push(TaskId tid) {
        size_t bottom = bottom_.load(std::memory_order_relaxed);
        buffer_[bottom % capacity_] = tid;
        std::atomic_thread_fence(std::memory_order_release);
        bottom_.store(bottom + 1, std::memory_order_relaxed);
    }

    /// Pop from bottom (owner only)
    std::optional<TaskId> pop() {
        size_t bottom = bottom_.load(std::memory_order_relaxed);
        if (bottom == 0) return std::nullopt;

        bottom--;
        bottom_.store(bottom, std::memory_order_relaxed);
        std::atomic_thread_fence(std::memory_order_seq_cst);

        size_t top = top_.load(std::memory_order_relaxed);
        if (top <= bottom) {
            TaskId tid = buffer_[bottom % capacity_];
            if (top == bottom) {
                // Last element - compete with stealers
                if (!top_.compare_exchange_strong(top, top + 1,
                    std::memory_order_seq_cst, std::memory_order_relaxed)) {
                    // Lost race to stealer
                    bottom_.store(top + 1, std::memory_order_relaxed);
                    return std::nullopt;
                }
                bottom_.store(top + 1, std::memory_order_relaxed);
            }
            return tid;
        } else {
            // Empty
            bottom_.store(top, std::memory_order_relaxed);
            return std::nullopt;
        }
    }

    /// Steal from top (other workers)
    std::optional<TaskId> steal() {
        size_t top = top_.load(std::memory_order_acquire);
        std::atomic_thread_fence(std::memory_order_seq_cst);
        size_t bottom = bottom_.load(std::memory_order_acquire);

        if (top >= bottom) return std::nullopt;  // Empty

        TaskId tid = buffer_[top % capacity_];
        if (!top_.compare_exchange_strong(top, top + 1,
            std::memory_order_seq_cst, std::memory_order_relaxed)) {
            // Lost race to another stealer or owner
            return std::nullopt;
        }
        return tid;
    }

    bool empty() const {
        return top_.load(std::memory_order_acquire) >=
               bottom_.load(std::memory_order_acquire);
    }

    size_t size() const {
        size_t top = top_.load(std::memory_order_acquire);
        size_t bottom = bottom_.load(std::memory_order_acquire);
        return (bottom > top) ? (bottom - top) : 0;
    }

private:
    size_t capacity_;
    std::vector<TaskId> buffer_;
    std::atomic<size_t> top_;    // Steal from here
    std::atomic<size_t> bottom_; // Push/pop from here
};

/// Work-stealing queue set with per-worker deques.
/// Implements L12 concurrent mechanism requirement.
class WorkStealingQueueSet : public ReadyQueueSet {
public:
    explicit WorkStealingQueueSet(size_t num_workers)
        : num_workers_(num_workers), shutdown_(false) {
        for (size_t i = 0; i < num_workers; ++i) {
            worker_deques_.push_back(std::make_unique<WorkStealingDeque>());
        }
        current_worker_.store(0, std::memory_order_relaxed);
    }

    /// Push task - round-robin to worker deques
    void push(ExecPool /*pool*/, TaskId tid) override {
        size_t worker = current_worker_.fetch_add(1, std::memory_order_relaxed) % num_workers_;
        worker_deques_[worker]->push(tid);
    }

    /// Try to pop from worker's own deque
    std::optional<TaskId> try_pop(ExecPool /*pool*/) override {
        // Each thread should have its own worker ID in production
        // For now, use thread-local approximation
        thread_local size_t my_worker = current_worker_.fetch_add(1) % num_workers_;
        return try_pop_worker(my_worker);
    }

    /// Try to pop from own deque, then steal from others
    std::optional<TaskId> try_pop_worker(size_t worker_id) {
        // Try own deque first
        auto result = worker_deques_[worker_id]->pop();
        if (result) return result;

        // Try stealing from other workers
        for (size_t i = 1; i < num_workers_; ++i) {
            size_t victim = (worker_id + i) % num_workers_;
            result = worker_deques_[victim]->steal();
            if (result) return result;
        }
        return std::nullopt;
    }

    std::optional<TaskId> try_pop_any() override {
        // Try all deques
        for (size_t i = 0; i < num_workers_; ++i) {
            auto result = worker_deques_[i]->pop();
            if (result) return result;
        }
        return std::nullopt;
    }

    TaskId pop(ExecPool pool) override {
        while (!shutdown_.load(std::memory_order_acquire)) {
            auto result = try_pop(pool);
            if (result) return *result;
            std::this_thread::yield();
        }
        return INVALID_TASK_ID;
    }

    bool all_empty() const override {
        for (const auto& deque : worker_deques_) {
            if (!deque->empty()) return false;
        }
        return true;
    }

    size_t total_size() const override {
        size_t total = 0;
        for (const auto& deque : worker_deques_) {
            total += deque->size();
        }
        return total;
    }

    void shutdown() override {
        shutdown_.store(true, std::memory_order_release);
    }

    size_t num_workers() const { return num_workers_; }

private:
    size_t num_workers_;
    std::vector<std::unique_ptr<WorkStealingDeque>> worker_deques_;
    std::atomic<size_t> current_worker_;
    std::atomic<bool> shutdown_;
};

}  // namespace pto::wsp::graph
