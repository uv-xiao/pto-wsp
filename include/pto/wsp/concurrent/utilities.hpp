// PTO Workload-Schedule Programming (PTO-WSP) framework v9 - Common Concurrency Utilities
// Copyright (c) 2026 PTO Project
// SPDX-License-Identifier: Apache-2.0
//
// This file extracts common concurrency patterns from pto-isa-lh and pto-isa-wc
// for reuse across backends. Implements L12 requirement for common concurrency utilities.

#pragma once

#include <atomic>
#include <thread>
#include <vector>
#include <functional>
#include <memory>
#include <condition_variable>
#include <mutex>
#include <queue>
#include <optional>

namespace pto::wsp::concurrent {

// ============================================================
// Thread-Safe Counter (for reference counting, completion tracking)
// ============================================================

/**
 * Atomic counter with completion notification.
 * Used for tracking task completion, reference counting, and synchronization.
 */
class CompletionCounter {
public:
    explicit CompletionCounter(int64_t initial = 0)
        : count_(initial) {}

    /// Increment counter
    void increment(int64_t delta = 1) {
        count_.fetch_add(delta, std::memory_order_acq_rel);
    }

    /// Decrement counter and return new value
    int64_t decrement(int64_t delta = 1) {
        return count_.fetch_sub(delta, std::memory_order_acq_rel) - delta;
    }

    /// Get current value
    [[nodiscard]] int64_t get() const {
        return count_.load(std::memory_order_acquire);
    }

    /// Check if counter is zero
    [[nodiscard]] bool is_zero() const {
        return get() == 0;
    }

    /// Wait until counter reaches zero (busy wait)
    void wait_for_zero() const {
        while (!is_zero()) {
            std::this_thread::yield();
        }
    }

    /// Wait with backoff
    void wait_for_zero_with_backoff(int max_spins = 1000) const {
        int spins = 0;
        while (!is_zero()) {
            if (++spins > max_spins) {
                std::this_thread::sleep_for(std::chrono::microseconds(10));
                spins = 0;
            } else {
                std::this_thread::yield();
            }
        }
    }

private:
    std::atomic<int64_t> count_;
};

// ============================================================
// Latch (Single-use barrier)
// ============================================================

/**
 * Single-use latch for synchronization.
 * Threads can wait on the latch until it's released.
 */
class Latch {
public:
    explicit Latch(int count = 1)
        : count_(count), released_(false) {}

    /// Count down the latch (release when reaches 0)
    void count_down() {
        std::unique_lock lock(mutex_);
        if (--count_ <= 0) {
            released_ = true;
            cv_.notify_all();
        }
    }

    /// Wait for latch to be released
    void wait() {
        std::unique_lock lock(mutex_);
        cv_.wait(lock, [this] { return released_; });
    }

    /// Try to wait with timeout
    template<typename Rep, typename Period>
    bool wait_for(const std::chrono::duration<Rep, Period>& timeout) {
        std::unique_lock lock(mutex_);
        return cv_.wait_for(lock, timeout, [this] { return released_; });
    }

    /// Check if released without blocking
    [[nodiscard]] bool is_released() const {
        std::lock_guard lock(mutex_);
        return released_;
    }

private:
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    int count_;
    bool released_;
};

// ============================================================
// Barrier (Reusable multi-phase synchronization)
// ============================================================

/**
 * Reusable barrier for multi-phase synchronization.
 * All threads must arrive before any can proceed.
 */
class Barrier {
public:
    explicit Barrier(int num_threads)
        : num_threads_(num_threads), count_(num_threads), generation_(0) {}

    /// Arrive at barrier and wait for all threads
    void arrive_and_wait() {
        std::unique_lock lock(mutex_);
        auto gen = generation_;

        if (--count_ == 0) {
            // Last thread - reset and release all
            generation_++;
            count_ = num_threads_;
            cv_.notify_all();
        } else {
            // Wait for others
            cv_.wait(lock, [this, gen] { return generation_ != gen; });
        }
    }

private:
    std::mutex mutex_;
    std::condition_variable cv_;
    const int num_threads_;
    int count_;
    int generation_;
};

// ============================================================
// MPMC Bounded Queue (Multi-Producer Multi-Consumer)
// ============================================================

/**
 * Thread-safe bounded queue with blocking operations.
 * Suitable for work distribution and producer-consumer patterns.
 */
template<typename T>
class BoundedQueue {
public:
    explicit BoundedQueue(size_t capacity)
        : capacity_(capacity), closed_(false) {}

    /// Push item (blocks if full)
    bool push(T item) {
        std::unique_lock lock(mutex_);
        not_full_.wait(lock, [this] { return queue_.size() < capacity_ || closed_; });

        if (closed_) return false;

        queue_.push(std::move(item));
        not_empty_.notify_one();
        return true;
    }

    /// Try to push without blocking
    bool try_push(T item) {
        std::unique_lock lock(mutex_);
        if (queue_.size() >= capacity_ || closed_) return false;

        queue_.push(std::move(item));
        not_empty_.notify_one();
        return true;
    }

    /// Pop item (blocks if empty)
    std::optional<T> pop() {
        std::unique_lock lock(mutex_);
        not_empty_.wait(lock, [this] { return !queue_.empty() || closed_; });

        if (queue_.empty()) return std::nullopt;

        T item = std::move(queue_.front());
        queue_.pop();
        not_full_.notify_one();
        return item;
    }

    /// Try to pop without blocking
    std::optional<T> try_pop() {
        std::unique_lock lock(mutex_);
        if (queue_.empty()) return std::nullopt;

        T item = std::move(queue_.front());
        queue_.pop();
        not_full_.notify_one();
        return item;
    }

    /// Close the queue (wake all waiters)
    void close() {
        std::lock_guard lock(mutex_);
        closed_ = true;
        not_empty_.notify_all();
        not_full_.notify_all();
    }

    /// Check if closed
    [[nodiscard]] bool is_closed() const {
        std::lock_guard lock(mutex_);
        return closed_;
    }

    /// Get current size
    [[nodiscard]] size_t size() const {
        std::lock_guard lock(mutex_);
        return queue_.size();
    }

private:
    mutable std::mutex mutex_;
    std::condition_variable not_empty_;
    std::condition_variable not_full_;
    std::queue<T> queue_;
    const size_t capacity_;
    bool closed_;
};

// ============================================================
// Thread Pool (Fixed-size worker pool)
// ============================================================

/**
 * Simple fixed-size thread pool for parallel task execution.
 * Extracted as common utility from CPU simulation backend.
 */
class ThreadPool {
public:
    explicit ThreadPool(size_t num_threads = 0)
        : running_(false) {
        if (num_threads == 0) {
            num_threads = std::thread::hardware_concurrency();
            if (num_threads == 0) num_threads = 4;
        }
        num_threads_ = num_threads;
    }

    ~ThreadPool() {
        shutdown();
    }

    /// Start the thread pool
    void start() {
        if (running_.exchange(true)) return;

        workers_.reserve(num_threads_);
        for (size_t i = 0; i < num_threads_; ++i) {
            workers_.emplace_back([this] { worker_loop(); });
        }
    }

    /// Shutdown and wait for completion
    void shutdown() {
        if (!running_.exchange(false)) return;

        tasks_.close();
        for (auto& w : workers_) {
            if (w.joinable()) w.join();
        }
        workers_.clear();
    }

    /// Submit a task
    void submit(std::function<void()> task) {
        if (running_) {
            tasks_.push(std::move(task));
        }
    }

    /// Get number of workers
    [[nodiscard]] size_t num_workers() const {
        return num_threads_;
    }

private:
    void worker_loop() {
        while (running_) {
            auto task = tasks_.pop();
            if (task) {
                (*task)();
            }
        }
    }

    std::vector<std::thread> workers_;
    BoundedQueue<std::function<void()>> tasks_{4096};
    size_t num_threads_;
    std::atomic<bool> running_;
};

// ============================================================
// Domain Handshake (Multi-domain execution coordination)
// ============================================================

/**
 * Execution domain enumeration for multi-domain coordination.
 */
enum class ExecDomain {
    HostCPU,
    AscendAICore,
    AMDAIETile,
    Generic
};

/**
 * Domain handshake for coordinating execution across CPU and accelerators.
 * Implements L12 multi-domain execution requirement.
 */
class DomainHandshake {
public:
    DomainHandshake()
        : cpu_ready_(false), accelerator_ready_(false), transfer_complete_(false) {}

    /// Signal that CPU side is ready
    void cpu_ready() {
        std::lock_guard lock(mutex_);
        cpu_ready_ = true;
        cv_.notify_all();
    }

    /// Signal that accelerator side is ready
    void accelerator_ready() {
        std::lock_guard lock(mutex_);
        accelerator_ready_ = true;
        cv_.notify_all();
    }

    /// Wait for CPU to be ready
    void wait_cpu_ready() {
        std::unique_lock lock(mutex_);
        cv_.wait(lock, [this] { return cpu_ready_; });
    }

    /// Wait for accelerator to be ready
    void wait_accelerator_ready() {
        std::unique_lock lock(mutex_);
        cv_.wait(lock, [this] { return accelerator_ready_; });
    }

    /// Signal data transfer complete
    void transfer_complete() {
        std::lock_guard lock(mutex_);
        transfer_complete_ = true;
        cv_.notify_all();
    }

    /// Wait for data transfer to complete
    void wait_transfer_complete() {
        std::unique_lock lock(mutex_);
        cv_.wait(lock, [this] { return transfer_complete_; });
    }

    /// Reset for next iteration
    void reset() {
        std::lock_guard lock(mutex_);
        cpu_ready_ = false;
        accelerator_ready_ = false;
        transfer_complete_ = false;
    }

    /// Check if both sides are ready
    [[nodiscard]] bool both_ready() const {
        std::lock_guard lock(mutex_);
        return cpu_ready_ && accelerator_ready_;
    }

private:
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    bool cpu_ready_;
    bool accelerator_ready_;
    bool transfer_complete_;
};

// ============================================================
// Parallel For (Utility from pto-isa)
// ============================================================

/**
 * Parallel for loop utility.
 * Based on pto::cpu::parallel_for_1d from pto-isa.
 */
template<typename Fn>
void parallel_for(size_t begin, size_t end, Fn&& fn, size_t grain_size = 1024) {
    const size_t count = (end > begin) ? (end - begin) : 0;
    if (count == 0) return;

    // Single-threaded for small workloads
    if (count < grain_size) {
        for (size_t i = begin; i < end; ++i) {
            fn(i);
        }
        return;
    }

    // Determine thread count
    unsigned hw = std::thread::hardware_concurrency();
    if (hw == 0) hw = 4;
    const unsigned threads = std::min<unsigned>(hw, static_cast<unsigned>(count / grain_size + 1));

    if (threads <= 1) {
        for (size_t i = begin; i < end; ++i) {
            fn(i);
        }
        return;
    }

    // Parallel execution
    const size_t chunk = (count + threads - 1) / threads;
    std::vector<std::thread> workers;
    workers.reserve(threads);

    for (unsigned t = 0; t < threads; ++t) {
        const size_t b = begin + static_cast<size_t>(t) * chunk;
        const size_t e = std::min(end, b + chunk);
        if (b >= e) break;

        workers.emplace_back([&fn, b, e]() {
            for (size_t i = b; i < e; ++i) {
                fn(i);
            }
        });
    }

    for (auto& w : workers) {
        w.join();
    }
}

}  // namespace pto::wsp::concurrent
