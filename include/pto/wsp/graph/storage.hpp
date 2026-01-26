// Copyright 2026 PTO-RT Authors
// SPDX-License-Identifier: Apache-2.0

#pragma once  // API-2 FIX: Consistent include guard style
// (was: #ifndef PTO_WSP_GRAPH_STORAGE_HPP)

#include "pto/rt/graph/types.hpp"

#include <vector>
#include <string>
#include <unordered_map>
#include <optional>
#include <cassert>  // DS-2 FIX: for finalization guards
#include <span>     // CPP-4 FIX: for fanout_span

namespace pto::wsp::graph {

// ============================================================
// Kernel Bundle
// ============================================================

/// Kernel metadata for dispatch
struct KernelInfo {
    std::string name;
    std::string symbol;       // Mangled symbol for linking
    uint16_t num_params;
    uint16_t num_io;
    ExecDomain default_domain;
    ExecPool default_pool;
};

/// Kernel table: KernelId → KernelInfo
class KernelBundle {
public:
    /// Register a kernel. Returns existing ID if already registered with same name.
    /// DS-3 FIX: Check for duplicates to prevent silent overwrites.
    KernelId register_kernel(const KernelInfo& info) {
        // Check if already registered - return existing ID instead of creating duplicate
        auto existing = find_kernel(info.name);
        if (existing) {
            return *existing;
        }
        KernelId id = static_cast<KernelId>(kernels_.size());
        name_to_id_[info.name] = id;
        kernels_.push_back(info);
        return id;
    }

    std::optional<KernelId> find_kernel(const std::string& name) const {
        auto it = name_to_id_.find(name);
        if (it != name_to_id_.end()) {
            return it->second;
        }
        return std::nullopt;
    }

    const KernelInfo& get_kernel(KernelId id) const {
        return kernels_.at(id);
    }

    size_t size() const { return kernels_.size(); }

    auto begin() const { return kernels_.begin(); }
    auto end() const { return kernels_.end(); }

private:
    std::vector<KernelInfo> kernels_;
    std::unordered_map<std::string, KernelId> name_to_id_;
};

// ============================================================
// Task Graph Storage (Immutable after build)
// ============================================================

/// Layer 1: Immutable task graph storage (device-copyable)
class TaskGraphStorage {
public:
    TaskGraphStorage() = default;

    // Builder methods

    /// Reserve space for tasks
    void reserve(size_t num_tasks, size_t num_edges) {
        tasks_.reserve(num_tasks);
        fanout_edges_.reserve(num_edges);
    }

    /// Add a task to the graph (returns task ID)
    TaskId add_task(TaskNodePod task) {
        assert(!finalized_ && "DS-2 FIX: Cannot add task after finalize()");
        TaskId id = static_cast<TaskId>(tasks_.size());
        task.id = id;
        tasks_.push_back(std::move(task));
        return id;
    }

    /// Set fanout edges for a task
    void set_fanout(TaskId id, const std::vector<TaskId>& fanout) {
        assert(!finalized_ && "DS-2 FIX: Cannot set fanout after finalize()");
        if (id >= tasks_.size()) return;

        tasks_[id].fanout_begin = static_cast<uint32_t>(fanout_edges_.size());
        tasks_[id].fanout_count = static_cast<uint16_t>(fanout.size());

        for (TaskId target : fanout) {
            fanout_edges_.push_back(target);
        }
    }

    /// Increment fanin count for a task
    void increment_fanin(TaskId id) {
        assert(!finalized_ && "DS-2 FIX: Cannot increment fanin after finalize()");
        if (id < tasks_.size()) {
            tasks_[id].fanin++;
        }
    }

    /// Finalize the graph (no more modifications)
    void finalize() {
        finalized_ = true;
        // Build ready list (tasks with fanin == 0)
        ready_tasks_.clear();
        for (const auto& task : tasks_) {
            if (task.fanin == 0) {
                ready_tasks_.push_back(task.id);
            }
        }
    }

    // Accessors

    const TaskNodePod& get_task(TaskId id) const {
        return tasks_.at(id);
    }

    TaskNodePod& get_task_mut(TaskId id) {
        return tasks_.at(id);
    }

    size_t num_tasks() const { return tasks_.size(); }
    size_t num_edges() const { return fanout_edges_.size(); }
    bool is_finalized() const { return finalized_; }

    const std::vector<TaskNodePod>& tasks() const { return tasks_; }
    const std::vector<TaskId>& fanout_edges() const { return fanout_edges_; }
    const std::vector<TaskId>& ready_tasks() const { return ready_tasks_; }

    /// Get fanout targets for a task (allocates - use fanout_span for performance)
    std::vector<TaskId> get_fanout(TaskId id) const {
        std::vector<TaskId> result;
        if (id < tasks_.size()) {
            const auto& task = tasks_[id];
            result.reserve(task.fanout_count);
            for (uint16_t i = 0; i < task.fanout_count; ++i) {
                result.push_back(fanout_edges_[task.fanout_begin + i]);
            }
        }
        return result;
    }

    /// CPP-4 FIX: Get fanout as span (zero allocation, use in hot paths)
    std::span<const TaskId> fanout_span(TaskId id) const {
        if (id >= tasks_.size()) {
            return {};
        }
        const auto& task = tasks_[id];
        return std::span<const TaskId>(
            fanout_edges_.data() + task.fanout_begin,
            task.fanout_count);
    }

    // Kernel bundle access
    KernelBundle& kernel_bundle() { return kernel_bundle_; }
    const KernelBundle& kernel_bundle() const { return kernel_bundle_; }

private:
    std::vector<TaskNodePod> tasks_;
    std::vector<TaskId> fanout_edges_;  // Flat adjacency list
    std::vector<TaskId> ready_tasks_;   // Initial ready tasks (fanin == 0)
    KernelBundle kernel_bundle_;
    bool finalized_ = false;
};

// ============================================================
// Task Graph Builder
// ============================================================

/// Helper for building task graphs
/// INT-1 NOTE: To auto-infer dependencies from I/O regions, use DependencyAnalyzer
/// from tensor_map.hpp. Call analyzer.register_outputs() after each submit(),
/// then analyzer.analyze_dependencies() to get deps before the next task.
class TaskGraphBuilder {
public:
    explicit TaskGraphBuilder(TaskGraphStorage& storage)
        : storage_(storage) {}

    /// Start building a new task
    TaskGraphBuilder& begin_task(KernelId kernel) {
        current_task_ = TaskNodePod{};
        current_task_.kernel = kernel;
        current_task_.domain = ExecDomain::HostCPU;
        current_task_.pool = ExecPool::Any;
        current_task_.stream = 0;
        current_task_.affinity = 0;
        current_task_.fanin = 0;
        current_task_.num_u64_args = 0;
        current_task_.num_io = 0;
        current_task_.sched_tags = SchedTag::None;
        return *this;
    }

    TaskGraphBuilder& set_domain(ExecDomain domain) {
        current_task_.domain = domain;
        return *this;
    }

    TaskGraphBuilder& set_pool(ExecPool pool) {
        current_task_.pool = pool;
        return *this;
    }

    TaskGraphBuilder& set_stream(StreamId stream) {
        current_task_.stream = stream;
        return *this;
    }

    TaskGraphBuilder& set_affinity(TargetId affinity) {
        current_task_.affinity = affinity;
        return *this;
    }

    TaskGraphBuilder& add_arg(uint64_t arg) {
        if (current_task_.num_u64_args < MAX_TASK_ARGS) {
            current_task_.args[current_task_.num_u64_args++] = arg;
        }
        return *this;
    }

    TaskGraphBuilder& add_input(const TensorRegion2D& region) {
        if (current_task_.num_io < MAX_TASK_IO) {
            current_task_.io[current_task_.num_io++] = TaskIO{region, 0, {}};
        }
        return *this;
    }

    TaskGraphBuilder& add_output(const TensorRegion2D& region) {
        if (current_task_.num_io < MAX_TASK_IO) {
            current_task_.io[current_task_.num_io++] = TaskIO{region, 1, {}};
        }
        return *this;
    }

    TaskGraphBuilder& set_tags(SchedTag tags) {
        current_task_.sched_tags = tags;
        return *this;
    }

    /// Finish building and add the task
    TaskId submit() {
        TaskId id = storage_.add_task(current_task_);
        current_task_ = TaskNodePod{};
        return id;
    }

    /// Add dependency edge: from → to (from produces, to consumes)
    void add_dependency(TaskId from, TaskId to) {
        pending_edges_.push_back({from, to});
    }

    /// Finalize all edges and the graph
    void finalize() {
        // Group edges by source task
        std::unordered_map<TaskId, std::vector<TaskId>> fanout_map;
        for (const auto& [from, to] : pending_edges_) {
            fanout_map[from].push_back(to);
            storage_.increment_fanin(to);
        }

        // Set fanout for each source task
        for (const auto& [from, targets] : fanout_map) {
            storage_.set_fanout(from, targets);
        }

        storage_.finalize();
        pending_edges_.clear();
    }

private:
    TaskGraphStorage& storage_;
    TaskNodePod current_task_{};
    std::vector<std::pair<TaskId, TaskId>> pending_edges_;
};

}  // namespace pto::wsp::graph

