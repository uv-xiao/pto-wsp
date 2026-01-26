// PTO Workload-Schedule Programming (PTO-WSP) framework v9 - Schedule IR Nodes
// Copyright (c) 2024 PTO Project
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "core.hpp"

namespace pto::wsp::ir {

// Dispatch policies
enum class DispatchPolicy {
    RoundRobin,
    Affinity,
    Hash,
    WorkSteal,
    Custom,
};

// Dispatch Node - Task → AICPU assignment
struct DispatchNode : ScheduleNode {
    const DispatchPolicy policy;
    const int num_targets;       // For RoundRobin
    const std::string key_expr;  // For Affinity, Hash, Custom

    DispatchNode(NodeId id, DispatchPolicy policy, int num_targets = 0,
                 std::string key_expr = "")
        : ScheduleNode(id, NodeKind::Dispatch, WorkloadLevel::CPU),
          policy(policy),
          num_targets(num_targets),
          key_expr(std::move(key_expr)) {}

    void print(std::ostream& os, int indent) const override {
        os << std::string(indent, ' ') << "dispatch = ";
        switch (policy) {
            case DispatchPolicy::RoundRobin:
                os << "round_robin(" << num_targets << ")";
                break;
            case DispatchPolicy::Affinity:
                os << "affinity(" << key_expr << ")";
                break;
            case DispatchPolicy::Hash:
                os << "hash(" << key_expr << ")";
                break;
            case DispatchPolicy::WorkSteal:
                os << "work_steal";
                break;
            case DispatchPolicy::Custom:
                os << "dispatch_by(" << key_expr << ")";
                break;
        }
    }
};

// Stream Node - Concurrent execution streams
struct StreamNode : ScheduleNode {
    const int num_streams;
    const std::string key_expr;  // For stream_by (empty = single stream)

    StreamNode(NodeId id, int num_streams, std::string key_expr = "")
        : ScheduleNode(id, NodeKind::Stream, WorkloadLevel::CPU),
          num_streams(num_streams),
          key_expr(std::move(key_expr)) {}

    void print(std::ostream& os, int indent) const override {
        os << std::string(indent, ' ') << "streams = " << num_streams;
        if (!key_expr.empty()) {
            os << "\n" << std::string(indent, ' ') << "stream_by = " << key_expr;
        }
    }
};

// Timing policies
enum class TimingPolicy {
    Immediate,
    Batched,
    Interleaved,
    RateLimited,
};

// Timing Node - When to issue tasks
struct TimingNode : ScheduleNode {
    const TimingPolicy policy;
    const int param;  // batch_size, num_streams, or rate

    TimingNode(NodeId id, TimingPolicy policy, int param = 0)
        : ScheduleNode(id, NodeKind::Timing, WorkloadLevel::CPU),
          policy(policy),
          param(param) {}

    void print(std::ostream& os, int indent) const override {
        os << std::string(indent, ' ') << "timing = ";
        switch (policy) {
            case TimingPolicy::Immediate:
                os << "immediate";
                break;
            case TimingPolicy::Batched:
                os << "batched(" << param << ")";
                break;
            case TimingPolicy::Interleaved:
                os << "interleaved(" << param << ")";
                break;
            case TimingPolicy::RateLimited:
                os << "rate_limit(" << param << ")";
                break;
        }
    }
};

// SpatialMap Node - Spatial mapping to tile grid
struct SpatialMapNode : ScheduleNode {
    const std::vector<int64_t> grid;  // e.g., {4, 4} for 4x4

    SpatialMapNode(NodeId id, std::vector<int64_t> grid)
        : ScheduleNode(id, NodeKind::SpatialMap, WorkloadLevel::NPU),
          grid(std::move(grid)) {}

    void print(std::ostream& os, int indent) const override {
        os << std::string(indent, ' ') << "spatial_map = (";
        for (size_t i = 0; i < grid.size(); ++i) {
            if (i > 0) os << ", ";
            os << grid[i];
        }
        os << ")";
    }
};

// Layout dimension kinds
enum class LayoutDimKind {
    Shard,
    Replicate,
};

// Layout dimension
struct LayoutDim {
    LayoutDimKind kind;
    int64_t mesh_axis;  // Only for Shard

    [[nodiscard]] std::string to_string() const {
        if (kind == LayoutDimKind::Replicate) {
            return "R";
        }
        return "S(" + std::to_string(mesh_axis) + ")";
    }
};

// Layout Node - Data layout for tensor
struct LayoutNode : ScheduleNode {
    const std::string tensor_name;
    const std::vector<LayoutDim> dims;

    LayoutNode(NodeId id, std::string tensor, std::vector<LayoutDim> dims)
        : ScheduleNode(id, NodeKind::Layout, WorkloadLevel::CPU),
          tensor_name(std::move(tensor)),
          dims(std::move(dims)) {}

    void print(std::ostream& os, int indent) const override {
        os << std::string(indent, ' ') << "layout " << tensor_name << " = (";
        for (size_t i = 0; i < dims.size(); ++i) {
            if (i > 0) os << ", ";
            os << dims[i].to_string();
        }
        os << ")";
    }
};

// Extended schedule primitives

// DispatchThreshold Node - Multi-level dispatch based on thresholds
struct DispatchThresholdNode : ScheduleNode {
    const std::vector<int64_t> thresholds;
    const std::vector<DispatchPolicy> policies;

    DispatchThresholdNode(NodeId id,
                          std::vector<int64_t> thresholds,
                          std::vector<DispatchPolicy> policies)
        : ScheduleNode(id, NodeKind::DispatchThreshold, WorkloadLevel::CPU),
          thresholds(std::move(thresholds)),
          policies(std::move(policies)) {}

    void print(std::ostream& os, int indent) const override {
        os << std::string(indent, ' ') << "dispatch_threshold = [";
        for (size_t i = 0; i < thresholds.size(); ++i) {
            if (i > 0) os << ", ";
            os << thresholds[i];
        }
        os << "]";
    }
};

// PipelineDepth Node - In-flight task control
struct PipelineDepthNode : ScheduleNode {
    const int depth;

    PipelineDepthNode(NodeId id, int depth)
        : ScheduleNode(id, NodeKind::PipelineDepth, WorkloadLevel::CPU),
          depth(depth) {}

    void print(std::ostream& os, int indent) const override {
        os << std::string(indent, ' ') << "pipeline_depth = " << depth;
    }
};

// TaskWindow Node - Metadata window management
enum class TaskWindowOverflowPolicy {
    Stall,
    Abort,
    Benchmark,
};

struct TaskWindowNode : ScheduleNode {
    const int64_t size;
    const std::string unit;  // "tasks", "bytes", "entries"
    const TaskWindowOverflowPolicy overflow;

    TaskWindowNode(NodeId id, int64_t size, std::string unit = "tasks",
                   TaskWindowOverflowPolicy overflow = TaskWindowOverflowPolicy::Stall)
        : ScheduleNode(id, NodeKind::TaskWindow, WorkloadLevel::CPU),
          size(size),
          unit(std::move(unit)),
          overflow(overflow) {}

    void print(std::ostream& os, int indent) const override {
        os << std::string(indent, ' ') << "task_window = " << size << " " << unit;
    }
};

// BatchDeps Node - Batched dependency resolution
struct BatchDepsNode : ScheduleNode {
    const int64_t threshold;
    const bool range_compression;

    BatchDepsNode(NodeId id, int64_t threshold, bool range_compression = false)
        : ScheduleNode(id, NodeKind::BatchDeps, WorkloadLevel::CPU),
          threshold(threshold),
          range_compression(range_compression) {}

    void print(std::ostream& os, int indent) const override {
        os << std::string(indent, ' ') << "batch_deps = " << threshold;
        if (range_compression) {
            os << " range_compress";
        }
    }
};

}  // namespace pto::wsp::ir
