// PTO Workload-Schedule Programming (PTO-WSP) framework v9 - Core IR Infrastructure
// Copyright (c) 2024 PTO Project
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <functional>
#include <atomic>
#include <ostream>
#include <optional>

namespace pto::wsp::ir {

// Forward declarations
struct IRNode;
struct AxisNode;
struct WorkloadNode;
struct ScheduleNode;

// Unique ID for each node
using NodeId = uint64_t;

// Workload hierarchy level
enum class WorkloadLevel : uint8_t {
    CPU,   // Outer level: orchestration, dispatch to cores
    NPU,   // Inner level: InCore functions, tile operations
    Any,   // Level-agnostic nodes (axis types, etc.)
};

// Node kinds for dynamic dispatch
enum class NodeKind {
    // Axis nodes
    DenseAxis,
    DenseDynAxis,
    RaggedAxis,
    SparseAxis,

    // Workload nodes
    Task,
    ParallelFor,
    ForEach,
    Select,
    Cond,
    Combine,
    Sequential,
    Call,       // Cross-level call (hierarchy)

    // CSP nodes
    Channel,
    Process,
    Send,
    Consume,
    Pipeline,

    // Schedule nodes
    Dispatch,
    Stream,
    Timing,
    SpatialMap,
    Layout,
    DispatchThreshold,  // Multi-level dispatch
    PipelineDepth,      // In-flight control
    TaskWindow,         // Metadata window
    BatchDeps,          // Batched dependency resolution

    // Extension
    Ext,        // Extension op (escape hatch)
};

// Element types
enum class DType : uint8_t {
    F16, BF16, F32, F64,
    I8, I16, I32, I64,
    U8, U16, U32, U64,
    Bool,
};

// Memory locations
enum class Location : uint8_t {
    Global,  // HBM
    L2,      // L2 cache
    UB,      // Unified buffer
    L1,      // L1 buffer
};

// Shape representation
struct Shape {
    std::vector<int64_t> dims;

    [[nodiscard]] bool is_static() const {
        for (auto d : dims) {
            if (d < 0) return false;
        }
        return true;
    }

    [[nodiscard]] int64_t numel() const {
        int64_t n = 1;
        for (auto d : dims) {
            if (d < 0) return -1;  // Dynamic
            n *= d;
        }
        return n;
    }

    [[nodiscard]] int64_t rank() const { return static_cast<int64_t>(dims.size()); }
};

// Tensor type
struct TensorType {
    Shape shape;
    DType dtype;
    Location location;

    [[nodiscard]] std::string to_string() const;
};

// Channel type
struct ChannelType {
    TensorType element_type;
    int64_t capacity;  // 0 = rendezvous

    [[nodiscard]] std::string to_string() const;
};

// Shared pointer type for IR nodes
template<typename T>
using IRPtr = std::shared_ptr<const T>;

// Base IR node
struct IRNode {
    const NodeId id;
    const NodeKind kind;
    const WorkloadLevel level;

    IRNode(NodeId id, NodeKind kind, WorkloadLevel level = WorkloadLevel::Any)
        : id(id), kind(kind), level(level) {}

    virtual ~IRNode() = default;

    // Printing
    virtual void print(std::ostream& os, int indent = 0) const = 0;

    // Child traversal (for visitor pattern)
    using ChildFn = std::function<void(const IRPtr<IRNode>&)>;
    virtual void forEachChild(const ChildFn&) const {}  // Default: leaf node
};

// Node factory with ID generation
class IRFactory {
    std::atomic<NodeId> next_id_{0};

public:
    template<typename T, typename... Args>
    IRPtr<T> create(Args&&... args) {
        return std::make_shared<const T>(next_id_++, std::forward<Args>(args)...);
    }

    // Reset ID counter (for testing)
    void reset() { next_id_ = 0; }
};

// Axis node base
struct AxisNode : IRNode {
    using IRNode::IRNode;
};

// Workload node base
struct WorkloadNode : IRNode {
    using IRNode::IRNode;
};

// Schedule node base
struct ScheduleNode : IRNode {
    using IRNode::IRNode;
};

// Utility functions
[[nodiscard]] const char* nodeKindToString(NodeKind kind);
[[nodiscard]] const char* dtypeToString(DType dtype);
[[nodiscard]] const char* locationToString(Location loc);
[[nodiscard]] const char* levelToString(WorkloadLevel level);

}  // namespace pto::wsp::ir
