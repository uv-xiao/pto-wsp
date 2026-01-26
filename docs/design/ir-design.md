# PTO Workload-Schedule Programming (PTO-WSP) v9: IR Design

## 1. Overview

This document specifies the C++ Intermediate Representation (IR) for v9 runtime extension. The IR serves as:

1. **Canonical representation** of workloads and schedules
2. **Serialization format** via `.pto` assembly
3. **Input to backend code generators**
4. **Optimization target** for IR-level transformations

### 1.1 Design Principles

| Principle | Description |
|-----------|-------------|
| **Explicit** | All information needed for code generation is in the IR |
| **Immutable** | IR nodes are immutable after construction |
| **Typed** | Strong typing for safety and optimization |
| **Serializable** | Bidirectional conversion to `.pto` assembly |

### 1.2 IR Layers

```
┌─────────────────────────────────────────┐
│           Workload IR                    │
│  (What to compute)                       │
├─────────────────────────────────────────┤
│           Schedule IR                    │
│  (How to execute)                        │
├─────────────────────────────────────────┤
│           Combined IR                    │
│  (Workload + Schedule → Backend input)   │
└─────────────────────────────────────────┘
```

---

## 2. Common Infrastructure

### 2.1 Base Classes

```cpp
namespace pto::wsp::ir {

// Forward declarations
struct IRNode;
struct AxisNode;
struct WorkloadNode;
struct ScheduleNode;

// Unique ID for each node
using NodeId = uint64_t;

// Workload hierarchy level (NEW in v9)
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
    Call,       // NEW: cross-level call (hierarchy)

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
    DispatchThreshold,  // NEW: multi-level dispatch
    PipelineDepth,      // NEW: in-flight control
    TaskWindow,         // NEW: metadata window
    BatchDeps,          // NEW: batched dependency resolution

    // Extension
    Ext,        // NEW: extension op (escape hatch)
};

// Base IR node
struct IRNode {
    const NodeId id;
    const NodeKind kind;
    const WorkloadLevel level;  // NEW: hierarchy level

    IRNode(NodeId id, NodeKind kind, WorkloadLevel level = WorkloadLevel::Any)
        : id(id), kind(kind), level(level) {}
    virtual ~IRNode() = default;

    // Printing
    virtual void print(std::ostream& os, int indent = 0) const = 0;

    // Child traversal (for visitor pattern)
    using ChildFn = std::function<void(const std::shared_ptr<const IRNode>&)>;
    virtual void forEachChild(const ChildFn&) const {}  // Default: leaf node

    // Visitor pattern
    template<typename Visitor>
    auto accept(Visitor& v) -> decltype(v.visit(*this));
};

// Shared pointer type for IR nodes
template<typename T>
using IRPtr = std::shared_ptr<T>;

// Node factory with ID generation
class IRFactory {
    std::atomic<NodeId> next_id{0};

public:
    template<typename T, typename... Args>
    IRPtr<T> create(Args&&... args) {
        return std::make_shared<T>(next_id++, std::forward<Args>(args)...);
    }
};

}  // namespace pto::wsp::ir
```

### 2.2 Type Representation

```cpp
namespace pto::wsp::ir {

// Element types
enum class DType {
    F16, BF16, F32, F64,
    I8, I16, I32, I64,
    U8, U16, U32, U64,
    Bool,
};

// Memory locations
enum class Location {
    Global,  // HBM
    L2,      // L2 cache
    UB,      // Unified buffer
    L1,      // L1 buffer
};

// Shape representation
struct Shape {
    std::vector<int64_t> dims;

    bool is_static() const;
    int64_t numel() const;
    int64_t rank() const { return dims.size(); }
};

// Tensor type
struct TensorType {
    Shape shape;
    DType dtype;
    Location location;

    std::string to_string() const;
};

// Stream/Channel type
struct ChannelType {
    TensorType element_type;  // Type of elements in channel
    int64_t capacity;         // Buffer depth (0 = rendezvous)

    std::string to_string() const;
};

}  // namespace pto::wsp::ir
```

---

## 3. Axis IR

Axis nodes represent iteration spaces.

### 3.1 Dense Axis (Static)

```cpp
namespace pto::wsp::ir {

struct DenseAxisNode : IRNode {
    const int64_t size;  // Static size

    DenseAxisNode(NodeId id, int64_t size)
        : IRNode(id, NodeKind::DenseAxis), size(size) {}

    void print(std::ostream& os, int indent) const override {
        os << std::string(indent, ' ') << "Dense[" << size << "]";
    }
};

}
```

### 3.2 DenseDyn Axis (Dynamic)

```cpp
namespace pto::wsp::ir {

struct DenseDynAxisNode : IRNode {
    const std::string size_var;  // Variable name for size

    DenseDynAxisNode(NodeId id, std::string size_var)
        : IRNode(id, NodeKind::DenseDynAxis), size_var(std::move(size_var)) {}

    void print(std::ostream& os, int indent) const override {
        os << std::string(indent, ' ') << "DenseDyn(" << size_var << ")";
    }
};

}
```

### 3.3 Ragged Axis

```cpp
namespace pto::wsp::ir {

struct RaggedAxisNode : IRNode {
    const std::string outer_size_var;
    const std::string lengths_var;

    RaggedAxisNode(NodeId id, std::string outer, std::string lengths)
        : IRNode(id, NodeKind::RaggedAxis),
          outer_size_var(std::move(outer)),
          lengths_var(std::move(lengths)) {}

    void print(std::ostream& os, int indent) const override {
        os << std::string(indent, ' ') << "Ragged(" << outer_size_var
           << ", " << lengths_var << ")";
    }
};

}
```

### 3.4 Sparse Axis

```cpp
namespace pto::wsp::ir {

struct SparseAxisNode : IRNode {
    const std::string outer_size_var;
    const std::string indptr_var;
    const std::string indices_var;

    SparseAxisNode(NodeId id, std::string outer, std::string indptr, std::string indices)
        : IRNode(id, NodeKind::SparseAxis),
          outer_size_var(std::move(outer)),
          indptr_var(std::move(indptr)),
          indices_var(std::move(indices)) {}

    void print(std::ostream& os, int indent) const override {
        os << std::string(indent, ' ') << "Sparse(" << outer_size_var
           << ", " << indptr_var << ", " << indices_var << ")";
    }
};

}
```

---

## 4. Workload IR

### 4.1 Task Node

```cpp
namespace pto::wsp::ir {

struct TaskNode : IRNode {
    const std::string kernel_name;
    const std::vector<std::string> params;
    const std::vector<std::string> resources;

    TaskNode(NodeId id, std::string kernel,
             std::vector<std::string> params,
             std::vector<std::string> resources)
        : IRNode(id, NodeKind::Task),
          kernel_name(std::move(kernel)),
          params(std::move(params)),
          resources(std::move(resources)) {}

    void print(std::ostream& os, int indent) const override {
        os << std::string(indent, ' ') << "task @" << kernel_name << "(";
        for (size_t i = 0; i < params.size(); ++i) {
            if (i > 0) os << ", ";
            os << params[i];
        }
        os << ") resources(";
        for (size_t i = 0; i < resources.size(); ++i) {
            if (i > 0) os << ", ";
            os << resources[i];
        }
        os << ")";
    }
};

}
```

### 4.2 ParallelFor Node

```cpp
namespace pto::wsp::ir {

struct ParallelForNode : IRNode {
    const IRPtr<AxisNode> axis;
    const std::string index_var;
    const IRPtr<WorkloadNode> body;

    ParallelForNode(NodeId id, IRPtr<AxisNode> axis, std::string index_var,
                    IRPtr<WorkloadNode> body)
        : IRNode(id, NodeKind::ParallelFor),
          axis(std::move(axis)),
          index_var(std::move(index_var)),
          body(std::move(body)) {}

    void print(std::ostream& os, int indent) const override {
        os << std::string(indent, ' ') << "parallel_for " << index_var << " in ";
        axis->print(os, 0);
        os << " {\n";
        body->print(os, indent + 2);
        os << "\n" << std::string(indent, ' ') << "}";
    }
};

}
```

### 4.3 ForEach Node

```cpp
namespace pto::wsp::ir {

struct ForEachNode : IRNode {
    const IRPtr<AxisNode> axis;
    const std::string index_var;
    const IRPtr<WorkloadNode> body;

    ForEachNode(NodeId id, IRPtr<AxisNode> axis, std::string index_var,
                IRPtr<WorkloadNode> body)
        : IRNode(id, NodeKind::ForEach),
          axis(std::move(axis)),
          index_var(std::move(index_var)),
          body(std::move(body)) {}

    void print(std::ostream& os, int indent) const override {
        os << std::string(indent, ' ') << "for_each " << index_var << " in ";
        axis->print(os, 0);
        os << " {\n";
        body->print(os, indent + 2);
        os << "\n" << std::string(indent, ' ') << "}";
    }
};

}
```

### 4.4 Select Node

```cpp
namespace pto::wsp::ir {

struct SelectNode : IRNode {
    const IRPtr<SparseAxisNode> sparse;
    const std::string index_var;
    const IRPtr<WorkloadNode> body;

    SelectNode(NodeId id, IRPtr<SparseAxisNode> sparse, std::string index_var,
               IRPtr<WorkloadNode> body)
        : IRNode(id, NodeKind::Select),
          sparse(std::move(sparse)),
          index_var(std::move(index_var)),
          body(std::move(body)) {}

    void print(std::ostream& os, int indent) const override {
        os << std::string(indent, ' ') << "select " << index_var << " in ";
        sparse->print(os, 0);
        os << " {\n";
        body->print(os, indent + 2);
        os << "\n" << std::string(indent, ' ') << "}";
    }
};

}
```

### 4.5 Cond Node

```cpp
namespace pto::wsp::ir {

struct CondNode : IRNode {
    const std::string predicate;  // Condition expression
    const IRPtr<WorkloadNode> then_branch;
    const IRPtr<WorkloadNode> else_branch;

    CondNode(NodeId id, std::string predicate,
             IRPtr<WorkloadNode> then_branch,
             IRPtr<WorkloadNode> else_branch)
        : IRNode(id, NodeKind::Cond),
          predicate(std::move(predicate)),
          then_branch(std::move(then_branch)),
          else_branch(std::move(else_branch)) {}

    void print(std::ostream& os, int indent) const override {
        os << std::string(indent, ' ') << "cond " << predicate << " {\n";
        then_branch->print(os, indent + 2);
        os << "\n" << std::string(indent, ' ') << "} else {\n";
        else_branch->print(os, indent + 2);
        os << "\n" << std::string(indent, ' ') << "}";
    }
};

}
```

### 4.6 Combine Node

```cpp
namespace pto::wsp::ir {

struct CombineNode : IRNode {
    const std::vector<IRPtr<WorkloadNode>> workloads;

    CombineNode(NodeId id, std::vector<IRPtr<WorkloadNode>> workloads)
        : IRNode(id, NodeKind::Combine), workloads(std::move(workloads)) {}

    void print(std::ostream& os, int indent) const override {
        os << std::string(indent, ' ') << "combine {\n";
        for (const auto& w : workloads) {
            w->print(os, indent + 2);
            os << "\n";
        }
        os << std::string(indent, ' ') << "}";
    }
};

}
```

### 4.7 Sequential Node

```cpp
namespace pto::wsp::ir {

struct SequentialNode : IRNode {
    const std::vector<IRPtr<WorkloadNode>> workloads;

    SequentialNode(NodeId id, std::vector<IRPtr<WorkloadNode>> workloads)
        : IRNode(id, NodeKind::Sequential), workloads(std::move(workloads)) {}

    void print(std::ostream& os, int indent) const override {
        os << std::string(indent, ' ') << "sequential {\n";
        for (const auto& w : workloads) {
            w->print(os, indent + 2);
            os << "\n";
        }
        os << std::string(indent, ' ') << "}";
    }
};

}
```

### 4.8 Call Node (Cross-Level Hierarchy)

The `CallNode` enables hierarchical workload composition by referencing another workload definition (typically at a different level, e.g., CPU calling NPU).

```cpp
namespace pto::wsp::ir {

struct CallNode : IRNode {
    const std::string callee_workload;          // Symbol name of target workload
    const std::optional<std::string> callee_schedule;  // Optional schedule override
    const WorkloadLevel callee_level;           // Expected level (usually NPU when called from CPU)
    const std::vector<std::string> args;        // Arguments passed to callee
    const std::vector<std::string> resources;   // Resource bindings

    CallNode(NodeId id, std::string callee, WorkloadLevel callee_level,
             std::vector<std::string> args, std::vector<std::string> resources,
             std::optional<std::string> schedule = std::nullopt)
        : IRNode(id, NodeKind::Call, WorkloadLevel::CPU),  // Call is CPU-level
          callee_workload(std::move(callee)),
          callee_schedule(std::move(schedule)),
          callee_level(callee_level),
          args(std::move(args)),
          resources(std::move(resources)) {}

    void print(std::ostream& os, int indent) const override {
        os << std::string(indent, ' ') << "call @" << callee_workload;
        if (callee_schedule) {
            os << " with @" << *callee_schedule;
        }
        os << "(";
        for (size_t i = 0; i < args.size(); ++i) {
            if (i > 0) os << ", ";
            os << args[i];
        }
        os << ") resources(";
        for (size_t i = 0; i < resources.size(); ++i) {
            if (i > 0) os << ", ";
            os << resources[i];
        }
        os << ")";
    }

    void forEachChild(const ChildFn&) const override {
        // No IR children; callee is resolved via symbol table
    }
};

}
```

**Usage**: The `CallNode` enables the two-level hierarchical model where CPU-level workloads invoke NPU-level InCore functions:

```python
# Python frontend
@workload(level=CPU)
def attention(batch, heads):
    for b, h in P.grid(batch, heads):
        call("attn_incore", level=NPU, args=[b, h], resources=[...])

@workload(level=NPU)
def attn_incore(b, h):
    # InCore tile operations
    ...
```

---

## 5. CSP IR

### 5.1 Channel Node

```cpp
namespace pto::wsp::ir {

struct ChannelNode : IRNode {
    const std::string name;
    const ChannelType type;

    ChannelNode(NodeId id, std::string name, ChannelType type)
        : IRNode(id, NodeKind::Channel),
          name(std::move(name)),
          type(std::move(type)) {}

    bool is_event() const { return type.capacity == 0; }

    void print(std::ostream& os, int indent) const override {
        os << std::string(indent, ' ') << "channel " << name << " : "
           << type.to_string();
    }
};

}
```

### 5.2 Process Node

```cpp
namespace pto::wsp::ir {

struct ProcessNode : IRNode {
    const std::string name;
    const std::vector<std::string> consumes;  // Channel names
    const std::vector<std::string> produces;  // Channel names
    const IRPtr<WorkloadNode> body;

    ProcessNode(NodeId id, std::string name,
                std::vector<std::string> consumes,
                std::vector<std::string> produces,
                IRPtr<WorkloadNode> body)
        : IRNode(id, NodeKind::Process),
          name(std::move(name)),
          consumes(std::move(consumes)),
          produces(std::move(produces)),
          body(std::move(body)) {}

    void print(std::ostream& os, int indent) const override {
        os << std::string(indent, ' ') << "process @" << name;
        if (!consumes.empty()) {
            os << " consumes(";
            for (size_t i = 0; i < consumes.size(); ++i) {
                if (i > 0) os << ", ";
                os << consumes[i];
            }
            os << ")";
        }
        if (!produces.empty()) {
            os << " produces(";
            for (size_t i = 0; i < produces.size(); ++i) {
                if (i > 0) os << ", ";
                os << produces[i];
            }
            os << ")";
        }
        os << " {\n";
        body->print(os, indent + 2);
        os << "\n" << std::string(indent, ' ') << "}";
    }
};

}
```

### 5.3 Send Node

```cpp
namespace pto::wsp::ir {

struct SendNode : IRNode {
    const std::string channel_name;
    const IRPtr<WorkloadNode> value;

    SendNode(NodeId id, std::string channel, IRPtr<WorkloadNode> value)
        : IRNode(id, NodeKind::Send),
          channel_name(std::move(channel)),
          value(std::move(value)) {}

    void print(std::ostream& os, int indent) const override {
        os << std::string(indent, ' ') << "send " << channel_name << ", ";
        value->print(os, 0);
    }
};

}
```

### 5.4 Consume Node

```cpp
namespace pto::wsp::ir {

struct ConsumeNode : IRNode {
    const std::string channel_name;
    const std::string value_var;
    const IRPtr<WorkloadNode> body;

    ConsumeNode(NodeId id, std::string channel, std::string value_var,
                IRPtr<WorkloadNode> body)
        : IRNode(id, NodeKind::Consume),
          channel_name(std::move(channel)),
          value_var(std::move(value_var)),
          body(std::move(body)) {}

    void print(std::ostream& os, int indent) const override {
        os << std::string(indent, ' ') << "consume " << channel_name
           << " as " << value_var << " {\n";
        body->print(os, indent + 2);
        os << "\n" << std::string(indent, ' ') << "}";
    }
};

}
```

### 5.5 Pipeline Node

```cpp
namespace pto::wsp::ir {

struct PipelineNode : IRNode {
    const std::string name;
    const std::vector<IRPtr<ChannelNode>> channels;
    const std::vector<IRPtr<ProcessNode>> processes;

    PipelineNode(NodeId id, std::string name,
                 std::vector<IRPtr<ChannelNode>> channels,
                 std::vector<IRPtr<ProcessNode>> processes)
        : IRNode(id, NodeKind::Pipeline),
          name(std::move(name)),
          channels(std::move(channels)),
          processes(std::move(processes)) {}

    void print(std::ostream& os, int indent) const override {
        os << std::string(indent, ' ') << "pipeline @" << name << " {\n";
        for (const auto& ch : channels) {
            ch->print(os, indent + 2);
            os << "\n";
        }
        os << "\n";
        for (const auto& p : processes) {
            p->print(os, indent + 2);
            os << "\n\n";
        }
        os << std::string(indent, ' ') << "}";
    }
};

}
```

---

## 6. Schedule IR

### 6.1 Dispatch Node

```cpp
namespace pto::wsp::ir {

enum class DispatchPolicy {
    RoundRobin,
    Affinity,
    Hash,
    WorkSteal,
    Custom,
};

struct DispatchNode : IRNode {
    const DispatchPolicy policy;
    const int num_targets;           // For RoundRobin
    const std::string key_expr;      // For Affinity, Hash, Custom

    DispatchNode(NodeId id, DispatchPolicy policy, int num_targets = 0,
                 std::string key_expr = "")
        : IRNode(id, NodeKind::Dispatch),
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

}
```

### 6.2 Stream Node

```cpp
namespace pto::wsp::ir {

struct StreamNode : IRNode {
    const int num_streams;
    const std::string key_expr;  // For stream_by (empty = single stream)

    StreamNode(NodeId id, int num_streams, std::string key_expr = "")
        : IRNode(id, NodeKind::Stream),
          num_streams(num_streams),
          key_expr(std::move(key_expr)) {}

    void print(std::ostream& os, int indent) const override {
        os << std::string(indent, ' ') << "streams = " << num_streams;
        if (!key_expr.empty()) {
            os << "\n" << std::string(indent, ' ') << "stream_by = " << key_expr;
        }
    }
};

}
```

### 6.3 Timing Node

```cpp
namespace pto::wsp::ir {

enum class TimingPolicy {
    Immediate,
    Batched,
    Interleaved,
    RateLimited,
};

struct TimingNode : IRNode {
    const TimingPolicy policy;
    const int param;  // batch_size, num_streams, or rate

    TimingNode(NodeId id, TimingPolicy policy, int param = 0)
        : IRNode(id, NodeKind::Timing),
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

}
```

### 6.4 SpatialMap Node (New in v9)

```cpp
namespace pto::wsp::ir {

struct SpatialMapNode : IRNode {
    const std::vector<int64_t> grid;  // e.g., {4, 4} for 4x4

    SpatialMapNode(NodeId id, std::vector<int64_t> grid)
        : IRNode(id, NodeKind::SpatialMap),
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

}
```

### 6.5 Layout Node (New in v9)

```cpp
namespace pto::wsp::ir {

enum class LayoutKind {
    Shard,
    Replicate,
};

struct LayoutDim {
    LayoutKind kind;
    int shard_dim;  // For Shard

    static LayoutDim shard(int dim) { return {LayoutKind::Shard, dim}; }
    static LayoutDim replicate() { return {LayoutKind::Replicate, -1}; }
};

struct LayoutNode : IRNode {
    const std::string tensor_name;
    const std::vector<LayoutDim> dims;

    LayoutNode(NodeId id, std::string tensor, std::vector<LayoutDim> dims)
        : IRNode(id, NodeKind::Layout),
          tensor_name(std::move(tensor)),
          dims(std::move(dims)) {}

    void print(std::ostream& os, int indent) const override {
        os << std::string(indent, ' ') << "layout " << tensor_name << " = (";
        for (size_t i = 0; i < dims.size(); ++i) {
            if (i > 0) os << ", ";
            if (dims[i].kind == LayoutKind::Shard) {
                os << "Shard(" << dims[i].shard_dim << ")";
            } else {
                os << "Replicate";
            }
        }
        os << ")";
    }
};

}
```

---

## 7. Module IR

### 7.1 Workload Definition

```cpp
namespace pto::wsp::ir {

struct WorkloadDef {
    std::string name;
    WorkloadLevel level;  // NEW: CPU or NPU level
    std::vector<std::pair<std::string, IRPtr<AxisNode>>> params;  // (name, axis)
    IRPtr<WorkloadNode> body;

    void print(std::ostream& os) const {
        os << "@workload";
        if (level == WorkloadLevel::NPU) os << "[npu]";
        os << " " << name << "(";
        for (size_t i = 0; i < params.size(); ++i) {
            if (i > 0) os << ", ";
            os << params[i].first << ": ";
            params[i].second->print(os, 0);
        }
        os << ") {\n";
        body->print(os, 2);
        os << "\n}\n";
    }
};

}
```

### 7.2 Schedule Definition

The schedule definition uses a **directives list** instead of fixed optional fields. This enables adding new schedule primitives without changing the struct:

```cpp
namespace pto::wsp::ir {

struct ScheduleDef {
    std::string name;
    std::string workload_name;
    WorkloadLevel level;  // Should match workload's level

    // Extensible: directives list instead of fixed fields
    std::vector<IRPtr<ScheduleNode>> directives;

    // Helper accessors (search directives by kind)
    IRPtr<DispatchNode> dispatch() const {
        for (const auto& d : directives) {
            if (d->kind == NodeKind::Dispatch)
                return std::static_pointer_cast<DispatchNode>(d);
        }
        return nullptr;
    }

    IRPtr<StreamNode> stream() const {
        for (const auto& d : directives) {
            if (d->kind == NodeKind::Stream)
                return std::static_pointer_cast<StreamNode>(d);
        }
        return nullptr;
    }

    IRPtr<TimingNode> timing() const {
        for (const auto& d : directives) {
            if (d->kind == NodeKind::Timing)
                return std::static_pointer_cast<TimingNode>(d);
        }
        return nullptr;
    }

    void print(std::ostream& os) const {
        os << "@schedule " << name << " for @" << workload_name << " {\n";
        for (const auto& d : directives) {
            d->print(os, 2);
            os << "\n";
        }
        os << "}\n";
    }
};

}
```

### 7.3 Module

```cpp
namespace pto::wsp::ir {

struct Module {
    std::string name;
    std::string version;
    std::vector<std::string> targets;  // cpu_sim, ascend_npu, amd_aie

    std::vector<WorkloadDef> workloads;
    std::vector<ScheduleDef> schedules;
    std::vector<IRPtr<PipelineNode>> pipelines;

    void print(std::ostream& os) const {
        os << "// PTO Module: " << name << "\n";
        os << "// Version: " << version << "\n";
        os << "// Target: ";
        for (size_t i = 0; i < targets.size(); ++i) {
            if (i > 0) os << " | ";
            os << targets[i];
        }
        os << "\n\n";

        for (const auto& w : workloads) {
            w.print(os);
            os << "\n";
        }

        for (const auto& s : schedules) {
            s.print(os);
            os << "\n";
        }

        for (const auto& p : pipelines) {
            p->print(os, 0);
            os << "\n\n";
        }
    }
};

}
```

---

## 8. Assembly Format (.pto)

### 8.1 Grammar (EBNF)

```ebnf
module          = header { type_def } { workload_def } { schedule_def } { pipeline_def } ;

header          = "// PTO Module:" IDENT
                  "// Version:" VERSION
                  "// Target:" target_list ;

target_list     = target { "|" target } ;
target          = "cpu_sim" | "ascend_npu" | "amd_aie" ;

type_def        = "!" IDENT "=" type_expr ;
type_expr       = axis_type | channel_type ;
axis_type       = "Dense[" INT "]" | "DenseDyn" | "Ragged" | "Sparse" ;
channel_type    = "Channel[" type_ref "," INT "]" ;
type_ref        = "!" IDENT | type_expr ;

workload_def    = "@workload" IDENT "(" param_list ")" "{" workload_body "}" ;
param_list      = [ param { "," param } ] ;
param           = IDENT ":" type_ref ;
workload_body   = workload_stmt { workload_stmt } ;

workload_stmt   = parallel_for | for_each | select | cond | task | combine | sequential
                | send | consume ;

parallel_for    = "parallel_for" IDENT "in" axis_expr "{" workload_body "}" ;
for_each        = "for_each" IDENT "in" axis_expr "{" workload_body "}" ;
select          = "select" IDENT "in" sparse_expr "{" workload_body "}" ;
cond            = "cond" expr "{" workload_body "}" "else" "{" workload_body "}" ;
task            = [ IDENT "=" ] "task" "@" IDENT "(" expr_list ")" "resources" "(" expr_list ")" ;
combine         = "combine" "{" workload_body "}" ;
sequential      = "sequential" "{" workload_body "}" ;
send            = "send" IDENT "," workload_stmt ;
consume         = "consume" IDENT "as" IDENT "{" workload_body "}" ;

axis_expr       = type_ref | "DenseDyn(" IDENT ")" | ... ;
sparse_expr     = IDENT "[" IDENT "]" ;
expr_list       = [ expr { "," expr } ] ;
expr            = IDENT | INT | IDENT "[" expr "]" | ... ;

schedule_def    = "@schedule" IDENT "for" "@" IDENT "{" schedule_body "}" ;
schedule_body   = { schedule_stmt } ;
schedule_stmt   = dispatch_stmt | stream_stmt | timing_stmt | spatial_stmt | layout_stmt ;
dispatch_stmt   = "dispatch" "=" dispatch_policy ;
dispatch_policy = "round_robin(" INT ")" | "affinity(" expr ")" | "hash(" expr ")"
                | "work_steal" | "dispatch_by(" expr ")" ;
stream_stmt     = "streams" "=" INT [ "stream_by" "=" expr ] ;
timing_stmt     = "timing" "=" timing_policy ;
timing_policy   = "immediate" | "batched(" INT ")" | "interleaved(" INT ")"
                | "rate_limit(" INT ")" ;
spatial_stmt    = "spatial_map" "=" "(" INT { "," INT } ")" ;
layout_stmt     = "layout" IDENT "=" "(" layout_dim { "," layout_dim } ")" ;
layout_dim      = "Shard(" INT ")" | "Replicate" ;

pipeline_def    = "@pipeline" IDENT "{" channel_decl { channel_decl }
                  process_def { process_def } "}" ;
channel_decl    = "channel" IDENT ":" type_ref ;
process_def     = "process" "@" IDENT [ "consumes(" IDENT_LIST ")" ]
                  [ "produces(" IDENT_LIST ")" ] "{" workload_body "}" ;
```

### 8.2 Example

```
// PTO Module: attention_example
// Version: 9.0
// Target: cpu_sim | ascend_npu

// Type definitions
!batch = DenseDyn
!heads = Dense[8]
!l2c = Channel[Task, 2]

// Workload definition
@workload attention(%batch: !batch, %heads: !heads) {
  parallel_for %b in %batch {
    parallel_for %h in %heads {
      %t = task @attn_kernel(%b, %h) resources(%Q[%b][%h], %K[%b], %V[%b], %O[%b][%h])
      yield %t
    }
  }
}

// Schedule definition
@schedule attention_sched for @attention {
  dispatch = affinity(%b)
  streams = 2
  stream_by = %h mod 2
  timing = immediate
}

// CSP Pipeline
@pipeline megakernel {
  channel %l2c : !l2c
  channel %c2s : !l2c

  process @loader produces(%l2c) {
    for_each %i in DenseDyn(%num_tiles) {
      %t = task @load_kernel(%i) resources(%input[%i], %buf)
      send %l2c, %t
    }
  }

  process @computer consumes(%l2c) produces(%c2s) {
    consume %l2c as %tile {
      %r = task @compute_kernel(%tile) resources(%buf, %out)
      send %c2s, %r
    }
  }

  process @storer consumes(%c2s) {
    consume %c2s as %result {
      task @store_kernel(%result) resources(%out, %final)
    }
  }
}
```

---

## 9. Serialization API

### 9.1 Printer

```cpp
namespace pto::wsp::ir {

class Printer {
public:
    static void print(const Module& module, std::ostream& os);
    static std::string to_string(const Module& module);
    static void to_file(const Module& module, const std::string& path);
};

}
```

### 9.2 Parser

```cpp
namespace pto::wsp::ir {

class Parser {
public:
    static Module parse(std::istream& is);
    static Module parse(const std::string& source);
    static Module from_file(const std::string& path);
};

// Parsing errors
class ParseError : public std::runtime_error {
    int line;
    int column;
    std::string message;
public:
    ParseError(int line, int column, std::string msg);
};

}
```

---

## 10. Extension Mechanism

The extension mechanism enables adding new IR node types without modifying core headers. Key design principles:

1. **Closed core, open extensions**: Core `NodeKind` enum is fixed; extensions use `NodeKind::Ext`
2. **Name-based dispatch**: Extended ops identified by qualified name (e.g., `"npu.double_buffer"`)
3. **Self-registration**: Extensions register via static initializers when linked
4. **Two-layer visitor**: Virtual dispatch for core nodes; registry lookup for extensions

### 10.1 ExtOpNode Base Class

```cpp
namespace pto::wsp::ir {

// Attribute value types
using AttrValue = std::variant<
    int64_t, double, bool, std::string,
    std::vector<int64_t>, std::vector<std::string>
>;
using AttrMap = std::unordered_map<std::string, AttrValue>;

// Extension op classification
enum class ExtClass : uint8_t {
    Axis,      // Custom axis types
    Workload,  // Custom workload ops
    Schedule,  // Custom schedule directives
    CSP,       // Custom CSP primitives
    Backend,   // Backend-specific annotations
};

// Base class for all extension operations
struct ExtOpNode : IRNode {
    const ExtClass ext_class;
    const std::string op_name;  // Qualified name: "npu.double_buffer"
    const AttrMap attrs;
    const std::vector<IRPtr<IRNode>> children;

    ExtOpNode(NodeId id, ExtClass cls, std::string name, AttrMap attrs,
              std::vector<IRPtr<IRNode>> children = {},
              WorkloadLevel level = WorkloadLevel::Any)
        : IRNode(id, NodeKind::Ext, level),
          ext_class(cls), op_name(std::move(name)),
          attrs(std::move(attrs)), children(std::move(children)) {}

    // Attribute accessors with type checking
    template<typename T>
    std::optional<T> getAttr(const std::string& key) const {
        auto it = attrs.find(key);
        if (it == attrs.end()) return std::nullopt;
        if (auto* v = std::get_if<T>(&it->second)) return *v;
        return std::nullopt;
    }

    template<typename T>
    T getAttrOr(const std::string& key, T default_val) const {
        return getAttr<T>(key).value_or(default_val);
    }

    void forEachChild(const ChildFn& fn) const override {
        for (const auto& c : children) fn(c);
    }

    void print(std::ostream& os, int indent) const override;
};

}
```

### 10.2 Typed Extension Nodes (Subclasses)

For frequently-used extensions, define typed subclasses with accessor methods:

```cpp
namespace pto::wsp::ir::ext {

// NPU-specific: Double buffering directive
struct DoubleBufferNode : ExtOpNode {
    std::string buffer_name() const { return getAttrOr<std::string>("buffer", "ub"); }
    int64_t depth() const { return getAttrOr<int64_t>("depth", 2); }

    DoubleBufferNode(NodeId id, std::string buffer, int64_t depth,
                     WorkloadLevel level = WorkloadLevel::NPU)
        : ExtOpNode(id, ExtClass::Schedule, "npu.double_buffer",
                    {{"buffer", buffer}, {"depth", depth}}, {}, level) {}

    // Factory for registration
    static IRPtr<ExtOpNode> create(IRFactory& f, const AttrMap& attrs);
};

// Schedule extension: pipeline_depth
struct PipelineDepthNode : ExtOpNode {
    int64_t depth() const { return getAttrOr<int64_t>("depth", 2); }
    std::string scope() const { return getAttrOr<std::string>("scope", "per_stream"); }

    PipelineDepthNode(NodeId id, int64_t depth, std::string scope)
        : ExtOpNode(id, ExtClass::Schedule, "schedule.pipeline_depth",
                    {{"depth", depth}, {"scope", std::move(scope)}}) {}
};

// Schedule extension: task_window
struct TaskWindowNode : ExtOpNode {
    int64_t size() const { return getAttrOr<int64_t>("size", 8192); }
    std::string overflow() const { return getAttrOr<std::string>("overflow", "stall"); }

    TaskWindowNode(NodeId id, int64_t size, std::string overflow)
        : ExtOpNode(id, ExtClass::Schedule, "schedule.task_window",
                    {{"size", size}, {"overflow", std::move(overflow)}}) {}
};

}
```

### 10.3 Extension Registry

Extensions self-register handlers and factories via a global registry:

```cpp
namespace pto::wsp::ir {

// Handler function types
using ExtVisitorFn = std::function<WalkControl(IRVisitor&, const ExtOpNode&)>;
using ExtLeaveFn = std::function<void(IRVisitor&, const ExtOpNode&)>;
using ExtFactoryFn = std::function<IRPtr<ExtOpNode>(IRFactory&, const AttrMap&)>;
using ExtRewriteFn = std::function<IRPtr<IRNode>(IRRewriter&, IRFactory&, const IRPtr<ExtOpNode>&)>;

class ExtOpRegistry {
public:
    static ExtOpRegistry& instance();

    // Visitor registration
    void registerVisitor(const std::string& op_name, ExtVisitorFn enter,
                         ExtLeaveFn leave = nullptr);

    // Factory registration (for parsing)
    void registerFactory(const std::string& op_name, ExtFactoryFn factory);

    // Rewriter registration
    void registerRewriter(const std::string& op_name, ExtRewriteFn rewrite);

    // Lookup
    std::optional<ExtVisitorFn> getEnterHandler(const std::string& op_name) const;
    std::optional<ExtLeaveFn> getLeaveHandler(const std::string& op_name) const;
    std::optional<ExtRewriteFn> getRewriter(const std::string& op_name) const;

    // Check if op is registered
    bool hasOp(const std::string& op_name) const;

    // Create extension node from attributes
    IRPtr<ExtOpNode> create(const std::string& op_name, IRFactory& f, const AttrMap& attrs);

private:
    struct Entry {
        ExtVisitorFn enter;
        ExtLeaveFn leave;
        ExtFactoryFn factory;
        ExtRewriteFn rewrite;
    };
    std::unordered_map<std::string, Entry> registry_;
};

// Self-registration macro
#define REGISTER_EXT_OP(op_name, enter_fn, leave_fn, factory_fn) \
    static bool _reg_##__COUNTER__ = []() { \
        ExtOpRegistry::instance().registerVisitor(op_name, enter_fn, leave_fn); \
        ExtOpRegistry::instance().registerFactory(op_name, factory_fn); \
        return true; \
    }()

#define REGISTER_EXT_REWRITER(op_name, rewrite_fn) \
    static bool _rewrite_reg_##__COUNTER__ = []() { \
        ExtOpRegistry::instance().registerRewriter(op_name, rewrite_fn); \
        return true; \
    }()

}
```

### 10.4 Extension Registration Example

```cpp
// src/pto/rt/ir/ext/npu/npu_ext.cpp
#include "pto/rt/ir/ext.hpp"
#include "pto/rt/ir/ext/npu/double_buffer.hpp"

namespace pto::wsp::ir::ext {

// Static registration - runs before main()
REGISTER_EXT_OP(
    "npu.double_buffer",
    // enter handler
    [](IRVisitor& v, const ExtOpNode& n) -> WalkControl {
        return v.enter(static_cast<const IRNode&>(n));
    },
    // leave handler
    nullptr,
    // factory
    DoubleBufferNode::create
);

}
```

### 10.5 CMake Integration

Extensions are built as separate libraries that self-register when linked:

```cmake
# Core IR library (always built)
add_library(pto_ir
    src/pto/rt/ir/core.cpp
    src/pto/rt/ir/visitor.cpp
    src/pto/rt/ir/rewriter.cpp
    src/pto/rt/ir/pass.cpp
)
target_include_directories(pto_ir PUBLIC include)
target_compile_features(pto_ir PUBLIC cxx_std_23)

# NPU extensions (optional)
option(PTO_WSP_NPU_EXT "Build NPU IR extensions" ON)
if(PTO_WSP_NPU_EXT)
    add_library(pto_ir_npu_ext
        src/pto/rt/ir/ext/npu/npu_ext.cpp
    )
    target_link_libraries(pto_ir_npu_ext PUBLIC pto_ir)
endif()

# AIE extensions (optional)
option(PTO_WSP_AIE_EXT "Build AIE IR extensions" OFF)
if(PTO_WSP_AIE_EXT)
    add_library(pto_ir_aie_ext
        src/pto/rt/ir/ext/aie/aie_ext.cpp
    )
    target_link_libraries(pto_ir_aie_ext PUBLIC pto_ir)
endif()

# Backend links extension library
target_link_libraries(pto_backend_npu PRIVATE pto_ir_npu_ext)
```

### 10.6 Directory Structure

```
include/pto/rt/ir/
  core.hpp          # NodeId, NodeKind, IRNode, IRFactory
  nodes.hpp         # All core node types
  ext.hpp           # ExtOpNode, ExtOpRegistry, macros
  visitor.hpp       # IRVisitor, walk()
  rewriter.hpp      # IRRewriter
  pass.hpp          # Pass, PassManager
  ir.hpp            # Umbrella include

  ext/              # Extension headers
    npu/
      double_buffer.hpp
      prefetch.hpp
    aie/
      tile_placement.hpp
    schedule/
      pipeline_depth.hpp
      task_window.hpp

src/pto/rt/ir/
  core.cpp
  visitor.cpp
  rewriter.cpp
  pass.cpp

  ext/
    npu/npu_ext.cpp     # NPU extension registration
    aie/aie_ext.cpp     # AIE extension registration
```

---

## 11. Visitor and Traversal

### 11.1 IRVisitor

The visitor pattern enables traversing the IR tree without modifying node classes:

```cpp
namespace pto::wsp::ir {

// Walk control for visitor callbacks
enum class WalkControl {
    Continue,      // Continue traversal
    SkipChildren,  // Skip children of this node
    Abort,         // Stop entire traversal
};

// Base visitor with typed hooks
struct IRVisitor {
    virtual ~IRVisitor() = default;

    // Generic fallback
    virtual WalkControl enter(const IRNode&) { return WalkControl::Continue; }
    virtual void leave(const IRNode&) {}

    // Axis nodes
    virtual WalkControl enter(const DenseAxisNode& n) { return enter(static_cast<const IRNode&>(n)); }
    virtual WalkControl enter(const DenseDynAxisNode& n) { return enter(static_cast<const IRNode&>(n)); }
    virtual WalkControl enter(const RaggedAxisNode& n) { return enter(static_cast<const IRNode&>(n)); }
    virtual WalkControl enter(const SparseAxisNode& n) { return enter(static_cast<const IRNode&>(n)); }

    // Workload nodes
    virtual WalkControl enter(const TaskNode& n) { return enter(static_cast<const IRNode&>(n)); }
    virtual WalkControl enter(const ParallelForNode& n) { return enter(static_cast<const IRNode&>(n)); }
    virtual WalkControl enter(const ForEachNode& n) { return enter(static_cast<const IRNode&>(n)); }
    virtual WalkControl enter(const SelectNode& n) { return enter(static_cast<const IRNode&>(n)); }
    virtual WalkControl enter(const CondNode& n) { return enter(static_cast<const IRNode&>(n)); }
    virtual WalkControl enter(const CombineNode& n) { return enter(static_cast<const IRNode&>(n)); }
    virtual WalkControl enter(const SequentialNode& n) { return enter(static_cast<const IRNode&>(n)); }
    virtual WalkControl enter(const CallNode& n) { return enter(static_cast<const IRNode&>(n)); }

    // CSP nodes
    virtual WalkControl enter(const ChannelNode& n) { return enter(static_cast<const IRNode&>(n)); }
    virtual WalkControl enter(const ProcessNode& n) { return enter(static_cast<const IRNode&>(n)); }
    virtual WalkControl enter(const SendNode& n) { return enter(static_cast<const IRNode&>(n)); }
    virtual WalkControl enter(const ConsumeNode& n) { return enter(static_cast<const IRNode&>(n)); }
    virtual WalkControl enter(const PipelineNode& n) { return enter(static_cast<const IRNode&>(n)); }

    // Schedule nodes
    virtual WalkControl enter(const DispatchNode& n) { return enter(static_cast<const IRNode&>(n)); }
    virtual WalkControl enter(const StreamNode& n) { return enter(static_cast<const IRNode&>(n)); }
    virtual WalkControl enter(const TimingNode& n) { return enter(static_cast<const IRNode&>(n)); }
    virtual WalkControl enter(const SpatialMapNode& n) { return enter(static_cast<const IRNode&>(n)); }
    virtual WalkControl enter(const LayoutNode& n) { return enter(static_cast<const IRNode&>(n)); }

    // Extension (default: dispatch via registry)
    virtual WalkControl enter(const ExtOpNode& n) {
        return enterExt(n);
    }

protected:
    // Extensible dispatch for ExtOpNode via registry
    WalkControl enterExt(const ExtOpNode& n) {
        if (auto handler = ExtOpRegistry::instance().getEnterHandler(n.op_name)) {
            return (*handler)(*this, n);
        }
        // Fallback to base IRNode handler
        return enter(static_cast<const IRNode&>(n));
    }
};

}
```

### 11.2 Walk Function

```cpp
namespace pto::wsp::ir {

// Dispatch to typed visitor methods by NodeKind
WalkControl dispatchEnter(const IRPtr<IRNode>& node, IRVisitor& v);

// Recursive tree walk
WalkControl walk(const IRPtr<IRNode>& root, IRVisitor& v) {
    // 1. Dispatch enter() by node kind
    WalkControl ctrl = dispatchEnter(root, v);
    if (ctrl == WalkControl::Abort) return ctrl;
    if (ctrl == WalkControl::SkipChildren) {
        v.leave(*root);
        return WalkControl::Continue;
    }

    // 2. Recurse into children
    root->forEachChild([&v, &ctrl](const IRPtr<IRNode>& child) {
        if (ctrl != WalkControl::Abort) {
            ctrl = walk(child, v);
        }
    });

    // 3. Leave hook
    v.leave(*root);

    return ctrl == WalkControl::Abort ? ctrl : WalkControl::Continue;
}

// Walk all workloads/schedules/pipelines in a module
void walkModule(const Module& m, IRVisitor& v) {
    for (const auto& w : m.workloads) {
        walk(w.body, v);
    }
    for (const auto& s : m.schedules) {
        for (const auto& d : s.directives) {
            walk(d, v);
        }
    }
    for (const auto& p : m.pipelines) {
        walk(p, v);
    }
}

}
```

### 11.3 Symbol Table

For resolving `CallNode` references across workload definitions:

```cpp
namespace pto::wsp::ir {

class SymbolTable {
    std::unordered_map<std::string, const WorkloadDef*> workloads_;
    std::unordered_map<std::string, const ScheduleDef*> schedules_;
    std::unordered_map<std::string, const PipelineNode*> pipelines_;

public:
    static SymbolTable build(const Module& m);

    const WorkloadDef* lookupWorkload(const std::string& name) const;
    const ScheduleDef* lookupSchedule(const std::string& name) const;
    const PipelineNode* lookupPipeline(const std::string& name) const;
};

}
```

---

## 12. Pass Infrastructure

### 12.1 Diagnostics

```cpp
namespace pto::wsp::ir {

enum class DiagnosticSeverity { Note, Warning, Error };

struct Diagnostic {
    DiagnosticSeverity severity;
    std::string pass_name;
    std::string message;
    std::optional<NodeId> node_id;  // Optional source node

    std::string format() const {
        std::string prefix;
        switch (severity) {
            case DiagnosticSeverity::Note: prefix = "note"; break;
            case DiagnosticSeverity::Warning: prefix = "warning"; break;
            case DiagnosticSeverity::Error: prefix = "error"; break;
        }
        return "[" + pass_name + "] " + prefix + ": " + message;
    }
};

}
```

### 12.2 Pass Context and Result

```cpp
namespace pto::wsp::ir {

struct PassResult {
    bool changed = false;  // Whether the pass modified the IR
    bool ok = true;        // Whether the pass completed successfully
};

struct PassContext {
    IRFactory& factory;                  // For creating new nodes
    std::vector<Diagnostic>& diagnostics;  // Collect diagnostics
    const SymbolTable* symbols = nullptr;  // Optional symbol table

    void note(const std::string& pass, const std::string& msg,
              std::optional<NodeId> node = std::nullopt) {
        diagnostics.push_back({DiagnosticSeverity::Note, pass, msg, node});
    }

    void warn(const std::string& pass, const std::string& msg,
              std::optional<NodeId> node = std::nullopt) {
        diagnostics.push_back({DiagnosticSeverity::Warning, pass, msg, node});
    }

    void error(const std::string& pass, const std::string& msg,
               std::optional<NodeId> node = std::nullopt) {
        diagnostics.push_back({DiagnosticSeverity::Error, pass, msg, node});
    }
};

}
```

### 12.3 Pass Interface

```cpp
namespace pto::wsp::ir {

class Pass {
public:
    virtual ~Pass() = default;
    virtual std::string_view name() const = 0;
    virtual PassResult run(Module& m, PassContext& ctx) = 0;
};

}
```

### 12.4 Pass Manager

```cpp
namespace pto::wsp::ir {

class PassManager {
    std::vector<std::unique_ptr<Pass>> passes_;

public:
    void add(std::unique_ptr<Pass> p) {
        passes_.push_back(std::move(p));
    }

    template<typename T, typename... Args>
    void add(Args&&... args) {
        passes_.push_back(std::make_unique<T>(std::forward<Args>(args)...));
    }

    PassResult run(Module& m) {
        IRFactory factory;
        std::vector<Diagnostic> diags;
        PassContext ctx{factory, diags};
        ctx.symbols = &SymbolTable::build(m);

        PassResult overall{.changed = false, .ok = true};

        for (const auto& p : passes_) {
            PassResult r = p->run(m, ctx);
            overall.changed |= r.changed;
            if (!r.ok) {
                overall.ok = false;
                break;  // Stop on first error
            }
        }

        // Print diagnostics
        for (const auto& d : diags) {
            std::cerr << d.format() << "\n";
        }

        return overall;
    }
};

}
```

---

## 13. IR Rewriter

### 13.1 IRRewriter Base Class

The rewriter enables tree transformations by recursively rewriting children and rebuilding nodes:

```cpp
namespace pto::wsp::ir {

class IRRewriter {
protected:
    IRFactory& factory_;

public:
    explicit IRRewriter(IRFactory& f) : factory_(f) {}
    virtual ~IRRewriter() = default;

    // Main entry point: dispatches by node kind
    IRPtr<IRNode> rewrite(const IRPtr<IRNode>& n);

    // Typed rewrite helpers
    IRPtr<WorkloadNode> rewriteWorkload(const IRPtr<WorkloadNode>& n);
    IRPtr<ScheduleNode> rewriteSchedule(const IRPtr<ScheduleNode>& n);

protected:
    // Override points (default: rebuild after rewriting children)
    virtual IRPtr<IRNode> rewriteTask(const IRPtr<TaskNode>& n) { return n; }
    virtual IRPtr<IRNode> rewriteParallelFor(const IRPtr<ParallelForNode>& n);
    virtual IRPtr<IRNode> rewriteForEach(const IRPtr<ForEachNode>& n);
    virtual IRPtr<IRNode> rewriteSelect(const IRPtr<SelectNode>& n);
    virtual IRPtr<IRNode> rewriteCond(const IRPtr<CondNode>& n);
    virtual IRPtr<IRNode> rewriteCombine(const IRPtr<CombineNode>& n);
    virtual IRPtr<IRNode> rewriteSequential(const IRPtr<SequentialNode>& n);
    virtual IRPtr<IRNode> rewriteCall(const IRPtr<CallNode>& n) { return n; }

    // Extension rewrite (default: dispatch via registry, then rebuild)
    virtual IRPtr<IRNode> rewriteExt(const IRPtr<ExtOpNode>& n) {
        // Check registry for extension-specific rewriter
        if (auto rewriter = ExtOpRegistry::instance().getRewriter(n->op_name)) {
            return (*rewriter)(*this, factory_, n);
        }
        // Default: rewrite children, rebuild if changed
        return rewriteExtDefault(n);
    }

    IRPtr<IRNode> rewriteExtDefault(const IRPtr<ExtOpNode>& n) {
        std::vector<IRPtr<IRNode>> newChildren;
        bool changed = false;
        for (const auto& child : n->children) {
            auto newChild = rewrite(child);
            if (newChild != child) changed = true;
            newChildren.push_back(newChild);
        }
        if (!changed) return n;
        return factory_.create<ExtOpNode>(n->ext_class, n->op_name, n->attrs,
                                           std::move(newChildren), n->level);
    }

    // Rebuild helpers (create new node if children changed)
    template<typename T>
    IRPtr<T> rebuildIfChanged(const IRPtr<T>& original,
                               const std::vector<IRPtr<IRNode>>& newChildren);
};

}
```

### 13.2 Example: Constant Folding Pass

```cpp
namespace pto::wsp::ir {

class ConstantFoldingRewriter : public IRRewriter {
public:
    using IRRewriter::IRRewriter;

protected:
    IRPtr<IRNode> rewriteCond(const IRPtr<CondNode>& n) override {
        // If predicate is constant "true" or "false", eliminate branch
        if (n->predicate == "true") {
            return rewriteWorkload(n->then_branch);
        }
        if (n->predicate == "false") {
            return rewriteWorkload(n->else_branch);
        }
        // Otherwise, default rewrite
        return IRRewriter::rewriteCond(n);
    }
};

class ConstantFoldingPass : public Pass {
public:
    std::string_view name() const override { return "constant-folding"; }

    PassResult run(Module& m, PassContext& ctx) override {
        ConstantFoldingRewriter rewriter(ctx.factory);
        bool changed = false;

        for (auto& w : m.workloads) {
            auto newBody = rewriter.rewriteWorkload(w.body);
            if (newBody != w.body) {
                w.body = newBody;
                changed = true;
            }
        }

        return {.changed = changed, .ok = true};
    }
};

}
```

---

## 14. Summary

### 14.1 IR Node Hierarchy

```
IRNode (with WorkloadLevel)
├── AxisNode
│   ├── DenseAxisNode
│   ├── DenseDynAxisNode
│   ├── RaggedAxisNode
│   └── SparseAxisNode
├── WorkloadNode
│   ├── TaskNode
│   ├── ParallelForNode
│   ├── ForEachNode
│   ├── SelectNode
│   ├── CondNode
│   ├── CombineNode
│   ├── SequentialNode
│   └── CallNode          [NEW: cross-level hierarchy]
├── CSPNode
│   ├── ChannelNode
│   ├── ProcessNode
│   ├── SendNode
│   ├── ConsumeNode
│   └── PipelineNode
├── ScheduleNode
│   ├── DispatchNode
│   ├── StreamNode
│   ├── TimingNode
│   ├── SpatialMapNode
│   └── LayoutNode
└── ExtOpNode              [NEW: extension mechanism]
```

### 14.2 New in v9

| Component | Description |
|-----------|-------------|
| `WorkloadLevel` | Hierarchy level (CPU, NPU, Any) on every node |
| `CallNode` | Cross-level call for CPU→NPU hierarchy |
| `ExtOpNode` | Extension mechanism for new primitives |
| `ExtOpRegistry` | Self-registration for extensions (visitor, rewriter, factory) |
| `SpatialMapNode` | Grid-based spatial mapping for AIE |
| `LayoutNode` | Data distribution (Shard/Replicate) |
| `.pto` format | Text-based assembly for serialization |
| `Module` | Top-level container for workloads, schedules, pipelines |

### 14.3 Infrastructure Components

| Component | Description |
|-----------|-------------|
| `IRVisitor` | Base visitor with typed hooks + extension dispatch |
| `walk()` | Recursive tree traversal with enter/leave hooks |
| `ExtOpRegistry` | Global registry for extension handlers |
| `SymbolTable` | Resolves CallNode references across definitions |
| `Pass` | Transformation pass interface |
| `PassManager` | Runs pass pipeline with diagnostics |
| `PassContext` | Factory + diagnostics for pass execution |
| `IRRewriter` | Tree rewriter with extension dispatch |

### 14.4 Header Organization (Recommended)

```text
include/pto/rt/ir/
├── core.hpp      // NodeId, WorkloadLevel, IRNode, IRFactory
├── nodes.hpp     // All core node types
├── ext.hpp       // ExtOpNode, ExtOpRegistry, REGISTER_EXT_OP macro
├── module.hpp    // Module, WorkloadDef, ScheduleDef, SymbolTable
├── visitor.hpp   // IRVisitor, walk(), walkModule()
├── pass.hpp      // Pass, PassManager, PassContext, Diagnostic
├── rewriter.hpp  // IRRewriter
├── ir.hpp        // Umbrella include
└── ext/          // Extension headers (npu/, aie/, schedule/)
```

### 14.5 Extension Design Principles

| Aspect | Decision |
|--------|----------|
| Core nodes | Fixed `NodeKind` enum, virtual dispatch |
| Extensions | `NodeKind::Ext` + name-based dispatch via registry |
| Registration | Static initializers (`REGISTER_EXT_OP` macro) |
| CMake | Separate libraries, auto-register when linked |

---

*Version: 9.1*
*Last Updated: 2026-01-25*
