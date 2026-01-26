// PTO Workload-Schedule Programming (PTO-WSP) framework v9 - Workload IR Nodes
// Copyright (c) 2024 PTO Project
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "core.hpp"
#include "axis.hpp"

namespace pto::wsp::ir {

// Task Node - Single kernel invocation
struct TaskNode : WorkloadNode {
    const std::string kernel_name;
    const std::vector<std::string> params;
    const std::vector<std::string> resources;

    TaskNode(NodeId id, std::string kernel,
             std::vector<std::string> params,
             std::vector<std::string> resources)
        : WorkloadNode(id, NodeKind::Task, WorkloadLevel::CPU),
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

// ParallelFor Node - Parallel iteration over axis
struct ParallelForNode : WorkloadNode {
    const IRPtr<AxisNode> axis;
    const std::string index_var;
    const IRPtr<WorkloadNode> body;

    ParallelForNode(NodeId id, IRPtr<AxisNode> axis, std::string index_var,
                    IRPtr<WorkloadNode> body)
        : WorkloadNode(id, NodeKind::ParallelFor, WorkloadLevel::CPU),
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

    void forEachChild(const ChildFn& fn) const override {
        fn(std::static_pointer_cast<const IRNode>(axis));
        fn(std::static_pointer_cast<const IRNode>(body));
    }
};

// ForEach Node - Sequential iteration over axis
struct ForEachNode : WorkloadNode {
    const IRPtr<AxisNode> axis;
    const std::string index_var;
    const IRPtr<WorkloadNode> body;

    ForEachNode(NodeId id, IRPtr<AxisNode> axis, std::string index_var,
                IRPtr<WorkloadNode> body)
        : WorkloadNode(id, NodeKind::ForEach, WorkloadLevel::CPU),
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

    void forEachChild(const ChildFn& fn) const override {
        fn(std::static_pointer_cast<const IRNode>(axis));
        fn(std::static_pointer_cast<const IRNode>(body));
    }
};

// Select Node - Sparse iteration
struct SelectNode : WorkloadNode {
    const IRPtr<SparseAxisNode> sparse;
    const std::string index_var;
    const IRPtr<WorkloadNode> body;

    SelectNode(NodeId id, IRPtr<SparseAxisNode> sparse, std::string index_var,
               IRPtr<WorkloadNode> body)
        : WorkloadNode(id, NodeKind::Select, WorkloadLevel::CPU),
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

    void forEachChild(const ChildFn& fn) const override {
        fn(std::static_pointer_cast<const IRNode>(sparse));
        fn(std::static_pointer_cast<const IRNode>(body));
    }
};

// Cond Node - Conditional workload
struct CondNode : WorkloadNode {
    const std::string predicate;
    const IRPtr<WorkloadNode> then_branch;
    const IRPtr<WorkloadNode> else_branch;

    CondNode(NodeId id, std::string predicate,
             IRPtr<WorkloadNode> then_branch,
             IRPtr<WorkloadNode> else_branch)
        : WorkloadNode(id, NodeKind::Cond, WorkloadLevel::CPU),
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

    void forEachChild(const ChildFn& fn) const override {
        fn(std::static_pointer_cast<const IRNode>(then_branch));
        fn(std::static_pointer_cast<const IRNode>(else_branch));
    }
};

// Combine Node - Parallel composition of workloads
struct CombineNode : WorkloadNode {
    const std::vector<IRPtr<WorkloadNode>> workloads;

    CombineNode(NodeId id, std::vector<IRPtr<WorkloadNode>> workloads)
        : WorkloadNode(id, NodeKind::Combine, WorkloadLevel::CPU),
          workloads(std::move(workloads)) {}

    void print(std::ostream& os, int indent) const override {
        os << std::string(indent, ' ') << "combine {\n";
        for (size_t i = 0; i < workloads.size(); ++i) {
            workloads[i]->print(os, indent + 2);
            if (i + 1 < workloads.size()) os << ",";
            os << "\n";
        }
        os << std::string(indent, ' ') << "}";
    }

    void forEachChild(const ChildFn& fn) const override {
        for (const auto& w : workloads) {
            fn(std::static_pointer_cast<const IRNode>(w));
        }
    }
};

// Sequential Node - Sequential composition of workloads
struct SequentialNode : WorkloadNode {
    const std::vector<IRPtr<WorkloadNode>> workloads;

    SequentialNode(NodeId id, std::vector<IRPtr<WorkloadNode>> workloads)
        : WorkloadNode(id, NodeKind::Sequential, WorkloadLevel::CPU),
          workloads(std::move(workloads)) {}

    void print(std::ostream& os, int indent) const override {
        os << std::string(indent, ' ') << "sequential {\n";
        for (size_t i = 0; i < workloads.size(); ++i) {
            workloads[i]->print(os, indent + 2);
            os << ";\n";
        }
        os << std::string(indent, ' ') << "}";
    }

    void forEachChild(const ChildFn& fn) const override {
        for (const auto& w : workloads) {
            fn(std::static_pointer_cast<const IRNode>(w));
        }
    }
};

// Call Node - Cross-level call to another workload
struct CallNode : WorkloadNode {
    const std::string target_name;  // Symbol reference to target workload
    const std::vector<std::string> args;

    CallNode(NodeId id, std::string target, std::vector<std::string> args,
             WorkloadLevel level = WorkloadLevel::CPU)
        : WorkloadNode(id, NodeKind::Call, level),
          target_name(std::move(target)),
          args(std::move(args)) {}

    void print(std::ostream& os, int indent) const override {
        os << std::string(indent, ' ') << "call @" << target_name << "(";
        for (size_t i = 0; i < args.size(); ++i) {
            if (i > 0) os << ", ";
            os << args[i];
        }
        os << ")";
    }
};

}  // namespace pto::wsp::ir
