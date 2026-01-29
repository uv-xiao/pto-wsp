// PTO Workload-Schedule Programming (PTO-WSP) framework v9 - Axis IR Nodes
// Copyright (c) 2024 PTO Project
// SPDX-License-Identifier: MIT

#pragma once

#include "core.hpp"

namespace pto::wsp::ir {

// Dense Axis (Static size)
struct DenseAxisNode : AxisNode {
    const int64_t size;

    DenseAxisNode(NodeId id, int64_t size)
        : AxisNode(id, NodeKind::DenseAxis, WorkloadLevel::Any), size(size) {}

    void print(std::ostream& os, int indent) const override {
        os << std::string(indent, ' ') << "dense[" << size << "]";
    }
};

// DenseDyn Axis (Dynamic size)
struct DenseDynAxisNode : AxisNode {
    const std::string size_var;

    DenseDynAxisNode(NodeId id, std::string size_var)
        : AxisNode(id, NodeKind::DenseDynAxis, WorkloadLevel::Any),
          size_var(std::move(size_var)) {}

    void print(std::ostream& os, int indent) const override {
        os << std::string(indent, ' ') << "dense_dyn(" << size_var << ")";
    }
};

// Ragged Axis
struct RaggedAxisNode : AxisNode {
    const std::string outer_size_var;
    const std::string lengths_var;

    RaggedAxisNode(NodeId id, std::string outer, std::string lengths)
        : AxisNode(id, NodeKind::RaggedAxis, WorkloadLevel::Any),
          outer_size_var(std::move(outer)),
          lengths_var(std::move(lengths)) {}

    void print(std::ostream& os, int indent) const override {
        os << std::string(indent, ' ') << "ragged(" << outer_size_var
           << ", " << lengths_var << ")";
    }
};

// Sparse Axis
struct SparseAxisNode : AxisNode {
    const std::string outer_size_var;
    const std::string indptr_var;
    const std::string indices_var;

    SparseAxisNode(NodeId id, std::string outer, std::string indptr, std::string indices)
        : AxisNode(id, NodeKind::SparseAxis, WorkloadLevel::Any),
          outer_size_var(std::move(outer)),
          indptr_var(std::move(indptr)),
          indices_var(std::move(indices)) {}

    void print(std::ostream& os, int indent) const override {
        os << std::string(indent, ' ') << "sparse(" << outer_size_var
           << ", " << indptr_var << ", " << indices_var << ")";
    }
};

}  // namespace pto::wsp::ir
