// PTO Workload-Schedule Programming (PTO-WSP) framework v9 - IR Main Header
// Copyright (c) 2024 PTO Project
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Core infrastructure
#include "core.hpp"

// Node types
#include "axis.hpp"
#include "workload.hpp"
#include "schedule.hpp"
#include "csp.hpp"

// Extension mechanism
#include "ext.hpp"

// NPU function IR
#include "npu.hpp"

// Module container
#include "module.hpp"

// Visitor and traversal
#include "visitor.hpp"

// Type checking pass
#include "type_check.hpp"

// Serialization
#include "printer.hpp"
#include "parser.hpp"

namespace pto::wsp::ir {

// Utility string conversions (inline implementations)
inline const char* nodeKindToString(NodeKind kind) {
    switch (kind) {
        case NodeKind::DenseAxis: return "DenseAxis";
        case NodeKind::DenseDynAxis: return "DenseDynAxis";
        case NodeKind::RaggedAxis: return "RaggedAxis";
        case NodeKind::SparseAxis: return "SparseAxis";
        case NodeKind::Task: return "Task";
        case NodeKind::ParallelFor: return "ParallelFor";
        case NodeKind::ForEach: return "ForEach";
        case NodeKind::Select: return "Select";
        case NodeKind::Cond: return "Cond";
        case NodeKind::Combine: return "Combine";
        case NodeKind::Sequential: return "Sequential";
        case NodeKind::Call: return "Call";
        case NodeKind::Channel: return "Channel";
        case NodeKind::Process: return "Process";
        case NodeKind::Send: return "Send";
        case NodeKind::Consume: return "Consume";
        case NodeKind::Pipeline: return "Pipeline";
        case NodeKind::Dispatch: return "Dispatch";
        case NodeKind::Stream: return "Stream";
        case NodeKind::Timing: return "Timing";
        case NodeKind::SpatialMap: return "SpatialMap";
        case NodeKind::Layout: return "Layout";
        case NodeKind::DispatchThreshold: return "DispatchThreshold";
        case NodeKind::PipelineDepth: return "PipelineDepth";
        case NodeKind::TaskWindow: return "TaskWindow";
        case NodeKind::BatchDeps: return "BatchDeps";
        case NodeKind::Ext: return "Ext";
    }
    return "Unknown";
}

inline const char* dtypeToString(DType dtype) {
    switch (dtype) {
        case DType::F16: return "f16";
        case DType::BF16: return "bf16";
        case DType::F32: return "f32";
        case DType::F64: return "f64";
        case DType::I8: return "i8";
        case DType::I16: return "i16";
        case DType::I32: return "i32";
        case DType::I64: return "i64";
        case DType::U8: return "u8";
        case DType::U16: return "u16";
        case DType::U32: return "u32";
        case DType::U64: return "u64";
        case DType::Bool: return "bool";
    }
    return "unknown";
}

inline const char* locationToString(Location loc) {
    switch (loc) {
        case Location::Global: return "global";
        case Location::L2: return "l2";
        case Location::UB: return "ub";
        case Location::L1: return "l1";
    }
    return "unknown";
}

inline const char* levelToString(WorkloadLevel level) {
    switch (level) {
        case WorkloadLevel::CPU: return "cpu";
        case WorkloadLevel::NPU: return "npu";
        case WorkloadLevel::Any: return "any";
    }
    return "unknown";
}

// TensorType string conversion
inline std::string TensorType::to_string() const {
    std::string s = dtypeToString(dtype);
    s += "[";
    for (size_t i = 0; i < shape.dims.size(); ++i) {
        if (i > 0) s += ", ";
        if (shape.dims[i] < 0) {
            s += "?";
        } else {
            s += std::to_string(shape.dims[i]);
        }
    }
    s += "]";
    s += "@";
    s += locationToString(location);
    return s;
}

// ChannelType string conversion
inline std::string ChannelType::to_string() const {
    std::string s = "Channel[";
    s += element_type.to_string();
    s += ", ";
    s += std::to_string(capacity);
    s += "]";
    return s;
}

}  // namespace pto::wsp::ir
