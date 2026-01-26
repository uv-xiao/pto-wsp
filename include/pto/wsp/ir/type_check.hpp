// PTO Workload-Schedule Programming (PTO-WSP) framework v9 - IR Type Checking Pass
// Copyright (c) 2026 PTO Project
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "pto/rt/ir/core.hpp"
#include "pto/rt/ir/axis.hpp"
#include "pto/rt/ir/workload.hpp"
#include "pto/rt/ir/schedule.hpp"
#include "pto/rt/ir/csp.hpp"
#include "pto/rt/ir/module.hpp"
#include "pto/rt/ir/visitor.hpp"

#include <string>
#include <vector>
#include <unordered_set>
#include <sstream>

namespace pto::wsp::ir {

// ============================================================
// Type Error Information
// ============================================================

enum class TypeCheckErrorKind {
    InvalidStructure,      // Malformed IR structure
    UndefinedReference,    // Reference to undefined symbol
    DuplicateDefinition,   // Duplicate workload/schedule name
    LayoutIncompatible,    // Layout compatibility error
    TypeMismatch,          // Type annotation mismatch
    AxisBoundsError,       // Axis out of bounds
    ChannelTypeMismatch,   // Channel element type mismatch
    WorkloadLevelError,    // Wrong workload level (CPU vs NPU)
    ScheduleMismatch,      // Schedule doesn't match workload
};

struct TypeCheckError {
    TypeCheckErrorKind kind;
    std::string message;
    std::string location;  // Node ID or symbol name
    std::string hint;

    [[nodiscard]] std::string to_string() const {
        std::ostringstream oss;
        oss << "TypeCheckError[";
        switch (kind) {
            case TypeCheckErrorKind::InvalidStructure: oss << "InvalidStructure"; break;
            case TypeCheckErrorKind::UndefinedReference: oss << "UndefinedReference"; break;
            case TypeCheckErrorKind::DuplicateDefinition: oss << "DuplicateDefinition"; break;
            case TypeCheckErrorKind::LayoutIncompatible: oss << "LayoutIncompatible"; break;
            case TypeCheckErrorKind::TypeMismatch: oss << "TypeMismatch"; break;
            case TypeCheckErrorKind::AxisBoundsError: oss << "AxisBoundsError"; break;
            case TypeCheckErrorKind::ChannelTypeMismatch: oss << "ChannelTypeMismatch"; break;
            case TypeCheckErrorKind::WorkloadLevelError: oss << "WorkloadLevelError"; break;
            case TypeCheckErrorKind::ScheduleMismatch: oss << "ScheduleMismatch"; break;
        }
        oss << "]";
        if (!location.empty()) {
            oss << " at " << location;
        }
        oss << ": " << message;
        if (!hint.empty()) {
            oss << " (hint: " << hint << ")";
        }
        return oss.str();
    }
};

// ============================================================
// Type Check Result
// ============================================================

struct TypeCheckResult {
    bool valid = true;
    std::vector<TypeCheckError> errors;

    void add_error(TypeCheckErrorKind kind, const std::string& message,
                   const std::string& location = "", const std::string& hint = "") {
        valid = false;
        errors.push_back({kind, message, location, hint});
    }

    [[nodiscard]] std::string to_string() const {
        if (valid) return "TypeCheck: OK";
        std::ostringstream oss;
        oss << "TypeCheck: " << errors.size() << " error(s)\n";
        for (const auto& e : errors) {
            oss << "  " << e.to_string() << "\n";
        }
        return oss.str();
    }
};

// ============================================================
// Type Check Pass Declaration
// ============================================================

class TypeCheckPass : public IRVisitor {
public:
    explicit TypeCheckPass(const Module& module);

    /// Run type checking on the module
    TypeCheckResult check();

    // ========== Visitor Overrides ==========
    WalkControl enter(const CallNode& node) override;
    WalkControl enter(const TaskNode& node) override;
    WalkControl enter(const ParallelForNode& node) override;
    WalkControl enter(const ForEachNode& node) override;
    WalkControl enter(const CondNode& node) override;
    WalkControl enter(const SendNode& node) override;
    WalkControl enter(const ConsumeNode& node) override;
    WalkControl enter(const DenseAxisNode& node) override;
    WalkControl enter(const PipelineDepthNode& node) override;
    WalkControl enter(const TaskWindowNode& node) override;

private:
    void checkDuplicates();
    void checkSchedule(const ScheduleDef& sched);
    void checkPipeline(const PipelineNode& pipeline);

    const Module& module_;
    SymbolTable symbols_;
    TypeCheckResult result_;
    std::string current_workload_;
    WorkloadLevel current_level_ = WorkloadLevel::CPU;
    std::unordered_set<std::string> workload_names_;
    std::unordered_set<std::string> channel_names_;
};

// ============================================================
// Convenience Function
// ============================================================

/// Run type checking on a module
TypeCheckResult type_check(const Module& module);

}  // namespace pto::wsp::ir
