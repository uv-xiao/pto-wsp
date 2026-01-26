// PTO Workload-Schedule Programming (PTO-WSP) framework v9 - IR Type Checking Pass Implementation
// Copyright (c) 2026 PTO Project
// SPDX-License-Identifier: Apache-2.0

#include "pto/rt/ir/type_check.hpp"

namespace pto::wsp::ir {

// ============================================================
// TypeCheckPass Implementation
// ============================================================

TypeCheckPass::TypeCheckPass(const Module& module)
    : module_(module), symbols_(SymbolTable::build(module)) {
    // Collect defined workload names
    for (const auto& w : module.workloads) {
        workload_names_.insert(w.name);
    }
    // Collect defined channel names from pipelines
    for (const auto& p : module.pipelines) {
        for (const auto& ch : p->channels) {
            channel_names_.insert(ch->name);
        }
    }
}

TypeCheckResult TypeCheckPass::check() {
    result_ = TypeCheckResult{};

    // Phase 1: Check for duplicate definitions
    checkDuplicates();

    // Phase 2: Walk all workload bodies
    for (const auto& w : module_.workloads) {
        current_workload_ = w.name;
        current_level_ = w.level;
        if (w.body) {
            walk(std::static_pointer_cast<const IRNode>(w.body), *this);
        }
    }

    // Phase 3: Check schedules reference valid workloads
    for (const auto& s : module_.schedules) {
        checkSchedule(s);
    }

    // Phase 4: Check pipelines
    for (const auto& p : module_.pipelines) {
        checkPipeline(*p);
    }

    return result_;
}

// ========== Visitor Overrides ==========

WalkControl TypeCheckPass::enter(const CallNode& node) {
    // Check that call target exists
    if (workload_names_.find(node.target_name) == workload_names_.end()) {
        result_.add_error(
            TypeCheckErrorKind::UndefinedReference,
            "Call to undefined workload: " + node.target_name,
            std::to_string(node.id),
            "Define workload '" + node.target_name + "' or check spelling"
        );
    }
    return WalkControl::Continue;
}

WalkControl TypeCheckPass::enter(const TaskNode& node) {
    // Check that kernel name is non-empty
    if (node.kernel_name.empty()) {
        result_.add_error(
            TypeCheckErrorKind::InvalidStructure,
            "Task has empty kernel name",
            std::to_string(node.id)
        );
    }
    return WalkControl::Continue;
}

WalkControl TypeCheckPass::enter(const ParallelForNode& node) {
    // Check axis exists
    if (!node.axis) {
        result_.add_error(
            TypeCheckErrorKind::InvalidStructure,
            "ParallelFor has null axis",
            std::to_string(node.id)
        );
    }
    // Check body exists
    if (!node.body) {
        result_.add_error(
            TypeCheckErrorKind::InvalidStructure,
            "ParallelFor has null body",
            std::to_string(node.id)
        );
    }
    // Check level consistency
    if (current_level_ == WorkloadLevel::NPU) {
        result_.add_error(
            TypeCheckErrorKind::WorkloadLevelError,
            "ParallelFor cannot appear in NPU-level workload",
            std::to_string(node.id),
            "Use tile operations instead"
        );
    }
    return WalkControl::Continue;
}

WalkControl TypeCheckPass::enter(const ForEachNode& node) {
    if (!node.axis) {
        result_.add_error(
            TypeCheckErrorKind::InvalidStructure,
            "ForEach has null axis",
            std::to_string(node.id)
        );
    }
    if (!node.body) {
        result_.add_error(
            TypeCheckErrorKind::InvalidStructure,
            "ForEach has null body",
            std::to_string(node.id)
        );
    }
    return WalkControl::Continue;
}

WalkControl TypeCheckPass::enter(const CondNode& node) {
    if (!node.then_branch) {
        result_.add_error(
            TypeCheckErrorKind::InvalidStructure,
            "Cond has null then_branch",
            std::to_string(node.id)
        );
    }
    // else_branch can be null (optional)
    return WalkControl::Continue;
}

WalkControl TypeCheckPass::enter(const SendNode& node) {
    // Check channel reference
    if (node.channel_name.empty()) {
        result_.add_error(
            TypeCheckErrorKind::InvalidStructure,
            "Send has empty channel name",
            std::to_string(node.id)
        );
    }
    return WalkControl::Continue;
}

WalkControl TypeCheckPass::enter(const ConsumeNode& node) {
    if (node.channel_name.empty()) {
        result_.add_error(
            TypeCheckErrorKind::InvalidStructure,
            "Consume has empty channel name",
            std::to_string(node.id)
        );
    }
    if (!node.body) {
        result_.add_error(
            TypeCheckErrorKind::InvalidStructure,
            "Consume has null body",
            std::to_string(node.id)
        );
    }
    return WalkControl::Continue;
}

WalkControl TypeCheckPass::enter(const DenseAxisNode& node) {
    // Check size is positive
    if (node.size <= 0) {
        result_.add_error(
            TypeCheckErrorKind::AxisBoundsError,
            "DenseAxis has non-positive size: " + std::to_string(node.size),
            std::to_string(node.id)
        );
    }
    return WalkControl::Continue;
}

WalkControl TypeCheckPass::enter(const PipelineDepthNode& node) {
    if (node.depth < 1) {
        result_.add_error(
            TypeCheckErrorKind::InvalidStructure,
            "PipelineDepth must be >= 1, got: " + std::to_string(node.depth),
            std::to_string(node.id)
        );
    }
    return WalkControl::Continue;
}

WalkControl TypeCheckPass::enter(const TaskWindowNode& node) {
    if (node.size <= 0) {
        result_.add_error(
            TypeCheckErrorKind::InvalidStructure,
            "TaskWindow size must be positive",
            std::to_string(node.id)
        );
    }
    return WalkControl::Continue;
}

// ========== Private Helper Methods ==========

void TypeCheckPass::checkDuplicates() {
    std::unordered_set<std::string> seen;

    // Check workload names
    for (const auto& w : module_.workloads) {
        if (seen.count(w.name)) {
            result_.add_error(
                TypeCheckErrorKind::DuplicateDefinition,
                "Duplicate workload definition: " + w.name,
                w.name
            );
        }
        seen.insert(w.name);
    }

    // Check schedule names
    seen.clear();
    for (const auto& s : module_.schedules) {
        if (seen.count(s.name)) {
            result_.add_error(
                TypeCheckErrorKind::DuplicateDefinition,
                "Duplicate schedule definition: " + s.name,
                s.name
            );
        }
        seen.insert(s.name);
    }
}

void TypeCheckPass::checkSchedule(const ScheduleDef& sched) {
    // Check that schedule references an existing workload
    if (workload_names_.find(sched.workload_name) == workload_names_.end()) {
        result_.add_error(
            TypeCheckErrorKind::UndefinedReference,
            "Schedule '" + sched.name + "' references undefined workload: " + sched.workload_name,
            sched.name,
            "Define workload '" + sched.workload_name + "' first"
        );
    }

    // Check schedule directives are well-formed
    for (const auto& d : sched.directives) {
        walk(std::static_pointer_cast<const IRNode>(d), *this);
    }
}

void TypeCheckPass::checkPipeline(const PipelineNode& pipeline) {
    // Check all channels have unique names
    std::unordered_set<std::string> ch_names;
    for (const auto& ch : pipeline.channels) {
        if (ch_names.count(ch->name)) {
            result_.add_error(
                TypeCheckErrorKind::DuplicateDefinition,
                "Duplicate channel name in pipeline: " + ch->name,
                std::to_string(pipeline.id)
            );
        }
        ch_names.insert(ch->name);
    }

    // Check processes
    for (const auto& proc : pipeline.processes) {
        // Check consumed channels exist
        for (const auto& ch_name : proc->consumes) {
            if (ch_names.find(ch_name) == ch_names.end()) {
                result_.add_error(
                    TypeCheckErrorKind::UndefinedReference,
                    "Process '" + proc->name + "' consumes undefined channel: " + ch_name,
                    proc->name
                );
            }
        }
        // Check produced channels exist
        for (const auto& ch_name : proc->produces) {
            if (ch_names.find(ch_name) == ch_names.end()) {
                result_.add_error(
                    TypeCheckErrorKind::UndefinedReference,
                    "Process '" + proc->name + "' produces undefined channel: " + ch_name,
                    proc->name
                );
            }
        }
    }
}

// ============================================================
// Convenience Function
// ============================================================

TypeCheckResult type_check(const Module& module) {
    TypeCheckPass pass(module);
    return pass.check();
}

}  // namespace pto::wsp::ir
