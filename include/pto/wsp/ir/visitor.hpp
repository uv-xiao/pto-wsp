// PTO Workload-Schedule Programming (PTO-WSP) framework v9 - Visitor and Traversal
// Copyright (c) 2024 PTO Project
// SPDX-License-Identifier: MIT

#pragma once

#include "core.hpp"
#include "axis.hpp"
#include "workload.hpp"
#include "schedule.hpp"
#include "csp.hpp"
#include "ext.hpp"
#include "module.hpp"

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
    virtual WalkControl enter(const SlotSetU64Node& n) { return enter(static_cast<const IRNode&>(n)); }
    virtual WalkControl enter(const SlotLoadU64Node& n) { return enter(static_cast<const IRNode&>(n)); }
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
    virtual WalkControl enter(const DispatchThresholdNode& n) { return enter(static_cast<const IRNode&>(n)); }
    virtual WalkControl enter(const PipelineDepthNode& n) { return enter(static_cast<const IRNode&>(n)); }
    virtual WalkControl enter(const TaskWindowNode& n) { return enter(static_cast<const IRNode&>(n)); }
    virtual WalkControl enter(const BatchDepsNode& n) { return enter(static_cast<const IRNode&>(n)); }

    // Extension (default: dispatch via registry)
    virtual WalkControl enter(const ExtOpNode& n) {
        return enterExt(n);
    }

protected:
    // Extensible dispatch for ExtOpNode via registry
    WalkControl enterExt(const ExtOpNode& n) {
        if (auto handler = ExtOpRegistry::instance().getEnterHandler(n.op_name)) {
            return static_cast<WalkControl>((*handler)(*this, n));
        }
        // Fallback to base IRNode handler
        return enter(static_cast<const IRNode&>(n));
    }
};

// Dispatch to typed visitor methods by NodeKind
inline WalkControl dispatchEnter(const IRPtr<IRNode>& node, IRVisitor& v) {
    switch (node->kind) {
        // Axis nodes
        case NodeKind::DenseAxis:
            return v.enter(static_cast<const DenseAxisNode&>(*node));
        case NodeKind::DenseDynAxis:
            return v.enter(static_cast<const DenseDynAxisNode&>(*node));
        case NodeKind::RaggedAxis:
            return v.enter(static_cast<const RaggedAxisNode&>(*node));
        case NodeKind::SparseAxis:
            return v.enter(static_cast<const SparseAxisNode&>(*node));

        // Workload nodes
        case NodeKind::Task:
            return v.enter(static_cast<const TaskNode&>(*node));
        case NodeKind::SlotSetU64:
            return v.enter(static_cast<const SlotSetU64Node&>(*node));
        case NodeKind::SlotLoadU64:
            return v.enter(static_cast<const SlotLoadU64Node&>(*node));
        case NodeKind::ParallelFor:
            return v.enter(static_cast<const ParallelForNode&>(*node));
        case NodeKind::ForEach:
            return v.enter(static_cast<const ForEachNode&>(*node));
        case NodeKind::Select:
            return v.enter(static_cast<const SelectNode&>(*node));
        case NodeKind::Cond:
            return v.enter(static_cast<const CondNode&>(*node));
        case NodeKind::Combine:
            return v.enter(static_cast<const CombineNode&>(*node));
        case NodeKind::Sequential:
            return v.enter(static_cast<const SequentialNode&>(*node));
        case NodeKind::Call:
            return v.enter(static_cast<const CallNode&>(*node));

        // CSP nodes
        case NodeKind::Channel:
            return v.enter(static_cast<const ChannelNode&>(*node));
        case NodeKind::Process:
            return v.enter(static_cast<const ProcessNode&>(*node));
        case NodeKind::Send:
            return v.enter(static_cast<const SendNode&>(*node));
        case NodeKind::Consume:
            return v.enter(static_cast<const ConsumeNode&>(*node));
        case NodeKind::Pipeline:
            return v.enter(static_cast<const PipelineNode&>(*node));

        // Schedule nodes
        case NodeKind::Dispatch:
            return v.enter(static_cast<const DispatchNode&>(*node));
        case NodeKind::Stream:
            return v.enter(static_cast<const StreamNode&>(*node));
        case NodeKind::Timing:
            return v.enter(static_cast<const TimingNode&>(*node));
        case NodeKind::SpatialMap:
            return v.enter(static_cast<const SpatialMapNode&>(*node));
        case NodeKind::Layout:
            return v.enter(static_cast<const LayoutNode&>(*node));
        case NodeKind::DispatchThreshold:
            return v.enter(static_cast<const DispatchThresholdNode&>(*node));
        case NodeKind::PipelineDepth:
            return v.enter(static_cast<const PipelineDepthNode&>(*node));
        case NodeKind::TaskWindow:
            return v.enter(static_cast<const TaskWindowNode&>(*node));
        case NodeKind::BatchDeps:
            return v.enter(static_cast<const BatchDepsNode&>(*node));

        // Extension
        case NodeKind::Ext:
            return v.enter(static_cast<const ExtOpNode&>(*node));
    }
    return WalkControl::Continue;
}

// Recursive tree walk
inline WalkControl walk(const IRPtr<IRNode>& root, IRVisitor& v) {
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
inline void walkModule(const Module& m, IRVisitor& v) {
    for (const auto& w : m.workloads) {
        walk(std::static_pointer_cast<const IRNode>(w.body), v);
    }
    for (const auto& s : m.schedules) {
        for (const auto& d : s.directives) {
            walk(std::static_pointer_cast<const IRNode>(d), v);
        }
    }
    for (const auto& p : m.pipelines) {
        walk(std::static_pointer_cast<const IRNode>(p), v);
    }
}

// Symbol table for resolving cross-workload references
class SymbolTable {
    std::unordered_map<std::string, const WorkloadDef*> workloads_;
    std::unordered_map<std::string, const ScheduleDef*> schedules_;
    std::unordered_map<std::string, IRPtr<PipelineNode>> pipelines_;

public:
    static SymbolTable build(const Module& m) {
        SymbolTable table;
        for (const auto& w : m.workloads) {
            table.workloads_[w.name] = &w;
        }
        for (const auto& s : m.schedules) {
            table.schedules_[s.name] = &s;
        }
        for (const auto& p : m.pipelines) {
            // Pipeline nodes don't have a name field in current design
            // This would need to be added if pipeline lookup is needed
        }
        return table;
    }

    [[nodiscard]] const WorkloadDef* lookupWorkload(const std::string& name) const {
        auto it = workloads_.find(name);
        return it != workloads_.end() ? it->second : nullptr;
    }

    [[nodiscard]] const ScheduleDef* lookupSchedule(const std::string& name) const {
        auto it = schedules_.find(name);
        return it != schedules_.end() ? it->second : nullptr;
    }
};

}  // namespace pto::wsp::ir
