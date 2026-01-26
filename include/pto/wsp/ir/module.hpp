// PTO Workload-Schedule Programming (PTO-WSP) framework v9 - Module IR
// Copyright (c) 2024 PTO Project
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "core.hpp"
#include "axis.hpp"
#include "workload.hpp"
#include "schedule.hpp"
#include "csp.hpp"

namespace pto::wsp::ir {

// Workload definition
struct WorkloadDef {
    std::string name;
    WorkloadLevel level;
    std::vector<std::pair<std::string, IRPtr<AxisNode>>> params;
    IRPtr<WorkloadNode> body;

    void print(std::ostream& os) const {
        os << "@workload";
        if (level == WorkloadLevel::NPU) os << "[npu]";
        os << " " << name << "(";
        for (size_t i = 0; i < params.size(); ++i) {
            if (i > 0) os << ", ";
            os << "%" << params[i].first << ": ";
            params[i].second->print(os, 0);
        }
        os << ") {\n";
        if (body) {
            body->print(os, 2);
        } else {
            os << "  // body not parsed";
        }
        os << "\n}\n";
    }
};

// Schedule definition
struct ScheduleDef {
    std::string name;
    std::string workload_name;
    WorkloadLevel level;
    std::vector<IRPtr<ScheduleNode>> directives;

    // Helper accessors
    [[nodiscard]] IRPtr<DispatchNode> dispatch() const {
        for (const auto& d : directives) {
            if (d->kind == NodeKind::Dispatch)
                return std::static_pointer_cast<const DispatchNode>(d);
        }
        return nullptr;
    }

    [[nodiscard]] IRPtr<StreamNode> stream() const {
        for (const auto& d : directives) {
            if (d->kind == NodeKind::Stream)
                return std::static_pointer_cast<const StreamNode>(d);
        }
        return nullptr;
    }

    [[nodiscard]] IRPtr<TimingNode> timing() const {
        for (const auto& d : directives) {
            if (d->kind == NodeKind::Timing)
                return std::static_pointer_cast<const TimingNode>(d);
        }
        return nullptr;
    }

    void print(std::ostream& os) const {
        os << "@schedule " << name << " for " << workload_name << " {\n";
        for (const auto& d : directives) {
            d->print(os, 2);
            os << "\n";
        }
        os << "}\n";
    }
};

// Module - top-level container
struct Module {
    std::string name;
    std::string version = "9.0";
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

    // Lookup helpers
    [[nodiscard]] const WorkloadDef* findWorkload(const std::string& n) const {
        for (const auto& w : workloads) {
            if (w.name == n) return &w;
        }
        return nullptr;
    }

    [[nodiscard]] const ScheduleDef* findSchedule(const std::string& n) const {
        for (const auto& s : schedules) {
            if (s.name == n) return &s;
        }
        return nullptr;
    }

    [[nodiscard]] const ScheduleDef* findScheduleForWorkload(const std::string& wl) const {
        for (const auto& s : schedules) {
            if (s.workload_name == wl) return &s;
        }
        return nullptr;
    }
};

}  // namespace pto::wsp::ir
