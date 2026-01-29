// PTO Workload-Schedule Programming (PTO-WSP) framework v9 - CSP IR Nodes
// Copyright (c) 2024 PTO Project
// SPDX-License-Identifier: MIT

#pragma once

#include "core.hpp"
#include "workload.hpp"

namespace pto::wsp::ir {

// Channel Node - CSP channel declaration
struct ChannelNode : IRNode {
    const std::string name;
    const ChannelType type;

    ChannelNode(NodeId id, std::string name, ChannelType type)
        : IRNode(id, NodeKind::Channel, WorkloadLevel::CPU),
          name(std::move(name)),
          type(std::move(type)) {}

    [[nodiscard]] bool is_event() const { return type.capacity == 0; }

    void print(std::ostream& os, int indent) const override {
        os << std::string(indent, ' ') << "channel " << name << " : "
           << type.to_string();
    }
};

// Process Node - CSP process with channels
struct ProcessNode : WorkloadNode {
    const std::string name;
    const std::vector<std::string> consumes;  // Channel names
    const std::vector<std::string> produces;  // Channel names
    const IRPtr<WorkloadNode> body;

    ProcessNode(NodeId id, std::string name,
                std::vector<std::string> consumes,
                std::vector<std::string> produces,
                IRPtr<WorkloadNode> body)
        : WorkloadNode(id, NodeKind::Process, WorkloadLevel::CPU),
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

    void forEachChild(const ChildFn& fn) const override {
        fn(std::static_pointer_cast<const IRNode>(body));
    }
};

// Send Node - Send value to channel
struct SendNode : WorkloadNode {
    const std::string channel_name;
    const IRPtr<WorkloadNode> value;

    SendNode(NodeId id, std::string channel, IRPtr<WorkloadNode> value)
        : WorkloadNode(id, NodeKind::Send, WorkloadLevel::CPU),
          channel_name(std::move(channel)),
          value(std::move(value)) {}

    void print(std::ostream& os, int indent) const override {
        os << std::string(indent, ' ') << "send " << channel_name << ", ";
        value->print(os, 0);
    }

    void forEachChild(const ChildFn& fn) const override {
        fn(std::static_pointer_cast<const IRNode>(value));
    }
};

// Consume Node - Receive from channel and process
struct ConsumeNode : WorkloadNode {
    const std::string channel_name;
    const std::string value_var;
    const IRPtr<WorkloadNode> body;

    ConsumeNode(NodeId id, std::string channel, std::string value_var,
                IRPtr<WorkloadNode> body)
        : WorkloadNode(id, NodeKind::Consume, WorkloadLevel::CPU),
          channel_name(std::move(channel)),
          value_var(std::move(value_var)),
          body(std::move(body)) {}

    void print(std::ostream& os, int indent) const override {
        os << std::string(indent, ' ') << "consume " << value_var << " from "
           << channel_name << " {\n";
        body->print(os, indent + 2);
        os << "\n" << std::string(indent, ' ') << "}";
    }

    void forEachChild(const ChildFn& fn) const override {
        fn(std::static_pointer_cast<const IRNode>(body));
    }
};

// Pipeline Node - CSP pipeline composition
struct PipelineNode : WorkloadNode {
    const std::vector<IRPtr<ChannelNode>> channels;
    const std::vector<IRPtr<ProcessNode>> processes;

    PipelineNode(NodeId id,
                 std::vector<IRPtr<ChannelNode>> channels,
                 std::vector<IRPtr<ProcessNode>> processes)
        : WorkloadNode(id, NodeKind::Pipeline, WorkloadLevel::CPU),
          channels(std::move(channels)),
          processes(std::move(processes)) {}

    void print(std::ostream& os, int indent) const override {
        os << std::string(indent, ' ') << "pipeline {\n";
        for (const auto& ch : channels) {
            ch->print(os, indent + 2);
            os << "\n";
        }
        os << "\n";
        for (const auto& proc : processes) {
            proc->print(os, indent + 2);
            os << "\n";
        }
        os << std::string(indent, ' ') << "}";
    }

    void forEachChild(const ChildFn& fn) const override {
        for (const auto& ch : channels) {
            fn(std::static_pointer_cast<const IRNode>(ch));
        }
        for (const auto& proc : processes) {
            fn(std::static_pointer_cast<const IRNode>(proc));
        }
    }
};

}  // namespace pto::wsp::ir
