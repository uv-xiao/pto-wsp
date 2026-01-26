// PTO Workload-Schedule Programming (PTO-WSP) framework v9 - Extension Mechanism
// Copyright (c) 2024 PTO Project
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "core.hpp"

#include <variant>
#include <unordered_map>

namespace pto::wsp::ir {

// Forward declarations
class IRVisitor;
class IRRewriter;
enum class WalkControl;

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
    [[nodiscard]] std::optional<T> getAttr(const std::string& key) const {
        auto it = attrs.find(key);
        if (it == attrs.end()) return std::nullopt;
        if (auto* v = std::get_if<T>(&it->second)) return *v;
        return std::nullopt;
    }

    template<typename T>
    [[nodiscard]] T getAttrOr(const std::string& key, T default_val) const {
        return getAttr<T>(key).value_or(default_val);
    }

    void forEachChild(const ChildFn& fn) const override {
        for (const auto& c : children) fn(c);
    }

    void print(std::ostream& os, int indent) const override {
        os << std::string(indent, ' ') << "ext." << op_name << "(";
        bool first = true;
        for (const auto& [k, v] : attrs) {
            if (!first) os << ", ";
            first = false;
            os << k << "=";
            std::visit([&os](auto&& arg) {
                using T = std::decay_t<decltype(arg)>;
                if constexpr (std::is_same_v<T, std::string>) {
                    os << '"' << arg << '"';
                } else if constexpr (std::is_same_v<T, bool>) {
                    os << (arg ? "true" : "false");
                } else if constexpr (std::is_same_v<T, std::vector<int64_t>>) {
                    os << "[";
                    for (size_t i = 0; i < arg.size(); ++i) {
                        if (i > 0) os << ", ";
                        os << arg[i];
                    }
                    os << "]";
                } else if constexpr (std::is_same_v<T, std::vector<std::string>>) {
                    os << "[";
                    for (size_t i = 0; i < arg.size(); ++i) {
                        if (i > 0) os << ", ";
                        os << '"' << arg[i] << '"';
                    }
                    os << "]";
                } else {
                    os << arg;
                }
            }, v);
        }
        os << ")";
        if (!children.empty()) {
            os << " {\n";
            for (const auto& c : children) {
                c->print(os, indent + 2);
                os << "\n";
            }
            os << std::string(indent, ' ') << "}";
        }
    }
};

// Handler function types (forward declarations for actual implementation)
using ExtVisitorFn = std::function<int(IRVisitor&, const ExtOpNode&)>;
using ExtLeaveFn = std::function<void(IRVisitor&, const ExtOpNode&)>;
using ExtFactoryFn = std::function<IRPtr<ExtOpNode>(IRFactory&, const AttrMap&)>;
using ExtRewriteFn = std::function<IRPtr<IRNode>(IRRewriter&, IRFactory&, const IRPtr<ExtOpNode>&)>;

// Extension registry for self-registration
class ExtOpRegistry {
public:
    static ExtOpRegistry& instance() {
        static ExtOpRegistry reg;
        return reg;
    }

    // Visitor registration
    void registerVisitor(const std::string& op_name, ExtVisitorFn enter,
                         ExtLeaveFn leave = nullptr) {
        auto& entry = registry_[op_name];
        entry.enter = std::move(enter);
        entry.leave = std::move(leave);
    }

    // Factory registration (for parsing)
    void registerFactory(const std::string& op_name, ExtFactoryFn factory) {
        registry_[op_name].factory = std::move(factory);
    }

    // Rewriter registration
    void registerRewriter(const std::string& op_name, ExtRewriteFn rewrite) {
        registry_[op_name].rewrite = std::move(rewrite);
    }

    // Lookup
    [[nodiscard]] std::optional<ExtVisitorFn> getEnterHandler(const std::string& op_name) const {
        auto it = registry_.find(op_name);
        if (it == registry_.end() || !it->second.enter) return std::nullopt;
        return it->second.enter;
    }

    [[nodiscard]] std::optional<ExtLeaveFn> getLeaveHandler(const std::string& op_name) const {
        auto it = registry_.find(op_name);
        if (it == registry_.end() || !it->second.leave) return std::nullopt;
        return it->second.leave;
    }

    [[nodiscard]] std::optional<ExtRewriteFn> getRewriter(const std::string& op_name) const {
        auto it = registry_.find(op_name);
        if (it == registry_.end() || !it->second.rewrite) return std::nullopt;
        return it->second.rewrite;
    }

    [[nodiscard]] bool hasOp(const std::string& op_name) const {
        return registry_.contains(op_name);
    }

    // Create extension node from attributes
    [[nodiscard]] IRPtr<ExtOpNode> create(const std::string& op_name, IRFactory& f, const AttrMap& attrs) {
        auto it = registry_.find(op_name);
        if (it == registry_.end() || !it->second.factory) {
            // Create generic ExtOpNode
            return f.create<ExtOpNode>(ExtClass::Backend, op_name, attrs);
        }
        return it->second.factory(f, attrs);
    }

private:
    struct Entry {
        ExtVisitorFn enter;
        ExtLeaveFn leave;
        ExtFactoryFn factory;
        ExtRewriteFn rewrite;
    };
    std::unordered_map<std::string, Entry> registry_;
};

// Self-registration macros
#define REGISTER_EXT_OP(op_name, enter_fn, leave_fn, factory_fn) \
    static bool _reg_##__COUNTER__ = []() { \
        ::pto::wsp::ir::ExtOpRegistry::instance().registerVisitor(op_name, enter_fn, leave_fn); \
        ::pto::wsp::ir::ExtOpRegistry::instance().registerFactory(op_name, factory_fn); \
        return true; \
    }()

#define REGISTER_EXT_REWRITER(op_name, rewrite_fn) \
    static bool _rewrite_reg_##__COUNTER__ = []() { \
        ::pto::wsp::ir::ExtOpRegistry::instance().registerRewriter(op_name, rewrite_fn); \
        return true; \
    }()

}  // namespace pto::wsp::ir
