// PTO Workload-Schedule Programming (PTO-WSP) framework v9 - Template-Based Code Generation
// Copyright (c) 2026 PTO Project
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "pto/rt/ir/ir.hpp"
#include "pto/rt/ir/npu.hpp"

#include <string>
#include <sstream>
#include <unordered_map>
#include <functional>
#include <memory>
#include <variant>

namespace pto::wsp::backend::codegen {

// ============================================================
// Code Generation Context
// ============================================================

/// Context for code generation, tracks state during traversal
struct CodeGenContext {
    std::ostringstream output;
    int indent_level = 0;
    std::string indent_str = "    ";

    // Symbol tables
    std::unordered_map<std::string, std::string> tile_names;      // IR name → generated name
    std::unordered_map<std::string, std::string> memref_names;    // IR name → generated name
    std::unordered_map<std::string, std::string> scalar_names;    // IR name → generated name

    // Counters for unique names
    int temp_counter = 0;
    int label_counter = 0;

    // Target configuration
    std::string target_triple;
    bool emit_comments = true;
    bool emit_debug_info = false;

    std::string gen_temp() { return "t" + std::to_string(temp_counter++); }
    std::string gen_label() { return "L" + std::to_string(label_counter++); }

    void emit(const std::string& code) {
        for (int i = 0; i < indent_level; ++i) output << indent_str;
        output << code << "\n";
    }

    void emit_raw(const std::string& code) { output << code; }

    void emit_comment(const std::string& comment) {
        if (emit_comments) {
            emit("// " + comment);
        }
    }

    void push_indent() { indent_level++; }
    void pop_indent() { if (indent_level > 0) indent_level--; }

    std::string get_output() const { return output.str(); }
    void clear() { output.str(""); output.clear(); }
};

// ============================================================
// Code Template System
// ============================================================

/// Template variable types
using TemplateValue = std::variant<
    std::string,
    int64_t,
    double,
    std::vector<std::string>
>;

/// Simple template engine
/// TMPL-1 NOTE: Values containing "${" are not escaped. If needed, use
/// a different placeholder pattern or pre-escape values. Consider using
/// std::format or AST-based emission for complex codegen (TMPL-2).
class Template {
public:
    explicit Template(std::string pattern) : pattern_(std::move(pattern)) {}

    /// Set a variable value
    Template& set(const std::string& key, const TemplateValue& value) {
        vars_[key] = value;
        return *this;
    }

    /// Render the template
    std::string render() const {
        std::string result = pattern_;
        for (const auto& [key, value] : vars_) {
            std::string placeholder = "${" + key + "}";
            std::string replacement = value_to_string(value);
            size_t pos;
            while ((pos = result.find(placeholder)) != std::string::npos) {
                result.replace(pos, placeholder.length(), replacement);
            }
        }
        return result;
    }

private:
    static std::string value_to_string(const TemplateValue& v) {
        return std::visit([](auto&& arg) -> std::string {
            using T = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<T, std::string>) {
                return arg;
            } else if constexpr (std::is_same_v<T, int64_t>) {
                return std::to_string(arg);
            } else if constexpr (std::is_same_v<T, double>) {
                return std::to_string(arg);
            } else if constexpr (std::is_same_v<T, std::vector<std::string>>) {
                std::string result;
                for (size_t i = 0; i < arg.size(); ++i) {
                    if (i > 0) result += ", ";
                    result += arg[i];
                }
                return result;
            }
            return "";
        }, v);
    }

    std::string pattern_;
    std::unordered_map<std::string, TemplateValue> vars_;
};

// ============================================================
// Code Emitter Interface
// ============================================================

/// Base class for target-specific code emitters
class CodeEmitter {
public:
    virtual ~CodeEmitter() = default;

    /// Get emitter name
    [[nodiscard]] virtual std::string name() const = 0;

    /// Get file extension for generated code
    [[nodiscard]] virtual std::string file_extension() const = 0;

    /// Emit header (includes, pragmas, etc.)
    virtual void emit_header(CodeGenContext& ctx) = 0;

    /// Emit footer (cleanup, main, etc.)
    virtual void emit_footer(CodeGenContext& ctx) = 0;

    /// Emit NPU function
    virtual void emit_npu_function(CodeGenContext& ctx, const ir::NPUFunction& func) = 0;

    /// Emit load operation
    virtual void emit_load(CodeGenContext& ctx, const ir::LoadOp& op) = 0;

    /// Emit store operation
    virtual void emit_store(CodeGenContext& ctx, const ir::StoreOp& op) = 0;

    /// Emit binary operation
    virtual void emit_binary(CodeGenContext& ctx, const ir::BinaryOp& op) = 0;

    /// Emit unary operation
    virtual void emit_unary(CodeGenContext& ctx, const ir::UnaryOp& op) = 0;

    /// Emit reduction operation
    virtual void emit_reduce(CodeGenContext& ctx, const ir::ReduceOp& op) = 0;

    /// Emit broadcast operation
    virtual void emit_broadcast(CodeGenContext& ctx, const ir::BroadcastOp& op) = 0;

    /// Emit matmul operation
    virtual void emit_matmul(CodeGenContext& ctx, const ir::MatmulOp& op) = 0;

    /// Emit for loop begin
    virtual void emit_for_begin(CodeGenContext& ctx, const ir::ForLoopBeginOp& op) = 0;

    /// Emit for loop end
    virtual void emit_for_end(CodeGenContext& ctx) = 0;

    /// Emit wait for DMA
    virtual void emit_wait(CodeGenContext& ctx, const ir::WaitOp& op) = 0;
};

// ============================================================
// Code Generator
// ============================================================

/// Main code generator - uses emitter for target-specific code
class CodeGenerator {
public:
    explicit CodeGenerator(std::unique_ptr<CodeEmitter> emitter)
        : emitter_(std::move(emitter)) {}

    /// Generate code for an NPU module
    std::string generate(const ir::NPUModule& module) {
        ctx_.clear();

        emitter_->emit_header(ctx_);

        for (const auto& func : module.functions) {
            generate_function(*func);
        }

        emitter_->emit_footer(ctx_);

        return ctx_.get_output();
    }

    /// Generate code for a single NPU function
    std::string generate_function(const ir::NPUFunction& func) {
        emitter_->emit_npu_function(ctx_, func);
        return ctx_.get_output();
    }

    /// Get the emitter
    CodeEmitter& emitter() { return *emitter_; }

    /// Get context
    CodeGenContext& context() { return ctx_; }

private:
    void emit_op(const ir::NPUOp& op) {
        switch (op.kind) {
            case ir::NPUOpKind::Load:
                emitter_->emit_load(ctx_, static_cast<const ir::LoadOp&>(op));
                break;
            case ir::NPUOpKind::Store:
                emitter_->emit_store(ctx_, static_cast<const ir::StoreOp&>(op));
                break;
            case ir::NPUOpKind::Add:
            case ir::NPUOpKind::Mul:
            case ir::NPUOpKind::Sub:
            case ir::NPUOpKind::Div:
                emitter_->emit_binary(ctx_, static_cast<const ir::BinaryOp&>(op));
                break;
            case ir::NPUOpKind::Exp:
            case ir::NPUOpKind::Rsqrt:
            case ir::NPUOpKind::Neg:
            case ir::NPUOpKind::Abs:
                emitter_->emit_unary(ctx_, static_cast<const ir::UnaryOp&>(op));
                break;
            case ir::NPUOpKind::RowSum:
            case ir::NPUOpKind::RowMax:
            case ir::NPUOpKind::RowMin:
            case ir::NPUOpKind::RowMean:
                emitter_->emit_reduce(ctx_, static_cast<const ir::ReduceOp&>(op));
                break;
            case ir::NPUOpKind::RowExpandMul:
            case ir::NPUOpKind::RowExpandAdd:
            case ir::NPUOpKind::RowExpandSub:
                emitter_->emit_broadcast(ctx_, static_cast<const ir::BroadcastOp&>(op));
                break;
            case ir::NPUOpKind::Matmul:
                emitter_->emit_matmul(ctx_, static_cast<const ir::MatmulOp&>(op));
                break;
            case ir::NPUOpKind::ForLoopBegin:
                emitter_->emit_for_begin(ctx_, static_cast<const ir::ForLoopBeginOp&>(op));
                ctx_.push_indent();
                break;
            case ir::NPUOpKind::ForLoopEnd:
                ctx_.pop_indent();
                emitter_->emit_for_end(ctx_);
                break;
            case ir::NPUOpKind::Wait:
                emitter_->emit_wait(ctx_, static_cast<const ir::WaitOp&>(op));
                break;
            default:
                ctx_.emit_comment("Unhandled op kind");
                break;
        }
    }

    std::unique_ptr<CodeEmitter> emitter_;
    CodeGenContext ctx_;
};

// ============================================================
// Emitter Registry
// ============================================================

/// Factory function type for creating emitters
using EmitterFactory = std::function<std::unique_ptr<CodeEmitter>()>;

/// Global registry for code emitters
class EmitterRegistry {
public:
    static EmitterRegistry& instance() {
        static EmitterRegistry registry;
        return registry;
    }

    void register_emitter(const std::string& name, EmitterFactory factory) {
        factories_[name] = std::move(factory);
    }

    std::unique_ptr<CodeEmitter> create_emitter(const std::string& name) const {
        auto it = factories_.find(name);
        if (it != factories_.end()) {
            return it->second();
        }
        return nullptr;
    }

    std::vector<std::string> available_emitters() const {
        std::vector<std::string> names;
        names.reserve(factories_.size());
        for (const auto& [name, _] : factories_) {
            names.push_back(name);
        }
        return names;
    }

private:
    EmitterRegistry() = default;
    std::unordered_map<std::string, EmitterFactory> factories_;
};

/// Helper macro for registering emitters
#define REGISTER_EMITTER(EmitterClass, name) \
    namespace { \
        static bool _registered_emitter_##EmitterClass = []() { \
            ::pto::wsp::backend::codegen::EmitterRegistry::instance().register_emitter( \
                name, []() { return std::make_unique<EmitterClass>(); }); \
            return true; \
        }(); \
    }

}  // namespace pto::wsp::backend::codegen
