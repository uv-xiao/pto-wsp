// Copyright 2026 PTO-WSP Authors
// SPDX-License-Identifier: MIT

#include "pto/wsp/codegen/cpp_ast.hpp"

#include <sstream>
#include <stdexcept>

namespace pto::wsp::codegen::cpp {
namespace {

class Emitter {
public:
    explicit Emitter(EmitOptions opt) : opt_(std::move(opt)) {}

    std::string run(const TranslationUnit& tu) {
        for (const auto& inc : tu.includes) {
            line("#include " + inc);
        }
        if (!tu.includes.empty()) {
            line("");
        }

        for (const auto& def : tu.defines) {
            line(def);
        }
        if (!tu.defines.empty()) {
            line("");
        }

        for (const auto& raw : tu.raw_toplevel) {
            line(raw);
        }
        if (!tu.raw_toplevel.empty()) {
            line("");
        }

        for (const auto& fn : tu.functions) {
            emit_function(fn);
            line("");
        }

        auto out = oss_.str();
        if (!opt_.trailing_newline && !out.empty() && out.back() == '\n') {
            out.pop_back();
        }
        return out;
    }

private:
    void indent() {
        for (int i = 0; i < indent_; ++i) oss_ << opt_.indent;
    }

    void line(const std::string& s) {
        indent();
        oss_ << s << "\n";
    }

    void open_block() {
        line("{");
        indent_++;
    }

    void close_block() {
        if (indent_ == 0) throw std::runtime_error("close_block: underflow");
        indent_--;
        line("}");
    }

    static std::string join_params(const std::vector<Param>& params) {
        std::ostringstream oss;
        for (size_t i = 0; i < params.size(); ++i) {
            if (i) oss << ", ";
            oss << params[i].type << " " << params[i].name;
        }
        return oss.str();
    }

    void emit_function(const Function& fn) {
        if (fn.extern_c) {
            line("extern \"C\" " + fn.return_type + " " + fn.name + "(" + join_params(fn.params) + ")");
        } else {
            line(fn.return_type + " " + fn.name + "(" + join_params(fn.params) + ")");
        }
        emit_block(fn.body);
    }

    void emit_block(const BlockStmt& b) {
        open_block();
        for (const auto& s : b.stmts) emit_stmt(s);
        close_block();
    }

    void emit_stmt(const Stmt& s) {
        if (!s.node) throw std::runtime_error("emit_stmt: null Stmt");
        std::visit([this](auto&& node) { emit_stmt_node(node); }, s.node->node);
    }

    void emit_stmt_node(const RawStmt& s) {
        line(s.code);
    }

    void emit_stmt_node(const ExprStmt& s) {
        line(emit_expr(s.expr) + ";");
    }

    void emit_stmt_node(const ReturnStmt& s) {
        if (s.value.has_value()) {
            line("return " + emit_expr(*s.value) + ";");
        } else {
            line("return;");
        }
    }

    void emit_stmt_node(const VarDeclStmt& s) {
        if (s.init.has_value()) {
            line(s.type + " " + s.name + " = " + emit_expr(*s.init) + ";");
        } else {
            line(s.type + " " + s.name + ";");
        }
    }

    void emit_stmt_node(const BlockStmt& s) {
        emit_block(s);
    }

    void emit_stmt_node(const IfStmt& s) {
        line("if (" + emit_expr(s.cond) + ")");
        emit_block(s.then_block);
        if (s.else_block.has_value()) {
            line("else");
            emit_block(*s.else_block);
        }
    }

    void emit_stmt_node(const ForStmt& s) {
        std::string init;
        if (s.init.init.has_value()) {
            init = s.init.type + " " + s.init.name + " = " + emit_expr(*s.init.init);
        } else {
            init = s.init.type + " " + s.init.name;
        }
        line("for (" + init + "; " + emit_expr(s.cond) + "; " + emit_expr(s.inc) + ")");
        emit_block(s.body);
    }

    std::string emit_expr(const Expr& e) {
        if (!e.node) throw std::runtime_error("emit_expr: null Expr");
        return std::visit([this](auto&& node) { return emit_expr_node(node); }, e.node->node);
    }

    std::string emit_expr_node(const RawExpr& e) { return e.code; }
    std::string emit_expr_node(const IdentExpr& e) { return e.name; }
    std::string emit_expr_node(const LiteralExpr& e) { return e.code; }

    std::string emit_expr_node(const CallExpr& e) {
        std::ostringstream oss;
        oss << emit_expr(e.callee) << "(";
        for (size_t i = 0; i < e.args.size(); ++i) {
            if (i) oss << ", ";
            oss << emit_expr(e.args[i]);
        }
        oss << ")";
        return oss.str();
    }

    std::string emit_expr_node(const MemberExpr& e) {
        return "(" + emit_expr(e.base) + ")" + (e.arrow ? "->" : ".") + e.member;
    }

    std::string emit_expr_node(const IndexExpr& e) {
        std::ostringstream oss;
        oss << "(" << emit_expr(e.base) << ")";
        for (const auto& idx : e.indices) {
            oss << "[" << emit_expr(idx) << "]";
        }
        return oss.str();
    }

    std::string emit_expr_node(const UnaryExpr& e) {
        if (e.postfix) {
            return "(" + emit_expr(e.expr) + ")" + e.op;
        }
        return e.op + "(" + emit_expr(e.expr) + ")";
    }

    std::string emit_expr_node(const BinaryExpr& e) {
        return "(" + emit_expr(e.lhs) + " " + e.op + " " + emit_expr(e.rhs) + ")";
    }

    std::string emit_expr_node(const CastExpr& e) {
        return "(" + e.type + ")(" + emit_expr(e.expr) + ")";
    }

    EmitOptions opt_;
    std::ostringstream oss_;
    int indent_ = 0;
};

}  // namespace

std::string emit_cpp(const TranslationUnit& tu, const EmitOptions& opt) {
    return Emitter(opt).run(tu);
}

}  // namespace pto::wsp::codegen::cpp
