// Copyright 2026 PTO-WSP Authors
// SPDX-License-Identifier: MIT

#pragma once

#include "pto/wsp/codegen/cpp_ast.hpp"

#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace pto::wsp::codegen::cpp {

// A small fluent builder for common C++ code patterns.
//
// This is intentionally minimal: it aims to eliminate ad-hoc string concatenation
// in codegen backends while keeping escape hatches (RawExpr/RawStmt) for complex
// constructs during bring-up.

inline Expr call(Expr callee, std::vector<Expr> args = {}) {
    return make_expr(CallExpr{std::move(callee), std::move(args)});
}

inline Expr call(std::string callee, std::vector<Expr> args = {}) {
    return call(ident(std::move(callee)), std::move(args));
}

inline Expr member(Expr base, std::string member_name, bool arrow = false) {
    return make_expr(MemberExpr{std::move(base), std::move(member_name), arrow});
}

inline Expr index(Expr base, std::vector<Expr> indices) {
    return make_expr(IndexExpr{std::move(base), std::move(indices)});
}

inline Expr unary(std::string op, Expr e, bool postfix = false) {
    return make_expr(UnaryExpr{std::move(op), std::move(e), postfix});
}

inline Expr binary(Expr lhs, std::string op, Expr rhs) {
    return make_expr(BinaryExpr{std::move(op), std::move(lhs), std::move(rhs)});
}

inline Expr cast(std::string type, Expr e) {
    return make_expr(CastExpr{std::move(type), std::move(e)});
}

inline Stmt expr_stmt(Expr e) { return make_stmt(ExprStmt{std::move(e)}); }

inline Stmt raw_stmt(std::string code) { return make_stmt(RawStmt{std::move(code)}); }

inline Stmt ret(std::optional<Expr> value = std::nullopt) { return make_stmt(ReturnStmt{std::move(value)}); }

inline Stmt vardecl(std::string type, std::string name, std::optional<Expr> init = std::nullopt) {
    return make_stmt(VarDeclStmt{std::move(type), std::move(name), std::move(init)});
}

inline BlockStmt block(std::vector<Stmt> stmts = {}) { return BlockStmt{std::move(stmts)}; }

inline Stmt if_stmt(Expr cond, BlockStmt then_block, std::optional<BlockStmt> else_block = std::nullopt) {
    return make_stmt(IfStmt{std::move(cond), std::move(then_block), std::move(else_block)});
}

inline Stmt for_stmt(VarDeclStmt init, Expr cond, Expr inc, BlockStmt body) {
    return make_stmt(ForStmt{std::move(init), std::move(cond), std::move(inc), std::move(body)});
}

// Helper to build a simple function with a body constructed by pushing statements.
class FunctionBuilder {
public:
    FunctionBuilder(std::string return_type, std::string name)
        : fn_{std::move(return_type), std::move(name), {}, BlockStmt{}, false} {}

    FunctionBuilder& extern_c(bool v = true) {
        fn_.extern_c = v;
        return *this;
    }

    FunctionBuilder& param(std::string type, std::string name) {
        fn_.params.push_back(Param{std::move(type), std::move(name)});
        return *this;
    }

    FunctionBuilder& stmt(Stmt s) {
        fn_.body.stmts.push_back(std::move(s));
        return *this;
    }

    Function build() && { return std::move(fn_); }

private:
    Function fn_;
};

}  // namespace pto::wsp::codegen::cpp
