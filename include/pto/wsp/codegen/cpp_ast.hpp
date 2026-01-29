// Copyright 2026 PTO-RT Authors
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace pto::wsp::codegen::cpp {

struct ExprNode;
struct StmtNode;

// Expr/Stmt are small value-semantic handles (shared ownership) so AST nodes can
// reference each other without recursive value types.
struct Expr {
    std::shared_ptr<const ExprNode> node;

    Expr() = default;
    explicit Expr(std::shared_ptr<const ExprNode> n) : node(std::move(n)) {}
};

struct Stmt {
    std::shared_ptr<const StmtNode> node;

    Stmt() = default;
    explicit Stmt(std::shared_ptr<const StmtNode> n) : node(std::move(n)) {}
};

// =============================================================================
// Expressions
// =============================================================================

struct RawExpr {
    std::string code;
};

struct IdentExpr {
    std::string name;
};

struct LiteralExpr {
    std::string code;  // already formatted (e.g. "0", "3.14f", "\"hi\"")
};

struct CallExpr {
    Expr callee;
    std::vector<Expr> args;
};

struct MemberExpr {
    Expr base;
    std::string member;
    bool arrow = false;                   // true -> "->", false -> "."
};

struct IndexExpr {
    Expr base;
    std::vector<Expr> indices;
};

struct UnaryExpr {
    std::string op;                       // "!", "-", "*", "&", "++", ...
    Expr expr;
    bool postfix = false;                 // true -> "x++", false -> "++x"
};

struct BinaryExpr {
    std::string op;                       // "+", "-", "*", "&&", "==", ...
    Expr lhs;
    Expr rhs;
};

struct CastExpr {
    std::string type;                     // "(T)" style cast
    Expr expr;
};

struct ExprNode {
    using Node = std::variant<
        RawExpr,
        IdentExpr,
        LiteralExpr,
        CallExpr,
        MemberExpr,
        IndexExpr,
        UnaryExpr,
        BinaryExpr,
        CastExpr>;

    Node node;

    template <class T>
    explicit ExprNode(T v) : node(std::move(v)) {}
};

template <class T>
inline Expr make_expr(T v) {
    return Expr(std::make_shared<ExprNode>(std::move(v)));
}

inline Expr raw_expr(std::string code) { return make_expr(RawExpr{std::move(code)}); }
inline Expr ident(std::string name) { return make_expr(IdentExpr{std::move(name)}); }
inline Expr lit(std::string code) { return make_expr(LiteralExpr{std::move(code)}); }

// =============================================================================
// Statements
// =============================================================================

struct RawStmt {
    std::string code;  // emitted as-is (must include trailing ';' if needed)
};

struct ExprStmt {
    Expr expr;
};

struct ReturnStmt {
    std::optional<Expr> value;
};

struct VarDeclStmt {
    std::string type;
    std::string name;
    std::optional<Expr> init;
};

struct BlockStmt {
    std::vector<Stmt> stmts;
};

struct IfStmt {
    Expr cond;
    BlockStmt then_block;
    std::optional<BlockStmt> else_block;
};

struct ForStmt {
    // Rendered as: for (<init>; <cond>; <inc>) { ... }
    // Init is restricted to var decl for now to keep emission predictable.
    VarDeclStmt init;
    Expr cond;
    Expr inc;
    BlockStmt body;
};

struct StmtNode {
    using Node = std::variant<
        RawStmt,
        ExprStmt,
        ReturnStmt,
        VarDeclStmt,
        BlockStmt,
        IfStmt,
        ForStmt>;

    Node node;

    template <class T>
    explicit StmtNode(T v) : node(std::move(v)) {}
};

template <class T>
inline Stmt make_stmt(T v) {
    return Stmt(std::make_shared<StmtNode>(std::move(v)));
}

// =============================================================================
// Decls / Translation Unit
// =============================================================================

struct Param {
    std::string type;
    std::string name;
};

struct Function {
    std::string return_type;
    std::string name;
    std::vector<Param> params;
    BlockStmt body;
    bool extern_c = false;
};

struct TranslationUnit {
    std::vector<std::string> includes;    // e.g. "<cstdint>" or "\"my.h\""
    std::vector<std::string> defines;     // raw lines: "#define X 1"
    std::vector<std::string> raw_toplevel;
    std::vector<Function> functions;
};

struct EmitOptions {
    std::string indent = "    ";
    bool trailing_newline = true;
};

std::string emit_cpp(const TranslationUnit& tu, const EmitOptions& opt = {});

}  // namespace pto::wsp::codegen::cpp
