// PTO Workload-Schedule Programming (PTO-WSP) framework v9 - Scalar Expression IR
// Copyright (c) 2026 PTO Project
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace pto::wsp::ir {

enum class ScalarType : uint8_t {
    Bool,
    I64,
    U64,
};

inline const char* scalarTypeToString(ScalarType t) {
    switch (t) {
    case ScalarType::Bool: return "bool";
    case ScalarType::I64: return "i64";
    case ScalarType::U64: return "u64";
    }
    return "unknown";
}

struct ScalarValue {
    ScalarType type{ScalarType::U64};
    union {
        bool b;
        int64_t i64;
        uint64_t u64;
    } v{};

    static ScalarValue from_bool(bool x) {
        ScalarValue out;
        out.type = ScalarType::Bool;
        out.v.b = x;
        return out;
    }
    static ScalarValue from_i64(int64_t x) {
        ScalarValue out;
        out.type = ScalarType::I64;
        out.v.i64 = x;
        return out;
    }
    static ScalarValue from_u64(uint64_t x) {
        ScalarValue out;
        out.type = ScalarType::U64;
        out.v.u64 = x;
        return out;
    }
};

enum class ScalarExprKind : uint8_t {
    LiteralBool,
    LiteralI64,
    LiteralU64,
    TaskParam,     // task.params[i]
    TaskTagU64,    // task.tags[name] (u64 / hashed)
    AxisVar,       // t.get("axis") / bindings
    SymbolU64,     // symbol_u64(name)
    Slot,          // slot_u64(i) / slot_i64(i) / slot_bool(i)
    Unary,
    Binary,
    Ternary,
    Cast,
};

struct ScalarExprNode;
using ScalarExpr = std::shared_ptr<ScalarExprNode>;

struct ScalarExprNode {
    const ScalarExprKind kind;
    const ScalarType type;

    ScalarExprNode(ScalarExprKind kind, ScalarType type) : kind(kind), type(type) {}
    virtual ~ScalarExprNode() = default;

    virtual void print(std::ostream& os) const = 0;
};

inline std::string to_string(const ScalarExpr& e) {
    if (!e) return "<null>";
    std::ostringstream oss;
    e->print(oss);
    return oss.str();
}

struct LiteralBoolExpr final : ScalarExprNode {
    const bool value;
    explicit LiteralBoolExpr(bool v) : ScalarExprNode(ScalarExprKind::LiteralBool, ScalarType::Bool), value(v) {}
    void print(std::ostream& os) const override { os << (value ? "true" : "false"); }
};

struct LiteralI64Expr final : ScalarExprNode {
    const int64_t value;
    explicit LiteralI64Expr(int64_t v) : ScalarExprNode(ScalarExprKind::LiteralI64, ScalarType::I64), value(v) {}
    void print(std::ostream& os) const override { os << value; }
};

struct LiteralU64Expr final : ScalarExprNode {
    const uint64_t value;
    explicit LiteralU64Expr(uint64_t v) : ScalarExprNode(ScalarExprKind::LiteralU64, ScalarType::U64), value(v) {}
    void print(std::ostream& os) const override { os << value << "u"; }
};

struct TaskParamExpr final : ScalarExprNode {
    const uint32_t index;
    explicit TaskParamExpr(uint32_t index, ScalarType t = ScalarType::I64)
        : ScalarExprNode(ScalarExprKind::TaskParam, t), index(index) {}
    void print(std::ostream& os) const override { os << "task_param(" << index << ")"; }
};

struct TaskTagU64Expr final : ScalarExprNode {
    const uint64_t tag_id;
    explicit TaskTagU64Expr(uint64_t tag_id) : ScalarExprNode(ScalarExprKind::TaskTagU64, ScalarType::U64), tag_id(tag_id) {}
    void print(std::ostream& os) const override { os << "task_tag_u64(0x" << std::hex << tag_id << std::dec << ")"; }
};

struct AxisVarExpr final : ScalarExprNode {
    const uint64_t axis_id;
    explicit AxisVarExpr(uint64_t axis_id, ScalarType t = ScalarType::I64)
        : ScalarExprNode(ScalarExprKind::AxisVar, t), axis_id(axis_id) {}
    void print(std::ostream& os) const override { os << "axis_var(0x" << std::hex << axis_id << std::dec << ")"; }
};

struct SymbolU64Expr final : ScalarExprNode {
    const uint64_t symbol_id;
    explicit SymbolU64Expr(uint64_t symbol_id) : ScalarExprNode(ScalarExprKind::SymbolU64, ScalarType::U64), symbol_id(symbol_id) {}
    void print(std::ostream& os) const override { os << "symbol_u64(0x" << std::hex << symbol_id << std::dec << ")"; }
};

struct SlotExpr final : ScalarExprNode {
    const uint32_t index;
    explicit SlotExpr(uint32_t index, ScalarType t)
        : ScalarExprNode(ScalarExprKind::Slot, t), index(index) {}
    void print(std::ostream& os) const override {
        const char* fn = "slot_u64";
        if (type == ScalarType::I64) fn = "slot_i64";
        if (type == ScalarType::Bool) fn = "slot_bool";
        os << fn << "(" << index << ")";
    }
};

enum class ScalarUnaryOp : uint8_t {
    Not,     // !
    Neg,     // -
    BitNot,  // ~
};

struct UnaryExpr final : ScalarExprNode {
    const ScalarUnaryOp op;
    const ScalarExpr expr;

    UnaryExpr(ScalarUnaryOp op, ScalarExpr expr, ScalarType type)
        : ScalarExprNode(ScalarExprKind::Unary, type), op(op), expr(std::move(expr)) {}

    void print(std::ostream& os) const override {
        switch (op) {
        case ScalarUnaryOp::Not: os << "(!"; break;
        case ScalarUnaryOp::Neg: os << "(-"; break;
        case ScalarUnaryOp::BitNot: os << "(~"; break;
        }
        if (expr) expr->print(os);
        os << ")";
    }
};

enum class ScalarBinaryOp : uint8_t {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    BitAnd,
    BitOr,
    BitXor,
    Shl,
    Shr,
    Lt,
    Le,
    Gt,
    Ge,
    Eq,
    Ne,
    And,
    Or,
};

inline const char* binaryOpToString(ScalarBinaryOp op) {
    switch (op) {
    case ScalarBinaryOp::Add: return "+";
    case ScalarBinaryOp::Sub: return "-";
    case ScalarBinaryOp::Mul: return "*";
    case ScalarBinaryOp::Div: return "/";
    case ScalarBinaryOp::Mod: return "%";
    case ScalarBinaryOp::BitAnd: return "&";
    case ScalarBinaryOp::BitOr: return "|";
    case ScalarBinaryOp::BitXor: return "^";
    case ScalarBinaryOp::Shl: return "<<";
    case ScalarBinaryOp::Shr: return ">>";
    case ScalarBinaryOp::Lt: return "<";
    case ScalarBinaryOp::Le: return "<=";
    case ScalarBinaryOp::Gt: return ">";
    case ScalarBinaryOp::Ge: return ">=";
    case ScalarBinaryOp::Eq: return "==";
    case ScalarBinaryOp::Ne: return "!=";
    case ScalarBinaryOp::And: return "&&";
    case ScalarBinaryOp::Or: return "||";
    }
    return "?";
}

struct BinaryExpr final : ScalarExprNode {
    const ScalarBinaryOp op;
    const ScalarExpr lhs;
    const ScalarExpr rhs;

    BinaryExpr(ScalarBinaryOp op, ScalarExpr lhs, ScalarExpr rhs, ScalarType type)
        : ScalarExprNode(ScalarExprKind::Binary, type), op(op), lhs(std::move(lhs)), rhs(std::move(rhs)) {}

    void print(std::ostream& os) const override {
        os << "(";
        if (lhs) lhs->print(os);
        os << " " << binaryOpToString(op) << " ";
        if (rhs) rhs->print(os);
        os << ")";
    }
};

struct TernaryExpr final : ScalarExprNode {
    const ScalarExpr cond;
    const ScalarExpr then_expr;
    const ScalarExpr else_expr;

    TernaryExpr(ScalarExpr cond, ScalarExpr then_expr, ScalarExpr else_expr, ScalarType type)
        : ScalarExprNode(ScalarExprKind::Ternary, type),
          cond(std::move(cond)),
          then_expr(std::move(then_expr)),
          else_expr(std::move(else_expr)) {}

    void print(std::ostream& os) const override {
        os << "(";
        if (cond) cond->print(os);
        os << " ? ";
        if (then_expr) then_expr->print(os);
        os << " : ";
        if (else_expr) else_expr->print(os);
        os << ")";
    }
};

struct CastExpr final : ScalarExprNode {
    const ScalarType to;
    const ScalarExpr expr;

    CastExpr(ScalarType to, ScalarExpr expr)
        : ScalarExprNode(ScalarExprKind::Cast, to), to(to), expr(std::move(expr)) {}

    void print(std::ostream& os) const override {
        os << "(" << scalarTypeToString(to) << ")(";
        if (expr) expr->print(os);
        os << ")";
    }
};

struct ScalarExprEvalContext {
    // Expected to return the requested values for the current task context.
    std::function<int64_t(uint32_t)> task_param_i64;
    std::function<uint64_t(uint64_t)> task_tag_u64;
    std::function<int64_t(uint64_t)> axis_var_i64;
    std::function<uint64_t(uint64_t)> symbol_u64;
    std::function<uint64_t(uint32_t)> slot_u64;
};

inline uint64_t as_u64(const ScalarValue& v) {
    if (v.type != ScalarType::U64) throw std::runtime_error("ScalarValue: expected u64");
    return v.v.u64;
}

inline int64_t as_i64(const ScalarValue& v) {
    if (v.type != ScalarType::I64) throw std::runtime_error("ScalarValue: expected i64");
    return v.v.i64;
}

inline bool as_bool(const ScalarValue& v) {
    if (v.type != ScalarType::Bool) throw std::runtime_error("ScalarValue: expected bool");
    return v.v.b;
}

inline ScalarValue eval_scalar_expr(const ScalarExpr& e, const ScalarExprEvalContext& ctx) {
    if (!e) throw std::runtime_error("ScalarExpr: null");

    switch (e->kind) {
    case ScalarExprKind::LiteralBool:
        return ScalarValue::from_bool(static_cast<const LiteralBoolExpr&>(*e).value);
    case ScalarExprKind::LiteralI64:
        return ScalarValue::from_i64(static_cast<const LiteralI64Expr&>(*e).value);
    case ScalarExprKind::LiteralU64:
        return ScalarValue::from_u64(static_cast<const LiteralU64Expr&>(*e).value);
    case ScalarExprKind::TaskParam: {
        const auto& n = static_cast<const TaskParamExpr&>(*e);
        if (!ctx.task_param_i64) throw std::runtime_error("ScalarExprEvalContext: task_param_i64 not set");
        const int64_t v = ctx.task_param_i64(n.index);
        if (n.type == ScalarType::I64) return ScalarValue::from_i64(v);
        if (n.type == ScalarType::U64) return ScalarValue::from_u64(static_cast<uint64_t>(v));
        throw std::runtime_error("TaskParamExpr: invalid type");
    }
    case ScalarExprKind::TaskTagU64: {
        const auto& n = static_cast<const TaskTagU64Expr&>(*e);
        if (!ctx.task_tag_u64) throw std::runtime_error("ScalarExprEvalContext: task_tag_u64 not set");
        return ScalarValue::from_u64(ctx.task_tag_u64(n.tag_id));
    }
    case ScalarExprKind::AxisVar: {
        const auto& n = static_cast<const AxisVarExpr&>(*e);
        if (!ctx.axis_var_i64) throw std::runtime_error("ScalarExprEvalContext: axis_var_i64 not set");
        const int64_t v = ctx.axis_var_i64(n.axis_id);
        if (n.type == ScalarType::I64) return ScalarValue::from_i64(v);
        if (n.type == ScalarType::U64) return ScalarValue::from_u64(static_cast<uint64_t>(v));
        throw std::runtime_error("AxisVarExpr: invalid type");
    }
    case ScalarExprKind::SymbolU64: {
        const auto& n = static_cast<const SymbolU64Expr&>(*e);
        if (!ctx.symbol_u64) throw std::runtime_error("ScalarExprEvalContext: symbol_u64 not set");
        return ScalarValue::from_u64(ctx.symbol_u64(n.symbol_id));
    }
    case ScalarExprKind::Slot: {
        const auto& n = static_cast<const SlotExpr&>(*e);
        if (!ctx.slot_u64) throw std::runtime_error("ScalarExprEvalContext: slot_u64 not set");
        const uint64_t v = ctx.slot_u64(n.index);
        if (n.type == ScalarType::U64) return ScalarValue::from_u64(v);
        if (n.type == ScalarType::I64) return ScalarValue::from_i64((int64_t)v);
        if (n.type == ScalarType::Bool) return ScalarValue::from_bool(v != 0);
        throw std::runtime_error("SlotExpr: invalid type");
    }
    case ScalarExprKind::Unary: {
        const auto& n = static_cast<const UnaryExpr&>(*e);
        const auto v = eval_scalar_expr(n.expr, ctx);
        switch (n.op) {
        case ScalarUnaryOp::Not:
            return ScalarValue::from_bool(!as_bool(v));
        case ScalarUnaryOp::Neg:
            if (n.type == ScalarType::I64) return ScalarValue::from_i64(-as_i64(v));
            if (n.type == ScalarType::U64) return ScalarValue::from_u64((uint64_t)(0ULL - as_u64(v)));
            throw std::runtime_error("Unary neg: invalid type");
        case ScalarUnaryOp::BitNot:
            if (n.type != ScalarType::U64) throw std::runtime_error("Unary bitnot: expected u64");
            return ScalarValue::from_u64(~as_u64(v));
        }
        throw std::runtime_error("Unary: unknown op");
    }
    case ScalarExprKind::Binary: {
        const auto& n = static_cast<const BinaryExpr&>(*e);
        const auto a = eval_scalar_expr(n.lhs, ctx);
        const auto b = eval_scalar_expr(n.rhs, ctx);

        const bool is_bool = (n.op == ScalarBinaryOp::And || n.op == ScalarBinaryOp::Or);
        if (is_bool) {
            if (n.op == ScalarBinaryOp::And) return ScalarValue::from_bool(as_bool(a) && as_bool(b));
            return ScalarValue::from_bool(as_bool(a) || as_bool(b));
        }

        const bool is_cmp =
            n.op == ScalarBinaryOp::Lt || n.op == ScalarBinaryOp::Le || n.op == ScalarBinaryOp::Gt ||
            n.op == ScalarBinaryOp::Ge || n.op == ScalarBinaryOp::Eq || n.op == ScalarBinaryOp::Ne;
        if (is_cmp) {
            if (a.type != b.type) throw std::runtime_error("ScalarExpr: comparison with mismatched types");
            if (a.type == ScalarType::I64) {
                const int64_t x = a.v.i64, y = b.v.i64;
                bool r = false;
                switch (n.op) {
                case ScalarBinaryOp::Lt: r = x < y; break;
                case ScalarBinaryOp::Le: r = x <= y; break;
                case ScalarBinaryOp::Gt: r = x > y; break;
                case ScalarBinaryOp::Ge: r = x >= y; break;
                case ScalarBinaryOp::Eq: r = x == y; break;
                case ScalarBinaryOp::Ne: r = x != y; break;
                default: break;
                }
                return ScalarValue::from_bool(r);
            }
            if (a.type == ScalarType::U64) {
                const uint64_t x = a.v.u64, y = b.v.u64;
                bool r = false;
                switch (n.op) {
                case ScalarBinaryOp::Lt: r = x < y; break;
                case ScalarBinaryOp::Le: r = x <= y; break;
                case ScalarBinaryOp::Gt: r = x > y; break;
                case ScalarBinaryOp::Ge: r = x >= y; break;
                case ScalarBinaryOp::Eq: r = x == y; break;
                case ScalarBinaryOp::Ne: r = x != y; break;
                default: break;
                }
                return ScalarValue::from_bool(r);
            }
            if (a.type == ScalarType::Bool) {
                const bool x = a.v.b, y = b.v.b;
                bool r = false;
                switch (n.op) {
                case ScalarBinaryOp::Eq: r = x == y; break;
                case ScalarBinaryOp::Ne: r = x != y; break;
                default: throw std::runtime_error("ScalarExpr: unsupported bool comparison op");
                }
                return ScalarValue::from_bool(r);
            }
            throw std::runtime_error("ScalarExpr: unsupported comparison type");
        }

        if (a.type != b.type) throw std::runtime_error("ScalarExpr: binary op with mismatched types");
        if (a.type == ScalarType::I64) {
            const int64_t x = a.v.i64, y = b.v.i64;
            switch (n.op) {
            case ScalarBinaryOp::Add: return ScalarValue::from_i64(x + y);
            case ScalarBinaryOp::Sub: return ScalarValue::from_i64(x - y);
            case ScalarBinaryOp::Mul: return ScalarValue::from_i64(x * y);
            case ScalarBinaryOp::Div: return ScalarValue::from_i64(y == 0 ? 0 : (x / y));
            case ScalarBinaryOp::Mod: return ScalarValue::from_i64(y == 0 ? 0 : (x % y));
            default: throw std::runtime_error("ScalarExpr: unsupported i64 binary op");
            }
        }
        if (a.type == ScalarType::U64) {
            const uint64_t x = a.v.u64, y = b.v.u64;
            switch (n.op) {
            case ScalarBinaryOp::Add: return ScalarValue::from_u64(x + y);
            case ScalarBinaryOp::Sub: return ScalarValue::from_u64(x - y);
            case ScalarBinaryOp::Mul: return ScalarValue::from_u64(x * y);
            case ScalarBinaryOp::Div: return ScalarValue::from_u64(y == 0 ? 0 : (x / y));
            case ScalarBinaryOp::Mod: return ScalarValue::from_u64(y == 0 ? 0 : (x % y));
            case ScalarBinaryOp::BitAnd: return ScalarValue::from_u64(x & y);
            case ScalarBinaryOp::BitOr: return ScalarValue::from_u64(x | y);
            case ScalarBinaryOp::BitXor: return ScalarValue::from_u64(x ^ y);
            case ScalarBinaryOp::Shl: return ScalarValue::from_u64(x << (y & 63ULL));
            case ScalarBinaryOp::Shr: return ScalarValue::from_u64(x >> (y & 63ULL));
            default: throw std::runtime_error("ScalarExpr: unsupported u64 binary op");
            }
        }
        throw std::runtime_error("ScalarExpr: unsupported binary types");
    }
    case ScalarExprKind::Ternary: {
        const auto& n = static_cast<const TernaryExpr&>(*e);
        const bool c = as_bool(eval_scalar_expr(n.cond, ctx));
        return eval_scalar_expr(c ? n.then_expr : n.else_expr, ctx);
    }
    case ScalarExprKind::Cast: {
        const auto& n = static_cast<const CastExpr&>(*e);
        const auto v = eval_scalar_expr(n.expr, ctx);
        if (n.to == ScalarType::Bool) {
            if (v.type == ScalarType::Bool) return v;
            if (v.type == ScalarType::I64) return ScalarValue::from_bool(v.v.i64 != 0);
            if (v.type == ScalarType::U64) return ScalarValue::from_bool(v.v.u64 != 0);
        }
        if (n.to == ScalarType::I64) {
            if (v.type == ScalarType::I64) return v;
            if (v.type == ScalarType::U64) return ScalarValue::from_i64((int64_t)v.v.u64);
            if (v.type == ScalarType::Bool) return ScalarValue::from_i64(v.v.b ? 1 : 0);
        }
        if (n.to == ScalarType::U64) {
            if (v.type == ScalarType::U64) return v;
            if (v.type == ScalarType::I64) return ScalarValue::from_u64((uint64_t)v.v.i64);
            if (v.type == ScalarType::Bool) return ScalarValue::from_u64(v.v.b ? 1ULL : 0ULL);
        }
        throw std::runtime_error("ScalarExpr: unsupported cast");
    }
    }

    throw std::runtime_error("ScalarExpr: unknown kind");
}

// Convenience constructors (minimal, keep API stable for pybind wrappers).
inline ScalarExpr lit_bool(bool v) { return std::make_shared<LiteralBoolExpr>(v); }
inline ScalarExpr lit_i64(int64_t v) { return std::make_shared<LiteralI64Expr>(v); }
inline ScalarExpr lit_u64(uint64_t v) { return std::make_shared<LiteralU64Expr>(v); }
inline ScalarExpr task_param(uint32_t i, ScalarType t = ScalarType::I64) { return std::make_shared<TaskParamExpr>(i, t); }
inline ScalarExpr task_tag_u64(uint64_t id) { return std::make_shared<TaskTagU64Expr>(id); }
inline ScalarExpr axis_var(uint64_t id, ScalarType t = ScalarType::I64) { return std::make_shared<AxisVarExpr>(id, t); }
inline ScalarExpr symbol_u64(uint64_t id) { return std::make_shared<SymbolU64Expr>(id); }
inline ScalarExpr slot_u64(uint32_t i) { return std::make_shared<SlotExpr>(i, ScalarType::U64); }
inline ScalarExpr slot_i64(uint32_t i) { return std::make_shared<SlotExpr>(i, ScalarType::I64); }
inline ScalarExpr slot_bool(uint32_t i) { return std::make_shared<SlotExpr>(i, ScalarType::Bool); }
inline ScalarExpr unary(ScalarUnaryOp op, ScalarExpr e, ScalarType t) { return std::make_shared<UnaryExpr>(op, std::move(e), t); }
inline ScalarExpr binary(ScalarBinaryOp op, ScalarExpr a, ScalarExpr b, ScalarType t) { return std::make_shared<BinaryExpr>(op, std::move(a), std::move(b), t); }
inline ScalarExpr ternary(ScalarExpr c, ScalarExpr t, ScalarExpr f, ScalarType ty) { return std::make_shared<TernaryExpr>(std::move(c), std::move(t), std::move(f), ty); }
inline ScalarExpr cast(ScalarType to, ScalarExpr e) { return std::make_shared<CastExpr>(to, std::move(e)); }

}  // namespace pto::wsp::ir
