"""
Scalar expression (ScalarExpr) DSL for PTO-RT v9.

This is a **symbolic** expression system used for:
- runtime predicates (cond)
- programmable scheduling keys (dispatch, stream_by, etc.)

Design constraints (v9):
- Expressions must be convertible to the C++ ScalarExpr IR (pto_ir_cpp.ScalarExpr).
- Expressions must not rely on Python callbacks at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Union


def fnv1a_64(s: str) -> int:
    """Stable 64-bit FNV-1a hash for identifier strings (matches codegen scheme)."""
    h = 0xCBF29CE484222325
    for b in s.encode("utf-8"):
        h ^= b
        h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


ScalarLiteral = Union[bool, int]


@dataclass(frozen=True)
class Expr:
    kind: str
    type: str  # "bool" | "i64" | "u64"
    a: Any = None
    b: Any = None
    c: Any = None

    # ------------------------------------------------------------
    # Conversions
    # ------------------------------------------------------------
    def to_cpp(self):
        try:
            from pto_wsp import pto_ir_cpp as cpp  # pip install -e .
        except Exception:  # pragma: no cover
            import pto_ir_cpp as cpp

        if self.kind == "lit_bool":
            return cpp.scalar_lit_bool(bool(self.a))
        if self.kind == "lit_i64":
            return cpp.scalar_lit_i64(int(self.a))
        if self.kind == "lit_u64":
            return cpp.scalar_lit_u64(int(self.a) & 0xFFFFFFFFFFFFFFFF)
        if self.kind == "task_param":
            return cpp.scalar_task_param(int(self.a), _cpp_type(cpp, self.type))
        if self.kind == "task_tag_u64":
            return cpp.scalar_task_tag_u64(int(self.a) & 0xFFFFFFFFFFFFFFFF)
        if self.kind == "axis_var":
            return cpp.scalar_axis_var(int(self.a) & 0xFFFFFFFFFFFFFFFF, _cpp_type(cpp, self.type))
        if self.kind == "symbol_u64":
            return cpp.scalar_symbol_u64(int(self.a) & 0xFFFFFFFFFFFFFFFF)
        if self.kind == "slot_u64":
            return cpp.scalar_slot_u64(int(self.a))
        if self.kind == "slot_i64":
            return cpp.scalar_slot_i64(int(self.a))
        if self.kind == "slot_bool":
            return cpp.scalar_slot_bool(int(self.a))
        if self.kind == "unary":
            op = _cpp_unary_op(cpp, str(self.a))
            return cpp.scalar_unary(op, _as_expr(self.b).to_cpp(), _cpp_type(cpp, self.type))
        if self.kind == "binary":
            op = _cpp_binary_op(cpp, str(self.a))
            return cpp.scalar_binary(op, _as_expr(self.b).to_cpp(), _as_expr(self.c).to_cpp(), _cpp_type(cpp, self.type))
        if self.kind == "ternary":
            return cpp.scalar_ternary(_as_expr(self.a).to_cpp(), _as_expr(self.b).to_cpp(), _as_expr(self.c).to_cpp(), _cpp_type(cpp, self.type))
        if self.kind == "cast":
            return cpp.scalar_cast(_cpp_type(cpp, self.type), _as_expr(self.a).to_cpp())
        raise TypeError(f"Unsupported Expr kind: {self.kind}")

    def to_debug_str(self) -> str:
        if self.kind.startswith("lit_"):
            return repr(self.a)
        if self.kind == "task_param":
            return f"task.params[{self.a}]"
        if self.kind == "axis_var":
            return f"axis({self.a:#x})"
        if self.kind == "task_tag_u64":
            return f"tag({self.a:#x})"
        if self.kind == "symbol_u64":
            return f"symbol({self.a:#x})"
        if self.kind.startswith("slot_"):
            return f"{self.kind}({self.a})"
        if self.kind == "unary":
            return f"({self.a} {_as_expr(self.b).to_debug_str()})"
        if self.kind == "binary":
            return f"({_as_expr(self.b).to_debug_str()} {self.a} {_as_expr(self.c).to_debug_str()})"
        if self.kind == "ternary":
            return f"({_as_expr(self.a).to_debug_str()} ? {_as_expr(self.b).to_debug_str()} : {_as_expr(self.c).to_debug_str()})"
        if self.kind == "cast":
            return f"({self.type})({_as_expr(self.a).to_debug_str()})"
        return f"<expr:{self.kind}>"

    # ------------------------------------------------------------
    # Operator overloads (note: Python 'and/or' cannot be overloaded)
    # ------------------------------------------------------------
    def __bool__(self) -> bool:  # pragma: no cover
        raise TypeError("ScalarExpr cannot be used with Python 'and/or' or in 'if'. Use '&'/'|' or Expr.where().")

    def where(self, then_expr: Any, else_expr: Any) -> "Expr":
        if self.type != "bool":
            raise TypeError("where() requires a bool predicate Expr")
        t = _as_expr(then_expr)
        f = _as_expr(else_expr)
        if t.type != f.type:
            raise TypeError("where() branch type mismatch")
        return Expr("ternary", t.type, self, t, f)

    # unary
    def __invert__(self) -> "Expr":
        if self.type == "bool":
            return Expr("unary", "bool", "!", self)
        if self.type == "u64":
            return Expr("unary", "u64", "~", self)
        raise TypeError("~ supported for bool/u64 only")

    def __neg__(self) -> "Expr":
        if self.type not in ("i64", "u64"):
            raise TypeError("Unary - supported for i64/u64 only")
        return Expr("unary", self.type, "-", self)

    # binary helpers
    def _bin(self, other: Any, op: str, out_type: Optional[str] = None) -> "Expr":
        rhs = _as_expr(other)
        lhs = self
        if out_type is None:
            out_type = lhs.type
        return Expr("binary", out_type, op, lhs, rhs)

    # arithmetic
    def __add__(self, other: Any) -> "Expr": return self._bin(other, "+")
    def __sub__(self, other: Any) -> "Expr": return self._bin(other, "-")
    def __mul__(self, other: Any) -> "Expr": return self._bin(other, "*")
    def __floordiv__(self, other: Any) -> "Expr": return self._bin(other, "/")
    def __mod__(self, other: Any) -> "Expr": return self._bin(other, "%")

    # bitwise / logical (use '&'/'|' for bool, bitwise for u64)
    def __and__(self, other: Any) -> "Expr":
        rhs = _as_expr(other)
        if self.type == "bool" and rhs.type == "bool":
            return Expr("binary", "bool", "&&", self, rhs)
        if self.type == "u64" and rhs.type == "u64":
            return Expr("binary", "u64", "&", self, rhs)
        raise TypeError("& requires both bool or both u64")

    def __or__(self, other: Any) -> "Expr":
        rhs = _as_expr(other)
        if self.type == "bool" and rhs.type == "bool":
            return Expr("binary", "bool", "||", self, rhs)
        if self.type == "u64" and rhs.type == "u64":
            return Expr("binary", "u64", "|", self, rhs)
        raise TypeError("| requires both bool or both u64")

    def __xor__(self, other: Any) -> "Expr":
        rhs = _as_expr(other)
        if self.type == "u64" and rhs.type == "u64":
            return Expr("binary", "u64", "^", self, rhs)
        raise TypeError("^ requires u64")

    def __lshift__(self, other: Any) -> "Expr":
        rhs = _as_expr(other).cast_u64()
        return Expr("binary", "u64", "<<", self.cast_u64(), rhs)

    def __rshift__(self, other: Any) -> "Expr":
        rhs = _as_expr(other).cast_u64()
        return Expr("binary", "u64", ">>", self.cast_u64(), rhs)

    # comparisons
    def __lt__(self, other: Any) -> "Expr": return Expr("binary", "bool", "<", self, _as_expr(other))
    def __le__(self, other: Any) -> "Expr": return Expr("binary", "bool", "<=", self, _as_expr(other))
    def __gt__(self, other: Any) -> "Expr": return Expr("binary", "bool", ">", self, _as_expr(other))
    def __ge__(self, other: Any) -> "Expr": return Expr("binary", "bool", ">=", self, _as_expr(other))
    def __eq__(self, other: Any) -> "Expr": return Expr("binary", "bool", "==", self, _as_expr(other))
    def __ne__(self, other: Any) -> "Expr": return Expr("binary", "bool", "!=", self, _as_expr(other))

    # casts
    def cast_bool(self) -> "Expr": return Expr("cast", "bool", self)
    def cast_i64(self) -> "Expr": return Expr("cast", "i64", self)
    def cast_u64(self) -> "Expr": return Expr("cast", "u64", self)


def _cpp_type(cpp, t: str):
    if t == "bool":
        return cpp.ScalarType.Bool
    if t == "i64":
        return cpp.ScalarType.I64
    if t == "u64":
        return cpp.ScalarType.U64
    raise TypeError(f"Unknown scalar type: {t}")


def _cpp_unary_op(cpp, op: str):
    if op == "!":
        return cpp.ScalarUnaryOp.Not
    if op == "-":
        return cpp.ScalarUnaryOp.Neg
    if op == "~":
        return cpp.ScalarUnaryOp.BitNot
    raise TypeError(f"Unsupported unary op: {op!r}")


def _cpp_binary_op(cpp, op: str):
    mapping = {
        "+": cpp.ScalarBinaryOp.Add,
        "-": cpp.ScalarBinaryOp.Sub,
        "*": cpp.ScalarBinaryOp.Mul,
        "/": cpp.ScalarBinaryOp.Div,
        "%": cpp.ScalarBinaryOp.Mod,
        "&": cpp.ScalarBinaryOp.BitAnd,
        "|": cpp.ScalarBinaryOp.BitOr,
        "^": cpp.ScalarBinaryOp.BitXor,
        "<<": cpp.ScalarBinaryOp.Shl,
        ">>": cpp.ScalarBinaryOp.Shr,
        "<": cpp.ScalarBinaryOp.Lt,
        "<=": cpp.ScalarBinaryOp.Le,
        ">": cpp.ScalarBinaryOp.Gt,
        ">=": cpp.ScalarBinaryOp.Ge,
        "==": cpp.ScalarBinaryOp.Eq,
        "!=": cpp.ScalarBinaryOp.Ne,
        "&&": cpp.ScalarBinaryOp.And,
        "||": cpp.ScalarBinaryOp.Or,
    }
    if op not in mapping:
        raise TypeError(f"Unsupported binary op: {op!r}")
    return mapping[op]


def _as_expr(x: Any) -> Expr:
    if isinstance(x, Expr):
        return x
    if isinstance(x, bool):
        return Expr("lit_bool", "bool", x)
    if isinstance(x, int):
        return Expr("lit_i64", "i64", x)
    raise TypeError(f"Cannot convert to ScalarExpr: {x!r}")


def as_expr(x: Any) -> Expr:
    return _as_expr(x)


class _ParamsProxy:
    def __getitem__(self, i: int) -> Expr:
        return Expr("task_param", "i64", int(i))


class _TagsProxy:
    def __getitem__(self, name: str) -> Expr:
        return Expr("task_tag_u64", "u64", fnv1a_64(str(name)))


class TaskExpr:
    """Proxy object for tracing schedule lambdas: lambda t: t.get('b') % 2."""

    def __init__(self):
        self.params = _ParamsProxy()
        self.tags = _TagsProxy()

    def get(self, axis: str, default: int = 0) -> Expr:
        _ = default
        return Expr("axis_var", "i64", fnv1a_64(str(axis)))


def symbol_u64(name: str) -> Expr:
    return Expr("symbol_u64", "u64", fnv1a_64(str(name)))


def slot_u64(i: int) -> Expr:
    return Expr("slot_u64", "u64", int(i))


def slot_i64(i: int) -> Expr:
    return Expr("slot_i64", "i64", int(i))


def slot_bool(i: int) -> Expr:
    return Expr("slot_bool", "bool", int(i))


def axis(name: str) -> Expr:
    return Expr("axis_var", "i64", fnv1a_64(str(name)))


def task_param(index: int) -> Expr:
    return Expr("task_param", "i64", int(index))


def trace_task_fn(fn: Callable[[TaskExpr], Any]) -> Expr:
    """Run fn with a TaskExpr proxy to build a symbolic expression."""
    out = fn(TaskExpr())
    return _as_expr(out)


def to_cpp(expr: Any):
    return _as_expr(expr).to_cpp()
