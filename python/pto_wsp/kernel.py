"""
JAX/Triton-Style JIT Kernel Programming for PTO Workload-Schedule Programming (PTO-WSP) framework.

This module replaces the string-based NPU function builder with typed IR.
Inspired by Triton's @triton.jit and JAX's @jax.jit patterns.

Example (Recommended API using pto namespace):
    from pto_wsp import kernel, In, Out, Tile, Scalar, pto, DType

    @kernel
    def rmsnorm(x: In[Tile[32, 128, DType.F16]],
                out: Out[Tile[32, 128, DType.F16]],
                eps: Scalar[DType.F32] = 1e-6):
        # Typed values using pto.* primitives
        sq = pto.mul(x, x)              # returns Value
        mean = pto.rowmean(sq)          # returns Value
        rsqrt_val = pto.rsqrt(mean)     # returns Value
        pto.store(out, pto.mul(x, rsqrt_val))

Note:
    The `tl` namespace is a legacy alias for `pto` and is DEPRECATED.
    Please migrate to `pto.*` primitives for new code.
"""

from __future__ import annotations
from typing import (
    Any, Optional, Union, Callable, TypeVar, Generic,
    get_type_hints, get_origin, get_args
)
from dataclasses import dataclass, field
from enum import Enum, auto
import functools
import inspect
import ast

from pto_wsp.types import DType, Location, Tensor

__all__ = [
    # Core types
    "Value", "TileType", "ScalarType",
    # Kernel decorators
    "jit_kernel", "kernel",
    # Type annotations
    "In", "Out", "InOut", "Tile", "Scalar", "Constexpr",
    # Tile language primitives (pto.* - recommended)
    "pto",
    # Legacy alias (tl.* - deprecated, use pto.*)
    "tl",
    # Compiled kernel
    "CompiledKernel", "KernelIR",
]


# ============================================================
# Type System for Kernel Programming
# ============================================================

@dataclass(frozen=True)
class TileType:
    """Tile shape and dtype specification.

    Example:
        Tile[32, 128, F16]  # 32x128 tile of float16
        Tile[M, N, dtype]   # Symbolic dimensions
    """
    rows: Union[int, str]  # int for concrete, str for symbolic
    cols: Union[int, str]
    dtype: DType
    location: Location = Location.UB


@dataclass(frozen=True)
class ScalarType:
    """Scalar type specification."""
    dtype: DType


class Constexpr:
    """Marker for compile-time constant parameters (like Triton's tl.constexpr)."""
    pass


# Type annotation helpers
T = TypeVar('T')

class In(Generic[T]):
    """Input parameter marker."""
    pass

class Out(Generic[T]):
    """Output parameter marker."""
    pass

class InOut(Generic[T]):
    """Input/Output parameter marker."""
    pass

class Tile(Generic[T]):
    """Tile type annotation. Use Tile[rows, cols, dtype]."""
    def __class_getitem__(cls, args):
        if not isinstance(args, tuple):
            args = (args,)
        if len(args) == 2:
            rows, cols = args
            dtype = DType.F16
        elif len(args) == 3:
            rows, cols, dtype = args
        else:
            raise TypeError(f"Tile expects 2-3 args, got {len(args)}")
        return TileType(rows, cols, dtype)


class Scalar:
    """Scalar type annotation. Use Scalar[dtype]."""
    def __class_getitem__(cls, dtype):
        return ScalarType(dtype)


# ============================================================
# Typed IR Values (SSA-style, no string names)
# ============================================================

class ValueId:
    """Auto-incrementing value ID generator."""
    _counter = 0

    @classmethod
    def next(cls) -> int:
        cls._counter += 1
        return cls._counter

    @classmethod
    def reset(cls):
        cls._counter = 0


@dataclass
class Value:
    """Typed IR value - replaces string-based references.

    Every operation returns a Value, enabling type-safe chaining:
        sq = tl.mul(x, x)  # sq is a Value
        mean = tl.rowmean(sq)  # mean is a Value
    """
    id: int
    dtype: DType
    shape: Optional[tuple] = None  # (rows, cols) for tiles, None for scalars
    location: Location = Location.UB
    debug_name: Optional[str] = None  # Optional for debugging

    @classmethod
    def tile(cls, rows: int, cols: int, dtype: DType = DType.F16,
             location: Location = Location.UB, name: str = None) -> "Value":
        """Create a tile value."""
        return cls(
            id=ValueId.next(),
            dtype=dtype,
            shape=(rows, cols),
            location=location,
            debug_name=name
        )

    @classmethod
    def scalar(cls, dtype: DType = DType.F32, name: str = None) -> "Value":
        """Create a scalar value."""
        return cls(
            id=ValueId.next(),
            dtype=dtype,
            shape=None,
            debug_name=name
        )

    def __repr__(self):
        if self.debug_name:
            return f"Value({self.debug_name}:{self.id})"
        return f"Value({self.id})"


# ============================================================
# Kernel IR Operations (typed, no strings)
# ============================================================

class OpKind(Enum):
    """Kernel operation kinds."""
    # Memory
    Load = auto()
    Store = auto()
    Alloc = auto()

    # Binary
    Add = auto()
    Sub = auto()
    Mul = auto()
    Div = auto()
    Max = auto()
    Min = auto()

    # Unary
    Neg = auto()
    Abs = auto()
    Exp = auto()
    Log = auto()
    Sqrt = auto()
    Rsqrt = auto()
    Tanh = auto()
    Sigmoid = auto()
    Relu = auto()
    Gelu = auto()
    Silu = auto()

    # Reduction
    RowSum = auto()
    RowMax = auto()
    RowMean = auto()
    ColSum = auto()
    ColMax = auto()

    # Broadcast
    RowBroadcastAdd = auto()
    RowBroadcastMul = auto()
    RowExpandMul = auto()
    ColBroadcastAdd = auto()
    ColBroadcastMul = auto()

    # Matrix
    Matmul = auto()

    # Special
    Constant = auto()
    Sin = auto()
    Cos = auto()

    # Control flow
    ForBegin = auto()
    ForEnd = auto()
    IfBegin = auto()
    IfEnd = auto()

    # PTO-ISA helpers (v9)
    IotaU32 = auto()
    TSort32 = auto()
    ExtractTopkIndices = auto()


@dataclass
class Op:
    """Single IR operation with typed operands and result."""
    kind: OpKind
    result: Optional[Value]  # None for store, control flow
    operands: list[Value]
    attrs: dict[str, Any] = field(default_factory=dict)

    def __repr__(self):
        operand_ids = [f"v{op.id}" for op in self.operands]
        if self.result:
            return f"v{self.result.id} = {self.kind.name}({', '.join(operand_ids)})"
        return f"{self.kind.name}({', '.join(operand_ids)})"


@dataclass
class KernelIR:
    """Complete kernel IR - typed operations with no string refs."""
    name: str
    params: list[tuple[str, Value]]  # (param_name, value) for function signature
    ops: list[Op] = field(default_factory=list)
    schedule_hints: dict[str, Any] = field(default_factory=dict)

    def add_op(self, kind: OpKind, operands: list[Value],
               result: Optional[Value] = None, **attrs) -> Optional[Value]:
        """Add operation to IR, return result value if any."""
        op = Op(kind=kind, result=result, operands=operands, attrs=attrs)
        self.ops.append(op)
        return result

    def dump(self) -> str:
        """Dump IR as text (for debugging)."""
        lines = [f"@kernel {self.name}("]
        for name, val in self.params:
            lines.append(f"  {name}: v{val.id}")
        lines.append(") {")
        for op in self.ops:
            lines.append(f"  {op}")
        lines.append("}")
        return "\n".join(lines)


# ============================================================
# Tile Language (tl.*) - Triton-style primitives
# ============================================================

class _TileLanguage:
    """Triton-style tile language primitives.

    All operations return typed Value objects (no strings!).

    Usage:
        @kernel
        def my_kernel(x: In[Tile[M, N, F16]], out: Out[Tile[M, N, F16]]):
            sq = tl.mul(x, x)        # Returns Value
            tl.store(out, sq)        # No string refs!
    """

    def __init__(self):
        self._current_ir: Optional[KernelIR] = None

    def _check_tracing(self):
        if self._current_ir is None:
            raise RuntimeError("tl.* must be used inside @kernel function")

    def _infer_result_type(self, *operands: Value) -> tuple:
        """Infer result dtype and shape from operands."""
        dtype = operands[0].dtype if operands else DType.F32
        shape = operands[0].shape if operands else None
        return dtype, shape

    # === Allocation ===

    def alloc(self, shape: tuple[int, int], dtype: DType = DType.F16,
              location: Location = Location.UB) -> Value:
        """Allocate a tile buffer.

        Example:
            temp = tl.alloc((32, 128), dtype=F16)  # Returns Value, not string!
        """
        self._check_tracing()
        result = Value.tile(shape[0], shape[1], dtype, location)
        self._current_ir.add_op(OpKind.Alloc, [], result, shape=shape)
        return result

    # === Memory Operations ===

    def load(self, src: Value, offset: tuple[int, int] = (0, 0),
             mask: Optional[Value] = None) -> Value:
        """Load from memory to tile.

        Example:
            x = tl.load(input_ptr)  # Returns Value
        """
        self._check_tracing()
        dtype, shape = src.dtype, src.shape
        result = Value.tile(shape[0] if shape else 32, shape[1] if shape else 128, dtype)
        operands = [src] if mask is None else [src, mask]
        self._current_ir.add_op(OpKind.Load, operands, result, offset=offset)
        return result

    def store(self, dst: Value, src: Value, offset: tuple[int, int] = (0, 0),
              mask: Optional[Value] = None):
        """Store tile to memory.

        Example:
            tl.store(output_ptr, result)  # No string refs!
        """
        self._check_tracing()
        operands = [dst, src] if mask is None else [dst, src, mask]
        self._current_ir.add_op(OpKind.Store, operands, None, offset=offset)

    # === Binary Operations ===

    def add(self, a: Value, b: Value) -> Value:
        """Element-wise addition."""
        self._check_tracing()
        dtype, shape = self._infer_result_type(a, b)
        result = Value(ValueId.next(), dtype, shape)
        self._current_ir.add_op(OpKind.Add, [a, b], result)
        return result

    def sub(self, a: Value, b: Value) -> Value:
        """Element-wise subtraction."""
        self._check_tracing()
        dtype, shape = self._infer_result_type(a, b)
        result = Value(ValueId.next(), dtype, shape)
        self._current_ir.add_op(OpKind.Sub, [a, b], result)
        return result

    def mul(self, a: Value, b: Value) -> Value:
        """Element-wise multiplication."""
        self._check_tracing()
        dtype, shape = self._infer_result_type(a, b)
        result = Value(ValueId.next(), dtype, shape)
        self._current_ir.add_op(OpKind.Mul, [a, b], result)
        return result

    def div(self, a: Value, b: Value) -> Value:
        """Element-wise division."""
        self._check_tracing()
        dtype, shape = self._infer_result_type(a, b)
        result = Value(ValueId.next(), dtype, shape)
        self._current_ir.add_op(OpKind.Div, [a, b], result)
        return result

    def maximum(self, a: Value, b: Value) -> Value:
        """Element-wise maximum."""
        self._check_tracing()
        dtype, shape = self._infer_result_type(a, b)
        result = Value(ValueId.next(), dtype, shape)
        self._current_ir.add_op(OpKind.Max, [a, b], result)
        return result

    def minimum(self, a: Value, b: Value) -> Value:
        """Element-wise minimum."""
        self._check_tracing()
        dtype, shape = self._infer_result_type(a, b)
        result = Value(ValueId.next(), dtype, shape)
        self._current_ir.add_op(OpKind.Min, [a, b], result)
        return result

    # === Unary Operations ===

    def exp(self, x: Value) -> Value:
        """Element-wise exponential."""
        self._check_tracing()
        result = Value(ValueId.next(), x.dtype, x.shape)
        self._current_ir.add_op(OpKind.Exp, [x], result)
        return result

    def log(self, x: Value) -> Value:
        """Element-wise logarithm."""
        self._check_tracing()
        result = Value(ValueId.next(), x.dtype, x.shape)
        self._current_ir.add_op(OpKind.Log, [x], result)
        return result

    def sqrt(self, x: Value) -> Value:
        """Element-wise square root."""
        self._check_tracing()
        result = Value(ValueId.next(), x.dtype, x.shape)
        self._current_ir.add_op(OpKind.Sqrt, [x], result)
        return result

    def rsqrt(self, x: Value) -> Value:
        """Element-wise reciprocal square root."""
        self._check_tracing()
        result = Value(ValueId.next(), x.dtype, x.shape)
        self._current_ir.add_op(OpKind.Rsqrt, [x], result)
        return result

    def tanh(self, x: Value) -> Value:
        """Element-wise tanh."""
        self._check_tracing()
        result = Value(ValueId.next(), x.dtype, x.shape)
        self._current_ir.add_op(OpKind.Tanh, [x], result)
        return result

    def sigmoid(self, x: Value) -> Value:
        """Element-wise sigmoid."""
        self._check_tracing()
        result = Value(ValueId.next(), x.dtype, x.shape)
        self._current_ir.add_op(OpKind.Sigmoid, [x], result)
        return result

    def relu(self, x: Value) -> Value:
        """Element-wise ReLU."""
        self._check_tracing()
        result = Value(ValueId.next(), x.dtype, x.shape)
        self._current_ir.add_op(OpKind.Relu, [x], result)
        return result

    def gelu(self, x: Value) -> Value:
        """Element-wise GELU."""
        self._check_tracing()
        result = Value(ValueId.next(), x.dtype, x.shape)
        self._current_ir.add_op(OpKind.Gelu, [x], result)
        return result

    def silu(self, x: Value) -> Value:
        """Element-wise SiLU (Swish)."""
        self._check_tracing()
        result = Value(ValueId.next(), x.dtype, x.shape)
        self._current_ir.add_op(OpKind.Silu, [x], result)
        return result

    def neg(self, x: Value) -> Value:
        """Element-wise negation."""
        self._check_tracing()
        result = Value(ValueId.next(), x.dtype, x.shape)
        self._current_ir.add_op(OpKind.Neg, [x], result)
        return result

    def abs(self, x: Value) -> Value:
        """Element-wise absolute value."""
        self._check_tracing()
        result = Value(ValueId.next(), x.dtype, x.shape)
        self._current_ir.add_op(OpKind.Abs, [x], result)
        return result

    # === Reduction Operations ===

    def rowsum(self, x: Value) -> Value:
        """Row-wise sum reduction."""
        self._check_tracing()
        rows = x.shape[0] if x.shape else 1
        out_dtype = DType.F32 if x.dtype == DType.F16 else x.dtype
        result = Value(ValueId.next(), out_dtype, (rows, 1))
        self._current_ir.add_op(OpKind.RowSum, [x], result)
        return result

    def rowmax(self, x: Value) -> Value:
        """Row-wise max reduction."""
        self._check_tracing()
        rows = x.shape[0] if x.shape else 1
        out_dtype = DType.F32 if x.dtype == DType.F16 else x.dtype
        result = Value(ValueId.next(), out_dtype, (rows, 1))
        self._current_ir.add_op(OpKind.RowMax, [x], result)
        return result

    def rowmean(self, x: Value) -> Value:
        """Row-wise mean reduction."""
        self._check_tracing()
        rows = x.shape[0] if x.shape else 1
        out_dtype = DType.F32 if x.dtype == DType.F16 else x.dtype
        result = Value(ValueId.next(), out_dtype, (rows, 1))
        self._current_ir.add_op(OpKind.RowMean, [x], result)
        return result

    def colsum(self, x: Value) -> Value:
        """Column-wise sum reduction."""
        self._check_tracing()
        cols = x.shape[1] if x.shape else 1
        out_dtype = DType.F32 if x.dtype == DType.F16 else x.dtype
        result = Value(ValueId.next(), out_dtype, (1, cols))
        self._current_ir.add_op(OpKind.ColSum, [x], result)
        return result

    def colmax(self, x: Value) -> Value:
        """Column-wise max reduction."""
        self._check_tracing()
        cols = x.shape[1] if x.shape else 1
        out_dtype = DType.F32 if x.dtype == DType.F16 else x.dtype
        result = Value(ValueId.next(), out_dtype, (1, cols))
        self._current_ir.add_op(OpKind.ColMax, [x], result)
        return result

    # === Broadcast Operations ===

    def rowmul(self, tile: Value, vec: Value) -> Value:
        """Multiply each row by vector: out[i,j] = tile[i,j] * vec[i]."""
        self._check_tracing()
        result = Value(ValueId.next(), tile.dtype, tile.shape)
        self._current_ir.add_op(OpKind.RowBroadcastMul, [tile, vec], result)
        return result

    def rowadd(self, tile: Value, vec: Value) -> Value:
        """Add vector to each row: out[i,j] = tile[i,j] + vec[i]."""
        self._check_tracing()
        result = Value(ValueId.next(), tile.dtype, tile.shape)
        self._current_ir.add_op(OpKind.RowBroadcastAdd, [tile, vec], result)
        return result

    def colmul(self, tile: Value, vec: Value) -> Value:
        """Multiply each column by vector: out[i,j] = tile[i,j] * vec[j]."""
        self._check_tracing()
        result = Value(ValueId.next(), tile.dtype, tile.shape)
        self._current_ir.add_op(OpKind.ColBroadcastMul, [tile, vec], result)
        return result

    def coladd(self, tile: Value, vec: Value) -> Value:
        """Add vector to each column: out[i,j] = tile[i,j] + vec[j]."""
        self._check_tracing()
        result = Value(ValueId.next(), tile.dtype, tile.shape)
        self._current_ir.add_op(OpKind.ColBroadcastAdd, [tile, vec], result)
        return result

    # === Matrix Operations ===

    def matmul(self, a: Value, b: Value, acc: Optional[Value] = None) -> Value:
        """Matrix multiplication.

        Example:
            c = tl.matmul(a, b)  # c = a @ b
            c = tl.matmul(a, b, acc=c)  # c += a @ b
        """
        self._check_tracing()
        # Output shape: (a.rows, b.cols)
        rows = a.shape[0] if a.shape else 32
        cols = b.shape[1] if b.shape else 32
        # DType rules (v9 direction):
        # - f16 @ f16 accumulates to f32
        # - f32 @ f32 stays f32
        # - other dtypes are not yet supported by the Python frontend
        if a.dtype == DType.F16 and b.dtype == DType.F16:
            out_dtype = DType.F32
        else:
            out_dtype = a.dtype

        if acc is not None and acc.dtype != out_dtype:
            raise TypeError(f"matmul acc dtype mismatch: acc={acc.dtype} out={out_dtype}")

        result = Value(ValueId.next(), out_dtype, (rows, cols))
        operands = [a, b] if acc is None else [a, b, acc]
        self._current_ir.add_op(OpKind.Matmul, operands, result)
        return result

    # === Special Operations ===

    def constant(self, value: float, dtype: DType = DType.F32) -> Value:
        """Create a constant scalar value.

        Example:
            scale = tl.constant(1.0 / 64.0, F32)
        """
        self._check_tracing()
        result = Value.scalar(dtype, name=f"const_{value}")
        self._current_ir.add_op(OpKind.Constant, [], result, value=value)
        return result

    def sin(self, x: Value) -> Value:
        """Element-wise sine."""
        self._check_tracing()
        result = Value(ValueId.next(), x.dtype, x.shape)
        self._current_ir.add_op(OpKind.Sin, [x], result)
        return result

    def cos(self, x: Value) -> Value:
        """Element-wise cosine."""
        self._check_tracing()
        result = Value(ValueId.next(), x.dtype, x.shape)
        self._current_ir.add_op(OpKind.Cos, [x], result)
        return result

    # Aliases for compatibility
    def max(self, a: Value, b: Value) -> Value:
        """Element-wise maximum (alias for maximum)."""
        return self.maximum(a, b)

    def min(self, a: Value, b: Value) -> Value:
        """Element-wise minimum (alias for minimum)."""
        return self.minimum(a, b)

    # === Slice and Interleave Operations (for RoPE) ===

    def slice_even(self, x: Value) -> Value:
        """Extract even-indexed elements along the last dimension.

        Used in RoPE (Rotary Position Embedding) computations.

        Example:
            x_even = pto.slice_even(x)  # x[:, 0::2]
        """
        self._check_tracing()
        rows = x.shape[0] if x.shape else 32
        cols = (x.shape[1] // 2) if x.shape else 64
        result = Value(ValueId.next(), x.dtype, (rows, cols))
        self._current_ir.add_op(OpKind.Constant, [x], result, slice_type="even")
        return result

    def slice_odd(self, x: Value) -> Value:
        """Extract odd-indexed elements along the last dimension.

        Used in RoPE (Rotary Position Embedding) computations.

        Example:
            x_odd = pto.slice_odd(x)  # x[:, 1::2]
        """
        self._check_tracing()
        rows = x.shape[0] if x.shape else 32
        cols = (x.shape[1] // 2) if x.shape else 64
        result = Value(ValueId.next(), x.dtype, (rows, cols))
        self._current_ir.add_op(OpKind.Constant, [x], result, slice_type="odd")
        return result

    def interleave(self, a: Value, b: Value) -> Value:
        """Interleave two tensors along the last dimension.

        Used to reconstruct tensor after RoPE transformation.

        Example:
            result = pto.interleave(new_even, new_odd)  # Combine even/odd back
        """
        self._check_tracing()
        rows = a.shape[0] if a.shape else 32
        cols = (a.shape[1] * 2) if a.shape else 128
        result = Value(ValueId.next(), a.dtype, (rows, cols))
        self._current_ir.add_op(OpKind.Constant, [a, b], result, op_type="interleave")
        return result

    def broadcast(self, x: Value, shape: tuple) -> Value:
        """Broadcast tensor to target shape.

        Example:
            weight = pto.broadcast(weights_tile, (TILE_SIZE, HIDDEN_DIM))
        """
        self._check_tracing()
        result = Value(ValueId.next(), x.dtype, shape)
        self._current_ir.add_op(OpKind.Constant, [x], result, broadcast_shape=shape)
        return result

    # === PTO-ISA helpers (v9) ===

    def iota_u32(self, cols: int) -> Value:
        """Create a [1,cols] u32 tile with values [0..cols-1]."""
        self._check_tracing()
        c = int(cols)
        if c <= 0:
            raise ValueError("iota_u32: cols must be > 0")
        result = Value(ValueId.next(), DType.U32, (1, c))
        self._current_ir.add_op(OpKind.IotaU32, [], result, cols=c)
        return result

    def tsort32(self, scores: Value, idx: Value) -> Value:
        """PTO-ISA TSORT32 helper.

        - scores: f32 tile [1,32]
        - idx: u32 tile [1,32]
        - returns: f32 tile [1,64] (score,index pairs; index stored as f32)
        """
        self._check_tracing()
        if scores.shape != (1, 32) or scores.dtype != DType.F32:
            raise ValueError("tsort32: scores must be Tile[1,32,F32]")
        if idx.shape != (1, 32) or idx.dtype != DType.U32:
            raise ValueError("tsort32: idx must be Tile[1,32,U32]")
        result = Value(ValueId.next(), DType.F32, (1, 64))
        self._current_ir.add_op(OpKind.TSort32, [scores, idx], result)
        return result

    def extract_topk_indices(self, pairs: Value, k: int) -> Value:
        """Extract top-k indices from TSORT32 pairs into an i32 tile [1,k]."""
        self._check_tracing()
        kk = int(k)
        if pairs.shape != (1, 64) or pairs.dtype != DType.F32:
            raise ValueError("extract_topk_indices: pairs must be Tile[1,64,F32] from tsort32")
        if kk <= 0 or kk > 32:
            raise ValueError("extract_topk_indices: k must be in [1,32]")
        result = Value(ValueId.next(), DType.I32, (1, kk))
        self._current_ir.add_op(OpKind.ExtractTopkIndices, [pairs], result, k=kk)
        return result


# Global pto instance for tile language operations (PTO-style API)
# This is the recommended namespace for kernel programming
pto = _TileLanguage()

# Legacy alias: tl is DEPRECATED, use pto instead
# Kept for backward compatibility but will emit deprecation warnings in future
tl = pto


# ============================================================
# @kernel Decorator with JIT Tracing
# ============================================================

@dataclass
class CompiledKernel:
    """Compiled kernel artifact."""
    name: str
    ir: KernelIR
    target: str
    code: str  # Generated backend code
    metadata: dict[str, Any] = field(default_factory=dict)

    def __call__(self, *args, **kwargs):
        """Execute compiled kernel (simulation mode)."""
        # In real execution, this would call the backend
        raise NotImplementedError("Kernel execution requires backend runtime")


class JITKernel:
    """JIT-compiled kernel reference.

    Wraps a Python function and traces it on first call to produce IR.
    Supports Triton-style specialization based on argument types.
    """

    def __init__(self, func: Callable, **options):
        self.func = func
        self.name = func.__name__
        self.options = options
        self._ir_cache: dict[tuple, KernelIR] = {}
        self._compiled_cache: dict[tuple, CompiledKernel] = {}

        # Extract type hints for parameter types
        self._signature = {}
        if hasattr(func, '__annotations__'):
            self._signature = func.__annotations__.copy()

    def _trace(self, *args, **kwargs) -> KernelIR:
        """Trace kernel function to produce IR."""
        ValueId.reset()
        ir = KernelIR(name=self.name, params=[])

        # Set up tracing context
        tl._current_ir = ir

        try:
            # Create Value objects for each parameter
            sig = inspect.signature(self.func)
            param_values = {}

            for i, (name, param) in enumerate(sig.parameters.items()):
                # Get type annotation
                ann = self._signature.get(name)

                # Create Value based on annotation or actual arg
                if i < len(args):
                    arg = args[i]
                    if isinstance(arg, Value):
                        val = arg
                    elif isinstance(arg, TileType):
                        val = Value.tile(arg.rows, arg.cols, arg.dtype, arg.location, name)
                    else:
                        # Default: infer from annotation or use default tile
                        val = Value.tile(32, 128, DType.F16, name=name)
                else:
                    val = Value.tile(32, 128, DType.F16, name=name)

                param_values[name] = val
                ir.params.append((name, val))

            # Execute function with Value parameters
            self.func(**param_values)

        finally:
            tl._current_ir = None

        return ir

    def compile(self, target: str = "ascend", **kwargs) -> CompiledKernel:
        raise RuntimeError(
            "Kernel.compile() is not supported in v9 codegen-first mode. "
            "Use workload.compile() to build codegen artifacts from C++ IR."
        )

    def _generate_code(self, ir: KernelIR, target: str) -> str:
        """Generate backend code from IR."""
        if target in ("ascend", "ascend_npu"):
            return self._generate_ascend_code(ir)
        elif target == "cpu_sim":
            return self._generate_cpu_sim_code(ir)
        else:
            raise ValueError(f"Unknown target: {target}")

    def _generate_ascend_code(self, ir: KernelIR) -> str:
        """Generate Ascend CANN code."""
        lines = [
            "// Generated by PTO-RT JIT Kernel Compiler",
            f"// Kernel: {ir.name}",
            "",
            '#include "kernel_operator.h"',
            "",
            f"__aicore__ void {ir.name}(GM_ADDR input, GM_ADDR output) {{",
        ]

        # Emit tile declarations
        for name, val in ir.params:
            if val.shape:
                lines.append(f"    Tile<half, {val.shape[0]}, {val.shape[1]}, UB> {name};")

        lines.append("")

        # Emit operations
        op_templates = {
            OpKind.Load: "DataCopy({result}, {src}[0], 1);",
            OpKind.Store: "DataCopy({dst}[0], {src}, 1);",
            OpKind.Add: "Add({result}, {a}, {b});",
            OpKind.Sub: "Sub({result}, {a}, {b});",
            OpKind.Mul: "Mul({result}, {a}, {b});",
            OpKind.Div: "Div({result}, {a}, {b});",
            OpKind.Exp: "Exp({result}, {src});",
            OpKind.Sqrt: "Sqrt({result}, {src});",
            OpKind.Rsqrt: "Rsqrt({result}, {src});",
            OpKind.RowSum: "ReduceSum({result}, {src}, 1);",
            OpKind.RowMax: "ReduceMax({result}, {src}, 1);",
            OpKind.RowMean: "ReduceMean({result}, {src}, 1);",
            OpKind.Matmul: "Cube::Matmul({result}, {a}, {b});",
            OpKind.RowBroadcastMul: "BroadcastMul({result}, {tile}, {vec});",
        }

        for op in ir.ops:
            template = op_templates.get(op.kind)
            if template:
                # Build substitution dict
                subs = {}
                if op.result:
                    subs["result"] = f"v{op.result.id}"
                for i, operand in enumerate(op.operands):
                    name = ["src", "dst", "a", "b", "tile", "vec"][i] if i < 6 else f"op{i}"
                    subs[name] = f"v{operand.id}"

                # Format template
                try:
                    line = template.format(**subs)
                    lines.append(f"    {line}")
                except KeyError:
                    lines.append(f"    // {op}")
            else:
                lines.append(f"    // Unhandled: {op}")

        lines.append("}")
        lines.append("")
        lines.append(f'REGISTER_KERNEL("{ir.name}", {ir.name});')

        return "\n".join(lines)

    def _generate_cpu_sim_code(self, ir: KernelIR) -> str:
        """Generate CPU simulation code (Python)."""
        lines = [
            f"# CPU Simulation for kernel: {ir.name}",
            f"def {ir.name}_sim(**kwargs):",
            "    import numpy as np",
        ]

        for op in ir.ops:
            lines.append(f"    # {op}")

        lines.append("    pass")
        return "\n".join(lines)

    def __call__(self, *args, **kwargs):
        """Call kernel - traces and compiles if needed."""
        # In @workload context, create a task reference
        from pto_wsp.builder import get_current_builder
        builder = get_current_builder()

        if builder is not None:
            # Inside @workload - create task node
            from pto_wsp.workload import Workload
            task = Workload("task",
                          kernel=self.name,
                          params=list(args),
                          resources=list(kwargs.values()))
            builder.add_child(task)
            return task
        else:
            # Direct call - trace and return IR
            return self._trace(*args, **kwargs)

    def __getitem__(self, axes):
        """Axis binding: kernel[b, h](...)."""
        if not isinstance(axes, tuple):
            axes = (axes,)
        return _KernelCallWithAxes(self, axes)


class _KernelCallWithAxes:
    """Helper for kernel[axes](...) syntax."""

    def __init__(self, kernel: JITKernel, axes: tuple):
        self.kernel = kernel
        self.axes = axes

    def __call__(self, *args, **kwargs):
        from pto_wsp.builder import get_current_builder
        builder = get_current_builder()

        if builder is not None:
            from pto_wsp.workload import Workload
            task = Workload("task",
                          kernel=self.kernel.name,
                          params=list(self.axes),
                          resources=list(args) + list(kwargs.values()))
            builder.add_child(task)
            return task
        else:
            raise RuntimeError("Axis-bound kernel call must be inside @workload")


def jit_kernel(func: Callable = None, *, num_warps: int = 4,
               num_stages: int = 2, **kwargs) -> JITKernel:
    """JIT kernel decorator (Triton-style).

    Example:
        @jit_kernel(num_warps=4)
        def matmul_kernel(a: In[Tile[M, K, F16]],
                         b: In[Tile[K, N, F16]],
                         c: Out[Tile[M, N, F16]]):
            c_val = tl.matmul(a, b)
            tl.store(c, c_val)
    """
    options = {"num_warps": num_warps, "num_stages": num_stages, **kwargs}

    def decorator(fn: Callable) -> JITKernel:
        return JITKernel(fn, **options)

    if func is not None:
        return decorator(func)
    return decorator


# Alias for simpler usage
kernel = jit_kernel
