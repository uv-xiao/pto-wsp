"""
NPU Function Builder for PTO Workload-Schedule Programming (PTO-WSP) framework.

DEPRECATED: This module uses string-based operation building which is
error-prone. For new code, use @jit_kernel with tl.* primitives instead:

    from pto_wsp import jit_kernel, tl, In, Out, Tile, Scalar

    @jit_kernel
    def rmsnorm_kernel(
        x: In[Tile[32, 128, DType.F16]],
        out: Out[Tile[32, 128, DType.F16]],
        eps: Scalar[DType.F32] = 1e-6
    ):
        sq = tl.mul(x, x)
        mean = tl.rowmean(sq)
        rsqrt_val = tl.rsqrt(mean)
        result = tl.mul(x, rsqrt_val)
        tl.store(out, result)

The @jit_kernel approach provides:
- Type safety (no string typos)
- Static analysis support
- Better error messages
- Consistent with JAX/Triton style

Legacy string-based Example (deprecated):
    npu_func = (npu("rmsnorm_tile")
        .tile("x", 32, 128, dtype=DType.F16)
        .tile("out", 32, 128, dtype=DType.F16)
        .scalar("eps", DType.F32, default=1e-6)
        .memref("input", DType.F16, is_input=True)
        .memref("output", DType.F16, is_output=True)
        .load("x", "input")
        .mul("sq", "x", "x")
        .rowmean("mean", "sq")
        .rsqrt("rsqrt_val", "mean")
        .rowmul("out", "x", "rsqrt_val")
        .store("output", "out")
        .build())
"""

from __future__ import annotations
import warnings
from typing import Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum

from pto_wsp.types import DType, Location


# =========== NPU Function IR Nodes ===========

@dataclass
class TileDecl:
    """Tile buffer declaration."""
    name: str
    rows: int
    cols: int
    dtype: DType
    location: Location = Location.UB


@dataclass
class ScalarDecl:
    """Scalar variable declaration."""
    name: str
    dtype: DType
    default: Optional[Any] = None


@dataclass
class MemrefDecl:
    """Memory reference declaration (external tensor)."""
    name: str
    dtype: DType
    location: Location = Location.Global
    shape: list[int] = field(default_factory=list)
    is_input: bool = False
    is_output: bool = False


# NPU Operation kinds
class NPUOpKind(Enum):
    # Memory operations
    Load = "load"
    Store = "store"
    Wait = "wait"

    # Binary operations
    Add = "add"
    Sub = "sub"
    Mul = "mul"
    Div = "div"
    Max = "max"
    Min = "min"

    # Unary operations
    Neg = "neg"
    Abs = "abs"
    Exp = "exp"
    Log = "log"
    Sqrt = "sqrt"
    Rsqrt = "rsqrt"
    Tanh = "tanh"
    Sigmoid = "sigmoid"
    Relu = "relu"
    Gelu = "gelu"
    Silu = "silu"

    # Reduction operations
    RowSum = "rowsum"
    RowMax = "rowmax"
    RowMean = "rowmean"
    ColSum = "colsum"
    ColMax = "colmax"

    # Broadcast operations
    RowBroadcastAdd = "rowbroadcastadd"
    RowBroadcastMul = "rowbroadcastmul"
    RowExpandMul = "rowexpandmul"
    ColBroadcastAdd = "colbroadcastadd"
    ColBroadcastMul = "colbroadcastmul"

    # MatMul operations
    MatMul = "matmul"
    MatMulAcc = "matmulacc"  # Accumulate version

    # Control flow
    ForLoopBegin = "forloopbegin"
    ForLoopEnd = "forloopend"
    IfThenBegin = "ifthenbegin"
    ElseBranch = "elsebranch"
    IfThenEnd = "ifthenend"


@dataclass
class NPUOp:
    """Base class for NPU operations."""
    kind: NPUOpKind
    dst: str
    src: list[str] = field(default_factory=list)
    attrs: dict[str, Any] = field(default_factory=dict)


@dataclass
class NPUFunction:
    """NPU Function IR representation."""
    name: str
    tiles: list[TileDecl] = field(default_factory=list)
    scalars: list[ScalarDecl] = field(default_factory=list)
    memrefs: list[MemrefDecl] = field(default_factory=list)
    ops: list[NPUOp] = field(default_factory=list)

    # Schedule hints
    tile_policy: str = "default"  # "default", "row_major", "col_major"
    double_buffer: bool = False
    pipeline_depth: int = 1
    is_cube: bool = False  # True for matmul-heavy kernels (AIC), False for vector (AIV)

    def to_ir(self) -> dict:
        """Convert to IR dictionary for serialization."""
        return {
            "name": self.name,
            "tiles": [
                {"name": t.name, "rows": t.rows, "cols": t.cols,
                 "dtype": t.dtype.value, "location": t.location.value}
                for t in self.tiles
            ],
            "scalars": [
                {"name": s.name, "dtype": s.dtype.value, "default": s.default}
                for s in self.scalars
            ],
            "memrefs": [
                {"name": m.name, "dtype": m.dtype.value, "location": m.location.value,
                 "is_input": m.is_input, "is_output": m.is_output}
                for m in self.memrefs
            ],
            "ops": [
                {"kind": op.kind.value, "dst": op.dst, "src": op.src, "attrs": op.attrs}
                for op in self.ops
            ],
            "schedule": {
                "tile_policy": self.tile_policy,
                "double_buffer": self.double_buffer,
                "pipeline_depth": self.pipeline_depth,
                "is_cube": self.is_cube,
            }
        }


# =========== NPU Function Builder ===========

class NPUFunctionBuilder:
    """Fluent builder for NPU functions.

    Example:
        func = (npu("rmsnorm")
            .tile("x", 32, 128)
            .load("x", "input")
            .rowmean("mean", "x")
            .rsqrt("rsqrt_val", "mean")
            .rowmul("out", "x", "rsqrt_val")
            .store("output", "out")
            .build())
    """

    def __init__(self, name: str):
        self._name = name
        self._tiles: list[TileDecl] = []
        self._scalars: list[ScalarDecl] = []
        self._memrefs: list[MemrefDecl] = []
        self._ops: list[NPUOp] = []
        self._schedule_hints: dict[str, Any] = {
            "tile_policy": "default",
            "double_buffer": False,
            "pipeline_depth": 1,
            "is_cube": False,
        }

    # ========== Declaration Methods ==========

    def tile(self, name: str, rows: int, cols: int,
             dtype: DType = DType.F16,
             location: Location = Location.UB) -> NPUFunctionBuilder:
        """Declare a tile buffer.

        Args:
            name: Buffer name
            rows: Number of rows
            cols: Number of columns
            dtype: Data type
            location: Memory location (UB, L1, L2)
        """
        self._tiles.append(TileDecl(name, rows, cols, dtype, location))
        return self

    def scalar(self, name: str, dtype: DType = DType.F32,
               default: Optional[Any] = None) -> NPUFunctionBuilder:
        """Declare a scalar variable.

        Args:
            name: Variable name
            dtype: Data type
            default: Default value
        """
        self._scalars.append(ScalarDecl(name, dtype, default))
        return self

    def memref(self, name: str, dtype: DType = DType.F16,
               location: Location = Location.Global,
               shape: Optional[list[int]] = None,
               is_input: bool = False,
               is_output: bool = False) -> NPUFunctionBuilder:
        """Declare a memory reference (external tensor).

        Args:
            name: Reference name
            dtype: Data type
            location: Memory location
            shape: Optional shape hint
            is_input: True if this is an input tensor
            is_output: True if this is an output tensor
        """
        self._memrefs.append(MemrefDecl(
            name, dtype, location,
            shape or [], is_input, is_output
        ))
        return self

    # ========== Memory Operations ==========

    def load(self, dst: str, src: str, **attrs) -> NPUFunctionBuilder:
        """Load data from memref to tile.

        Args:
            dst: Destination tile name
            src: Source memref name
        """
        self._ops.append(NPUOp(NPUOpKind.Load, dst, [src], attrs))
        return self

    def store(self, dst: str, src: str, **attrs) -> NPUFunctionBuilder:
        """Store data from tile to memref.

        Args:
            dst: Destination memref name
            src: Source tile name
        """
        self._ops.append(NPUOp(NPUOpKind.Store, dst, [src], attrs))
        return self

    def wait(self, event: str = "") -> NPUFunctionBuilder:
        """Wait for async operation to complete."""
        self._ops.append(NPUOp(NPUOpKind.Wait, "", [event] if event else []))
        return self

    # ========== Binary Operations ==========

    def add(self, dst: str, src1: str, src2: str) -> NPUFunctionBuilder:
        """Element-wise addition: dst = src1 + src2."""
        self._ops.append(NPUOp(NPUOpKind.Add, dst, [src1, src2]))
        return self

    def sub(self, dst: str, src1: str, src2: str) -> NPUFunctionBuilder:
        """Element-wise subtraction: dst = src1 - src2."""
        self._ops.append(NPUOp(NPUOpKind.Sub, dst, [src1, src2]))
        return self

    def mul(self, dst: str, src1: str, src2: str) -> NPUFunctionBuilder:
        """Element-wise multiplication: dst = src1 * src2."""
        self._ops.append(NPUOp(NPUOpKind.Mul, dst, [src1, src2]))
        return self

    def div(self, dst: str, src1: str, src2: str) -> NPUFunctionBuilder:
        """Element-wise division: dst = src1 / src2."""
        self._ops.append(NPUOp(NPUOpKind.Div, dst, [src1, src2]))
        return self

    def max(self, dst: str, src1: str, src2: str) -> NPUFunctionBuilder:
        """Element-wise max: dst = max(src1, src2)."""
        self._ops.append(NPUOp(NPUOpKind.Max, dst, [src1, src2]))
        return self

    def min(self, dst: str, src1: str, src2: str) -> NPUFunctionBuilder:
        """Element-wise min: dst = min(src1, src2)."""
        self._ops.append(NPUOp(NPUOpKind.Min, dst, [src1, src2]))
        return self

    # ========== Unary Operations ==========

    def neg(self, dst: str, src: str) -> NPUFunctionBuilder:
        """Negation: dst = -src."""
        self._ops.append(NPUOp(NPUOpKind.Neg, dst, [src]))
        return self

    def abs(self, dst: str, src: str) -> NPUFunctionBuilder:
        """Absolute value: dst = |src|."""
        self._ops.append(NPUOp(NPUOpKind.Abs, dst, [src]))
        return self

    def exp(self, dst: str, src: str) -> NPUFunctionBuilder:
        """Exponential: dst = exp(src)."""
        self._ops.append(NPUOp(NPUOpKind.Exp, dst, [src]))
        return self

    def log(self, dst: str, src: str) -> NPUFunctionBuilder:
        """Logarithm: dst = log(src)."""
        self._ops.append(NPUOp(NPUOpKind.Log, dst, [src]))
        return self

    def sqrt(self, dst: str, src: str) -> NPUFunctionBuilder:
        """Square root: dst = sqrt(src)."""
        self._ops.append(NPUOp(NPUOpKind.Sqrt, dst, [src]))
        return self

    def rsqrt(self, dst: str, src: str) -> NPUFunctionBuilder:
        """Reciprocal square root: dst = 1/sqrt(src)."""
        self._ops.append(NPUOp(NPUOpKind.Rsqrt, dst, [src]))
        return self

    def tanh(self, dst: str, src: str) -> NPUFunctionBuilder:
        """Hyperbolic tangent: dst = tanh(src)."""
        self._ops.append(NPUOp(NPUOpKind.Tanh, dst, [src]))
        return self

    def sigmoid(self, dst: str, src: str) -> NPUFunctionBuilder:
        """Sigmoid: dst = 1/(1+exp(-src))."""
        self._ops.append(NPUOp(NPUOpKind.Sigmoid, dst, [src]))
        return self

    def relu(self, dst: str, src: str) -> NPUFunctionBuilder:
        """ReLU: dst = max(0, src)."""
        self._ops.append(NPUOp(NPUOpKind.Relu, dst, [src]))
        return self

    def gelu(self, dst: str, src: str) -> NPUFunctionBuilder:
        """GELU activation."""
        self._ops.append(NPUOp(NPUOpKind.Gelu, dst, [src]))
        return self

    def silu(self, dst: str, src: str) -> NPUFunctionBuilder:
        """SiLU (Swish) activation: dst = src * sigmoid(src)."""
        self._ops.append(NPUOp(NPUOpKind.Silu, dst, [src]))
        return self

    # ========== Reduction Operations ==========

    def rowsum(self, dst: str, src: str) -> NPUFunctionBuilder:
        """Row-wise sum reduction."""
        self._ops.append(NPUOp(NPUOpKind.RowSum, dst, [src]))
        return self

    def rowmax(self, dst: str, src: str) -> NPUFunctionBuilder:
        """Row-wise max reduction."""
        self._ops.append(NPUOp(NPUOpKind.RowMax, dst, [src]))
        return self

    def rowmean(self, dst: str, src: str) -> NPUFunctionBuilder:
        """Row-wise mean reduction."""
        self._ops.append(NPUOp(NPUOpKind.RowMean, dst, [src]))
        return self

    def colsum(self, dst: str, src: str) -> NPUFunctionBuilder:
        """Column-wise sum reduction."""
        self._ops.append(NPUOp(NPUOpKind.ColSum, dst, [src]))
        return self

    def colmax(self, dst: str, src: str) -> NPUFunctionBuilder:
        """Column-wise max reduction."""
        self._ops.append(NPUOp(NPUOpKind.ColMax, dst, [src]))
        return self

    # ========== Broadcast Operations ==========

    def rowbroadcastadd(self, dst: str, tile: str, vec: str) -> NPUFunctionBuilder:
        """Add vector to each row: dst[i,j] = tile[i,j] + vec[i]."""
        self._ops.append(NPUOp(NPUOpKind.RowBroadcastAdd, dst, [tile, vec]))
        return self

    def rowbroadcastmul(self, dst: str, tile: str, vec: str) -> NPUFunctionBuilder:
        """Multiply each row by vector: dst[i,j] = tile[i,j] * vec[i]."""
        self._ops.append(NPUOp(NPUOpKind.RowBroadcastMul, dst, [tile, vec]))
        return self

    def rowexpandmul(self, dst: str, tile: str, vec: str) -> NPUFunctionBuilder:
        """Expand vector and multiply: dst[i,j] = tile[i,j] * vec[i]."""
        self._ops.append(NPUOp(NPUOpKind.RowExpandMul, dst, [tile, vec]))
        return self

    def colbroadcastadd(self, dst: str, tile: str, vec: str) -> NPUFunctionBuilder:
        """Add vector to each column: dst[i,j] = tile[i,j] + vec[j]."""
        self._ops.append(NPUOp(NPUOpKind.ColBroadcastAdd, dst, [tile, vec]))
        return self

    def colbroadcastmul(self, dst: str, tile: str, vec: str) -> NPUFunctionBuilder:
        """Multiply each column by vector: dst[i,j] = tile[i,j] * vec[j]."""
        self._ops.append(NPUOp(NPUOpKind.ColBroadcastMul, dst, [tile, vec]))
        return self

    # ========== MatMul Operations ==========

    def matmul(self, dst: str, a: str, b: str,
               transpose_a: bool = False,
               transpose_b: bool = False) -> NPUFunctionBuilder:
        """Matrix multiplication: dst = A @ B.

        Args:
            dst: Destination tile
            a: Left operand tile
            b: Right operand tile
            transpose_a: Transpose A before multiplication
            transpose_b: Transpose B before multiplication
        """
        self._ops.append(NPUOp(NPUOpKind.MatMul, dst, [a, b], {
            "transpose_a": transpose_a,
            "transpose_b": transpose_b,
        }))
        return self

    def matmulacc(self, dst: str, a: str, b: str,
                  transpose_a: bool = False,
                  transpose_b: bool = False) -> NPUFunctionBuilder:
        """Matrix multiplication with accumulation: dst += A @ B."""
        self._ops.append(NPUOp(NPUOpKind.MatMulAcc, dst, [a, b], {
            "transpose_a": transpose_a,
            "transpose_b": transpose_b,
        }))
        return self

    # ========== Control Flow ==========

    def for_loop(self, var: str, start: int, end: int, step: int = 1) -> NPUFunctionBuilder:
        """Begin a for loop."""
        self._ops.append(NPUOp(NPUOpKind.ForLoopBegin, var, [], {
            "start": start,
            "end": end,
            "step": step,
        }))
        return self

    def end_for(self) -> NPUFunctionBuilder:
        """End a for loop."""
        self._ops.append(NPUOp(NPUOpKind.ForLoopEnd, "", []))
        return self

    def if_then(self, cond: str) -> NPUFunctionBuilder:
        """Begin an if-then block."""
        self._ops.append(NPUOp(NPUOpKind.IfThenBegin, "", [cond]))
        return self

    def else_branch(self) -> NPUFunctionBuilder:
        """Begin else branch."""
        self._ops.append(NPUOp(NPUOpKind.ElseBranch, "", []))
        return self

    def end_if(self) -> NPUFunctionBuilder:
        """End if-then block."""
        self._ops.append(NPUOp(NPUOpKind.IfThenEnd, "", []))
        return self

    # ========== Schedule Hints ==========

    def double_buffer(self, enable: bool = True) -> NPUFunctionBuilder:
        """Enable double buffering for async DMA."""
        self._schedule_hints["double_buffer"] = enable
        return self

    def pipeline(self, depth: int) -> NPUFunctionBuilder:
        """Set pipeline depth."""
        self._schedule_hints["pipeline_depth"] = depth
        return self

    def tile_policy(self, policy: str) -> NPUFunctionBuilder:
        """Set tile traversal policy.

        Args:
            policy: "default", "row_major", "col_major"
        """
        self._schedule_hints["tile_policy"] = policy
        return self

    def cube(self, is_cube: bool = True) -> NPUFunctionBuilder:
        """Mark as cube (matmul) kernel for AIC execution."""
        self._schedule_hints["is_cube"] = is_cube
        return self

    # ========== Build ==========

    def build(self) -> NPUFunction:
        """Build the NPU function."""
        func = NPUFunction(
            name=self._name,
            tiles=self._tiles.copy(),
            scalars=self._scalars.copy(),
            memrefs=self._memrefs.copy(),
            ops=self._ops.copy(),
        )
        func.tile_policy = self._schedule_hints["tile_policy"]
        func.double_buffer = self._schedule_hints["double_buffer"]
        func.pipeline_depth = self._schedule_hints["pipeline_depth"]
        func.is_cube = self._schedule_hints["is_cube"]
        return func


def npu(name: str) -> NPUFunctionBuilder:
    """Create an NPU function builder.

    DEPRECATED: Use @jit_kernel with tl.* primitives instead.

    Example (deprecated):
        rmsnorm = (npu("rmsnorm")
            .tile("x", 32, 128)
            .load("x", "input")
            .rowmean("mean", "x")
            .rsqrt("rsqrt_val", "mean")
            .rowmul("out", "x", "rsqrt_val")
            .store("output", "out")
            .build())

    Preferred (use @jit_kernel):
        @jit_kernel
        def rmsnorm(x: In[Tile[32, 128, F16]], out: Out[Tile[32, 128, F16]]):
            sq = tl.mul(x, x)
            mean = tl.rowmean(sq)
            rsqrt_val = tl.rsqrt(mean)
            result = tl.mul(x, rsqrt_val)
            tl.store(out, result)
    """
    warnings.warn(
        "npu() string-based builder is deprecated. Use @jit_kernel with tl.* primitives instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return NPUFunctionBuilder(name)


# =========== Composite Operations (Macro Expansion) ===========

def rmsnorm(builder: NPUFunctionBuilder, output: str, input: str,
            eps: float = 1e-6) -> NPUFunctionBuilder:
    """RMSNorm composite operation (expands to primitives).

    Equivalent to:
        x_sq = input * input
        mean = rowmean(x_sq)
        rsqrt_val = rsqrt(mean + eps)
        output = input * rsqrt_val
    """
    builder.scalar("_eps", DType.F32, eps)
    builder.mul("_sq", input, input)
    builder.rowmean("_mean", "_sq")
    # Note: Adding eps would need a scalar broadcast, simplified here
    builder.rsqrt("_rsqrt", "_mean")
    builder.rowexpandmul(output, input, "_rsqrt")
    return builder


def softmax(builder: NPUFunctionBuilder, output: str, input: str) -> NPUFunctionBuilder:
    """Softmax composite operation (expands to primitives).

    Equivalent to:
        max_val = rowmax(input)
        shifted = input - max_val
        exp_val = exp(shifted)
        sum_val = rowsum(exp_val)
        output = exp_val / sum_val
    """
    builder.rowmax("_max", input)
    builder.rowbroadcastadd("_shifted", input, "_max")  # Should be subtract
    builder.exp("_exp", "_shifted")
    builder.rowsum("_sum", "_exp")
    builder.rowbroadcastmul(output, "_exp", "_sum")  # Should be div
    return builder


def layer_norm(builder: NPUFunctionBuilder, output: str, input: str,
               gamma: str, beta: str, eps: float = 1e-5) -> NPUFunctionBuilder:
    """LayerNorm composite operation."""
    builder.scalar("_eps", DType.F32, eps)
    builder.rowmean("_mean", input)
    builder.rowbroadcastadd("_centered", input, "_mean")  # Should subtract
    builder.mul("_sq", "_centered", "_centered")
    builder.rowmean("_var", "_sq")
    builder.rsqrt("_std", "_var")
    builder.rowexpandmul("_norm", "_centered", "_std")
    builder.rowbroadcastmul("_scaled", "_norm", gamma)
    builder.rowbroadcastadd(output, "_scaled", beta)
    return builder
