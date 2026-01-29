"""
Workload builder infrastructure for PTO Workload-Schedule Programming (PTO-WSP) framework.

Provides the stack-based builder pattern for @workload decorator
and the unified @kernel decorator with JIT support.
"""

from __future__ import annotations
from typing import Any, Callable, Optional, TypeVar, Union, get_type_hints
from dataclasses import dataclass, field
from contextvars import ContextVar
import functools
import inspect

# Context variable for the current builder stack
_builder_stack: ContextVar[list["WorkloadBuilder"]] = ContextVar("builder_stack", default=[])


@dataclass
class LoopFrame:
    """A single loop frame in the builder stack."""
    kind: str  # "parallel", "sequential", "select", "conditional"
    axis: Any
    var_name: str
    children: list[Any] = field(default_factory=list)


@dataclass
class ConditionalFrame:
    """Conditional frame for P.when()."""
    predicate: Any
    then_body: list[Any] = field(default_factory=list)
    else_body: Optional[list[Any]] = None


@dataclass
class KernelParam:
    """Structured kernel parameter with direction and type info.

    This dataclass captures complete type information for kernel parameters,
    enabling type checking and code generation.

    Attributes:
        name: Parameter name in the kernel signature
        direction: "in", "out", or "inout"
        inner_type: The underlying type (TileType, ScalarType, etc.)
        dtype: Data type (F16, F32, etc.)
        shape: Shape tuple for tiles, None for scalars

    Example:
        # For a parameter: x: In[Tile[32, 128, F16]]
        KernelParam(
            name="x",
            direction="in",
            inner_type=TileType(32, 128, DType.F16),
            dtype=DType.F16,
            shape=(32, 128)
        )
    """
    name: str
    direction: str  # "in", "out", "inout", "constexpr"
    inner_type: Any  # TileType, ScalarType, or actual type
    dtype: Any = None  # DType if known
    shape: Optional[tuple] = None  # (rows, cols) for tiles


def extract_kernel_params(func: Callable) -> list[KernelParam]:
    """Extract structured kernel parameters from function type hints.

    Parses type hints like In[Tile[32, 128, F16]] into KernelParam objects.

    Args:
        func: Kernel function with type annotations

    Returns:
        List of KernelParam objects describing each parameter
    """
    from pto_wsp.types import DType
    from pto_wsp.kernel import TileType, ScalarType

    params = []
    hints = get_type_hints(func) if hasattr(func, '__annotations__') else {}
    sig = inspect.signature(func)

    for param_name in sig.parameters:
        if param_name == 'return':
            continue

        hint = hints.get(param_name)
        if hint is None:
            # No type hint - create basic param
            params.append(KernelParam(name=param_name, direction="in", inner_type=None))
            continue

        # Check for direction markers (In, Out, InOut, Constexpr)
        direction = "in"
        inner_type = hint

        # Check class attributes from _DirectionMeta
        if hasattr(hint, '_direction'):
            direction = hint._direction
        if hasattr(hint, '_inner_type'):
            inner_type = hint._inner_type

        # Extract dtype and shape from inner_type
        dtype = None
        shape = None

        if isinstance(inner_type, TileType):
            dtype = inner_type.dtype
            shape = (inner_type.rows, inner_type.cols)
        elif isinstance(inner_type, ScalarType):
            dtype = inner_type.dtype

        params.append(KernelParam(
            name=param_name,
            direction=direction,
            inner_type=inner_type,
            dtype=dtype,
            shape=shape
        ))

    return params


class WorkloadBuilder:
    """Stack-based workload builder.

    This builder maintains a stack of loop frames that gets populated
    as the workload function body executes with P() loops.
    """

    def __init__(self, name: str, type_check: bool = True):
        self.name = name
        self.frames: list[LoopFrame | ConditionalFrame] = []
        self.root_children: list[Any] = []
        self._pending_cond_workload: Any | None = None
        self._var_counter = 0
        # Type checking integration
        self._type_check = type_check
        self._type_errors: list[Any] = []
        self._type_checker = None
        if type_check:
            from pto_wsp.type_checker import TypeChecker, TypeCheckContext
            self._type_checker = TypeChecker(TypeCheckContext(
                fail_fast=False,
                workload_name=name
            ))

    def push_frame(self, frame: LoopFrame | ConditionalFrame) -> None:
        """Push a new frame onto the stack."""
        self.frames.append(frame)

    def pop_frame(self) -> LoopFrame | ConditionalFrame:
        """Pop and return the top frame."""
        return self.frames.pop()

    def current_frame(self) -> Optional[LoopFrame | ConditionalFrame]:
        """Get the current (top) frame."""
        return self.frames[-1] if self.frames else None

    def add_child(self, node: Any) -> None:
        """Add a child node to the current frame or root."""
        if self.frames:
            frame = self.frames[-1]
            if isinstance(frame, LoopFrame):
                frame.children.append(node)
            elif isinstance(frame, ConditionalFrame):
                if frame.else_body is not None:
                    frame.else_body.append(node)
                else:
                    frame.then_body.append(node)
        else:
            self.root_children.append(node)

    def generate_var_name(self, prefix: str = "i") -> str:
        """Generate a unique variable name."""
        self._var_counter += 1
        return f"{prefix}_{self._var_counter}"

    def check_kernel_call(self, kernel_ref: "Kernel", axes: tuple, args: dict) -> None:
        """Type check a kernel call if type checking is enabled."""
        if self._type_checker:
            self._type_checker.ctx.current_location = f"{self.name}.{kernel_ref.name}"
            self._type_checker.check_kernel_call(kernel_ref, axes, args)

    def check_axis_bounds(self, axis: Any, index: Any) -> None:
        """Type check axis bounds if type checking is enabled."""
        if self._type_checker:
            self._type_checker.check_axis_bounds(axis, index)

    def check_tensor_access(self, tensor: Any, indices: tuple) -> None:
        """Type check tensor access if type checking is enabled."""
        if self._type_checker:
            from pto_wsp.types import Tensor
            if isinstance(tensor, Tensor):
                self._type_checker.check_tensor_access(tensor, indices)

    def get_type_errors(self) -> list:
        """Get collected type errors."""
        if self._type_checker:
            return self._type_checker.get_errors()
        return []

    def has_type_errors(self) -> bool:
        """Check if any type errors were collected."""
        return len(self.get_type_errors()) > 0

    def build(self) -> "WorkloadIR":
        """Build the final workload IR from collected frames."""
        from pto_wsp.workload import Workload

        if len(self.root_children) == 0:
            return Workload("empty")
        elif len(self.root_children) == 1:
            return self.root_children[0]
        else:
            return Workload("combine", workloads=self.root_children)


def get_current_builder() -> Optional[WorkloadBuilder]:
    """Get the current workload builder from context."""
    stack = _builder_stack.get()
    return stack[-1] if stack else None


def push_builder(builder: WorkloadBuilder) -> None:
    """Push a builder onto the context stack."""
    stack = _builder_stack.get()
    _builder_stack.set(stack + [builder])


def pop_builder() -> Optional[WorkloadBuilder]:
    """Pop and return the top builder from the context stack."""
    stack = _builder_stack.get()
    if stack:
        builder = stack[-1]
        _builder_stack.set(stack[:-1])
        return builder
    return None


# =========== @workload decorator ===========

def workload_decorator(
    func: Callable[..., None] = None,
    *,
    type_check: bool = True,
    fail_on_type_error: bool = False
) -> Callable[..., Any]:
    """Decorator for defining workloads using @workload + P() syntax.

    The decorated function uses P() and kernel calls to define a workload.

    Args:
        func: The workload function to decorate
        type_check: Whether to perform type checking (default: True)
        fail_on_type_error: Whether to raise TypeError on type errors (default: False)

    Example:
        @workload
        def attention(batch, heads):
            for b, h in P(batch, heads):
                attn[b, h](Q=Q[b,h], K=K[b], V=V[b], O=O[b,h])

        @workload(type_check=True, fail_on_type_error=True)
        def strict_attention(batch, heads):
            ...

    Returns:
        A callable that returns a Workload when invoked with axis arguments.
    """
    def decorator(fn: Callable[..., None]) -> Callable[..., Any]:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs) -> Any:
            from pto_wsp.workload import Workload

            # Create new builder and push to stack
            builder = WorkloadBuilder(fn.__name__, type_check=type_check)
            push_builder(builder)

            try:
                # Execute the function body - P() calls will populate the builder
                fn(*args, **kwargs)

                # Check for type errors
                if builder.has_type_errors():
                    errors = builder.get_type_errors()
                    if fail_on_type_error:
                        error_messages = "\n".join(str(e) for e in errors)
                        raise TypeError(f"Type errors in workload '{fn.__name__}':\n{error_messages}")
                    else:
                        # Store errors on workload for later inspection
                        result = builder.build()
                        result._type_errors = errors
                        return result

                # Build the workload from collected structure
                result = builder.build()
                result._type_errors = []  # No errors
                return result
            finally:
                # Always pop the builder
                pop_builder()

        # Attach metadata
        wrapper._workload_name = fn.__name__
        wrapper._type_check = type_check
        return wrapper

    # Handle both @workload and @workload() syntax
    if func is not None:
        return decorator(func)
    return decorator


# =========== Unified @kernel decorator with JIT support ===========

# Import JIT components from kernel.py
from pto_wsp.kernel import (
    _TileLanguage,
    KernelIR,
    CompiledKernel,
    Value,
    ValueId,
    Op,
    OpKind,
    TileType,
    ScalarType,
    tl,  # Use the global tl instance from kernel.py
)


class Kernel:
    """Unified kernel with JIT tracing support.

    This class provides a single @kernel decorator that:
    1. Traces function body with tl.* primitives to produce KernelIR
    2. Can be used in workloads with axis binding: kernel[b, m, n](...)
    3. Compiles to target backend code (Ascend/CPU via pto-isa)

    CPU simulation is handled by pto-isa's built-in CPU simulation backend
    (compiled with -D__CPU_SIM flag), not by separate Python implementations.

    Example:
        @kernel
        def gemm_tile(a: In[Tile[32, 32, F16]], b: In[Tile[32, 32, F16]], c: Out[Tile[32, 32, F16]]):
            result = tl.matmul(a, b)
            tl.store(c, result)

        # Use in workload
        @workload
        def bgemm():
            for b, m, n in P(batch, tile_m, tile_n):
                gemm_tile[b, m, n](a=A[b], b=B[b], c=C[b])

        # Compile and execute - CPU simulation uses pto-isa backend
        program = bgemm().compile()
        program.execute()
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

        # Structured parameter info (direction + dtype/shape) for codegen.
        self._kernel_params: list[KernelParam] = extract_kernel_params(func)

        functools.update_wrapper(self, func)

    @property
    def signature(self) -> dict:
        """Get the kernel parameter signature (for backward compatibility)."""
        return self._signature

    def _trace(self, *args, **kwargs) -> KernelIR:
        """Trace kernel function to produce IR."""
        ValueId.reset()
        ir = KernelIR(name=self.name, params=[])

        # Set up tracing context - use the global tl from kernel.py
        from pto_wsp.kernel import tl as kernel_tl
        kernel_tl._current_ir = ir

        try:
            # Create Value objects for each parameter
            sig = inspect.signature(self.func)
            param_values = {}

            params_by_name = {p.name: p for p in self._kernel_params}

            for name in sig.parameters:
                meta = params_by_name.get(name)
                if meta is None or meta.inner_type is None:
                    val = Value.tile(32, 128, DType.F16, name=name)
                else:
                    if isinstance(meta.inner_type, TileType) and meta.shape is not None:
                        rows, cols = meta.shape
                        val = Value.tile(rows, cols, meta.dtype or DType.F16, meta.inner_type.location, name)
                    elif isinstance(meta.inner_type, ScalarType):
                        val = Value.scalar(meta.dtype or DType.F32, name=name)
                    else:
                        # Unknown inner type (e.g., plain Python scalar) - represent as scalar.
                        val = Value.scalar(DType.F32, name=name)

                param_values[name] = val
                ir.params.append((name, val))

            # Execute function with Value parameters unless this kernel provides
            # an explicit C++ implementation.
            if self.options.get("cpp_src") is None and self.options.get("cpp_file_src") is None:
                self.func(**param_values)

        finally:
            kernel_tl._current_ir = None

        return ir

    def trace(self, **kwargs) -> KernelIR:
        """Trace kernel to get IR (public API)."""
        return self._trace(**kwargs)

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
            "// Generated by PTO-RT Kernel Compiler",
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

    def __call__(self, *args, **kwargs) -> Any:
        """Call kernel - either create task in workload context or trace."""
        builder = get_current_builder()

        if builder is not None:
            # Inside @workload - create task node
            from pto_wsp.workload import Workload

            # Type check the kernel call if enabled
            if builder._type_checker:
                builder.check_kernel_call(self, (), kwargs)

            # Preserve argument-to-parameter mapping for codegen.
            sig = inspect.signature(self.func)
            param_names = [n for n in sig.parameters.keys() if n != "return"]
            resources: dict[str, Any] = {}
            for i, arg in enumerate(args):
                if i < len(param_names):
                    resources[param_names[i]] = arg
            resources.update(kwargs)

            task = Workload("task",
                          kernel=self,
                          params=[],
                          resources=resources)
            builder.add_child(task)
            return task
        else:
            # Direct call - trace and return IR
            return self._trace(*args, **kwargs)

    def __getitem__(self, axes) -> "_KernelCallWithAxes":
        """Axis binding: kernel[b, h](...)."""
        if not isinstance(axes, tuple):
            axes = (axes,)
        return _KernelCallWithAxes(self, axes)


class _KernelCallWithAxes:
    """Helper for kernel[axes](...) syntax."""

    def __init__(self, kernel: Kernel, axes: tuple):
        self.kernel = kernel
        self.kernel_ref = kernel  # Alias for backward compatibility
        self.axes = axes

    def __call__(self, *args, **kwargs) -> Any:
        """Complete the kernel call with arguments."""
        builder = get_current_builder()

        if builder is not None:
            from pto_wsp.workload import Workload

            # Type check the kernel call if enabled
            if builder._type_checker:
                builder.check_kernel_call(self.kernel, self.axes, kwargs)

            sig = inspect.signature(self.kernel.func)
            param_names = [n for n in sig.parameters.keys() if n != "return"]
            resources: dict[str, Any] = {}
            for i, arg in enumerate(args):
                if i < len(param_names):
                    resources[param_names[i]] = arg
            resources.update(kwargs)

            task = Workload("task",
                          kernel=self.kernel,
                          params=list(self.axes),
                          resources=resources)
            builder.add_child(task)
            return task
        else:
            raise RuntimeError("Axis-bound kernel call must be inside @workload")


def kernel(func: Callable = None, *, num_warps: int = 4,
           num_stages: int = 2,
           cpp_src: str | None = None,
           cpp_body_path: str | None = None,
           cpp_tu_path: str | None = None,
           cpp_includes: Optional[list[str]] = None,
           **kwargs) -> Kernel:
    """Unified kernel decorator with JIT support.

    This decorator creates kernels that:
    1. Can be traced to produce IR with tl.* primitives
    2. Can be used directly in workloads with axis binding
    3. Can be compiled to target backends
    4. Support CPU implementation registration for simulation

    Example:
        @kernel
        def gemm_tile(a: In[Tile[M, K, F16]], b: In[Tile[K, N, F16]], c: Out[Tile[M, N, F16]]):
            result = tl.matmul(a, b)
            tl.store(c, result)

        # Register CPU implementation
        @gemm_tile.register_cpu
        def gemm_cpu(a, b, c):
            c[:] = np.matmul(a, b)

        # Or directly
        gemm_tile.cpu_impl = lambda a, b, c: np.matmul(a, b, out=c)

        # Use in workload
        @workload
        def bgemm():
            for b in P(batch):
                gemm_tile[b](a=A[b], b=B[b], c=C[b])
    """
    import os

    options = {"num_warps": num_warps, "num_stages": num_stages, **kwargs}

    if cpp_src is not None and (cpp_body_path is not None or cpp_tu_path is not None):
        raise ValueError("kernel(): use only one of cpp_src / cpp_body_path / cpp_tu_path")

    if cpp_body_path is not None and cpp_tu_path is not None:
        raise ValueError("kernel(): use only one of cpp_body_path / cpp_tu_path")

    if cpp_body_path is not None:
        p = os.path.abspath(os.path.expanduser(str(cpp_body_path)))
        with open(p, "r", encoding="utf-8") as f:
            options["cpp_src"] = f.read()
        options["cpp_src_path"] = p
    elif cpp_tu_path is not None:
        p = os.path.abspath(os.path.expanduser(str(cpp_tu_path)))
        with open(p, "r", encoding="utf-8") as f:
            options["cpp_file_src"] = f.read()
        options["cpp_file_path"] = p
    elif cpp_src is not None:
        options["cpp_src"] = str(cpp_src)

    if cpp_includes is not None:
        options["cpp_includes"] = [str(x) for x in cpp_includes]

    def decorator(fn: Callable) -> Kernel:
        return Kernel(fn, **options)

    if func is not None:
        return decorator(func)
    return decorator


# Backward compatibility: KernelRef is now Kernel
KernelRef = Kernel


# =========== Type annotations for kernel signatures ===========

T = TypeVar("T")


class _DirectionMeta(type):
    """Metaclass that enables subscript syntax and stores type info."""

    def __getitem__(cls, item):
        """Support Foo[Type] syntax, returning a special annotated type."""
        # Create a new class that remembers both the direction and inner type
        class_name = f"{cls.__name__}[{item}]"

        # Create a new class that carries the type information
        new_cls = type(class_name, (cls,), {
            '_inner_type': item,
            '_direction': cls.__name__.lower(),
        })
        return new_cls


class In(metaclass=_DirectionMeta):
    """Input tensor annotation for kernel parameters.

    Marks a parameter as read-only input.

    Usage:
        @kernel
        def my_kernel(x: In[Tile[32, 32, F32]]): ...

    The type can be introspected:
        hint = In[Tile[32, 32, F32]]
        hint._inner_type  # Tile[32, 32, F32]
        hint._direction   # "in"
    """
    _inner_type = None
    _direction = "in"


class Out(metaclass=_DirectionMeta):
    """Output tensor annotation for kernel parameters.

    Marks a parameter as write-only output.

    Usage:
        @kernel
        def my_kernel(y: Out[Tile[32, 32, F32]]): ...
    """
    _inner_type = None
    _direction = "out"


class InOut(metaclass=_DirectionMeta):
    """In-place tensor annotation for kernel parameters.

    Marks a parameter as read-write (in-place modification).

    Usage:
        @kernel
        def my_kernel(x: InOut[Tile[32, 32, F32]]): ...
    """
    _inner_type = None
    _direction = "inout"


class Constexpr(metaclass=_DirectionMeta):
    """Compile-time constant annotation for kernel parameters.

    Marks a parameter as a compile-time constant value.

    Usage:
        @kernel
        def my_kernel(tile_size: Constexpr[int]): ...
    """
    _inner_type = None
    _direction = "constexpr"


def get_direction(hint) -> str:
    """Extract direction from a type hint.

    Args:
        hint: Type hint like In[Tile[...]] or Out[Tile[...]]

    Returns:
        Direction string: "in", "out", "inout", "constexpr", or "unknown"
    """
    if hasattr(hint, '_direction'):
        return hint._direction
    # Handle string annotations for backward compatibility
    if isinstance(hint, str):
        if hint.startswith("In["):
            return "in"
        elif hint.startswith("Out["):
            return "out"
        elif hint.startswith("InOut["):
            return "inout"
        elif hint.startswith("Constexpr["):
            return "constexpr"
    return "unknown"


def get_inner_type(hint):
    """Extract inner type from a direction-annotated type hint.

    Args:
        hint: Type hint like In[Tile[...]]

    Returns:
        Inner type (e.g., Tile[...]) or None
    """
    if hasattr(hint, '_inner_type'):
        return hint._inner_type
    return None


# Import DType for type inference in tracing
from pto_wsp.types import DType
