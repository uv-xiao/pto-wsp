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


class WorkloadBuilder:
    """Stack-based workload builder.

    This builder maintains a stack of loop frames that gets populated
    as the workload function body executes with P() loops.
    """

    def __init__(self, name: str, type_check: bool = True):
        self.name = name
        self.frames: list[LoopFrame | ConditionalFrame] = []
        self.root_children: list[Any] = []
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
            kernel_tl._current_ir = None

        return ir

    def trace(self, **kwargs) -> KernelIR:
        """Trace kernel to get IR (public API)."""
        return self._trace(**kwargs)

    def compile(self, target: str = "ascend", **kwargs) -> CompiledKernel:
        """Compile kernel for target backend."""
        # Create specialization key from kwargs
        spec_key = (target, tuple(sorted(kwargs.items())))

        if spec_key in self._compiled_cache:
            return self._compiled_cache[spec_key]

        # Trace to get IR
        ir = self._trace(**kwargs)

        # Generate code for target
        code = self._generate_code(ir, target)

        compiled = CompiledKernel(
            name=self.name,
            ir=ir,
            target=target,
            code=code,
            metadata=kwargs
        )
        self._compiled_cache[spec_key] = compiled
        return compiled

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

            task = Workload("task",
                          kernel=self.name,
                          params=[],
                          resources=list(args) + list(kwargs.values()))
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

            task = Workload("task",
                          kernel=self.kernel.name,
                          params=list(self.axes),
                          resources=list(args) + list(kwargs.values()))
            builder.add_child(task)
            return task
        else:
            raise RuntimeError("Axis-bound kernel call must be inside @workload")


def kernel(func: Callable = None, *, num_warps: int = 4,
           num_stages: int = 2, **kwargs) -> Kernel:
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
    options = {"num_warps": num_warps, "num_stages": num_stages, **kwargs}

    def decorator(fn: Callable) -> Kernel:
        return Kernel(fn, **options)

    if func is not None:
        return decorator(func)
    return decorator


# Backward compatibility: KernelRef is now Kernel
KernelRef = Kernel


# =========== Type annotations for kernel signatures ===========

T = TypeVar("T")


class In:
    """Input tensor annotation for kernel parameters."""
    def __class_getitem__(cls, item):
        return f"In[{item}]"


class Out:
    """Output tensor annotation for kernel parameters."""
    def __class_getitem__(cls, item):
        return f"Out[{item}]"


class InOut:
    """In-place tensor annotation for kernel parameters."""
    def __class_getitem__(cls, item):
        return f"InOut[{item}]"


class Constexpr:
    """Compile-time constant annotation for kernel parameters."""
    def __class_getitem__(cls, item):
        return f"Constexpr[{item}]"


# Import DType for type inference in tracing
from pto_wsp.types import DType
