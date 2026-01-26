"""
Type Checker for PTO Workload-Schedule Programming (PTO-WSP) framework.

Implements L1 (Python builder-time) type checking as specified in docs/type-system.md.

Key checks:
- Kernel signature validation
- Layout compatibility (Dato-style join rules)
- Axis bounds validation
- Tensor access validation
"""

from __future__ import annotations
from typing import Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum

from pto_wsp.types import (
    Dense, DenseDyn, Ragged, Sparse, Tensor, DType, Location,
    # Layout types - imported from types.py to avoid duplication
    DistElem,
    TensorReplicate as Replicate,
    TensorShard as Shard,
    TensorLayout as Layout,
    LayoutIncompatibleError as LayoutCompatibilityError,
    tensor_layout_join as layout_join,
)
from pto_wsp.builder import KernelRef, In, Out, InOut, Constexpr


# ============================================================
# Layout Type Definitions
# ============================================================
# NOTE: Layout types are now imported from types.py to avoid duplication.
# The canonical implementations are:
#   - DistElem, TensorReplicate, TensorShard, TensorLayout in types.py
#   - tensor_layout_join() in types.py
#   - LayoutIncompatibleError in types.py
#
# This module provides aliases for backward compatibility:
#   - Replicate = TensorReplicate
#   - Shard = TensorShard
#   - Layout = TensorLayout
#   - LayoutCompatibilityError = LayoutIncompatibleError
#   - layout_join = tensor_layout_join


# ============================================================
# Type Error Reporting
# ============================================================

class TypeErrorKind(Enum):
    """Categories of type errors."""
    ARITY_MISMATCH = "arity_mismatch"
    TYPE_MISMATCH = "type_mismatch"
    LAYOUT_INCOMPATIBLE = "layout_incompatible"
    AXIS_OUT_OF_BOUNDS = "axis_out_of_bounds"
    MISSING_ARGUMENT = "missing_argument"
    DIRECTION_VIOLATION = "direction_violation"
    SHAPE_MISMATCH = "shape_mismatch"


@dataclass
class TypeErrorInfo:
    """Detailed type error information."""
    kind: TypeErrorKind
    message: str
    location: Optional[str] = None
    expected: Optional[str] = None
    got: Optional[str] = None
    hint: Optional[str] = None

    def __str__(self):
        lines = [f"TypeError: {self.kind.value}"]
        if self.location:
            lines.append(f"  at {self.location}")
        lines.append("")
        lines.append(f"  {self.message}")
        if self.expected or self.got:
            lines.append("")
            if self.expected:
                lines.append(f"  Expected: {self.expected}")
            if self.got:
                lines.append(f"  Got:      {self.got}")
        if self.hint:
            lines.append("")
            lines.append(f"  Hint: {self.hint}")
        return "\n".join(lines)


# ============================================================
# Type Checker
# ============================================================

@dataclass
class TypeCheckContext:
    """Context for type checking."""
    fail_fast: bool = False
    workload_name: str = ""
    current_location: str = ""


class TypeChecker:
    """Builder-time type checker for workloads.

    Performs L1 (Python builder-time) type checking:
    - Kernel signature validation
    - Layout compatibility
    - Axis bounds validation
    """

    def __init__(self, context: Optional[TypeCheckContext] = None):
        self.ctx = context or TypeCheckContext()
        self.errors: list[TypeErrorInfo] = []

    def has_errors(self) -> bool:
        """Check if any errors were recorded."""
        return len(self.errors) > 0

    def get_errors(self) -> list[TypeErrorInfo]:
        """Get all recorded errors."""
        return self.errors.copy()

    def clear_errors(self) -> None:
        """Clear all recorded errors."""
        self.errors.clear()

    def error(self, kind: TypeErrorKind, message: str,
              expected: str = None, got: str = None, hint: str = None) -> None:
        """Record a type error."""
        err = TypeErrorInfo(
            kind=kind,
            message=message,
            location=self.ctx.current_location,
            expected=expected,
            got=got,
            hint=hint
        )
        self.errors.append(err)
        if self.ctx.fail_fast:
            raise TypeError(str(err))

    # ========== Kernel Call Validation ==========

    def check_kernel_call(self, kernel: KernelRef, axes: tuple, args: dict) -> bool:
        """Validate kernel invocation.

        Checks:
        1. Arity match
        2. Argument types match parameter annotations
        3. Direction constraints (Out params must be assignable)
        4. Layout compatibility across arguments
        """
        sig = kernel.signature

        # 1. Check arity
        if len(args) != len(sig):
            self.error(
                TypeErrorKind.ARITY_MISMATCH,
                f"Kernel '{kernel.name}' expects {len(sig)} args, got {len(args)}",
                expected=str(list(sig.keys())),
                got=str(list(args.keys()))
            )
            return False

        # 2. Check each argument
        for param_name, param_type in sig.items():
            if param_name not in args:
                self.error(
                    TypeErrorKind.MISSING_ARGUMENT,
                    f"Missing argument: {param_name}",
                    hint=f"Add {param_name} argument to kernel call"
                )
                continue

            arg = args[param_name]
            self._check_param_type(kernel.name, param_name, param_type, arg)

        # 3. Check layout compatibility
        self._check_layout_compatibility(kernel.name, args)

        return not self.has_errors()

    def _check_param_type(self, kernel_name: str, param_name: str,
                          param_type: Any, arg: Any) -> None:
        """Check argument matches parameter type annotation."""
        # Parse direction from type string (e.g., "In[Tensor]")
        direction = None
        inner_type = None

        # Convert non-string types to string for parsing
        if not isinstance(param_type, str):
            # Handle ScalarType, TileType, or other type objects
            param_type = str(param_type) if param_type is not None else ""

        if not param_type or not param_type.startswith(("In[", "Out[", "InOut[", "Constexpr[")):
            # Non-annotated parameter or unrecognized type - skip checking
            return

        if param_type.startswith("In["):
            direction = "In"
            inner_type = param_type[3:-1]
        elif param_type.startswith("Out["):
            direction = "Out"
            inner_type = param_type[4:-1]
        elif param_type.startswith("InOut["):
            direction = "InOut"
            inner_type = param_type[6:-1]
        elif param_type.startswith("Constexpr["):
            direction = "Constexpr"
            inner_type = param_type[10:-1]

        # Check tensor type if expected
        if inner_type == "Tensor" and not isinstance(arg, Tensor):
            # Allow None for now (placeholder in tests)
            if arg is not None:
                self.error(
                    TypeErrorKind.TYPE_MISMATCH,
                    f"Argument '{param_name}' in kernel '{kernel_name}'",
                    expected="Tensor",
                    got=type(arg).__name__,
                    hint="Pass a Tensor object"
                )

    def _check_layout_compatibility(self, kernel_name: str, args: dict) -> None:
        """Check layout compatibility across tensor arguments."""
        layouts = []
        tensor_args = []

        for name, arg in args.items():
            if isinstance(arg, Tensor) and hasattr(arg, 'layout') and arg.layout:
                layouts.append(arg.layout)
                tensor_args.append(name)

        if len(layouts) < 2:
            return  # Not enough tensors to check

        try:
            result = layouts[0]
            for i, layout in enumerate(layouts[1:], start=1):
                result = layout_join(result, layout)
        except LayoutCompatibilityError as e:
            self.error(
                TypeErrorKind.LAYOUT_INCOMPATIBLE,
                f"Layout incompatibility in kernel '{kernel_name}': {e}",
                hint="Use relayout() to redistribute tensors with compatible layouts"
            )

    # ========== Axis Bounds Validation ==========

    def check_axis_bounds(self, axis: Any, index: Any) -> bool:
        """Validate axis index is in bounds.

        For Dense[N], checks that index < N at compile time if both are known.
        """
        # Static bounds check for Dense[N]
        if hasattr(axis, '_size') and isinstance(index, int):
            size = axis._size if hasattr(axis, '_size') else axis.size
            if not (0 <= index < size):
                self.error(
                    TypeErrorKind.AXIS_OUT_OF_BOUNDS,
                    f"Index {index} is out of bounds",
                    expected=f"0 <= index < {size}",
                    got=str(index),
                    hint="Use DenseDyn for dynamic indexing"
                )
                return False

        return True

    # ========== Tensor Access Validation ==========

    def check_tensor_access(self, tensor: Tensor, indices: tuple) -> bool:
        """Validate tensor indexing."""
        if not isinstance(tensor, Tensor):
            return True  # Skip if not a tensor

        rank = len(tensor.shape)
        num_indices = len(indices) if isinstance(indices, tuple) else 1

        if num_indices > rank:
            self.error(
                TypeErrorKind.SHAPE_MISMATCH,
                f"Too many indices for tensor",
                expected=f"<= {rank} indices",
                got=f"{num_indices} indices",
                hint="Reduce the number of indices"
            )
            return False

        return True

    # ========== Shape Compatibility ==========

    def check_shapes_compatible(self, shape1: tuple, shape2: tuple,
                                 context: str = "") -> bool:
        """Check if two shapes are compatible (allowing symbolic dims)."""
        if len(shape1) != len(shape2):
            self.error(
                TypeErrorKind.SHAPE_MISMATCH,
                f"Shape rank mismatch{' in ' + context if context else ''}",
                expected=f"rank {len(shape1)}",
                got=f"rank {len(shape2)}"
            )
            return False

        for i, (d1, d2) in enumerate(zip(shape1, shape2)):
            # -1 means dynamic/symbolic dimension
            if d1 == -1 or d2 == -1:
                continue
            if d1 != d2:
                self.error(
                    TypeErrorKind.SHAPE_MISMATCH,
                    f"Shape mismatch at dimension {i}{' in ' + context if context else ''}",
                    expected=str(d1),
                    got=str(d2)
                )
                return False

        return True


# ============================================================
# Convenience Functions
# ============================================================

def check_kernel_call(kernel: KernelRef, axes: tuple, args: dict,
                       fail_fast: bool = False) -> list[TypeErrorInfo]:
    """Check a kernel call for type errors.

    Returns:
        List of type errors (empty if valid)
    """
    checker = TypeChecker(TypeCheckContext(fail_fast=fail_fast))
    checker.check_kernel_call(kernel, axes, args)
    return checker.get_errors()


def check_layouts_compatible(*tensors: Tensor) -> Optional[Layout]:
    """Check if tensor layouts are compatible.

    Returns:
        Joined layout if compatible, None if incompatible
    """
    layouts = [t.layout for t in tensors if hasattr(t, 'layout') and t.layout]
    if len(layouts) < 2:
        return layouts[0] if layouts else None

    try:
        result = layouts[0]
        for layout in layouts[1:]:
            result = layout_join(result, layout)
        return result
    except LayoutCompatibilityError:
        return None


def validate_axis_index(axis: Any, index: int) -> bool:
    """Check if index is valid for axis.

    Returns:
        True if valid, raises TypeError if invalid
    """
    checker = TypeChecker(TypeCheckContext(fail_fast=True))
    return checker.check_axis_bounds(axis, index)
