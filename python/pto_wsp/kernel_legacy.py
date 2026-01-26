"""
Legacy kernel registration API for PTO Workload-Schedule Programming (PTO-WSP) framework.

DEPRECATED: This module is deprecated. For new code, use:
    - @kernel decorator from pto_wsp.builder for kernel definitions
    - @jit_kernel decorator from pto_wsp.kernel for JIT-compiled kernels

This module is retained only for backward compatibility and will be
removed in a future version.
"""

import warnings
from typing import Callable, Any


def register_kernel(name: str, impl_path: str = None) -> Callable:
    """Register a kernel implementation.

    DEPRECATED: Use @kernel or @jit_kernel decorator instead.

    .. deprecated::
        This function is deprecated. Use @kernel from pto_wsp.builder
        or @jit_kernel from pto_wsp.kernel instead.

    This legacy API allows registering kernel implementations by name,
    with optional path to external implementation.

    Args:
        name: Unique kernel name for registration
        impl_path: Optional path to external implementation file.
            If None, the decorated function is used as the implementation.

    Returns:
        Decorator function that registers the kernel

    Example:
        # Deprecated usage:
        @register_kernel("my_kernel")
        def my_kernel_impl(a, b):
            b[:] = a * 2

        # Preferred new usage:
        from pto_wsp import kernel, In, Out, Tensor

        @kernel
        def my_kernel(a: In[Tensor], b: Out[Tensor]):
            ...
    """
    warnings.warn(
        "register_kernel is deprecated. Use @kernel or @jit_kernel instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    def decorator(fn: Callable) -> Callable:
        # In the legacy API, we just return the function unchanged.
        # The actual kernel registration happens at compile time
        # when the workload is lowered to the backend.
        return fn
    return decorator


class ExternalKernel:
    """Reference to an externally implemented kernel.

    DEPRECATED: Use @kernel decorator with KernelRef instead.
    This class is not used anywhere in the codebase and will be
    removed in a future version.

    ExternalKernel represents a kernel implemented outside of Python,
    typically in C++, CUDA, or as a compiled binary. The kernel is
    identified by name and loaded from the specified path at runtime.

    Attributes:
        name: Kernel name for dispatch
        impl_path: Path to the compiled kernel implementation
        input_idx: List of indices for input arguments
        output_idx: List of indices for output arguments

    Example:
        # Deprecated usage:
        flash_attn = ExternalKernel(
            name="flash_attention_v2",
            impl_path="/path/to/flash_attn.so",
            input_idx=[0, 1, 2],  # Q, K, V
            output_idx=[3]         # O
        )

        # Preferred new usage:
        from pto_wsp import kernel, In, Out, Tensor

        @kernel
        def flash_attn_v2(
            Q: In[Tensor],
            K: In[Tensor],
            V: In[Tensor],
            O: Out[Tensor]
        ):
            '''Flash Attention v2 kernel (external implementation).'''
            ...
    """

    def __init__(
        self,
        name: str,
        impl_path: str,
        input_idx: list[int],
        output_idx: list[int]
    ):
        warnings.warn(
            "ExternalKernel is deprecated and unused. Use @kernel decorator instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        """Initialize an external kernel reference.

        Args:
            name: Kernel name for dispatch and identification
            impl_path: Path to the compiled kernel implementation
            input_idx: List of argument indices that are inputs
            output_idx: List of argument indices that are outputs
        """
        self.name = name
        self.impl_path = impl_path
        self.input_idx = input_idx
        self.output_idx = output_idx

    def __repr__(self) -> str:
        """Return string representation of the kernel."""
        return (
            f"ExternalKernel(name={self.name!r}, "
            f"impl_path={self.impl_path!r}, "
            f"inputs={self.input_idx}, "
            f"outputs={self.output_idx})"
        )
