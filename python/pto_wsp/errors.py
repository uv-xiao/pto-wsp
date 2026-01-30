"""
Exception Hierarchy for PTO Workload-Schedule Programming (PTO-WSP) framework.

This module defines the exception taxonomy for the PTO-WSP system.
All exceptions inherit from PtoError for consistent error handling.

Exception Hierarchy:
    PtoError (base)
    ├── CompileError (compilation failures)
    │   ├── TypeCheckError (type mismatch)
    │   └── IRConversionError (IR bridge failures)
    ├── ExecutionError (runtime failures)
    │   └── KernelError (kernel execution failures)
    ├── ScheduleError (scheduling failures)
    └── ChannelError (CSP channel errors)
        ├── ChannelClosed (send/recv on closed channel)
        └── ChannelFull (non-blocking send on full channel)
"""

from __future__ import annotations


class PtoError(Exception):
    """Base exception for all PTO-WSP errors.

    All PTO-WSP specific exceptions inherit from this class to allow
    for unified exception handling.

    Example:
        try:
            workload.compile()
        except PtoError as e:
            print(f"PTO error: {e}")
    """
    pass


# ============================================================
# Compilation Errors
# ============================================================

class CompileError(PtoError):
    """Error during workload/kernel compilation.

    Raised when compilation fails due to invalid workload structure,
    unsupported primitives, or backend limitations.

    Attributes:
        message: Error description
        workload_name: Name of the failing workload (if available)
        target: Target backend (if available)
    """
    def __init__(self, message: str, workload_name: str = None, target: str = None):
        self.workload_name = workload_name
        self.target = target
        super().__init__(message)


class TypeCheckError(CompileError):
    """Type checking failure during compilation.

    Raised when type checking detects incompatible types, invalid
    layouts, or mismatched tensor shapes.

    Attributes:
        message: Error description
        expected: Expected type/layout (if available)
        actual: Actual type/layout found
    """
    def __init__(self, message: str, expected: str = None, actual: str = None):
        self.expected = expected
        self.actual = actual
        super().__init__(message)


class IRConversionError(CompileError):
    """Error during Python → C++ IR conversion.

    Raised when the IR bridge fails to convert a Python workload
    to C++ IR representation.
    """
    pass


# ============================================================
# Execution Errors
# ============================================================

class ExecutionError(PtoError):
    """Error during program execution.

    Raised when a compiled program fails during execution,
    such as kernel crashes or resource exhaustion.

    Attributes:
        message: Error description
        task_id: ID of the failing task (if available)
        kernel_name: Name of the kernel that failed (if available)
    """
    def __init__(self, message: str, task_id: int = None, kernel_name: str = None):
        self.task_id = task_id
        self.kernel_name = kernel_name
        super().__init__(message)


class KernelError(ExecutionError):
    """Error in kernel execution.

    Raised when a kernel implementation fails, such as invalid
    memory access or unsupported operation.
    """
    pass


# ============================================================
# Scheduling Errors
# ============================================================

class ScheduleError(PtoError):
    """Error in schedule configuration.

    Raised when a schedule configuration is invalid or unsupported
    for the target backend.

    Attributes:
        message: Error description
        primitive: The unsupported schedule primitive (if available)
        target: Target backend that doesn't support the primitive
    """
    def __init__(self, message: str, primitive: str = None, target: str = None):
        self.primitive = primitive
        self.target = target
        super().__init__(message)


# ============================================================
# Channel Errors (CSP)
# ============================================================

class ChannelError(PtoError):
    """Base error for CSP channel operations.

    Raised when channel operations fail.
    """
    pass


class ChannelClosed(ChannelError):
    """Error when operating on a closed channel.

    Raised when attempting to send/receive on a channel that
    has been closed.

    Attributes:
        channel_name: Name of the closed channel (if available)
    """
    def __init__(self, message: str = "Operation on closed channel", channel_name: str = None):
        self.channel_name = channel_name
        super().__init__(message)


class ChannelFull(ChannelError):
    """Error when non-blocking send fails on full channel.

    Raised by try_send() when the channel buffer is full.

    Attributes:
        channel_name: Name of the full channel (if available)
        capacity: Channel capacity
    """
    def __init__(self, message: str = "Channel is full", channel_name: str = None, capacity: int = None):
        self.channel_name = channel_name
        self.capacity = capacity
        super().__init__(message)


# ============================================================
# __all__ exports
# ============================================================

__all__ = [
    "PtoError",
    "CompileError",
    "TypeCheckError",
    "IRConversionError",
    "ExecutionError",
    "KernelError",
    "ScheduleError",
    "ChannelError",
    "ChannelClosed",
    "ChannelFull",
]
