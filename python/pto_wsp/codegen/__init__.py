"""
Codegen frontend helpers for PTO-RT v9.

The v9 contract is that compilation + codegen + artifact builds are owned by C++
(via `pto_ir_cpp`). Python remains a frontend for workload/kernel construction
and provides only runtime ABI helpers (`runtime.py`).
"""

from .runtime import CodegenRuntime, RuntimeContextC  # noqa: F401
