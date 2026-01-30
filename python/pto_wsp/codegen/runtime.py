from __future__ import annotations

import ctypes
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from pto_wsp.types import DType, Tensor


def _dtype_to_numpy(dtype: DType) -> np.dtype:
    if dtype == DType.F32:
        return np.dtype(np.float32)
    if dtype == DType.F16:
        return np.dtype(np.float16)
    if dtype == DType.I32:
        return np.dtype(np.int32)
    if dtype == DType.I64:
        return np.dtype(np.int64)
    raise ValueError(f"Unsupported dtype for runtime: {dtype}")


def _ensure_numpy(t: Tensor) -> np.ndarray:
    if t.data is None:
        t.data = np.zeros(t.shape, dtype=_dtype_to_numpy(t.dtype))
    if not isinstance(t.data, np.ndarray):
        raise TypeError(f"Tensor.data must be a numpy.ndarray for codegen runtime (got {type(t.data)})")
    return t.data


# -----------------------------------------------------------------------------
# ctypes ABI structs (mirror include/pto/wsp/codegen/abi/*.hpp)
# -----------------------------------------------------------------------------


class RuntimeContextC(ctypes.Structure):
    pass


GET_AXIS_SIZE = ctypes.CFUNCTYPE(ctypes.c_int64, ctypes.c_void_p, ctypes.c_char_p)
GET_SYMBOL_U64 = ctypes.CFUNCTYPE(ctypes.c_uint64, ctypes.c_void_p, ctypes.c_uint64)
GET_SYMBOL_PTR = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint64)
GET_TENSOR_PTR = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32)
GET_TENSOR_STRIDE = ctypes.CFUNCTYPE(ctypes.c_uint64, ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32)
GET_KERNEL = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32)


RuntimeContextC._fields_ = [
    ("get_axis_size", GET_AXIS_SIZE),
    ("get_symbol_u64", GET_SYMBOL_U64),
    ("get_symbol_ptr", GET_SYMBOL_PTR),
    ("get_tensor_ptr", GET_TENSOR_PTR),
    ("get_tensor_stride", GET_TENSOR_STRIDE),
    ("get_kernel", GET_KERNEL),
    ("ctx", ctypes.c_void_p),
]


class CSPTContextC(ctypes.Structure):
    pass


class TimingConfigC(ctypes.Structure):
    _fields_ = [
        ("loop_overhead", ctypes.c_uint64),
        ("dispatch_overhead", ctypes.c_uint64),
        ("channel_latency", ctypes.c_uint64),
    ]


ADVANCE_CYCLES = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_uint64)
GET_TIME = ctypes.CFUNCTYPE(ctypes.c_uint64, ctypes.c_void_p)


CSPTContextC._fields_ = [
    ("ctx", ctypes.c_void_p),
    ("advance_cycles", ADVANCE_CYCLES),
    ("get_time", GET_TIME),
    ("timing", ctypes.POINTER(TimingConfigC)),
]


class _TimingState:
    __slots__ = ("cycles",)

    def __init__(self) -> None:
        self.cycles: int = 0


@dataclass
class CodegenRuntime:
    """Python-side runtime that backs the `RuntimeContext` ABI."""

    tensors: List[Tensor]
    axis_sizes: Dict[str, int]

    def __post_init__(self) -> None:
        self._handle = id(self)
        _RUNTIME_REGISTRY[self._handle] = self

        # Ensure all tensors have numpy storage.
        self._arrays: List[np.ndarray] = [_ensure_numpy(t) for t in self.tensors]

        # Keep C callbacks alive.
        self._cb_get_axis_size = GET_AXIS_SIZE(self._get_axis_size)
        self._cb_get_symbol_u64 = GET_SYMBOL_U64(self._get_symbol_u64)
        self._cb_get_symbol_ptr = GET_SYMBOL_PTR(self._get_symbol_ptr)
        self._cb_get_tensor_ptr = GET_TENSOR_PTR(self._get_tensor_ptr)
        self._cb_get_tensor_stride = GET_TENSOR_STRIDE(self._get_tensor_stride)
        self._cb_get_kernel = GET_KERNEL(self._get_kernel)

        # Timing state + callbacks (CSPT)
        self._timing_state = _TimingState()
        self._cb_advance_cycles = ADVANCE_CYCLES(self._advance_cycles)
        self._cb_get_time = GET_TIME(self._get_time)
        self._timing_cfg = TimingConfigC(
            loop_overhead=1,
            dispatch_overhead=2,
            channel_latency=10,
        )
        self.cspt = CSPTContextC(
            ctypes.c_void_p(self._handle),
            self._cb_advance_cycles,
            self._cb_get_time,
            ctypes.pointer(self._timing_cfg),
        )

        self.c_ctx = RuntimeContextC(
            self._cb_get_axis_size,
            self._cb_get_symbol_u64,
            self._cb_get_symbol_ptr,
            self._cb_get_tensor_ptr,
            self._cb_get_tensor_stride,
            self._cb_get_kernel,
            ctypes.c_void_p(self._handle),
        )

    def close(self) -> None:
        _RUNTIME_REGISTRY.pop(self._handle, None)

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    @staticmethod
    def _from_handle(ctx_ptr: ctypes.c_void_p) -> "CodegenRuntime":
        handle = int(ctx_ptr)
        rt = _RUNTIME_REGISTRY.get(handle)
        if rt is None:
            raise RuntimeError("Invalid runtime context handle")
        return rt

    @staticmethod
    def _get_axis_size(ctx: ctypes.c_void_p, name: bytes) -> int:
        rt = CodegenRuntime._from_handle(ctx)
        key = name.decode("utf-8")
        return int(rt.axis_sizes.get(key, 0))

    @staticmethod
    def _get_symbol_u64(ctx: ctypes.c_void_p, symbol_id: int) -> int:
        rt = CodegenRuntime._from_handle(ctx)
        return int(getattr(rt, "symbols_u64", {}).get(int(symbol_id), 0))

    @staticmethod
    def _get_symbol_ptr(ctx: ctypes.c_void_p, symbol_id: int) -> int:
        rt = CodegenRuntime._from_handle(ctx)
        arr = getattr(rt, "symbols_ptr", {}).get(int(symbol_id))
        if arr is None:
            return 0
        return int(arr.ctypes.data)

    @staticmethod
    def _get_tensor_ptr(ctx: ctypes.c_void_p, tensor_id: int) -> int:
        rt = CodegenRuntime._from_handle(ctx)
        arr = rt._arrays[int(tensor_id)]
        return int(arr.ctypes.data)

    @staticmethod
    def _get_tensor_stride(ctx: ctypes.c_void_p, tensor_id: int, dim: int) -> int:
        rt = CodegenRuntime._from_handle(ctx)
        arr = rt._arrays[int(tensor_id)]
        # numpy strides are bytes; convert to element strides.
        itemsize = arr.dtype.itemsize
        return int(arr.strides[int(dim)] // itemsize)

    @staticmethod
    def _get_kernel(ctx: ctypes.c_void_p, kernel_id: int) -> int:
        # Unused in current workload emission (direct calls).
        return 0

    @staticmethod
    def _advance_cycles(ctx: ctypes.c_void_p, cycles: int) -> None:
        rt = CodegenRuntime._from_handle(ctx)
        rt._timing_state.cycles += int(cycles)

    @staticmethod
    def _get_time(ctx: ctypes.c_void_p) -> int:
        rt = CodegenRuntime._from_handle(ctx)
        return int(rt._timing_state.cycles)


_RUNTIME_REGISTRY: Dict[int, CodegenRuntime] = {}
