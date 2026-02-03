from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def build_orch_func_args(arrays: Iterable[np.ndarray]) -> list[int]:
    """Build Phase 1 pto-runtime orchestration args.

    ABI: `[ptr0, nbytes0, ptr1, nbytes1, ...]`
    """
    out: list[int] = []
    for a in arrays:
        if not isinstance(a, np.ndarray):
            raise TypeError(f"expected numpy.ndarray, got {type(a)}")
        out.append(int(a.ctypes.data))
        out.append(int(a.nbytes))
    return out
