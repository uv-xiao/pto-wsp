from __future__ import annotations

import numpy as np


def bgemm_ref(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a.astype(np.float32) @ b.astype(np.float32)).astype(np.float32)

