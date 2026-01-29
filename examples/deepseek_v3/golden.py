from __future__ import annotations

import numpy as np


def moe_ref(x: np.ndarray, w0: np.ndarray, w1: np.ndarray, g: np.ndarray) -> np.ndarray:
    e0 = x @ w0
    e1 = x @ w1
    return g * e0 + (1.0 - g) * e1

