from __future__ import annotations

import numpy as np


def add_ref(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a + b


def square_ref(x: np.ndarray) -> np.ndarray:
    return x * x

