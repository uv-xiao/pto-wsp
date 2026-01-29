from __future__ import annotations

import numpy as np


def rmsnorm_ref(x: np.ndarray) -> np.ndarray:
    x_f = x.astype(np.float32)
    mean_sq = np.mean(x_f * x_f, axis=-1, keepdims=True)
    inv = 1.0 / np.sqrt(mean_sq)
    return (x_f * inv).astype(np.float32)
