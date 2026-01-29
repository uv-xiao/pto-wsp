from __future__ import annotations

import numpy as np


def square_pipeline_ref(x: np.ndarray) -> np.ndarray:
    if x.dtype != np.float32:
        x = x.astype(np.float32)
    return x * x

