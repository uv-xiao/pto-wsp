from __future__ import annotations

import math
import numpy as np


def decode_attention_ref(q: np.ndarray, k: np.ndarray, v: np.ndarray) -> np.ndarray:
    d = q.shape[1]
    scores = (q @ k.T) * (1.0 / math.sqrt(d))
    scores = scores - np.max(scores, axis=-1, keepdims=True)
    w = np.exp(scores) / np.sum(np.exp(scores), axis=-1, keepdims=True)
    return w @ v

