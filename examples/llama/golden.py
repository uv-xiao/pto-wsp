from __future__ import annotations

import math
import numpy as np


def llama_block_ref(x: np.ndarray, w1: np.ndarray, w2: np.ndarray, *, tile_seq: int, eps: float) -> np.ndarray:
    # Shapes: x[SEQ,D], w1[D,H], w2[H,D]
    seq, d = x.shape
    if w1.shape[0] != d or w2.shape[1] != d:
        raise ValueError("weight shapes incompatible with x")
    if seq % tile_seq:
        raise ValueError("seq must be divisible by tile_seq")
    tiles = seq // tile_seq

    out = np.zeros_like(x, dtype=np.float32)
    for t in range(tiles):
        sl = slice(t * tile_seq, (t + 1) * tile_seq)
        xt = x[sl].astype(np.float32)

        scores = (xt @ xt.T) * (1.0 / math.sqrt(d))
        scores = scores - np.max(scores, axis=-1, keepdims=True)
        w = np.exp(scores) / np.sum(np.exp(scores), axis=-1, keepdims=True)
        attn = w @ xt

        res1 = xt + attn
        mean_sq = np.mean(res1 * res1, axis=-1, keepdims=True)
        xn = res1 / np.sqrt(mean_sq + eps)

        h = xn @ w1.astype(np.float32)
        relu = np.maximum(h, 0.0)
        mlp = relu @ w2.astype(np.float32)
        out[sl] = res1 + mlp
    return out

