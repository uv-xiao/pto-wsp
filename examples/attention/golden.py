from __future__ import annotations

import math
import numpy as np


def tiled_attention_ref(q: np.ndarray, k: np.ndarray, v: np.ndarray, *, tile_seq: int) -> np.ndarray:
    # Shapes: [B,H,S,D]
    bsz, heads, seq, d = q.shape
    if k.shape != q.shape or v.shape != q.shape:
        raise ValueError("Q/K/V must have the same shape")
    if seq % tile_seq:
        raise ValueError("seq must be divisible by tile_seq")

    out = np.zeros_like(q, dtype=np.float32)
    tiles = seq // tile_seq
    for b in range(bsz):
        for h in range(heads):
            for t in range(tiles):
                sl = slice(t * tile_seq, (t + 1) * tile_seq)
                qt = q[b, h, sl, :].astype(np.float32)
                kt = k[b, h, sl, :].astype(np.float32)
                vt = v[b, h, sl, :].astype(np.float32)
                scores = (qt @ kt.T) * (1.0 / math.sqrt(d))
                scores = scores - np.max(scores, axis=-1, keepdims=True)
                w = np.exp(scores) / np.sum(np.exp(scores), axis=-1, keepdims=True)
                out[b, h, sl, :] = w @ vt
    return out

