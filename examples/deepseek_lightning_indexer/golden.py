from __future__ import annotations

import numpy as np


def stable_topk_indices_desc(scores: np.ndarray, k: int) -> np.ndarray:
    """Stable TopK indices per row (descending), matching PTO-ISA TSORT32 semantics.

    Tie-break: lower original index first.
    """
    if scores.ndim != 3:
        raise ValueError(f"scores must be [B,S,C], got shape={scores.shape}")
    bsz, st, cand = scores.shape
    if k > cand:
        raise ValueError(f"k={k} > candidates={cand}")

    out = np.empty((bsz, st, k), dtype=np.int32)
    idx = np.arange(cand, dtype=np.int32)
    for b in range(bsz):
        for s in range(st):
            # lexsort uses last key as primary sort key.
            # Sort by score desc then index asc.
            order = np.lexsort((idx, -scores[b, s].astype(np.float64)))
            out[b, s] = order[:k].astype(np.int32)
    return out


def make_inputs(seed: int, bsz: int, tiles: int, cand: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    scores = rng.standard_normal((bsz, tiles, cand), dtype=np.float32)

    # Pick values that cover all 4 tiers (2K/8K/64K/128K) deterministically.
    # Shape: [B,S,1,1] as u64 (read via slot_load_u64 in the artifact).
    eff_seq_choices = np.array([512, 4096, 16384, 131072], dtype=np.uint64)
    eff_seq = rng.choice(eff_seq_choices, size=(bsz, tiles, 1, 1), replace=True)
    return scores, eff_seq

