#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "_harness"))
sys.path.insert(0, str(_HERE))

from harness import CycleCheck, run_example  # noqa: E402
from golden import llama_block_ref  # noqa: E402
from pto_wsp_impl import run_llama_block  # noqa: E402


def main() -> bool:
    seq, d, tile_seq, mlp_dim = 16, 8, 8, 16
    eps = 1e-6
    seed = 0

    rng = np.random.default_rng(seed)
    x = rng.standard_normal((seq, d), dtype=np.float32)
    w1 = rng.standard_normal((d, mlp_dim), dtype=np.float32)
    w2 = rng.standard_normal((mlp_dim, d), dtype=np.float32)

    try:
        run_example(
            "llama",
            run_pto=lambda: run_llama_block(x, w1, w2, tile_seq=tile_seq, eps=eps),
            run_golden=lambda: llama_block_ref(x, w1, w2, tile_seq=tile_seq, eps=eps),
            rtol=1e-5,
            atol=1e-6,
            cycles=CycleCheck(expected=4752, rel_tol=0.20, min_cycles=1),
        )
        return True
    except Exception as e:  # noqa: BLE001
        print(f"llama: FAIL ({e})")
        return False


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
