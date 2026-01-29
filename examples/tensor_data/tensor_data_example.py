#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "_harness"))
sys.path.insert(0, str(_HERE))

from harness import CycleCheck, run_example  # noqa: E402
from golden import add_ref, square_ref  # noqa: E402
from pto_wsp_impl import run_add_tiles, run_square_tiles  # noqa: E402


def main() -> bool:
    seed = 0
    rng = np.random.default_rng(seed)

    # Tile-add input: [B,TM,TN,R,C]
    b, tm, tn, r, c = 2, 2, 2, 32, 32
    a = rng.standard_normal((b, tm, tn, r, c), dtype=np.float32)
    b_in = rng.standard_normal((b, tm, tn, r, c), dtype=np.float32)

    # Tile-square input: [B,T,R,C]
    x = rng.standard_normal((2, tm * tn, r, c), dtype=np.float32)

    try:
        run_example(
            "tensor_data:add",
            run_pto=lambda: run_add_tiles(a, b_in),
            run_golden=lambda: add_ref(a, b_in),
            rtol=1e-5,
            atol=1e-6,
            cycles=CycleCheck(expected=8192, rel_tol=0.20, min_cycles=1),
        )
        run_example(
            "tensor_data:square",
            run_pto=lambda: run_square_tiles(x),
            run_golden=lambda: square_ref(x),
            rtol=1e-5,
            atol=1e-6,
            cycles=CycleCheck(expected=6144, rel_tol=0.20, min_cycles=1),
        )
        return True
    except Exception as e:  # noqa: BLE001
        print(f"tensor_data: FAIL ({e})")
        return False


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
