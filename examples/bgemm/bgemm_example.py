#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "_harness"))
sys.path.insert(0, str(_HERE))

from harness import CycleCheck, run_example  # noqa: E402
from golden import bgemm_ref  # noqa: E402
from pto_wsp_impl import run_bgemm  # noqa: E402


def main() -> bool:
    batch_size, m, n, k = 1, 64, 64, 64
    seed = 0

    rng = np.random.default_rng(seed)
    a = rng.standard_normal((batch_size, m, k), dtype=np.float32).astype(np.float16)
    b = rng.standard_normal((batch_size, k, n), dtype=np.float32).astype(np.float16)

    try:
        run_example(
            "bgemm",
            run_pto=lambda: run_bgemm(a, b),
            run_golden=lambda: bgemm_ref(a, b),
            rtol=1e-2,
            atol=1e-2,
            cycles=CycleCheck(expected=275968, rel_tol=0.20, min_cycles=1),
        )
        return True
    except Exception as e:  # noqa: BLE001
        print(f"bgemm: FAIL ({e})")
        return False


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
