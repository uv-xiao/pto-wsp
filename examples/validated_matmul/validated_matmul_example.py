#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "_harness"))
sys.path.insert(0, str(_HERE))

from harness import CycleCheck, run_example  # noqa: E402
from golden import matmul_ref  # noqa: E402
from pto_wsp_impl import run_matmul  # noqa: E402


def main() -> bool:
    batch_size, m, n, k = 1, 16, 16, 16
    seed = 42

    np.random.seed(seed)
    a = np.random.randn(batch_size, m, k).astype(np.float32)
    b = np.random.randn(batch_size, k, n).astype(np.float32)

    try:
        run_example(
            "validated_matmul",
            run_pto=lambda: run_matmul(a, b),
            run_golden=lambda: matmul_ref(a, b),
            rtol=1e-5,
            atol=1e-6,
            cycles=CycleCheck(expected=4864, rel_tol=0.20, min_cycles=1),
        )
        return True
    except Exception as e:  # noqa: BLE001
        print(f"validated_matmul: FAIL ({e})")
        return False


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
