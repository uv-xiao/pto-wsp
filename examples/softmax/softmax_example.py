#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "_harness"))
sys.path.insert(0, str(_HERE))

from harness import CycleCheck, run_example  # noqa: E402
from golden import softmax_ref  # noqa: E402
from pto_wsp_impl import run_softmax  # noqa: E402


def main() -> bool:
    batch, seq, vocab = 1, 32, 256
    seed = 0

    rng = np.random.default_rng(seed)
    logits = (0.1 * rng.standard_normal((batch, seq, vocab), dtype=np.float32)).astype(np.float32)

    try:
        run_example(
            "softmax",
            run_pto=lambda: run_softmax(logits),
            run_golden=lambda: softmax_ref(logits),
            rtol=1e-2,
            atol=1e-2,
            cycles=CycleCheck(expected=39848, rel_tol=0.20, min_cycles=1),
        )
        return True
    except Exception as e:  # noqa: BLE001
        print(f"softmax: FAIL ({e})")
        return False


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
