#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "_harness"))
sys.path.insert(0, str(_HERE))

from harness import CycleCheck, run_example  # noqa: E402
from golden import decode_attention_ref  # noqa: E402
from pto_wsp_impl import run_decode_attention  # noqa: E402


def main() -> bool:
    d = 16
    kv = 64
    seed = 0

    rng = np.random.default_rng(seed)
    q = rng.standard_normal((1, d), dtype=np.float32)
    k = rng.standard_normal((kv, d), dtype=np.float32)
    v = rng.standard_normal((kv, d), dtype=np.float32)

    try:
        run_example(
            "flashinfer_decode",
            run_pto=lambda: run_decode_attention(q, k, v),
            run_golden=lambda: decode_attention_ref(q, k, v),
            rtol=1e-5,
            atol=1e-6,
            cycles=CycleCheck(expected=4514, rel_tol=0.20, min_cycles=1),
        )
        return True
    except Exception as e:  # noqa: BLE001
        print(f"flashinfer_decode: FAIL ({e})")
        return False


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
