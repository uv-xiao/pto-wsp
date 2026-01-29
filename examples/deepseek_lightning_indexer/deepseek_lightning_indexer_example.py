#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import os

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "_harness"))
sys.path.insert(0, str(_HERE))

from harness import CycleCheck, assert_cycles  # noqa: E402
from golden import make_inputs, stable_topk_indices_desc  # noqa: E402
from pto_wsp_impl import run  # noqa: E402


def main() -> bool:
    bsz = 2
    tiles = 4
    cand = 32
    k = 8
    seed = 0

    scores, eff_seq = make_inputs(seed=seed, bsz=bsz, tiles=tiles, cand=cand)
    ref = stable_topk_indices_desc(scores, k=k)

    only = os.environ.get("PTO_WSP_DEEPSEEK_LIGHTNING_INDEXER_IMPL", "").strip().lower()
    all_impls = ["pto", "ptoisa", "cppfile"]
    if not only:
        impls = all_impls
    else:
        if only not in all_impls:
            print(f"Invalid PTO_WSP_DEEPSEEK_LIGHTNING_INDEXER_IMPL={only!r} (expected one of {all_impls})")
            return False
        impls = [only]

    ok = True
    expected_cycles = {
        # Tolerance-based baselines. Keep a generous window because PTO-ISA
        # CPU-sim timing may shift across toolchains.
        "pto": 800,
        "ptoisa": 1319680,
        "cppfile": 1319680,
    }

    for impl in impls:
        got, cycles = run(scores, eff_seq, k=k, impl=impl)
        if not np.array_equal(got, ref):
            print(f"{impl}: TopK indices mismatch")
            print("got:\n", got)
            print("ref:\n", ref)
            ok = False

        try:
            assert_cycles(
                int(cycles),
                CycleCheck(expected=int(expected_cycles[impl]), rel_tol=0.25, min_cycles=1),
                name=f"deepseek_lightning_indexer:{impl}:cycles",
            )
            print(f"deepseek_lightning_indexer:{impl}: total_cycles={int(cycles)}")
        except Exception as e:  # noqa: BLE001
            print(f"deepseek_lightning_indexer:{impl}: cycles check failed ({e})")
            ok = False

    print("Status: PASS" if ok else "Status: FAIL")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
