import os
import sys

import numpy as np


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))


def test_build_orch_func_args_layout():
    from pto_wsp.pto_runtime_abi import build_orch_func_args

    a = np.zeros((4, 4), dtype=np.float32)
    b = np.zeros((4, 4), dtype=np.float32)

    args = build_orch_func_args([a, b])
    assert args == [int(a.ctypes.data), a.nbytes, int(b.ctypes.data), b.nbytes]
