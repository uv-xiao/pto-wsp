import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from pto_wsp import Dense, DType, Tensor
from pto_wsp.p_namespace import LoopVar


def test_tensor_views_preserve_base_and_symbolic_indices():
    axis = Dense[4]()
    b = LoopVar("b", axis)

    base = Tensor(data=None, shape=(4, 2, 3), dtype=DType.F32)
    v1 = base[b]
    v2 = v1[1]

    assert v2.base is base
    assert v2.index_exprs[0] is b
    assert v2.index_exprs[1] == 1

