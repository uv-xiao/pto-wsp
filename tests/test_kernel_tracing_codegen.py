import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from pto_wsp import kernel, In, Out, Tile, DType, pto


def test_kernel_trace_uses_type_annotations_for_shapes():
    @kernel
    def mm(a: In[Tile[4, 8, DType.F32]], b: In[Tile[8, 4, DType.F32]], c: Out[Tile[4, 4, DType.F32]]):
        out = pto.matmul(a, b)
        pto.store(c, out)

    ir = mm.trace()
    assert ir.params[0][1].shape == (4, 8)
    assert ir.params[1][1].shape == (8, 4)
    assert ir.params[2][1].shape == (4, 4)
    assert ir.params[0][1].dtype == DType.F32

