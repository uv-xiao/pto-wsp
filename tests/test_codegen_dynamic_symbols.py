import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from pto_wsp import DType, In, Out, Scalar, Symbol, Tensor, Tile, kernel, pto, workload


def test_scalar_symbol_can_change_without_recompile():
    x_data = np.arange(16, dtype=np.float32).reshape(4, 4)
    y_data = np.zeros((4, 4), dtype=np.float32)

    x = Tensor(data=x_data, shape=x_data.shape, dtype=DType.F32)
    y = Tensor(data=y_data, shape=y_data.shape, dtype=DType.F32)

    @kernel
    def scale(
        x_tile: In[Tile[4, 4, DType.F32]],
        y_tile: Out[Tile[4, 4, DType.F32]],
        s: Scalar[DType.F32],
    ):
        out = pto.mul(x_tile, s)
        pto.store(y_tile, out)

    @workload
    def wl():
        scale(x_tile=x, y_tile=y, s=Symbol("scale_s"))

    program = wl().named("dyn_scalar_symbol").compile(target="cpu_sim")
    assert program.using_cpp_backend

    # Run 1: s=2.0
    y_data[:] = 0
    program.set_symbol_f32("scale_s", 2.0)
    program.execute()
    program.synchronize()
    np.testing.assert_allclose(y_data, x_data * 2.0, rtol=0, atol=0)
    cycles2 = int(program.stats.total_cycles)
    assert cycles2 > 0

    # Run 2: s=3.0 (same compiled artifact; just different runtime symbol)
    y_data[:] = 0
    program.set_symbol_f32("scale_s", 3.0)
    program.execute()
    program.synchronize()
    np.testing.assert_allclose(y_data, x_data * 3.0, rtol=0, atol=0)
    cycles3 = int(program.stats.total_cycles)
    assert cycles3 == cycles2
