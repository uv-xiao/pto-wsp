import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from pto_wsp import DType, DenseDyn, In, Out, P, Tensor, Tile, kernel, pto, workload


def test_dense_dyn_axis_size_can_change_without_recompile():
    # Keep everything tiny and deterministic.
    max_batch = 4
    m = 4
    n = 4
    k = 4

    np.random.seed(0)
    a_data = np.random.randn(max_batch, m, k).astype(np.float32)
    b_data = np.random.randn(max_batch, k, n).astype(np.float32)
    c_data = np.zeros((max_batch, m, n), dtype=np.float32)

    a = Tensor(data=a_data, shape=a_data.shape, dtype=DType.F32)
    b = Tensor(data=b_data, shape=b_data.shape, dtype=DType.F32)
    c = Tensor(data=c_data, shape=c_data.shape, dtype=DType.F32)

    batch = DenseDyn(0)  # size is provided at runtime via Program.set_axis_sizes

    @kernel
    def mm(
        a_tile: In[Tile[4, 4, DType.F32]],
        b_tile: In[Tile[4, 4, DType.F32]],
        c_tile: Out[Tile[4, 4, DType.F32]],
    ):
        out = pto.matmul(a_tile, b_tile)
        pto.store(c_tile, out)

    @workload
    def wl():
        for bidx in P(batch):
            mm[bidx](a_tile=a[bidx], b_tile=b[bidx], c_tile=c[bidx])

    program = wl().named("dyn_axes").compile(target="cpu_sim")
    assert program.using_cpp_backend

    # Run 1: batch = 2
    c_data[:] = 0
    program.set_axis_sizes({"b": 2})
    program.execute()
    program.synchronize()

    ref2 = np.matmul(a_data[:2], b_data[:2])
    np.testing.assert_allclose(c_data[:2], ref2, rtol=0, atol=1e-5)
    np.testing.assert_allclose(c_data[2:], 0, rtol=0, atol=0)
    cycles2 = int(program.stats.total_cycles)
    assert cycles2 > 0
    per_task2 = cycles2 / 2.0

    # Run 2: batch = 4 (same compiled artifact; just different runtime axis size)
    c_data[:] = 0
    program.set_axis_sizes({"b": 4})
    program.execute()
    program.synchronize()

    ref4 = np.matmul(a_data[:4], b_data[:4])
    np.testing.assert_allclose(c_data[:4], ref4, rtol=0, atol=1e-5)
    cycles4 = int(program.stats.total_cycles)
    assert cycles4 > cycles2
    per_task4 = cycles4 / 4.0

    # v9 timing: per-task cycles come from PTO-ISA cycle reporting; should be stable
    # across runs (dynamic axis values change loop trip count, not kernel shape).
    assert per_task4 == per_task2
