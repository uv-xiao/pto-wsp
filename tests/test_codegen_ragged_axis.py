import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from pto_wsp import DType, InOut, P, Ragged, Tensor, Tile, kernel, pto, workload


def test_ragged_axis_total_can_change_without_recompile():
    y_data = np.zeros((3, 1, 1), dtype=np.int32)
    y = Tensor(data=y_data, shape=y_data.shape, dtype=DType.I32)

    ragged = Ragged(outer_size=2, lengths=[1, 1])

    @kernel
    def mark(y_tile: InOut[Tile[1, 1, DType.I32]]):
        v = pto.load(y_tile)
        one = pto.constant(1.0, DType.I32)
        v1 = pto.add(v, one)
        pto.store(y_tile, v1)

    @workload
    def wl():
        for t in P.seq(ragged, name="t"):
            mark[t](y_tile=y[t])

    program = wl().named("dyn_ragged").compile(target="cpu_sim")
    assert program.using_cpp_backend

    rid = id(ragged)

    def bind_ragged(outer_size: int, lengths: list[int]) -> None:
        program.set_symbol_u64(f"outer_{rid}", outer_size)
        program.set_symbol_ptr(f"lengths_{rid}", np.asarray(lengths, dtype=np.int32))

    # Run 1: total = 2
    y_data[:] = 0
    bind_ragged(2, [1, 1])
    program.execute()
    program.synchronize()
    np.testing.assert_array_equal(y_data.reshape(3), np.array([1, 1, 0], dtype=np.int32))

    # Run 2: total = 3 (same compiled artifact; just different runtime ragged lengths)
    y_data[:] = 0
    bind_ragged(2, [1, 2])
    program.execute()
    program.synchronize()
    np.testing.assert_array_equal(y_data.reshape(3), np.array([1, 1, 1], dtype=np.int32))

