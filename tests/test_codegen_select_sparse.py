import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from pto_wsp import DType, InOut, P, Sparse, Tensor, Tile, kernel, pto, workload


def test_sparse_select_indices_can_change_without_recompile():
    # y is shaped so that y[e] yields a 2D (1x1) tile view.
    y_data = np.zeros((4, 1, 1), dtype=np.int32)
    y = Tensor(data=y_data, shape=y_data.shape, dtype=DType.I32)

    sparse = Sparse(outer_size=2, indptr=[0, 2, 3], indices=[0, 2, 1])  # nnz=3

    @kernel
    def mark(y_tile: InOut[Tile[1, 1, DType.I32]]):
        v = pto.load(y_tile)
        one = pto.constant(1.0, DType.I32)
        v1 = pto.add(v, one)
        pto.store(y_tile, v1)

    @workload
    def wl():
        for e in P.sel(sparse, name="e"):
            mark[e](y_tile=y[e])

    program = wl().named("dyn_sparse_select").compile(target="cpu_sim")
    assert program.using_cpp_backend

    sid = id(sparse)

    def bind_sparse(outer_size: int, indptr: list[int], indices: list[int]) -> None:
        program.set_symbol_u64(f"outer_{sid}", outer_size)
        program.set_symbol_ptr(f"indptr_{sid}", np.asarray(indptr, dtype=np.int32))
        program.set_symbol_ptr(f"indices_{sid}", np.asarray(indices, dtype=np.int32))

    # Run 1: indices [0,2,1]
    y_data[:] = 0
    bind_sparse(2, [0, 2, 3], [0, 2, 1])
    program.execute()
    program.synchronize()
    np.testing.assert_array_equal(y_data.reshape(4), np.array([1, 1, 1, 0], dtype=np.int32))
    cycles1 = int(program.stats.total_cycles)
    assert cycles1 > 0

    # Run 2: indices [3] (change runtime symbols; no recompile)
    y_data[:] = 0
    bind_sparse(1, [0, 1], [3])
    program.execute()
    program.synchronize()
    np.testing.assert_array_equal(y_data.reshape(4), np.array([0, 0, 0, 1], dtype=np.int32))
    cycles2 = int(program.stats.total_cycles)
    assert cycles2 > 0
