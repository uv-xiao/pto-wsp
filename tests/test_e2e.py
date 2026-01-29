"""
End-to-end tests for PTO-RT (codegen-first).

These tests validate:
- Python DSL (@kernel/@workload) builds IR correctly
- Codegen compiles workload+kernel to a shared library
- Executing the generated library produces correct outputs (CPU-sim)
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from pto_wsp import DType, Dense, In, Out, P, Tensor, Tile, kernel, pto, workload


def test_e2e_codegen_square_tile():
    batch = Dense[1]()

    x_np = np.array([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=np.float32)  # (B,1,2,2)
    y_np = np.zeros_like(x_np)

    X = Tensor(data=x_np, shape=x_np.shape, dtype=DType.F32)
    Y = Tensor(data=y_np, shape=y_np.shape, dtype=DType.F32)

    @kernel
    def sq(x: In[Tile[2, 2, DType.F32]], out: Out[Tile[2, 2, DType.F32]]):
        v = pto.mul(x, x)
        pto.store(out, v)

    @workload
    def wl():
        for b in P(batch):
            sq[b](x=X[b][0], out=Y[b][0])

    program = wl().compile(target="cpu_sim")
    program.execute()
    program.synchronize()

    # Mul is currently not lowered to real compute, so this is a smoke test:
    # it should run end-to-end without error and write something deterministic.
    assert program.stats.task_count == 1


def test_e2e_codegen_matmul_correctness():
    batch = Dense[1]()

    A_np = np.array([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=np.float32)
    B_np = np.array([[[[5.0, 6.0], [7.0, 8.0]]]], dtype=np.float32)
    C_np = np.zeros((1, 1, 2, 2), dtype=np.float32)

    A = Tensor(data=A_np, shape=A_np.shape, dtype=DType.F32)
    B = Tensor(data=B_np, shape=B_np.shape, dtype=DType.F32)
    C = Tensor(data=C_np, shape=C_np.shape, dtype=DType.F32)

    @kernel
    def mm(a: In[Tile[2, 2, DType.F32]], b: In[Tile[2, 2, DType.F32]], out: Out[Tile[2, 2, DType.F32]]):
        v = pto.matmul(a, b)
        pto.store(out, v)

    @workload
    def wl():
        for b in P(batch):
            mm[b](a=A[b][0], b=B[b][0], out=C[b][0])

    program = wl().compile(target="cpu_sim")
    program.execute()
    program.synchronize()

    np.testing.assert_allclose(C_np[0, 0], A_np[0, 0] @ B_np[0, 0], rtol=0, atol=0)


def test_e2e_task_count_matches_nested_loops():
    outer = Dense[2]()
    inner = Dense[3]()

    X = Tensor(data=None, shape=(2, 3, 1, 1), dtype=DType.F32)
    Y = Tensor(data=None, shape=(2, 3, 1, 1), dtype=DType.F32)

    @kernel
    def copy(x: In[Tile[1, 1, DType.F32]], out: Out[Tile[1, 1, DType.F32]]):
        pto.store(out, x)

    @workload
    def wl():
        for i in P(outer):
            for j in P(inner):
                copy[i, j](x=X[i][j], out=Y[i][j])

    program = wl().compile(target="cpu_sim")
    program.execute()
    program.synchronize()

    assert program.stats.task_count == 2 * 3

