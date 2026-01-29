import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from pto_wsp import DType, Dense, In, Out, P, Tensor, Tile, kernel, pto, workload
from pto_wsp.scalar_expr import axis, symbol_u64


def test_codegen_cond_runtime_predicate_by_axis_var():
    batch = Dense[4]()

    A0_np = np.full((4, 1, 2, 2), 1.0, dtype=np.float32)
    A1_np = np.full((4, 1, 2, 2), 2.0, dtype=np.float32)
    C_np = np.zeros((4, 1, 2, 2), dtype=np.float32)

    A0 = Tensor(data=A0_np, shape=A0_np.shape, dtype=DType.F32)
    A1 = Tensor(data=A1_np, shape=A1_np.shape, dtype=DType.F32)
    C = Tensor(data=C_np, shape=C_np.shape, dtype=DType.F32)

    @kernel
    def copy(src: In[Tile[2, 2, DType.F32]], out: Out[Tile[2, 2, DType.F32]]):
        v = pto.load(src)
        pto.store(out, v)

    @workload
    def wl():
        for b in P(batch):
            with P.when(axis("b") < 2):
                copy[b](src=A0[b][0], out=C[b][0])
            with P.otherwise():
                copy[b](src=A1[b][0], out=C[b][0])

    program = wl().compile(target="cpu_sim")
    program.execute()
    program.synchronize()

    np.testing.assert_allclose(C_np[0, 0], A0_np[0, 0], rtol=0, atol=0)
    np.testing.assert_allclose(C_np[1, 0], A0_np[1, 0], rtol=0, atol=0)
    np.testing.assert_allclose(C_np[2, 0], A1_np[2, 0], rtol=0, atol=0)
    np.testing.assert_allclose(C_np[3, 0], A1_np[3, 0], rtol=0, atol=0)


def test_codegen_cond_runtime_predicate_by_symbol_without_rebuild():
    batch = Dense[4]()

    A0_np = np.full((4, 1, 2, 2), 1.0, dtype=np.float32)
    A1_np = np.full((4, 1, 2, 2), 2.0, dtype=np.float32)
    C_np = np.zeros((4, 1, 2, 2), dtype=np.float32)

    A0 = Tensor(data=A0_np, shape=A0_np.shape, dtype=DType.F32)
    A1 = Tensor(data=A1_np, shape=A1_np.shape, dtype=DType.F32)
    C = Tensor(data=C_np, shape=C_np.shape, dtype=DType.F32)

    @kernel
    def copy(src: In[Tile[2, 2, DType.F32]], out: Out[Tile[2, 2, DType.F32]]):
        v = pto.load(src)
        pto.store(out, v)

    @workload
    def wl():
        for b in P(batch):
            with P.when(axis("b") < symbol_u64("cut").cast_i64()):
                copy[b](src=A0[b][0], out=C[b][0])
            with P.otherwise():
                copy[b](src=A1[b][0], out=C[b][0])

    program = wl().compile(target="cpu_sim")

    program.set_symbol_u64("cut", 2)
    program.execute()
    program.synchronize()
    np.testing.assert_allclose(C_np[0, 0], A0_np[0, 0], rtol=0, atol=0)
    np.testing.assert_allclose(C_np[1, 0], A0_np[1, 0], rtol=0, atol=0)
    np.testing.assert_allclose(C_np[2, 0], A1_np[2, 0], rtol=0, atol=0)
    np.testing.assert_allclose(C_np[3, 0], A1_np[3, 0], rtol=0, atol=0)

    C_np.fill(0)
    program.set_symbol_u64("cut", 1)
    program.execute()
    program.synchronize()
    np.testing.assert_allclose(C_np[0, 0], A0_np[0, 0], rtol=0, atol=0)
    np.testing.assert_allclose(C_np[1, 0], A1_np[1, 0], rtol=0, atol=0)
    np.testing.assert_allclose(C_np[2, 0], A1_np[2, 0], rtol=0, atol=0)
    np.testing.assert_allclose(C_np[3, 0], A1_np[3, 0], rtol=0, atol=0)
