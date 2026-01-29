import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from pto_wsp import DType, Dense, In, Out, P, Tensor, Tile, kernel, pto, workload


def test_codegen_executes_matmul_f32():
    batch = Dense[1]()

    A_np = np.array([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=np.float32)  # (B,1,2,2)
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

