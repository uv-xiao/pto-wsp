import os
import sys

import pytest


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))


ASCEND_HOME_PATH = os.environ.get("ASCEND_HOME_PATH")


@pytest.mark.skipif(not ASCEND_HOME_PATH, reason="ASCEND_HOME_PATH not set (Ascend toolkit not available)")
def test_pto_runtime_a2a3_scale_smoke():
    import numpy as np

    from pto_wsp import DType, In, Out, Scalar, Tensor, Tile, kernel, pto, workload

    x_data = np.arange(16, dtype=np.float32).reshape(4, 4)
    y_data = np.zeros((4, 4), dtype=np.float32)

    x = Tensor(data=x_data, shape=x_data.shape, dtype=DType.F32)
    y = Tensor(data=y_data, shape=y_data.shape, dtype=DType.F32)

    @kernel
    def scale(x_tile: In[Tile[4, 4, DType.F32]], y_tile: Out[Tile[4, 4, DType.F32]], s: Scalar[DType.F32]):
        out = pto.mul(x_tile, s)
        pto.store(y_tile, out)

    @workload
    def wl():
        scale(x_tile=x, y_tile=y, s=2.0)

    program = wl().named("pto_runtime_scale_a2a3_smoke").compile(target="pto_runtime_a2a3")
    program.execute()
    program.synchronize()

    assert np.allclose(y.data, x.data * 2.0)
