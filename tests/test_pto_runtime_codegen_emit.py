import os
import sys

import numpy as np


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))


from pto_wsp import DType, In, Out, Scalar, Tensor, Tile, kernel, pto, workload


def test_a2a3sim_codegen_emits_pto_runtime_tree():
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
        scale(x_tile=x, y_tile=y, s=1.0)

    program = wl().named("pto_runtime_emit_smoke").compile(target="a2a3sim_codegen")
    assert program.using_cpp_backend
    assert program.codegen_artifact_dir

    artifact_dir = program.codegen_artifact_dir
    expected = [
        os.path.join("kernels", "kernel_config.py"),
        os.path.join("kernels", "orchestration", "pto_wsp_orch.cpp"),
    ]
    for rel in expected:
        assert os.path.exists(os.path.join(artifact_dir, rel))

    with open(os.path.join(artifact_dir, "kernels", "orchestration", "pto_wsp_orch.cpp"), "r", encoding="utf-8") as f:
        src = f.read()
    assert "TODO_PTO_RUNTIME_MULTI_AICPU_DISPATCH" in src

