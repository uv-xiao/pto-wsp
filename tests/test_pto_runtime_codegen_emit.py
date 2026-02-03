import os
import sys

import numpy as np


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))


from pto_wsp import DType, In, Out, Scalar, Tensor, Tile, kernel, pto, workload


def _compile_scale(target: str):
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

    program = wl().named("pto_runtime_emit_smoke").compile(target=target)
    assert program.using_cpp_backend
    assert program.codegen_artifact_dir
    return program.codegen_artifact_dir


def test_a2a3sim_codegen_emits_pto_runtime_tree():
    artifact_dir = _compile_scale("a2a3sim_codegen")

    expected = [
        os.path.join("kernels", "kernel_config.py"),
        os.path.join("kernels", "orchestration", "pto_wsp_orch.cpp"),
    ]
    for rel in expected:
        assert os.path.exists(os.path.join(artifact_dir, rel))

    with open(os.path.join(artifact_dir, "kernels", "orchestration", "pto_wsp_orch.cpp"), "r", encoding="utf-8") as f:
        src = f.read()
    assert "TODO_PTO_RUNTIME_MULTI_AICPU_DISPATCH" in src


def test_a2a3sim_codegen_emits_sim_kernels():
    artifact_dir = _compile_scale("a2a3sim_codegen")
    k = os.path.join(artifact_dir, "kernels", "aiv_sim", "kernel_scale.cpp")
    assert os.path.exists(k)
    with open(k, "r", encoding="utf-8") as f:
        s = f.read()
    assert "#include <pto/pto-inst.hpp>" not in s
    assert 'extern "C" void kernel_scale' in s


def test_a2a3sim_codegen_orchestration_is_not_stub():
    artifact_dir = _compile_scale("a2a3sim_codegen")
    p = os.path.join(artifact_dir, "kernels", "orchestration", "pto_wsp_orch.cpp")
    with open(p, "r", encoding="utf-8") as f:
        src = f.read()
    import re

    assert re.search(r"^[^/\n]*runtime->add_task\(", src, flags=re.MULTILINE)
