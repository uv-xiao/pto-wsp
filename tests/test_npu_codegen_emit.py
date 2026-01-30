import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from pto_wsp import DType, In, Out, Scalar, Tensor, Tile, kernel, pto, workload, ptoisa, ptoisa_kernel


def test_ascend_npu_codegen_emits_source_tree():
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

    program = wl().named("npu_codegen_smoke").compile(target="ascend_npu")
    assert program.using_cpp_backend

    artifact_dir = program.codegen_artifact_dir
    assert artifact_dir
    assert os.path.isdir(artifact_dir)

    expected = [
        "CMakeLists.txt",
        os.path.join("host", "runner_stub.cpp"),
        os.path.join("aicpu", "expand.cpp"),
        os.path.join("aicore", "dispatch.cpp"),
    ]
    for rel in expected:
        assert os.path.exists(os.path.join(artifact_dir, rel))

    with open(os.path.join(artifact_dir, "CMakeLists.txt"), "r", encoding="utf-8") as f:
        cm = f.read()
    assert "PTO_WSP_ENABLE_ASCEND" in cm
    assert "pto_wsp_npu_aicore" in cm

    # Codegen-only: cannot execute without real Ascend runtime/toolchain.
    try:
        program.execute()
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected RuntimeError when executing codegen-only target")


def test_ascend_npu_codegen_emits_custom_kernel_source():
    x_data = np.arange(16, dtype=np.float32).reshape(4, 4)
    y_data = np.zeros((4, 4), dtype=np.float32)

    x = Tensor(data=x_data, shape=x_data.shape, dtype=DType.F32)
    y = Tensor(data=y_data, shape=y_data.shape, dtype=DType.F32)

    @kernel(
        cpp_src=r"""
  // MARK_CUSTOM_KERNEL
  float* x_ptr = (float*)task->tensor_ptrs[0];
  float* y_ptr = (float*)task->tensor_ptrs[1];
  const uint64_t xs3 = task->tensor_strides[0];
  const uint64_t xs4 = task->tensor_strides[1];
  const uint64_t ys3 = task->tensor_strides[2];
  const uint64_t ys4 = task->tensor_strides[3];
  for (int r = 0; r < 4; ++r) {
    for (int c = 0; c < 4; ++c) {
      y_ptr[r * (int64_t)ys3 + c * (int64_t)ys4] = x_ptr[r * (int64_t)xs3 + c * (int64_t)xs4];
    }
  }
""",
    )
    def custom_copy(x_tile: In[Tile[4, 4, DType.F32]], y_tile: Out[Tile[4, 4, DType.F32]]):
        pass

    @workload
    def wl():
        custom_copy(x_tile=x, y_tile=y)

    program = wl().named("npu_codegen_custom_kernel").compile(target="ascend_npu")
    artifact_dir = program.codegen_artifact_dir
    assert artifact_dir
    kernel_cpp = os.path.join(artifact_dir, "aicore", "kernel_custom_copy.cpp")
    assert os.path.exists(kernel_cpp)
    with open(kernel_cpp, "r", encoding="utf-8") as f:
        src = f.read()
    assert "Custom kernel (Path A)" in src
    assert "MARK_CUSTOM_KERNEL" in src


def test_ascend_npu_codegen_emits_ptoisa_kernel_source():
    x_data = np.arange(16, dtype=np.float32).reshape(4, 4)
    y_data = np.zeros((4, 4), dtype=np.float32)

    x = Tensor(data=x_data, shape=x_data.shape, dtype=DType.F32)
    y = Tensor(data=y_data, shape=y_data.shape, dtype=DType.F32)

    @ptoisa_kernel
    def square(src: In[Tile[4, 4, DType.F32]], dst: Out[Tile[4, 4, DType.F32]]):
        a = ptoisa.tload(src)
        b = ptoisa.TMUL(a, a)
        ptoisa.tstore(dst, b)

    @workload
    def wl():
        square(src=x, dst=y)

    program = wl().named("npu_codegen_ptoisa_kernel").compile(target="ascend_npu")
    artifact_dir = program.codegen_artifact_dir
    assert artifact_dir
    kernel_cpp = os.path.join(artifact_dir, "aicore", "kernel_square.cpp")
    assert os.path.exists(kernel_cpp)
    with open(kernel_cpp, "r", encoding="utf-8") as f:
        src = f.read()
    assert "pto_wsp::ptoisa::TLOAD" in src
    assert "pto_wsp::ptoisa::TMUL" in src
    assert "pto_wsp::ptoisa::TSTORE" in src


def test_ascend_npu_codegen_emits_cpp_body_path_kernel_source():
    x_data = np.arange(16, dtype=np.float32).reshape(4, 4)
    y_data = np.zeros((4, 4), dtype=np.float32)

    x = Tensor(data=x_data, shape=x_data.shape, dtype=DType.F32)
    y = Tensor(data=y_data, shape=y_data.shape, dtype=DType.F32)

    @kernel(
        cpp_body_path=os.path.join(os.path.dirname(__file__), "assets", "cpp_body_kernel_snippet.cpp"),
    )
    def body_copy(x_tile: In[Tile[4, 4, DType.F32]], y_tile: Out[Tile[4, 4, DType.F32]]):
        pass

    @workload
    def wl():
        body_copy(x_tile=x, y_tile=y)

    program = wl().named("npu_codegen_cpp_body_path").compile(target="ascend_npu")
    artifact_dir = program.codegen_artifact_dir
    assert artifact_dir
    kernel_cpp = os.path.join(artifact_dir, "aicore", "kernel_body_copy.cpp")
    assert os.path.exists(kernel_cpp)
    with open(kernel_cpp, "r", encoding="utf-8") as f:
        src = f.read()
    assert "MARK_CPP_BODY_KERNEL_SNIPPET" in src


def test_ascend_npu_codegen_emits_cpp_tu_path_kernel_source():
    x_data = np.arange(16, dtype=np.float32).reshape(4, 4)
    y_data = np.zeros((4, 4), dtype=np.float32)

    x = Tensor(data=x_data, shape=x_data.shape, dtype=DType.F32)
    y = Tensor(data=y_data, shape=y_data.shape, dtype=DType.F32)

    # Full translation unit with the kernel definition (no wrapper generation).
    tu_src = r'''
#include "pto/wsp/codegen/abi/kernel_abi.hpp"
#include "pto/wsp/codegen/abi/ptoisa_bridge.hpp"
#include <cstdint>
#include <cstddef>

extern "C" PTO_WSP_KERNEL_ATTR uint64_t tu_copy(const KernelTaskDesc* task, CSPTContext* cspt) {
  (void)cspt;
  float* x_ptr = (float*)task->tensor_ptrs[0];
  float* y_ptr = (float*)task->tensor_ptrs[1];
  const uint64_t xs3 = task->tensor_strides[0];
  const uint64_t xs4 = task->tensor_strides[1];
  const uint64_t ys3 = task->tensor_strides[2];
  const uint64_t ys4 = task->tensor_strides[3];
  for (int r = 0; r < 4; ++r) {
    for (int c = 0; c < 4; ++c) {
      y_ptr[r * (int64_t)ys3 + c * (int64_t)ys4] = x_ptr[r * (int64_t)xs3 + c * (int64_t)xs4];
    }
  }
  return 0;
}
'''

    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "tu_copy.cpp")
        with open(p, "w", encoding="utf-8") as f:
            f.write(tu_src)

        @kernel(cpp_tu_path=p)
        def tu_copy(x_tile: In[Tile[4, 4, DType.F32]], y_tile: Out[Tile[4, 4, DType.F32]]):
            pass

        @workload
        def wl():
            tu_copy(x_tile=x, y_tile=y)

        program = wl().named("npu_codegen_cpp_tu_path").compile(target="ascend_npu")
        artifact_dir = program.codegen_artifact_dir
        assert artifact_dir
        kernel_cpp = os.path.join(artifact_dir, "aicore", "kernel_tu_copy.cpp")
        assert os.path.exists(kernel_cpp)
        with open(kernel_cpp, "r", encoding="utf-8") as f:
            src = f.read()
        assert "extern \"C\" PTO_WSP_KERNEL_ATTR uint64_t tu_copy" in src


def test_ascend_npu_codegen_emits_slot_ops_and_plan_storage():
    from pto_wsp.primitives import cond, sequential, slot_load_u64, slot_set_u64
    from pto_wsp.scalar_expr import slot_u64
    from pto_wsp.workload import Workload

    x_data = np.zeros((1, 1, 1, 1), dtype=np.int32)
    y_data = np.zeros((1, 1, 1, 1), dtype=np.int32)

    x = Tensor(data=x_data, shape=x_data.shape, dtype=DType.I32)
    y = Tensor(data=y_data, shape=y_data.shape, dtype=DType.I32)

    @kernel
    def nop(src: In[Tile[1, 1, DType.I32]], dst: Out[Tile[1, 1, DType.I32]]):
        pto.store(dst, src)

    # Force slot ops into the emitted AICPU expander:
    # - load a tensor element into slot 0
    # - set slot 1 to a constant
    # - branch on slot 0
    w = sequential(
        slot_load_u64(0, x[0][0], row=0, col=0),
        slot_set_u64(1, 123),
        cond(slot_u64(0) != 0, Workload("task", kernel=nop, params=[0], resources={"src": x[0][0], "dst": y[0][0]}), Workload("task", kernel=nop, params=[0], resources={"src": x[0][0], "dst": y[0][0]})),
    ).named("npu_codegen_slots_smoke")

    program = w.compile(target="ascend_npu")
    artifact_dir = program.codegen_artifact_dir
    assert artifact_dir
    expand_cpp = os.path.join(artifact_dir, "aicpu", "expand.cpp")
    with open(expand_cpp, "r", encoding="utf-8") as f:
        src = f.read()

    assert "PTO_WSP_NPU_MAX_SLOTS" in src or "slots_u64" in src
    assert "plan->slots_u64" in src
