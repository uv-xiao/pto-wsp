import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from pto_wsp import DType, Dense, In, Out, P, Tensor, Tile, kernel, workload


def test_codegen_custom_cpp_kernel_compiles_and_runs_cpu_sim():
    batch = Dense[1]()

    x_np = np.arange(32, dtype=np.float32).reshape(1, 1, 1, 32)
    y_np = np.zeros((1, 1, 1, 32), dtype=np.float32)

    x = Tensor(data=x_np, shape=x_np.shape, dtype=DType.F32)
    y = Tensor(data=y_np, shape=y_np.shape, dtype=DType.F32)

    @kernel(
        cpp_src=r"""
  // Copy a 1xN vector tile using PTO-ISA primitives (CPU-sim cycle counted).
  float* x_ptr = (float*)task->tensor_ptrs[0];
  float* y_ptr = (float*)task->tensor_ptrs[1];

  const int vr = 1;
  const int vc = 32;

  using Shape = pto::Shape<1, 1, 1, pto::DYNAMIC, pto::DYNAMIC>;
  using Stride = pto::Stride<1, 1, 1, pto::DYNAMIC, pto::DYNAMIC>;

  pto::GlobalTensor<float, Shape, Stride, pto::Layout::ND> gx(
      x_ptr, Shape(vr, vc), Stride((int)task->tensor_strides[0], (int)task->tensor_strides[1]));
  pto::GlobalTensor<float, Shape, Stride, pto::Layout::ND> gy(
      y_ptr, Shape(vr, vc), Stride((int)task->tensor_strides[2], (int)task->tensor_strides[3]));

  pto::Tile<pto::TileType::Vec, float, 1, 32, pto::BLayout::RowMajor, pto::DYNAMIC, pto::DYNAMIC> t(
      (size_t)vr, (size_t)vc);

  pto_wsp::ptoisa::TLOAD(t, gx);
  pto_wsp::ptoisa::TSTORE(gy, t);
""",
    )
    def copy_vec(inp: In[Tile[1, 32, DType.F32]], out: Out[Tile[1, 32, DType.F32]]):
        # C++-implemented kernel: body intentionally empty.
        pass

    @workload
    def wl():
        for b in P(batch):
            copy_vec[b](inp=x[b][0], out=y[b][0])

    program = wl().named("custom_cpp_kernel_copy").compile(target="cpu_sim")
    program.execute()
    program.synchronize()

    np.testing.assert_allclose(y_np, x_np, rtol=0, atol=0)
    assert int(program.stats.total_cycles) > 0

