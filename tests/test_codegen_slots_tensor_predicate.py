import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from pto_wsp import DType, Dense, In, Out, Tensor, Tile, kernel
from pto_wsp.primitives import cond, sequential, slot_load_u64
from pto_wsp.scalar_expr import slot_u64
from pto_wsp.workload import Workload


def test_codegen_slot_load_from_tensor_drives_cond_without_rebuild():
    batch = Dense[1]()

    x_np = np.zeros((1, 1, 1, 1), dtype=np.float32)
    flag_np = np.zeros((1, 1, 1, 1), dtype=np.int32)
    y_np = np.zeros((1, 1, 1, 1), dtype=np.int32)

    x = Tensor(data=x_np, shape=x_np.shape, dtype=DType.F32)
    flag = Tensor(data=flag_np, shape=flag_np.shape, dtype=DType.I32)
    y = Tensor(data=y_np, shape=y_np.shape, dtype=DType.I32)

    @kernel(
        cpp_src=r"""
  float* x_ptr = (float*)task->tensor_ptrs[0];
  int32_t* flag_ptr = (int32_t*)task->tensor_ptrs[1];
  const uint64_t x_s3 = task->tensor_strides[0];
  const uint64_t x_s4 = task->tensor_strides[1];
  const uint64_t f_s3 = task->tensor_strides[2];
  const uint64_t f_s4 = task->tensor_strides[3];
  const float v = x_ptr[0 * (int64_t)x_s3 + 0 * (int64_t)x_s4];
  flag_ptr[0 * (int64_t)f_s3 + 0 * (int64_t)f_s4] = (v > 0.0f) ? 1 : 0;
#if defined(__CPU_SIM)
  pto::cpu_sim::add_cycles(1);
#endif
""",
    )
    def write_flag(x_in: In[Tile[1, 1, DType.F32]], flag_out: Out[Tile[1, 1, DType.I32]]):
        pass

    @kernel(
        cpp_src=r"""
  int32_t* y_ptr = (int32_t*)task->tensor_ptrs[0];
  const uint64_t y_s3 = task->tensor_strides[0];
  const uint64_t y_s4 = task->tensor_strides[1];
  y_ptr[0 * (int64_t)y_s3 + 0 * (int64_t)y_s4] = 7;
#if defined(__CPU_SIM)
  pto::cpu_sim::add_cycles(1);
#endif
""",
    )
    def fill_true(y_out: Out[Tile[1, 1, DType.I32]]):
        pass

    @kernel(
        cpp_src=r"""
  int32_t* y_ptr = (int32_t*)task->tensor_ptrs[0];
  const uint64_t y_s3 = task->tensor_strides[0];
  const uint64_t y_s4 = task->tensor_strides[1];
  y_ptr[0 * (int64_t)y_s3 + 0 * (int64_t)y_s4] = 3;
#if defined(__CPU_SIM)
  pto::cpu_sim::add_cycles(1);
#endif
""",
    )
    def fill_false(y_out: Out[Tile[1, 1, DType.I32]]):
        pass

    def build_workload():
        # Sequential:
        # 1) write flag tensor from x
        # 2) load flag[0,0] -> slot 0
        # 3) branch on slot 0
        write = Workload("task", kernel=write_flag, params=[0], resources={"x_in": x[0][0], "flag_out": flag[0][0]})
        load = slot_load_u64(0, flag[0][0], row=0, col=0)
        t = Workload("task", kernel=fill_true, params=[0], resources={"y_out": y[0][0]})
        f = Workload("task", kernel=fill_false, params=[0], resources={"y_out": y[0][0]})
        br = cond(slot_u64(0) != 0, t, f)
        return sequential(write, load, br).named("slot_tensor_predicate")

    program = build_workload().compile(target="cpu_sim")

    # Run 1: x <= 0 -> false branch.
    x_np.fill(-1.0)
    flag_np.fill(0)
    y_np.fill(0)
    program.execute()
    program.synchronize()
    assert int(flag_np[0, 0, 0, 0]) == 0
    assert int(y_np[0, 0, 0, 0]) == 3

    # Run 2: x > 0 -> true branch (same compiled artifact).
    x_np.fill(1.0)
    flag_np.fill(0)
    y_np.fill(0)
    program.execute()
    program.synchronize()
    assert int(flag_np[0, 0, 0, 0]) == 1
    assert int(y_np[0, 0, 0, 0]) == 7
