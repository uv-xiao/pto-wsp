import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from pto_wsp import DType, Dense, DispatchPolicy, In, Out, P, TaskWindow, Tensor, Tile, WindowMode, kernel, pto, workload


def test_npu_codegen_preserves_dispatch_and_task_window_in_sources():
    tiles = Dense[2]()
    a_np = np.arange(2 * 4 * 4, dtype=np.float32).reshape(2, 4, 4)
    b_np = np.zeros_like(a_np)

    a = Tensor(data=a_np, shape=a_np.shape, dtype=DType.F32)
    b = Tensor(data=b_np, shape=b_np.shape, dtype=DType.F32)

    @kernel
    def copy_tile(src: In[Tile[4, 4, DType.F32]], dst: Out[Tile[4, 4, DType.F32]]):
        pto.store(dst, src)

    @workload
    def wl():
        for t in P(tiles):
            copy_tile[t](src=a[t], dst=b[t])

    program = (
        wl()
        .named("npu_sched_preserve")
        .dispatch(DispatchPolicy.round_robin(4))
        .task_graph(window=TaskWindow(16, "tasks", WindowMode.STALL))
        .compile(target="ascend_npu")
    )
    assert program.using_cpp_backend
    artifact_dir = program.codegen_artifact_dir
    assert artifact_dir is not None

    expand_cpp = os.path.join(artifact_dir, "aicpu", "expand.cpp")
    assert os.path.exists(expand_cpp)
    with open(expand_cpp, "r", encoding="utf-8") as f:
        src = f.read()

    # Plan-level schedule fields preserved.
    assert "plan->dispatch_policy" in src
    assert "plan->dispatch_num_targets" in src
    assert "plan->task_window_tasks" in src

    # Task-level dispatch assignment preserved.
    assert "task->target_id" in src

