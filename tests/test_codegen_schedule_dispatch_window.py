import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from pto_wsp import DType, Dense, DispatchPolicy, In, Out, P, TaskWindow, Tensor, Tile, WindowMode, kernel, pto, workload


def test_dispatch_and_task_window_change_cycles_semantics():
    tiles = Dense[4]()
    a_np = np.arange(4 * 4 * 4, dtype=np.float32).reshape(4, 4, 4)
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

    # Baseline: single worker => makespan == sum(kernel_cycles).
    b_np[:] = 0
    p0 = wl().named("sched_baseline").compile(target="cpu_sim")
    p0.execute()
    p0.synchronize()
    np.testing.assert_allclose(b_np, a_np, rtol=0, atol=0)
    cycles_serial = int(p0.stats.total_cycles)
    assert cycles_serial > 0

    # Dispatch round_robin(2): makespan should drop ~2x for uniform tasks.
    b_np[:] = 0
    p1 = wl().named("sched_rr2").dispatch(DispatchPolicy.round_robin(2)).compile(target="cpu_sim")
    p1.execute()
    p1.synchronize()
    np.testing.assert_allclose(b_np, a_np, rtol=0, atol=0)
    cycles_rr2 = int(p1.stats.total_cycles)
    assert cycles_rr2 > 0
    assert cycles_rr2 * 2 == cycles_serial

    # TaskWindow(size=1, STALL): forces serial issue even with multiple workers.
    b_np[:] = 0
    p2 = (
        wl()
        .named("sched_rr2_w1")
        .dispatch(DispatchPolicy.round_robin(2))
        .task_graph(window=TaskWindow(1, "tasks", WindowMode.STALL))
        .compile(target="cpu_sim")
    )
    p2.execute()
    p2.synchronize()
    np.testing.assert_allclose(b_np, a_np, rtol=0, atol=0)
    cycles_rr2_w1 = int(p2.stats.total_cycles)
    assert cycles_rr2_w1 == cycles_serial

