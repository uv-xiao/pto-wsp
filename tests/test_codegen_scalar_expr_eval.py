import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from pto_wsp import DType, Dense, DispatchPolicy, In, Out, P, Tensor, Tile, kernel, pto, workload
from pto_wsp.scalar_expr import symbol_u64


def test_scalar_expr_dispatch_reads_task_params_tags_and_symbols_without_rebuild():
    tiles = Dense[6]()
    a_np = np.arange(6 * 4 * 4, dtype=np.float32).reshape(6, 4, 4)
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

    b_np[:] = 0
    p0 = wl().named("scalar_expr_serial").compile(target="cpu_sim")
    p0.execute()
    p0.synchronize()
    np.testing.assert_allclose(b_np, a_np, rtol=0, atol=0)
    cycles_serial = int(p0.stats.total_cycles)
    assert cycles_serial > 0
    assert cycles_serial % 6 == 0
    cycles_per_task = cycles_serial // 6

    # dispatch affinity uses a ScalarExpr key; v9 codegen evaluates it in the artifact.
    #
    # t.tags["mask"] is modeled as a u64 tag lookup (v9: implemented via symbol_u64(tag_id)).
    # With mask=0, all tasks map to worker0 => makespan = 6*cycles_per_task.
    # With mask=1, tasks split by parity => makespan = 3*cycles_per_task.
    b_np[:] = 0
    p1 = (
        wl()
        .named("scalar_expr_dispatch_tags_params")
        .dispatch(DispatchPolicy.affinity(lambda t: (t.tags["mask"] & t.params[0].cast_u64()), num_aicpus=2))
        .compile(target="cpu_sim")
    )

    p1.set_symbol_u64("mask", 0)
    p1.execute()
    p1.synchronize()
    np.testing.assert_allclose(b_np, a_np, rtol=0, atol=0)
    assert int(p1.stats.total_cycles) == 6 * cycles_per_task

    b_np[:] = 0
    p1.set_symbol_u64("mask", 1)
    p1.execute()
    p1.synchronize()
    np.testing.assert_allclose(b_np, a_np, rtol=0, atol=0)
    assert int(p1.stats.total_cycles) == 3 * cycles_per_task

    # Also exercise ternary + comparisons + casts (still no rebuild).
    b_np[:] = 0
    p2 = (
        wl()
        .named("scalar_expr_dispatch_ternary")
        .dispatch(
            DispatchPolicy.affinity(
                lambda t: (t.params[0] == symbol_u64("special").cast_i64()).where(0, 1).cast_u64(),
                num_aicpus=2,
            )
        )
        .compile(target="cpu_sim")
    )

    p2.set_symbol_u64("special", 0)
    p2.execute()
    p2.synchronize()
    np.testing.assert_allclose(b_np, a_np, rtol=0, atol=0)
    assert int(p2.stats.total_cycles) == 5 * cycles_per_task

    b_np[:] = 0
    p2.set_symbol_u64("special", 7)
    p2.execute()
    p2.synchronize()
    np.testing.assert_allclose(b_np, a_np, rtol=0, atol=0)
    assert int(p2.stats.total_cycles) == 6 * cycles_per_task

