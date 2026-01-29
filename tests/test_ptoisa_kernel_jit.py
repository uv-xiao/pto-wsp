from __future__ import annotations

import numpy as np

from pto_wsp import (
    Dense,
    DType,
    In,
    Out,
    P,
    Tensor,
    Tile,
    DispatchPolicy,
    TaskWindow,
    WindowMode,
    ptoisa,
    ptoisa_kernel,
    workload,
)


def test_ptoisa_kernel_jit_square_tile_cpu_sim():
    tiles = Dense[4]()

    x_np = np.random.default_rng(0).standard_normal((4, 1, 32, 32), dtype=np.float32)
    y_np = np.zeros_like(x_np, dtype=np.float32)

    X = Tensor(data=x_np, shape=x_np.shape, dtype=DType.F32)
    Y = Tensor(data=y_np, shape=y_np.shape, dtype=DType.F32)

    @ptoisa_kernel
    def square(src: In[Tile[32, 32, DType.F32]], dst: Out[Tile[32, 32, DType.F32]]):
        a = ptoisa.tload(src)
        b = ptoisa.TMUL(a, a)
        ptoisa.tstore(dst, b)

    @workload
    def wl():
        for t in P(tiles):
            square[t](src=X[t][0], dst=Y[t][0])

    program = (
        wl()
        .dispatch(DispatchPolicy.round_robin(num_aicpus=2))
        .task_graph(window=TaskWindow(2048, "tasks", WindowMode.STALL))
        .compile(target="cpu_sim")
    )
    program.execute()
    program.synchronize()

    ref = x_np * x_np
    assert np.allclose(y_np, ref, rtol=0, atol=0)
    stats = program.stats() if callable(program.stats) else program.stats
    assert int(getattr(stats, "total_cycles", 0) or 0) > 0


def test_ptoisa_kernel_jit_tsort32_topk_indices_cpu_sim():
    batch = Dense[2]()
    tiles = Dense[3]()

    rng = np.random.default_rng(0)
    scores_np = rng.standard_normal((2, 3, 1, 32), dtype=np.float32)
    out_np = np.zeros((2, 3, 1, 8), dtype=np.int32)

    Scores = Tensor(data=scores_np, shape=scores_np.shape, dtype=DType.F32)
    Out = Tensor(data=out_np, shape=out_np.shape, dtype=DType.I32)

    @ptoisa_kernel
    def topk8(scores: In[Tile[1, 32, DType.F32]], out_idx: Out[Tile[1, 8, DType.I32]]):
        s = ptoisa.tload(scores)
        idx = ptoisa.iota_u32(32)
        pairs = ptoisa.TSORT32(s, idx)
        ptoisa.store_topk_indices_i32(out_idx, pairs, 8)

    @workload
    def wl():
        for b in P(batch):
            for t in P(tiles):
                topk8[b, t](scores=Scores[b][t], out_idx=Out[b][t])

    program = (
        wl()
        .dispatch(DispatchPolicy.round_robin(num_aicpus=2))
        .task_graph(window=TaskWindow(4096, "tasks", WindowMode.STALL))
        .compile(target="cpu_sim")
    )
    program.execute()
    program.synchronize()

    ref = np.argsort(-scores_np.reshape(2, 3, 32), axis=-1)[:, :, :8].astype(np.int32)
    got = out_np.reshape(2, 3, 8)
    assert np.array_equal(got, ref)

    stats = program.stats() if callable(program.stats) else program.stats
    assert int(getattr(stats, "total_cycles", 0) or 0) > 0
