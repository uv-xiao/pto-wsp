import math
import numpy as np

from pto_wsp import (
    kernel,
    pto,
    In,
    Out,
    Tile,
    workload,
    P,
    Dense,
    Tensor,
    DType,
    DispatchPolicy,
    TaskWindow,
    WindowMode,
)


def run_tiled_attention(q_base: np.ndarray, k_base: np.ndarray, v_base: np.ndarray, *, tile_seq: int) -> tuple[np.ndarray, int]:
    f32 = DType.F32

    if q_base.shape != k_base.shape or q_base.shape != v_base.shape:
        raise ValueError("Q/K/V must have the same shape")
    if q_base.ndim != 4:
        raise ValueError("Q/K/V must be [B,H,S,D]")

    batch, heads_n, seq, d = q_base.shape
    if seq % tile_seq:
        raise ValueError("seq must be divisible by tile_seq")
    seq_tiles_n = seq // tile_seq

    out_base = np.zeros_like(q_base, dtype=np.float32)

    q_tiles = q_base.reshape(batch, heads_n, seq_tiles_n, tile_seq, d).astype(np.float32)
    k_tiles_t = k_base.reshape(batch, heads_n, seq_tiles_n, tile_seq, d).transpose(0, 1, 2, 4, 3).astype(np.float32)
    v_tiles = v_base.reshape(batch, heads_n, seq_tiles_n, tile_seq, d).astype(np.float32)
    out_tiles = out_base.reshape(batch, heads_n, seq_tiles_n, tile_seq, d)

    Q = Tensor(data=q_tiles, shape=q_tiles.shape, dtype=f32)
    K_t = Tensor(data=k_tiles_t, shape=k_tiles_t.shape, dtype=f32)
    V = Tensor(data=v_tiles, shape=v_tiles.shape, dtype=f32)
    O = Tensor(data=out_tiles, shape=out_tiles.shape, dtype=f32)

    batch_ax = Dense[batch]()
    heads_ax = Dense[heads_n]()
    seq_tiles_ax = Dense[seq_tiles_n]()

    @kernel
    def attention_tile(
        q: In[Tile[tile_seq, d, f32]],
        k_t: In[Tile[d, tile_seq, f32]],
        v: In[Tile[tile_seq, d, f32]],
        out: Out[Tile[tile_seq, d, f32]],
    ):
        scores = pto.matmul(q, k_t)
        scale = pto.constant(1.0 / math.sqrt(d), f32)
        scores = pto.mul(scores, scale)

        max_s = pto.rowmax(scores)
        scores = pto.sub(scores, max_s)
        exp_s = pto.exp(scores)
        sum_s = pto.rowsum(exp_s)
        w = pto.div(exp_s, sum_s)

        out_tile = pto.matmul(w, v)
        pto.store(out, out_tile)

    @workload
    def attention():
        for b, h in P(batch_ax, heads_ax):
            for s in P(seq_tiles_ax):
                attention_tile[b, h, s](q=Q[b][h][s], k_t=K_t[b][h][s], v=V[b][h][s], out=O[b][h][s])

    program = (
        attention()
        .dispatch(DispatchPolicy.round_robin(num_aicpus=4))
        .task_graph(window=TaskWindow(8192, "tasks", WindowMode.STALL))
        .compile(target="cpu_sim")
    )
    program.execute()
    program.synchronize()

    stats = program.stats() if callable(program.stats) else program.stats
    total_cycles = int(getattr(stats, "total_cycles", 0) or 0)
    return out_base, total_cycles

