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


def run_llama_block(x_base: np.ndarray, w1_base: np.ndarray, w2_base: np.ndarray, *, tile_seq: int, eps: float) -> tuple[np.ndarray, int]:
    f32 = DType.F32

    seq, d = x_base.shape
    if seq % tile_seq:
        raise ValueError("seq must be divisible by tile_seq")
    seq_tiles_n = seq // tile_seq
    mlp_dim = w1_base.shape[1]
    if w1_base.shape != (d, mlp_dim) or w2_base.shape != (mlp_dim, d):
        raise ValueError("weight shapes mismatch")

    out_base = np.zeros_like(x_base, dtype=np.float32)

    x_tiles = x_base.reshape(seq_tiles_n, tile_seq, d).astype(np.float32)
    out_tiles = out_base.reshape(seq_tiles_n, tile_seq, d)
    k_tiles_t = x_tiles.transpose(0, 2, 1)

    X = Tensor(data=x_tiles, shape=x_tiles.shape, dtype=f32)
    K_t = Tensor(data=k_tiles_t, shape=k_tiles_t.shape, dtype=f32)
    W1 = Tensor(data=w1_base.astype(np.float32), shape=w1_base.shape, dtype=f32)
    W2 = Tensor(data=w2_base.astype(np.float32), shape=w2_base.shape, dtype=f32)
    OutT = Tensor(data=out_tiles, shape=out_tiles.shape, dtype=f32)

    Xn = Tensor(data=np.zeros_like(x_tiles), shape=x_tiles.shape, dtype=f32)
    Attn = Tensor(data=np.zeros_like(x_tiles), shape=x_tiles.shape, dtype=f32)
    Res1 = Tensor(data=np.zeros_like(x_tiles), shape=x_tiles.shape, dtype=f32)
    Mlp = Tensor(data=np.zeros_like(x_tiles), shape=x_tiles.shape, dtype=f32)

    seq_tiles = Dense[seq_tiles_n]()

    @kernel
    def rmsnorm_tile(
        x: In[Tile[tile_seq, d, f32]],
        out: Out[Tile[tile_seq, d, f32]],
    ):
        x_sq = pto.mul(x, x)
        mean_sq = pto.rowmean(x_sq)
        eps_v = pto.constant(eps, f32)
        mean_sq_eps = pto.add(mean_sq, eps_v)
        inv = pto.rsqrt(mean_sq_eps)
        y = pto.mul(x, inv)
        pto.store(out, y)

    @kernel
    def attn_tile(
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

        out_v = pto.matmul(w, v)
        pto.store(out, out_v)

    @kernel
    def add_tile(
        a: In[Tile[tile_seq, d, f32]],
        b: In[Tile[tile_seq, d, f32]],
        out: Out[Tile[tile_seq, d, f32]],
    ):
        y = pto.add(a, b)
        pto.store(out, y)

    @kernel
    def mlp_tile(
        x: In[Tile[tile_seq, d, f32]],
        w1: In[Tile[d, mlp_dim, f32]],
        w2: In[Tile[mlp_dim, d, f32]],
        out: Out[Tile[tile_seq, d, f32]],
    ):
        h = pto.matmul(x, w1)
        relu = pto.max(h, pto.constant(0.0, f32))
        y = pto.matmul(relu, w2)
        pto.store(out, y)

    @workload
    def llama_block():
        for s in P(seq_tiles):
            attn_tile[s](q=X[s], k_t=K_t[s], v=X[s], out=Attn[s])
            add_tile[s](a=X[s], b=Attn[s], out=Res1[s])
            rmsnorm_tile[s](x=Res1[s], out=Xn[s])
            mlp_tile[s](x=Xn[s], w1=W1, w2=W2, out=Mlp[s])
            add_tile[s](a=Res1[s], b=Mlp[s], out=OutT[s])

    program = (
        llama_block()
        .dispatch(DispatchPolicy.round_robin(num_aicpus=4))
        .task_graph(window=TaskWindow(8192, "tasks", WindowMode.STALL))
        .compile(target="cpu_sim")
    )
    program.execute()
    program.synchronize()

    stats = program.stats() if callable(program.stats) else program.stats
    total_cycles = int(getattr(stats, "total_cycles", 0) or 0)
    return out_base, total_cycles

