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


def run_moe_block(x_base: np.ndarray, w0: np.ndarray, w1: np.ndarray, g_base: np.ndarray) -> tuple[np.ndarray, int]:
    f32 = DType.F32

    if x_base.ndim != 2:
        raise ValueError("x must be [SEQ,D]")
    seq, d = x_base.shape
    if w0.shape != (d, d) or w1.shape != (d, d):
        raise ValueError("w0/w1 must be [D,D]")
    if g_base.shape != (seq, 1):
        raise ValueError("g must be [SEQ,1]")

    tile_seq = 8
    if seq % tile_seq:
        raise ValueError("seq must be divisible by 8 for this example")
    seq_tiles_n = seq // tile_seq

    x_tiles = x_base.reshape(seq_tiles_n, tile_seq, d).astype(np.float32)
    g_tiles = g_base.reshape(seq_tiles_n, tile_seq, 1).astype(np.float32)
    y_base = np.zeros_like(x_base, dtype=np.float32)
    y_tiles = y_base.reshape(seq_tiles_n, tile_seq, d)

    X = Tensor(data=x_tiles, shape=x_tiles.shape, dtype=f32)
    G = Tensor(data=g_tiles, shape=g_tiles.shape, dtype=f32)
    W0 = Tensor(data=w0.astype(np.float32), shape=w0.shape, dtype=f32)
    W1 = Tensor(data=w1.astype(np.float32), shape=w1.shape, dtype=f32)
    Y = Tensor(data=y_tiles, shape=y_tiles.shape, dtype=f32)

    E0 = Tensor(data=np.zeros_like(x_tiles), shape=x_tiles.shape, dtype=f32)
    E1 = Tensor(data=np.zeros_like(x_tiles), shape=x_tiles.shape, dtype=f32)

    seq_tiles = Dense[seq_tiles_n]()

    @kernel
    def expert_linear(
        x: In[Tile[tile_seq, d, f32]],
        w: In[Tile[d, d, f32]],
        out: Out[Tile[tile_seq, d, f32]],
    ):
        y = pto.matmul(x, w)
        pto.store(out, y)

    @kernel
    def moe_mix(
        e0: In[Tile[tile_seq, d, f32]],
        e1: In[Tile[tile_seq, d, f32]],
        g: In[Tile[tile_seq, 1, f32]],
        out: Out[Tile[tile_seq, d, f32]],
    ):
        g1 = pto.mul(g, pto.constant(-1.0, f32))
        g1 = pto.add(g1, pto.constant(1.0, f32))
        y0 = pto.mul(e0, g)
        y1 = pto.mul(e1, g1)
        y = pto.add(y0, y1)
        pto.store(out, y)

    @workload
    def moe_block():
        for s in P(seq_tiles):
            expert_linear[s](x=X[s], w=W0, out=E0[s])
            expert_linear[s](x=X[s], w=W1, out=E1[s])
            moe_mix[s](e0=E0[s], e1=E1[s], g=G[s], out=Y[s])

    program = (
        moe_block()
        .dispatch(DispatchPolicy.round_robin(num_aicpus=4))
        .task_graph(window=TaskWindow(8192, "tasks", WindowMode.STALL))
        .compile(target="cpu_sim")
    )
    program.execute()
    program.synchronize()

    stats = program.stats() if callable(program.stats) else program.stats
    total_cycles = int(getattr(stats, "total_cycles", 0) or 0)
    return y_base, total_cycles

