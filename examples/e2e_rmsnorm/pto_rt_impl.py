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


def run_rmsnorm(x_base: np.ndarray) -> tuple[np.ndarray, int]:
    f32 = DType.F32

    if x_base.ndim != 3:
        raise ValueError("x must be [B,SEQ,H]")
    batch, seq, hidden = x_base.shape

    tile_seq = 16
    if seq % tile_seq:
        raise ValueError("seq must be divisible by 16 for this example")

    seq_tiles_n = seq // tile_seq
    y_base = np.zeros((batch, seq, hidden), dtype=np.float32)

    x_tiles = x_base.reshape(batch, seq_tiles_n, tile_seq, hidden)
    y_tiles = y_base.reshape(batch, seq_tiles_n, tile_seq, hidden)

    X = Tensor(data=x_tiles.astype(np.float32), shape=x_tiles.shape, dtype=f32)
    Y = Tensor(data=y_tiles, shape=y_tiles.shape, dtype=f32)

    batch_ax = Dense[batch]()
    seq_tiles = Dense[seq_tiles_n]()

    @kernel
    def rmsnorm_tile(
        x: In[Tile[tile_seq, hidden, f32]],
        out: Out[Tile[tile_seq, hidden, f32]],
    ):
        sq = pto.mul(x, x)
        mean_sq = pto.rowmean(sq)
        inv = pto.rsqrt(mean_sq)
        y = pto.mul(x, inv)
        pto.store(out, y)

    @workload
    def wl():
        for b in P(batch_ax):
            for t in P(seq_tiles):
                rmsnorm_tile[b, t](x=X[b][t], out=Y[b][t])

    program = (
        wl()
        .dispatch(DispatchPolicy.round_robin(num_aicpus=4))
        .task_graph(window=TaskWindow(8192, "tasks", WindowMode.STALL))
        .compile(target="cpu_sim")
    )
    program.execute()
    program.synchronize()

    stats = program.stats() if callable(program.stats) else program.stats
    total_cycles = int(getattr(stats, "total_cycles", 0) or 0)
    return y_base, total_cycles
