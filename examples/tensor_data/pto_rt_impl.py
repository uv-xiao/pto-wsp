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


def run_add_tiles(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, int]:
    if a.shape != b.shape:
        raise ValueError("A and B shapes must match")
    if a.ndim != 5:
        raise ValueError("A/B must be [B,TM,TN,R,C] tiled")

    f32 = DType.F32
    batch, tm, tn, tile_m, tile_n = a.shape
    out = np.zeros_like(a, dtype=np.float32)

    A = Tensor(data=a.astype(np.float32), shape=a.shape, dtype=f32)
    B = Tensor(data=b.astype(np.float32), shape=b.shape, dtype=f32)
    C = Tensor(data=out, shape=out.shape, dtype=f32)

    batch_ax = Dense[batch]()
    tm_ax = Dense[tm]()
    tn_ax = Dense[tn]()

    @kernel
    def add_tiles(
        a_tile: In[Tile[tile_m, tile_n, f32]],
        b_tile: In[Tile[tile_m, tile_n, f32]],
        c_tile: Out[Tile[tile_m, tile_n, f32]],
    ):
        z = pto.add(a_tile, b_tile)
        pto.store(c_tile, z)

    @workload
    def wl():
        for bb in P(batch_ax):
            for i in P(tm_ax):
                for j in P(tn_ax):
                    add_tiles[bb, i, j](a_tile=A[bb][i][j], b_tile=B[bb][i][j], c_tile=C[bb][i][j])

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
    return out, total_cycles


def run_square_tiles(x: np.ndarray) -> tuple[np.ndarray, int]:
    if x.ndim != 4:
        raise ValueError("X must be [B,T,R,C] tiled")

    f32 = DType.F32
    batch, tiles, tile_m, tile_n = x.shape
    out = np.zeros_like(x, dtype=np.float32)

    X = Tensor(data=x.astype(np.float32), shape=x.shape, dtype=f32)
    Y = Tensor(data=out, shape=out.shape, dtype=f32)

    batch_ax = Dense[batch]()
    tiles_ax = Dense[tiles]()

    @kernel
    def square_tile(
        in_tile: In[Tile[tile_m, tile_n, f32]],
        out_tile: Out[Tile[tile_m, tile_n, f32]],
    ):
        y = pto.mul(in_tile, in_tile)
        pto.store(out_tile, y)

    @workload
    def wl():
        for bb in P(batch_ax):
            for t in P(tiles_ax):
                square_tile[bb, t](in_tile=X[bb][t], out_tile=Y[bb][t])

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
    return out, total_cycles

