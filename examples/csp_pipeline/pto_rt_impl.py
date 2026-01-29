import numpy as np

from pto_wsp import (
    DType,
    Dense,
    In,
    Out,
    P,
    TaskWindow,
    Tile,
    WindowMode,
    DispatchPolicy,
    Tensor,
    kernel,
    pto,
    workload,
    for_each,
    task,
)
from pto_wsp.csp import Channel, consume, connect, process, send


TILE_H = 64
TILE_W = 64


def run_csp_square_pipeline(
    x: np.ndarray,
    *,
    channel_depth: int = 2,
) -> tuple[np.ndarray, int]:
    """Run a 3-stage CSP pipeline (load -> square -> store) in CPU-sim codegen-first mode.

    CSP channels carry *tokens* (tile indices) for synchronization; the actual data lives
    in global tensors (shared memory).
    """
    if x.ndim != 3:
        raise ValueError("x must be [T, H, W]")
    if x.dtype != np.float32:
        x = x.astype(np.float32)

    tiles_n, h, w = x.shape
    if h != TILE_H or w != TILE_W:
        raise ValueError(f"x must be [T,{TILE_H},{TILE_W}] (got {x.shape})")
    out = np.zeros_like(x, dtype=np.float32)

    f32 = DType.F32

    X = Tensor(data=x, shape=x.shape, dtype=f32)
    T0 = Tensor(data=np.zeros_like(x, dtype=np.float32), shape=x.shape, dtype=f32)
    T1 = Tensor(data=np.zeros_like(x, dtype=np.float32), shape=x.shape, dtype=f32)
    Y = Tensor(data=out, shape=out.shape, dtype=f32)

    tiles = Dense[tiles_n]()

    @kernel
    def load_tile(src: In[Tile[TILE_H, TILE_W, f32]], dst: Out[Tile[TILE_H, TILE_W, f32]]):
        pto.store(dst, src)

    @kernel
    def square_tile(src: In[Tile[TILE_H, TILE_W, f32]], dst: Out[Tile[TILE_H, TILE_W, f32]]):
        y = pto.mul(src, src)
        pto.store(dst, y)

    @kernel
    def store_tile(src: In[Tile[TILE_H, TILE_W, f32]], dst: Out[Tile[TILE_H, TILE_W, f32]]):
        pto.store(dst, src)

    l2c = Channel("l2c", depth=int(channel_depth))
    c2s = Channel("c2s", depth=int(channel_depth))

    loader = process("loader").produces(l2c).body(
        for_each(
            tiles,
            lambda t: send(l2c, task(load_tile, [t], {"src": X[t], "dst": T0[t]})),
        )
    )

    computer = process("computer").consumes(l2c).produces(c2s).body(
        consume(
            l2c,
            lambda t: send(c2s, task(square_tile, [t], {"src": T0[t], "dst": T1[t]})),
        )
    )

    storer = process("storer").consumes(c2s).body(
        consume(
            c2s,
            lambda t: task(store_tile, [t], {"src": T1[t], "dst": Y[t]}),
        )
    )

    pipe = connect([loader, computer, storer], [l2c, c2s])

    program = (
        pipe
        .dispatch(DispatchPolicy.round_robin(num_aicpus=4))
        .task_graph(window=TaskWindow(8192, "tasks", WindowMode.STALL))
        .compile(target="cpu_sim")
    )
    program.set_symbol_u64("__pto_wsp_channel_latency_cycles", 0)
    program.execute()
    program.synchronize()
    stats = program.stats() if callable(program.stats) else program.stats
    return out, int(getattr(stats, "total_cycles", 0) or 0)
