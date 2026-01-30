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


def run_matmul(a_base: np.ndarray, b_base: np.ndarray) -> tuple[np.ndarray, int]:
    # Configuration: fixed to keep this example fast and deterministic.
    batch_size, m, k = a_base.shape
    _, k2, n = b_base.shape
    if k2 != k:
        raise ValueError(f"matmul shape mismatch: A is {a_base.shape}, B is {b_base.shape}")

    tile_m = m
    tile_n = n
    tile_k = k

    f32 = DType.F32

    c_base = np.zeros((batch_size, m, n), dtype=np.float32)

    a_tiles = a_base.reshape(batch_size, 1, tile_m, tile_k)
    b_tiles = b_base.reshape(batch_size, 1, tile_k, tile_n)
    c_tiles = c_base.reshape(batch_size, 1, tile_m, tile_n)

    A = Tensor(data=a_tiles, shape=a_tiles.shape, dtype=f32)
    B = Tensor(data=b_tiles, shape=b_tiles.shape, dtype=f32)
    C = Tensor(data=c_tiles, shape=c_tiles.shape, dtype=f32)

    batch = Dense[batch_size]()
    one = Dense[1]()

    @kernel
    def matmul_kernel(
        a_tile: In[Tile[tile_m, tile_k, f32]],
        b_tile: In[Tile[tile_k, tile_n, f32]],
        c_tile: Out[Tile[tile_m, tile_n, f32]],
    ):
        out = pto.matmul(a_tile, b_tile)
        pto.store(c_tile, out)

    @workload
    def tiled_matmul():
        for b in P(batch):
            for _ in P(one):
                matmul_kernel[b](
                    a_tile=A[b][0],
                    b_tile=B[b][0],
                    c_tile=C[b][0],
                )

    program = (
        tiled_matmul()
        .dispatch(DispatchPolicy.round_robin(num_aicpus=4))
        .task_graph(window=TaskWindow(1024, "tasks", WindowMode.STALL))
        .compile(target="cpu_sim")
    )
    program.execute()
    program.synchronize()

    stats = program.stats() if callable(program.stats) else program.stats
    total_cycles = int(getattr(stats, "total_cycles", 0) or 0)
    return c_base, total_cycles

