import numpy as np

from pto_wsp import (
    kernel,
    pto,
    In,
    InOut,
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


def run_bgemm(a_base: np.ndarray, b_base: np.ndarray) -> tuple[np.ndarray, int]:
    # Shapes: A[b,m,k], B[b,k,n]
    if a_base.ndim != 3 or b_base.ndim != 3:
        raise ValueError("A and B must be rank-3 (batched)")
    batch_size, m, k = a_base.shape
    batch2, k2, n = b_base.shape
    if batch2 != batch_size or k2 != k:
        raise ValueError(f"shape mismatch: A={a_base.shape}, B={b_base.shape}")

    f16 = DType.F16
    f32 = DType.F32

    tile_m = 16
    tile_n = 16
    tile_k = 16
    if m % tile_m or n % tile_n or k % tile_k:
        raise ValueError("This example requires M,N,K divisible by 16.")

    tiles_m = Dense[m // tile_m]()
    tiles_n = Dense[n // tile_n]()
    tiles_k = Dense[k // tile_k]()
    batch = Dense[batch_size]()

    c_base = np.zeros((batch_size, m, n), dtype=np.float32)

    # A tiles: [b, tm, tk, TILE_M, TILE_K]
    a_tiles = a_base.reshape(batch_size, m // tile_m, tile_m, k // tile_k, tile_k).transpose(0, 1, 3, 2, 4)
    # B tiles: [b, tn, tk, TILE_K, TILE_N]
    b_tiles = b_base.reshape(batch_size, k // tile_k, tile_k, n // tile_n, tile_n).transpose(0, 3, 1, 2, 4)
    # C tiles: [b, tm, tn, TILE_M, TILE_N]
    c_tiles = c_base.reshape(batch_size, m // tile_m, tile_m, n // tile_n, tile_n).transpose(0, 1, 3, 2, 4)

    A = Tensor(data=a_tiles, shape=a_tiles.shape, dtype=f16)
    B = Tensor(data=b_tiles, shape=b_tiles.shape, dtype=f16)
    C = Tensor(data=c_tiles, shape=c_tiles.shape, dtype=f32)

    @kernel
    def gemm_tile(
        a_tile: In[Tile[tile_m, tile_k, f16]],
        b_tile: In[Tile[tile_k, tile_n, f16]],
        c_tile: InOut[Tile[tile_m, tile_n, f32]],
    ):
        acc = pto.matmul(a_tile, b_tile)
        out = pto.add(c_tile, acc)
        pto.store(c_tile, out)

    @workload
    def bgemm():
        for b in P(batch):
            for tm in P(tiles_m):
                for tn in P(tiles_n):
                    for tk in P.seq(tiles_k):
                        gemm_tile[b, tm, tn, tk](
                            a_tile=A[b][tm][tk],
                            b_tile=B[b][tn][tk],
                            c_tile=C[b][tm][tn],
                        )

    program = (
        bgemm()
        .dispatch(DispatchPolicy.round_robin(num_aicpus=4))
        .task_graph(window=TaskWindow(8192, "tasks", WindowMode.STALL))
        .compile(target="cpu_sim")
    )
    program.execute()
    program.synchronize()

    stats = program.stats() if callable(program.stats) else program.stats
    total_cycles = int(getattr(stats, "total_cycles", 0) or 0)
    return c_base, total_cycles

