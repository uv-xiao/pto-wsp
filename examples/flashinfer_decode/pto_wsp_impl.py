import math
import numpy as np

from pto_wsp import (
    kernel,
    pto,
    In,
    Out,
    Tile,
    workload,
    Tensor,
    DType,
    DispatchPolicy,
    TaskWindow,
    WindowMode,
)


def run_decode_attention(q: np.ndarray, k: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, int]:
    f32 = DType.F32

    if q.shape[0] != 1:
        raise ValueError("q must be [1,D]")
    if k.ndim != 2 or v.ndim != 2 or k.shape != v.shape:
        raise ValueError("k and v must be [KV,D] and same shape")

    d = q.shape[1]
    kv = k.shape[0]

    out = np.zeros((1, d), dtype=np.float32)
    k_t = k.T.copy()

    Q = Tensor(data=q.astype(np.float32), shape=q.shape, dtype=f32)
    K_t = Tensor(data=k_t.astype(np.float32), shape=k_t.shape, dtype=f32)
    V = Tensor(data=v.astype(np.float32), shape=v.shape, dtype=f32)
    O = Tensor(data=out, shape=out.shape, dtype=f32)

    @kernel
    def decode_attention(
        q_tile: In[Tile[1, d, f32]],
        k_t_tile: In[Tile[d, kv, f32]],
        v_tile: In[Tile[kv, d, f32]],
        out_tile: Out[Tile[1, d, f32]],
    ):
        scores = pto.matmul(q_tile, k_t_tile)
        scale = pto.constant(1.0 / math.sqrt(d), f32)
        scores = pto.mul(scores, scale)
        max_s = pto.rowmax(scores)
        scores = pto.sub(scores, max_s)
        exp_s = pto.exp(scores)
        sum_s = pto.rowsum(exp_s)
        w = pto.div(exp_s, sum_s)
        out_v = pto.matmul(w, v_tile)
        pto.store(out_tile, out_v)

    @workload
    def decode():
        decode_attention[0](q_tile=Q, k_t_tile=K_t, v_tile=V, out_tile=O)

    program = (
        decode()
        .dispatch(DispatchPolicy.round_robin(num_aicpus=4))
        .task_graph(window=TaskWindow(1024, "tasks", WindowMode.STALL))
        .compile(target="cpu_sim")
    )
    program.execute()
    program.synchronize()

    stats = program.stats() if callable(program.stats) else program.stats
    total_cycles = int(getattr(stats, "total_cycles", 0) or 0)
    return out, total_cycles

