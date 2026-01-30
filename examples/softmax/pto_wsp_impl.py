import numpy as np

from pto_wsp import (
    kernel,
    pto,
    In,
    Out,
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


def run_softmax(logits_base: np.ndarray) -> tuple[np.ndarray, int]:
    f32 = DType.F32

    if logits_base.ndim != 3:
        raise ValueError("logits must be [B,S,V]")
    batch_size, seq_len, vocab = logits_base.shape

    tile_row = 8
    tile_col = 64
    if seq_len % tile_row or vocab % tile_col:
        raise ValueError("This example requires SEQ divisible by 8 and VOCAB divisible by 64.")

    seq_tiles_n = seq_len // tile_row
    vocab_tiles_n = vocab // tile_col

    probs_base = np.zeros_like(logits_base, dtype=np.float32)

    logits_tiles = logits_base.reshape(batch_size, seq_tiles_n, tile_row, vocab_tiles_n, tile_col)
    probs_tiles = probs_base.reshape(batch_size, seq_tiles_n, tile_row, vocab_tiles_n, tile_col)

    running_max_base = np.full((batch_size, seq_tiles_n, tile_row, 1), -np.inf, dtype=np.float32)
    running_sum_base = np.zeros((batch_size, seq_tiles_n, tile_row, 1), dtype=np.float32)

    logits = Tensor(data=logits_tiles, shape=logits_tiles.shape, dtype=f32)
    probs = Tensor(data=probs_tiles, shape=probs_tiles.shape, dtype=f32)
    running_max = Tensor(data=running_max_base, shape=running_max_base.shape, dtype=f32)
    running_sum = Tensor(data=running_sum_base, shape=running_sum_base.shape, dtype=f32)

    batch = Dense[batch_size]()
    seq_tiles = Dense[seq_tiles_n]()
    vocab_tiles = Dense[vocab_tiles_n]()

    @kernel
    def update_rowmax(
        x_tile: In[Tile[tile_row, tile_col, f32]],
        m_acc: InOut[Tile[tile_row, 1, f32]],
    ):
        m_tile = pto.rowmax(x_tile)
        m_new = pto.max(m_acc, m_tile)
        pto.store(m_acc, m_new)

    @kernel
    def exp_and_accumulate(
        x_tile: In[Tile[tile_row, tile_col, f32]],
        m_tile: In[Tile[tile_row, 1, f32]],
        y_tile: Out[Tile[tile_row, tile_col, f32]],
        l_acc: InOut[Tile[tile_row, 1, f32]],
    ):
        x_shifted = pto.sub(x_tile, m_tile)
        exp_x = pto.exp(x_shifted)
        s_tile = pto.rowsum(exp_x)
        l_new = pto.add(l_acc, s_tile)
        pto.store(l_acc, l_new)
        pto.store(y_tile, exp_x)

    @kernel
    def rescale(
        y_tile: InOut[Tile[tile_row, tile_col, f32]],
        l_tile: In[Tile[tile_row, 1, f32]],
    ):
        y_scaled = pto.div(y_tile, l_tile)
        pto.store(y_tile, y_scaled)

    @workload
    def softmax_workload():
        for b in P(batch):
            for s in P(seq_tiles):
                for v in P.seq(vocab_tiles):
                    update_rowmax[b, s, v](x_tile=logits[b][s][v], m_acc=running_max[b][s])

        for b in P(batch):
            for s in P(seq_tiles):
                for v in P.seq(vocab_tiles):
                    exp_and_accumulate[b, s, v](
                        x_tile=logits[b][s][v],
                        m_tile=running_max[b][s],
                        y_tile=probs[b][s][v],
                        l_acc=running_sum[b][s],
                    )

        for b in P(batch):
            for s in P(seq_tiles):
                for v in P(vocab_tiles):
                    rescale[b, s, v](y_tile=probs[b][s][v], l_tile=running_sum[b][s])

    program = (
        softmax_workload()
        .dispatch(DispatchPolicy.round_robin(num_aicpus=4))
        .task_graph(window=TaskWindow(8192, "tasks", WindowMode.STALL))
        .compile(target="cpu_sim")
    )
    program.execute()
    program.synchronize()

    stats = program.stats() if callable(program.stats) else program.stats
    total_cycles = int(getattr(stats, "total_cycles", 0) or 0)
    return probs_base, total_cycles

