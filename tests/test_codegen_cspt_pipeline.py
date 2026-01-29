import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from pto_wsp import DType, Dense, In, Out, P, Tensor, Tile, kernel, pto, workload
from pto_wsp.csp import Channel, consume, connect, process, send
from pto_wsp.primitives import for_each
from pto_wsp.workload import Workload


def test_cspt_pipeline_codegen_first_cpu_sim_runs_and_overlaps():
    tiles = Dense[4]()

    x_np = np.arange(4 * 4 * 4, dtype=np.float32).reshape(4, 4, 4)
    tmp_np = np.zeros_like(x_np)
    y_np = np.zeros_like(x_np)

    x = Tensor(data=x_np, shape=x_np.shape, dtype=DType.F32)
    tmp = Tensor(data=tmp_np, shape=tmp_np.shape, dtype=DType.F32)
    y = Tensor(data=y_np, shape=y_np.shape, dtype=DType.F32)

    @kernel
    def load(src: In[Tile[4, 4, DType.F32]], dst: Out[Tile[4, 4, DType.F32]]):
        pto.store(dst, src)

    @kernel
    def square(inp: In[Tile[4, 4, DType.F32]], out: Out[Tile[4, 4, DType.F32]]):
        v = pto.mul(inp, inp)
        pto.store(out, v)

    ch = Channel("l2c", depth=1)

    def make_load_task(i):
        return Workload("task", kernel=load, params=[i], resources={"src": x[i], "dst": tmp[i]})

    def make_scale_task(i):
        return Workload("task", kernel=square, params=[i], resources={"inp": tmp[i], "out": y[i]})

    producer = process("producer").produces(ch).body(for_each(tiles, lambda i: send(ch, make_load_task(i))))
    consumer = process("consumer").consumes(ch).body(consume(ch, lambda i: make_scale_task(i)))

    pipeline = connect([producer, consumer], [ch])

    # Codegen-first CSPT pipeline.
    y_np[:] = 0
    tmp_np[:] = 0
    p_pipe = pipeline.named("cspt_pipe").compile(target="cpu_sim")
    assert p_pipe.using_cpp_backend
    p_pipe.execute()
    p_pipe.synchronize()
    np.testing.assert_allclose(y_np, x_np * x_np, rtol=0, atol=0)
    cycles_pipe = int(p_pipe.stats.total_cycles)
    assert cycles_pipe > 0

    # Equivalent sequential workload should not be faster than the pipeline makespan.
    @workload
    def seq_wl():
        for i in P(tiles):
            load[i](src=x[i], dst=tmp[i])
            square[i](inp=tmp[i], out=y[i])

    y_np[:] = 0
    tmp_np[:] = 0
    p_seq = seq_wl().named("cspt_seq").compile(target="cpu_sim")
    p_seq.execute()
    p_seq.synchronize()
    np.testing.assert_allclose(y_np, x_np * x_np, rtol=0, atol=0)
    cycles_seq = int(p_seq.stats.total_cycles)
    assert cycles_seq >= cycles_pipe


def test_cspt_channel_latency_symbol_increases_cycles_without_recompile():
    tiles = Dense[4]()

    x_np = np.arange(4 * 4 * 4, dtype=np.float32).reshape(4, 4, 4)
    tmp_np = np.zeros_like(x_np)
    y_np = np.zeros_like(x_np)

    x = Tensor(data=x_np, shape=x_np.shape, dtype=DType.F32)
    tmp = Tensor(data=tmp_np, shape=tmp_np.shape, dtype=DType.F32)
    y = Tensor(data=y_np, shape=y_np.shape, dtype=DType.F32)

    @kernel
    def load(src: In[Tile[4, 4, DType.F32]], dst: Out[Tile[4, 4, DType.F32]]):
        pto.store(dst, src)

    @kernel
    def square(inp: In[Tile[4, 4, DType.F32]], out: Out[Tile[4, 4, DType.F32]]):
        v = pto.mul(inp, inp)
        pto.store(out, v)

    ch = Channel("l2c", depth=1)

    def make_load_task(i):
        return Workload("task", kernel=load, params=[i], resources={"src": x[i], "dst": tmp[i]})

    def make_square_task(i):
        return Workload("task", kernel=square, params=[i], resources={"inp": tmp[i], "out": y[i]})

    pipeline = connect(
        [
            process("producer").produces(ch).body(for_each(tiles, lambda i: send(ch, make_load_task(i)))),
            process("consumer").consumes(ch).body(consume(ch, lambda i: make_square_task(i))),
        ],
        [ch],
    )

    program = pipeline.named("cspt_latency").compile(target="cpu_sim")
    assert program.using_cpp_backend

    # Run with default latency=0.
    y_np[:] = 0
    tmp_np[:] = 0
    program.set_symbol_u64("__pto_wsp_channel_latency_cycles", 0)
    program.execute()
    program.synchronize()
    np.testing.assert_allclose(y_np, x_np * x_np, rtol=0, atol=0)
    cycles0 = int(program.stats.total_cycles)
    assert cycles0 > 0

    # Run again without rebuild, but with increased channel latency.
    y_np[:] = 0
    tmp_np[:] = 0
    program.set_symbol_u64("__pto_wsp_channel_latency_cycles", 50)
    program.execute()
    program.synchronize()
    np.testing.assert_allclose(y_np, x_np * x_np, rtol=0, atol=0)
    cycles50 = int(program.stats.total_cycles)
    assert cycles50 > cycles0
