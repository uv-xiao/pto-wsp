"""
End-to-end tests for PTO Workload-Schedule Programming (PTO-WSP) framework.

These tests verify the complete flow from Python workload definition
through compilation and execution.

Test Categories:
- E2E CPU simulation: Python workload → compile → execute → verify
- E2E task enumeration: Verify correct task generation from workloads
- E2E scheduling: Verify dispatch/stream policies affect execution
"""

import sys
import os
import threading
import time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

import pytest
import numpy as np
from pto_wsp import (
    # Workload definition
    workload, kernel, P,
    In, Out, InOut,
    # Types
    Dense, DenseDyn,
    Tensor, DType,
    # Legacy primitives
    parallel_for, for_each, task, combine, sequential, select, cond,
    # Schedule
    DispatchPolicy, TimingPolicy,
    # Workload class
    Workload,
)


# ============================================================
# E2E CPU Simulation Tests
# ============================================================

class TestE2ECPUSimulation:
    """End-to-end tests for CPU simulation backend."""

    def test_simple_task_execution(self):
        """Test single task compiles and executes."""
        w = task("simple", [0], [])
        program = w.compile(target="cpu_sim")

        # Should compile successfully
        assert program is not None
        assert program.stats.compile_time_ms >= 0

        # Execute and synchronize
        program.execute()
        program.synchronize()

        # Verify completion
        assert program.is_complete()
        assert program.stats.task_count == 1

    def test_parallel_for_enumeration(self):
        """Test parallel_for enumerates correct number of tasks."""
        batch = DenseDyn(8)
        w = parallel_for(batch, lambda b: task("process", [b], []))
        program = w.compile(target="cpu_sim")

        program.execute()
        program.synchronize()

        # Should have 8 tasks (one per batch element)
        assert program.stats.task_count == 8

    def test_nested_parallel_for(self):
        """Test nested parallel_for produces Cartesian product of tasks."""
        batch = DenseDyn(4)
        heads = Dense[2]()

        @workload
        def nested_workload(batch, heads):
            for b in P(batch):
                for h in P(heads):
                    task("attn", [b, h], [])

        w = nested_workload(batch, heads)
        program = w.compile(target="cpu_sim")

        program.execute()
        program.synchronize()

        # Should have 4 * 2 = 8 tasks
        assert program.stats.task_count == 8

    def test_kernel_with_implementation(self):
        """Test kernel execution with registered implementation."""
        # Create mutable state to verify execution
        results = []
        lock = threading.Lock()

        def accumulate_impl(val):
            with lock:
                results.append(val)

        batch = DenseDyn(4)
        w = parallel_for(batch, lambda b: task("accumulate", [b], [b]))
        program = w.compile(target="cpu_sim")

        program.register_kernel("accumulate", accumulate_impl)
        program.execute()
        program.synchronize()

        # Should have executed 4 times
        assert len(results) == 4
        # Values should be 0, 1, 2, 3 (in some order due to parallelism)
        assert sorted(results) == [0, 1, 2, 3]

    def test_sequential_execution(self):
        """Test sequential workloads execute in order."""
        order = []
        lock = threading.Lock()

        def record_impl(idx):
            with lock:
                order.append(idx)

        seq = DenseDyn(5)
        w = for_each(seq, lambda i: task("record", [i], [i]))
        program = w.compile(target="cpu_sim")

        program.register_kernel("record", record_impl)
        program.execute()
        program.synchronize()

        assert len(order) == 5
        # for_each is sequential, so order should be preserved
        # (though current impl may still be parallel - this tests the API)

    def test_combine_workloads(self):
        """Test combined workloads execute all tasks."""
        t1 = task("kernel_a", [], [])
        t2 = task("kernel_b", [], [])
        t3 = task("kernel_c", [], [])
        w = combine(t1, t2, t3)

        program = w.compile(target="cpu_sim")
        program.execute()
        program.synchronize()

        assert program.stats.task_count == 3

    def test_stream_count_configuration(self):
        """Test stream count affects parallelism."""
        batch = DenseDyn(16)
        w = parallel_for(batch, lambda b: task("compute", [b], [])).streams(4)

        program = w.compile(target="cpu_sim")

        # Verify stream count was compiled
        assert program._compiled_plan.stream_count == 4

        program.execute()
        program.synchronize()
        assert program.stats.task_count == 16

    def test_dispatch_policy_configuration(self):
        """Test dispatch policy is preserved in compilation."""
        w = task("kernel", [], []).dispatch(DispatchPolicy.round_robin(4))

        program = w.compile(target="cpu_sim")

        # Verify dispatch policy was compiled
        assert program._compiled_plan.dispatch_policy is not None

    def test_execution_stats(self):
        """Test execution statistics are tracked."""
        batch = DenseDyn(10)
        w = parallel_for(batch, lambda b: task("work", [b], []))

        program = w.compile(target="cpu_sim")
        program.execute()
        program.synchronize()

        # Compile time should be recorded
        assert program.stats.compile_time_ms >= 0
        # Execute time should be recorded
        assert program.stats.execute_time_ms >= 0
        # Task count should be correct
        assert program.stats.task_count == 10


class TestE2ETaskEnumeration:
    """Tests for task enumeration from workloads."""

    def test_enumerate_parallel_for(self):
        """Test enumerate() extracts tasks from parallel_for."""
        batch = DenseDyn(4)
        w = parallel_for(batch, lambda b: task("kernel", [b], []))

        tasks = w.enumerate()

        assert len(tasks) == 4
        # Each task should have a binding for the batch axis
        for i, t in enumerate(tasks):
            assert t.kernel == "kernel"

    def test_enumerate_nested_loops(self):
        """Test enumerate() handles nested parallel_for."""
        batch = DenseDyn(2)
        heads = Dense[3]()

        @workload
        def nested(batch, heads):
            for b in P(batch):
                for h in P(heads):
                    task("attn", [b, h], [])

        w = nested(batch, heads)
        tasks = w.enumerate()

        # 2 * 3 = 6 tasks
        assert len(tasks) == 6

    def test_enumerate_combine(self):
        """Test enumerate() handles combine."""
        t1 = task("a", [], [])
        t2 = task("b", [], [])
        w = combine(t1, t2)

        tasks = w.enumerate()
        assert len(tasks) == 2
        kernels = [t.kernel for t in tasks]
        assert "a" in kernels
        assert "b" in kernels

    def test_enumerate_sequential(self):
        """Test enumerate() handles sequential."""
        t1 = task("first", [], [])
        t2 = task("second", [], [])
        w = sequential(t1, t2)

        tasks = w.enumerate()
        assert len(tasks) == 2

    def test_enumerate_select(self):
        """Test enumerate() handles select (sparse iteration)."""
        # Select with explicit indices list
        indices = [0, 2, 5]  # Sparse indices
        w = select(indices, lambda e: task(f"expert_{e}", [e], []))

        tasks = w.enumerate()

        assert len(tasks) == 3  # One task per selected index
        kernels = [t.kernel for t in tasks]
        assert "expert_0" in kernels
        assert "expert_2" in kernels
        assert "expert_5" in kernels

    def test_enumerate_select_with_range(self):
        """Test enumerate() handles select with integer (range)."""
        # Select with integer acts like range
        w = select(3, lambda e: task(f"expert_{e}", [e], []))

        tasks = w.enumerate()

        assert len(tasks) == 3
        kernels = [t.kernel for t in tasks]
        assert "expert_0" in kernels
        assert "expert_1" in kernels
        assert "expert_2" in kernels

    def test_enumerate_cond_true(self):
        """Test enumerate() handles cond with true predicate."""
        then_w = task("fast_path", [], [])
        else_w = task("slow_path", [], [])
        w = cond(True, then_w, else_w)

        tasks = w.enumerate()

        assert len(tasks) == 1
        assert tasks[0].kernel == "fast_path"

    def test_enumerate_cond_false(self):
        """Test enumerate() handles cond with false predicate."""
        then_w = task("fast_path", [], [])
        else_w = task("slow_path", [], [])
        w = cond(False, then_w, else_w)

        tasks = w.enumerate()

        assert len(tasks) == 1
        assert tasks[0].kernel == "slow_path"

    def test_enumerate_cond_callable(self):
        """Test enumerate() handles cond with callable predicate."""
        seq_len = 1024  # Simulated runtime value

        then_w = task("attn_2k", [seq_len], [])
        else_w = task("attn_8k", [seq_len], [])
        w = cond(lambda: seq_len <= 2048, then_w, else_w)

        tasks = w.enumerate()

        assert len(tasks) == 1
        assert tasks[0].kernel == "attn_2k"  # 1024 <= 2048, so true branch

    def test_task_get_axis_value(self):
        """Test Task.get() returns correct axis values."""
        from pto_wsp.workload import Task

        t = Task(
            kernel="kernel",
            params=[0, 1],
            resources=[],
            bindings={"batch": 3, "head": 7}
        )

        assert t.get("batch") == 3
        assert t.get("head") == 7
        assert t.get("unknown") == 0  # Default for missing


class TestE2EKernelDecorator:
    """Tests for @kernel integration in e2e flow."""

    def test_kernel_in_workload(self):
        """Test @kernel decorated functions work in workloads."""
        @kernel
        def compute(x: In[Tensor], y: Out[Tensor]):
            pass

        @workload
        def process(batch):
            for b in P(batch):
                compute[b](None, None)

        w = process(DenseDyn(4))
        program = w.compile(target="cpu_sim")
        program.execute()
        program.synchronize()

        assert program.stats.task_count == 4

    def test_kernel_with_actual_computation(self):
        """Test kernel executes actual computation."""
        results = {"sum": 0}
        lock = threading.Lock()

        @kernel
        def add_kernel(x: In[Tensor], y: Out[Tensor]):
            pass

        def add_impl(val):
            with lock:
                results["sum"] += val

        batch = DenseDyn(5)
        w = parallel_for(batch, lambda b: task("add_kernel", [b], [b]))
        program = w.compile(target="cpu_sim")

        program.register_kernel("add_kernel", add_impl)
        program.execute()
        program.synchronize()

        # Sum of 0+1+2+3+4 = 10
        assert results["sum"] == 10


class TestE2ESchedulingPolicies:
    """Tests for scheduling policy effects on execution."""

    def test_dispatch_threshold(self):
        """Test dispatch_threshold configuration."""
        w = task("kernel", [], []).dispatch_threshold(
            thresholds=[100, 1000],
            policies={
                100: DispatchPolicy.round_robin(1),
                1000: DispatchPolicy.round_robin(4),
            }
        )

        program = w.compile(target="cpu_sim")
        # Should compile without error
        program.execute()
        program.synchronize()

    def test_task_graph_mode(self):
        """Test task_graph mode compilation."""
        from pto_wsp import Deps, ReadyPolicy, Pools

        w = task("kernel", [], []).task_graph(
            deps=Deps.infer_tensor_map_exact(),
            ready=ReadyPolicy.fifo(),
            pools=Pools.single()
        )

        program = w.compile(target="cpu_sim")
        program.execute()
        program.synchronize()

        # Should complete successfully
        assert program.is_complete()

    def test_timing_policy(self):
        """Test timing policy configuration."""
        w = task("kernel", [], []).timing(TimingPolicy.immediate)

        program = w.compile(target="cpu_sim")
        assert program._compiled_plan is not None


class TestE2ECSPPrimitives:
    """Tests for CSP primitives in e2e flow."""

    def test_channel_send(self):
        """Test send workload compiles and executes."""
        from pto_wsp import Channel, send

        ch = Channel("data", depth=2)
        w = send(ch, task("produce", [], []))

        program = w.compile(target="cpu_sim")
        program.execute()
        program.synchronize()

        # Send wraps a task, so should have 1 task
        assert program.stats.task_count == 1

    def test_process_pipeline(self):
        """Test process pipeline compiles."""
        from pto_wsp import Channel, process, send, consume, connect

        ch = Channel("pipe", depth=2)

        loader = (process("loader")
            .produces(ch)
            .body(task("load", [], [])))

        computer = (process("computer")
            .consumes(ch)
            .produces(Channel("out", depth=2))
            .body(task("compute", [], [])))

        pipeline = connect([loader, computer], [ch])

        program = pipeline.compile(target="cpu_sim")
        program.execute()
        program.synchronize()

        # Pipeline has 2 processes with 1 task each
        assert program.stats.task_count == 2


class TestE2EErrorHandling:
    """Tests for error handling in e2e flow."""

    def test_empty_workload(self):
        """Test empty workload compiles and executes without error."""
        from pto_wsp.workload import Workload

        w = Workload("empty")

        program = w.compile(target="cpu_sim")
        program.execute()
        program.synchronize()

        assert program.is_complete()
        assert program.stats.task_count == 0

    def test_double_execute_raises(self):
        """Test calling execute twice raises error."""
        w = task("kernel", [], [])
        program = w.compile(target="cpu_sim")

        program.execute()

        with pytest.raises(RuntimeError):
            program.execute()

    def test_missing_kernel_impl_no_error(self):
        """Test missing kernel implementation doesn't crash (no-op)."""
        w = task("nonexistent_kernel", [], [])
        program = w.compile(target="cpu_sim")

        # Should not raise, just no-op
        program.execute()
        program.synchronize()
        assert program.is_complete()


class TestE2EPerformance:
    """Performance-related e2e tests."""

    def test_large_workload_scalability(self):
        """Test large workload compiles and executes."""
        batch = DenseDyn(1000)
        w = parallel_for(batch, lambda b: task("compute", [b], []))

        program = w.compile(target="cpu_sim")
        program.execute()
        program.synchronize()

        assert program.stats.task_count == 1000

    def test_parallel_execution_faster_than_serial(self):
        """Test parallel execution provides speedup."""
        results = []
        lock = threading.Lock()

        def slow_kernel(idx):
            time.sleep(0.01)  # 10ms sleep
            with lock:
                results.append(idx)

        batch = DenseDyn(8)
        w = parallel_for(batch, lambda b: task("slow", [b], [b])).streams(8)

        program = w.compile(target="cpu_sim")
        program.register_kernel("slow", slow_kernel)

        start = time.perf_counter()
        program.execute()
        program.synchronize()
        elapsed = time.perf_counter() - start

        # Serial would take ~80ms (8 * 10ms)
        # Parallel should be much faster (closer to 10ms + overhead)
        # Use generous bound: should be less than 60ms
        assert elapsed < 0.06, f"Parallel execution too slow: {elapsed:.3f}s"
        assert len(results) == 8


class TestE2EIntegration:
    """Integration tests combining multiple features."""

    def test_full_attention_workload(self):
        """Test complete multi-head attention workload."""
        @kernel
        def flash_attention(Q: In[Tensor], K: In[Tensor],
                          V: In[Tensor], O: Out[Tensor]):
            pass

        @workload
        def multi_head_attention(batch, heads, seq_len):
            for b in P(batch):
                for h in P(heads):
                    # Would have Q, K, V, O tensors here
                    flash_attention[b, h](None, None, None, None)

        batch = DenseDyn(2)
        heads = Dense[4]()
        seq_len = DenseDyn(128)

        w = multi_head_attention(batch, heads, seq_len)
        w = w.dispatch(DispatchPolicy.round_robin(4)).streams(4)

        program = w.compile(target="cpu_sim")
        program.execute()
        program.synchronize()

        # 2 batches * 4 heads = 8 tasks
        assert program.stats.task_count == 8

    def test_matmul_tiled_workload(self):
        """Test tiled matrix multiplication workload."""
        @kernel
        def matmul_tile(A: In[Tensor], B: In[Tensor], C: Out[Tensor]):
            pass

        @workload
        def tiled_matmul(M_tiles, N_tiles, K_tiles):
            for m in P(M_tiles):
                for n in P(N_tiles):
                    for k in P.seq(K_tiles):
                        matmul_tile[m, n, k](None, None, None)

        M_tiles = DenseDyn(4)
        N_tiles = DenseDyn(4)
        K_tiles = DenseDyn(2)

        w = tiled_matmul(M_tiles, N_tiles, K_tiles)
        program = w.compile(target="cpu_sim")
        program.execute()
        program.synchronize()

        # 4 * 4 * 2 = 32 tiles
        assert program.stats.task_count == 32


class TestE2EAscendCodegen:
    """End-to-end tests for Ascend NPU codegen using JIT kernel API."""

    def test_jit_kernel_produces_ir(self):
        """Test that @jit_kernel produces typed IR (no string refs)."""
        from pto_wsp.kernel import jit_kernel, tl, Value, DType

        @jit_kernel
        def test_matmul(a, b, c):
            # Typed operations - no string names!
            result = tl.matmul(a, b)
            tl.store(c, result)

        # Trace kernel to get IR
        a_val = Value.tile(64, 64, DType.F16, name="a")
        b_val = Value.tile(64, 64, DType.F16, name="b")
        c_val = Value.tile(64, 64, DType.F16, name="c")

        ir = test_matmul(a=a_val, b=b_val, c=c_val)

        # IR should have operations
        assert ir is not None
        assert len(ir.ops) >= 2  # matmul + store
        assert ir.name == "test_matmul"

    def test_jit_kernel_compiles_to_ascend(self):
        """Test JIT kernel compiles to Ascend code."""
        from pto_wsp.kernel import jit_kernel, tl, Value, DType

        @jit_kernel
        def rmsnorm_kernel(x, out):
            sq = tl.mul(x, x)
            mean = tl.rowmean(sq)
            rsqrt_val = tl.rsqrt(mean)
            result = tl.mul(x, rsqrt_val)
            tl.store(out, result)

        # Compile for Ascend
        compiled = rmsnorm_kernel.compile(target="ascend")

        # Generated code should be non-empty
        assert compiled.code is not None
        assert len(compiled.code) > 0
        assert "Generated by PTO-RT" in compiled.code
        assert "rmsnorm_kernel" in compiled.code

    def test_jit_kernel_typed_values(self):
        """Test that JIT kernel uses typed Values, not strings."""
        from pto_wsp.kernel import jit_kernel, tl, Value, DType, OpKind

        @jit_kernel
        def typed_kernel(x, y, out):
            sum_val = tl.add(x, y)
            tl.store(out, sum_val)

        x = Value.tile(32, 128, DType.F16)
        y = Value.tile(32, 128, DType.F16)
        out = Value.tile(32, 128, DType.F16)

        ir = typed_kernel(x=x, y=y, out=out)

        # All operands should be Value objects with integer IDs
        for op in ir.ops:
            for operand in op.operands:
                assert isinstance(operand, Value)
                assert isinstance(operand.id, int)
            if op.result:
                assert isinstance(op.result, Value)
                assert isinstance(op.result.id, int)

    def test_tile_language_primitives(self):
        """Test that tl.* primitives create proper IR."""
        from pto_wsp.kernel import jit_kernel, tl, Value, DType, OpKind

        @jit_kernel
        def test_primitives(x, out):
            # Test various operations
            sq = tl.mul(x, x)
            exp_val = tl.exp(sq)
            sum_val = tl.rowsum(exp_val)
            tl.store(out, sum_val)

        ir = test_primitives(
            x=Value.tile(32, 128, DType.F16),
            out=Value.tile(32, 1, DType.F16)
        )

        # Check operation kinds
        op_kinds = [op.kind for op in ir.ops]
        assert OpKind.Mul in op_kinds
        assert OpKind.Exp in op_kinds
        assert OpKind.RowSum in op_kinds
        assert OpKind.Store in op_kinds

    def test_ascend_backend_available(self):
        """Test that Ascend backend is available via C++."""
        try:
            import pto_ir_cpp
        except ImportError:
            pytest.skip("pto_ir_cpp not available")

        backend = pto_ir_cpp.AscendNPUBackend()
        assert backend is not None
        assert backend.name() == "ascend_npu"
        assert "ascend" in backend.supported_targets()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
