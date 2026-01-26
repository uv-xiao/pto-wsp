"""
Tests for PTO Workload-Schedule Programming (PTO-WSP) framework Python Frontend.

Tests cover:
- @workload decorator and P namespace
- @kernel decorator for JIT kernels
- NPU function builder
- Schedule combinator API
- CSP primitives
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

import pytest
from pto_wsp import (
    # New API
    workload, kernel, P, KernelRef,
    In, Out, InOut, Constexpr,
    # Types
    Dense, DenseDyn, Ragged, Sparse,
    Tensor, DType, Location,
    # R10: Layout types
    MemLayout, TensorLayout, TensorShard, TensorReplicate,
    LayoutIncompatibleError, tensor_layout_join,
    relayout, allreduce, allgather, reduce_scatter,
    # NPU builder
    npu, NPUFunction, NPUOpKind,
    # Legacy API
    parallel_for, for_each, select, cond, task, combine, sequential,
    # CSP
    Channel, process, send, consume, connect,
    # Schedule
    DispatchPolicy, TimingPolicy,
    WindowMode, TaskWindow,
    # Task graph (R9)
    DepsMode, Deps, ReadyPolicy, StartPolicy, TracePolicy, Pools, TaskGraphConfig,
    # Workload
    Workload, Task,
    # Type Checker
    TypeChecker, TypeCheckContext, TypeErrorKind, TypeErrorInfo,
    Layout, LayoutShard, LayoutReplicate, layout_join, LayoutCompatibilityError,
    check_kernel_call, check_layouts_compatible, validate_axis_index,
)


# ============================================================
# Test @workload decorator + P namespace
# ============================================================

class TestWorkloadDecorator:
    """Tests for @workload decorator."""

    def test_simple_workload(self):
        """Test basic @workload usage."""
        @workload
        def simple(batch):
            pass

        result = simple(DenseDyn(4))
        assert result is not None

    def test_parallel_grid(self):
        """Test P() parallel grid syntax."""
        @kernel
        def attn(Q, K, V, O):
            pass

        @workload
        def attention(batch, heads):
            for b, h in P(batch, heads):
                attn[b, h](None, None, None, None)

        batch = DenseDyn(4)
        heads = Dense[8]()
        result = attention(batch, heads)
        assert result is not None
        assert result._kind == "parallel_for"

    def test_sequential_loop(self):
        """Test P.seq() sequential iteration."""
        @kernel
        def scan_step(x, y):
            pass

        @workload
        def scan(seq_len):
            for i in P.seq(seq_len):
                scan_step[i](None, None)

        result = scan(DenseDyn(32))
        assert result is not None
        assert result._kind == "for_each"

    def test_nested_loops(self):
        """Test nested P() loops."""
        @kernel
        def compute(x, y):
            pass

        @workload
        def nested(outer, inner):
            for i in P(outer):
                for j in P.seq(inner):
                    compute[i, j](None, None)

        result = nested(DenseDyn(4), DenseDyn(8))
        assert result is not None


# ============================================================
# Test @kernel decorator
# ============================================================

class TestKernelDecorator:
    """Tests for @kernel decorator."""

    def test_kernel_creates_ref(self):
        """Test @kernel creates KernelRef."""
        @kernel
        def my_kernel(x: In[Tensor], y: Out[Tensor]):
            pass

        assert isinstance(my_kernel, KernelRef)
        assert my_kernel.name == "my_kernel"

    def test_kernel_signature(self):
        """Test kernel signature extraction."""
        @kernel
        def typed_kernel(a: In[Tensor], b: Out[Tensor], c: Constexpr[int]):
            pass

        assert "a" in typed_kernel.signature
        assert "b" in typed_kernel.signature
        assert "c" in typed_kernel.signature

    def test_kernel_axis_binding(self):
        """Test kernel[axes] syntax."""
        @kernel
        def indexed_kernel(x):
            pass

        builder = indexed_kernel[1, 2]
        assert builder.axes == (1, 2)
        assert builder.kernel_ref is indexed_kernel


# ============================================================
# Test NPU Function Builder
# ============================================================

class TestNPUBuilder:
    """Tests for NPU function builder."""

    def test_basic_npu_function(self):
        """Test basic NPU function creation."""
        func = (npu("test_kernel")
            .tile("x", 32, 128, dtype=DType.F16)
            .tile("out", 32, 128, dtype=DType.F16)
            .memref("input", DType.F16, is_input=True)
            .memref("output", DType.F16, is_output=True)
            .load("x", "input")
            .store("output", "out")
            .build())

        assert isinstance(func, NPUFunction)
        assert func.name == "test_kernel"
        assert len(func.tiles) == 2
        assert len(func.memrefs) == 2
        assert len(func.ops) == 2

    def test_rmsnorm_kernel(self):
        """Test RMSNorm kernel pattern."""
        func = (npu("rmsnorm")
            .tile("x", 32, 128, dtype=DType.F16)
            .tile("out", 32, 128, dtype=DType.F16)
            .scalar("eps", DType.F32, default=1e-6)
            .memref("input", DType.F16, is_input=True)
            .memref("output", DType.F16, is_output=True)
            .load("x", "input")
            .mul("sq", "x", "x")
            .rowmean("mean", "sq")
            .rsqrt("rsqrt_val", "mean")
            .rowexpandmul("out", "x", "rsqrt_val")
            .store("output", "out")
            .build())

        assert func.name == "rmsnorm"
        assert len(func.tiles) == 2
        assert len(func.scalars) == 1
        # Verify operations
        op_kinds = [op.kind for op in func.ops]
        assert NPUOpKind.Load in op_kinds
        assert NPUOpKind.Mul in op_kinds
        assert NPUOpKind.RowMean in op_kinds
        assert NPUOpKind.Rsqrt in op_kinds
        assert NPUOpKind.Store in op_kinds

    def test_matmul_kernel(self):
        """Test MatMul kernel with cube flag."""
        func = (npu("qk_matmul")
            .tile("q", 32, 64, dtype=DType.F16)
            .tile("k", 64, 32, dtype=DType.F16)
            .tile("s", 32, 32, dtype=DType.F16, location=Location.L1)
            .memref("Q", DType.F16, is_input=True)
            .memref("K", DType.F16, is_input=True)
            .memref("S", DType.F16, is_output=True)
            .load("q", "Q")
            .load("k", "K")
            .matmul("s", "q", "k", transpose_b=True)
            .store("S", "s")
            .cube()  # Mark as cube kernel
            .build())

        assert func.name == "qk_matmul"
        assert func.is_cube == True
        # Verify matmul operation
        matmul_ops = [op for op in func.ops if op.kind == NPUOpKind.MatMul]
        assert len(matmul_ops) == 1
        assert matmul_ops[0].attrs["transpose_b"] == True

    def test_schedule_hints(self):
        """Test NPU function schedule hints."""
        func = (npu("pipelined")
            .tile("a", 32, 32)
            .double_buffer()
            .pipeline(depth=2)
            .tile_policy("row_major")
            .build())

        assert func.double_buffer == True
        assert func.pipeline_depth == 2
        assert func.tile_policy == "row_major"

    def test_control_flow(self):
        """Test NPU function control flow."""
        func = (npu("loop_kernel")
            .tile("a", 32, 32)
            .for_loop("i", 0, 4, 1)
            .load("a", "input")
            .end_for()
            .build())

        assert len(func.ops) == 3  # for_begin, load, for_end
        assert func.ops[0].kind == NPUOpKind.ForLoopBegin
        assert func.ops[2].kind == NPUOpKind.ForLoopEnd

    def test_to_ir_serialization(self):
        """Test NPU function IR serialization."""
        func = (npu("serialize_test")
            .tile("x", 32, 64, dtype=DType.F16)
            .scalar("alpha", DType.F32, default=1.0)
            .memref("input", DType.F16, is_input=True)
            .load("x", "input")
            .build())

        ir = func.to_ir()
        assert ir["name"] == "serialize_test"
        assert len(ir["tiles"]) == 1
        assert ir["tiles"][0]["rows"] == 32
        assert ir["tiles"][0]["cols"] == 64
        assert ir["scalars"][0]["default"] == 1.0


# ============================================================
# Test Axis Types
# ============================================================

class TestAxisTypes:
    """Tests for axis types."""

    def test_dense_static(self):
        """Test Dense static axis."""
        heads = Dense[8]()
        assert heads.size == 8

    def test_dense_dyn(self):
        """Test DenseDyn dynamic axis."""
        batch = DenseDyn(16)
        assert batch.size == 16

    def test_ragged(self):
        """Test Ragged axis."""
        tokens = Ragged(4, [10, 15, 8, 12])
        assert tokens.outer_size == 4
        assert tokens.length(0) == 10
        assert tokens.total() == 45

    def test_sparse(self):
        """Test Sparse axis."""
        # CSR format: 2 batches, batch 0 has indices [1,3], batch 1 has indices [0,2,4]
        routing = Sparse(2, [0, 2, 5], [1, 3, 0, 2, 4])
        assert routing.outer_size == 2
        assert routing.nnz() == 5
        assert routing.row_nnz(0) == 2
        assert routing[0] == [1, 3]
        assert routing[1] == [0, 2, 4]


# ============================================================
# Test Legacy Primitives
# ============================================================

class TestLegacyPrimitives:
    """Tests for legacy workload primitives."""

    def test_parallel_for(self):
        """Test parallel_for primitive."""
        batch = DenseDyn(4)
        w = parallel_for(batch, lambda b: task("kernel", [b], []))
        assert w._kind == "parallel_for"

    def test_for_each(self):
        """Test for_each primitive."""
        seq = DenseDyn(8)
        w = for_each(seq, lambda i: task("scan", [i], []))
        assert w._kind == "for_each"

    def test_task(self):
        """Test task primitive."""
        w = task("my_kernel", [1, 2], ["tensor1", "tensor2"])
        assert w._kind == "task"
        assert w._kwargs["kernel"] == "my_kernel"

    def test_combine(self):
        """Test combine primitive."""
        t1 = task("k1", [], [])
        t2 = task("k2", [], [])
        w = combine(t1, t2)
        assert w._kind == "combine"

    def test_sequential(self):
        """Test sequential primitive."""
        t1 = task("k1", [], [])
        t2 = task("k2", [], [])
        w = sequential(t1, t2)
        assert w._kind == "sequential"


# ============================================================
# Test Schedule Combinator API
# ============================================================

class TestScheduleAPI:
    """Tests for schedule combinator API."""

    def test_dispatch_policy(self):
        """Test dispatch policy."""
        w = task("kernel", [], [])
        w2 = w.dispatch(DispatchPolicy.round_robin(4))
        assert w2._dispatch_policy is not None
        # Verify immutability (original unchanged)
        assert w._dispatch_policy is None

    def test_streams(self):
        """Test streams configuration."""
        w = task("kernel", [], [])
        w2 = w.streams(4)
        assert w2._stream_count == 4

    def test_stream_by(self):
        """Test stream_by function."""
        w = task("kernel", [], [])
        w2 = w.stream_by(lambda t: t.params[0] % 2)
        assert w2._stream_by_fn is not None

    def test_timing(self):
        """Test timing policy."""
        w = task("kernel", [], [])
        w2 = w.timing(TimingPolicy.immediate)
        assert w2._timing_policy is not None

    def test_chained_schedule(self):
        """Test chained schedule API."""
        w = (task("kernel", [], [])
            .dispatch(DispatchPolicy.round_robin(4))
            .streams(2)
            .timing(TimingPolicy.immediate))
        assert w._dispatch_policy is not None
        assert w._stream_count == 2
        assert w._timing_policy is not None


# ============================================================
# Test Extended Schedule Primitives (R5)
# ============================================================

class TestExtendedSchedulePrimitives:
    """Tests for extended schedule primitives (R5)."""

    def test_dispatch_threshold(self):
        """Test dispatch_threshold for multi-level dispatch."""
        w = task("kernel", [], [])
        w2 = w.dispatch_threshold(
            thresholds=[256, 1024, 4096],
            policies={
                256: DispatchPolicy.round_robin(1),
                1024: DispatchPolicy.round_robin(4),
                4096: DispatchPolicy.work_steal(),
            }
        )
        assert w2._dispatch_threshold is not None
        assert w2._dispatch_threshold.thresholds == [256, 1024, 4096]
        # Test policy selection
        policy = w2._dispatch_threshold.select_policy(500)
        assert policy is not None

    def test_pipeline_depth(self):
        """Test pipeline_depth for in-flight control."""
        w = task("kernel", [], [])
        w2 = w.pipeline_depth(2, scope="per_stream")
        assert w2._pipeline_depth is not None
        assert w2._pipeline_depth.depth == 2
        from pto_wsp.schedule import GateScope
        assert w2._pipeline_depth.scope == GateScope.PER_STREAM

    def test_pipeline_depth_global(self):
        """Test pipeline_depth with global scope."""
        w = task("kernel", [], []).pipeline_depth(3)
        assert w._pipeline_depth.depth == 3
        from pto_wsp.schedule import GateScope
        assert w._pipeline_depth.scope == GateScope.GLOBAL

    def test_task_window(self):
        """Test task_window for metadata management."""
        w = task("kernel", [], [])
        w2 = w.task_window(8192, unit="tasks", mode="stall")
        assert w2._task_window is not None
        assert w2._task_window.size == 8192
        assert w2._task_window.unit == "tasks"
        from pto_wsp.schedule import WindowMode
        assert w2._task_window.mode == WindowMode.STALL

    def test_task_window_benchmark_mode(self):
        """Test task_window with benchmark mode."""
        w = task("kernel", [], []).task_window(4096, mode="benchmark")
        from pto_wsp.schedule import WindowMode
        assert w._task_window.mode == WindowMode.BENCHMARK

    def test_batch_deps(self):
        """Test batch_deps for batched dependency resolution."""
        w = task("kernel", [], [])
        w2 = w.batch_deps(64, range_compression=True)
        assert w2._batch_deps is not None
        assert w2._batch_deps.threshold == 64
        assert w2._batch_deps.range_compression == True

    def test_batch_deps_default(self):
        """Test batch_deps with default range_compression."""
        w = task("kernel", [], []).batch_deps(128)
        assert w._batch_deps.threshold == 128
        assert w._batch_deps.range_compression == False

    def test_chained_extended_schedule(self):
        """Test chaining extended schedule primitives."""
        w = (task("kernel", [], [])
            .dispatch(DispatchPolicy.round_robin(4))
            .streams(2)
            .timing(TimingPolicy.immediate)
            .pipeline_depth(2, scope="per_stream")
            .task_window(8192)
            .batch_deps(64))

        assert w._dispatch_policy is not None
        assert w._stream_count == 2
        assert w._timing_policy is not None
        assert w._pipeline_depth is not None
        assert w._task_window is not None
        assert w._batch_deps is not None

    def test_schedule_primitives_immutable(self):
        """Test that schedule methods return new Workload instances."""
        w1 = task("kernel", [], [])
        w2 = w1.pipeline_depth(2)
        w3 = w2.task_window(4096)

        # Original unchanged
        assert w1._pipeline_depth is None
        assert w1._task_window is None
        # w2 has pipeline_depth but not task_window
        assert w2._pipeline_depth is not None
        assert w2._task_window is None
        # w3 has both
        assert w3._pipeline_depth is not None
        assert w3._task_window is not None


# ============================================================
# Test Task Graph Primitives (R9)
# ============================================================

class TestTaskGraphPrimitives:
    """Tests for task graph primitives (R9)."""

    def test_deps_infer_tensor_map_exact(self):
        """Test Deps.infer_tensor_map_exact() (pto-isa-lh compatible)."""
        deps = Deps.infer_tensor_map_exact()
        assert deps.mode == DepsMode.INFER_TENSOR_MAP_EXACT

    def test_deps_infer_bytes_overlap(self):
        """Test Deps.infer_bytes_overlap() (v9 extension)."""
        deps = Deps.infer_bytes_overlap()
        assert deps.mode == DepsMode.INFER_BYTES_OVERLAP

    def test_deps_explicit(self):
        """Test Deps.explicit() for manual dependencies."""
        deps = Deps.explicit()
        assert deps.mode == DepsMode.EXPLICIT

    def test_deps_hybrid(self):
        """Test Deps.hybrid() for combined inference + explicit."""
        deps = Deps.hybrid(infer=DepsMode.INFER_TENSOR_MAP_EXACT, explicit=True)
        assert deps.mode == DepsMode.HYBRID
        assert deps._kwargs["infer"] == DepsMode.INFER_TENSOR_MAP_EXACT
        assert deps._kwargs["explicit"] == True

    def test_ready_policy_fifo(self):
        """Test ReadyPolicy.fifo()."""
        policy = ReadyPolicy.fifo()
        assert policy._kind == "fifo"

    def test_ready_policy_work_steal(self):
        """Test ReadyPolicy.work_steal()."""
        policy = ReadyPolicy.work_steal()
        assert policy._kind == "work_steal"

    def test_ready_policy_priority(self):
        """Test ReadyPolicy.priority() with custom function."""
        policy = ReadyPolicy.priority(lambda t: t.id)
        assert policy._kind == "priority"
        assert policy._kwargs["priority_fn"] is not None

    def test_start_policy_threshold(self):
        """Test StartPolicy.threshold() for pipelined execution."""
        policy = StartPolicy.threshold(100)
        assert policy._kind == "threshold"
        assert policy._kwargs["n"] == 100

    def test_start_policy_after_orchestration(self):
        """Test StartPolicy.after_orchestration()."""
        policy = StartPolicy.after_orchestration()
        assert policy._kind == "after_orchestration"

    def test_trace_policy_none(self):
        """Test TracePolicy.none() for no tracing."""
        policy = TracePolicy.none()
        assert policy._kind == "none"

    def test_trace_policy_cycles(self):
        """Test TracePolicy.cycles() for cycle-level simulation."""
        policy = TracePolicy.cycles(lambda t: 100)
        assert policy._kind == "cycles"
        assert policy._kwargs["cost_fn"] is not None

    def test_pools_single(self):
        """Test Pools.single() for unified queue."""
        pools = Pools.single()
        assert pools._kind == "single"

    def test_pools_by_exec_unit(self):
        """Test Pools.by_exec_unit() for vector/cube separation."""
        pools = Pools.by_exec_unit()
        assert pools._kind == "by_exec_unit"

    def test_pools_custom(self):
        """Test Pools.custom() with routing function."""
        pools = Pools.custom(lambda t: "vector" if t.is_cube else "cube")
        assert pools._kind == "custom"
        assert pools._kwargs["pool_fn"] is not None

    def test_task_graph_config_defaults(self):
        """Test TaskGraphConfig with default values."""
        config = TaskGraphConfig()
        assert config.deps.mode == DepsMode.INFER_TENSOR_MAP_EXACT
        assert config.window.size == 8192
        assert config.window.mode == WindowMode.STALL
        assert config.pools._kind == "single"
        assert config.ready._kind == "fifo"
        assert config.start._kind == "after_orchestration"
        assert config.trace._kind == "none"

    def test_task_graph_config_custom(self):
        """Test TaskGraphConfig with custom values."""
        config = TaskGraphConfig(
            deps=Deps.hybrid(),
            window=TaskWindow(4096, "tasks", WindowMode.ABORT),
            pools=Pools.by_exec_unit(),
            ready=ReadyPolicy.work_steal(),
            start=StartPolicy.threshold(50),
            trace=TracePolicy.cycles()
        )
        assert config.deps.mode == DepsMode.HYBRID
        assert config.window.size == 4096
        assert config.window.mode == WindowMode.ABORT
        assert config.pools._kind == "by_exec_unit"
        assert config.ready._kind == "work_steal"
        assert config.start._kind == "threshold"
        assert config.trace._kind == "cycles"


class TestWorkloadTaskGraph:
    """Tests for Workload.task_graph() method (R9)."""

    def test_basic_task_graph(self):
        """Test basic task_graph() call."""
        w = task("kernel", [], [])
        w2 = w.task_graph()
        assert w2._task_graph_config is not None
        # Original unchanged
        assert w._task_graph_config is None

    def test_task_graph_clears_streams(self):
        """Test that task_graph() clears stream-based settings."""
        w = task("kernel", [], []).streams(4).stream_by(lambda t: t.id % 4)
        assert w._stream_count == 4
        assert w._stream_by_fn is not None

        w2 = w.task_graph()
        assert w2._stream_count is None
        assert w2._stream_by_fn is None
        assert w2._task_graph_config is not None

    def test_task_graph_with_deps(self):
        """Test task_graph() with custom deps."""
        w = task("kernel", [], []).task_graph(deps=Deps.hybrid())
        assert w._task_graph_config.deps.mode == DepsMode.HYBRID

    def test_task_graph_with_window(self):
        """Test task_graph() with custom window."""
        w = task("kernel", [], []).task_graph(
            window=TaskWindow(4096, "tasks", WindowMode.ABORT)
        )
        assert w._task_graph_config.window.size == 4096
        assert w._task_graph_config.window.mode == WindowMode.ABORT

    def test_task_graph_with_pools(self):
        """Test task_graph() with custom pools."""
        w = task("kernel", [], []).task_graph(pools=Pools.by_exec_unit())
        assert w._task_graph_config.pools._kind == "by_exec_unit"

    def test_task_graph_with_ready(self):
        """Test task_graph() with custom ready policy."""
        w = task("kernel", [], []).task_graph(ready=ReadyPolicy.work_steal())
        assert w._task_graph_config.ready._kind == "work_steal"

    def test_task_graph_with_start(self):
        """Test task_graph() with custom start policy."""
        w = task("kernel", [], []).task_graph(start=StartPolicy.threshold(100))
        assert w._task_graph_config.start._kind == "threshold"
        assert w._task_graph_config.start._kwargs["n"] == 100

    def test_task_graph_with_trace(self):
        """Test task_graph() with tracing enabled."""
        w = task("kernel", [], []).task_graph(trace=TracePolicy.cycles())
        assert w._task_graph_config.trace._kind == "cycles"

    def test_task_graph_full_config(self):
        """Test task_graph() with all options."""
        w = (task("kernel", [], [])
            .dispatch(DispatchPolicy.work_steal())
            .task_graph(
                deps=Deps.infer_tensor_map_exact(),
                window=TaskWindow(8192, "tasks", WindowMode.STALL),
                pools=Pools.by_exec_unit(),
                ready=ReadyPolicy.work_steal(),
                start=StartPolicy.threshold(100),
                trace=TracePolicy.cycles()
            ))

        assert w._dispatch_policy is not None
        assert w._task_graph_config is not None
        config = w._task_graph_config
        assert config.deps.mode == DepsMode.INFER_TENSOR_MAP_EXACT
        assert config.window.size == 8192
        assert config.pools._kind == "by_exec_unit"
        assert config.ready._kind == "work_steal"
        assert config.start._kwargs["n"] == 100
        assert config.trace._kind == "cycles"

    def test_task_graph_immutable(self):
        """Test that task_graph() returns a new Workload instance."""
        w1 = task("kernel", [], [])
        w2 = w1.task_graph()
        w3 = w2.task_graph(deps=Deps.explicit())

        # w1 unchanged
        assert w1._task_graph_config is None
        # w2 has default config
        assert w2._task_graph_config is not None
        assert w2._task_graph_config.deps.mode == DepsMode.INFER_TENSOR_MAP_EXACT
        # w3 has explicit deps
        assert w3._task_graph_config.deps.mode == DepsMode.EXPLICIT


# ============================================================
# Test CSP Primitives
# ============================================================

class TestCSPPrimitives:
    """Tests for CSP primitives."""

    def test_channel_creation(self):
        """Test Channel creation."""
        ch = Channel("data", depth=2)
        assert ch.name == "data"
        assert ch.depth == 2
        assert ch.is_open()

    def test_channel_buffering(self):
        """Test Channel buffer operations."""
        ch = Channel("test", depth=2)
        assert ch.empty()
        assert not ch.full()

    def test_process_builder(self):
        """Test process builder API."""
        ch_in = Channel("input", depth=2)
        ch_out = Channel("output", depth=2)

        proc = (process("compute")
            .consumes(ch_in)
            .produces(ch_out)
            .body(task("process", [], [])))

        assert proc.name == "compute"
        assert ch_in in proc.consumes
        assert ch_out in proc.produces

    def test_send_workload(self):
        """Test send creates workload."""
        ch = Channel("ch", depth=2)
        w = send(ch, task("loader", [], []))
        assert w._kind == "send"

    def test_consume_workload(self):
        """Test consume creates workload."""
        ch = Channel("ch", depth=2)
        w = consume(ch, lambda t: task("process", [t], []))
        assert w._kind == "consume"

    def test_connect_pipeline(self):
        """Test connect creates pipeline."""
        ch = Channel("ch", depth=2)
        p1 = (process("loader").produces(ch).body(task("load", [], [])))
        p2 = (process("compute").consumes(ch).body(task("compute", [], [])))
        pipeline = connect([p1, p2], [ch])
        assert pipeline._kind == "connect"


# ============================================================
# Test Integration
# ============================================================

class TestIntegration:
    """Integration tests for complete workflows."""

    def test_attention_workload(self):
        """Test complete attention workload definition."""
        @kernel
        def attention(Q: In[Tensor], K: In[Tensor], V: In[Tensor], O: Out[Tensor]):
            pass

        @workload
        def multi_head_attention(batch, heads):
            for b, h in P(batch, heads):
                attention[b, h](None, None, None, None)

        batch = DenseDyn(4)
        heads = Dense[8]()
        w = multi_head_attention(batch, heads)
        assert w is not None

    def test_attention_with_schedule(self):
        """Test attention workload with schedule."""
        batch = DenseDyn(4)
        w = (parallel_for(batch, lambda b: task("attn", [b], []))
            .dispatch(DispatchPolicy.round_robin(4))
            .streams(2)
            .timing(TimingPolicy.immediate))

        assert w._dispatch_policy is not None
        assert w._stream_count == 2

    def test_npu_kernel_integration(self):
        """Test NPU kernel with workload."""
        # Define NPU kernel
        rmsnorm_kernel = (npu("rmsnorm")
            .tile("x", 32, 128)
            .load("x", "input")
            .rowmean("mean", "x")
            .rsqrt("rsqrt", "mean")
            .rowexpandmul("out", "x", "rsqrt")
            .store("output", "out")
            .build())

        # Use in workload
        @kernel
        def rmsnorm(x: In[Tensor], out: Out[Tensor]):
            # This would reference rmsnorm_kernel in a real implementation
            pass

        @workload
        def batch_rmsnorm(batch):
            for b in P(batch):
                rmsnorm[b](None, None)

        w = batch_rmsnorm(DenseDyn(4))
        assert w is not None


# ============================================================
# Test Type Checker
# ============================================================

class TestWorkloadTypeCheckIntegration:
    """Tests for type checking integration with @workload decorator."""

    def test_workload_type_check_enabled_by_default(self):
        """Test that type checking is enabled by default."""
        @workload
        def simple_workload():
            pass

        w = simple_workload()
        # Type errors list should be empty (no errors)
        assert hasattr(w, '_type_errors')
        assert w._type_errors == []

    def test_workload_detects_missing_argument(self):
        """Test that type checking detects missing arguments."""
        @kernel
        def my_kernel(a: In[Tensor], b: In[Tensor]):
            pass

        @workload
        def bad_workload():
            # Call with missing argument (should trigger type error)
            my_kernel(a=None)  # missing b

        w = bad_workload()
        # Should have collected type error about missing argument
        assert w.has_type_errors()
        errors = w.type_errors
        assert len(errors) >= 1

    def test_workload_detects_arity_mismatch(self):
        """Test that type checking detects arity mismatch."""
        @kernel
        def two_arg_kernel(x: In[Tensor], y: Out[Tensor]):
            pass

        @workload
        def workload_arity_mismatch():
            # Call with wrong number of args
            two_arg_kernel(x=None, y=None, z=None)  # extra arg

        w = workload_arity_mismatch()
        assert w.has_type_errors()

    def test_workload_type_check_disabled(self):
        """Test that type checking can be disabled."""
        @kernel
        def strict_kernel(a: In[Tensor]):
            pass

        @workload(type_check=False)
        def no_check_workload():
            # This would fail type check but checking is disabled
            strict_kernel()  # missing a

        w = no_check_workload()
        # Should NOT have collected errors since checking was disabled
        assert not w.has_type_errors()

    def test_workload_fail_on_type_error(self):
        """Test that fail_on_type_error raises TypeError."""
        @kernel
        def fail_kernel(x: In[Tensor]):
            pass

        @workload(type_check=True, fail_on_type_error=True)
        def fail_workload():
            fail_kernel()  # missing x

        # Should raise TypeError due to fail_on_type_error=True
        try:
            fail_workload()
            assert False, "Should have raised TypeError"
        except TypeError as e:
            assert "Type errors in workload" in str(e)

    def test_workload_valid_kernel_call(self):
        """Test that valid kernel calls pass type checking."""
        @kernel
        def valid_kernel(x: In[Tensor], y: Out[Tensor]):
            pass

        t1 = Tensor(data=None, shape=(16, 32), dtype=DType.F16)
        t2 = Tensor(data=None, shape=(16, 32), dtype=DType.F16)

        @workload
        def valid_workload():
            valid_kernel(x=t1, y=t2)

        w = valid_workload()
        # Should pass type checking - no errors
        assert not w.has_type_errors()

    def test_workload_with_axis_binding_type_check(self):
        """Test type checking with axis-bound kernel calls."""
        @kernel
        def indexed_kernel(q: In[Tensor], k: In[Tensor]):
            pass

        batch = Dense[4]()
        q = Tensor(data=None, shape=(4, 64), dtype=DType.F16)
        k = Tensor(data=None, shape=(4, 64), dtype=DType.F16)

        @workload
        def indexed_workload():
            for b in P(batch):
                indexed_kernel[b](q=q, k=k)

        w = indexed_workload()
        # Should pass - correct args with axis binding
        assert not w.has_type_errors()


class TestTypeChecker:
    """Tests for type checker."""

    def test_layout_join_replicate(self):
        """Test R ⊔ R = R."""
        a = Layout((LayoutReplicate(), LayoutReplicate()))
        b = Layout((LayoutReplicate(), LayoutReplicate()))
        result = layout_join(a, b)
        assert all(isinstance(d, LayoutReplicate) for d in result.dist)

    def test_layout_join_replicate_shard(self):
        """Test R ⊔ S(i) = S(i)."""
        a = Layout((LayoutReplicate(), LayoutReplicate()))
        b = Layout((LayoutShard(0), LayoutReplicate()))
        result = layout_join(a, b)
        assert isinstance(result.dist[0], LayoutShard)
        assert result.dist[0].mesh_axis == 0

    def test_layout_join_shard_shard_same(self):
        """Test S(i) ⊔ S(i) = S(i)."""
        a = Layout((LayoutShard(0), LayoutReplicate()))
        b = Layout((LayoutShard(0), LayoutReplicate()))
        result = layout_join(a, b)
        assert isinstance(result.dist[0], LayoutShard)
        assert result.dist[0].mesh_axis == 0

    def test_layout_join_shard_shard_different(self):
        """Test S(i) ⊔ S(j) = error if i ≠ j."""
        a = Layout((LayoutShard(0), LayoutReplicate()))
        b = Layout((LayoutShard(1), LayoutReplicate()))
        with pytest.raises(LayoutCompatibilityError):
            layout_join(a, b)

    def test_layout_join_rank_mismatch(self):
        """Test layout join with rank mismatch."""
        a = Layout((LayoutReplicate(),))
        b = Layout((LayoutReplicate(), LayoutReplicate()))
        with pytest.raises(LayoutCompatibilityError):
            layout_join(a, b)

    def test_type_checker_axis_bounds(self):
        """Test axis bounds checking."""
        checker = TypeChecker()
        axis = Dense[8]()

        # Valid index
        assert checker.check_axis_bounds(axis, 5)
        assert not checker.has_errors()

        # Invalid index (out of bounds)
        checker.check_axis_bounds(axis, 10)
        assert checker.has_errors()
        assert checker.errors[0].kind == TypeErrorKind.AXIS_OUT_OF_BOUNDS

    def test_type_checker_kernel_call(self):
        """Test kernel call type checking."""
        @kernel
        def my_kernel(a: In[Tensor], b: Out[Tensor]):
            pass

        checker = TypeChecker()
        # Valid call (with None placeholders - accepted for now)
        result = checker.check_kernel_call(my_kernel, (), {"a": None, "b": None})
        assert result  # No errors

    def test_type_checker_missing_argument(self):
        """Test missing argument detection."""
        @kernel
        def my_kernel(a: In[Tensor], b: Out[Tensor]):
            pass

        checker = TypeChecker()
        checker.check_kernel_call(my_kernel, (), {"a": None})
        assert checker.has_errors()
        # Note: Arity mismatch is caught first (1 vs 2 args)
        assert any(e.kind == TypeErrorKind.ARITY_MISMATCH for e in checker.errors)

    def test_type_checker_arity_mismatch(self):
        """Test arity mismatch detection."""
        @kernel
        def my_kernel(a: In[Tensor], b: Out[Tensor]):
            pass

        checker = TypeChecker()
        checker.check_kernel_call(my_kernel, (), {"a": None, "b": None, "c": None})
        assert checker.has_errors()
        assert any(e.kind == TypeErrorKind.ARITY_MISMATCH for e in checker.errors)

    def test_validate_axis_index(self):
        """Test convenience function for axis index validation."""
        axis = Dense[4]()
        assert validate_axis_index(axis, 3)  # Valid

        with pytest.raises(TypeError):
            validate_axis_index(axis, 5)  # Out of bounds

    def test_check_layouts_compatible(self):
        """Test convenience function for layout compatibility."""
        # Create tensors with layouts (mock)
        t1 = Tensor(None, (4, 4), DType.F16)
        t1.layout = Layout((LayoutReplicate(), LayoutReplicate()))

        t2 = Tensor(None, (4, 4), DType.F16)
        t2.layout = Layout((LayoutShard(0), LayoutReplicate()))

        result = check_layouts_compatible(t1, t2)
        assert result is not None
        assert isinstance(result.dist[0], LayoutShard)


# ============================================================
# Test Layout as Refinement Types (R10)
# ============================================================

class TestMemLayout:
    """Tests for MemLayout (Triton-style memory layout)."""

    def test_row_major(self):
        """Test row-major layout creation."""
        layout = MemLayout.row_major((4, 8))
        assert layout.strides == (8, 1)
        assert layout.order == (0, 1)

    def test_col_major(self):
        """Test column-major layout creation."""
        layout = MemLayout.col_major((4, 8))
        assert layout.strides == (1, 4)
        assert layout.order == (1, 0)

    def test_permute(self):
        """Test layout permutation."""
        layout = MemLayout.row_major((4, 8, 16))
        permuted = layout.permute((2, 0, 1))
        assert permuted.strides == (1, 128, 16)

    def test_compose(self):
        """Test layout composition."""
        l1 = MemLayout(strides=(8, 1))
        l2 = MemLayout(swizzle="XOR")
        composed = l1.compose(l2)
        assert composed.strides == (8, 1)
        assert composed.swizzle == "XOR"


class TestTensorLayout:
    """Tests for TensorLayout (unified distribution + memory)."""

    def test_default_replicated(self):
        """Test default replicated layout."""
        layout = TensorLayout.default(3)
        assert len(layout.dist) == 3
        assert all(isinstance(d, TensorReplicate) for d in layout.dist)
        assert layout.is_replicated()

    def test_sharded(self):
        """Test sharded layout creation."""
        layout = TensorLayout.sharded(dim=0, rank=2, mesh_axis=0)
        assert isinstance(layout.dist[0], TensorShard)
        assert layout.dist[0].mesh_axis == 0
        assert isinstance(layout.dist[1], TensorReplicate)
        assert not layout.is_replicated()
        assert layout.sharded_dims() == [0]

    def test_with_mem_layout(self):
        """Test TensorLayout with memory layout."""
        layout = TensorLayout(
            dist=(TensorShard(0), TensorReplicate()),
            mem=MemLayout.row_major((4, 8))
        )
        assert layout.mem is not None
        assert layout.mem.strides == (8, 1)


class TestTensorLayoutJoin:
    """Tests for tensor_layout_join (Dato-style join rules)."""

    def test_join_replicate_replicate(self):
        """Test R ⊔ R = R."""
        a = TensorLayout((TensorReplicate(), TensorReplicate()))
        b = TensorLayout((TensorReplicate(), TensorReplicate()))
        result = tensor_layout_join(a, b)
        assert all(isinstance(d, TensorReplicate) for d in result.dist)

    def test_join_replicate_shard(self):
        """Test R ⊔ S(i) = S(i)."""
        a = TensorLayout((TensorReplicate(), TensorReplicate()))
        b = TensorLayout((TensorShard(0), TensorReplicate()))
        result = tensor_layout_join(a, b)
        assert isinstance(result.dist[0], TensorShard)
        assert result.dist[0].mesh_axis == 0

    def test_join_shard_replicate(self):
        """Test S(i) ⊔ R = S(i)."""
        a = TensorLayout((TensorShard(0), TensorReplicate()))
        b = TensorLayout((TensorReplicate(), TensorReplicate()))
        result = tensor_layout_join(a, b)
        assert isinstance(result.dist[0], TensorShard)

    def test_join_shard_shard_same(self):
        """Test S(i) ⊔ S(i) = S(i)."""
        a = TensorLayout((TensorShard(0), TensorReplicate()))
        b = TensorLayout((TensorShard(0), TensorReplicate()))
        result = tensor_layout_join(a, b)
        assert isinstance(result.dist[0], TensorShard)
        assert result.dist[0].mesh_axis == 0

    def test_join_shard_shard_different_error(self):
        """Test S(i) ⊔ S(j) = error if i ≠ j."""
        a = TensorLayout((TensorShard(0), TensorReplicate()))
        b = TensorLayout((TensorShard(1), TensorReplicate()))
        with pytest.raises(LayoutIncompatibleError):
            tensor_layout_join(a, b)

    def test_join_rank_mismatch_error(self):
        """Test join with rank mismatch."""
        a = TensorLayout((TensorReplicate(),))
        b = TensorLayout((TensorReplicate(), TensorReplicate()))
        with pytest.raises(LayoutIncompatibleError):
            tensor_layout_join(a, b)


class TestTensorWithLayout:
    """Tests for Tensor with layout refinement (R10)."""

    def test_tensor_default_layout(self):
        """Test tensor gets default replicated layout."""
        t = Tensor(data=None, shape=(4, 8), dtype=DType.F16)
        assert t.layout is not None
        assert t.layout.is_replicated()

    def test_tensor_explicit_layout(self):
        """Test tensor with explicit layout."""
        layout = TensorLayout.sharded(dim=0, rank=2, mesh_axis=0)
        t = Tensor(data=None, shape=(4, 8), dtype=DType.F16, layout=layout)
        assert not t.layout.is_replicated()
        assert t.layout.sharded_dims() == [0]

    def test_tensor_with_layout_method(self):
        """Test Tensor.with_layout() returns new tensor."""
        t1 = Tensor(data=None, shape=(4, 8), dtype=DType.F16)
        layout = TensorLayout.sharded(dim=0, rank=2, mesh_axis=0)
        t2 = t1.with_layout(layout)
        # Original unchanged
        assert t1.layout.is_replicated()
        # New tensor has new layout
        assert not t2.layout.is_replicated()


class TestLayoutTransitionOperations:
    """Tests for layout transition operations (R10)."""

    def test_relayout(self):
        """Test relayout() for explicit redistribution."""
        t1 = Tensor(data=None, shape=(4, 8), dtype=DType.F16)
        assert t1.layout.is_replicated()

        new_layout = TensorLayout.sharded(dim=0, rank=2, mesh_axis=0)
        t2 = relayout(t1, new_layout)
        assert not t2.layout.is_replicated()
        assert t2.layout.sharded_dims() == [0]

    def test_allreduce(self):
        """Test allreduce() collective."""
        layout = TensorLayout.sharded(dim=0, rank=2, mesh_axis=0)
        t1 = Tensor(data=None, shape=(4, 8), dtype=DType.F16, layout=layout)
        assert not t1.layout.is_replicated()

        t2 = allreduce(t1, mesh_axis=0)
        assert t2.layout.is_replicated()

    def test_allgather(self):
        """Test allgather() collective."""
        layout = TensorLayout.sharded(dim=0, rank=2, mesh_axis=0)
        t1 = Tensor(data=None, shape=(4, 8), dtype=DType.F16, layout=layout)

        t2 = allgather(t1, dim=0, mesh_axis=0)
        assert isinstance(t2.layout.dist[0], TensorReplicate)

    def test_reduce_scatter(self):
        """Test reduce_scatter() collective."""
        t1 = Tensor(data=None, shape=(4, 8), dtype=DType.F16)
        assert t1.layout.is_replicated()

        t2 = reduce_scatter(t1, dim=0, mesh_axis=0)
        assert isinstance(t2.layout.dist[0], TensorShard)
        assert t2.layout.dist[0].mesh_axis == 0


class TestTensorMethods:
    """Tests for Tensor methods (indexing, slicing, nbytes)."""

    def test_tensor_getitem_reduces_rank(self):
        """Test that indexing reduces tensor rank."""
        t = Tensor(data=None, shape=(4, 8, 16), dtype=DType.F32)
        t_indexed = t[0]
        assert t_indexed.shape == (8, 16)

    def test_tensor_getitem_propagates_layout(self):
        """Test that indexing propagates layout correctly."""
        layout = TensorLayout(
            dist=(TensorShard(0), TensorReplicate(), TensorReplicate())
        )
        t = Tensor(data=None, shape=(4, 8, 16), dtype=DType.F32, layout=layout)
        t_indexed = t[0]
        # First dimension removed, so layout should have 2 dims
        assert len(t_indexed.layout.dist) == 2
        # The sharded dimension was removed, remaining should be replicated
        assert isinstance(t_indexed.layout.dist[0], TensorReplicate)
        assert isinstance(t_indexed.layout.dist[1], TensorReplicate)

    def test_tensor_slice_preserves_rank(self):
        """Test that slicing preserves tensor rank."""
        t = Tensor(data=None, shape=(10, 8), dtype=DType.F32)
        t_sliced = t.slice(2, 6)
        assert len(t_sliced.shape) == 2
        assert t_sliced.shape == (4, 8)

    def test_tensor_slice_preserves_layout(self):
        """Test that slicing preserves layout."""
        layout = TensorLayout.sharded(dim=0, rank=2, mesh_axis=0)
        t = Tensor(data=None, shape=(10, 8), dtype=DType.F32, layout=layout)
        t_sliced = t.slice(2, 6)
        # Layout should be preserved
        assert t_sliced.layout == t.layout

    def test_tensor_nbytes_fp32(self):
        """Test nbytes calculation for FP32."""
        t = Tensor(data=None, shape=(4, 8), dtype=DType.F32)
        # 4 * 8 * 4 bytes = 128
        assert t.nbytes() == 128

    def test_tensor_nbytes_fp16(self):
        """Test nbytes calculation for FP16."""
        t = Tensor(data=None, shape=(4, 8), dtype=DType.F16)
        # 4 * 8 * 2 bytes = 64
        assert t.nbytes() == 64

    def test_tensor_nbytes_int8(self):
        """Test nbytes calculation for INT8."""
        t = Tensor(data=None, shape=(4, 8, 16), dtype=DType.I8)
        # 4 * 8 * 16 * 1 byte = 512
        assert t.nbytes() == 512

    def test_tensor_nbytes_empty(self):
        """Test nbytes returns 0 for empty shape."""
        t = Tensor(data=None, shape=(), dtype=DType.F32)
        assert t.nbytes() == 0


class TestWorkloadLayoutDeprecation:
    """Tests for deprecated Workload.layout() method (R10)."""

    def test_layout_method_warns(self):
        """Test that Workload.layout() emits deprecation warning."""
        import warnings
        t = Tensor(data=None, shape=(4, 8), dtype=DType.F16)
        w = task("kernel", [], [])

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            w.layout(t, TensorShard(0), TensorReplicate())
            assert len(caught) == 1
            assert issubclass(caught[0].category, DeprecationWarning)
            assert "deprecated" in str(caught[0].message).lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
