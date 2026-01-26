"""
Python↔C++ Integration Tests via pybind11 (L5).

These tests verify that the C++ IR classes are correctly exposed
to Python through the pybind11 bindings in pto_ir_bindings.cpp.

Test Categories:
- IR node creation via IRFactory
- Module parsing and round-trip
- Type checking across Python/C++ boundary
- Backend compilation flow
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build'))

import pytest


# Try to import the C++ module
try:
    import pto_ir_cpp as cpp
    HAS_CPP_MODULE = True
except ImportError as e:
    HAS_CPP_MODULE = False
    IMPORT_ERROR = str(e)


@pytest.mark.skipif(not HAS_CPP_MODULE, reason="C++ module not built")
class TestIRNodeCreation:
    """Test IR node creation through pybind11 bindings."""

    def test_ir_factory_create_dense_axis(self):
        """Test creating DenseAxisNode via IRFactory."""
        factory = cpp.IRFactory()
        axis = factory.create_dense_axis(8)

        assert axis is not None
        assert axis.size == 8
        assert axis.kind == cpp.NodeKind.DenseAxis

    def test_ir_factory_create_dense_dyn_axis(self):
        """Test creating DenseDynAxisNode via IRFactory."""
        factory = cpp.IRFactory()
        axis = factory.create_dense_dyn_axis("batch")

        assert axis is not None
        assert axis.size_var == "batch"
        assert axis.kind == cpp.NodeKind.DenseDynAxis

    def test_ir_factory_create_ragged_axis(self):
        """Test creating RaggedAxisNode via IRFactory."""
        factory = cpp.IRFactory()
        axis = factory.create_ragged_axis("outer", "lengths")

        assert axis is not None
        assert axis.outer_size_var == "outer"
        assert axis.lengths_var == "lengths"
        assert axis.kind == cpp.NodeKind.RaggedAxis

    def test_ir_factory_create_sparse_axis(self):
        """Test creating SparseAxisNode via IRFactory."""
        factory = cpp.IRFactory()
        axis = factory.create_sparse_axis("outer", "indptr", "indices")

        assert axis is not None
        assert axis.outer_size_var == "outer"
        assert axis.indptr_var == "indptr"
        assert axis.indices_var == "indices"
        assert axis.kind == cpp.NodeKind.SparseAxis

    def test_ir_factory_create_task(self):
        """Test creating TaskNode via IRFactory."""
        factory = cpp.IRFactory()
        task = factory.create_task("kernel", ["i", "j"], ["input", "output"])

        assert task is not None
        assert task.kernel_name == "kernel"
        assert task.params == ["i", "j"]
        assert task.resources == ["input", "output"]
        assert task.kind == cpp.NodeKind.Task

    def test_ir_factory_create_parallel_for(self):
        """Test creating ParallelForNode via IRFactory."""
        factory = cpp.IRFactory()
        axis = factory.create_dense_axis(4)
        task = factory.create_task("kernel", ["i"], ["data"])
        pfor = factory.create_parallel_for(axis, "i", task)

        assert pfor is not None
        assert pfor.index_var == "i"
        assert pfor.kind == cpp.NodeKind.ParallelFor

    def test_ir_factory_create_for_each(self):
        """Test creating ForEachNode via IRFactory."""
        factory = cpp.IRFactory()
        axis = factory.create_dense_axis(4)
        task = factory.create_task("kernel", ["i"], ["data"])
        foreach = factory.create_for_each(axis, "i", task)

        assert foreach is not None
        assert foreach.index_var == "i"
        assert foreach.kind == cpp.NodeKind.ForEach


@pytest.mark.skipif(not HAS_CPP_MODULE, reason="C++ module not built")
class TestScheduleNodes:
    """Test schedule node creation and properties."""

    def test_create_dispatch_node(self):
        """Test creating DispatchNode."""
        factory = cpp.IRFactory()
        dispatch = factory.create_dispatch(cpp.DispatchPolicy.RoundRobin, 4, "")

        assert dispatch is not None
        assert dispatch.policy == cpp.DispatchPolicy.RoundRobin
        assert dispatch.num_targets == 4

    def test_create_stream_node(self):
        """Test creating StreamNode."""
        factory = cpp.IRFactory()
        stream = factory.create_stream(2, "i % 2")

        assert stream is not None
        assert stream.num_streams == 2
        assert stream.key_expr == "i % 2"

    def test_create_timing_node(self):
        """Test creating TimingNode."""
        factory = cpp.IRFactory()
        timing = factory.create_timing(cpp.TimingPolicy.Immediate, 0)

        assert timing is not None
        assert timing.policy == cpp.TimingPolicy.Immediate


@pytest.mark.skipif(not HAS_CPP_MODULE, reason="C++ module not built")
class TestModuleParsing:
    """Test module parsing through pybind11."""

    def test_parse_simple_workload(self):
        """Test parsing simple workload from string."""
        source = """@workload simple(%N: Dense[8]) {
  parallel_for i in Dense[8] {
    task @kernel(i) resources(input, output)
  }
}"""
        module = cpp.parse_string(source)

        assert module is not None
        assert len(module.workloads) == 1
        assert module.workloads[0].name == "simple"

    def test_parse_with_schedule(self):
        """Test parsing workload with schedule."""
        source = """@workload compute(%N: Dense[4]) {
  parallel_for i in Dense[4] {
    task @kernel(i) resources(data)
  }
}

@schedule compute_sched for compute {
  dispatch = round_robin(4)
  streams = 2
  timing = immediate
}"""
        module = cpp.parse_string(source)

        assert len(module.workloads) == 1
        assert len(module.schedules) == 1
        assert module.schedules[0].name == "compute_sched"
        assert module.schedules[0].workload_name == "compute"

    def test_module_round_trip(self):
        """Test parse -> print -> parse round-trip."""
        source = """@workload test(%N: Dense[4]) {
  parallel_for i in Dense[4] {
    task @kernel(i) resources(data)
  }
}"""
        # First parse
        module1 = cpp.parse_string(source)

        # Print via WorkloadDef.print()
        printed1 = module1.workloads[0].print()

        # Parse again
        module2 = cpp.parse_string(printed1)

        # Verify structure preserved
        assert len(module2.workloads) == 1
        assert module2.workloads[0].name == module1.workloads[0].name

    def test_find_workload(self):
        """Test Module.find_workload method."""
        source = """@workload first(%N: Dense[4]) {
  task @k1() resources(a)
}

@workload second(%M: Dense[8]) {
  task @k2() resources(b)
}"""
        module = cpp.parse_string(source)

        first = module.find_workload("first")
        second = module.find_workload("second")
        missing = module.find_workload("nonexistent")

        assert first is not None
        assert first.name == "first"
        assert second is not None
        assert second.name == "second"
        assert missing is None


@pytest.mark.skipif(not HAS_CPP_MODULE, reason="C++ module not built")
class TestTypeChecking:
    """Test type checking across Python/C++ boundary."""

    def test_type_check_valid_module(self):
        """Test type checking a valid module."""
        factory = cpp.IRFactory()

        module = cpp.Module()
        module.name = "valid_test"

        axis = factory.create_dense_axis(4)
        task = factory.create_task("kernel", ["i"], ["data"])
        pfor = factory.create_parallel_for(axis, "i", task)

        workload = cpp.WorkloadDef()
        workload.name = "test_workload"
        workload.level = cpp.WorkloadLevel.CPU
        workload.body = pfor
        module.workloads.append(workload)

        result = cpp.type_check(module)

        assert result.valid
        assert len(result.errors) == 0

    def test_type_check_invalid_module(self):
        """Test type checking detects errors."""
        factory = cpp.IRFactory()

        module = cpp.Module()
        module.name = "invalid_test"

        # Create schedule referencing nonexistent workload
        sched = cpp.ScheduleDef()
        sched.name = "bad_schedule"
        sched.workload_name = "nonexistent"
        sched.level = cpp.WorkloadLevel.CPU
        module.schedules.append(sched)

        result = cpp.type_check(module)

        assert not result.valid
        assert len(result.errors) > 0


@pytest.mark.skipif(not HAS_CPP_MODULE, reason="C++ module not built")
class TestEnums:
    """Test enum bindings."""

    def test_node_kind_enum(self):
        """Test NodeKind enum values."""
        assert cpp.NodeKind.DenseAxis is not None
        assert cpp.NodeKind.ParallelFor is not None
        assert cpp.NodeKind.Task is not None

    def test_dtype_enum(self):
        """Test DType enum values."""
        assert cpp.DType.F16 is not None
        assert cpp.DType.F32 is not None
        assert cpp.DType.I32 is not None

    def test_dispatch_policy_enum(self):
        """Test DispatchPolicy enum values."""
        assert cpp.DispatchPolicy.RoundRobin is not None
        assert cpp.DispatchPolicy.Affinity is not None
        assert cpp.DispatchPolicy.WorkSteal is not None

    def test_timing_policy_enum(self):
        """Test TimingPolicy enum values."""
        assert cpp.TimingPolicy.Immediate is not None
        assert cpp.TimingPolicy.Batched is not None


@pytest.mark.skipif(not HAS_CPP_MODULE, reason="C++ module not built")
class TestGraphBindings:
    """Test task graph bindings."""

    def test_task_graph_storage(self):
        """Test TaskGraphStorage creation."""
        storage = cpp.TaskGraphStorage()
        storage.reserve(100, 200)

        assert storage.num_tasks() == 0
        assert storage.num_edges() == 0

    def test_task_graph_builder(self):
        """Test TaskGraphBuilder usage."""
        storage = cpp.TaskGraphStorage()
        builder = cpp.TaskGraphBuilder(storage)

        # Create a task
        tid = builder.begin_task(0).add_arg(42).submit()

        builder.finalize()

        assert storage.num_tasks() == 1
        assert storage.is_finalized()

    def test_kernel_bundle(self):
        """Test KernelBundle for kernel registration."""
        bundle = cpp.KernelBundle()

        info = cpp.KernelInfo()
        info.name = "test_kernel"
        info.symbol = "test_kernel_impl"
        info.num_params = 2
        info.num_io = 3

        kid = bundle.register_kernel(info)

        assert bundle.size() == 1
        assert bundle.find_kernel("test_kernel") == kid


@pytest.mark.skipif(not HAS_CPP_MODULE, reason="C++ module not built")
class TestBackendBindings:
    """Test backend compilation bindings."""

    def test_compile_options(self):
        """Test CompileOptions struct."""
        opts = cpp.CompileOptions()
        opts.target = "cpu_sim"
        opts.num_threads = 4
        opts.num_streams = 2

        assert opts.target == "cpu_sim"
        assert opts.num_threads == 4
        assert opts.num_streams == 2

    def test_available_backends(self):
        """Test backend registry."""
        backends = cpp.available_backends()

        # Should have at least cpu_sim and ascend_npu
        assert len(backends) >= 2
        assert "cpu_sim" in backends or "ascend_npu" in backends

    def test_get_backend(self):
        """Test getting backend by name."""
        # Try to get a registered backend
        backends = cpp.available_backends()
        if backends:
            backend = cpp.get_backend(backends[0])
            assert backend is not None
            assert backend.name() == backends[0]


@pytest.mark.skipif(not HAS_CPP_MODULE, reason="C++ module not built")
class TestNodePrinting:
    """Test node print methods."""

    def test_axis_node_print(self):
        """Test AxisNode print method."""
        factory = cpp.IRFactory()
        axis = factory.create_dense_axis(8)

        output = axis.print()
        assert "Dense[8]" in output

    def test_task_node_print(self):
        """Test TaskNode print method."""
        factory = cpp.IRFactory()
        task = factory.create_task("kernel", ["i"], ["data"])

        output = task.print()
        assert "@kernel" in output

    def test_workload_def_print(self):
        """Test WorkloadDef print method."""
        factory = cpp.IRFactory()

        axis = factory.create_dense_axis(4)
        task = factory.create_task("kernel", ["i"], ["data"])
        pfor = factory.create_parallel_for(axis, "i", task)

        workload = cpp.WorkloadDef()
        workload.name = "test"
        workload.level = cpp.WorkloadLevel.CPU
        workload.body = pfor

        output = workload.print()
        assert "@workload test" in output
        assert "parallel_for" in output


# Utility functions
@pytest.mark.skipif(not HAS_CPP_MODULE, reason="C++ module not built")
class TestUtilityFunctions:
    """Test utility functions."""

    def test_node_kind_to_string(self):
        """Test nodeKindToString conversion."""
        assert cpp.node_kind_to_string(cpp.NodeKind.Task) == "Task"
        assert cpp.node_kind_to_string(cpp.NodeKind.ParallelFor) == "ParallelFor"

    def test_dtype_to_string(self):
        """Test dtypeToString conversion."""
        assert cpp.dtype_to_string(cpp.DType.F16) == "f16"
        assert cpp.dtype_to_string(cpp.DType.F32) == "f32"

    def test_level_to_string(self):
        """Test levelToString conversion."""
        assert cpp.level_to_string(cpp.WorkloadLevel.CPU) == "cpu"
        assert cpp.level_to_string(cpp.WorkloadLevel.NPU) == "npu"


if __name__ == "__main__":
    if not HAS_CPP_MODULE:
        print(f"Skipping tests: C++ module not available ({IMPORT_ERROR})")
        print("Build with: cmake -B build && cmake --build build")
        sys.exit(0)

    pytest.main([__file__, "-v"])
