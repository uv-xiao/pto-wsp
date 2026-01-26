"""
Python → C++ IR Bridge for PTO Workload-Schedule Programming (PTO-WSP) framework.

This module converts Python workload definitions to C++ IR for
multi-backend compilation and execution.

Usage:
    from pto_wsp.ir_bridge import workload_to_ir, schedule_to_ir

    # Convert Python workload to C++ Module
    module = workload_to_ir(my_workload)

    # Compile and execute via C++ backend
    from pto_ir_cpp import compile
    program = compile(module)
    program.execute()
"""

from __future__ import annotations
from typing import Any, Optional, TYPE_CHECKING

# Try to import C++ bindings - may fail if not built
try:
    import pto_ir_cpp as cpp
    HAS_CPP_BINDINGS = True
except ImportError:
    HAS_CPP_BINDINGS = False
    cpp = None

from pto_wsp.workload import Workload
from pto_wsp.types import Dense, DenseDyn, Ragged, Sparse


class IRBridgeError(Exception):
    """Error during Python → C++ IR conversion."""
    pass


def check_cpp_bindings():
    """Check if C++ bindings are available."""
    if not HAS_CPP_BINDINGS:
        raise IRBridgeError(
            "C++ bindings not available. Build the project with CMake first:\n"
            "  cmake -B build && cmake --build build"
        )


def workload_to_ir(workload: Workload, module_name: str = "main") -> "cpp.Module":
    """Convert a Python Workload to C++ IR Module.

    Args:
        workload: Python Workload object
        module_name: Name for the generated module

    Returns:
        cpp.Module: C++ IR Module ready for compilation

    Raises:
        IRBridgeError: If conversion fails or C++ bindings unavailable

    Example:
        from pto_wsp import workload, P, kernel, DenseDyn
        from pto_wsp.ir_bridge import workload_to_ir
        import pto_ir_cpp as cpp

        @workload
        def my_workload():
            for b in P(batch):
                my_kernel[b](...)

        module = workload_to_ir(my_workload())
        program = cpp.compile(module)
        program.execute()
    """
    check_cpp_bindings()

    # Create C++ IR factory and module
    factory = cpp.IRFactory()
    module = cpp.Module()
    module.name = module_name
    module.version = "1.0"
    module.targets = ["cpu", "ascend"]

    # Convert workload to WorkloadDef
    workload_def = _convert_workload_def(factory, workload)
    module.workloads.append(workload_def)

    # Convert schedule if present
    if workload._schedule:
        schedule_def = _convert_schedule_def(factory, workload)
        module.schedules.append(schedule_def)

    return module


def _convert_workload_def(factory: "cpp.IRFactory", workload: Workload) -> "cpp.WorkloadDef":
    """Convert Python Workload to C++ WorkloadDef."""
    wdef = cpp.WorkloadDef()
    wdef.name = workload._name if workload._name else "unnamed"
    wdef.level = cpp.WorkloadLevel.CPU

    # Convert axis parameters
    for name, axis in workload._params if hasattr(workload, '_params') else []:
        axis_node = _convert_axis(factory, axis)
        wdef.params.append((name, axis_node))

    # Convert workload body
    wdef.body = _convert_workload_body(factory, workload)

    return wdef


def _convert_axis(factory: "cpp.IRFactory", axis: Any) -> "cpp.AxisNode":
    """Convert Python axis to C++ AxisNode."""
    if isinstance(axis, Dense):
        return factory.create_dense_axis(axis.size)
    elif isinstance(axis, DenseDyn):
        # For DenseDyn, use the actual size as a string variable
        return factory.create_dense_dyn_axis(f"dyn_{id(axis)}")
    elif isinstance(axis, Ragged):
        return factory.create_ragged_axis(
            f"outer_{id(axis)}",
            f"lengths_{id(axis)}"
        )
    elif isinstance(axis, Sparse):
        return factory.create_sparse_axis(
            f"outer_{id(axis)}",
            f"indptr_{id(axis)}",
            f"indices_{id(axis)}"
        )
    elif isinstance(axis, int):
        return factory.create_dense_axis(axis)
    else:
        # Try to get size attribute
        size = getattr(axis, 'size', 1)
        return factory.create_dense_axis(size)


def _convert_workload_body(factory: "cpp.IRFactory", workload: Workload) -> "cpp.WorkloadNode":
    """Convert Python Workload body to C++ WorkloadNode."""
    kind = workload._kind

    if kind == "task":
        # Leaf task node
        kernel_name = workload._kwargs.get("kernel", "unknown")
        params = [str(p) for p in workload._kwargs.get("params", [])]
        resources = [str(r) for r in workload._kwargs.get("resources", [])]
        return factory.create_task(kernel_name, params, resources)

    elif kind == "parallel_for":
        axis = workload._kwargs.get("axis")
        var_name = workload._kwargs.get("var_name", "i")
        body = workload._kwargs.get("body")

        axis_node = _convert_axis(factory, axis)

        # Handle body - may be lambda or Workload
        if callable(body):
            # Call lambda with 0 to get a sample body
            body_workload = body(0)
            if isinstance(body_workload, Workload):
                body_node = _convert_workload_body(factory, body_workload)
            else:
                # Create empty task for unknown body
                body_node = factory.create_task("unknown", [], [])
        elif isinstance(body, Workload):
            body_node = _convert_workload_body(factory, body)
        else:
            body_node = factory.create_task("unknown", [], [])

        return factory.create_parallel_for(axis_node, var_name, body_node)

    elif kind == "for_each" or kind == "sequential":
        # Sequential loop or list
        workloads = workload._kwargs.get("workloads")
        if workloads:
            # Sequential list of workloads
            nodes = [_convert_workload_body(factory, w) for w in workloads
                     if isinstance(w, Workload)]
            return factory.create_sequential(nodes)
        else:
            # Sequential loop
            axis = workload._kwargs.get("axis")
            var_name = workload._kwargs.get("var_name", "i")
            body = workload._kwargs.get("body")

            axis_node = _convert_axis(factory, axis)

            if callable(body):
                body_workload = body(0)
                if isinstance(body_workload, Workload):
                    body_node = _convert_workload_body(factory, body_workload)
                else:
                    body_node = factory.create_task("unknown", [], [])
            elif isinstance(body, Workload):
                body_node = _convert_workload_body(factory, body)
            else:
                body_node = factory.create_task("unknown", [], [])

            return factory.create_for_each(axis_node, var_name, body_node)

    elif kind == "combine":
        workloads = workload._kwargs.get("workloads", [])
        nodes = [_convert_workload_body(factory, w) for w in workloads
                 if isinstance(w, Workload)]
        return factory.create_combine(nodes)

    elif kind == "select":
        sparse = workload._kwargs.get("sparse")
        body = workload._kwargs.get("body")

        # Create sparse axis node
        sparse_node = factory.create_sparse_axis(
            f"outer_{id(sparse)}",
            f"indptr_{id(sparse)}",
            f"indices_{id(sparse)}"
        )

        if callable(body):
            body_workload = body(0)
            if isinstance(body_workload, Workload):
                body_node = _convert_workload_body(factory, body_workload)
            else:
                body_node = factory.create_task("unknown", [], [])
        elif isinstance(body, Workload):
            body_node = _convert_workload_body(factory, body)
        else:
            body_node = factory.create_task("unknown", [], [])

        return factory.create_select(sparse_node, "_select_idx", body_node)

    elif kind == "cond":
        predicate = workload._kwargs.get("predicate")
        then_w = workload._kwargs.get("then_workload")
        else_w = workload._kwargs.get("else_workload")

        predicate_str = str(predicate) if not callable(predicate) else "runtime"

        then_node = _convert_workload_body(factory, then_w) if isinstance(then_w, Workload) else factory.create_task("then", [], [])
        else_node = _convert_workload_body(factory, else_w) if isinstance(else_w, Workload) else factory.create_task("else", [], [])

        return factory.create_cond(predicate_str, then_node, else_node)

    else:
        # Unknown kind - create empty task
        return factory.create_task(f"unknown_{kind}", [], [])


def _convert_schedule_def(factory: "cpp.IRFactory", workload: Workload) -> "cpp.ScheduleDef":
    """Convert Python schedule configuration to C++ ScheduleDef."""
    sdef = cpp.ScheduleDef()
    sdef.name = f"{workload._name}_schedule" if workload._name else "default_schedule"
    sdef.workload_name = workload._name if workload._name else "unnamed"
    sdef.level = cpp.WorkloadLevel.CPU

    schedule = workload._schedule

    # Convert dispatch policy
    if "dispatch" in schedule:
        dispatch_info = schedule["dispatch"]
        policy = _convert_dispatch_policy(dispatch_info)
        num_targets = getattr(dispatch_info, 'num_targets', 4)
        key_expr = getattr(dispatch_info, 'key_expr', "")
        sdef.directives.append(factory.create_dispatch(policy, num_targets, key_expr))

    # Convert streams
    if "streams" in schedule:
        num_streams = schedule["streams"]
        key_expr = schedule.get("stream_by", "")
        if callable(key_expr):
            key_expr = ""  # Can't serialize lambda
        sdef.directives.append(factory.create_stream(num_streams, key_expr))

    # Convert timing
    if "timing" in schedule:
        timing_info = schedule["timing"]
        policy = _convert_timing_policy(timing_info)
        param = getattr(timing_info, 'param', 0)
        sdef.directives.append(factory.create_timing(policy, param))

    return sdef


def _convert_dispatch_policy(policy: Any) -> "cpp.DispatchPolicy":
    """Convert Python DispatchPolicy to C++ enum."""
    # Get policy name
    policy_name = getattr(policy, 'name', str(type(policy).__name__)).lower()

    if 'round' in policy_name:
        return cpp.DispatchPolicy.RoundRobin
    elif 'affinity' in policy_name:
        return cpp.DispatchPolicy.Affinity
    elif 'hash' in policy_name:
        return cpp.DispatchPolicy.Hash
    elif 'steal' in policy_name:
        return cpp.DispatchPolicy.WorkSteal
    else:
        return cpp.DispatchPolicy.RoundRobin  # Default


def _convert_timing_policy(policy: Any) -> "cpp.TimingPolicy":
    """Convert Python TimingPolicy to C++ enum."""
    policy_name = getattr(policy, 'name', str(type(policy).__name__)).lower()

    if 'immediate' in policy_name:
        return cpp.TimingPolicy.Immediate
    elif 'batch' in policy_name:
        return cpp.TimingPolicy.Batched
    elif 'interleav' in policy_name:
        return cpp.TimingPolicy.Interleaved
    elif 'rate' in policy_name:
        return cpp.TimingPolicy.RateLimited
    else:
        return cpp.TimingPolicy.Immediate  # Default


def module_to_string(module: "cpp.Module") -> str:
    """Convert C++ Module to string representation.

    Args:
        module: C++ IR Module

    Returns:
        String representation in PTO IR format
    """
    check_cpp_bindings()
    printer = cpp.Printer()
    return printer.print_module(module)


def parse_ir_string(source: str) -> "cpp.Module":
    """Parse PTO IR from string.

    Args:
        source: PTO IR source code

    Returns:
        cpp.Module: Parsed C++ IR Module
    """
    check_cpp_bindings()
    return cpp.parse_string(source)
