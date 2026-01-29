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
from dataclasses import dataclass
import inspect
import struct
from typing import Any, Optional, TYPE_CHECKING

# Try to import C++ bindings - may fail if not built
# Try submodule import first (pip install -e .), then fallback to top-level
try:
    from pto_wsp import pto_ir_cpp as cpp
    HAS_CPP_BINDINGS = True
except ImportError:
    try:
        import pto_ir_cpp as cpp
        HAS_CPP_BINDINGS = True
    except ImportError:
        HAS_CPP_BINDINGS = False
        cpp = None

from pto_wsp.workload import Workload
from pto_wsp.types import Dense, DenseDyn, Ragged, Sparse, Symbol, Tensor, DType


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
    # Note: pybind11 doesn't support list.append() on def_readwrite vectors
    # Use direct assignment instead
    module.workloads = [workload_def]

    # Convert schedule if present (check for non-default schedule)
    if workload._schedule and workload._schedule.to_dict():
        schedule_def = _convert_schedule_def(factory, workload)
        module.schedules = [schedule_def]

    return module


@dataclass(frozen=True)
class CodegenIRBuild:
    module: "cpp.Module"
    tensors: list[Tensor]


def workload_to_codegen_ir(workload: Workload, module_name: str = "main") -> CodegenIRBuild:
    """Convert a Python Workload to a C++ IR Module with attached codegen inputs.

    v9 formal input: C++ owns the compilation/codegen pipeline; Python only
    provides a typed IR module and runtime metadata (tensors/kernels/bindings).
    """
    check_cpp_bindings()

    factory = cpp.IRFactory()
    module = cpp.Module()
    module.name = module_name
    module.version = "1.0"
    module.targets = ["cpu_sim", "ascend_npu"]

    collector = _CodegenCollector()
    collector.collect(workload)

    module.tensors = [_tensor_to_codegen_info(t) for t in collector.tensors]
    module.kernels = [_kernel_to_codegen_def(k) for _, k in sorted(collector.kernels.items())]

    workload_def = _convert_workload_def_codegen(factory, workload, collector.tensor_ids)
    module.workloads = [workload_def]

    if workload._schedule and workload._schedule.to_dict():
        schedule_def = _convert_schedule_def(factory, workload)
        module.schedules = [schedule_def]

    return CodegenIRBuild(module=module, tensors=list(collector.tensors))


def _convert_workload_def(factory: "cpp.IRFactory", workload: Workload) -> "cpp.WorkloadDef":
    """Convert Python Workload to C++ WorkloadDef.

    Args:
        factory: C++ IRFactory for creating IR nodes
        workload: Python Workload object with _name, _params, and body

    Returns:
        cpp.WorkloadDef: C++ workload definition ready for compilation
    """
    wdef = cpp.WorkloadDef()
    wdef.name = workload._name if workload._name else "unnamed"
    wdef.level = cpp.WorkloadLevel.CPU

    # Convert axis parameters (list of (name, axis) tuples)
    params = []
    for name, axis in workload._params:
        if isinstance(axis, DenseDyn):
            axis_node = factory.create_dense_dyn_axis(str(name))
        else:
            axis_node = _convert_axis(factory, axis)
        params.append((name, axis_node))
    wdef.params = params

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
        kernel = workload._kwargs.get("kernel", "unknown")
        kernel_name = kernel.name if hasattr(kernel, "name") else str(kernel)
        params = [str(p) for p in workload._kwargs.get("params", [])]
        resources = [str(r) for r in workload._kwargs.get("resources", [])]
        return factory.create_task(kernel_name, params, resources)

    elif kind == "parallel_for":
        axis = workload._kwargs.get("axis")
        var_name = workload._kwargs.get("var_name", "i")
        body = workload._kwargs.get("body")

        if isinstance(axis, DenseDyn):
            # DenseDyn size is provided at runtime via the loop variable name.
            axis_node = factory.create_dense_dyn_axis(str(var_name))
        else:
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

            if isinstance(axis, DenseDyn):
                axis_node = factory.create_dense_dyn_axis(str(var_name))
            else:
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

        from pto_wsp.scalar_expr import as_expr

        expr = as_expr(predicate)
        predicate_str = expr.to_debug_str()
        predicate_cpp = expr.to_cpp()

        then_node = _convert_workload_body(factory, then_w) if isinstance(then_w, Workload) else factory.create_task("then", [], [])
        else_node = _convert_workload_body(factory, else_w) if isinstance(else_w, Workload) else factory.create_task("else", [], [])

        return factory.create_cond(predicate_str, predicate_cpp, then_node, else_node)

    else:
        # Unknown kind - create empty task
        return factory.create_task(f"unknown_{kind}", [], [])


def _convert_workload_def_codegen(
    factory: "cpp.IRFactory",
    workload: Workload,
    tensor_ids: dict[int, int],
) -> "cpp.WorkloadDef":
    wdef = cpp.WorkloadDef()
    wdef.name = workload._name if workload._name else "unnamed"
    wdef.level = cpp.WorkloadLevel.CPU

    params = []
    for name, axis in workload._params:
        if isinstance(axis, DenseDyn):
            axis_node = factory.create_dense_dyn_axis(str(name))
        else:
            axis_node = _convert_axis(factory, axis)
        params.append((name, axis_node))
    wdef.params = params

    wdef.body = _convert_workload_body_codegen(factory, workload, tensor_ids)
    return wdef


def _convert_workload_body_codegen(
    factory: "cpp.IRFactory",
    workload: Workload,
    tensor_ids: dict[int, int],
) -> "cpp.WorkloadNode":
    kind = workload._kind

    if kind == "slot_set_u64":
        slot = int(workload._kwargs.get("slot", 0))
        value = int(workload._kwargs.get("value", 0)) & 0xFFFFFFFFFFFFFFFF
        return factory.create_slot_set_u64(slot, value)

    if kind == "slot_load_u64":
        slot = int(workload._kwargs.get("slot", 0))
        t = workload._kwargs.get("tensor")
        if not isinstance(t, Tensor):
            raise IRBridgeError("slot_load_u64 requires tensor=Tensor view")

        if len(t.shape) < 2:
            raise IRBridgeError(f"slot_load_u64 tensor view must be rank>=2 (got shape={t.shape!r})")

        base = t.base
        tid = tensor_ids[id(base)]

        ta = cpp.CodegenTensorArg()
        ta.param = "tensor"
        ta.tensor_id = int(tid)
        ta.base_rank = int(len(base.shape))
        ta.view_rank = int(len(t.shape))
        ta.index_exprs = [_index_expr_to_codegen_expr(e) for e in t.index_exprs]

        row = int(workload._kwargs.get("row", 0))
        col = int(workload._kwargs.get("col", 0))
        return factory.create_slot_load_u64(slot, ta, row, col)

    if kind == "task":
        kernel = workload._kwargs.get("kernel", "unknown")
        if kernel is None or not hasattr(kernel, "name"):
            raise IRBridgeError("Task node missing kernel for codegen")
        kernel_name = str(kernel.name)

        params = [str(p) for p in workload._kwargs.get("params", [])]
        resources_map = workload._kwargs.get("resources", {})
        if not isinstance(resources_map, dict):
            raise IRBridgeError("Task resources must be dict[param_name -> Tensor] for codegen")
        resources = [str(k) for k in resources_map.keys()]

        axis_args = [_axis_val_to_codegen_arg(v) for v in workload._kwargs.get("params", []) or []]

        ir = kernel.trace()

        # Scalar args (kernel scalar params) are passed via the u64 args ABI,
        # after axis args.
        scalar_args = []
        sig = inspect.signature(kernel.func)
        for pname, pval in ir.params:
            if pval.shape is not None:
                continue
            if pname in resources_map:
                sval = resources_map[pname]
            else:
                default = sig.parameters.get(pname).default if pname in sig.parameters else inspect._empty
                if default is inspect._empty:
                    raise IRBridgeError(f"Missing scalar arg for param '{pname}' in task {kernel_name}")
                sval = default
            scalar_args.append(_scalar_val_to_codegen_arg(sval, pval.dtype))

        tensor_args = []
        for pname, pval in ir.params:
            if pval.shape is None:
                continue
            t = resources_map.get(pname)
            if not isinstance(t, Tensor):
                raise IRBridgeError(f"Missing tensor arg for param '{pname}' in task {kernel_name}")

            base = t.base
            tid = tensor_ids[id(base)]

            ta = cpp.CodegenTensorArg()
            ta.param = str(pname)
            ta.tensor_id = int(tid)
            ta.base_rank = int(len(base.shape))
            ta.view_rank = int(len(t.shape))
            ta.index_exprs = [_index_expr_to_codegen_expr(e) for e in t.index_exprs]
            tensor_args.append(ta)

        return factory.create_task_codegen(kernel_name, params, resources, axis_args, scalar_args, tensor_args)

    if kind == "parallel_for":
        axis = workload._kwargs.get("axis")
        var_name = workload._kwargs.get("var_name", "i")
        body = workload._kwargs.get("body")

        if isinstance(axis, DenseDyn):
            axis_node = factory.create_dense_dyn_axis(str(var_name))
        else:
            axis_node = _convert_axis(factory, axis)

        if callable(body):
            # v9: for non-@workload constructed workloads (e.g. primitives),
            # trace loop bodies with a symbolic LoopVar so tasks can bind to
            # the loop index without sampling with a concrete integer.
            from pto_wsp.p_namespace import LoopVar

            body = body(LoopVar(str(var_name), axis))
        if not isinstance(body, Workload):
            raise IRBridgeError("Loop body must be a Workload for codegen")

        return factory.create_parallel_for(
            axis_node,
            var_name,
            _convert_workload_body_codegen(factory, body, tensor_ids),
        )

    if kind == "for_each":
        axis = workload._kwargs.get("axis")
        var_name = workload._kwargs.get("var_name", "i")
        body = workload._kwargs.get("body")

        if isinstance(axis, DenseDyn):
            axis_node = factory.create_dense_dyn_axis(str(var_name))
        else:
            axis_node = _convert_axis(factory, axis)

        if callable(body):
            from pto_wsp.p_namespace import LoopVar

            body = body(LoopVar(str(var_name), axis))
        if not isinstance(body, Workload):
            raise IRBridgeError("Loop body must be a Workload for codegen")

        return factory.create_for_each(
            axis_node,
            var_name,
            _convert_workload_body_codegen(factory, body, tensor_ids),
        )

    if kind in ("combine", "sequential"):
        nodes = []
        for child in workload._kwargs.get("workloads", []) or []:
            if isinstance(child, Workload):
                nodes.append(_convert_workload_body_codegen(factory, child, tensor_ids))
        if kind == "combine":
            return factory.create_combine(nodes)
        return factory.create_sequential(nodes)

    if kind == "cond":
        predicate = workload._kwargs.get("predicate")
        then_w = workload._kwargs.get("then_workload")
        else_w = workload._kwargs.get("else_workload")

        from pto_wsp.scalar_expr import as_expr

        expr = as_expr(predicate)
        predicate_str = expr.to_debug_str()
        predicate_cpp = expr.to_cpp()
        then_node = _convert_workload_body_codegen(factory, then_w, tensor_ids) if isinstance(then_w, Workload) else factory.create_task("then", [], [])
        else_node = _convert_workload_body_codegen(factory, else_w, tensor_ids) if isinstance(else_w, Workload) else factory.create_task("else", [], [])
        return factory.create_cond(predicate_str, predicate_cpp, then_node, else_node)

    if kind == "select":
        sparse = workload._kwargs.get("sparse")
        var_name = workload._kwargs.get("var_name", "e")
        body = workload._kwargs.get("body")
        sparse_node = factory.create_sparse_axis(
            f"outer_{id(sparse)}",
            f"indptr_{id(sparse)}",
            f"indices_{id(sparse)}",
        )
        if callable(body):
            from pto_wsp.p_namespace import LoopVar

            body = body(LoopVar(str(var_name), sparse))
        if not isinstance(body, Workload):
            raise IRBridgeError("Select body must be a Workload for codegen")
        return factory.create_select(
            sparse_node,
            str(var_name),
            _convert_workload_body_codegen(factory, body, tensor_ids),
        )

    # ------------------------------------------------------------------
    # CSP (codegen-first): connect / send / consume
    # ------------------------------------------------------------------
    if kind == "send":
        ch = workload._kwargs.get("channel")
        value = workload._kwargs.get("value")
        if ch is None or not hasattr(ch, "name"):
            raise IRBridgeError("send(...) missing channel")
        if not isinstance(value, Workload):
            raise IRBridgeError("send(...) value must be a Workload for codegen-first CSP")
        value_node = _convert_workload_body_codegen(factory, value, tensor_ids)
        return factory.create_send(str(ch.name), value_node)

    if kind == "consume":
        ch = workload._kwargs.get("channel")
        body_fn = workload._kwargs.get("body")
        if ch is None or not hasattr(ch, "name"):
            raise IRBridgeError("consume(...) missing channel")
        if not callable(body_fn):
            raise IRBridgeError("consume(...) missing body callable")

        from pto_wsp.p_namespace import LoopVar

        def _sanitize_ident(s: str) -> str:
            out = []
            for c in s:
                out.append(c if (c.isalnum() or c == "_") else "_")
            ident = "".join(out)
            if not ident or ident[0].isdigit():
                ident = "v_" + ident
            return ident

        value_var = f"csp_{_sanitize_ident(str(ch.name))}_{abs(id(workload))}"
        body_workload = body_fn(LoopVar(value_var, axis=None))
        if not isinstance(body_workload, Workload):
            raise IRBridgeError("consume(...) body must return a Workload")
        body_node = _convert_workload_body_codegen(factory, body_workload, tensor_ids)
        return factory.create_consume(str(ch.name), value_var, body_node)

    if kind == "connect":
        processes = workload._kwargs.get("processes") or []
        channels = workload._kwargs.get("channels") or []

        # Create channel nodes first (by name; type is placeholder for v9 bring-up).
        ch_nodes = []
        for ch in channels:
            if ch is None or not hasattr(ch, "name"):
                raise IRBridgeError("connect(...) channels must be Channel objects with .name")
            depth = int(getattr(ch, "depth", 0))
            ch_nodes.append(factory.create_channel(str(ch.name), depth))

        proc_nodes = []
        for p in processes:
            if p is None or not hasattr(p, "name"):
                raise IRBridgeError("connect(...) processes must be Process objects with .name")
            consumes = [str(c.name) for c in getattr(p, "consumes", []) or []]
            produces = [str(c.name) for c in getattr(p, "produces", []) or []]
            body = getattr(p, "body", None)
            if not isinstance(body, Workload):
                raise IRBridgeError("Process.body must be a Workload for codegen-first CSP")
            body_node = _convert_workload_body_codegen(factory, body, tensor_ids)
            proc_nodes.append(factory.create_process(str(p.name), consumes, produces, body_node))

        return factory.create_pipeline(ch_nodes, proc_nodes)

    return factory.create_task(f"unknown_{kind}", [], [])


def _convert_schedule_def(factory: "cpp.IRFactory", workload: Workload) -> "cpp.ScheduleDef":
    """Convert Python Schedule dataclass to C++ ScheduleDef.

    Args:
        factory: C++ IRFactory for creating IR nodes
        workload: Python Workload with _schedule: Schedule dataclass

    Returns:
        cpp.ScheduleDef: C++ schedule definition with directives
    """
    sdef = cpp.ScheduleDef()
    sdef.name = f"{workload._name}_schedule" if workload._name else "default_schedule"
    sdef.workload_name = workload._name if workload._name else "unnamed"
    sdef.level = cpp.WorkloadLevel.CPU

    schedule = workload._schedule
    directives = []  # Build list, then assign (pybind11 limitation)

    # Convert dispatch policy
    if schedule.dispatch is not None:
        policy = _convert_dispatch_policy(schedule.dispatch)
        kind = getattr(schedule.dispatch, "_kind", "")
        kwargs = getattr(schedule.dispatch, "_kwargs", {}) or {}

        num_targets = 0
        key_expr = ""
        key = None

        if kind == "round_robin":
            num_targets = int(kwargs.get("num_aicpus", 0))
        elif kind in ("affinity", "hash"):
            num_targets = int(kwargs.get("num_aicpus", 0))
            key_fn = kwargs.get("key_fn")
            if key_fn is None:
                raise IRBridgeError(f"DispatchPolicy.{kind} missing key_fn")
            if callable(key_fn):
                from pto_wsp.scalar_expr import trace_task_fn

                expr = trace_task_fn(key_fn)
            else:
                from pto_wsp.scalar_expr import as_expr

                expr = as_expr(key_fn)
            key_expr = expr.to_debug_str()
            key = expr.to_cpp()
        elif kind == "dispatch_by":
            num_targets = int(kwargs.get("num_aicpus", 0))
            fn = kwargs.get("fn")
            if fn is None:
                raise IRBridgeError("DispatchPolicy.dispatch_by missing fn")
            if callable(fn):
                from pto_wsp.scalar_expr import trace_task_fn

                expr = trace_task_fn(fn)
            else:
                from pto_wsp.scalar_expr import as_expr

                expr = as_expr(fn)
            key_expr = expr.to_debug_str()
            key = expr.to_cpp()
        elif kind == "work_steal":
            pass
        else:
            # If this is not one of our known python policy wrappers, keep it as a string-only debug hint.
            key_expr = str(schedule.dispatch)

        directives.append(factory.create_dispatch(policy, num_targets, key_expr, key))

    # Convert task_window (R9 task_graph config) into a schedule directive.
    tg = getattr(workload, "_task_graph_config", None)
    window = getattr(tg, "window", None) if tg is not None else None
    if window is not None:
        overflow = cpp.TaskWindowOverflowPolicy.Stall
        mode = getattr(window, "mode", None)
        mode_val = getattr(mode, "value", str(mode)).lower() if mode is not None else "stall"
        if "abort" in mode_val:
            overflow = cpp.TaskWindowOverflowPolicy.Abort
        elif "bench" in mode_val:
            overflow = cpp.TaskWindowOverflowPolicy.Benchmark
        directives.append(factory.create_task_window(int(window.size), str(window.unit), overflow))

    # Convert streams
    if schedule.streams != 1:
        num_streams = schedule.streams
        key_expr = ""
        key = None
        if schedule.stream_by is not None:
            if callable(schedule.stream_by):
                from pto_wsp.scalar_expr import trace_task_fn

                expr = trace_task_fn(schedule.stream_by)
                key_expr = expr.to_debug_str()
                key = expr.to_cpp()
            else:
                key_expr = str(schedule.stream_by)
        directives.append(factory.create_stream(num_streams, key_expr, key))

    # Convert timing
    if schedule.timing is not None:
        policy = _convert_timing_policy(schedule.timing)
        param = getattr(schedule.timing, 'param', 0)
        directives.append(factory.create_timing(policy, param))

    # Assign directives (pybind11 doesn't support append on def_readwrite vectors)
    sdef.directives = directives

    return sdef


def _convert_dispatch_policy(policy: Any) -> "cpp.DispatchPolicy":
    """Convert Python DispatchPolicy to C++ enum."""
    kind = getattr(policy, "_kind", "").lower()
    if kind == "round_robin":
        return cpp.DispatchPolicy.RoundRobin
    if kind == "affinity":
        return cpp.DispatchPolicy.Affinity
    if kind == "hash":
        return cpp.DispatchPolicy.Hash
    if kind == "work_steal":
        return cpp.DispatchPolicy.WorkSteal
    if kind == "dispatch_by":
        return cpp.DispatchPolicy.Custom

    # Fallback: legacy heuristic based on class name.
    policy_name = getattr(policy, "name", str(type(policy).__name__)).lower()
    if "affinity" in policy_name:
        return cpp.DispatchPolicy.Affinity
    if "hash" in policy_name:
        return cpp.DispatchPolicy.Hash
    if "steal" in policy_name:
        return cpp.DispatchPolicy.WorkSteal
    return cpp.DispatchPolicy.RoundRobin


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


class _CodegenCollector:
    def __init__(self) -> None:
        self.tensor_ids: dict[int, int] = {}
        self.tensors: list[Tensor] = []
        self.kernels: dict[str, Any] = {}

    def collect(self, node: Workload) -> None:
        if not isinstance(node, Workload):
            return

        if node._kind == "task":
            kernel = node._kwargs.get("kernel")
            if kernel is not None and hasattr(kernel, "name"):
                self.kernels[str(kernel.name)] = kernel

            resources = node._kwargs.get("resources", {})
            if isinstance(resources, dict):
                for v in resources.values():
                    self._register_tensor(v)
            else:
                for v in resources:
                    self._register_tensor(v)

        kind = node._kind
        if kind in ("parallel_for", "for_each"):
            body = node._kwargs.get("body")
            if isinstance(body, Workload):
                self.collect(body)
            elif callable(body):
                maybe = body(0)
                if isinstance(maybe, Workload):
                    self.collect(maybe)
        elif kind in ("combine", "sequential"):
            for child in node._kwargs.get("workloads", []) or []:
                if isinstance(child, Workload):
                    self.collect(child)
        elif kind == "cond":
            then_w = node._kwargs.get("then_workload")
            else_w = node._kwargs.get("else_workload")
            if isinstance(then_w, Workload):
                self.collect(then_w)
            if isinstance(else_w, Workload):
                self.collect(else_w)
        elif kind == "select":
            body = node._kwargs.get("body")
            if isinstance(body, Workload):
                self.collect(body)
            elif callable(body):
                maybe = body(0)
                if isinstance(maybe, Workload):
                    self.collect(maybe)
        elif kind == "send":
            value = node._kwargs.get("value")
            if isinstance(value, Workload):
                self.collect(value)
        elif kind == "consume":
            body = node._kwargs.get("body")
            if isinstance(body, Workload):
                self.collect(body)
            elif callable(body):
                maybe = body(0)
                if isinstance(maybe, Workload):
                    self.collect(maybe)
        elif kind == "connect":
            procs = node._kwargs.get("processes") or []
            for p in procs:
                body = getattr(p, "body", None)
                if isinstance(body, Workload):
                    self.collect(body)
        elif kind == "slot_load_u64":
            self._register_tensor(node._kwargs.get("tensor"))

    def _register_tensor(self, t: Any) -> None:
        if not isinstance(t, Tensor):
            return
        base = t.base
        key = id(base)
        if key not in self.tensor_ids:
            self.tensor_ids[key] = len(self.tensors)
            self.tensors.append(base)


def _tensor_to_codegen_info(t: Tensor) -> "cpp.CodegenTensorInfo":
    info = cpp.CodegenTensorInfo()
    info.dtype = _dtype_to_cpp(t.dtype)
    info.shape = [int(d) for d in t.shape]
    return info


def _dtype_to_cpp(dtype: Any) -> "cpp.DType":
    v = getattr(dtype, "value", str(dtype))
    by = {
        "f16": cpp.DType.F16,
        "bf16": cpp.DType.BF16,
        "f32": cpp.DType.F32,
        "f64": cpp.DType.F64,
        "i8": cpp.DType.I8,
        "i16": cpp.DType.I16,
        "i32": cpp.DType.I32,
        "i64": cpp.DType.I64,
        "u8": cpp.DType.U8,
        "u16": cpp.DType.U16,
        "u32": cpp.DType.U32,
        "u64": cpp.DType.U64,
        "bool": cpp.DType.Bool,
    }
    if v not in by:
        raise IRBridgeError(f"Unsupported dtype for codegen: {dtype!r}")
    return by[v]


def _fnv1a_64(s: str) -> int:
    h = 0xCBF29CE484222325
    for b in s.encode("utf-8"):
        h ^= b
        h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


def _axis_val_to_codegen_arg(val: Any) -> "cpp.CodegenAxisArg":
    a = cpp.CodegenAxisArg()
    if hasattr(val, "name"):
        a.is_var = True
        a.var = str(val.name)
        a.u64 = 0
        return a
    if isinstance(val, str):
        a.is_var = False
        a.var = ""
        a.u64 = int(_fnv1a_64(val))
        return a
    try:
        a.is_var = False
        a.var = ""
        a.u64 = int(val) & 0xFFFFFFFFFFFFFFFF
        return a
    except Exception as e:  # noqa: BLE001
        raise IRBridgeError(f"Unsupported axis binding value: {val!r}") from e


def _scalar_val_to_codegen_arg(val: Any, dtype: Any) -> "cpp.CodegenAxisArg":
    """Encode a scalar kernel argument into the u64 args ABI.

    For floats, store IEEE bits in the low bits of u64; kernel code decodes
    based on the scalar param's dtype.
    """
    a = cpp.CodegenAxisArg()
    if isinstance(val, Symbol):
        a.is_var = False
        a.var = str(val.sym)
        a.u64 = int(_fnv1a_64(a.var))
        return a
    if hasattr(val, "name"):
        a.is_var = True
        a.var = str(val.name)
        a.u64 = 0
        return a

    a.is_var = False
    a.var = ""

    dt = getattr(dtype, "value", str(dtype))
    if dt == "f32" or dtype == DType.F32:
        bits_u32 = struct.unpack("<I", struct.pack("<f", float(val)))[0]
        a.u64 = int(bits_u32)
        return a
    if dt == "i32" or dtype == DType.I32:
        a.u64 = int(val) & 0xFFFFFFFF
        return a
    if dt == "i64" or dtype == DType.I64:
        a.u64 = int(val) & 0xFFFFFFFFFFFFFFFF
        return a
    if dt == "bool" or dtype == DType.Bool:
        a.u64 = 1 if bool(val) else 0
        return a

    raise IRBridgeError(f"Unsupported scalar dtype for codegen args: {dtype!r}")


def _index_expr_to_codegen_expr(expr: Any) -> "cpp.CodegenIndexExpr":
    e = cpp.CodegenIndexExpr()
    if hasattr(expr, "name"):
        e.is_var = True
        e.var = str(expr.name)
        e.constant = 0
        return e
    try:
        e.is_var = False
        e.var = ""
        e.constant = int(expr)
        return e
    except Exception as ex:  # noqa: BLE001
        raise IRBridgeError(f"Unsupported index expr: {expr!r}") from ex


def _kernel_ir_to_codegen_def(kernel_ir: Any) -> "cpp.CodegenKernelDef":
    if kernel_ir is None:
        raise IRBridgeError("KernelIR is None - kernel may not have been traced")

    kd = cpp.CodegenKernelDef()
    kd.name = str(kernel_ir.name)

    values: dict[int, Any] = {}
    for _, v in kernel_ir.params:
        values[int(v.id)] = v
    for op in kernel_ir.ops:
        if op.result is not None:
            values[int(op.result.id)] = op.result

    kd_values = {}
    for vid, v in values.items():
        vi = cpp.CodegenKernelValueInfo()
        vi.dtype = _dtype_to_cpp(v.dtype)
        if v.shape is None:
            vi.has_shape = False
            vi.rows = 0
            vi.cols = 0
        else:
            if not isinstance(v.shape[0], int) or not isinstance(v.shape[1], int):
                raise IRBridgeError(f"Kernel value shape must be concrete ints (got {v.shape!r})")
            vi.has_shape = True
            vi.rows = int(v.shape[0])
            vi.cols = int(v.shape[1])
        kd_values[int(vid)] = vi
    kd.values = kd_values

    params = []
    for pname, v in kernel_ir.params:
        pi = cpp.CodegenKernelParamInfo()
        pi.name = str(pname)
        pi.id = int(v.id)
        params.append(pi)
    kd.params = params

    ops = []
    for op in kernel_ir.ops:
        oi = cpp.CodegenKernelOpInfo()
        oi.kind = str(op.kind.name)
        if op.result is None:
            oi.has_result = False
            oi.result = 0
        else:
            oi.has_result = True
            oi.result = int(op.result.id)
        oi.operands = [int(v.id) for v in op.operands]
        oi.attrs = _attrs_to_cpp(op.attrs or {})
        ops.append(oi)
    kd.ops = ops

    return kd


def _kernel_to_codegen_def(kernel: Any) -> "cpp.CodegenKernelDef":
    kd = _kernel_ir_to_codegen_def(kernel.trace())
    attrs: dict[str, Any] = {}
    opts = getattr(kernel, "options", {}) or {}
    if "cpp_src" in opts and opts["cpp_src"] is not None:
        attrs["cpp_src"] = str(opts["cpp_src"])
        if "cpp_includes" in opts and opts["cpp_includes"] is not None:
            attrs["cpp_includes"] = [str(x) for x in opts["cpp_includes"]]
    if "cpp_file_src" in opts and opts["cpp_file_src"] is not None:
        attrs["cpp_file_src"] = str(opts["cpp_file_src"])
        if "cpp_includes" in opts and opts["cpp_includes"] is not None:
            attrs["cpp_includes"] = [str(x) for x in opts["cpp_includes"]]
    kd.attrs = _attrs_to_cpp(attrs)
    return kd


def _attrs_to_cpp(attrs: dict[str, Any]) -> Any:
    # AttrValue = variant<int64,double,bool,string,vector<int64>,vector<string>>
    out = {}
    for k, v in attrs.items():
        if isinstance(v, bool):
            out[str(k)] = bool(v)
        elif isinstance(v, int):
            out[str(k)] = int(v)
        elif isinstance(v, float):
            out[str(k)] = float(v)
        elif isinstance(v, str):
            out[str(k)] = str(v)
        elif isinstance(v, (list, tuple)):
            if all(isinstance(x, int) for x in v):
                out[str(k)] = [int(x) for x in v]
            elif all(isinstance(x, str) for x in v):
                out[str(k)] = [str(x) for x in v]
            else:
                raise IRBridgeError(f"Unsupported attr list type for '{k}': {v!r}")
        else:
            raise IRBridgeError(f"Unsupported attr type for '{k}': {type(v)}")
    return out
