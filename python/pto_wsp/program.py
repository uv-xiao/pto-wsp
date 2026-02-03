"""
Program execution for PTO Workload-Schedule Programming (PTO-WSP) framework.

Provides compiled program execution for various backends.

Codegen-first implementation:
- `@kernel` and `@workload` lower to C++ (CPU-sim today) and are compiled into a
  shared library that is loaded/executed via the `pto_ir_cpp` C++ bindings
  (no Python `ctypes` runtime path).
- Python fallback execution has been removed.
"""

from __future__ import annotations
from typing import Any, Optional, Callable, List
from dataclasses import dataclass, field
from enum import IntEnum, auto
import threading
import time


class TraceLevel(IntEnum):
    """Trace level for program execution profiling.

    Controls the granularity of tracing information collected during execution.
    IntEnum allows comparison with >= for level checking.
    """
    NONE = 0           # No tracing
    SUMMARY = 1        # Only summary statistics
    TIMING = 2         # Per-task timing
    FULL = 3           # Full trace with all details


@dataclass
class TraceEvent:
    """A single trace event during program execution.

    Attributes:
        name: Event name (e.g., kernel name, phase name)
        category: Event category (task, kernel, sync, etc.)
        start_ns: Start timestamp in nanoseconds
        end_ns: End timestamp in nanoseconds
        task_id: Optional task ID
        thread_id: Thread that executed the event
        metadata: Additional event metadata
    """
    name: str
    category: str
    start_ns: int
    end_ns: int
    task_id: Optional[int] = None
    thread_id: Optional[int] = None
    metadata: dict = field(default_factory=dict)

    @property
    def duration_us(self) -> float:
        """Duration in microseconds."""
        return (self.end_ns - self.start_ns) / 1000

    @property
    def duration_ms(self) -> float:
        """Duration in milliseconds."""
        return (self.end_ns - self.start_ns) / 1_000_000


@dataclass
class ExecutionTrace:
    """Collection of trace events from program execution.

    Provides analysis methods for profiling data.
    Thread-safe for concurrent event recording.
    """
    events: List[TraceEvent] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def add_event(self, event: TraceEvent) -> None:
        """Add a trace event (thread-safe)."""
        with self._lock:
            self.events.append(event)

    def get_events(self, category: Optional[str] = None) -> List[TraceEvent]:
        """Get events, optionally filtered by category."""
        if category is None:
            return self.events
        return [e for e in self.events if e.category == category]

    def get_task_events(self) -> List[TraceEvent]:
        """Get all task execution events."""
        return self.get_events("task")

    def get_kernel_events(self, kernel_name: Optional[str] = None) -> List[TraceEvent]:
        """Get kernel execution events, optionally filtered by name."""
        kernel_events = self.get_events("kernel")
        if kernel_name is None:
            return kernel_events
        return [e for e in kernel_events if e.name == kernel_name]

    def total_time_ms(self) -> float:
        """Total wall-clock time from first to last event."""
        if not self.events:
            return 0.0
        start = min(e.start_ns for e in self.events)
        end = max(e.end_ns for e in self.events)
        return (end - start) / 1_000_000

    def kernel_time_ms(self) -> float:
        """Total time spent in kernel execution."""
        return sum(e.duration_ms for e in self.get_events("kernel"))

    def summary(self) -> dict:
        """Generate summary statistics."""
        task_events = self.get_task_events()
        kernel_events = self.get_events("kernel")

        return {
            "total_events": len(self.events),
            "task_count": len(task_events),
            "kernel_invocations": len(kernel_events),
            "total_time_ms": self.total_time_ms(),
            "kernel_time_ms": self.kernel_time_ms(),
            "avg_task_time_us": (
                sum(e.duration_us for e in task_events) / len(task_events)
                if task_events else 0.0
            ),
            "unique_kernels": len(set(e.name for e in kernel_events)),
        }

    def to_chrome_trace(self) -> dict:
        """Export trace in Chrome Tracing format for visualization.

        The output can be loaded in chrome://tracing or Perfetto.

        Returns:
            Dict in Chrome Tracing JSON format
        """
        chrome_events = []
        for event in self.events:
            chrome_events.append({
                "name": event.name,
                "cat": event.category,
                "ph": "X",  # Complete event
                "ts": event.start_ns / 1000,  # Microseconds
                "dur": (event.end_ns - event.start_ns) / 1000,
                "pid": 1,
                "tid": event.thread_id or 0,
                "args": event.metadata,
            })
        return {"traceEvents": chrome_events}

    def print_summary(self) -> None:
        """Print a human-readable summary."""
        s = self.summary()
        print(f"Execution Trace Summary:")
        print(f"  Total events: {s['total_events']}")
        print(f"  Tasks executed: {s['task_count']}")
        print(f"  Kernel invocations: {s['kernel_invocations']}")
        print(f"  Unique kernels: {s['unique_kernels']}")
        print(f"  Total time: {s['total_time_ms']:.3f} ms")
        print(f"  Kernel time: {s['kernel_time_ms']:.3f} ms")
        print(f"  Avg task time: {s['avg_task_time_us']:.2f} us")


@dataclass
class ProgramStats:
    """Execution statistics for a program run.

    Attributes:
        compile_time_ms: Time spent compiling (ms)
        execute_time_ms: Time spent executing (ms)
        task_count: Number of tasks executed
        parallel_tasks: Number of tasks executed in parallel
    """
    compile_time_ms: float = 0.0
    execute_time_ms: float = 0.0
    task_count: int = 0
    parallel_tasks: int = 0
    total_cycles: int = 0


@dataclass
class ExecutionContext:
    """Context for task execution.

    Tracks current loop variable bindings during task enumeration.
    """
    bindings: dict[str, int] = field(default_factory=dict)

    def with_binding(self, name: str, value: int) -> "ExecutionContext":
        """Create new context with additional binding."""
        new_bindings = self.bindings.copy()
        new_bindings[name] = value
        return ExecutionContext(bindings=new_bindings)


class Program:
    """Compiled program ready for execution.

    The Program class represents a compiled workload that can be executed
    on a target backend via the v9 **codegen-first** pipeline:

    - Python authors workloads/kernels and bridges them to a C++ `ir::Module`.
    - C++ codegen emits backend artifacts:
      - `target="cpu_sim"`: builds a cached shared library and executes it via `dlopen`.
      - `target="ascend_npu"`: emits a source tree for inspection (device build/run requires Ascend/CANN).

    Attributes:
        workload: Source workload definition
        target: Target backend name
        options: Compilation options
        stats: Execution statistics
        trace: Execution trace (if tracing enabled)
        using_cpp_backend: True if backed by generated artifacts

    Example:
        # Basic execution
        program = workload.compile(target="cpu_sim")
        print(f"Using codegen backend: {program.using_cpp_backend}")
        program.execute()
        program.synchronize()
        print(f"Executed {program.stats.task_count} tasks")

        # With profiling (requires C++ backend)
        program = workload.compile(target="cpu_sim")
        program.enable_tracing(TraceLevel.TIMING)
        program.execute()
        program.synchronize()
        program.trace.print_summary()

        # Export trace for visualization
        import json
        with open("trace.json", "w") as f:
            json.dump(program.trace.to_chrome_trace(), f)
    """

    def __init__(self, workload: Any, target: str, options: dict):
        """Initialize a compiled program.

        Args:
            workload: Source Workload instance
            target: Target backend ("cpu_sim", "ascend_npu")
            options: Compilation options dict
        """
        self.workload = workload
        self.target = target
        self.options = options
        self.stats = ProgramStats()
        self.trace = ExecutionTrace()
        self._pto_runtime_platform: str | None = None

        # Execution state
        self._lock = threading.Lock()
        self._complete = threading.Event()
        self._started = False
        self._trace_level = TraceLevel.NONE

        # Dynamic axis sizes (for DenseDyn codegen). Can be updated between runs
        # without recompiling the generated workload artifact.
        self._axis_sizes: dict[str, int] = self._infer_axis_sizes(workload)
        axis_sizes_opt = options.get("axis_sizes") if isinstance(options, dict) else None
        if isinstance(axis_sizes_opt, dict):
            for k, v in axis_sizes_opt.items():
                self._axis_sizes[str(k)] = int(v)

        # Runtime-bound symbols (hashed IDs) used by dynamic codegen artifacts.
        # These can be changed between runs without rebuilding the artifact.
        self._symbols_u64: dict[int, int] = {}
        self._symbols_ptr: dict[int, Any] = {}

        # Codegen-first runtime artifacts
        self._codegen_exec = None
        self._codegen_tensors = []
        self._codegen_thread = None
        self._codegen_artifact_dir: str | None = None
        self._can_execute: bool = True

        # Compile the workload
        start = time.perf_counter()
        self._compiled_plan = self._compile()
        self.stats.compile_time_ms = (time.perf_counter() - start) * 1000

    @property
    def using_cpp_backend(self) -> bool:
        """Check if this program is backed by generated C++ code.

        Returns:
            True if compiled and ready to execute.
        """
        return self._codegen_exec is not None or self._codegen_artifact_dir is not None

    @property
    def codegen_artifact_dir(self) -> str | None:
        """Path to the emitted codegen artifact directory (if codegen-only target)."""
        return self._codegen_artifact_dir

    def enable_tracing(self, level: TraceLevel = TraceLevel.TIMING) -> "Program":
        """Enable execution tracing.

        Args:
            level: Trace level (NONE, SUMMARY, TIMING, FULL)

        Returns:
            self for method chaining
        """
        self._trace_level = level
        return self

    def disable_tracing(self) -> "Program":
        """Disable execution tracing.

        Returns:
            self for method chaining
        """
        self._trace_level = TraceLevel.NONE
        return self

    def clear_trace(self) -> None:
        """Clear collected trace events."""
        self.trace = ExecutionTrace()

    def register_kernel(self, name: str, impl: Callable, *, pass_task: bool = False) -> None:
        """Register a kernel implementation for CPU simulation.

        Args:
            name: Kernel name (must match @kernel function name)
            impl: Python callable implementing the kernel
            pass_task: If True, pass the full Task object to impl instead of resources.
                       This allows access to task.params (loop indices) and task.bindings.

        Example (resource-based - default):
            @kernel
            def my_kernel(a: In[Tensor], b: Out[Tensor]): ...

            def my_kernel_impl(a, b):
                b[:] = a * 2

            program.register_kernel("my_kernel", my_kernel_impl)

        Example (task-based):
            def my_kernel_impl(task):
                # Access loop indices via task.params or task.bindings
                batch_idx = task.bindings.get("b", 0)
                # Access resources via task.resources
                a, b = task.resources
                b[:] = a * 2

            program.register_kernel("my_kernel", my_kernel_impl, pass_task=True)
        """
        raise RuntimeError("program.register_kernel() is not supported in codegen-first mode")

    def set_axis_sizes(self, axis_sizes: dict[str, int]) -> None:
        """Update runtime axis sizes for the next execute().

        This is primarily used by DenseDyn codegen: generated workload loops can
        query axis sizes at runtime via `RuntimeContext.get_axis_size`.
        """
        for k, v in axis_sizes.items():
            self._axis_sizes[str(k)] = int(v)

    @staticmethod
    def _fnv1a_64(s: str) -> int:
        h = 0xCBF29CE484222325
        for b in s.encode("utf-8"):
            h ^= b
            h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
        return h

    def set_symbol_u64(self, name: str, value: int) -> None:
        """Bind a runtime symbol (u64) for codegen artifacts.

        The generated code treats the symbol ID as a stable 64-bit hash of `name`.
        """
        self._symbols_u64[self._fnv1a_64(str(name))] = int(value) & 0xFFFFFFFFFFFFFFFF

    def set_symbol_f32(self, name: str, value: float) -> None:
        """Bind a runtime symbol as an f32 bit-pattern stored in u64."""
        import struct

        bits_u32 = struct.unpack("<I", struct.pack("<f", float(value)))[0]
        self.set_symbol_u64(name, int(bits_u32))

    def set_symbol_ptr(self, name: str, arr: Any) -> None:
        """Bind a runtime symbol to a pointer-backed buffer (numpy array expected)."""
        self._symbols_ptr[self._fnv1a_64(str(name))] = arr

    @staticmethod
    def _infer_axis_sizes(workload: Any) -> dict[str, int]:
        """Infer axis sizes from the workload tree (best-effort)."""
        from pto_wsp.workload import Workload as WorkloadNode

        out: dict[str, int] = {}

        def walk(node: Any) -> None:
            if not isinstance(node, WorkloadNode):
                return

            kind = node._kind
            if kind in ("parallel_for", "for_each"):
                axis = node._kwargs.get("axis")
                var_name = node._kwargs.get("var_name")
                if var_name and hasattr(axis, "size"):
                    try:
                        out[str(var_name)] = int(axis.size)
                    except Exception:
                        pass
                walk(node._kwargs.get("body"))
                return

            if kind in ("combine", "sequential"):
                for child in node._kwargs.get("workloads", []) or []:
                    walk(child)
                return

        walk(workload)
        return out

    def _compile(self) -> "ExecutionPlan":
        """Compile workload into an execution plan.

        Compiles the workload to a codegen shared library and also enumerates
        tasks for stats (e.g. `Program.stats.task_count`).
        """
        from pto_wsp.workload import Workload
        from pto_wsp.errors import CompileError

        try:
            self._compile_codegen()

            # Build ExecutionPlan for stats
            plan = ExecutionPlan()
            plan.tasks = self._enumerate_workload(self.workload, ExecutionContext())
            if hasattr(self.workload, "_stream_count") and self.workload._stream_count:
                plan.stream_count = self.workload._stream_count
            if hasattr(self.workload, "_dispatch_policy") and self.workload._dispatch_policy:
                plan.dispatch_policy = self.workload._dispatch_policy
            return plan
        except Exception as e:
            raise CompileError(f"Codegen compilation failed: {e}") from e

    def _compile_codegen(self) -> None:
        """Compile workload + kernels into a shared library and load entrypoint."""
        from pto_wsp.ir_bridge import check_cpp_bindings, cpp, workload_to_codegen_ir

        module_name = self.workload._name if getattr(self.workload, "_name", None) else "main"

        check_cpp_bindings()
        build = workload_to_codegen_ir(self.workload, module_name=module_name)
        opts = cpp.CompileOptions()
        target = str(self.target)
        if target == "pto_runtime_a2a3sim":
            opts.target = "a2a3sim_codegen"
            self._pto_runtime_platform = "a2a3sim"
        elif target == "pto_runtime_a2a3":
            opts.target = "a2a3_codegen"
            self._pto_runtime_platform = "a2a3"
        else:
            opts.target = target
        result = cpp.compile_codegen(build.module, opts)
        self._can_execute = bool(result.get("can_execute", True))
        if self._can_execute:
            self._codegen_exec = cpp.CodegenExecutable(str(result["so_path"]), str(result["entrypoint"]))
        else:
            self._codegen_exec = None
            self._codegen_artifact_dir = str(result.get("artifact_dir", ""))
        self._codegen_tensors = build.tensors

    def _enumerate_workload(self, workload: Any, ctx: ExecutionContext) -> list["TaskInstance"]:
        """Recursively enumerate tasks from a workload.

        Args:
            workload: Workload node to enumerate
            ctx: Current execution context with loop bindings

        Returns:
            List of TaskInstance objects
        """
        from pto_wsp.workload import Workload, Task
        from pto_wsp.builder import LoopFrame, ConditionalFrame

        if not isinstance(workload, Workload):
            return []

        kind = workload._kind
        tasks = []

        if kind == "task":
            # Leaf task node - create TaskInstance
            kernel = workload._kwargs.get("kernel", "unknown")
            kernel_name = kernel.name if hasattr(kernel, "name") else str(kernel)
            params = workload._kwargs.get("params", [])
            resources = workload._kwargs.get("resources", [])

            # Resolve loop variables in params
            resolved_params = []
            for p in params:
                if hasattr(p, 'index'):
                    # LoopVar - get value from context
                    name = getattr(p, 'name', str(p))
                    resolved_params.append(ctx.bindings.get(name, p.index))
                elif isinstance(p, int):
                    resolved_params.append(p)
                else:
                    resolved_params.append(p)

            task = TaskInstance(
                kernel=kernel_name,
                params=resolved_params,
                resources=resources,
                bindings=ctx.bindings.copy()
            )
            tasks.append(task)

        elif kind == "parallel_for":
            # Parallel loop - enumerate all iterations
            axis = workload._kwargs.get("axis")
            var_name = workload._kwargs.get("var_name", "i")
            body = workload._kwargs.get("body")

            if axis is not None and body:
                size = self._get_axis_size(axis)
                for i in range(size):
                    new_ctx = ctx.with_binding(var_name, i)
                    # Body can be a lambda (legacy API) or a Workload
                    if callable(body):
                        body_workload = body(i)
                        tasks.extend(self._enumerate_workload(body_workload, new_ctx))
                    else:
                        tasks.extend(self._enumerate_workload(body, new_ctx))

        elif kind == "sequential":
            # Sequential can be either:
            # 1. A list of workloads: sequential(w1, w2, w3)
            # 2. A loop with axis and body
            workloads = workload._kwargs.get("workloads")
            if workloads:
                # Sequential list of workloads
                for w in workloads:
                    tasks.extend(self._enumerate_workload(w, ctx))
            else:
                # Sequential loop
                axis = workload._kwargs.get("axis")
                var_name = workload._kwargs.get("var_name", "i")
                body = workload._kwargs.get("body")

                if axis is not None and body:
                    size = self._get_axis_size(axis)
                    for i in range(size):
                        new_ctx = ctx.with_binding(var_name, i)
                        # Body can be a lambda (legacy API) or a Workload
                        if callable(body):
                            body_workload = body(i)
                            tasks.extend(self._enumerate_workload(body_workload, new_ctx))
                        else:
                            tasks.extend(self._enumerate_workload(body, new_ctx))

        elif kind == "for_each":
            # Sequential iteration (legacy API)
            axis = workload._kwargs.get("axis")
            var_name = workload._kwargs.get("var_name", "i")
            body = workload._kwargs.get("body")

            if axis is not None and body:
                size = self._get_axis_size(axis)
                for i in range(size):
                    new_ctx = ctx.with_binding(var_name, i)
                    # Body can be a lambda (legacy API) or a Workload
                    if callable(body):
                        body_workload = body(i)
                        tasks.extend(self._enumerate_workload(body_workload, new_ctx))
                    else:
                        tasks.extend(self._enumerate_workload(body, new_ctx))

        elif kind == "combine":
            # Combined workloads - enumerate each
            workloads = workload._kwargs.get("workloads", [])
            for w in workloads:
                tasks.extend(self._enumerate_workload(w, ctx))

        elif kind == "select":
            # Sparse selection - enumerate selected indices
            sparse = workload._kwargs.get("sparse")
            var_name = workload._kwargs.get("var_name", "e")
            body = workload._kwargs.get("body")

            if sparse is not None and body:
                # Get selected indices from Sparse axis
                indices = self._get_sparse_indices(sparse)
                for idx in indices:
                    new_ctx = ctx.with_binding(var_name, idx)
                    if callable(body):
                        body_workload = body(idx)
                        tasks.extend(self._enumerate_workload(body_workload, new_ctx))
                    else:
                        tasks.extend(self._enumerate_workload(body, new_ctx))

        elif kind == "cond":
            # Conditional workload - evaluate predicate to select branch
            predicate = workload._kwargs.get("predicate")
            then_workload = workload._kwargs.get("then_workload")
            else_workload = workload._kwargs.get("else_workload")

            # Evaluate predicate (may be runtime value or constant)
            # In builder mode, we take the then branch by default
            # In execution mode with real data, we would evaluate the predicate
            branch_taken = True
            if callable(predicate):
                try:
                    branch_taken = bool(predicate())
                except Exception:
                    branch_taken = True  # Default to then branch
            elif isinstance(predicate, bool):
                branch_taken = predicate

            if branch_taken and then_workload:
                tasks.extend(self._enumerate_workload(then_workload, ctx))
            elif not branch_taken and else_workload:
                tasks.extend(self._enumerate_workload(else_workload, ctx))

        elif kind == "pipeline":
            # Pipeline workload - enumerate all stages
            stages = workload._kwargs.get("stages", [])
            for stage in stages:
                tasks.extend(self._enumerate_workload(stage, ctx))

        elif kind == "empty":
            pass

        elif kind == "send":
            # CSP send - enumerate the value workload
            value = workload._kwargs.get("value")
            if value:
                tasks.extend(self._enumerate_workload(value, ctx))

        elif kind == "consume":
            # CSP consume - handled at runtime
            pass

        elif kind == "connect":
            # CSP connect - enumerate process bodies
            processes = workload._kwargs.get("processes", [])
            for proc in processes:
                if hasattr(proc, 'body') and proc.body:
                    tasks.extend(self._enumerate_workload(proc.body, ctx))

        return tasks

    def _get_axis_size(self, axis: Any) -> int:
        """Get the size of an axis.

        Args:
            axis: Axis object or integer size

        Returns:
            Integer size of the axis
        """
        if isinstance(axis, int):
            return axis
        if hasattr(axis, 'size'):
            return axis.size
        if hasattr(axis, '__len__'):
            return len(axis)
        return 0

    def _get_sparse_indices(self, sparse: Any) -> list[int]:
        """Get the indices from a Sparse axis.

        Args:
            sparse: Sparse axis object or list of indices

        Returns:
            List of selected indices
        """
        if isinstance(sparse, (list, tuple)):
            return list(sparse)
        if hasattr(sparse, 'indices'):
            return list(sparse.indices)
        if hasattr(sparse, '__iter__'):
            return list(sparse)
        return []

    def execute(self) -> None:
        """Execute the compiled program (codegen-first).

        This method is non-blocking. Use synchronize() to wait for completion.

        Raises:
            RuntimeError: If program was already started or not compiled.
        """
        with self._lock:
            if self._started and not self._complete.is_set():
                raise RuntimeError("Program already started")
            self._started = True
            self._complete.clear()

        start = time.perf_counter()

        if not self._can_execute:
            if self._pto_runtime_platform and self._codegen_artifact_dir:
                self._execute_pto_runtime()
                self.stats.execute_time_ms = (time.perf_counter() - start) * 1000
                return
            raise RuntimeError(
                f"Target '{self.target}' is codegen-only in this environment; "
                f"artifact_dir={self._codegen_artifact_dir!r}"
            )
        if self._codegen_exec is None:
            raise RuntimeError("Program not compiled (missing codegen executable)")

        self._execute_codegen()

        self.stats.execute_time_ms = (time.perf_counter() - start) * 1000

    def _execute_pto_runtime(self) -> None:
        import numpy as np
        from pto_wsp.types import DType, Tensor
        from pto_wsp import pto_runtime_runner

        if not self._pto_runtime_platform or not self._codegen_artifact_dir:
            raise RuntimeError("pto-runtime execution requested but no artifact/platform is available")

        self.stats.task_count = len(self._compiled_plan.tasks)

        def ensure_numpy(t: Tensor) -> np.ndarray:
            if t.data is None:
                if t.dtype == DType.F32:
                    t.data = np.zeros(t.shape, dtype=np.float32)
                elif t.dtype == DType.F16:
                    t.data = np.zeros(t.shape, dtype=np.float16)
                elif t.dtype == DType.I32:
                    t.data = np.zeros(t.shape, dtype=np.int32)
                elif t.dtype == DType.I64:
                    t.data = np.zeros(t.shape, dtype=np.int64)
                else:
                    raise ValueError(f"Unsupported dtype for pto-runtime backend: {t.dtype}")
            if not isinstance(t.data, np.ndarray):
                raise TypeError(f"Tensor.data must be a numpy.ndarray for pto-runtime backend (got {type(t.data)})")
            return t.data

        def run():
            try:
                arrays = [ensure_numpy(t) for t in self._codegen_tensors]
                pto_runtime_runner.run_host_build_graph(
                    artifact_dir=self._codegen_artifact_dir,
                    platform=self._pto_runtime_platform,
                    arrays=arrays,
                )
            finally:
                self._complete.set()

        self._codegen_thread = threading.Thread(target=run, daemon=True)
        self._codegen_thread.start()

    def _execute_codegen(self) -> None:
        """Execute generated workload code in a background thread."""
        import numpy as np
        from pto_wsp.types import DType, Tensor

        self.stats.task_count = len(self._compiled_plan.tasks)

        def ensure_numpy(t: Tensor) -> np.ndarray:
            if t.data is None:
                if t.dtype == DType.F32:
                    t.data = np.zeros(t.shape, dtype=np.float32)
                elif t.dtype == DType.F16:
                    t.data = np.zeros(t.shape, dtype=np.float16)
                elif t.dtype == DType.I32:
                    t.data = np.zeros(t.shape, dtype=np.int32)
                elif t.dtype == DType.I64:
                    t.data = np.zeros(t.shape, dtype=np.int64)
                else:
                    raise ValueError(f"Unsupported dtype for codegen runtime: {t.dtype}")
            if not isinstance(t.data, np.ndarray):
                raise TypeError(f"Tensor.data must be a numpy.ndarray for codegen runtime (got {type(t.data)})")
            return t.data

        def run():
            try:
                if self._codegen_exec is None:
                    raise RuntimeError("Program not compiled (missing codegen executable)")
                arrays = [ensure_numpy(t) for t in self._codegen_tensors]
                axis_sizes = {str(k): int(v) for k, v in self._axis_sizes.items()}
                cycles = self._codegen_exec.run_with_symbols(arrays, axis_sizes, dict(self._symbols_u64), dict(self._symbols_ptr))
                self.stats.total_cycles = int(cycles)
            finally:
                self._complete.set()

        self._codegen_thread = threading.Thread(target=run, daemon=True)
        self._codegen_thread.start()

    def synchronize(self) -> None:
        """Wait for all tasks to complete.

        Blocks until all dispatched tasks have finished executing.
        """
        self._complete.wait()
        t = self._codegen_thread
        if t is not None:
            t.join(timeout=5.0)

    def is_complete(self) -> bool:
        """Check if all tasks have completed.

        Returns:
            True if all tasks are done, False otherwise
        """
        return self._complete.is_set()


@dataclass
class TaskInstance:
    """A concrete task instance for execution.

    Represents a single kernel invocation with resolved parameters.

    Attributes:
        kernel: Kernel name
        params: Resolved parameter values (loop indices)
        resources: Input/output resources
        bindings: Loop variable bindings at creation time
    """
    kernel: str
    params: list[Any]
    resources: Any
    bindings: dict[str, int] = field(default_factory=dict)

    def get(self, axis: str) -> int:
        """Get axis value for this task.

        Args:
            axis: Axis/variable name (e.g., "batch", "head")

        Returns:
            Integer index value for the axis

        Example:
            # Inside dispatch policy
            policy = DispatchPolicy.affinity(lambda t: t.get("batch"))
        """
        return self.bindings.get(axis, 0)


@dataclass
class ExecutionPlan:
    """Compiled execution plan for a workload.

    Contains enumerated tasks and scheduling configuration.
    """
    tasks: list[TaskInstance] = field(default_factory=list)
    stream_count: int = 1
    dispatch_policy: Any = None
    timing_policy: Any = None
