"""
Program execution for PTO Workload-Schedule Programming (PTO-WSP) framework.

Provides compiled program execution for various backends.
"""

from __future__ import annotations
from typing import Any, Optional, Callable, List
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    on a target backend. It handles:
    - Task enumeration from workload IR
    - Scheduling based on dispatch/stream policies
    - Execution on the target backend
    - Synchronization and completion tracking
    - Profiling and tracing (optional)

    Attributes:
        workload: Source workload definition
        target: Target backend name
        options: Compilation options
        stats: Execution statistics
        trace: Execution trace (if tracing enabled)

    Example:
        # Basic execution
        program = workload.compile(target="cpu_sim")
        program.execute()
        program.synchronize()
        print(f"Executed {program.stats.task_count} tasks")

        # With profiling
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
            target: Target backend ("cpu_sim", "ascend_npu", "amd_aie")
            options: Compilation options dict
        """
        self.workload = workload
        self.target = target
        self.options = options
        self.stats = ProgramStats()
        self.trace = ExecutionTrace()

        # Execution state
        self._executor: Optional[ThreadPoolExecutor] = None
        self._futures: list = []
        self._lock = threading.Lock()
        self._complete = threading.Event()
        self._started = False
        self._trace_level = TraceLevel.NONE
        self._task_counter = 0

        # Kernel registry for CPU simulation
        self._kernels: dict[str, Callable] = {}

        # Compile the workload
        start = time.perf_counter()
        self._compiled_plan = self._compile()
        self.stats.compile_time_ms = (time.perf_counter() - start) * 1000

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

    def register_kernel(self, name: str, impl: Callable) -> None:
        """Register a kernel implementation for CPU simulation.

        Args:
            name: Kernel name (must match @kernel function name)
            impl: Python callable implementing the kernel

        Example:
            @kernel
            def my_kernel(a: In[Tensor], b: Out[Tensor]): ...

            def my_kernel_impl(a, b):
                b[:] = a * 2

            program.register_kernel("my_kernel", my_kernel_impl)
        """
        self._kernels[name] = impl

    def _compile(self) -> "ExecutionPlan":
        """Compile workload into an execution plan.

        Returns:
            ExecutionPlan with task enumeration and scheduling
        """
        from pto_wsp.workload import Workload

        # Extract workload structure
        plan = ExecutionPlan()

        # Enumerate tasks from workload
        tasks = self._enumerate_workload(self.workload, ExecutionContext())
        plan.tasks = tasks

        # Apply scheduling based on workload configuration
        if hasattr(self.workload, '_stream_count') and self.workload._stream_count:
            plan.stream_count = self.workload._stream_count

        if hasattr(self.workload, '_dispatch_policy') and self.workload._dispatch_policy:
            plan.dispatch_policy = self.workload._dispatch_policy

        return plan

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
            kernel_name = workload._kwargs.get("kernel", "unknown")
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

    def execute(self) -> None:
        """Execute the compiled program.

        Dispatches tasks according to the schedule and begins execution
        on the target backend. For CPU simulation, uses a thread pool.

        This method is non-blocking. Use synchronize() to wait for completion.

        Raises:
            RuntimeError: If program was already started
        """
        with self._lock:
            if self._started:
                raise RuntimeError("Program already started")
            self._started = True
            self._complete.clear()

        start = time.perf_counter()

        if self.target == "cpu_sim":
            self._execute_cpu_sim()
        else:
            # For other backends, just mark complete (placeholder)
            self._complete.set()

        self.stats.execute_time_ms = (time.perf_counter() - start) * 1000

    def _execute_cpu_sim(self) -> None:
        """Execute using CPU simulation backend.

        Uses ThreadPoolExecutor for parallel task execution.
        """
        plan = self._compiled_plan

        if not plan.tasks:
            self._complete.set()
            return

        self.stats.task_count = len(plan.tasks)

        # Determine parallelism level
        max_workers = plan.stream_count if plan.stream_count else 4

        # Create executor
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

        # Check if workload has sequential dependencies
        has_sequential = self._has_sequential_deps()

        if has_sequential:
            # Execute sequentially
            for task in plan.tasks:
                self._execute_task(task)
            self._complete.set()
        else:
            # Execute in parallel
            self.stats.parallel_tasks = len(plan.tasks)
            self._futures = [
                self._executor.submit(self._execute_task, task)
                for task in plan.tasks
            ]

            # Wait for all in background thread
            def wait_for_all():
                for f in as_completed(self._futures):
                    try:
                        f.result()
                    except Exception as e:
                        pass  # Log error in production
                self._complete.set()

            threading.Thread(target=wait_for_all, daemon=True).start()

    def _has_sequential_deps(self) -> bool:
        """Check if workload has sequential dependencies."""
        if hasattr(self.workload, '_kind'):
            return self.workload._kind == "sequential"
        return False

    def _execute_task(self, task: "TaskInstance") -> None:
        """Execute a single task.

        Args:
            task: TaskInstance to execute
        """
        # Assign task ID
        with self._lock:
            task_id = self._task_counter
            self._task_counter += 1

        thread_id = threading.current_thread().ident

        # Record task start if tracing
        if self._trace_level >= TraceLevel.TIMING:
            task_start = time.perf_counter_ns()

        kernel_impl = self._kernels.get(task.kernel)
        kernel_start = None
        kernel_end = None

        if kernel_impl:
            # Record kernel start if full tracing
            if self._trace_level >= TraceLevel.TIMING:
                kernel_start = time.perf_counter_ns()

            # Call the registered kernel implementation
            try:
                kernel_impl(*task.resources)
            except Exception as e:
                if self._trace_level >= TraceLevel.FULL:
                    # Record error in trace
                    self.trace.add_event(TraceEvent(
                        name=f"error:{task.kernel}",
                        category="error",
                        start_ns=time.perf_counter_ns(),
                        end_ns=time.perf_counter_ns(),
                        task_id=task_id,
                        thread_id=thread_id,
                        metadata={"error": str(e)},
                    ))

            # Record kernel end if full tracing
            if self._trace_level >= TraceLevel.TIMING:
                kernel_end = time.perf_counter_ns()

        # Record trace events
        if self._trace_level >= TraceLevel.TIMING:
            task_end = time.perf_counter_ns()

            # Task event
            self.trace.add_event(TraceEvent(
                name=f"task:{task.kernel}",
                category="task",
                start_ns=task_start,
                end_ns=task_end,
                task_id=task_id,
                thread_id=thread_id,
                metadata={"bindings": task.bindings} if self._trace_level >= TraceLevel.FULL else {},
            ))

            # Kernel event (only if kernel was executed)
            if kernel_start is not None and kernel_end is not None:
                self.trace.add_event(TraceEvent(
                    name=task.kernel,
                    category="kernel",
                    start_ns=kernel_start,
                    end_ns=kernel_end,
                    task_id=task_id,
                    thread_id=thread_id,
                    metadata={"params": task.params} if self._trace_level >= TraceLevel.FULL else {},
                ))

    def synchronize(self) -> None:
        """Wait for all tasks to complete.

        Blocks until all dispatched tasks have finished executing.
        """
        self._complete.wait()

        # Cleanup executor
        if self._executor:
            self._executor.shutdown(wait=True)
            self._executor = None

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
    resources: list[Any]
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
