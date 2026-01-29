"""
Workload class for PTO Workload-Schedule Programming (PTO-WSP) framework.

Workloads are typed expressions describing task generation.
"""

from __future__ import annotations
from typing import Any, Callable, Optional
from dataclasses import dataclass, field


@dataclass
class Schedule:
    """Schedule configuration for workload execution.

    Contains all scheduling parameters that affect how a workload
    is compiled and executed on the target backend.

    Attributes:
        dispatch: Dispatch policy for task → executor assignment
        streams: Number of concurrent streams
        stream_by: Function for task → stream assignment
        timing: Timing policy for task issue scheduling
    """
    dispatch: Any = None
    streams: int = 1
    stream_by: Optional[Callable] = None
    timing: Any = None

    def to_dict(self) -> dict:
        """Convert to dict for IR bridge compatibility."""
        result = {}
        if self.dispatch is not None:
            result["dispatch"] = self.dispatch
        if self.streams != 1:
            result["streams"] = self.streams
        if self.stream_by is not None:
            result["stream_by"] = self.stream_by
        if self.timing is not None:
            result["timing"] = self.timing
        return result


@dataclass
class Task:
    """Single kernel invocation.

    Represents a task in the workload with kernel reference,
    parameters (loop indices), and resources (tensors).

    Attributes:
        kernel: Kernel name or KernelRef
        params: List of parameter values (typically loop indices)
        resources: List of input/output resources (tensors)
        id: Unique task identifier (assigned during enumeration)
        bindings: Axis name to value bindings for dispatch/stream policies
    """
    kernel: str
    params: list[Any]
    resources: list[Any]
    id: int = 0
    bindings: dict[str, int] = None

    def __post_init__(self):
        """Initialize bindings if not provided."""
        if self.bindings is None:
            self.bindings = {}

    def get(self, axis: str, default: int = 0) -> int:
        """Get axis value for dispatch/stream policies.

        Args:
            axis: Axis/variable name (e.g., "batch", "head", "b", "h")
            default: Default value if axis not found

        Returns:
            Integer index value for the axis, or default if not found

        Example:
            # Inside dispatch policy
            policy = DispatchPolicy.affinity(lambda t: t.get("batch"))

            # Inside stream_by function
            stream_fn = lambda t: t.get("head") % 4
        """
        # First check bindings dict
        if self.bindings and axis in self.bindings:
            return self.bindings[axis]

        # Fall back to checking params by position (for legacy compatibility)
        # This assumes params are ordered by axis order
        return default

    def get_axis_value(self, index: int) -> int:
        """Get axis value by position in params.

        Args:
            index: Position index in the params list

        Returns:
            Integer value at that position, or 0 if out of bounds

        Example:
            # For a task with params [batch_idx, tile_m, tile_n]
            batch_idx = task.get_axis_value(0)  # First axis
            tile_m = task.get_axis_value(1)     # Second axis
            tile_n = task.get_axis_value(2)     # Third axis
        """
        if self.params and 0 <= index < len(self.params):
            return self.params[index]
        return 0


class Workload:
    """Typed workload expression.

    Workloads can be composed and scheduled using combinator-style API:

        program = (workload
            .dispatch(DispatchPolicy.round_robin(4))
            .streams(2)
            .stream_by(lambda t: t.params[0] % 2)
            .compile())
    """

    def __init__(self, kind: str, name: Optional[str] = None, **kwargs):
        """Initialize a Workload.

        Args:
            kind: Type of workload node (task, parallel_for, sequential, etc.)
            name: Optional name for the workload (used in IR generation)
            **kwargs: Additional keyword arguments specific to the workload kind
        """
        self._kind = kind
        self._kwargs = kwargs
        # Workload metadata for IR bridge
        self._name: Optional[str] = name
        self._params: list[tuple[str, Any]] = []  # List of (name, axis) tuples
        # Structured schedule configuration
        self._schedule: Schedule = Schedule()
        # Legacy attributes (kept for compatibility during transition)
        self._dispatch_policy = None
        self._stream_count = 1
        self._stream_by_fn = None
        self._timing_policy = None
        self._spatial_grid = None
        self._layouts = []
        # Task graph config (R9)
        self._task_graph_config = None
        # Type checking integration (R3)
        self._type_errors = []

    def enumerate(self) -> list[Task]:
        """Enumerate all tasks in the workload.

        Recursively traverses the workload tree and generates concrete
        Task instances for each kernel invocation. Loop variables are
        resolved to their concrete values.

        Returns:
            List of Task instances in execution order

        Example:
            @workload
            def attention(batch, heads):
                for b, h in P(batch, heads):
                    attn[b, h](Q=Q[b,h], K=K[b], V=V[b], O=O[b,h])

            w = attention(4, 8)  # 4 batches, 8 heads
            tasks = w.enumerate()
            print(len(tasks))  # 32 tasks
        """
        return self._enumerate_recursive({})

    def _enumerate_recursive(self, bindings: dict[str, int]) -> list[Task]:
        """Recursively enumerate tasks with current bindings.

        Args:
            bindings: Current loop variable bindings

        Returns:
            List of Task instances
        """
        tasks = []

        if self._kind == "task":
            # Leaf task node - create Task instance
            kernel_name = self._kwargs.get("kernel", "unknown")
            params = self._kwargs.get("params", [])
            resources = self._kwargs.get("resources", [])

            # Resolve loop variables in params
            resolved_params = []
            for p in params:
                if hasattr(p, 'index'):
                    # LoopVar - get value from bindings
                    name = getattr(p, 'name', str(p))
                    resolved_params.append(bindings.get(name, p.index))
                elif isinstance(p, int):
                    resolved_params.append(p)
                else:
                    resolved_params.append(p)

            task = Task(
                kernel=kernel_name,
                params=resolved_params,
                resources=resources,
                id=len(tasks),
                bindings=bindings.copy()
            )
            tasks.append(task)

        elif self._kind == "parallel_for":
            # Parallel loop - enumerate all iterations
            axis = self._kwargs.get("axis")
            var_name = self._kwargs.get("var_name", "i")
            body = self._kwargs.get("body")

            if axis is not None and body:
                size = self._get_axis_size(axis)
                for i in range(size):
                    new_bindings = bindings.copy()
                    new_bindings[var_name] = i
                    # Body can be a lambda (legacy API) or a Workload
                    if callable(body):
                        # Call lambda with index to get actual body workload
                        body_workload = body(i)
                        if hasattr(body_workload, '_enumerate_recursive'):
                            tasks.extend(body_workload._enumerate_recursive(new_bindings))
                    elif hasattr(body, '_enumerate_recursive'):
                        tasks.extend(body._enumerate_recursive(new_bindings))

        elif self._kind == "sequential":
            # Sequential can be either:
            # 1. A list of workloads: sequential(w1, w2, w3)
            # 2. A loop with axis and body
            workloads = self._kwargs.get("workloads")
            if workloads:
                # Sequential list of workloads
                for w in workloads:
                    if hasattr(w, '_enumerate_recursive'):
                        tasks.extend(w._enumerate_recursive(bindings))
            else:
                # Sequential loop
                axis = self._kwargs.get("axis")
                var_name = self._kwargs.get("var_name", "i")
                body = self._kwargs.get("body")

                if axis is not None and body:
                    size = self._get_axis_size(axis)
                    for i in range(size):
                        new_bindings = bindings.copy()
                        new_bindings[var_name] = i
                        # Body can be a lambda (legacy API) or a Workload
                        if callable(body):
                            body_workload = body(i)
                            if hasattr(body_workload, '_enumerate_recursive'):
                                tasks.extend(body_workload._enumerate_recursive(new_bindings))
                        elif hasattr(body, '_enumerate_recursive'):
                            tasks.extend(body._enumerate_recursive(new_bindings))

        elif self._kind == "for_each":
            # Sequential iteration - enumerate in order (legacy API)
            axis = self._kwargs.get("axis")
            var_name = self._kwargs.get("var_name", "i")
            body = self._kwargs.get("body")

            if axis is not None and body:
                size = self._get_axis_size(axis)
                for i in range(size):
                    new_bindings = bindings.copy()
                    new_bindings[var_name] = i
                    # Body can be a lambda (legacy API) or a Workload
                    if callable(body):
                        body_workload = body(i)
                        if hasattr(body_workload, '_enumerate_recursive'):
                            tasks.extend(body_workload._enumerate_recursive(new_bindings))
                    elif hasattr(body, '_enumerate_recursive'):
                        tasks.extend(body._enumerate_recursive(new_bindings))

        elif self._kind == "combine":
            # Combined workloads - enumerate each
            workloads = self._kwargs.get("workloads", [])
            for w in workloads:
                if hasattr(w, '_enumerate_recursive'):
                    tasks.extend(w._enumerate_recursive(bindings))

        elif self._kind == "send":
            # CSP send - enumerate the value workload
            value = self._kwargs.get("value")
            if value and hasattr(value, '_enumerate_recursive'):
                tasks.extend(value._enumerate_recursive(bindings))

        elif self._kind == "consume":
            # CSP consume - handled at runtime
            pass

        elif self._kind == "connect":
            # CSP connect - enumerate process bodies
            processes = self._kwargs.get("processes", [])
            for proc in processes:
                if hasattr(proc, 'body') and proc.body:
                    if hasattr(proc.body, '_enumerate_recursive'):
                        tasks.extend(proc.body._enumerate_recursive(bindings))

        elif self._kind == "pipeline":
            # CSP pipeline - enumerate all processes in order
            processes = self._kwargs.get("processes", [])
            for proc in processes:
                if hasattr(proc, 'body') and proc.body:
                    if hasattr(proc.body, '_enumerate_recursive'):
                        tasks.extend(proc.body._enumerate_recursive(bindings))
                elif hasattr(proc, '_enumerate_recursive'):
                    # Process might be a workload itself
                    tasks.extend(proc._enumerate_recursive(bindings))

        elif self._kind == "select":
            # Sparse iteration over selected indices (MoE routing)
            sparse = self._kwargs.get("sparse")
            body = self._kwargs.get("body")

            if sparse is not None and body:
                # Get indices from sparse object
                if hasattr(sparse, 'indices'):
                    indices = sparse.indices
                elif hasattr(sparse, '__iter__'):
                    indices = list(sparse)
                elif isinstance(sparse, int):
                    indices = range(sparse)
                else:
                    indices = []

                for idx in indices:
                    new_bindings = bindings.copy()
                    new_bindings["_select_idx"] = idx
                    # Body is a lambda mapping index to workload
                    if callable(body):
                        body_workload = body(idx)
                        if hasattr(body_workload, '_enumerate_recursive'):
                            tasks.extend(body_workload._enumerate_recursive(new_bindings))
                    elif hasattr(body, '_enumerate_recursive'):
                        tasks.extend(body._enumerate_recursive(new_bindings))

        elif self._kind == "cond":
            # Conditional workload selection
            predicate = self._kwargs.get("predicate")
            then_workload = self._kwargs.get("then_workload")
            else_workload = self._kwargs.get("else_workload")

            # Evaluate predicate (may be bool or callable)
            if callable(predicate):
                cond_result = predicate()
            else:
                cond_result = bool(predicate)

            # Enumerate the selected branch
            if cond_result:
                if then_workload and hasattr(then_workload, '_enumerate_recursive'):
                    tasks.extend(then_workload._enumerate_recursive(bindings))
            else:
                if else_workload and hasattr(else_workload, '_enumerate_recursive'):
                    tasks.extend(else_workload._enumerate_recursive(bindings))

        # Assign sequential IDs
        for i, task in enumerate(tasks):
            task.id = i

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

    # ========== Workload Metadata API ==========

    def named(self, name: str) -> Workload:
        """Set workload name for IR generation.

        Args:
            name: Name for this workload

        Returns:
            New Workload with name set
        """
        w = self._copy()
        w._name = name
        return w

    def with_params(self, *params: tuple[str, Any]) -> Workload:
        """Set axis parameters for IR generation.

        Args:
            *params: (name, axis) tuples describing workload parameters

        Returns:
            New Workload with params set

        Example:
            workload.with_params(("batch", batch_axis), ("heads", heads_axis))
        """
        w = self._copy()
        w._params = list(params)
        return w

    # ========== Combinator-style Schedule API ==========

    def dispatch(self, policy: Any) -> Workload:
        """Set dispatch policy (Task → AICPU assignment).

        Args:
            policy: DispatchPolicy instance

        Returns:
            New Workload with dispatch policy applied
        """
        w = self._copy()
        w._dispatch_policy = policy
        w._schedule.dispatch = policy
        return w

    def streams(self, count: int) -> Workload:
        """Set number of concurrent streams.

        Args:
            count: Number of streams

        Returns:
            New Workload with stream count set
        """
        w = self._copy()
        w._stream_count = count
        w._schedule.streams = count
        return w

    def stream_by(self, key_fn: Callable[[Task], int]) -> Workload:
        """Set stream assignment function.

        Args:
            key_fn: Function mapping Task to stream index

        Returns:
            New Workload with stream assignment function set
        """
        w = self._copy()
        w._stream_by_fn = key_fn
        w._schedule.stream_by = key_fn
        return w

    def timing(self, policy: Any) -> Workload:
        """Set timing policy (when to issue tasks).

        Args:
            policy: TimingPolicy instance

        Returns:
            New Workload with timing policy applied
        """
        w = self._copy()
        w._timing_policy = policy
        w._schedule.timing = policy
        return w

    def spatial_map(self, grid: tuple[int, int]) -> Workload:
        """Set spatial mapping (workload → tile grid).

        Args:
            grid: (rows, cols) of tile array

        Returns:
            New Workload with spatial mapping set
        """
        w = self._copy()
        w._spatial_grid = grid
        return w

    def layout(self, tensor: Any, *dims: Any) -> Workload:
        """Set data layout for a tensor.

        DEPRECATED (R10): Use TensorLayout as part of Tensor type instead.
        Layout should be a type refinement, not a schedule primitive.

        Example migration:
            # Old (deprecated):
            workload.layout(tensor, Shard(0), Replicate())

            # New (R10):
            from pto_wsp.types import TensorLayout, TensorShard, TensorReplicate
            tensor = Tensor(
                data=..., shape=..., dtype=...,
                layout=TensorLayout((TensorShard(0), TensorReplicate()))
            )

        Args:
            tensor: Tensor to configure layout for
            *dims: Layout specification per dimension (Shard/Replicate)

        Returns:
            New Workload with layout added
        """
        import warnings
        warnings.warn(
            "Workload.layout() is deprecated (R10). "
            "Use TensorLayout as part of Tensor type instead.",
            DeprecationWarning,
            stacklevel=2
        )
        w = self._copy()
        w._layouts.append((tensor, dims))
        return w

    # ========== Task Graph API (R9) ==========

    def task_graph(
        self,
        deps: Any = None,
        window: Any = None,
        pools: Any = None,
        ready: Any = None,
        start: Any = None,
        trace: Any = None,
    ) -> Workload:
        """Use task graph execution instead of streams (R9).

        Task graph mode provides pto-isa-lh compatible execution with:
        - DAG-based dependency tracking (fanin/fanout counters)
        - TensorMap exact-match dependency inference
        - Sliding window task metadata management
        - Heterogeneous execution pools (vector/cube)
        - Optional cycle-level simulation

        This is an alternative to `.streams()` for workloads that need
        general DAG patterns with fanin/fanout, rather than mostly-linear
        per-key pipelines.

        Args:
            deps: Dependency inference mode (Deps.infer_tensor_map_exact() default)
            window: Task metadata window (TaskWindow(8192) default)
            pools: Execution pool routing (Pools.single() default)
            ready: Ready queue policy (ReadyPolicy.fifo() default)
            start: Execution start policy (StartPolicy.after_orchestration() default)
            trace: Tracing/simulation policy (TracePolicy.none() default)

        Returns:
            New Workload with task graph execution configured

        Example:
            # Basic task graph (pto-isa-lh compatible)
            program = (workload
                .dispatch(DispatchPolicy.round_robin(4))
                .task_graph()
                .compile())

            # Full configuration
            program = (workload
                .dispatch(DispatchPolicy.work_steal())
                .task_graph(
                    deps=Deps.infer_tensor_map_exact(),
                    window=TaskWindow(8192, "tasks", WindowMode.STALL),
                    pools=Pools.by_exec_unit(),
                    ready=ReadyPolicy.work_steal(),
                    start=StartPolicy.threshold(100),
                    trace=TracePolicy.cycles()
                )
                .compile())
        """
        from pto_wsp.schedule import (
            TaskGraphConfig, Deps, TaskWindow, Pools,
            ReadyPolicy, StartPolicy, TracePolicy, WindowMode
        )
        w = self._copy()
        w._task_graph_config = TaskGraphConfig(
            deps=deps or Deps.infer_tensor_map_exact(),
            window=window or TaskWindow(8192, "tasks", WindowMode.STALL),
            pools=pools or Pools.single(),
            ready=ready or ReadyPolicy.fifo(),
            start=start or StartPolicy.after_orchestration(),
            trace=trace or TracePolicy.none(),
        )
        # Clear stream-based settings when using task_graph
        w._stream_count = None
        w._stream_by_fn = None
        return w

    def compile(self, target: str = "cpu_sim", **options) -> Any:
        """Compile workload to executable program.

        Args:
            target: Target backend ("cpu_sim", "ascend_npu")
            **options: Additional compilation options

        Returns:
            Program instance ready for execution
        """
        from pto_wsp.program import Program
        return Program(self, target, options)

    @property
    def type_errors(self) -> list:
        """Get type errors detected during workload construction."""
        return self._type_errors

    def has_type_errors(self) -> bool:
        """Check if any type errors were detected."""
        return len(self._type_errors) > 0

    def _copy(self) -> Workload:
        """Create a shallow copy of this workload."""
        w = Workload(self._kind, name=self._name, **self._kwargs)
        # Workload metadata
        w._params = self._params.copy()
        # Schedule (copy the dataclass)
        w._schedule = Schedule(
            dispatch=self._schedule.dispatch,
            streams=self._schedule.streams,
            stream_by=self._schedule.stream_by,
            timing=self._schedule.timing,
        )
        # Legacy attributes (kept for compatibility)
        w._dispatch_policy = self._dispatch_policy
        w._stream_count = self._stream_count
        w._stream_by_fn = self._stream_by_fn
        w._timing_policy = self._timing_policy
        w._spatial_grid = self._spatial_grid
        w._layouts = self._layouts.copy()
        # Task graph config (R9)
        w._task_graph_config = self._task_graph_config
        # Type checking (R3)
        w._type_errors = self._type_errors.copy()
        return w
