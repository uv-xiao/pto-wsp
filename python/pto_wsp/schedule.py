"""
Schedule policies for PTO Workload-Schedule Programming (PTO-WSP) framework.

This module provides schedule primitives for controlling:
- Task dispatch (which executor handles each task)
- Issue policy (task ordering within executors)
- Timing policy (when tasks are issued)
- Extended primitives: dispatch_threshold, pipeline_depth, task_window, batch_deps
"""

from __future__ import annotations
from typing import Callable, Any, Dict, List
from enum import Enum


class WindowMode(Enum):
    """Task window overflow policy."""
    STALL = "stall"       # Block until window has space
    ABORT = "abort"       # Fail immediately on overflow
    BENCHMARK = "benchmark"  # Record overflow events for profiling


class GateScope(Enum):
    """Pipeline depth scope."""
    GLOBAL = "global"         # Single global depth limit
    PER_STREAM = "per_stream" # Per-stream depth limits
    PER_POOL = "per_pool"     # Per-pool depth limits (vector/cube)


class DispatchPolicy:
    """Dispatch policy: determines which AICPU handles each task."""

    def __init__(self, kind: str, **kwargs):
        self._kind = kind
        self._kwargs = kwargs

    @staticmethod
    def round_robin(num_aicpus: int) -> DispatchPolicy:
        """Round-robin across AICPUs."""
        return DispatchPolicy("round_robin", num_aicpus=num_aicpus)

    @staticmethod
    def affinity(key_fn: Callable[[Any], Any]) -> DispatchPolicy:
        """Same key value → same AICPU."""
        return DispatchPolicy("affinity", key_fn=key_fn)

    @staticmethod
    def hash(key_fn: Callable[[Any], Any]) -> DispatchPolicy:
        """Hash-based assignment."""
        return DispatchPolicy("hash", key_fn=key_fn)

    @staticmethod
    def work_steal() -> DispatchPolicy:
        """Dynamic load balancing with work stealing."""
        return DispatchPolicy("work_steal")

    @staticmethod
    def dispatch_by(fn: Callable[[Any], int]) -> DispatchPolicy:
        """Custom dispatch function."""
        return DispatchPolicy("dispatch_by", fn=fn)


class IssuePolicy:
    """Issue policy: determines task ordering within each AICPU."""

    def __init__(self, kind: str, **kwargs):
        self._kind = kind
        self._kwargs = kwargs

    @staticmethod
    def stream_by(key_fn: Callable[[Any], int]) -> IssuePolicy:
        """Stream assignment by key function."""
        return IssuePolicy("stream_by", key_fn=key_fn)

    @staticmethod
    def single_stream() -> IssuePolicy:
        """All tasks in single stream."""
        return IssuePolicy("single_stream")

    @staticmethod
    def per_axis(axis: Any) -> IssuePolicy:
        """Each axis value → separate stream."""
        return IssuePolicy("per_axis", axis=axis)

    @staticmethod
    def fifo() -> IssuePolicy:
        """First-in-first-out ordering."""
        return IssuePolicy("fifo")

    @staticmethod
    def priority(priority_fn: Callable[[Any], int]) -> IssuePolicy:
        """Priority-based ordering."""
        return IssuePolicy("priority", priority_fn=priority_fn)


class TimingPolicy:
    """Timing policy: controls when tasks are issued."""

    def __init__(self, kind: str, **kwargs):
        self._kind = kind
        self._kwargs = kwargs

    @staticmethod
    @property
    def immediate() -> TimingPolicy:
        """Issue as soon as dependencies satisfied."""
        return TimingPolicy("immediate")

    @staticmethod
    def batched(n: int) -> TimingPolicy:
        """Batch N tasks before issuing."""
        return TimingPolicy("batched", n=n)

    @staticmethod
    def interleaved(streams: int) -> TimingPolicy:
        """Round-robin across streams."""
        return TimingPolicy("interleaved", streams=streams)

    @staticmethod
    def rate_limit(tasks_per_ms: int) -> TimingPolicy:
        """Rate limiting."""
        return TimingPolicy("rate_limit", tasks_per_ms=tasks_per_ms)


# For convenience, expose immediate as a constant
TimingPolicy.immediate = TimingPolicy("immediate")


# ============================================================
# Extended Schedule Primitives (R5)
# ============================================================

class DispatchThreshold:
    """Multi-level dispatch based on workload size thresholds.

    Different dispatch policies are applied based on thresholds:
    - Small workloads: single executor (threshold[0])
    - Medium workloads: few executors with affinity
    - Large workloads: many executors with work-stealing

    Example:
        DispatchThreshold(
            thresholds=[256, 1024, 4096],
            policies={
                256: DispatchPolicy.round_robin(1),
                1024: DispatchPolicy.affinity(lambda t: t.batch),
                4096: DispatchPolicy.work_steal(),
            }
        )
    """

    def __init__(self, thresholds: List[int], policies: Dict[int, DispatchPolicy]):
        self.thresholds = sorted(thresholds)
        self.policies = policies

    def select_policy(self, workload_size: int) -> DispatchPolicy:
        """Select the appropriate policy based on workload size."""
        for threshold in reversed(self.thresholds):
            if workload_size >= threshold:
                return self.policies.get(threshold)
        return self.policies.get(self.thresholds[0]) if self.thresholds else None


class PipelineDepth:
    """In-flight task control (double/triple buffering).

    Controls maximum number of tasks in-flight per scope:
    - GLOBAL: single limit across all streams
    - PER_STREAM: each stream has its own limit
    - PER_POOL: separate limits for vector/cube pools

    Example:
        PipelineDepth(depth=2, scope=GateScope.PER_STREAM)
    """

    def __init__(self, depth: int, scope: GateScope = GateScope.GLOBAL):
        self.depth = depth
        self.scope = scope


class TaskWindow:
    """Metadata window management for task tracking.

    Controls memory allocation for task metadata:
    - size: maximum window size (tasks, bytes, or entries)
    - unit: "tasks", "bytes", or "entries"
    - mode: overflow behavior (STALL, ABORT, BENCHMARK)

    Example:
        TaskWindow(size=8192, unit="tasks", mode=WindowMode.STALL)
    """

    def __init__(self, size: int, unit: str = "tasks",
                 mode: WindowMode = WindowMode.STALL):
        if unit not in ("tasks", "bytes", "entries"):
            raise ValueError(f"Invalid unit: {unit}")
        self.size = size
        self.unit = unit
        self.mode = mode


class BatchDeps:
    """Batched dependency resolution.

    Defers dependency resolution until threshold tasks accumulated:
    - Reduces per-task resolution overhead
    - Optional range compression for consecutive task IDs

    Example:
        BatchDeps(threshold=64, range_compression=True)
    """

    def __init__(self, threshold: int, range_compression: bool = False):
        self.threshold = threshold
        self.range_compression = range_compression


# ============================================================
# Task Graph Primitives (R9)
# ============================================================

class DepsMode(Enum):
    """Dependency inference mode for task graphs."""
    INFER_TENSOR_MAP_EXACT = "infer_tensor_map_exact"  # pto-isa-lh compatible
    INFER_BYTES_OVERLAP = "infer_bytes_overlap"        # Extended overlap detection
    EXPLICIT = "explicit"                              # Only explicit deps
    HYBRID = "hybrid"                                  # Union of inferred + explicit


class Deps:
    """Dependency inference configuration for task graphs.

    Controls how dependencies between tasks are discovered:
    - infer_tensor_map_exact(): pto-isa-lh compatible exact region matching
    - infer_bytes_overlap(): Extended overlap detection (v9 extension)
    - explicit(): Only user-specified explicit dependencies
    - hybrid(): Union of inferred and explicit dependencies

    Example:
        Deps.infer_tensor_map_exact()  # Default, pto-isa-lh compatible
        Deps.hybrid(infer=DepsMode.INFER_TENSOR_MAP_EXACT, explicit=True)
    """

    def __init__(self, mode: DepsMode, **kwargs):
        self.mode = mode
        self._kwargs = kwargs

    @staticmethod
    def infer_tensor_map_exact() -> "Deps":
        """TensorMap exact-match lookup (pto-isa-lh compatible).

        Dependencies are inferred from tensor region access:
        - Key = (buffer_id, offsets, extents) with extents in key
        - Matches only when all fields are exactly equal
        - No overlap detection (same as pto-isa-lh limitation)
        """
        return Deps(DepsMode.INFER_TENSOR_MAP_EXACT)

    @staticmethod
    def infer_bytes_overlap() -> "Deps":
        """Extended overlap detection for dependencies.

        V9 extension that detects overlapping memory regions
        even when shapes differ. Addresses false negatives from
        pto-isa-lh's exact-match approach.
        """
        return Deps(DepsMode.INFER_BYTES_OVERLAP)

    @staticmethod
    def explicit() -> "Deps":
        """Only explicit dependencies.

        No inference; all dependencies must be specified via
        structural workload primitives or explicit `after=[]`.
        """
        return Deps(DepsMode.EXPLICIT)

    @staticmethod
    def hybrid(infer: DepsMode = DepsMode.INFER_TENSOR_MAP_EXACT,
               explicit: bool = True) -> "Deps":
        """Hybrid inference + explicit dependencies.

        Union of:
        - Structural/workload dependencies
        - TensorMap inferred RAW dependencies
        - User-specified explicit edges

        Args:
            infer: Inference mode for tensor-based deps
            explicit: Whether to include explicit deps

        Example:
            Deps.hybrid()  # Default hybrid mode
        """
        return Deps(DepsMode.HYBRID, infer=infer, explicit=explicit)


class ReadyPolicy:
    """Ready queue policy for task graph execution.

    Controls how ready tasks (fanin==0) are selected for execution.
    """

    def __init__(self, kind: str, **kwargs):
        self._kind = kind
        self._kwargs = kwargs

    @staticmethod
    def fifo() -> "ReadyPolicy":
        """First-in-first-out ordering."""
        return ReadyPolicy("fifo")

    @staticmethod
    def work_steal() -> "ReadyPolicy":
        """Dynamic load balancing with work stealing."""
        return ReadyPolicy("work_steal")

    @staticmethod
    def priority(priority_fn: Callable[[Any], int]) -> "ReadyPolicy":
        """Priority-based ordering.

        Args:
            priority_fn: Function mapping task to priority (lower = higher priority)
        """
        return ReadyPolicy("priority", priority_fn=priority_fn)


class StartPolicy:
    """Execution start policy for task graph.

    Controls when workers start executing tasks relative to orchestration.
    """

    def __init__(self, kind: str, **kwargs):
        self._kind = kind
        self._kwargs = kwargs

    @staticmethod
    def threshold(n: int) -> "StartPolicy":
        """Start execution after N tasks are ready.

        Enables pipelined build+execute where workers start before
        orchestration completes. Must handle producer-complete races.

        Args:
            n: Number of ready tasks before execution starts
        """
        return StartPolicy("threshold", n=n)

    @staticmethod
    def after_orchestration() -> "StartPolicy":
        """Start execution only after all tasks are submitted."""
        return StartPolicy("after_orchestration")


class TracePolicy:
    """Tracing policy for task graph simulation.

    Controls cycle-level simulation and trace recording.
    """

    def __init__(self, kind: str, **kwargs):
        self._kind = kind
        self._kwargs = kwargs

    @staticmethod
    def none() -> "TracePolicy":
        """No tracing or simulation."""
        return TracePolicy("none")

    @staticmethod
    def cycles(cost_fn: Callable[[Any], int] = None) -> "TracePolicy":
        """Enable cycle-level simulation.

        Records earliest_start_cycle and end_cycle for each task.
        Completion propagates timing to dependents.

        Args:
            cost_fn: Optional function mapping task to cycle cost
        """
        return TracePolicy("cycles", cost_fn=cost_fn)


class Pools:
    """Execution pool configuration for task graphs.

    Controls task routing to heterogeneous execution units.
    Generalizes pto-isa-lh's dual-queue (vector/cube) to N pools.
    """

    def __init__(self, kind: str, **kwargs):
        self._kind = kind
        self._kwargs = kwargs

    @staticmethod
    def single() -> "Pools":
        """Single unified ready queue."""
        return Pools("single")

    @staticmethod
    def by_exec_unit() -> "Pools":
        """Route tasks by execution unit (vector/cube).

        Reproduces A2A3's dual-queue semantics where vector
        and cube operations have separate ready queues.
        """
        return Pools("by_exec_unit")

    @staticmethod
    def custom(pool_fn: Callable[[Any], str]) -> "Pools":
        """Custom pool routing.

        Args:
            pool_fn: Function mapping task to pool name
        """
        return Pools("custom", pool_fn=pool_fn)


class TaskGraphConfig:
    """Configuration for task graph execution mode.

    Encapsulates all task graph settings as an alternative to streams.

    Example:
        TaskGraphConfig(
            deps=Deps.infer_tensor_map_exact(),
            window=TaskWindow(8192, "tasks", WindowMode.STALL),
            pools=Pools.by_exec_unit(),
            ready=ReadyPolicy.work_steal(),
            start=StartPolicy.threshold(100),
            trace=TracePolicy.none()
        )
    """

    def __init__(
        self,
        deps: Deps = None,
        window: TaskWindow = None,
        pools: Pools = None,
        ready: ReadyPolicy = None,
        start: StartPolicy = None,
        trace: TracePolicy = None,
    ):
        self.deps = deps or Deps.infer_tensor_map_exact()
        self.window = window or TaskWindow(8192, "tasks", WindowMode.STALL)
        self.pools = pools or Pools.single()
        self.ready = ready or ReadyPolicy.fifo()
        self.start = start or StartPolicy.after_orchestration()
        self.trace = trace or TracePolicy.none()
