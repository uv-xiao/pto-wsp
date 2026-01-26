"""
Automatic Scheduling Exploration for PTO Workload-Schedule Programming (PTO-WSP) framework.

This module provides tools for automatically exploring scheduling configurations
to find optimal execution parameters for a given workload.

Example:
    from pto_wsp import workload, P, Dense
    from pto_wsp.auto_schedule import AutoScheduler, SearchSpace

    @workload
    def my_workload():
        ...

    # Define search space
    space = SearchSpace()
    space.add_streams([1, 2, 4, 8])
    space.add_dispatch(["round_robin", "work_steal"])

    # Run exploration
    scheduler = AutoScheduler(my_workload())
    best_config, results = scheduler.explore(space, trials=10)

    print(f"Best config: {best_config}")
    print(f"Speedup: {results.speedup:.2f}x")
"""

from __future__ import annotations
from typing import Any, Callable, Optional, List, Dict, Tuple
from dataclasses import dataclass, field
from enum import Enum, auto
import time
import statistics


class DispatchOption(Enum):
    """Available dispatch policies for exploration."""
    ROUND_ROBIN = auto()
    WORK_STEAL = auto()
    AFFINITY = auto()


class TimingOption(Enum):
    """Available timing policies for exploration."""
    IMMEDIATE = auto()
    BATCHED = auto()


@dataclass
class ScheduleConfig:
    """A specific scheduling configuration.

    Attributes:
        streams: Number of concurrent streams
        dispatch: Dispatch policy type
        dispatch_param: Parameter for dispatch policy (e.g., executor count)
        timing: Timing policy type
        timing_param: Parameter for timing policy (e.g., batch size)
    """
    streams: int = 1
    dispatch: str = "round_robin"
    dispatch_param: int = 4
    timing: str = "immediate"
    timing_param: Optional[int] = None

    def __hash__(self):
        return hash((self.streams, self.dispatch, self.dispatch_param,
                    self.timing, self.timing_param))

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "streams": self.streams,
            "dispatch": self.dispatch,
            "dispatch_param": self.dispatch_param,
            "timing": self.timing,
            "timing_param": self.timing_param,
        }


@dataclass
class ExplorationResult:
    """Result of a single scheduling exploration trial.

    Attributes:
        config: The schedule configuration tested
        execute_time_ms: Execution time in milliseconds
        compile_time_ms: Compilation time in milliseconds
        task_count: Number of tasks executed
        error: Error message if execution failed
    """
    config: ScheduleConfig
    execute_time_ms: float = 0.0
    compile_time_ms: float = 0.0
    task_count: int = 0
    error: Optional[str] = None

    @property
    def total_time_ms(self) -> float:
        """Total time including compile and execute."""
        return self.compile_time_ms + self.execute_time_ms

    @property
    def success(self) -> bool:
        """Whether the trial succeeded."""
        return self.error is None


@dataclass
class ExplorationSummary:
    """Summary of scheduling exploration results.

    Attributes:
        results: All trial results
        best_config: Best performing configuration
        best_time_ms: Best execution time
        baseline_time_ms: Baseline (default config) time
        speedup: Speedup over baseline
    """
    results: List[ExplorationResult] = field(default_factory=list)
    best_config: Optional[ScheduleConfig] = None
    best_time_ms: float = float('inf')
    baseline_time_ms: float = 0.0
    speedup: float = 1.0

    def print_summary(self) -> None:
        """Print exploration summary."""
        print(f"Scheduling Exploration Summary")
        print(f"=" * 50)
        print(f"Configurations tested: {len(self.results)}")
        print(f"Successful trials: {sum(1 for r in self.results if r.success)}")
        print(f"Baseline time: {self.baseline_time_ms:.3f} ms")
        print(f"Best time: {self.best_time_ms:.3f} ms")
        print(f"Speedup: {self.speedup:.2f}x")
        print()
        if self.best_config:
            print(f"Best configuration:")
            print(f"  Streams: {self.best_config.streams}")
            print(f"  Dispatch: {self.best_config.dispatch}({self.best_config.dispatch_param})")
            print(f"  Timing: {self.best_config.timing}")

    def top_configs(self, n: int = 5) -> List[Tuple[ScheduleConfig, float]]:
        """Get top N configurations by execution time."""
        successful = [r for r in self.results if r.success]
        sorted_results = sorted(successful, key=lambda r: r.execute_time_ms)
        return [(r.config, r.execute_time_ms) for r in sorted_results[:n]]


class SearchSpace:
    """Defines the search space for scheduling exploration.

    Example:
        space = SearchSpace()
        space.add_streams([1, 2, 4, 8])
        space.add_dispatch(["round_robin", "work_steal"])
        space.add_timing(["immediate", "batched"])
    """

    def __init__(self):
        self._stream_options: List[int] = [1]
        self._dispatch_options: List[Tuple[str, int]] = [("round_robin", 4)]
        self._timing_options: List[Tuple[str, Optional[int]]] = [("immediate", None)]

    def add_streams(self, values: List[int]) -> "SearchSpace":
        """Add stream count options.

        Args:
            values: List of stream counts to try

        Returns:
            self for chaining
        """
        self._stream_options = values
        return self

    def add_dispatch(
        self,
        policies: List[str],
        params: Optional[List[int]] = None
    ) -> "SearchSpace":
        """Add dispatch policy options.

        Args:
            policies: List of policy names ("round_robin", "work_steal", "affinity")
            params: List of parameters (defaults to [4] for all)

        Returns:
            self for chaining
        """
        if params is None:
            params = [4]

        self._dispatch_options = []
        for policy in policies:
            for param in params:
                self._dispatch_options.append((policy, param))
        return self

    def add_timing(
        self,
        policies: List[str],
        params: Optional[List[Optional[int]]] = None
    ) -> "SearchSpace":
        """Add timing policy options.

        Args:
            policies: List of policy names ("immediate", "batched")
            params: List of parameters (batch sizes)

        Returns:
            self for chaining
        """
        if params is None:
            params = [None]

        self._timing_options = []
        for policy in policies:
            for param in params:
                if policy == "immediate":
                    self._timing_options.append(("immediate", None))
                else:
                    self._timing_options.append((policy, param))
        return self

    def enumerate_configs(self) -> List[ScheduleConfig]:
        """Enumerate all configurations in the search space.

        Returns:
            List of ScheduleConfig objects
        """
        configs = []
        for streams in self._stream_options:
            for dispatch, dispatch_param in self._dispatch_options:
                for timing, timing_param in self._timing_options:
                    configs.append(ScheduleConfig(
                        streams=streams,
                        dispatch=dispatch,
                        dispatch_param=dispatch_param,
                        timing=timing,
                        timing_param=timing_param,
                    ))
        return configs

    @property
    def size(self) -> int:
        """Total number of configurations in the space."""
        return (len(self._stream_options) *
                len(self._dispatch_options) *
                len(self._timing_options))


class AutoScheduler:
    """Automatic scheduling exploration for workloads.

    Explores different scheduling configurations and finds the best one
    based on execution time.

    Example:
        scheduler = AutoScheduler(my_workload())

        # Simple exploration with defaults
        best_config = scheduler.find_best()

        # Custom search space
        space = SearchSpace()
        space.add_streams([1, 2, 4, 8])
        space.add_dispatch(["round_robin", "work_steal"])

        best_config, summary = scheduler.explore(space, trials=5)
        summary.print_summary()
    """

    def __init__(self, workload: Any):
        """Initialize auto scheduler.

        Args:
            workload: Source Workload instance
        """
        self.workload = workload
        self._kernel_impls: Dict[str, Callable] = {}

    def register_kernel(self, name: str, impl: Callable) -> "AutoScheduler":
        """Register kernel implementation for CPU simulation.

        Args:
            name: Kernel name
            impl: Python callable

        Returns:
            self for chaining
        """
        self._kernel_impls[name] = impl
        return self

    def _apply_config(self, workload: Any, config: ScheduleConfig) -> Any:
        """Apply a schedule configuration to a workload.

        Args:
            workload: Source workload
            config: Schedule configuration

        Returns:
            Configured workload
        """
        from pto_wsp import DispatchPolicy, TimingPolicy

        # Start with base workload (clone if possible)
        w = workload

        # Apply dispatch policy
        if config.dispatch == "round_robin":
            w = w.dispatch(DispatchPolicy.round_robin(config.dispatch_param))
        elif config.dispatch == "work_steal":
            w = w.dispatch(DispatchPolicy.work_steal())
        elif config.dispatch == "affinity":
            # Default affinity by first loop variable
            w = w.dispatch(DispatchPolicy.affinity(lambda t: t.get("batch", 0)))

        # Apply streams
        w = w.streams(config.streams)

        # Apply timing policy
        if config.timing == "immediate":
            w = w.timing(TimingPolicy.immediate)
        elif config.timing == "batched" and config.timing_param:
            w = w.timing(TimingPolicy.batched(config.timing_param))

        return w

    def _run_trial(self, config: ScheduleConfig, warmup: int = 1) -> ExplorationResult:
        """Run a single trial with a configuration.

        Args:
            config: Schedule configuration to test
            warmup: Number of warmup runs

        Returns:
            ExplorationResult with timing data
        """
        try:
            # Apply configuration
            configured = self._apply_config(self.workload, config)

            # Compile
            compile_start = time.perf_counter()
            program = configured.compile()
            compile_time = (time.perf_counter() - compile_start) * 1000

            # Register kernels
            for name, impl in self._kernel_impls.items():
                program.register_kernel(name, impl)

            # Warmup runs
            for _ in range(warmup):
                program._started = False  # Reset for re-run
                program._complete.clear()
                program.execute()
                program.synchronize()

            # Timed run
            program._started = False
            program._complete.clear()
            exec_start = time.perf_counter()
            program.execute()
            program.synchronize()
            exec_time = (time.perf_counter() - exec_start) * 1000

            return ExplorationResult(
                config=config,
                execute_time_ms=exec_time,
                compile_time_ms=compile_time,
                task_count=program.stats.task_count,
            )

        except Exception as e:
            return ExplorationResult(
                config=config,
                error=str(e),
            )

    def explore(
        self,
        space: SearchSpace,
        trials: int = 3,
        warmup: int = 1,
        verbose: bool = False,
    ) -> Tuple[ScheduleConfig, ExplorationSummary]:
        """Explore scheduling configurations.

        Args:
            space: Search space to explore
            trials: Number of trials per configuration
            warmup: Warmup runs before timing
            verbose: Print progress

        Returns:
            Tuple of (best_config, summary)
        """
        configs = space.enumerate_configs()

        if verbose:
            print(f"Exploring {len(configs)} configurations ({trials} trials each)")

        # Get baseline (default config)
        baseline_config = ScheduleConfig()
        baseline_times = []
        for _ in range(trials):
            result = self._run_trial(baseline_config, warmup)
            if result.success:
                baseline_times.append(result.execute_time_ms)
        baseline_time = statistics.mean(baseline_times) if baseline_times else float('inf')

        # Explore all configurations
        results: List[ExplorationResult] = []
        best_time = float('inf')
        best_config = baseline_config

        for i, config in enumerate(configs):
            if verbose:
                print(f"  [{i+1}/{len(configs)}] Testing {config.dispatch}({config.dispatch_param}) "
                      f"streams={config.streams} timing={config.timing}")

            times = []
            for t in range(trials):
                result = self._run_trial(config, warmup)
                results.append(result)

                if result.success:
                    times.append(result.execute_time_ms)

                    if verbose:
                        print(f"    Trial {t+1}: {result.execute_time_ms:.3f}ms")

            if times:
                avg_time = statistics.mean(times)
                if avg_time < best_time:
                    best_time = avg_time
                    best_config = config

                if verbose:
                    print(f"    Average: {avg_time:.3f}ms")

        # Compute speedup
        speedup = baseline_time / best_time if best_time > 0 else 1.0

        summary = ExplorationSummary(
            results=results,
            best_config=best_config,
            best_time_ms=best_time,
            baseline_time_ms=baseline_time,
            speedup=speedup,
        )

        return best_config, summary

    def find_best(
        self,
        max_streams: int = 8,
        trials: int = 3,
    ) -> ScheduleConfig:
        """Quick exploration with default search space.

        Args:
            max_streams: Maximum stream count to try
            trials: Trials per configuration

        Returns:
            Best schedule configuration
        """
        space = SearchSpace()
        space.add_streams([1, 2, 4, min(8, max_streams)])
        space.add_dispatch(["round_robin", "work_steal"], [4])
        space.add_timing(["immediate"])

        config, _ = self.explore(space, trials=trials, warmup=1)
        return config


def auto_schedule(workload: Any, verbose: bool = False) -> Any:
    """Convenience function for automatic scheduling.

    Args:
        workload: Source workload
        verbose: Print exploration progress

    Returns:
        Workload with best schedule applied

    Example:
        scheduled = auto_schedule(my_workload())
        program = scheduled.compile()
    """
    scheduler = AutoScheduler(workload)
    best_config = scheduler.find_best()

    if verbose:
        print(f"Auto-schedule selected: streams={best_config.streams}, "
              f"dispatch={best_config.dispatch}")

    return scheduler._apply_config(workload, best_config)
