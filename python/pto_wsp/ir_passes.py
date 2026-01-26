"""
IR Pass Infrastructure for PTO Workload-Schedule Programming (PTO-WSP) framework.

This module provides a simple pass manager and rewriter infrastructure
for transforming workload IR.

Example:
    from pto_wsp.ir_passes import PassManager, Pass

    # Define a pass
    class SimplifyPass(Pass):
        def run(self, workload):
            # Transform workload
            return workload

    # Run passes
    pm = PassManager()
    pm.add(SimplifyPass())
    result = pm.run(my_workload)
"""

from __future__ import annotations
from typing import Callable, Any, Optional, List
from abc import ABC, abstractmethod
from dataclasses import dataclass

from pto_wsp.workload import Workload


class Pass(ABC):
    """Base class for IR transformation passes.

    Subclasses implement the `run` method to transform workloads.
    """

    def __init__(self, name: str = None):
        self.name = name or self.__class__.__name__

    @abstractmethod
    def run(self, workload: Workload) -> Workload:
        """Transform a workload and return the result.

        Args:
            workload: Input workload to transform

        Returns:
            Transformed workload (may be same object or new)
        """
        pass


class FunctionPass(Pass):
    """Pass defined by a function rather than a class."""

    def __init__(self, func: Callable[[Workload], Workload], name: str = None):
        super().__init__(name or func.__name__)
        self._func = func

    def run(self, workload: Workload) -> Workload:
        return self._func(workload)


class PassManager:
    """Manages and runs a sequence of IR passes.

    Example:
        pm = PassManager()
        pm.add(DeadCodeElimination())
        pm.add(LoopFusion())
        result = pm.run(workload)
    """

    def __init__(self):
        self._passes: List[Pass] = []
        self._stats: dict = {}

    def add(self, p: Pass) -> "PassManager":
        """Add a pass to the pipeline.

        Args:
            p: Pass to add

        Returns:
            self for chaining
        """
        self._passes.append(p)
        return self

    def add_func(self, func: Callable[[Workload], Workload], name: str = None) -> "PassManager":
        """Add a function as a pass.

        Args:
            func: Function to add as a pass
            name: Optional name for the pass

        Returns:
            self for chaining
        """
        self._passes.append(FunctionPass(func, name))
        return self

    def run(self, workload: Workload, verbose: bool = False) -> Workload:
        """Run all passes on a workload.

        Args:
            workload: Input workload
            verbose: Print pass names as they run

        Returns:
            Transformed workload
        """
        result = workload
        for p in self._passes:
            if verbose:
                print(f"Running pass: {p.name}")
            result = p.run(result)
            self._stats[p.name] = self._stats.get(p.name, 0) + 1
        return result

    def clear(self):
        """Remove all passes."""
        self._passes.clear()

    @property
    def stats(self) -> dict:
        """Return pass execution statistics."""
        return self._stats.copy()


# ============================================================================
# Built-in Passes
# ============================================================================

class IdentityPass(Pass):
    """Pass that returns the workload unchanged."""

    def run(self, workload: Workload) -> Workload:
        return workload


class PrintPass(Pass):
    """Pass that prints workload info and returns it unchanged."""

    def __init__(self, label: str = ""):
        super().__init__(f"PrintPass({label})" if label else "PrintPass")
        self._label = label

    def run(self, workload: Workload) -> Workload:
        prefix = f"[{self._label}] " if self._label else ""
        print(f"{prefix}Workload: kind={workload._kind}, name={workload._name}")
        return workload


class FlattenCombinePass(Pass):
    """Flatten nested combine nodes.

    Transforms:
        combine(combine(a, b), c) -> combine(a, b, c)
    """

    def run(self, workload: Workload) -> Workload:
        if workload._kind != "combine":
            return workload

        workloads = workload._kwargs.get("workloads", [])
        flattened = []

        for w in workloads:
            if isinstance(w, Workload) and w._kind == "combine":
                # Recursively flatten nested combines
                inner = self.run(w)
                inner_workloads = inner._kwargs.get("workloads", [])
                flattened.extend(inner_workloads)
            else:
                flattened.append(w)

        return Workload("combine", workloads=flattened)


class DeadTaskElimination(Pass):
    """Remove tasks with no outputs.

    This is a placeholder - actual implementation would need
    to analyze data dependencies.
    """

    def run(self, workload: Workload) -> Workload:
        # Placeholder: actual implementation would analyze dependencies
        return workload


class LoopInterchange(Pass):
    """Interchange nested parallel loops.

    Transforms:
        for a in P(A):
            for b in P(B):
                task(a, b)

    To:
        for b in P(B):
            for a in P(A):
                task(a, b)

    This is a placeholder - actual implementation would check
    legality of interchange.
    """

    def run(self, workload: Workload) -> Workload:
        # Placeholder: actual implementation would analyze dependencies
        return workload


# ============================================================================
# Rewriter Infrastructure
# ============================================================================

@dataclass
class RewritePattern:
    """Pattern for matching and rewriting workload nodes."""
    name: str
    match: Callable[[Workload], bool]
    rewrite: Callable[[Workload], Workload]

    def apply(self, workload: Workload) -> Optional[Workload]:
        """Apply pattern if it matches.

        Returns:
            Rewritten workload if matched, None otherwise
        """
        if self.match(workload):
            return self.rewrite(workload)
        return None


class PatternRewriter(Pass):
    """Pass that applies rewrite patterns until fixpoint.

    Example:
        rewriter = PatternRewriter()
        rewriter.add_pattern(
            "flatten_combine",
            match=lambda w: w._kind == "combine" and any(
                isinstance(c, Workload) and c._kind == "combine"
                for c in w._kwargs.get("workloads", [])
            ),
            rewrite=lambda w: flatten_combine(w)
        )
        result = rewriter.run(workload)
    """

    def __init__(self, max_iterations: int = 100):
        super().__init__("PatternRewriter")
        self._patterns: List[RewritePattern] = []
        self._max_iterations = max_iterations

    def add_pattern(
        self,
        name: str,
        match: Callable[[Workload], bool],
        rewrite: Callable[[Workload], Workload]
    ) -> "PatternRewriter":
        """Add a rewrite pattern.

        Args:
            name: Pattern name for debugging
            match: Function returning True if pattern applies
            rewrite: Function to rewrite matched workload

        Returns:
            self for chaining
        """
        self._patterns.append(RewritePattern(name, match, rewrite))
        return self

    def run(self, workload: Workload) -> Workload:
        """Apply patterns until no more changes."""
        result = workload
        for _ in range(self._max_iterations):
            changed = False
            for pattern in self._patterns:
                new_result = pattern.apply(result)
                if new_result is not None:
                    result = new_result
                    changed = True
                    break
            if not changed:
                break
        return result


# ============================================================================
# Utility Functions
# ============================================================================

def pass_func(func: Callable[[Workload], Workload]) -> Pass:
    """Decorator to create a pass from a function.

    Example:
        @pass_func
        def my_pass(workload):
            return workload
    """
    return FunctionPass(func)
