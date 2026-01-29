"""
P namespace for symbolic loops in PTO Workload-Schedule Programming (PTO-WSP) framework.

The P namespace provides loop constructors with explicit parallelism semantics,
inspired by TileLang's T namespace.

Example:
    @workload
    def attention(batch, heads):
        for b, h in P(batch, heads):       # Parallel grid
            attn[b, h](Q[b,h], K[b], V[b], O[b,h])

    @workload
    def scan(seq_len):
        for i in P.seq(seq_len):           # Sequential
            scan_step[i](input[i], output[i])
"""

from __future__ import annotations
from typing import Any, Iterator, Union, Optional
from contextlib import contextmanager
from dataclasses import dataclass

from pto_wsp.builder import (
    get_current_builder,
    LoopFrame,
    ConditionalFrame,
)


@dataclass
class LoopVar:
    """Symbolic loop variable.

    Represents an iteration variable that can be used for indexing
    and passed to kernel calls.
    """
    name: str
    axis: Any
    index: int = 0  # Current index during iteration

    def __hash__(self):
        return hash(self.name)

    def __repr__(self):
        return f"LoopVar({self.name})"


class ParallelGridIterator:
    """Iterator for P() parallel grid.

    Yields tuples of loop variables for multi-dimensional grids.
    """

    def __init__(self, axes: tuple[Any, ...], var_names: list[str]):
        self.axes = axes
        self.var_names = var_names
        self._started = False
        self._loop_vars: list[LoopVar] = []
        self._frame: Optional[LoopFrame] = None

    def __iter__(self) -> Iterator[Union[LoopVar, tuple[LoopVar, ...]]]:
        builder = get_current_builder()
        if builder is None:
            raise RuntimeError("P() used outside @workload context")

        # Create loop variables
        self._loop_vars = [
            LoopVar(name, axis)
            for name, axis in zip(self.var_names, self.axes)
        ]

        # Create and push frame for this parallel loop
        self._frame = LoopFrame(
            kind="parallel",
            axis=self.axes,
            var_name=",".join(self.var_names),
        )
        builder.push_frame(self._frame)

        # Yield the loop variables (single iteration in builder mode)
        if len(self._loop_vars) == 1:
            yield self._loop_vars[0]
        else:
            yield tuple(self._loop_vars)

        # Pop frame and add completed node to parent
        builder.pop_frame()

        # Build workload node from frame
        from pto_wsp.workload import Workload

        # Create nested parallel_for for each axis
        body = self._build_body(self._frame.children)
        workload = body

        # Wrap in nested parallel_for for each axis (from innermost to outermost)
        # Note: body is stored directly (not as lambda) since P namespace already knows
        # all iterations and the body is the same for all
        for axis, var in zip(reversed(self.axes), reversed(self._loop_vars)):
            # Store body directly for enumeration (not as callable)
            workload = Workload("parallel_for", axis=axis, body=workload, var_name=var.name)

        builder.add_child(workload)

    def _build_body(self, children: list) -> Any:
        """Build body workload from children."""
        from pto_wsp.workload import Workload

        if len(children) == 0:
            return Workload("empty")
        elif len(children) == 1:
            return children[0]
        else:
            return Workload("combine", workloads=children)


class SequentialIterator:
    """Iterator for P.seq() sequential iteration."""

    def __init__(self, axis: Any, var_name: str):
        self.axis = axis
        self.var_name = var_name
        self._frame: Optional[LoopFrame] = None

    def __iter__(self) -> Iterator[LoopVar]:
        builder = get_current_builder()
        if builder is None:
            raise RuntimeError("P.seq() used outside @workload context")

        loop_var = LoopVar(self.var_name, self.axis)

        # Create and push frame
        self._frame = LoopFrame(
            kind="sequential",
            axis=self.axis,
            var_name=self.var_name,
        )
        builder.push_frame(self._frame)

        yield loop_var

        # Pop frame and build node
        builder.pop_frame()

        from pto_wsp.workload import Workload

        body = self._build_body(self._frame.children)
        # Store body directly (not as callable) for enumeration
        workload = Workload("for_each", axis=self.axis, body=body, var_name=self.var_name)
        builder.add_child(workload)

    def _build_body(self, children: list) -> Any:
        from pto_wsp.workload import Workload

        if len(children) == 0:
            return Workload("empty")
        elif len(children) == 1:
            return children[0]
        else:
            return Workload("sequential", workloads=children)


class SelectIterator:
    """Iterator for P.sel() sparse iteration."""

    def __init__(self, sparse: Any, var_name: str):
        self.sparse = sparse
        self.var_name = var_name
        self._frame: Optional[LoopFrame] = None

    def __iter__(self) -> Iterator[LoopVar]:
        builder = get_current_builder()
        if builder is None:
            raise RuntimeError("P.sel() used outside @workload context")

        loop_var = LoopVar(self.var_name, self.sparse)

        self._frame = LoopFrame(
            kind="select",
            axis=self.sparse,
            var_name=self.var_name,
        )
        builder.push_frame(self._frame)

        yield loop_var

        builder.pop_frame()

        from pto_wsp.workload import Workload

        body = self._build_body(self._frame.children)
        # Store body directly (not as callable) for enumeration
        workload = Workload("select", sparse=self.sparse, body=body, var_name=self.var_name)
        builder.add_child(workload)

    def _build_body(self, children: list) -> Any:
        from pto_wsp.workload import Workload

        if len(children) == 0:
            return Workload("empty")
        elif len(children) == 1:
            return children[0]
        else:
            return Workload("combine", workloads=children)


class PNamespace:
    """P namespace for loop constructors.

    Provides explicit parallelism semantics through different loop constructors:
    - P(axes...) or P.grid(axes...) - Parallel grid
    - P.seq(axis) - Sequential iteration
    - P.sel(sparse) - Sparse selection
    - P.pipe() - Pipeline scope
    - P.when(pred) - Conditional branch
    """

    def __call__(self, *axes, **kwargs) -> ParallelGridIterator:
        """Create parallel grid iterator.

        Equivalent to P.grid(*axes).

        Example:
            for b, h in P(batch, heads):
                ...
        """
        return self.grid(*axes, **kwargs)

    def grid(self, *axes, names: Optional[list[str]] = None) -> ParallelGridIterator:
        """Parallel grid iteration - all iterations can run in parallel.

        Type: Independent

        Example:
            for b, h in P.grid(batch, heads):
                attn[b, h](Q[b,h], K[b], V[b], O[b,h])
        """
        builder = get_current_builder()
        if builder is None:
            raise RuntimeError("P.grid() used outside @workload context")

        if names is None:
            # Generate default names using builder's counter for uniqueness
            default_names = ["b", "h", "s", "d", "i", "j", "k", "l", "m", "n"]
            names = []
            for _ in axes:
                idx = builder._var_counter % len(default_names)
                names.append(default_names[idx])
                builder._var_counter += 1
        else:
            names = list(names)

        return ParallelGridIterator(axes, names)

    def seq(self, axis: Any, name: str = "i") -> SequentialIterator:
        """Sequential iteration - task[i] depends on task[i-1].

        Type: Sequential

        Example:
            for i in P.seq(seq_len):
                scan_step[i](input[i], output[i])
        """
        return SequentialIterator(axis, name)

    def sel(self, sparse: Any, name: str = "e") -> SelectIterator:
        """Sparse iteration over selected indices (MoE routing).

        Type: Selected

        Example:
            for e in P.sel(routing[b]):
                expert[b, e](tokens[b], weights[e])
        """
        return SelectIterator(sparse, name)

    @contextmanager
    def pipe(self):
        """Pipeline scope for CSP-style programming.

        Example:
            with P.pipe():
                for i in P.seq(tiles):
                    send(ch, load[i](data))
                for tile in recv(ch):
                    compute(tile)
        """
        builder = get_current_builder()
        if builder is None:
            raise RuntimeError("P.pipe() used outside @workload context")

        from pto_wsp.workload import Workload

        # Push pipeline frame
        frame = LoopFrame(kind="pipeline", axis=None, var_name="")
        builder.push_frame(frame)

        try:
            yield
        finally:
            builder.pop_frame()

            # Build pipeline workload
            if frame.children:
                workload = Workload("pipeline", stages=frame.children)
                builder.add_child(workload)

    @contextmanager
    def when(self, predicate: Any):
        """Conditional workload branch.

        Example:
            with P.when(seq_lens[b] <= 2048):
                attn_2k[b](...)
        """
        builder = get_current_builder()
        if builder is None:
            raise RuntimeError("P.when() used outside @workload context")

        # Push conditional frame
        frame = ConditionalFrame(predicate=predicate)
        builder.push_frame(frame)

        try:
            yield
        finally:
            builder.pop_frame()

            from pto_wsp.workload import Workload

            # Build conditional workload with an explicit empty else branch by default.
            # `P.otherwise()` may overwrite the else branch by mutating the stored workload.
            then_body = self._build_body(frame.then_body)
            else_body = Workload("empty")

            w = Workload(
                "cond",
                predicate=predicate,
                then_workload=then_body,
                else_workload=else_body,
            )
            builder.add_child(w)
            builder._pending_cond_workload = w

    @contextmanager
    def otherwise(self):
        """Else branch for conditional workload.

        Example:
            with P.when(seq_lens[b] <= 2048):
                attn_2k[b](...)
            with P.otherwise():
                attn_8k[b](...)
        """
        builder = get_current_builder()
        if builder is None:
            raise RuntimeError("P.otherwise() used outside @workload context")

        pending = getattr(builder, "_pending_cond_workload", None)
        if pending is None or not hasattr(pending, "_kind") or pending._kind != "cond":
            raise RuntimeError("P.otherwise() must follow P.when()")

        # Collect else branch under a conditional frame, but only the else side.
        frame = ConditionalFrame(predicate=None, then_body=[], else_body=[])
        builder.push_frame(frame)

        try:
            yield
        finally:
            builder.pop_frame()

            # Replace else branch of the most-recently-built cond workload.
            pending._kwargs["else_workload"] = self._build_body(frame.else_body or [])
            builder._pending_cond_workload = None

    def _build_body(self, children: list) -> Any:
        from pto_wsp.workload import Workload

        if len(children) == 0:
            return Workload("empty")
        elif len(children) == 1:
            return children[0]
        else:
            return Workload("combine", workloads=children)


# Global P instance
P = PNamespace()
