"""
Workload primitives for PTO Workload-Schedule Programming (PTO-WSP) framework.

Declarative functions for defining workloads.
"""

from __future__ import annotations
from typing import Callable, Any, TypeVar
from pto_wsp.workload import Workload

T = TypeVar("T")


def parallel_for(axis: Any, body: Callable[[int], Any]) -> Workload:
    """Independent iteration over axis.

    Creates workloads where all iterations can run in parallel.

    Args:
        axis: Iteration axis (Dense, DenseDyn, etc.)
        body: Lambda function mapping index to workload

    Returns:
        Workload with Independent dependency type

    Example:
        attention = parallel_for(batch, lambda b:
            parallel_for(heads, lambda h:
                task("attn", [b, h], resources)))
    """
    return Workload("parallel_for", axis=axis, body=body)


def for_each(axis: Any, body: Callable[[int], Any]) -> Workload:
    """Sequential iteration over axis.

    Creates workloads where task[i] depends on task[i-1].

    Args:
        axis: Iteration axis
        body: Lambda function mapping index to workload

    Returns:
        Workload with Sequential dependency type

    Example:
        scan = for_each(seq, lambda i:
            task("scan", [i], resources))
    """
    return Workload("for_each", axis=axis, body=body)


def select(sparse: Any, body: Callable[[int], Any]) -> Workload:
    """Sparse iteration over selected indices.

    Used for MoE routing where only certain experts are invoked.

    Args:
        sparse: Sparse axis or list of indices
        body: Lambda function mapping index to workload

    Returns:
        Workload with Independent dependency type

    Example:
        moe = select(routing[b], lambda e:
            task(f"expert_{e}", [b, e], resources))
    """
    return Workload("select", sparse=sparse, body=body)


def cond(predicate: bool, then_workload: Any, else_workload: Any) -> Workload:
    """Conditional workload selection.

    Args:
        predicate: Runtime condition
        then_workload: Workload if predicate is true
        else_workload: Workload if predicate is false

    Returns:
        Workload that selects between then/else based on predicate

    Example:
        tiered = cond(seq_len <= 2048,
            task("attn_2k", [b, seq_len], resources),
            task("attn_8k", [b, seq_len], resources))
    """
    return Workload("cond", predicate=predicate,
                    then_workload=then_workload,
                    else_workload=else_workload)


def task(kernel: str, params: list[Any], resources: list[Any]) -> Workload:
    """Single kernel invocation (leaf workload).

    Args:
        kernel: Kernel name (string)
        params: Kernel parameters
        resources: Input/output tensors

    Returns:
        Workload with None dependency type

    Example:
        t = task("attention_kernel", [batch, head, seq_len], [Q, K, V, O])
    """
    from pto_wsp.builder import get_current_builder

    w = Workload("task", kernel=kernel, params=params, resources=resources)

    # If inside a @workload context with P loops, add to builder
    builder = get_current_builder()
    if builder is not None and builder.frames:
        builder.add_child(w)

    return w


def combine(*workloads: Any) -> Workload:
    """Compose workloads (schedule determines order).

    Use when workloads have no inherent ordering dependency.

    Args:
        *workloads: Workloads to combine

    Returns:
        Workload with Combined dependency type

    Example:
        layer = combine(
            rms_norm_workload,
            attention_workload,
            ffn_workload)
    """
    return Workload("combine", workloads=list(workloads))


def sequential(*workloads: Any) -> Workload:
    """Explicit ordering where later workloads depend on earlier ones.

    Args:
        *workloads: Workloads in order (each waits for previous to complete)

    Returns:
        Workload with Sequential dependency type

    Example:
        pipeline = sequential(
            parallel_for(batch, lambda b: task("load", [b], [...])),
            parallel_for(batch, lambda b: task("compute", [b], [...])),
            parallel_for(batch, lambda b: task("store", [b], [...])))
    """
    return Workload("sequential", workloads=list(workloads))
