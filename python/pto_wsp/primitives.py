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


def cond(predicate: Any, then_workload: Any, else_workload: Any) -> Workload:
    """Conditional workload selection.

    Args:
        predicate: Runtime predicate (v9: ScalarExpr or convertible literal)
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


def task(kernel: Any, params: list[Any], resources: Any) -> Workload:
    """Single kernel invocation (leaf workload).

    This function supports two modes:

    1) Legacy enumerate-only mode (no codegen): `kernel` is a `str` name and
       `resources` is a positional `list` of tensors/values.

    2) Codegen-first mode: `kernel` is a `@kernel` object and `resources` is a
       `dict[param_name -> Tensor/scalar]` matching the kernel signature.

    Returns a `Workload("task", ...)` node. When invoked inside a `@workload`
    builder loop, the task is also added to the builder.
    """
    from pto_wsp.builder import get_current_builder

    if kernel is not None and hasattr(kernel, "trace") and hasattr(kernel, "name"):
        if not isinstance(resources, dict):
            raise TypeError("task(kernel=@kernel, ...) requires resources=dict[param_name -> Tensor/scalar]")
        w = Workload("task", kernel=kernel, params=params, resources=resources)
    else:
        w = Workload("task", kernel=str(kernel), params=params, resources=list(resources))

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


def slot_set_u64(slot: int, value: int) -> Workload:
    """Write a runtime u64 slot (side effect).

    Slots live inside codegen-first artifacts and can be used by ScalarExpr
    predicates/keys (v9) to enable data-dependent control.
    """
    from pto_wsp.builder import get_current_builder

    w = Workload("slot_set_u64", slot=int(slot), value=int(value) & 0xFFFFFFFFFFFFFFFF)
    builder = get_current_builder()
    if builder is not None:
        builder.add_child(w)
    return w


def slot_load_u64(slot: int, tensor: Any, row: int = 0, col: int = 0) -> Workload:
    """Load a u64 value from a tensor element into a runtime slot (side effect).

    The tensor must be a `pto_wsp.types.Tensor` view that resolves to a 2D tile
    view (last two dims). `row/col` are indices within that tile view.
    """
    from pto_wsp.builder import get_current_builder

    w = Workload("slot_load_u64", slot=int(slot), tensor=tensor, row=int(row), col=int(col))
    builder = get_current_builder()
    if builder is not None:
        builder.add_child(w)
    return w
