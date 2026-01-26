## 1) How TVM, Triton, and JAX use context managers for DSL-ish patterns

### TVM
TVM uses **two distinct “Python DSL” styles**:

1) **Runtime builder + `with` scopes (pure Python execution)**  
`tvm.tir.ir_builder.create()` exposes scopes like `with ib.for_range(...) as i:` and `with ib.if_scope(...):` to build a TIR statement list, then `ib.get()` returns the built IR. ([apache.googlesource.com](https://apache.googlesource.com/tvm/%2B/refs/heads/v0.14.0/tests/python/unittest/test_tir_stmt_functor_ir_transform.py?utm_source=openai))  
Under the hood this is the classic “push a scope frame on enter / pop and wrap on exit” pattern.

2) **AST-embedded DSL (“TVMScript”), using `with` as syntactic markers**  
TVMScript uses `with T.block("..."):` and `with T.init():` inside a function that’s **interpreted as a DSL embedded in Python AST**, not as normal imperative Python semantics. ([tvm.apache.org](https://tvm.apache.org/docs/deep_dive/tensor_ir/learning.html?utm_source=openai))  
This is great for rich syntax, but it’s heavier (AST parsing, stricter restrictions, tooling implications).

Also note: TVM’s newer IRBuilder infrastructure explicitly documents a **thread-local “current builder”** scope mechanism (RAII-style) to route builder operations to the active builder. ([tvm.apache.org](https://tvm.apache.org/docs/reference/api/doxygen/classtvm_1_1script_1_1ir__builder_1_1IRBuilder.html?utm_source=openai))

### Triton
Triton primarily uses **decorators + ordinary Python control flow**, plus special iterators/helpers that influence compilation. Example: `for i in tl.static_range(...):` is an *iterator* (not a context manager) used to encourage unrolling inside `@triton.jit` kernels. ([triton-lang.org](https://triton-lang.org/main/python-api/generated/triton.language.static_range.html?utm_source=openai))  
So Triton is closer to “Python executes while staging a kernel” than “`with`-block builds an IR tree”.

### JAX
JAX uses context managers mostly for **staging/debugging annotations and configuration**, not to define structured control-flow syntax. For example:
- `with jax.named_scope("..."):` annotates ops in the compiled/staged program with names for profiling/debugging. ([docs.jax.dev](https://docs.jax.dev/en/latest/_autosummary/jax.named_scope.html?utm_source=openai))
- `with jax.ensure_compile_time_eval():` forces eager evaluation at trace time or errors. ([docs.jax.dev](https://docs.jax.dev/en/latest/_autosummary/jax.ensure_compile_time_eval.html?utm_source=openai))

JAX’s *actual* staged control flow is typically expressed via functional primitives (e.g. `lax.cond`, `lax.select`, `lax.switch`), because Python `if`/`for` have “execute now” semantics during tracing.

---

## 2) Capturing the workload tree when contexts exit

If you want `with parallel_for(batch) as b:` to **build a workload IR**, you need a **builder state** that records “what got emitted inside this scope” and finalizes it on `__exit__`.

The proven pattern (TVM-style) is:

- Maintain a **stack of frames**, each frame has a list `stmts` (your workload children).
- On `__enter__`: push a new frame; return a **symbolic var** for the user to reference (`b`).
- While inside the block: primitives call `emit(node)` which appends to the current frame.
- On `__exit__`: pop the frame’s children, wrap them into a node (`ParallelFor(axis, var, body)`), and `emit()` that node into the *parent* frame.

Use `contextvars.ContextVar` (preferred) or thread-local storage to hold the “current builder”, so nested/composed code is safe and doesn’t require explicitly passing `builder` everywhere.

---

## 3) Nested context managers for composition

With a stack-based builder, nesting is automatic:

```python
with parallel_for(batch) as b:
    with for_each(tokens) as t:
        task("...", [b, t], ...)
```

- `for_each` builds its node first and emits it into the `parallel_for` frame.
- Exiting `parallel_for` wraps the whole inner structure as its body.

The only tricky part is **multi-body constructs** (`cond`, `select` cases): you need *branch-specific frames* so that “then body” and “else body” don’t mix.

---

## Concrete Python implementation patterns (drop-in skeleton)

### A. A minimal “structured builder” with `with` scopes

```python
from __future__ import annotations
from dataclasses import dataclass, field
from contextlib import AbstractContextManager
from contextvars import ContextVar
from typing import Any, Optional

_current_builder: ContextVar["WorkloadBuilder|None"] = ContextVar("workload_builder", default=None)

@dataclass
class Node:
    kind: str
    attrs: dict[str, Any] = field(default_factory=dict)
    body: list["Node"] = field(default_factory=list)

@dataclass(frozen=True)
class SymVar:
    name: str
    axis: Any

@dataclass
class _Frame:
    stmts: list[Node] = field(default_factory=list)

class WorkloadBuilder:
    def __init__(self) -> None:
        self._stack: list[_Frame] = [_Frame()]

    def emit(self, node: Node) -> None:
        self._stack[-1].stmts.append(node)

    def _push(self) -> None:
        self._stack.append(_Frame())

    def _pop(self) -> list[Node]:
        return self._stack.pop().stmts

    def finish(self) -> Node:
        (root,) = self._stack  # single root frame
        return Node("root", body=root.stmts)

class workload(AbstractContextManager[WorkloadBuilder]):
    def __enter__(self) -> WorkloadBuilder:
        self._b = WorkloadBuilder()
        self._tok = _current_builder.set(self._b)
        return self._b
    def __exit__(self, exc_type, exc, tb) -> bool:
        _current_builder.reset(self._tok)
        return False

def _bld() -> WorkloadBuilder:
    b = _current_builder.get()
    if b is None:
        raise RuntimeError("No active workload builder. Use `with workload(): ...`")
    return b

class parallel_for(AbstractContextManager[SymVar]):
    def __init__(self, axis: Any, name: str = "i") -> None:
        self.axis, self.var = axis, SymVar(name, axis)
    def __enter__(self) -> SymVar:
        _bld()._push()
        return self.var
    def __exit__(self, exc_type, exc, tb) -> bool:
        body = _bld()._pop()
        if exc_type is None:
            _bld().emit(Node("parallel_for", {"axis": self.axis, "var": self.var}, body))
        return False  # never swallow exceptions

class for_each(AbstractContextManager[SymVar]):
    def __init__(self, axis: Any, name: str = "i") -> None:
        self.axis, self.var = axis, SymVar(name, axis)
    def __enter__(self) -> SymVar:
        _bld()._push()
        return self.var
    def __exit__(self, exc_type, exc, tb) -> bool:
        body = _bld()._pop()
        if exc_type is None:
            _bld().emit(Node("for_each", {"axis": self.axis, "var": self.var}, body))
        return False

def task(kernel: str, params: list[Any], resources: list[Any]) -> None:
    _bld().emit(Node("task", {"kernel": kernel, "params": params, "resources": resources}))
```

Usage:

```python
with workload() as w:
    with parallel_for(batch, name="b") as b:
        task("load", [b], [Q])
        task("compute", [b], [Q, O])
root = w.finish()
```

Key point: `with parallel_for(...)` is a **statement**, so something else (a builder context or decorator) must produce the “final Workload value”.

### B. `cond` pattern (explicit then/else sub-scopes)

Make `cond(pred)` return a handle with two inner context managers:

```python
class cond(AbstractContextManager["cond"]):
    def __init__(self, pred: Any) -> None:
        self.pred = pred
        self._then: Optional[list[Node]] = None
        self._else: Optional[list[Node]] = None

    def __enter__(self) -> "cond":
        return self

    def then_(self) -> AbstractContextManager[None]:
        return _Branch(self, which="then")

    def else_(self) -> AbstractContextManager[None]:
        return _Branch(self, which="else")

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc_type is None:
            if self._then is None or self._else is None:
                raise RuntimeError("cond requires both then_() and else_() blocks")
            _bld().emit(Node("cond", {"pred": self.pred}, body=[
                Node("then", body=self._then),
                Node("else", body=self._else),
            ]))
        return False

class _Branch(AbstractContextManager[None]):
    def __init__(self, parent: cond, which: str) -> None:
        self.parent, self.which = parent, which
    def __enter__(self) -> None:
        _bld()._push()
    def __exit__(self, exc_type, exc, tb) -> bool:
        body = _bld()._pop()
        if exc_type is None:
            if self.which == "then": self.parent._then = body
            else: self.parent._else = body
        return False
```

Usage:

```python
with cond(seq_len <= 2048) as c:
    with c.then_():
        task("attn_2k", [b, seq_len], resources)
    with c.else_():
        task("attn_8k", [b, seq_len], resources)
```

This avoids the “dangling else” problem and composes cleanly under nesting.

### C. `select` pattern (multi-case + default)

Same idea as `cond`, but multiple `case(value)` blocks + optional `default()`; on `__exit__`, wrap all cases into a single `select` node. This mirrors how many IR builders model `switch`.

---

## Design advice specific to your current lambda-based API

Your current `Workload("parallel_for", axis=..., body=<callable>)` representation (see `python/pto_wsp/primitives.py`) can’t naturally become `with parallel_for(axis) as i:` *without* introducing either:
- a **builder** that produces an IR tree (recommended), or
- an **AST-based DSL** (TVMScript-like), which is much heavier.

If you want to keep a smooth “expression returns Workload” feel, the cleanest compromise is:

- Add `with workload() as w:` as the “expression boundary”.
- Inside, all primitives are statement-like and record nodes.
- `w.finish()` returns the root `Workload`/IR tree.

If you want, I can sketch a migration path that keeps backward compatibility (`parallel_for(axis, lambda ...)` still works) while enabling the `with parallel_for(axis) as i:` form.