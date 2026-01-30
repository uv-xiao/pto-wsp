# JIT Kernel Design Research for PTO-WSP v9

## Overview

This document analyzes JIT kernel patterns from Triton, JAX, and TileLang to design PTO-WSP v9's `@kernel` decorator that eliminates string-based task definitions (addressing requirement R7).

---

## 1) Kernel Definition / Decoration

- **Triton (`@triton.jit`)**: A Python function *is* the kernel body; it’s decorated with `@triton.jit` and written using `triton.language` (`tl.*`) primitives. Meta-parameters are typically marked `tl.constexpr` so they become compile-time constants. citeturn0search0turn0search1
- **JAX (`@jax.jit`)**: A normal Python function is decorated with `@jax.jit` (or wrapped by `jax.jit(f)`); the body is expressed in JAX ops and must be traceable/pure-ish. citeturn0search3turn0search4
- **TileLang (`with T.Kernel(...)`)**: Kernel code lives inside a `@T.prim_func` (TIR-like) function; `with T.Kernel(grid) as (bx, by):` is the *single obvious launch scope marker* (matching the pattern in `docs/research/tilelang_syntax_research.md`). Often the outer function is wrapped with `@tilelang.jit`, which compiles/returns a callable kernel object. citeturn1open0turn0search2

**2) Type inference from function signatures**

- **Triton**: Runtime tensor arguments determine pointer element types; compile-time specialization happens via `tl.constexpr` parameters (their *values* become part of the specialization key). It’s less “Python typing” and more “trace types from actual call + constexpr markers”. citeturn0search0
- **JAX**: JIT traces the function with *abstract values* derived from the actual call arguments (shape/dtype/pytree structure). `static_argnums/static_argnames` force some args to be compile-time constants and therefore part of the compilation cache key. citeturn0search3turn0search4
- **TileLang**: Buffer dtypes/shapes are explicit in the `@T.prim_func` signature (e.g., `T.Tensor((m, n), "float16")`). Specialization usually comes from the *Python “factory” arguments* that produce different TIR, rather than inference from a single polymorphic signature. citeturn1open0turn0search2

**3) Host-call style vs string task names**

- **Triton**: You call a kernel object directly (no strings): `kernel[grid](...)`. citeturn0search0
- **JAX**: You call the jitted function directly like a normal function: `f_jit(x)`. citeturn0search3
- **TileLang**: You typically get a compiled kernel object (often from `@tilelang.jit`) and call it directly: `kernel(a, b, c)`. citeturn0search2
- **PTO-WSP v9 (current repo state)**: user-facing docs still show `task("attn_kernel", ...)` (e.g. `README.md`, `docs/spec.md`), but `docs/npu-design.md` already sketches a non-string path: `KernelHandle.task(**bindings)` with a `kernel_handle` stored on the task node.

**4) Compile-time vs runtime compilation strategies**

- **Triton**: Runtime JIT on first use; cached per specialization (dtypes + `constexpr`). Autotuning adds “compile multiple variants then select”. citeturn0search0turn0search1
- **JAX**: Runtime compilation on first call for a given abstract signature; supports ahead-of-time-ish via `.lower(...).compile()` to force compilation. citeturn0search4
- **TileLang**: `tilelang.jit` supports both **lazy** compilation (compile at first call) and eager-ish behavior depending on the decorator configuration. citeturn0search2

**5) A PTO-WSP v9 `@kernel` decorator design (direct calls inside workload contexts, no `task("name", ...)`)**

Leverage the split already implied by `docs/npu-design.md`: *kernel definition artifact* (like Triton/TileLang) + *workload task emission* (builder context).

Core idea: `@kernel` returns a **callable kernel reference** whose `__call__` emits a workload task when a workload builder context is active.

Recommended shape:

- **Definition time**
  - `@kernel(module=mod, name=...)` wraps a Python function and produces a `KernelRef` (or reuses `KernelHandle`) containing:
    - `symbol/name` (for linking/debug only; not user-facing strings)
    - `signature` inferred from Python annotations (inputs/outputs/constexpr/meta)
    - `lower()` hook to produce NPUFunction IR (builder-style now; AST/DSL later)
- **Use time (inside workload context managers)**
  - `KernelRef.__call__(..., **bindings)` checks the current `WorkloadBuilder` (contextvar-style as described in `docs/spec.md`) and emits a `task` node whose attrs carry the kernel object/handle directly (not a string).
  - Mapping axes can be explicit, TileLang-inspired:
    - `attn_kernel.at(b=b, h=h)(Q=..., K=..., V=..., O=...)`
    - or Triton-inspired:
    - `attn_kernel[b, h](Q=..., K=..., V=..., O=...)`
  - Bind resources by **keyword** (like `KernelHandle.task(**bindings)` in `docs/npu-design.md`) so interface mismatches are caught early.

Compilation model that matches Triton/JAX expectations while fitting PTO-WSP:

- Default: **AOT at `Workload.compile(target=...)`** = compile all referenced kernels once (deterministic builds, good for deployment).
- Optional: **runtime JIT** fallback = if a kernel hasn’t been compiled for `(target, dtypes/layouts, constexpr values, schedule hints)`, compile on first `Program.execute()` and cache (Triton/JAX-like behavior).

This gets you the UX you want:

```python
@kernel(module=mod)
def attn_kernel(Q: In[Tensor], K: In[Tensor], V: In[Tensor], O: Out[Tensor], *, BLOCK: Constexpr[int]):
    ...

with workload() as w:
    with parallel_for(batch) as b:
        with parallel_for(heads) as h:
            attn_kernel.at(b=b, h=h)(Q=Q[b][h], K=K[b], V=V[b], O=O[b][h])  # no strings
```

---

## 6) Key Design Decisions for PTO-WSP v9

### 6.1 KernelRef Object

```python
class KernelRef:
    """Reference to a registered kernel (returned by @kernel decorator)."""

    # Set at decoration time
    symbol: str                    # Internal name (for linking/debug)
    module: Optional[Module]       # Parent module if specified
    signature: KernelSignature     # Inferred from annotations

    # Methods
    def lower(self) -> NPUFunction: ...     # Produce NPU IR
    def __call__(self, **bindings) -> None: # Emit task in workload context
    def at(self, **axes) -> BoundKernel: ...  # Bind iteration axes
    def __getitem__(self, axes) -> BoundKernel: ...  # Triton-style axis binding
```

### 6.2 Signature Type System

```python
from typing import Annotated

# Direction markers
In = Annotated[T, "input"]
Out = Annotated[T, "output"]
InOut = Annotated[T, "inout"]

# Compile-time constant
Constexpr = Annotated[T, "constexpr"]

# Axis variable (from outer parallel_for)
AxisVar = Annotated[int, "axis"]

# Example signature
@kernel(module=attention_ops)
def flash_attn(
    b: AxisVar,           # From parallel_for(batch)
    h: AxisVar,           # From parallel_for(heads)
    Q: In[Tensor],
    K: In[Tensor],
    V: In[Tensor],
    O: Out[Tensor],
    *,
    BLOCK_M: Constexpr[int] = 128,
    BLOCK_N: Constexpr[int] = 64,
):
    # NPU function body or reference to NPUFunction
    ...
```

### 6.3 Compilation Strategy

| Mode | When | Use Case |
|------|------|----------|
| **AOT (default)** | At `workload.compile(target=...)` | Deterministic builds, deployment |
| **Runtime JIT** | First `program.execute()` | Interactive development |
| **Cached** | On specialization match | Both modes |

Specialization key: `(target, dtypes, layouts, constexpr_values, schedule_hints)`

### 6.4 Integration with Context Managers

```python
# The @kernel decorator returns KernelRef
# KernelRef.__call__ checks for active WorkloadBuilder

with workload() as w:
    with parallel_for(batch) as b:
        with parallel_for(heads) as h:
            # Option A: TileLang-style .at()
            flash_attn.at(b=b, h=h)(Q=Q[b][h], K=K[b], V=V[b], O=O[b][h])

            # Option B: Triton-style subscript
            flash_attn[b, h](Q=Q[b][h], K=K[b], V=V[b], O=O[b][h])

            # Option C: Direct call (axes inferred from context)
            flash_attn(b, h, Q=Q[b][h], K=K[b], V=V[b], O=O[b][h])
```

### 6.5 Backward Compatibility

```python
# Legacy string-based task() still works (deprecated)
task("kernel_name", [b, h], [Q, K, V, O])  # ⚠️ Deprecated

# New direct kernel call (preferred)
flash_attn[b, h](Q=Q, K=K, V=V, O=O)  # ✅ Recommended
```

---

## 7) Recommendations for Implementation

1. **Phase 1**: Implement `@kernel` decorator returning `KernelRef`
2. **Phase 2**: Implement context-aware `__call__` that emits to WorkloadBuilder
3. **Phase 3**: Add `.at()` and `__getitem__` for axis binding
4. **Phase 4**: Wire up signature inference from type annotations
5. **Phase 5**: Implement AOT compilation path in `Schedule.compile()`

The design leverages existing `docs/npu-design.md` infrastructure while providing Triton/JAX-like UX.