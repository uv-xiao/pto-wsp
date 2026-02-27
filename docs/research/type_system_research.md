# Unified Type System Research for PTO-WSP v9

## Overview

This document analyzes type system design patterns from Triton, Dato, and existing v8 architecture to design a unified type system for PTO-WSP v9 that addresses requirements R8 (Unified Type System) and R10 (Layout as Refinement Types).

---

## 1) Triton "Linear Layout" (What to Adopt)

The core idea to import is: represent *any* tile memory layout as a **first-class, composable index→offset function**, not as ad-hoc flags (“row-major”, “col-major”, “NZ”, “swizzled”, etc.). Concretely:

- A **LinearLayout** is a mapping `f: (i0, i1, ..., ik) -> (base + Σ ij * stridej)` plus:
  - an **iteration order** (which dims are contiguous / vectorized),
  - optional **swizzle/bank-conflict** permutations,
  - optional **packing** (e.g., vector lanes, MMA fragment packing),
  - and a **composition** operator (tile/reshape/permute = layout transforms).

Why it matters for v9 types: it gives you a single “layout algebra” object you can use at **both** levels:
- CPU-level tensor layout reasoning (stride/order/contiguity)
- NPU-level tile/fragment layouts (UB/L1/L0A/L0B/L0C formats), including “fractal/NZ-like” transforms already noted in PTO-ISA research (`docs/research/04_pto_isa.md`).

This matches the repo’s existing direction to learn from CuTe-style layout composition (`docs/research/02_gpu_patterns.md`) while adding a *typed* representation instead of “schedule knobs”.

---

## 2) Dato refinement layouts (S|R) and the missing piece in current v9

Dato’s layout type is a **refinement** on a tensor type (`τ ::= D[n̄]@L`) where each tensor dimension is `S(axis)` or `R` (`docs/research/16_dato.md`). Key rules you likely want to preserve:

- Elementwise: result layout is a “join” (`Lx ⊔ Ly`) — intuitively, `R ⊔ S(a) = S(a)` and `S(a) ⊔ S(b)` is invalid unless `a=b` (otherwise you need an explicit relayout/collective).
- Reduce on sharded dim produces a **pending collective effect** that must be discharged by `allreduce` (Dato models this as an effect set Π).

What v9 currently has: `Shard(dim=0)` / `Replicate()` as *schedule directives* (`docs/spec.md` “Spatial Primitives”, and IR `LayoutNode` in `docs/ir-design.md`). That’s the opposite of R10/R10-like guidance in `docs/archive/comments.md`: layouts should be **types**, not schedule primitives.

---

## 3) Unifying NPU-level types with CPU-level types: one `Layout` with two facets

Right now you conceptually have two separate “layout” worlds:
- CPU world: `AxisType` (`Dense/DenseDyn/Ragged/Sparse`) and a (planned) Dato-like `TensorLayout`
- NPU world: `TileShape`, `BufferLocation`, `TileLayout` (PTO-ISA tile formats, UB/L1/L0*, NZ/ND conversions)

A clean unification is to define **one** `Layout` refinement with two orthogonal components:

### A. Distribution (Dato facet)
Per tensor dimension:
- `Replicate`
- `Shard(mesh_axis=i)`  (Dato’s `S0`, `S1`, … notion; note: this is *not* the tensor dim, it’s the **mesh** axis)

This requires a notion of a **Mesh** (from `spatial_map(grid=...)`), but you don’t need to bake it into `Workload[...]` generics: it can be a constraint checked once the schedule (mesh) exists.

### B. Memory (Triton/CuTe facet)
A `MemLayout` that is (at minimum):
- `shape` + `strides` + `order`
- optional `swizzle` / `pack` / `fractal` tags
- composable transforms (`permute`, `tile`, `reshape`, `compose`)

Then you can define:
- `TensorLayout = Layout[Dist=..., Mem=...]` (CPU-visible, NPU-lowering-relevant)
- `TileLayout = MemLayout[...]` (NPU tile buffers / fragments)

And keep `BufferLocation` / `Location` as a separate refinement on the storage object (e.g., UB vs GM), which is already modeled in `docs/spec.md` and `python/pto_wsp/types.py`.

---

## 4) Python `typing` patterns that actually work for a DSL (and what to avoid)

Static typing in Python won’t give you true dependent/refinement types, so the practical pattern is:

1) **Runtime type objects + validation** (primary)
- `Layout`, `MemLayout`, `Shard/Replicate`, `TileShape`, etc. are runtime values.
- The DSL builder validates constraints immediately (during workload/kernel construction), so errors surface early (as requested in `docs/archive/comments.md`).

2) **Type hints for UX + IDE help** (secondary)
Use typing where it helps without fighting the language:
- `typing.Annotated` to attach refinements: `Annotated[Tensor, Layout(...)]`
- `Literal[int]` / `typing_extensions.TypeVarTuple` for constexpr-ish shape parameters (mostly for docs/IDE; still validate at runtime)
- `Protocol` for `KernelRef`/`BoundKernel` callability (mirrors the spec’s `KernelRef` API in `docs/spec.md`)
- `@overload` for ergonomic frontends (e.g., `.at(...)` vs `__getitem__` axis binding)

Avoid promising that mypy/pyright can prove layout correctness; instead, make the **builder/typechecker** the source of truth, and treat typing as hints.

---

## 5) Layout as a refinement type (R10): what changes in v9 design

Target state (aligning `docs/archive/comments.md` + Dato):
- **Remove** `schedule.layout(...)` as a schedule primitive (and likely remove/repurpose IR `LayoutNode` as a “tensor type annotation” node).
- Make layout part of tensor types: `Tensor[DType, Shape, Layout]`.
- `spatial_map(grid=...)` remains schedule-level, but it becomes the *environment* that validates `Shard(mesh_axis=...)` refinements.

You still need an *explicit* operation when layouts are incompatible:
- `relayout(x, to=Layout(...))` or a collective op (`allreduce`, `allgather`, etc.), so the programmer (or an optimizer pass) makes communication explicit rather than silently changing distribution.

---

## 6) Preserving v8 `Workload[Axes, Task, Deps]` signatures while adding layouts

You can keep v8’s typing rules exactly as-is (`docs/archive/v8/analysis.md`, and mirrored across `docs/spec.md`), by treating layout as part of the **Task**’s resource types, not as an extra `Workload[...]` generic.

Concretely:
- `Axes` stays “iteration axes” (Dense/DenseDyn/… products)
- `Deps` stays `Independent | Sequential | Combined | ...`
- `Task` becomes richer: `Task[K, Params, IO]` where `IO` references typed tensors:
  - `Q: Tensor[F16, (B,H,S,D), Layout(...)]`, etc.

So `parallel_for/for_each/select/combine/sequential` keep the same `Workload[...]` shape, but the *leaf* tasks carry stronger tensor/layout refinements. That preserves the v8 mental model and signatures while enabling Dato/Triton-style layout checking inside kernel calls and tensor ops.

---

---

## 7) Proposed Type System Design

### 7.1 Core Types

```python
from typing import Generic, TypeVar, Annotated
from enum import Enum

# Data types
class DType(Enum):
    F16 = "f16"
    BF16 = "bf16"
    F32 = "f32"
    # ...

# Buffer locations
class Location(Enum):
    Global = "global"  # HBM
    L2 = "l2"
    UB = "ub"          # Unified Buffer (NPU)
    L1 = "l1"

# Distribution types (Dato-style)
class Replicate:
    """Full data on each tile."""
    pass

class Shard:
    """Partitioned along mesh axis."""
    def __init__(self, mesh_axis: int):
        self.mesh_axis = mesh_axis

DistElem = Replicate | Shard

# Memory layout (Triton-style)
class MemLayout:
    """Physical memory arrangement."""
    strides: tuple[int, ...]
    order: tuple[int, ...]  # Iteration order
    swizzle: Optional[SwizzleSpec] = None

    def compose(self, other: "MemLayout") -> "MemLayout": ...
    def permute(self, perm: tuple[int, ...]) -> "MemLayout": ...
    def tile(self, tile_shape: tuple[int, ...]) -> "MemLayout": ...

# Unified Layout = Distribution × Memory
class Layout:
    """Unified layout type with distribution and memory facets."""
    dist: tuple[DistElem, ...]  # Per-dimension distribution
    mem: Optional[MemLayout] = None  # Physical layout (optional)

    def __init__(self, *dist: DistElem, mem: MemLayout = None):
        self.dist = dist
        self.mem = mem

# Tensor with layout refinement
class Tensor(Generic[D, S, L]):
    """Tensor type with dtype, shape, and layout refinements."""
    dtype: D
    shape: S
    layout: L
    location: Location = Location.Global
```

### 7.2 Layout Compatibility Rules (Dato-style)

```python
def layout_join(a: Layout, b: Layout) -> Layout:
    """Compute layout join for elementwise operations."""
    if len(a.dist) != len(b.dist):
        raise TypeError("Layout dimension mismatch")

    result = []
    for da, db in zip(a.dist, b.dist):
        if isinstance(da, Replicate):
            result.append(db)
        elif isinstance(db, Replicate):
            result.append(da)
        elif da.mesh_axis == db.mesh_axis:
            result.append(da)
        else:
            raise TypeError(f"Incompatible sharding: S({da.mesh_axis}) vs S({db.mesh_axis})")

    return Layout(*result)
```

### 7.3 Type Checking Locations

| Level | What's Checked | When |
|-------|---------------|------|
| **Python Builder** | Layout compatibility for kernel calls | Immediately at workload construction |
| **IR Type Pass** | Cross-workload layout consistency | At `workload.compile()` |
| **Backend Lowering** | Physical layout feasibility for target | At backend `lower()` |

---

## 8) Recommendations for Implementation

1. **Phase 1**: Define `Layout`, `MemLayout`, `Shard`, `Replicate` in `python/pto_wsp/types.py`
2. **Phase 2**: Add `Layout` parameter to `Tensor` class
3. **Phase 3**: Remove `schedule.layout()` from schedule API (per R10)
4. **Phase 4**: Update IR `LayoutNode` to be type annotation, not schedule directive
5. **Phase 5**: Implement `layout_join()` and compatibility checking in builder
6. **Phase 6**: Add `relayout()` and collective primitives for explicit redistribution

This design unifies NPU-level and CPU-level types while preserving v8's `Workload[Axes, Task, Deps]` signatures.
