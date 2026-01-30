# PTO-WSP v9: Formal Type System

## 1. Overview

This document formally specifies the type system for PTO-WSP v9, covering:

1. **Type definitions** for axes, tensors, layouts, workloads, and tasks
2. **Type checking rules** at Python builder and IR compiler levels
3. **Type inference** for workload composition
4. **Compatibility rules** for layout joins and redistribution

### 1.1 Design Principles

| Principle | Description |
|-----------|-------------|
| **Refinement Types** | Layout is a type refinement on Tensor, not a schedule directive |
| **Gradual Typing** | Static hints where possible; runtime validation always |
| **Early Errors** | Type errors surface at builder time, not execution time |
| **Composable** | Layout algebra enables composable layout transformations |

### 1.2 Type Checking Levels

| Level | Location | When | What's Checked |
|-------|----------|------|----------------|
| **L1: Python Builder** | `python/pto_wsp/` | Workload construction | Kernel signatures, layout compatibility |
| **L2: IR Type Pass** | `src/pto/wsp/ir/` | `workload.compile()` | Cross-workload consistency, axis bounds |
| **L3: Backend Lowering** | `src/pto/wsp/backend/` | Backend `lower()` | Physical layout feasibility, memory constraints |

---

## 2. Core Type Definitions

### 2.1 Axis Types

Axes describe iteration spaces with structural guarantees.

```
AxisType ::= Dense[N]           -- Static size N (compile-time constant)
           | DenseDyn           -- Dynamic size (runtime value)
           | Ragged             -- Variable lengths per outer element
           | Sparse             -- CSR format (sparse iteration)
```

**Formal definitions:**

```python
# Dense: statically-known iteration count
Dense[N] : AxisType where N : ℕ
  ├── size : N (compile-time constant)
  └── indices : {0, 1, ..., N-1}

# DenseDyn: runtime iteration count
DenseDyn : AxisType
  ├── size : IntVar (runtime symbolic or concrete)
  └── indices : {0, 1, ..., size-1}

# Ragged: variable lengths per outer element
Ragged : AxisType
  ├── outer_size : int
  ├── lengths : list[int]  where len(lengths) = outer_size
  └── indices(i) : {0, 1, ..., lengths[i]-1} for i in 0..outer_size-1

# Sparse: CSR-format iteration
Sparse : AxisType
  ├── outer_size : int
  ├── indptr : list[int]  where len(indptr) = outer_size + 1
  ├── indices : list[int]
  └── selected(i) : {indices[j] for j in indptr[i]..indptr[i+1]-1}
```

**Typing rules:**

```
├─────────────────────────────────────────────────────────────
│ Γ ⊢ n : ℕ  (n is a compile-time constant)
│ ─────────────────────────────────────────
│ Γ ⊢ Dense[n] : AxisType
├─────────────────────────────────────────────────────────────
│ Γ ⊢ s : int | IntVar
│ ──────────────────────
│ Γ ⊢ DenseDyn(s) : AxisType
├─────────────────────────────────────────────────────────────
│ Γ ⊢ n : int   Γ ⊢ lens : list[int]   len(lens) = n
│ ─────────────────────────────────────────────────────
│ Γ ⊢ Ragged(n, lens) : AxisType
├─────────────────────────────────────────────────────────────
│ Γ ⊢ n : int   Γ ⊢ indptr : list[int]   Γ ⊢ indices : list[int]
│ len(indptr) = n + 1   indptr[0] = 0   indptr[n] = len(indices)
│ ──────────────────────────────────────────────────────────────
│ Γ ⊢ Sparse(n, indptr, indices) : AxisType
└─────────────────────────────────────────────────────────────
```

### 2.2 Data Types

```
DType ::= F16 | BF16 | F32 | F64           -- Floating point
        | I8 | I16 | I32 | I64             -- Signed integer
        | U8 | U16 | U32 | U64             -- Unsigned integer
        | Bool                              -- Boolean

Location ::= Global | L2 | UB | L1          -- Memory hierarchy
```

### 2.3 Layout Types

Layout is a **type-level refinement** with two orthogonal facets:

```
DistElem ::= Replicate          -- Full copy on each worker
           | Shard(mesh_axis)   -- Partitioned along mesh axis

MemLayout ::= {
    strides: tuple[int, ...],   -- Per-dimension strides
    order: tuple[int, ...],     -- Iteration order (contiguity)
    swizzle: Option[SwizzleSpec] -- Bank conflict avoidance
}

Layout ::= Layout(dist: tuple[DistElem, ...], mem: Option[MemLayout])
```

**Formal definition:**

```python
# Distribution element
Replicate : DistElem
  └── semantics: full data replicated on all workers

Shard(i) : DistElem where i : ℕ (mesh axis index)
  └── semantics: data partitioned along mesh axis i

# Memory layout (Triton-style composable)
MemLayout : MemType
  ├── strides : tuple[int, ...]
  ├── order : tuple[int, ...]
  ├── swizzle : Option[SwizzleSpec]
  └── operations:
      ├── compose : MemLayout × MemLayout → MemLayout
      ├── permute : MemLayout × tuple[int, ...] → MemLayout
      └── tile : MemLayout × tuple[int, ...] → MemLayout

# Unified layout
Layout(d₁, d₂, ..., dₙ; mem=m) : LayoutType
  where dᵢ : DistElem for i in 1..n
        m : Option[MemLayout]
```

### 2.4 Tensor Types

Tensors are parameterized by dtype, shape, and layout refinement:

```
Tensor[D, S, L] : TensorType
  where D : DType
        S : tuple[int | IntVar, ...]  -- Shape
        L : LayoutType                -- Layout refinement
```

**Examples:**

```python
# Tensor with explicit layout refinement
Q : Tensor[F16, (B, H, S, D), Layout(Shard(0), Replicate(), Replicate(), Replicate())]
K : Tensor[F16, (B, S, D), Layout(Replicate(), Shard(1), Replicate())]
O : Tensor[F16, (B, H, S, D), Layout(Shard(0), Shard(1), Replicate(), Replicate())]
```

### 2.5 Kernel Types

Kernels have typed signatures with direction annotations:

```
KernelType ::= (params: list[ParamType]) → Effect
  where ParamType ::= In[T] | Out[T] | InOut[T] | Constexpr[T]
        Effect ::= Pure | Mutates[list[str]]
```

**Formal definition:**

```python
# Parameter direction annotations
In[T] : ParamType      -- Read-only input
Out[T] : ParamType     -- Write-only output
InOut[T] : ParamType   -- Read-write
Constexpr[T] : ParamType  -- Compile-time constant

# Kernel signature
@kernel
def f(p₁: In[T₁], p₂: Out[T₂], ...) : KernelRef
  where f.signature = KernelType((In[T₁], Out[T₂], ...), Mutates[out_params])
```

### 2.6 Workload Types

Workloads have the v8-style type signature:

```
Workload[Axes, Task, Deps] : WorkloadType
  where Axes : AxisType × AxisType × ...  -- Product of iteration axes
        Task : TaskType                    -- Task type with kernel + resources
        Deps : DependencyType              -- Dependency classification
```

**Dependency types:**

```
DependencyType ::= Independent    -- All tasks can run in parallel
                 | Sequential     -- Task[i] depends on Task[i-1]
                 | Combined       -- Schedule determines order
                 | Pipeline       -- Channel-based dependencies
                 | Conditional    -- Predicate-based selection
```

---

## 3. Type Checking Rules

### 3.1 Layout Compatibility (Dato Rules)

The join operator computes compatible layouts for elementwise operations:

```
layout_join : Layout × Layout → Layout | TypeError

Rules:
├─ R ⊔ R = R                           -- Replicate ⊔ Replicate
├─ R ⊔ S(i) = S(i)                     -- Replicate ⊔ Shard
├─ S(i) ⊔ R = S(i)                     -- Shard ⊔ Replicate
├─ S(i) ⊔ S(i) = S(i)                  -- Same shard axis
└─ S(i) ⊔ S(j) = TypeError if i ≠ j   -- Incompatible sharding
```

**Implementation:**

```python
def layout_join(a: Layout, b: Layout) -> Layout:
    """Join layouts for elementwise operations."""
    if len(a.dist) != len(b.dist):
        raise TypeError(f"Layout rank mismatch: {len(a.dist)} vs {len(b.dist)}")

    result = []
    for i, (da, db) in enumerate(zip(a.dist, b.dist)):
        if isinstance(da, Replicate):
            result.append(db)
        elif isinstance(db, Replicate):
            result.append(da)
        elif isinstance(da, Shard) and isinstance(db, Shard):
            if da.mesh_axis == db.mesh_axis:
                result.append(da)
            else:
                raise TypeError(
                    f"Incompatible sharding at dim {i}: "
                    f"S({da.mesh_axis}) vs S({db.mesh_axis})"
                )
        else:
            result.append(da)  # Both same type

    return Layout(*result)
```

### 3.2 Kernel Call Type Checking

When a kernel is called in a workload, the following must be verified:

```
├─────────────────────────────────────────────────────────────
│ KERNEL-CALL
│
│ Γ ⊢ k : KernelRef with signature (p₁: D₁[T₁], ..., pₙ: Dₙ[Tₙ])
│ Γ ⊢ aᵢ : Tᵢ  for each argument i
│ Γ ⊢ axes : tuple[AxisVar, ...]  (bound loop variables)
│
│ Checks:
│ 1. Arity: n = |arguments|
│ 2. Type compatibility: typeof(aᵢ) <: Tᵢ
│ 3. Direction: Out/InOut params must be assignable lvalues
│ 4. Layout: layout_join(tensor_layouts) succeeds
│ ──────────────────────────────────────────────────────────
│ Γ ⊢ k[axes](a₁, ..., aₙ) : TaskType
└─────────────────────────────────────────────────────────────
```

### 3.3 Loop Constructor Type Rules

```
├─────────────────────────────────────────────────────────────
│ P-GRID (parallel)
│
│ Γ ⊢ axes : tuple[AxisType, ...]
│ Γ, vars : AxisVars ⊢ body : Workload[A, T, D]
│ ───────────────────────────────────────────────
│ Γ ⊢ for vars in P(axes): body : Workload[axes × A, T, Independent]
├─────────────────────────────────────────────────────────────
│ P-SEQ (sequential)
│
│ Γ ⊢ axis : AxisType
│ Γ, i : AxisVar ⊢ body : Workload[A, T, D]
│ ───────────────────────────────────────────────
│ Γ ⊢ for i in P.seq(axis): body : Workload[axis × A, T, Sequential]
├─────────────────────────────────────────────────────────────
│ P-SEL (sparse selection)
│
│ Γ ⊢ sparse : Sparse
│ Γ, e : AxisVar ⊢ body : Workload[A, T, D]
│ ───────────────────────────────────────────────
│ Γ ⊢ for e in P.sel(sparse): body : Workload[sparse × A, T, Independent]
├─────────────────────────────────────────────────────────────
│ P-WHEN (conditional)
│
│ Γ ⊢ pred : bool
│ Γ ⊢ then_body : Workload[A₁, T₁, D₁]
│ Γ ⊢ else_body : Workload[A₂, T₂, D₂]
│ ───────────────────────────────────────────────
│ Γ ⊢ with P.when(pred): then_body; else: else_body
│     : Workload[A₁ ∪ A₂, T₁ ∪ T₂, Conditional]
└─────────────────────────────────────────────────────────────
```

### 3.4 Composition Type Rules

```
├─────────────────────────────────────────────────────────────
│ COMBINE
│
│ Γ ⊢ w₁ : Workload[A₁, T₁, D₁]
│ Γ ⊢ w₂ : Workload[A₂, T₂, D₂]
│ ──────────────────────────────────────────
│ Γ ⊢ combine(w₁, w₂) : Workload[A₁ × A₂, T₁ ∪ T₂, Combined]
├─────────────────────────────────────────────────────────────
│ SEQUENTIAL
│
│ Γ ⊢ w₁ : Workload[A₁, T₁, D₁]
│ Γ ⊢ w₂ : Workload[A₂, T₂, D₂]
│ ──────────────────────────────────────────
│ Γ ⊢ sequential(w₁, w₂) : Workload[A₁ × A₂, T₁ ∪ T₂, Sequential]
└─────────────────────────────────────────────────────────────
```

---

## 4. Python-Level Type Checking

### 4.1 Builder-Time Validation

The Python builder validates types **immediately** during workload construction:

```python
# python/pto_wsp/type_checker.py

class TypeChecker:
    """Builder-time type checker for workloads."""

    def __init__(self, context: BuilderContext):
        self.ctx = context
        self.errors: list[TypeError] = []

    def check_kernel_call(self, kernel: KernelRef, axes: tuple, args: dict) -> None:
        """Validate kernel invocation."""
        sig = kernel.signature

        # 1. Check arity
        if len(args) != len(sig.params):
            self.error(f"Expected {len(sig.params)} args, got {len(args)}")

        # 2. Check argument types
        for name, param in sig.params.items():
            if name not in args:
                self.error(f"Missing argument: {name}")
                continue

            arg = args[name]
            self._check_param_type(name, param, arg)

        # 3. Check layout compatibility
        self._check_layout_compatibility(args)

    def check_axis_bounds(self, axis: AxisType, index: AxisVar) -> None:
        """Validate axis index is in bounds."""
        # Static bounds check for Dense[N]
        if isinstance(axis, Dense) and isinstance(index, int):
            if not (0 <= index < axis.size):
                self.error(f"Index {index} out of bounds for Dense[{axis.size}]")

    def check_tensor_access(self, tensor: Tensor, indices: tuple) -> None:
        """Validate tensor indexing."""
        if len(indices) > len(tensor.shape):
            self.error(f"Too many indices: {len(indices)} for rank-{len(tensor.shape)} tensor")

    def _check_param_type(self, name: str, param: ParamType, arg: Any) -> None:
        """Check argument matches parameter type."""
        expected_type = param.inner_type
        actual_type = type(arg)

        # Tensor type check
        if isinstance(expected_type, TensorType):
            if not isinstance(arg, Tensor):
                self.error(f"Argument {name}: expected Tensor, got {actual_type}")
                return

            # DType check
            if arg.dtype != expected_type.dtype:
                self.error(f"Argument {name}: dtype mismatch: {arg.dtype} vs {expected_type.dtype}")

            # Shape compatibility (allow symbolic)
            if not self._shapes_compatible(arg.shape, expected_type.shape):
                self.error(f"Argument {name}: shape mismatch")

    def _check_layout_compatibility(self, args: dict[str, Tensor]) -> None:
        """Check layout compatibility across arguments."""
        layouts = [arg.layout for arg in args.values() if hasattr(arg, 'layout')]
        if len(layouts) < 2:
            return

        try:
            result = layouts[0]
            for layout in layouts[1:]:
                result = layout_join(result, layout)
        except TypeError as e:
            self.error(f"Layout incompatibility: {e}")

    def error(self, msg: str) -> None:
        self.errors.append(TypeError(msg))
        if self.ctx.fail_fast:
            raise TypeError(msg)
```

### 4.2 Runtime Type Annotations

Use Python typing for IDE support and documentation:

```python
# python/pto_wsp/types.py

from typing import Generic, TypeVar, Annotated, Protocol, overload

D = TypeVar('D', bound=DType)
S = TypeVar('S')  # Shape tuple
L = TypeVar('L', bound=Layout)

class Tensor(Generic[D, S, L]):
    """Tensor with type parameters for dtype, shape, and layout."""

    def __init__(self, data: Any, dtype: D, shape: S, layout: L = None):
        self.data = data
        self.dtype = dtype
        self.shape = shape
        self.layout = layout or Layout.default(len(shape))

    @overload
    def __getitem__(self, idx: int) -> "Tensor[D, S, L]": ...
    @overload
    def __getitem__(self, idx: tuple) -> "Tensor[D, S, L]": ...

    def __getitem__(self, idx):
        # Runtime indexing with type preservation
        ...

class KernelRef(Protocol):
    """Protocol for callable kernel references."""

    @property
    def signature(self) -> KernelSignature: ...

    def __getitem__(self, axes: tuple) -> "BoundKernel": ...
    def __call__(self, **kwargs: Tensor) -> Task: ...
```

---

## 5. Compiler-Level Type Checking

### 5.1 IR Type Pass

The IR type pass validates type consistency after Python-to-IR conversion:

```cpp
// include/pto/wsp/ir/passes/type_check_pass.hpp

namespace pto::wsp::ir {

class TypeCheckPass : public Pass {
public:
    std::string_view name() const override { return "type-check"; }

    PassResult run(Module& m, PassContext& ctx) override;

private:
    // Check workload type consistency
    void checkWorkload(const WorkloadDef& w, PassContext& ctx);

    // Check task type signatures
    void checkTask(const TaskNode& t, PassContext& ctx);

    // Check axis bounds
    void checkAxisBounds(const ParallelForNode& pf, PassContext& ctx);

    // Check layout compatibility across workload
    void checkLayoutConsistency(const WorkloadDef& w, PassContext& ctx);

    // Check CSP channel types
    void checkChannelTypes(const PipelineNode& p, PassContext& ctx);
};

// Type representation in IR
struct IRType {
    enum Kind { Axis, Tensor, Kernel, Workload, Layout };
    Kind kind;

    // Type-specific data
    std::variant<
        AxisTypeInfo,
        TensorTypeInfo,
        KernelTypeInfo,
        WorkloadTypeInfo,
        LayoutTypeInfo
    > info;
};

struct TensorTypeInfo {
    DType dtype;
    std::vector<int64_t> shape;  // -1 for dynamic dims
    std::optional<LayoutInfo> layout;
};

struct LayoutInfo {
    std::vector<DistElem> dist;
    std::optional<MemLayoutInfo> mem;
};

}
```

### 5.2 Type Inference

The compiler infers workload types from structure:

```cpp
// include/pto/wsp/ir/passes/type_inference_pass.hpp

namespace pto::wsp::ir {

class TypeInferencePass : public Pass {
public:
    std::string_view name() const override { return "type-inference"; }

    PassResult run(Module& m, PassContext& ctx) override;

private:
    // Infer workload type from body structure
    WorkloadTypeInfo inferWorkloadType(const IRPtr<WorkloadNode>& body);

    // Infer dependency type from loop constructor
    DependencyType inferDependency(NodeKind kind);

    // Infer axis product from nested loops
    AxisProduct inferAxes(const IRPtr<WorkloadNode>& body);
};

// Inferred dependency types
DependencyType inferDependency(NodeKind kind) {
    switch (kind) {
        case NodeKind::ParallelFor: return DependencyType::Independent;
        case NodeKind::ForEach:     return DependencyType::Sequential;
        case NodeKind::Select:      return DependencyType::Independent;
        case NodeKind::Cond:        return DependencyType::Conditional;
        case NodeKind::Combine:     return DependencyType::Combined;
        case NodeKind::Sequential:  return DependencyType::Sequential;
        case NodeKind::Pipeline:    return DependencyType::Pipeline;
        default:                    return DependencyType::Unknown;
    }
}

}
```

### 5.3 Backend Type Constraints

Each backend enforces additional type constraints:

```cpp
// include/pto/wsp/backend/type_constraints.hpp

namespace pto::wsp::backend {

class BackendTypeConstraints {
public:
    virtual ~BackendTypeConstraints() = default;

    // Check if dtype is supported
    virtual bool supportsDType(DType dtype) const = 0;

    // Check if layout is feasible
    virtual bool supportsLayout(const LayoutInfo& layout) const = 0;

    // Check memory constraints
    virtual bool checkMemoryConstraints(
        const TensorTypeInfo& tensor,
        Location loc) const = 0;

    // Maximum supported dimensions
    virtual int maxRank() const = 0;
};

// Ascend NPU constraints
class AscendNPUConstraints : public BackendTypeConstraints {
public:
    bool supportsDType(DType dtype) const override {
        return dtype == DType::F16 || dtype == DType::BF16 ||
               dtype == DType::F32 || dtype == DType::I32;
    }

    bool supportsLayout(const LayoutInfo& layout) const override {
        // Check for supported tile layouts (ND, NZ, etc.)
        ...
    }

    int maxRank() const override { return 8; }
};

}
```

---

## 6. Error Messages

### 6.1 Type Error Format

```
TypeError: <category>
  at <location>

  <description>

  Expected: <expected_type>
  Got:      <actual_type>

  Hint: <suggestion>
```

### 6.2 Example Errors

```python
# Layout incompatibility
TypeError: Layout incompatibility in kernel call 'attn_kernel'
  at attention.py:15, column 8

  Incompatible sharding at dimension 1:
  - Q has Shard(0)
  - K has Shard(1)

  Hint: Use relayout() to redistribute K, or change K's layout to match Q

# Type mismatch
TypeError: Argument type mismatch
  at attention.py:15, argument 'Q'

  Expected: Tensor[F16, (B, H, S, D), _]
  Got:      Tensor[F32, (B, H, S, D), _]

  Hint: Cast Q to F16 using cast(Q, DType.F16)

# Axis bounds
TypeError: Index out of bounds
  at moe.py:23, column 12

  Index 8 is out of bounds for Dense[8] (valid range: 0-7)

  Hint: Use DenseDyn for dynamic indexing
```

---

## 7. Summary

### 7.1 Type Hierarchy

```
Type
├── AxisType
│   ├── Dense[N]
│   ├── DenseDyn
│   ├── Ragged
│   └── Sparse
├── TensorType[D, S, L]
├── LayoutType
│   ├── DistElem (Replicate | Shard)
│   └── MemLayout
├── KernelType
│   └── ParamType (In | Out | InOut | Constexpr)
└── WorkloadType[Axes, Task, Deps]
    └── DependencyType
```

### 7.2 Type Checking Pipeline

```
Python Code
    │
    ▼
┌─────────────────────┐
│ L1: Builder Check   │  ← Immediate validation
│   - Kernel sigs     │
│   - Layout compat   │
│   - Axis bounds     │
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│ to_ir()             │  ← Convert to IR
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│ L2: IR Type Pass    │  ← Cross-workload checks
│   - Type inference  │
│   - Consistency     │
│   - Channel types   │
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│ L3: Backend Lower   │  ← Physical feasibility
│   - Memory limits   │
│   - Layout support  │
│   - DType support   │
└─────────────────────┘
    │
    ▼
Compiled Program
```

---

*Version: 1.0*
*Last Updated: 2026-01-25*
