# PTO Workload-Schedule Programming (PTO-WSP) framework v9: In-Python NPU Function Design

> **Note**: This document describes the legacy NPU function builder API (`npu()`).
> For new code, use the **JIT Kernel API** (`@jit_kernel` with `tl.*` primitives) in `python/pto_wsp/kernel.py`.
> See `docs/features.md` Section 6 for the recommended approach.

## 1. Overview

This document specifies the in-Python NPU function builder API for v9. The design follows pto-isa-lh's `PTOFunctionBuilder` patterns while integrating with the outer Workload-Schedule model.

### 1.1 Design Goals

1. **Builder Pattern**: Fluent API like `PTOFunctionBuilder` for defining NPU kernels
2. **Two-Level Model**: Workload IR builds task graph; NPUFunction IR defines kernels
3. **Multi-Backend**: Same kernel IR targets CPU simulation, Ascend NPU, AMD AIE
4. **Separate Compilation**: Kernels are compilation units referenced by tasks (not inlined IR)
5. **Schedule Separation**: Functional IR (portable) + Schedule annotations (backend-specific)

### 1.2 Module Structure

```python
from pto.rt.npu import (
    # Module container
    NPUModule,

    # Kernel builder
    NPUFunctionBuilder,

    # Built kernel (immutable)
    NPUFunction,

    # Memory locations
    Location,  # Global, L2, UB, L1

    # Data types
    DType,  # F16, BF16, F32, etc.
)
```

---

## 2. NPU Module and Function Builder

### 2.1 NPUModule

Container for kernel definitions, similar to pto-isa-lh's `PTOModule`.

```python
class NPUModule:
    """Container for NPU kernel definitions."""

    def __init__(self, name: str):
        self.name = name
        self._functions: dict[str, NPUFunction] = {}
        self._kernel_id_counter = 0

    def register(self, func: NPUFunction) -> KernelHandle:
        """Register a function and return a handle for workload tasks."""
        kernel_id = self._kernel_id_counter
        self._kernel_id_counter += 1
        self._functions[func.name] = func
        return KernelHandle(kernel_id, func.name, func.signature, self)

    def kernel(self, name: str) -> KernelHandle:
        """Get a registered kernel by name."""
        func = self._functions[name]
        return KernelHandle(func.kernel_id, name, func.signature, self)

    def compile(self, target: str) -> CompiledModule:
        """Compile all kernels for a target backend."""
        ...
```

### 2.2 NPUFunctionBuilder

Fluent builder for NPU kernels, following pto-isa-lh's `PTOFunctionBuilder` style.

```python
class NPUFunctionBuilder:
    """Builder for NPU kernel IR."""

    def __init__(self, name: str, module: NPUModule | None = None):
        self.name = name
        self.module = module
        self._tiles: list[TileDecl] = []
        self._scalars: list[ScalarDecl] = []
        self._memrefs: list[MemrefDecl] = []
        self._ops: list[Op] = []
        self._schedule: ScheduleHints = ScheduleHints()

    # ============ Declarations ============

    def tile(self, name: str, rows: int, cols: int, dtype: DType,
             location: Location = Location.UB) -> "NPUFunctionBuilder":
        """Declare a local tile buffer."""
        self._tiles.append(TileDecl(name, rows, cols, dtype, location))
        return self

    def scalar(self, name: str, dtype: DType) -> "NPUFunctionBuilder":
        """Declare a scalar variable."""
        self._scalars.append(ScalarDecl(name, dtype))
        return self

    def scalar_li(self, name: str, value: int | float, dtype: DType) -> "NPUFunctionBuilder":
        """Declare a scalar with immediate value."""
        self._scalars.append(ScalarDecl(name, dtype, immediate=value))
        return self

    def memref(self, name: str, dtype: DType, location: Location = Location.Global,
               shape: tuple[int, ...] | None = None) -> "NPUFunctionBuilder":
        """Declare a memory reference (tensor pointer/view)."""
        self._memrefs.append(MemrefDecl(name, dtype, location, shape))
        return self

    # ============ Memory Operations ============

    def load(self, dst_tile: str, src_memref: str,
             row: int = 0, col: int = 0, *,
             async_: bool = False, tag: str | None = None) -> "NPUFunctionBuilder":
        """Load data from global memory to local tile."""
        self._ops.append(LoadOp(dst_tile, src_memref, row, col, async_, tag))
        return self

    def store(self, src_tile: str, dst_memref: str,
              row: int = 0, col: int = 0, *,
              async_: bool = False, tag: str | None = None) -> "NPUFunctionBuilder":
        """Store data from local tile to global memory."""
        self._ops.append(StoreOp(src_tile, dst_memref, row, col, async_, tag))
        return self

    def wait(self, tag: str) -> "NPUFunctionBuilder":
        """Wait for async DMA operation to complete."""
        self._ops.append(WaitOp(tag))
        return self

    # ============ Compute Operations ============

    # Elementwise
    def add(self, dst: str, a: str, b: str) -> "NPUFunctionBuilder":
        """Element-wise addition."""
        self._ops.append(BinaryOp("add", dst, a, b))
        return self

    def mul(self, dst: str, a: str, b: str) -> "NPUFunctionBuilder":
        """Element-wise multiplication."""
        self._ops.append(BinaryOp("mul", dst, a, b))
        return self

    def sub(self, dst: str, a: str, b: str) -> "NPUFunctionBuilder":
        """Element-wise subtraction."""
        self._ops.append(BinaryOp("sub", dst, a, b))
        return self

    def div(self, dst: str, a: str, b: str) -> "NPUFunctionBuilder":
        """Element-wise division."""
        self._ops.append(BinaryOp("div", dst, a, b))
        return self

    def exp(self, dst: str, src: str) -> "NPUFunctionBuilder":
        """Element-wise exponential."""
        self._ops.append(UnaryOp("exp", dst, src))
        return self

    def rsqrt(self, dst: str, src: str) -> "NPUFunctionBuilder":
        """Element-wise reciprocal square root."""
        self._ops.append(UnaryOp("rsqrt", dst, src))
        return self

    # Reductions
    def rowsum(self, dst: str, src: str) -> "NPUFunctionBuilder":
        """Sum reduction along rows."""
        self._ops.append(ReduceOp("rowsum", dst, src))
        return self

    def rowmax(self, dst: str, src: str) -> "NPUFunctionBuilder":
        """Max reduction along rows."""
        self._ops.append(ReduceOp("rowmax", dst, src))
        return self

    # Broadcast
    def rowexpandmul(self, dst: str, a: str, b: str) -> "NPUFunctionBuilder":
        """Broadcast multiply: dst[i,j] = a[i,j] * b[i]."""
        self._ops.append(BroadcastOp("rowexpandmul", dst, a, b))
        return self

    def rowexpandadd(self, dst: str, a: str, b: str) -> "NPUFunctionBuilder":
        """Broadcast add: dst[i,j] = a[i,j] + b[i]."""
        self._ops.append(BroadcastOp("rowexpandadd", dst, a, b))
        return self

    # Matrix operations
    def matmul(self, dst: str, a: str, b: str, acc: str | None = None, *,
               use_cube: bool = True) -> "NPUFunctionBuilder":
        """Matrix multiplication (optionally on cube unit)."""
        self._ops.append(MatmulOp(dst, a, b, acc, use_cube))
        return self

    # ============ Control Flow ============

    def for_loop(self, iv: str, lb: int, ub: int, step: int = 1,
                 **attrs) -> "NPUFunctionBuilder":
        """Begin a for loop."""
        self._ops.append(ForLoopBeginOp(iv, lb, ub, step, attrs))
        return self

    def end_for(self) -> "NPUFunctionBuilder":
        """End a for loop."""
        self._ops.append(ForLoopEndOp())
        return self

    def if_then(self, cond: str, **attrs) -> "NPUFunctionBuilder":
        """Begin if-then block."""
        self._ops.append(IfThenBeginOp(cond, attrs))
        return self

    def else_branch(self) -> "NPUFunctionBuilder":
        """Begin else block."""
        self._ops.append(ElseBranchOp())
        return self

    def end_if(self) -> "NPUFunctionBuilder":
        """End if-then-else block."""
        self._ops.append(IfThenEndOp())
        return self

    # ============ Composite Operations (Macros) ============

    def rmsnorm(self, out: str, x: str, weight: str, eps: float = 1e-6) -> "NPUFunctionBuilder":
        """RMS normalization (expands to primitives).

        out = x * rsqrt(mean(x^2) + eps) * weight
        """
        # Expand to primitive ops
        tmp_sq = f"_{out}_sq"
        tmp_mean = f"_{out}_mean"
        tmp_rsqrt = f"_{out}_rsqrt"
        tmp_norm = f"_{out}_norm"

        return (self
            .mul(tmp_sq, x, x)
            .rowsum(tmp_mean, tmp_sq)  # Note: should be rowmean
            .rsqrt(tmp_rsqrt, tmp_mean)
            .rowexpandmul(tmp_norm, x, tmp_rsqrt)
            .mul(out, tmp_norm, weight))

    def softmax(self, out: str, x: str) -> "NPUFunctionBuilder":
        """Softmax (expands to primitives).

        out = exp(x - max(x)) / sum(exp(x - max(x)))
        """
        tmp_max = f"_{out}_max"
        tmp_sub = f"_{out}_sub"
        tmp_exp = f"_{out}_exp"
        tmp_sum = f"_{out}_sum"

        return (self
            .rowmax(tmp_max, x)
            .rowexpandadd(tmp_sub, x, tmp_max)  # Note: should be sub
            .exp(tmp_exp, tmp_sub)
            .rowsum(tmp_sum, tmp_exp)
            .rowexpandmul(out, tmp_exp, tmp_sum))  # Note: should be div

    # ============ Schedule Directives ============

    def set_tile_policy(self, tm: int = 0, tn: int = 0, tk: int = 0,
                        **params) -> "NPUFunctionBuilder":
        """Set tile size policy (NPU-specific, ignored by CPU sim)."""
        self._schedule.tile_policy = {"tm": tm, "tn": tn, "tk": tk, **params}
        return self

    def double_buffer(self, tile_or_memref: str, depth: int = 2) -> "NPUFunctionBuilder":
        """Enable double buffering for a tile/memref."""
        self._schedule.double_buffer[tile_or_memref] = depth
        return self

    def pipeline(self, loop: str, stages: int = 2,
                 overlap: tuple[str, ...] = ("load", "compute", "store")) -> "NPUFunctionBuilder":
        """Enable software pipelining for a loop."""
        self._schedule.pipeline[loop] = {"stages": stages, "overlap": overlap}
        return self

    def hint_sram_budget(self, kb: int) -> "NPUFunctionBuilder":
        """Hint the SRAM budget for this kernel."""
        self._schedule.sram_budget_kb = kb
        return self

    def hint_cube(self, use_cube: bool = True) -> "NPUFunctionBuilder":
        """Hint whether to use cube unit (vs vector only)."""
        self._schedule.use_cube = use_cube
        return self

    # ============ Build ============

    def build(self) -> "NPUFunction":
        """Build and return immutable NPUFunction."""
        func = NPUFunction(
            name=self.name,
            tiles=self._tiles,
            scalars=self._scalars,
            memrefs=self._memrefs,
            ops=self._ops,
            schedule=self._schedule
        )

        # Auto-register with module if provided
        if self.module is not None:
            self.module.register(func)

        return func
```

---

## 3. NPUFunction and Kernel Handle

### 3.1 NPUFunction

Immutable kernel IR representation.

```python
@dataclass(frozen=True)
class NPUFunction:
    """Immutable NPU kernel IR."""
    name: str
    tiles: tuple[TileDecl, ...]
    scalars: tuple[ScalarDecl, ...]
    memrefs: tuple[MemrefDecl, ...]
    ops: tuple[Op, ...]
    schedule: ScheduleHints
    kernel_id: int = -1  # Assigned by module registration

    @property
    def signature(self) -> KernelSignature:
        """Return kernel signature for task binding."""
        inputs = [m for m in self.memrefs if m.is_input]
        outputs = [m for m in self.memrefs if m.is_output]
        return KernelSignature(
            params=[s.name for s in self.scalars],
            inputs=[m.name for m in inputs],
            outputs=[m.name for m in outputs]
        )
```

### 3.2 KernelHandle

Reference to a registered kernel, used in workload tasks.

```python
@dataclass(frozen=True)
class KernelHandle:
    """Handle to a registered kernel."""
    kernel_id: int
    name: str
    signature: KernelSignature
    module: NPUModule

    def task(self, **bindings) -> Workload:
        """Create a task workload invoking this kernel.

        Args:
            **bindings: Map from memref names to tensor regions

        Returns:
            Workload node representing a single task
        """
        # Validate bindings match signature
        for name in self.signature.inputs + self.signature.outputs:
            if name not in bindings:
                raise ValueError(f"Missing binding for '{name}'")

        # Create task with resolved resources
        resources = [bindings[name] for name in self.signature.inputs + self.signature.outputs]

        return Workload(
            kind="task",
            attrs={
                "kernel_handle": self,
                "kernel": self.name,
            },
            resources=resources
        )
```

---

## 4. Integration with Workload-Schedule Model

### 4.1 Two-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  Outer Level: Workload + Schedule (CPU/AICPU Orchestration)                  │
│                                                                              │
│    @workload                                                                 │
│    def attention(batch):                                                     │
│        for b in P(batch):                                                    │
│            attn_kernel[b](Q=Q[b], K=K[b], V=V[b], O=O[b])                   │
│                                                                              │
│    program = attention(batch).dispatch(policy).streams(4).compile()          │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │  Inner Level: NPUFunction (InCore Kernel)                                │ │
│  │                                                                          │ │
│  │    attn_kernel = (NPUFunctionBuilder("attn")                             │ │
│  │        .tile("q", 32, 128, DType.F32)                                    │ │
│  │        .tile("k", 128, 128, DType.F32)                                   │ │
│  │        ...                                                               │ │
│  │        .load("q", "Q").load("k", "K")                                    │ │
│  │        .matmul("s", "q", "k")                                            │ │
│  │        .softmax("p", "s")                                                │ │
│  │        .matmul("o", "p", "v")                                            │ │
│  │        .store("o", "O")                                                  │ │
│  │        .double_buffer("q", 2).pipeline("tile_loop", 2)                   │ │
│  │        .build())                                                         │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Compilation Flow

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   Python     │    │   Workload   │    │   Backend    │    │   Runtime    │
│   Builder    │───▸│     IR       │───▸│   Codegen    │───▸│   Program    │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
       │                   │                   │                   │
       │                   │                   │                   │
       ▼                   ▼                   ▼                   ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ NPUFunction  │    │  Task nodes  │    │  Kernel      │    │  Compiled    │
│   Builder    │───▸│  reference   │───▸│  compilation │───▸│  artifacts   │
│              │    │  KernelHandle│    │  + linking   │    │              │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
```

### 4.3 Example: Complete Workflow

```python
from pto_wsp import *
from pto.rt.npu import NPUModule, NPUFunctionBuilder, DType, Location

# 1. Define NPU kernels
mod = NPUModule("attention_ops")

# Flash attention tile kernel
flash_attn = (NPUFunctionBuilder("flash_attn_tile", module=mod)
    # Declarations
    .tile("q", 32, 128, DType.F32)
    .tile("k", 128, 128, DType.F32)
    .tile("v", 128, 128, DType.F32)
    .tile("s", 32, 128, DType.F32)
    .tile("p", 32, 128, DType.F32)
    .tile("o", 32, 128, DType.F32)
    .memref("Q", DType.F32, Location.Global)
    .memref("K", DType.F32, Location.Global)
    .memref("V", DType.F32, Location.Global)
    .memref("O", DType.F32, Location.Global)

    # Compute
    .load("q", "Q")
    .load("k", "K")
    .load("v", "V")
    .matmul("s", "q", "k", use_cube=True)
    .softmax("p", "s")
    .matmul("o", "p", "v", use_cube=True)
    .store("o", "O")

    # Schedule hints
    .double_buffer("q", 2)
    .hint_cube(True)
    .build())

# 2. Build workload using kernel reference
from pto_wsp import workload, P, DenseDyn, Dense, DispatchPolicy

batch = DenseDyn(batch_size)
heads = Dense[8]

# Get kernel reference from module
flash_attn_tile = mod.kernel("flash_attn_tile")

@workload
def attention(batch, heads):
    for b, h in P(batch, heads):
        flash_attn_tile[b, h](Q=Q[b,h], K=K[b], V=V[b], O=O[b,h])

# 3. Schedule and compile
program = (attention(batch, heads)
    .dispatch(DispatchPolicy.affinity(lambda t: t.get("batch")))
    .streams(2)
    .compile(target="ascend_npu"))

# 4. Execute
program.execute()
```

---

## 5. Backend Code Generation

### 5.1 CPU Simulation Backend

The CPU backend interprets NPUFunction IR directly or lowers to NumPy/PyTorch operations.

```python
class CPUKernelInterpreter:
    """Interpret NPUFunction as Python operations."""

    def interpret(self, func: NPUFunction, bindings: dict[str, np.ndarray]) -> None:
        tiles: dict[str, np.ndarray] = {}
        scalars: dict[str, Any] = {}

        # Allocate tiles
        for tile in func.tiles:
            tiles[tile.name] = np.zeros((tile.rows, tile.cols), dtype=tile.dtype.numpy)

        # Execute ops
        for op in func.ops:
            match op:
                case LoadOp(dst, src, row, col, _, _):
                    tiles[dst] = bindings[src][row:row+tiles[dst].shape[0],
                                               col:col+tiles[dst].shape[1]]
                case StoreOp(src, dst, row, col, _, _):
                    bindings[dst][row:row+tiles[src].shape[0],
                                  col:col+tiles[src].shape[1]] = tiles[src]
                case BinaryOp("add", dst, a, b):
                    tiles[dst] = tiles[a] + tiles[b]
                case BinaryOp("mul", dst, a, b):
                    tiles[dst] = tiles[a] * tiles[b]
                case MatmulOp(dst, a, b, acc, _):
                    tiles[dst] = tiles[a] @ tiles[b]
                    if acc:
                        tiles[dst] += tiles[acc]
                # ... other ops
```

### 5.2 NPU Backend

The NPU backend lowers NPUFunction to PTO-ISA operations and generates AICore code.

```python
class NPUKernelCodegen:
    """Generate Ascend C code from NPUFunction."""

    def lower(self, func: NPUFunction) -> PTOProgram:
        """Lower to PTO-ISA representation."""
        program = PTOProgram(func.name)

        for op in func.ops:
            match op:
                case LoadOp(dst, src, row, col, async_, tag):
                    program.add_instr(InstrLoadGM(
                        dst=self.resolve_tile(dst),
                        src=self.resolve_memref(src),
                        offset=(row, col),
                        async_=async_
                    ))
                case MatmulOp(dst, a, b, acc, use_cube):
                    if use_cube:
                        program.add_instr(InstrCubeGEMM(dst, a, b, acc))
                    else:
                        program.add_instr(InstrVecMatmul(dst, a, b, acc))
                # ... other ops

        return program

    def codegen(self, func: NPUFunction) -> str:
        """Generate Ascend C code."""
        pto_program = self.lower(func)
        return PTOCodegenAscend().emit(pto_program)
```

---

## 6. Key Design Decisions

### D6: NPU Function Integration (Resolved)

**Question**: How do NPU functions relate to outer workload tasks?

**Decision**: NPU functions are **separate compilation units** referenced by `KernelHandle` in tasks.

**Rationale**:
- Caching: Kernels compiled once, reused across tasks
- Scheduling: Per-kernel tuning without affecting task graph
- Clarity: Kernel-level vs graph-level passes have clear ownership
- Matches pto-isa-lh: InCore functions are definitions; orchestration builds task graph of calls

### NPU-Level Scheduling (Inner Schedule)

**Approach**: Two-layer IR with schedule annotations

1. **Functional IR**: load/store/compute/control flow (portable)
2. **Schedule Annotations**: tiling params, pipeline stages, buffering

**Implementation**:
- Tile sizes: Specialize kernel variants (`attn_32x128`, `attn_64x256`) or use symbolic shapes
- Double buffering: Directive lowers to async DMA + buffer swap
- Pipelining: Loop stage splitting with overlap specification

### Buffer Analysis

Integrate `TileBufferAnalyzer` pattern from pto-isa-lh:
- Track tile liveness (first write, last read)
- Detect buffer reuse opportunities
- Validate SRAM budget constraints
- Report memory pressure for autotuning

---

## 7. Summary

| Component | Purpose | Reference |
|-----------|---------|-----------|
| `NPUModule` | Container for kernel definitions | pto-isa-lh `PTOModule` |
| `NPUFunctionBuilder` | Fluent kernel builder | pto-isa-lh `PTOFunctionBuilder` |
| `NPUFunction` | Immutable kernel IR | pto-isa-lh `PTOProgram` |
| `KernelHandle` | Reference for workload tasks | New in v9 |
| `ScheduleHints` | Inner-level scheduling directives | v9 hierarchical model |

---

*Version: 9.1*
*Last Updated: 2026-01-24*
