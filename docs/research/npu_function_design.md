## 0) Context from `pto-isa-lh`
The pattern you want already exists there as:
- **Kernel IR + builder**: `PTOFunctionBuilder` builds a `PTOProgram` with `.tile/.scalar/.memref`, `.load/.store`, compute ops, and `.for_loop/.if_then` (`references/pto-isa-lh/src/compile/pto_compile.py`).
- **Kernel-module linkage**: `PTOModule` is a symbol table for functions (`references/pto-isa-lh/src/compile/pto_compile_common.py`).
- **Outer scheduling ↔ kernels**: orchestration “calls” become **tasks** with I/O regions (`gen_task_scheduling_code` in `references/pto-isa-lh/src/compile/pto_codegen_arm64.py`), and there’s buffer sizing (`TileBufferAnalyzer` in `references/pto-isa-lh/src/compile/pto_compile_common.py`).
- **Examples** show the intended split: small InCore kernels + an outer orchestration layer (e.g. `references/pto-isa-lh/examples/bgemm/pto_bgemm.py`, `references/pto-isa-lh/examples/llama/pto_llama7B_dynamic.py`).

Your `pto_wsp` repo already has the *outer* concept (Workload/Task, kernel registry stubs, scheduling policies), so the clean design is: **Workload IR builds the task graph; NPUFunction IR defines kernels.**

---

## 1) Proposed API surface (Python)
### 1.1 Kernel IR objects
- `NPUModule(name)` — holds kernel definitions + optional library macros.
- `NPUFunction` — immutable kernel IR (declarations + ops + schedule hints).
- `NPUFunctionBuilder(name, module=...)` — fluent builder, PTOFunctionBuilder-style.

### 1.2 Core builder methods (match PTOFunctionBuilder feel)
Declarations:
- `.tile(name, rows, cols, dtype, location=UB)`  (UB/local tile)
- `.scalar(name, dtype)` / `.scalar_li(name, value, dtype)`
- `.memref(name, dtype, location=Global, shape=None)` (tensor pointer/view)

Memory:
- `.load(dst_tile, src_memref, row=0, col=0, *, async=False, tag=None)`
- `.store(src_tile, dst_memref, row=0, col=0, *, async=False, tag=None)`
- optional: `.wait(tag)` (for async DMA backends)

Compute (primitive “lowerable” ops):
- elementwise: `.add/.mul/.sub/.div/.exp/.rsqrt/...`
- reductions/broadcast: `.rowsum/.rowmax/.rowexpandmul/.rowexpandadd/...`
- matmul: `.matmul(dst, a, b, acc=None, *, use_cube=True)`
- composites as **macros** (recommended): `.rmsnorm(...)` expands to primitives unless backend has a native op

Control flow:
- `.for_loop(iv, lb, ub, step=1, **attrs)` + `.end_for()`
- `.if_then(cond, **attrs)` + `.else_branch()` + `.end_if()`

Scheduling directives (NPU-only semantics, ignored by CPU sim):
- `.set_tile_policy(**params)` (e.g., `tm/tn/tk`, vector lanes alignment)
- `.double_buffer(memref_or_tile, depth=2)`
- `.pipeline(loop=iv, stages=2, overlap=("load","compute","store"))`
- `.hint_sram_budget(kb=256)` / `.hint_cube(True)` etc.

Build:
- `.build() -> NPUFunction` (and auto-attach to module if provided)

### 1.3 Workload integration ergonomics
Make kernels directly usable as tasks:
- `kernel = module.kernel("tile_add")` (returns `KernelHandle`)
- `kernel.task(**bindings) -> Workload` (returns `Workload("task", ...)` with correct param/resource ordering)

Example (shape intentionally PTOFunctionBuilder-like):
```python
mod = NPUModule("ops")

tile_add = (NPUFunctionBuilder("tile_add", module=mod)
  .tile("a", 64, 128, DType.F32).tile("b", 64, 128, DType.F32).tile("c", 64, 128, DType.F32)
  .memref("A", DType.F32, Location.Global).memref("B", DType.F32, Location.Global).memref("C", DType.F32, Location.Global)
  .load("a", "A").load("b", "B").add("c", "a", "b").store("c", "C")
  .build())

work = parallel_for(num_tiles, lambda t:
  mod.kernel("tile_add").task(A=A.tile(t), B=B.tile(t), C=C.tile(t)))
```

---

## 2) Codegen model (CPU sim + NPU)
### 2.1 Single IR, multiple lowerings
- **CPU sim backend**: interpret or lower `NPUFunction` to a Python callable (NumPy/Torch), then run the task graph scheduler in-process.
- **NPU backend**: lower `NPUFunction` to a backend dialect (e.g., PTO-ISA-ish ops) + emit:
  1) per-kernel AICore wrapper/code
  2) an outer “graph/orchestration” artifact that instantiates tasks + deps

### 2.2 Keep composites as macros unless proven necessary
Implement `rmsnorm/softmax/...` as library functions that *emit primitive ops into the builder* (exactly like `pto-isa-lh` builds composed kernels from level-1 tile ops). This keeps every backend smaller: only primitives need real lowering.

---

## 3) Key questions (recommended answers)

### Q1: How should NPU functions register with outer workload tasks?
Use a **kernel registry that returns stable handles**, not raw strings.
- `NPUModule` (or a global `KernelRegistry`) assigns each `NPUFunction` a `KernelId` and stores:
  - name, signature (params/resources), input/output indices (for dependency inference), schedule metadata, and per-backend artifacts.
- `Workload.task(...)` stores `KernelHandle` (or `KernelId`) plus bound `TensorRegion`s.
- At `Workload.compile(target=...)`, resolve `KernelHandle → (compile if needed) → artifact`, then build the runtime task graph.

This matches the intent in your `docs/spec.md` “Kernel Registration” section and avoids “stringly-typed” linkage errors.

### Q2: Should NPU functions be first-class IR nodes or separate compilation units?
Make them **separate compilation units (kernel definitions)**, referenced by **first-class task nodes** in the workload IR.
- Pros: caching, reuse, per-kernel scheduling/tuning, clearer ownership (kernel-level vs graph-level passes).
- Still allow optional *late* fusion/inlining as an optimization pass (kernel fusion or “megakernel” mode), but don’t require it structurally.

This mirrors `pto-isa-lh`: InCore functions are definitions; orchestration builds a task graph of calls.

### Q3: How to handle NPU-level scheduling (tile sizes, double buffering)?
Treat scheduling as a **separate “schedule dialect” attached to the kernel IR**, with two layers:
1) **Functional IR**: load/store/compute/control flow (portable).
2) **Schedule annotations**: tiling params, pipeline stages, buffering, memory level placement.

Implementation strategy:
- Tile sizes: either (a) **specialize** multiple kernel variants (`tile_add_32/64/128` like `pto-isa-lh` does) and let the workload choose, or (b) keep one kernel with symbolic shapes + backend specialization.
- Double buffering: model as either:
  - explicit async DMA ops (`load_async`, `wait`, `swap`) in IR, or
  - a directive (`.double_buffer(...)`) that the NPU backend lowers into explicit buffers + async copies.
- Validate with a buffer analyzer (like `TileBufferAnalyzer`) to enforce UB/SRAM budgets and to decide when reuse/double-buffering is legal.

If you want, I can sketch the concrete class/type definitions (`NPUFunction`, `Op`, `TensorRegion`, `KernelHandle`, backend interfaces) in a way that drops into `python/pto_wsp/` cleanly and matches `docs/spec.md`/`docs/backend-arch.md`.