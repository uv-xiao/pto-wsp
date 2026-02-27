# PTO‑WSP v10: Compiler / Codegen Pipeline (draft)

This doc captures v10’s compilation-pipeline direction discussed during v10 planning: **Python-driven**, **codegen-first**,
and backend-extensible, while keeping PTO‑WSP’s existing typed IR and making MLIR an **optional internal route** (not a rewrite).

## 0) Non-negotiables

- v10 semantics must be implemented by **compiled artifacts** (kernels + orchestration/schedule runtime payload), not by host
  Python “driving” execution.
- Regardless of backend target, v10 is **always codegen**: generate C++ (and then backend-specific binaries) and build an
  executable artifact for execution.
- “Reducing the Python↔C++ bridge” is a **compilation pipeline** goal (what Python does vs what artifacts do), not a “target
  backend” choice.

## 1) Workload-schedule vs “kernel”

v10 keeps the **workload-schedule pattern** as the common cross-backend programming model:

- workload construction and schedule directives are portable (typed IR surface),
- backends enforce (or record/diagnose) schedule knobs via a capability matrix.

“Kernel” in v10 is not a fixed decorator taxonomy. Kernel bodies are treated as **backend-specific intrinsic programs** authored
in Python, which are lowered by a backend emitter into C++/binaries.

Implication:
- keep the workload-schedule IR stable,
- make intrinsic authoring **flexible and backend-specific**, with a general mechanism to decide codegen/lowering.

## 1.1) Where this fits in the v10 architecture

For v10, treat the pipeline as the “Compilation” layer that sits between:

- **DSL layer** (Python authoring: workload–schedule + intrinsic programs), and
- **Emitter/Backend layer** (codegen + artifact packaging + toolchain build/run).

This is intentional decoupling: new DSL features or intrinsics should not force a rewrite of the backend packaging story,
and adding a new backend should not force a rewrite of the DSL.

## 2) Pipeline overview (conceptual)

```
Python authoring
  |  (capture source, parse AST)
  v
AST pass pipeline (rewrite + analysis)
  |  - constfold / inline / desugar
  |  - type inference (when possible)
  |  - intrinsic recognition + lowering decisions
  |  - region selection for optional MLIR islands
  v
Typed IR (existing C++ IR layer)
  |  - workload/schedule is common
  |  - intrinsic nodes reference builtin symbols + typed contracts
  v
Backend codegen + packaging (per target)
  |  - emit C++ + metadata into an artifact package/tree
  |  - compile toolchain outputs (or delegate to pto-runtime tooling)
  v
Execution (artifact runtime)
  |  - schedule/CSP semantics enforced here
  v
Stats (cycles, stalls, diagnostics)
```

## 3) Intrinsic / builtin mechanism (backend-extensible)

v10 needs a general “mark and lower” mechanism for intrinsic operations:

- A Python-visible symbol (function) represents an intrinsic.
- The intrinsic has a typed contract:
  - return type rule and (optional) input type constraints,
  - side-effect / memory-scope metadata if needed by scheduling/analysis.
- Backends register emitter handlers for intrinsics:
  - `intrinsic_id → emitter(codegen_ctx, typed_args, attrs) → C++/backend code`.

This avoids ad-hoc coupling to `@kernel` while still supporting existing v9 authoring styles as *sources* of intrinsic IR:
- traced `pto.*` kernels
- instruction-traced `ptoisa.*`
- file-based C++ escape hatches

## 4) Optional MLIR islands (AST ↔ MLIR bridge)

We delay any “rewrite the compiler into MLIR” decision, but we want the option to route some regions through a stronger IR
pipeline when beneficial.

Key constraints from v10 discussion:

- No user-facing region marker is required.
- Region selection is decided **dynamically by an AST pass** (heuristic).
- The pass should not impose hard constraints; it may apply optional heuristics (e.g., “fully typed boundary”) to increase
  success rate and determinism.

Conceptual flow:

1) AST pass identifies a candidate sub-tree (whole function/block first; sub-block fallback if needed).
2) Convert that AST region into MLIR (an internal dialect set).
3) Run MLIR passes (canonicalization, fusion, vectorization, etc.).
4) Lower back into the main pipeline as either:
   - a generated intrinsic/kernel (C++ emitted from MLIR), or
   - a typed IR fragment equivalent to the original region.

Fallback:
- If conversion/lowering fails, the pass leaves the region in the default AST→typed-IR lowering path.

## 5) Backend packaging is part of the backend

v10 treats “kernel/orchestration artifact packaging” as a backend responsibility that must be easily extended:

- each backend defines an artifact layout and manifest fields it consumes,
- the compiler produces a consistent *conceptual* package:
  - kernels (binaries or sources)
  - orchestration/schedule payload (CSP bodies, policy IDs, task_window config)
  - slots/symbol schema (runtime predicates)
  - capability declarations and diagnostics policy

For `pto_runtime_*` targets, the packaging contract is additionally constrained by `pto-runtime`:
- Phase 1: visible `host_build_graph`-shaped source tree + PTO‑WSP wraps pto-runtime tooling to build/run.
- Phase 2: versioned package aligned to pto-runtime task-buffer direction.

## 6) References (what to learn from)

### 6.1 PyPTO (python-driven pipeline + packaging)

PyPTO demonstrates a Python-driven compilation API that:
- runs a pass manager strategy,
- emits a **visible artifact directory**,
- supports multiple codegen backends (e.g., PTO vs CCE), and
- generates an explicit `kernel_config.py` describing orchestration + kernels + runtime config.

Local pointers:
- `docs/reference/18_pypto.md`

### 6.2 LittleKernel (Python AST as IR + intrinsic registry)

LittleKernel demonstrates a lightweight pipeline:
- capture Python source with `inspect.getsource`,
- apply AST passes,
- run type inference over the AST,
- emit C++/CUDA with a backend emitter,
- allow “intrinsics” to be registered with custom codegen behavior.

Local pointers:
- `references/triton-distributed-knowingnothing/python/little_kernel/core/compile.py`
- `references/triton-distributed-knowingnothing/python/little_kernel/codegen/codegen_cuda.py`

### 6.3 Arknife + Axe + TL (multi-backend / multi-level mapping references)

These references motivate v10’s separation of:
- explicit hardware/layout abstractions, and
- backend-specific emitters, and
- (optional) multi-target composition.

Local pointers:
- `docs/reference/19_arknife.md`
- `docs/reference/20_axe.md`
- `docs/reference/21_tl_spatial_compiler.md`

## 7) Open questions (track in v10)

- Intrinsic representation in typed IR: minimal stable node set vs “escape hatch” payload.
- How to express intrinsic side-effects and memory scopes for dependency inference.
- How to ensure deterministic region selection (AST pass config + logging/diagnostics).
- Minimal MLIR dialect set needed for islands (and where to lower back into C++).
- How to version the artifact manifest so backends evolve without breaking older artifacts.
