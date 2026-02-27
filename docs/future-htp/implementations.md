# HTP (Heterogeneous Tile Programming) — Architecture (HOW)

## 0. Architectural snapshot

HTP is a Python-driven compiler framework with:

- **AST-first IR**: Python AST + attached typed metadata.
- **Capability typing**: every pass/pipeline/backend declares `requires/provides`.
- **Unified layout typing**: distribution + memory + hardware facets.
- **Artifact-first output**: compilation emits a package with a manifest and dumps.
- **Backend plugins**: codegen emitters + bindings are backend-registered.

This file describes components and data flow; implementation deep dives are in `docs/future-htp/impls/`.

Related design references already captured in this repo:

- Artifact packaging patterns: `docs/reference/18_pypto.md`
- Hardware abstraction (hierarchy + memory spaces): `docs/reference/19_arknife.md`
- Unified layout across layers: `docs/reference/20_axe.md`
- Stream/layout typing and mapping concepts: `docs/reference/16_dato.md`
- AIE mapping + FIFOs reference shape: `docs/reference/15_allo.md`

---

## 1. Top-level data flow

1. **Parse & capture**: the Python entrypoints (kernels/workloads/routines) are captured as AST + symbol tables.
2. **Dialect expansion**: dialect-level syntactic sugar expands into canonical AST forms and attaches metadata.
3. **Type/effect checking**: layout facets + stream effects are validated; required collectives/effects are made explicit.
4. **Pipeline selection**: given `target_backend`, choose a pipeline whose `requires` are satisfied by the program’s
   dialects/intrinsics/layout capabilities.
5. **Lowering and codegen**:
   - AST passes perform canonicalization, scheduling, partitioning, and lowering into codegen-ready forms.
   - Optional islands lower regions into MLIR or other toolchains.
6. **Emit artifact package**: a stable directory tree with manifest, IR dumps, codegen outputs, and build/run metadata.
7. **Bind**: backend binding builds/loads/executes (or returns a build recipe) from the package.

---

## 2. Core component model

### 2.1 Program model

A **Program** (design-time concept) includes:

- entrypoints:
  - kernels (device functions)
  - workloads (task graphs)
  - routines (host orchestration)
- enabled dialects (e.g., WSP, CSP)
- imported intrinsic sets (portable + backend-specific)
- layout declarations/annotations
- target selection (backend + hardware profile)

### 2.2 AST representation + metadata

HTP treats AST as the canonical source and attaches metadata:

- type info:
  - scalar/tile dtypes and shapes
  - layout facets (distribution/memory/hardware)
- schedule directives:
  - fusion hints, pipelining stages, buffering, resource constraints
- lowering state:
  - normalized forms, explicit collectives, explicit stream ops

Metadata must be:

- serializable into artifact dumps (for reproducibility),
- stable under minor AST reshaping (keyed by node identity + symbol path).

### 2.3 Dialect packages

Each dialect package includes:

- Python surface API (decorators, helper functions)
- AST patterns and canonical forms
- typing rules (layout + effects)
- canonicalization/lowering passes
- declared capabilities (what this dialect provides)

### 2.4 Intrinsic sets

An intrinsic set is:

- a typed API surface (Python calls) with contracts,
- a registry of backend handlers:
  - lowering rules (AST → backend form)
  - emitter code (backend form → artifact files)

Intrinsic contracts are the key compatibility boundary:

- required layout facets
- permitted memory spaces
- required scheduling constraints

### 2.5 Layout & effect typing

HTP should separate:

- **Layout values** attached to tensors/tiles.
- **Typing rules**:
  - distribution join/compatibility and explicit relayout
  - memory layout legality for vectorization/packing
  - hardware placement constraints (e.g., “this buffer must be in UB”)
- **Effects**:
  - stream effects (linear/affine usage)
  - collective effects (pending reductions, barriers)

### 2.6 Passes

A pass is a pure(ish) transform with a contract:

- `requires`: capabilities that must exist beforehand
- `provides`: capabilities established after it runs
- invariants:
  - AST shape invariants
  - type invariants (e.g., “all distribution layouts normalized”)
  - effect invariants (e.g., “no pending collectives”)

Passes come in classes:

- AST passes (primary)
- IR-island passes (enter/exit MLIR or external pipelines)
- packaging passes (manifest, file layout, dependency closure)

### 2.7 Pipelines

A pipeline is:

- a target-specific pass list with parameters,
- a declared output artifact contract,
- a binding requirement (which binding can run this package).

Pipeline selection is constraint solving:

- program capabilities + target backend requirements must satisfy the pipeline’s `requires`.

### 2.8 Backends and bindings

Backends provide:

- hardware model definition(s)
- supported dialects/intrinsics
- codegen emitters
- package contract

Bindings provide:

- build toolchain integration (compile/assemble/package)
- load/execute/simulate integration
- trace and diagnostics hooks

---

## 3. Artifact package contract (recommended baseline)

Directory layout (illustrative):

```
<out>/
  manifest.json
  ir/
    ast_original.pyast.json
    ast_canonical.pyast.json
    pass_trace.jsonl
  codegen/
    <backend>/
      ... backend-specific outputs ...
  build/
    ... build recipes / toolchain metadata ...
  logs/
    ... optional build/run logs ...
```

The manifest must include:

- compilation identity (versions, git hashes, environment capture)
- enabled dialects and intrinsic sets
- pipeline list and pass parameters
- backend target and hardware profile
- entrypoints (kernel/routine symbols)
- produced artifacts with paths and semantics

Deep dive: `docs/future-htp/impls/04_artifact_manifest.md`.

---

## 4. Target-specific packaging sketches

### 4.1 PTO / Ascend backend

Recommended shape (mirrors proven kernel/orchestration separation patterns):

```
codegen/pto/
  kernels/
    ... kernel sources / pto binaries ...
  orchestration/
    ... host orchestration code ...
  package_manifest.json
  kernel_registry.json
```

Key design point: “package manifests” are the stable integration boundary with runtime tooling.

Deep dive: `docs/future-htp/impls/05_backend_pto.md`.

### 4.2 AIE backend (MLIR-AIE)

Recommended shape:

```
codegen/aie/
  aie.mlir
  host.cpp (or host.mlir)
  mapping.json
  fifos.json
  build.sh / cmake/ (optional)
```

Deep dive: `docs/future-htp/impls/06_backend_aie.md`.

---

## 5. Cross-cutting concerns

### 5.1 Diagnostics

- “capability mismatch”: pipeline requires missing dialect/intrinsic/layout capability.
- “layout incompatibility”: distribution join failures; missing relayout.
- “effect mismatch”: stream protocol mismatch; pending collectives not discharged.
- “backend handler missing”: intrinsic lacks lowering/emitter for backend.

### 5.2 Testing strategy (docs-only requirement)

- Golden artifact tests: compile example → compare manifest + key emitted files.
- Type-check tests: known-bad programs produce stable diagnostics.
- Backend contract tests: package validates against binding expectations.

Deep dive: `docs/future-htp/impls/08_testing.md`.

---

## 6. Deep dives index

Implementation deep dives live in:

- `docs/future-htp/impls/01_ir_model.md`
- `docs/future-htp/impls/02_pass_manager.md`
- `docs/future-htp/impls/03_capability_solver.md`
- `docs/future-htp/impls/04_artifact_manifest.md`
- `docs/future-htp/impls/05_backend_pto.md`
- `docs/future-htp/impls/06_backend_aie.md`
- `docs/future-htp/impls/07_binding_interface.md`
- `docs/future-htp/impls/08_testing.md`
- `docs/future-htp/impls/09_transition_plan.md`
