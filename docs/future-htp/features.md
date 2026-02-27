# HTP (Heterogeneous Tile Programming) — Feature Catalog (WHAT)

## 0. Feature principles

- **Extensibility-first**: every major axis must be extensible (dialects, intrinsics, layout facets, passes, pipelines, backends, bindings).
- **Typed composition**: extension compatibility is checked via `requires/provides` capability typing plus layout/effect typing. 
<!-- Here, we need more typing for extension. For example, new dialects (e.g. WSP and CSP) for programming also introduce new typing. Meanwhile, we should have some basic typing, as those in CuTile about basic data types and tensor/tile elements. -->
- **Artifact-first**: compilation output is always a package with a manifest and inspectable intermediate dumps.
- **AST-first**: Python AST is the base IR; optional external compilation islands are explicit.
<!-- The external compilation should have extension mechanism. For example, an island MLIR pipeline may be defined with Python classes to specify matcher logic and bridge (bidirectional) logic (e.g., through MLIR's Python binding) -->

---

## 1. Programming entry (Python)

### 1.1 Kernel programming (tile-level)

<!-- @kernel is not something special. Instead, it should be an instance of the decarator-based programming extension mechanism. In other words, it shouldn't be hardcoded, but should be defined based on an extension framework (e.g., defining classes with bases). -->
- `@kernel` defines a tile kernel with a typed signature.
- Kernel bodies call intrinsics (portable or backend-specific) such as:
  - arithmetic/elementwise
  - loads/stores
  - asynchronous copies, barriers, events
- Kernel parameters can carry:
  - element dtype
  - tile shape
  - layout facets (distribution/memory/hardware constraints)

Rationale: kernels are the reusable unit for megakernels and pipelines.
<!-- Although kernels are reusable abstraction, they are not the minimal unit for extensions. -->

### 1.2 WSP: workload vs schedule programming

Two interlocked but separate layers:

- **Workload**: pure declarative definition of tasks and logical parallelism.
- **Schedule**: explicit, composable directives that constrain mapping, fusion, pipelining, buffering, and resource use.

Rationale: preserve the “workload first; schedule later” workflow across backends.

<!-- This should be a dialect to be plugged at the programming layer. -->

### 1.3 CSP: process/channel pipelines

- Processes with typed channels (bounded FIFO / rendezvous events).
- Static typing for:
  - stream element types
  - capacity/packing constraints (backend-dependent)
  - linear/affine usage to prevent deadlock / mismatch (Dato-style).

Rationale: pipelines are the natural representation for megakernels and serving routines.

<!-- This should also be a dialect to be extended. -->

### 1.4 Serving routine programming (host orchestration)

- A “routine” is a program that composes:
  - backend-built kernels/megakernels
  - data movement
  - CSP pipelines
  - runtime dispatch decisions (e.g., batch splitting, dynamic shapes)

Rationale: serving is “things above kernels” and must be in scope.

<!-- This don't need to be a standalone dialect/extension. Instead, it can be represented with basic + WSP + CSP. So this should be a case study. -->
---

## 2. Layout system (unified, multi-facet)

### 2.1 Distribution facet (Dato/Axe)

- Per-dimension distribution elements:
  - Shard on a mesh axis
  - Replicate
- Join/compatibility rules and explicit relayout operations.
- Effect tracking for collectives (e.g., pending reductions that must be discharged).

### 2.2 Memory facet (Triton/CuTe)

- Strides/order/swizzle/pack as a physical memory description.
- Composition operations (tile, reshape, permute) that remain explicit.

### 2.3 Hardware facet (Arknife-style)

- Explicit hardware hierarchy:
  - parallel levels (grid/block/warp/tile or backend equivalents)
  - memory spaces with capacity/alignment constraints

Rationale: backends differ; layout must bridge distributed ↔ on-device ↔ runtime constraints.

<!-- We need to think about how we integrate a specific layout with the program. We just use the Python's typing system and add type inference/checking for a specific layout as passes in the pipeline? But how should we easily move to another backend? For example, we can definitely build different pipelines to compile program of Axe layout targetting GPU and Ascend NPU. But if we want to switch the program of Axe to program of Arknife to target Qualcomm Hexagon NPU, can we do this automatically? This is about a unified defination of different layouts and their automatic conversion, which is very interesting. If we cannot achieve this, we should make best preparation for this at the code architecture level to leave the challenge for future solving. -->

---

## 3. Intrinsics & dialect libraries

### 3.1 Intrinsic sets with typed contracts

Intrinsics are declared with:

- collection naming
- semantic meaning (e.g., “vector add” on tiles),
- layout constraints (required facets),
- backend handler availability (which backend provides lowering/emitters).

### 3.2 Dialects as pluggable semantic layers

Dialects are packages that:

- define AST constructs / decorators / helper APIs,
- define typing rules / legality,
<!-- - define canonicalization/lowering passes into core representations. This is not correct. We don't have such a thing as the core representations. -->
<!-- - instead: define their own data structure (Python classes) which should be built and maintained for analysis during mutator passes. -->

Rationale: CSP and WSP should be optional and independently evolvable.

---

## 4. Pass system and compiler pipeline

### 4.1 AST passes (primary)

- Match/apply on AST with attached metadata.
- Each pass declares `requires/provides` capabilities and invariants.

### 4.2 External compilation islands (optional)

- A pass can lower a region into:
  - MLIR module(s) + dialects
  - external toolchain invocations
- The pass also defines how artifacts rejoin the main pipeline (manifested outputs).

### 4.3 Pipeline selection via capability typing

- A “target backend” selects a **pipeline template**.
- The pipeline is instantiated only if:
  - required dialects are present,
  - intrinsic handlers exist,
  - layout/effect typing is satisfied.

Rationale: extensibility without “if backend == …” branching.

---

## 5. Backends and artifacts

### 5.1 Backend abstraction

A backend defines:

- hardware model + supported layout facets,
- accepted dialects and required lowerings,
- artifact packaging contract,
- binding interface (build/run/simulate).

### 5.2 Artifact-first packaging

Each compile emits:

- `manifest.json` (or equivalent) describing:
  - inputs, versions, enabled dialects, pipeline, pass list
  - backend target and hardware model
  - produced artifact set and entrypoints
- `ir/` dumps and pass snapshots
- `codegen/` backend outputs

Rationale: reproducibility, debugging, and stable runtime integration.

---

## 6. “Must-support” targets (initial)

### 6.1 Ascend PTO ecosystem

- Emit artifacts consumable by a PTO runtime/toolchain (simulation and device).
- Packaging should mirror proven patterns (kernel/orchestration separation + manifest).

### 6.2 AIE via MLIR-AIE

- Emit MLIR-AIE oriented artifacts:
  - compute tile mapping, FIFO/stream wiring, host runtime glue
- Reuse known mapping concepts (kernel grid mapping + layout annotations).

---

## 7. Debuggability and introspection

- Standard pass tracing (“before/after” dumps).
- Type-check diagnostics:
  - layout incompatibility
  - stream protocol mismatches
  - missing backend handlers/capabilities
- Optional execution tracing hooks in bindings.

---

## 8. Deep dives index

Feature deep dives live in:

- `docs/future-htp/feats/01_extensibility.md`
- `docs/future-htp/feats/02_dialects_wsp.md`
- `docs/future-htp/feats/03_dialects_csp.md`
- `docs/future-htp/feats/04_intrinsics.md`
- `docs/future-htp/feats/05_layout.md`
- `docs/future-htp/feats/06_passes_pipelines.md`
- `docs/future-htp/feats/07_backends_artifacts.md`
- `docs/future-htp/feats/08_binding_runtime.md`
- `docs/future-htp/feats/09_debuggability.md`

