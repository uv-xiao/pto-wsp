# v10: PTO‑WSP ↔ pto-runtime Interface Contract (draft)

This document is the **interface/spec checkpoint** for v10. It exists because v10’s success depends on a clean boundary between:

- **PTO‑WSP**: workload + schedule programming model, typed IR, codegen, semantic contracts
- **pto-runtime**: execution substrate (host runtime + AICPU scheduler + AICore kernels), with sim parity (`a2a3sim`)

The goal is to avoid “gluing until it works” and instead define a small, versioned contract.

## 1) Layer boundaries (non-negotiable)

### 1.1 PTO‑WSP responsibilities

PTO‑WSP owns:

- **Workload/Schedule IR** semantics:
  - `dispatch(policy)` is behavior-changing (intended as AICPU scheduler assignment; see §5 hazards)
  - `task_window(stall-only)` is behavior-changing
  - CSP/CSPT semantics are artifact/runtime semantics
- **C++ codegen boundary**:
  - Phase 1: emit a pto-runtime-compatible `host_build_graph` source tree and build an orchestration `.so`
  - Phase 2: emit a compact plan + expander payload aligned to pto-runtime task-buffer roadmap
- **Python integration layer**:
  - imports pto-runtime tooling (builder/compiler) for building/running generated artifacts
  - must not become the execution engine for schedule/CSP semantics
- **Kernel programming**:
  - `pto.*` (high-level) lowering to PTO‑ISA tile kernels
  - `ptoisa.*` (instruction-traced) lowering to PTO‑ISA tile kernels
  - file-based custom kernels (escape hatch)
- **Predicate/slots/symbols semantics**:
  - tensor→scalar materialization boundaries
  - slots/symbol tables updated between runs without rebuilding kernels

PTO‑WSP must not push these semantics into Python host execution “as a convenience”.

### 1.2 pto-runtime responsibilities

pto-runtime owns:

- **Execution substrate**:
  - device memory allocation and H2D/D2H
  - registering/loading kernel binaries
  - launching the AICPU scheduler + AICore worker code
  - simulation parity via `a2a3sim` (host threads, executable mmap)
- **Resource bounding mechanisms** (task-buffer direction):
  - bounded task tables / buffers
  - bounded dep structures / arenas
  - backpressure + diagnostics

pto-runtime should not embed PTO‑WSP’s scheduling policies as ad-hoc special cases; it should execute policy IDs/bytecode/data
provided by PTO‑WSP artifacts.

## 2) What “integration” means

For PTO‑WSP, “integrate with pto-runtime” means:

- PTO‑WSP emits a **package** (binaries + metadata)
- pto-runtime executes it on:
  - **`a2a3sim`** for local semantics testing
  - **`a2a3`** for real Ascend devices

No “emit-only” backend is acceptable for v10 completeness; `emit-only` is allowed only as a degraded experience when toolchains
are absent locally.

## 3) Artifact package shape (v10 contract)

PTO‑WSP v10 should define a stable “package” layout that pto-runtime can consume.

Minimum contents:

0) **Manifest**
   - machine-readable manifest (`.json`/`.toml`) describing:
     - `wsp_runtime_abi` (string)
     - `target` (`a2a3` / `a2a3sim` / future)
     - kernel registry entries
     - schedule/CSP payload locations
     - slot/symbol schema
     - entry payload type (Phase 1 `.so` vs Phase 2 expander payload)

1) **Kernel registry payload**
   - mapping `kernel_id → binary` per executor type (AIC/AIV, possibly DMA later)
   - for `a2a3sim`, binaries are host-executable `.text` blobs or `.so` + symbol name (choose one and standardize)
   - for `a2a3`, binaries are the device-compatible kernel images expected by pto-runtime

2) **Schedule/runtime payload**
   - policy IDs and parameters (dispatch policy, etc.)
   - task_window config (size, overflow=STALL, unit=tasks)
    - CSP channel definitions (IDs, capacity, latency model)
   - slot/symbol table schema (IDs, widths, pointer vs u64)

3) **Entry payload**
   - either:
     - a host orchestration `.so` (Phase 1), or
     - an AICPU expander/scheduler payload consuming a compact plan (Phase 2)

The package must carry an explicit ABI version, e.g. `wsp_runtime_abi = "v10.0"`.

### 3.1 ABI versioning and compatibility rules (required)

v10 requires explicit compatibility rules:

- A package with unknown `wsp_runtime_abi` must fail fast with a clear error.
- v10 may allow “minor” additions (extra optional fields) while keeping “major” incompatible schema changes gated by a new ABI string.
- Slots/symbol IDs must be stable across rebuilds (hash-based IDs preferred) so “update between runs” remains valid.

## 4) Phase compatibility (status vs v10 target)

pto-runtime’s **current** “host_build_graph” path and the **future** “task-buffer” direction are not the same interface.
To keep v10 realistic but correct, we explicitly define phases.

### Phase 1 (compatibility; runnable backend fast)

Mechanism:
- PTO‑WSP generates a **host orchestration `.so`** that:
  - allocates tensors via pto-runtime host APIs,
  - builds a static task graph via pto-runtime’s `Runtime` object (add_task/add_successor style),
  - configures per-task placement (`core_type` / executor id).

Pros:
- gets a runnable Ascend backend quickly via existing pto-runtime APIs
- enables local semantics runs via `a2a3sim`

Cons / must-not-lie:
- does **not** provide PTO‑WSP’s v9 “on-device task expansion” thesis
- does **not** provide true `task_window` *backpressure to orchestration*, because orchestration runs to completion to build
  the full graph
- cannot represent unbounded/dynamic workloads without precomputing the full graph

Therefore Phase 1 is acceptable only as an intermediate state; v10 completeness requires Phase 2.

### Phase 2 (v10 target; true semantics)

Mechanism:
- PTO‑WSP generates a compact plan + runtime symbols/slots, and an AICPU-side expander/scheduler that:
  - expands tasks incrementally into a bounded task-buffer,
  - enforces `task_window` stall-only semantics as real backpressure,
  - integrates CSP edges (channel waits/signals) into readiness rules,
  - exposes actionable diagnostics when bounded resources stall/deadlock.

This phase aligns with:
- PTO‑WSP v9 direction (on-device expansion, dynamic axes, slot-driven predicates), and
- pto-runtime’s “task-buffer” roadmap (`docs/future/pto-runtime-task-buffer.md`).

## 5) Key interface hazards to avoid

1) **Graph-vs-buffer semantic drift**
   - A static task graph builder is not equivalent to a streaming task-buffer orchestrator; don’t claim task_window works if
     it only limits scheduling, not orchestration.

2) **Dispatch semantic drift (AICPU scheduler assignment vs `core_type`)**
   - PTO‑WSP `dispatch(policy)` is intended to decide **which AICPU scheduler instance** issues a task.
   - If pto-runtime cannot express/execute this yet, PTO‑WSP must treat it as **recorded/diagnosed** and must not claim it is
     enforced. Track the gap explicitly in `docs/future/pto_runtime_gaps.md`.

3) **Policy leakage**
   - Avoid “if policy == round_robin do X” hard-coded into pto-runtime; policies should be registered modules or IDs with a
     stable evaluation protocol.

4) **Cycles/time accounting mismatch**
   - PTO‑WSP canonical time is kernel cycle reports; pto-runtime must surface them without reinterpretation.

5) **Slots/symbol ABI instability**
   - Make symbol IDs stable (hash) and slots indexed; define update rules clearly.

6) **Over-coupling build systems**
   - Prefer submodule + adapter layer over copying sources or importing Python modules across repos without versioning.

## 6) Concrete pto-runtime touchpoints (Phase 1 reality check)

Phase 1 integration must be written against real pto-runtime APIs; these are the practical anchor points:

- Orchestration ABI (host_build_graph): `OrchestrationFunc(Runtime*, uint64_t* args, int arg_count)`
  - runtime loads orchestration code via `dlopen` and calls the named symbol.
- Host runtime entrypoints:
  - init/build graph (orchestration call) and finalize/validate path
  - device allocation and copy APIs exposed via the runtime’s host API table

Reference files (read-only):
- `references/pto-runtime/src/runtime/host_build_graph/runtime/runtime.h`
- `references/pto-runtime/src/runtime/host_build_graph/host/runtime_maker.cpp`

## 7) Next doc links

- pto-runtime status: `docs/future/pto_runtime_analysis.md`
- pto-runtime integration overview: `docs/future/pto_runtime_integration.md`
- task-buffer preview/reference: `docs/future/pto-runtime-task-buffer.md`
- v10 spec: `docs/future/v10_spec.md`
