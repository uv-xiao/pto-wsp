# PTO‑WSP v10: Feature Catalog (draft)

This document is an at-a-glance catalog for v10. For rationale see `docs/future/v10_analysis.md`.

## Status vocabulary (v10 planning)

- **Must**: required for v10 completeness.
- **Should**: high-value if time permits; safe to slip if needed.
- **Candidate**: intentionally exploratory; document but do not block v10.

## 0) Positioning and layers (Must)

v10 positions PTO‑WSP as a **general DL programming/compilation framework** with explicit decoupling:

1) **DSL (Python)**: workload–schedule as the portable surface + backend-specific intrinsic programming.
2) **Compilation (Python AST + optional MLIR)**: pass infrastructure over source AST + internal AST↔MLIR bridge for
   pass-selected regions.
3) **Emitter/Backend**: backend registry + artifact packaging + toolchain build/run integration.

## A) Programming model

### A1) Workloads and kernels (Must)

- Declarative workload construction (workload–schedule pattern; existing v9 decorators/DSL remain valid authoring inputs)
- Kernel authoring is **intrinsic programming** (backend-specific) rather than a fixed `@kernel` taxonomy:
  - front-end exposes typed “builtins/intrinsics” callable from Python
  - lowering/codegen for intrinsics is selected by backend emitter(s)
  - the common contract is **codegen to a compiled artifact** (no Python execution engine fallback)
  - existing v9 kernel styles (`pto.*`, `ptoisa.*`, file-based C++ escape hatches) remain valid inputs, but v10 treats them
    as different *sources* for the same “intrinsic → emitter” mechanism

### A2) CSP pipeline model (Must)

- `Channel`, `process`, `send`, `consume`, `connect`, `replicate`
- CSP semantics are **artifact semantics**, not Python runtime semantics
- Cross-backend: CPU-sim + Ascend + AIE

### A3) Runtime predicates + tensor→scalar (Must)

- Explicit tensor→scalar materialization
- ScalarExpr coverage expanded (easy missing ops)
- Artifact slots/symbols as the single runtime “value injection” mechanism

## A4) Compilation layer features (Must)

- Source-AST pass pipeline as the primary transformation layer (desugar/constfold/inline/type inference where possible)
- Backend-extensible intrinsic registry + emitter hooks
- Optional MLIR islands:
  - a pass dynamically selects AST regions to route through `AST → MLIR → C++/artifact`
  - no user-facing marker required; constraints are optional heuristics

## B) Scheduling and execution (Must)

### B1) Core schedule semantics (Must)

- `dispatch(policy)` is behavior-changing across backends
- `task_window(TaskWindow(..., unit="tasks", mode=STALL))` is behavior-changing
- Additional schedule knobs:
  - **Must**: explicitly categorized as enforced / ignored / unsupported per backend
  - **Should**: expand enforcement beyond v9’s narrow subset where safe

### B2) Policy registry (Must)

- Scheduling policies are modular:
  - dispatch policy module
  - ready-queue policy module
  - start policy module
- Per-backend capability matrix and diagnostics

### B3) Semantic time model (Must)

- `total_cycles` is canonical semantic time:
  - CPU-sim/Ascend: PTO‑ISA cycle reports
  - AIE: AIE cycle reports (or equivalent timing source)
- CSPT semantics: makespan across process-local time + channel latency model

## C) Backend architecture (Must)

### C1) Runtime core (Must)

- Unified runtime core shared by backends:
  - orchestrator/scheduler/worker split (conceptual; threading may vary per backend)
  - task metadata window (bounded)
  - dependency tracking (tensor map + explicit CSP edges)
  - **multi-ring flow control** (task ring + dep pools + heap/buffer arena + ready queues)
  - **stall statistics + high-water marks** for all bounded resources
  - deadlock detection + diagnostics (actionable)
  - **target runtime**: align to the decoupled `pto-runtime` project (real device + a2a3sim parity)

### C2) NPU architecture model (Must)

- Arch model representation:
  - memory hierarchy
  - exec units and concurrency constraints
  - sync model
  - dispatch/launch model
  - cycle accounting hooks
- Concrete instances:
  - Ascend
  - AIE / AMD NPU-style dataflow accelerator

### C3) Backend targets (Must)

- `pto_runtime_a2a3sim`: pto-runtime simulation backend (AICPU/AICore semantics on host threads) (**reference simulation backend**)
- `pto_runtime_a2a3`: pto-runtime Ascend backend (real device execution; toolchain-gated)
- `aie`: runnable in AIE environment (hardware/emulator; emit-only fallback allowed locally)

Notes:
- v10 explicitly treats native `cpu_sim` as **v9 legacy**: it is not a v10 target backend and is planned to be removed once
  v9 semantics are validated on `pto_runtime_a2a3sim` (parity-gated deletion).

Bootstrap status (as of 2026-02-02):
- emit-only scaffold target exists: `target="a2a3sim_codegen"` (emits pto-runtime `host_build_graph`-shaped sources)
- runnable pto-runtime targets exist:
  - `target="pto_runtime_a2a3sim"` runs end-to-end in tests (CI-capable)
  - `target="pto_runtime_a2a3"` is wired and toolchain-gated (requires `ASCEND_HOME_PATH`)
  - the behavior is: emit visible source tree → PTO‑WSP wraps pto-runtime tooling to compile+run

## D) Tooling and quality (Should)

- Artifact introspection tooling (print plan, schedule summary, CSP graph summary)
- Graph/trace exporters:
  - task graph visualization
  - semantic-time traces (cycles)
- Deterministic replay / seed controls for testing

## E) Optional future candidates (Candidate)

- Auto mapping search hooks (Dato-like), without full global search in v10
- Rich stream capacities/typechecking beyond sync-only channels
- Multi-device/multi-host orchestration
