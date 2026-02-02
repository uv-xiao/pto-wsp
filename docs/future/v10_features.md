# PTO‑WSP v10: Feature Catalog (draft)

This document is an at-a-glance catalog for v10. For rationale see `docs/future/v10_analysis.md`.

## Status vocabulary (v10 planning)

- **Must**: required for v10 completeness.
- **Should**: high-value if time permits; safe to slip if needed.
- **Candidate**: intentionally exploratory; document but do not block v10.

## A) Programming model

### A1) Workloads and kernels (Must)

- Declarative workload construction (`@workload`, `P`, composition primitives)
- Typed kernel authoring:
  - `pto.*` traced kernels (lowered to backend codegen)
  - `ptoisa.*` instruction-traced kernels
  - file-based C++ PTO‑ISA kernels (escape hatch)

### A2) CSP pipeline model (Must)

- `Channel`, `process`, `send`, `consume`, `connect`, `replicate`
- CSP semantics are **artifact semantics**, not Python runtime semantics
- Cross-backend: CPU-sim + Ascend + AIE

### A3) Runtime predicates + tensor→scalar (Must)

- Explicit tensor→scalar materialization
- ScalarExpr coverage expanded (easy missing ops)
- Artifact slots/symbols as the single runtime “value injection” mechanism

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

- `cpu_sim`: implemented via `pto-runtime` `a2a3sim` (host-thread simulation of AICPU/AICore)
- `ascend_npu`: runnable in CANN environment (not emit-only), via `pto-runtime` `a2a3`
- `aie`: runnable in AIE environment (hardware/emulator; emit-only fallback allowed locally)

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
