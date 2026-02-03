# PTO‑WSP v10 ↔ pto-runtime: Known Gaps / Missing Features

This document lists the **pto-runtime features PTO‑WSP v10 needs** that are **not yet available** (or not exposed cleanly)
in pto-runtime today.

It is a living checklist used to keep the integration **semantics-honest** (see `docs/future/v10_pto_runtime_interface.md`):
if a feature is missing, PTO‑WSP must treat it as **recorded/diagnosed** (or “unsupported”), not “implemented”.

## 1) Dispatch maps to AICPU scheduler assignment (not `core_type`)

**PTO‑WSP requirement:**
`dispatch(policy)` must decide **which AICPU scheduler instance** issues a task (task division across AICPU schedulers/queues).

**What exists today:**
- `host_build_graph` can represent task dependencies and (at most) per-task executor metadata (e.g., AICore vs AIV).
- The host launch API can start N AICPU threads, but there is no stable API surface for “task belongs to AICPU #k”.

**Gap:**
- First-class “AICPU shard / scheduler id” on tasks and a deterministic policy hook to compute it.
- Validation tooling to prove the mapping is enforced (not just serialized).
- PTO‑WSP placeholders (current scaffold):
  - generated orchestration contains `TODO_PTO_RUNTIME_MULTI_AICPU_DISPATCH` in `kernels/orchestration/pto_wsp_orch.cpp`
  - generated helper symbol name: `pto_wsp_select_aicpu_scheduler_id` (currently returns 0)
  - emitter entrypoint: `pto::wsp::codegen::pto_runtime::emit_host_build_graph_sources` (`include/pto/wsp/codegen/pto_runtime_host_build_graph.hpp`)

## 2) CSP requires auto-generated orchestrator/scheduler logic

**PTO‑WSP requirement:**
Pipeline workloads must execute with CSP semantics inside the artifact/runtime (channel empty/full waits, close/termination).

**What exists today:**
- pto-runtime has a host-built graph runtime that schedules tasks by dependency counters.

**Gaps:**
- Channel concepts in runtime: channel IDs, logical capacity, empty/full waits, close semantics.
- Integration points so the AICPU scheduler can treat channel waits as readiness constraints (not host-side control).
- Deadlock detection + structured diagnostics that report channel/blocking state.

## 3) Phase 2 task-buffer / true `task_window` backpressure

**PTO‑WSP requirement:**
`task_window(mode=STALL)` must provide **real backpressure** (stall-only) to orchestration/expansion, not just bound
scheduling after a full graph is built.

**What exists today:**
- Phase 1 host-built graph (`host_build_graph`) where orchestration runs to completion to build the whole graph.

**Gaps:**
- A bounded task-buffer (“multi-ring” resources) expansion model where orchestration can stall.
- Runtime flow-control stats and stable stall-reason vocabulary aligned to v10 spec.

## 4) Policy registry / extensible dispatch & issue semantics

**PTO‑WSP requirement:**
Dispatch/issue behavior must be implemented as named policy modules with stable IDs/parameters (no hard-coded if/else trees).

**Gap:**
- A policy registry protocol usable by the AICPU scheduler (policy ID + params + evaluation contract).
- A mechanism to add new policies without modifying pto-runtime core logic.

## 5) Slots/symbol ABI (predicates, tensor→scalar materialization)

**PTO‑WSP requirement:**
Dynamic values (predicates, routing keys) must flow via a **slot/symbol table** that can be updated between runs without
recompiling kernels.

**Gaps:**
- A stable, versioned slot schema in the runtime package (IDs, widths, update protocol).
- A runtime mechanism to “materialize tensor→slot” (load selected tensor element(s) into u64 slots) on device.

## 6) Versioned package/manifest contract consumable by pto-runtime

**PTO‑WSP requirement:**
A single artifact “package” format with a manifest that identifies:
ABI version, target (`a2a3`/`a2a3sim`), kernel registry entries, schedule/CSP payload, slot schema, and entry payload type.

**Gap:**
- Standardized package layout + manifest parser/validator on the pto-runtime side (or a stable handoff API).

## 7) Python-level integration ergonomics

**PTO‑WSP requirement:**
At the Python level, PTO‑WSP should be able to import and use pto-runtime tooling (e.g., compilers/builders) in a way that
works both with the submodule layout and external installs.

**Gap:**
- A supported import story (package vs path-based modules), version compatibility checks, and clear error messages when
  toolchains are missing.
