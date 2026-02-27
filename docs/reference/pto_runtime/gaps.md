# Reference: pto-runtime Gaps PTO‑WSP v10 Cares About

This document tracks **missing or not-yet-stable pto-runtime features** required by PTO‑WSP v10. It exists to keep the
integration **semantics-honest**: if something is not supported, PTO‑WSP should treat it as *recorded/diagnosed* (or
explicitly unsupported), not “implemented”.

Related v10 checkpoint:
- `docs/future/v10_pto_runtime_interface.md`

## 1) User-controlled `dispatch(policy)` → scheduler-domain assignment

PTO‑WSP requirement:
- `dispatch(policy)` must decide **which scheduler domain** issues the task (not the same as `core_type` / worker type).
- v10 needs both:
  - a **static fast-path** (compile-time decidable `scheduler_id`), and
  - a **dynamic hook path** (runtime dispatch hook when mapping depends on runtime state).

pto-runtime status:
- It can launch multiple AICPU threads, and the PTO2 runtime (`tensormap_and_ringbuffer`) has an explicit “3 schedulers +
  1 orchestrator” configuration, but there is **no stable ABI** for PTO‑WSP to:
  - annotate tasks with a scheduler shard id, or
  - register a per-task dispatch hook callable by the scheduler.

## 2) CSP/CSPT channel semantics

PTO‑WSP requirement:
- Channels are explicit ops (send/consume/close) that create blocking readiness constraints.
- CSPT time semantics: kernel cycles + channel latency model; stalls must be measurable and diagnosable.

pto-runtime status:
- Current runtimes focus on dependency-graph readiness (fanin counters) rather than channel readiness.
- PTO2 runtime is the likely place to integrate channels, but:
  - channel state (empty/full/closed) does not exist as a first-class runtime concept yet,
  - scheduler readiness checks don’t include channel waits,
  - deadlock diagnostics do not report CSP channel state.

## 3) True `task_window(mode=STALL)` backpressure to orchestration/expansion

PTO‑WSP requirement:
- `task_window` bounds in-flight tasks and creates **real backpressure** to the orchestrator/expander (stall-only baseline).

pto-runtime status:
- `host_build_graph`: orchestration builds the full graph first → no meaningful backpressure.
- PTO2 runtime (`tensormap_and_ringbuffer`) has a task ring + flow-control pointers, but PTO‑WSP still needs a stable
  contract to:
  - size/configure the window per artifact,
  - map runtime stall reasons into PTO‑WSP’s schedule semantics + diagnostics vocabulary.

## 4) Versioned package / manifest contract

PTO‑WSP requirement:
- A single backend package format with a manifest that identifies:
  - ABI version
  - platform (`a2a3` / `a2a3sim`)
  - runtime flavor (host_build_graph / PTO2 / …)
  - kernel registry (func_id, worker/core_type, binary/source references)
  - orchestration entry payload type (host `.so` vs device orchestration linked into AICPU)
  - schedule/CSP payload
  - slot schema

pto-runtime status:
- Example `kernel_config.py` exists and works well as a minimal config, but a versioned manifest/ABI contract is not yet
  formalized as a stable interface.

## 5) Slot/symbol ABI (predicates + tensor→scalar)

PTO‑WSP requirement:
- A runtime slot table that the artifact can read/write, and the host can update between runs without recompilation.
- A device/runtime mechanism for tensor→slot materialization (e.g., load 1 element into a u64 slot).

pto-runtime status:
- No general slot ABI yet in the runtime shared memory / package contract.

## 6) Multi-target / cross-backend artifact story (optional v10 layer)

PTO‑WSP direction:
- A program may be lowered into multiple backends; multi-target composition becomes a separate layer.

pto-runtime status:
- pto-runtime is a single-runtime substrate (Ascend + sim) rather than a multi-target framework.
- PTO‑WSP needs to define its own “multi-target layer” while treating pto-runtime as one backend target.

