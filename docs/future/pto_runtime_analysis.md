# PTO Runtime (decoupled) — Analysis Notes for PTO‑WSP v10

This document captures what the **decoupled `pto-runtime`** project provides today, what it is planning next (task-buffer
direction), and how PTO‑WSP v10 should target it as the backend runtime architecture.

Reference repo (read-only in this workspace): `references/pto-runtime/`.

## 1) What pto-runtime is (today)

`pto-runtime` is a modular task runtime intended to run on Ascend platforms using a **3-program split**:

- **Host runtime** (shared library, loaded by Python via ctypes)
- **AICPU kernel** (device-side scheduler / orchestrator)
- **AICore kernels** (device-side compute)

It supports two platforms:

- `a2a3` — real Ascend hardware (requires CANN toolchain)
- `a2a3sim` — host-thread simulation of AICPU/AICore (gcc/g++ only; no CANN)

This is critical for PTO‑WSP v10 because it means:

- we can have **device-faithful scheduling semantics** (AICPU ↔ AICore handshake) even without hardware (`a2a3sim`), and
- we can run the **same runtime implementation** in simulation and on real devices.

Reference entrypoint: `references/pto-runtime/README.md`.

## 2) Execution model (host_build_graph runtime)

The current example runtime `host_build_graph` builds a task graph on host via a dynamically loaded orchestration function:

1) Python compiles:
   - host runtime `.so` (platform-specific)
   - AICPU binary (scheduler)
   - AICore binary (dispatcher/executor)
2) Python compiles an **orchestration `.so`** and calls into `init_runtime(...)`.
3) The orchestration function allocates device memory via host APIs, copies inputs, and populates task records.
4) Python launches the runtime (`launch_runtime`) which starts AICPU scheduling and AICore execution.
5) Finalization copies recorded tensors back and frees device memory.

Key detail: `a2a3sim` can load kernels into executable memory and run AICPU/AICore logic in threads, enabling correctness
testing without Ascend.

Reference entrypoints:
- `references/pto-runtime/src/runtime/host_build_graph/runtime/runtime.h`
- `references/pto-runtime/src/runtime/host_build_graph/host/runtime_maker.cpp`
- `references/pto-runtime/examples/host_build_graph_sim_example/main.py`

### 2.1 Toolchain + build workflow (important for real-device testing)

`pto-runtime` formalizes “how to build the 3 binaries” via Python tooling:

- `RuntimeBuilder(platform=...)` discovers runtimes and builds:
  - host `.so`
  - AICPU binary
  - AICore binary
- `BinaryCompiler(platform=...)` selects toolchains:
  - `a2a3`: requires `ASCEND_HOME_PATH` and uses `ccec` + aarch64 cross toolchain + host gcc/g++
  - `a2a3sim`: uses host gcc/g++ for all three, no Ascend SDK required
- `PTOCompiler` compiles:
  - orchestration `.so` (g++)
  - incore kernels (`ccec` on real device, g++ on sim) and extracts `.text` sections for registration

This is exactly the kind of “real device testing plumbing” PTO‑WSP v10 needs to standardize.

## 3) Data model (host_build_graph runtime.h)

The simplified `host_build_graph` runtime stores tasks in a fixed-size array and schedules using a ready-queue-like loop.
Each task carries:

- `task_id`, `func_id`
- `args[]` as `uint64_t` (stable ABI)
- dependency tracking (`fanin`, `fanout[]`)
- execution placement (`core_type` AIC/AIV)
- kernel entry address (`function_bin_addr`) supplied by host after registering kernel binary
- timing fields (`start_time`, `end_time`) (currently DFX-oriented)

This is *not yet* PTO‑WSP’s “dynamic schedule + bounded window” story, but it is a useful baseline ABI:

- stable u64 args convention aligns well with PTO‑WSP’s kernel ABI
- explicit placement hints match `dispatch` / multi-executor semantics

## 4) Upcoming direction: task-buffer / bounded runtime (why it matters)

PTO‑WSP v10 requires:

- bounded in-flight execution (`task_window` stall-only semantics),
- bounded dependency tracking structures,
- bounded intermediate allocation arenas,
- actionable deadlock diagnostics (CSP + bounded resources).

The upcoming `pto-runtime` direction is previewed in:
- `docs/future/pto-runtime-task-buffer.md`

This direction aligns with the “multi-ring flow control” model (task ring + dep pools + heap arena + ready queues), but
**we should treat pto-runtime as the primary target**, and avoid building a parallel bespoke runtime core inside PTO‑WSP.

## 5) What PTO‑WSP must add on top (v10 gap analysis)

`pto-runtime` provides runtime execution plumbing; PTO‑WSP provides a novel programming model.

PTO‑WSP v10 gaps that must map into `pto-runtime`:

- **Workload–schedule semantics**:
  - `dispatch(policy)` + `task_window(stall-only)` are behavior-changing.
  - runtime must preserve/execute those semantics, not treat them as metadata.
- **CSP/CSPT**:
  - channel operations are explicit scheduling edges, not tensor deps.
  - cycle semantics must be derived from kernel cycle reports (PTO‑ISA) and constant channel latency.
- **Runtime predicates and tensor→scalar**:
  - slot/symbol tables that can be updated between runs without recompiling the artifact.
  - scalar-driven control must be artifact/runtime semantics (not Python host fallback).
- **Multi-backend story**:
  - `a2a3sim` becomes the default local “cpu_sim” validation backend for scheduler/CSP semantics.
  - `a2a3` becomes the real Ascend backend (toolchain-gated execution).

## 6) Recommendation for v10 direction

Treat `pto-runtime` as the backend runtime architecture for v10:

- “cpu_sim” correctness backend should be implemented via `pto-runtime` `a2a3sim` (AICPU/AICore semantics on host threads).
- Ascend backend should use `pto-runtime` `a2a3`.
- PTO‑WSP codegen should emit artifacts that are **consumable by pto-runtime** (task buffer/ABI, kernels, orchestration).

This keeps PTO‑WSP focused on its core: **workload–schedule programming**, while sharing runtime maturity work across repos.
