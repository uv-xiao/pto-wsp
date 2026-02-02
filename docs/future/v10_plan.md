# PTO‑WSP v10 Plan (draft)

> **Status:** draft planning document (not yet scheduled)  
> **Baseline:** v9.3 design / 0.1.0 implementation (“codegen-first” CPU-sim; NPU emit-only here)

## Status update (2026-02-02)

v10 execution has started with **pto-runtime integration bootstrap**:

- Integration shape chosen: **Option A (git submodule)** at `3rdparty/pto-runtime`
- Phase 1 scaffolding started:
  - emit-only codegen target `target="a2a3sim_codegen"` that emits a pto-runtime `host_build_graph`-shaped source tree
  - Python import bridge for pto-runtime tooling (`python/pto_wsp/pto_runtime_bridge.py`)
- Build plumbing updated:
  - repo-local default codegen cache (`build/.pto_wsp_codegen_cache`) to avoid `$HOME` writes in sandboxed environments
  - CMake option `PTO_RUNTIME_PATH` (default submodule path)

Near-term execution plan is tracked in: `docs/plans/2026-02-02-pto-runtime-integration-v10.md`.

v10 focuses on turning PTO‑WSP into a **mature multi-backend workload+schedule compiler** while **hardening and completing** the v9
headline features (CSP + dispatch/task_window-based scheduling). Concretely, v10 targets:

1) a more solid backend/runtime architecture (by targeting the decoupled `pto-runtime`), and  
2) making v9’s **CSP/CSPT** and **dispatch/issue** semantics “production-grade” (not “new”), and  
3) an additional accelerator target: **AMD AIE / AMD NPU-style dataflow accelerators** (Dato-style stream + layout model).

This is **not** a commitment to build a bespoke runtime inside PTO‑WSP; it is an explicit plan to adopt a decoupled runtime
(`pto-runtime`) as the execution substrate while keeping PTO‑WSP’s programming model goals.

## v9 status (why this is *not* “new”)

v9 already implements the *core direction*:
- **CSP/CSPT exists** and is artifact-first on CPU-sim:
  - pipeline workloads execute inside the generated artifact with CSPT time semantics
  - time is derived from PTO‑ISA kernel cycle reports + constant channel latency (default 0)
- **Programmable dispatch/issue exists** but enforcement is intentionally narrow:
  - `dispatch(...)` is behavior-changing for CPU-sim artifacts (round_robin/hash/affinity/custom are enforced)
  - `task_window(TaskWindow(..., mode=STALL, unit="tasks"))` is behavior-changing (stall-only)
  - most other schedule knobs are API-only/diagnostic (e.g., `timing`, `stream_by`, advanced task-graph policies)
- **NPU backend is emit-only here**: it preserves `dispatch` + `task_window` but cannot be executed without Ascend/CANN.

So v10 is about **completeness, robustness, and portability**, not introducing CSP/dispatch as brand-new features.

v10 also requires a **semantics-honest** integration with the decoupled `pto-runtime` project:

- Phase 1 (host graph build) can unblock runnable backends quickly, but must not claim true `task_window` backpressure-to-orchestration.
- Phase 2 (task-buffer + on-device expansion) is required for v10 completeness and restores the v9 thesis.

See `docs/future/v10_pto_runtime_interface.md`.

## Scope

### In scope (v10)

- **Backend maturity**
  - Make the “artifact runtime” architecture clearly layered and testable:
    - frontend → typed IR → codegen (CPU-sim / NPU / AIE) → artifact runtime
  - Define a **versioned artifact package + manifest** aligned to `pto-runtime` (kernels, schedule/CSP payload, slots/symbols ABI)
  - Improve runtime concurrency/flow-control fundamentals to support CSP safely:
    - explicit task window semantics (stall-only is baseline)
    - deadlock detection + diagnostics (esp. for CSP)
    - deterministic replay knobs for testing
  - Improve codegen build hygiene:
    - deterministic artifact structure
    - consistent compilation plumbing (logs, caching, versioning)
    - “emit-only” backends can still compile host-side shims for sanity

- **CSP as a complete feature (no compromise)**
  - Artifact-first CSP/CSPT semantics across backends:
    - channel operations are explicit (send/consume), not inferred from tensors
    - time is defined by PTO‑ISA cycle reports (no parallel wall-clock notion)
    - default per-channel latency model is constant 0 cycles (stall-only semantics)
  - Backend applicability (v10 target):
    - CPU-sim: runnable and fully validated (baseline)
    - Ascend NPU: runnable in a proper Ascend/CANN environment (not emit-only in v10)
    - AIE: runnable in an AIE-capable environment (hardware or emulator), with the same CSP semantics

- **Programmable dispatch/issue as first-class schedule (beyond the v9 enforced subset)**
  - Preserve `dispatch` + `task_window` as the behavior-changing core.
  - Make dispatch/issue policies *extensible* without changing IR format:
    - new policies become new codegen/runtime policy modules, not ad-hoc branches.

- **Runtime predicates + tensor→scalar bridge**
  - Expand ScalarExpr to cover more “easy” missing pieces.
  - Support tensor→scalar conversion explicitly (reductions/comparisons) and use it in:
    - conditional scheduling
    - data-dependent execution (e.g., MoE routing primitives)
  - Define how runtime scalar values feed into codegen artifacts **without recompiling the entire artifact**:
    - runtime symbol table / “slots” (already used for dynamic axes) becomes the single mechanism
    - conditions/dispatch windows read from slots (AICPU / host), kernels remain unchanged

- **AMD AIE / dataflow accelerator backend (Dato-inspired)**
  - Add an AIE backend target that:
    - preserves stream/channel semantics in emitted artifacts
    - preserves layout/sharding type information
    - is runnable in an AIE-capable environment (hardware or emulator); local environments may still fall back to “emit-only”
  - Adopt Dato’s conceptual model as the north star:
    - streams as first-class
    - layout as type refinement
    - explicit communication edges (put/get)

- **NPU architecture representation (multi-NPU support)**
  - Introduce a first-class representation of NPU architecture that codegen/runtime can target, rather than hard-coding
    “Ascend-only” assumptions:
    - memory hierarchy (HBM/GM, L2, scratchpads/UB, register files)
    - execution units (vector, matrix/cube, DMA/MTE) and their concurrency model
    - synchronization primitives and ordering domains
    - kernel launch/dispatch model (threads/cores/tiles, waves, queues)
    - data movement costs / bandwidth models (for scheduling + CSP backpressure)
  - Use this as the basis for:
    - runnable Ascend backend (CANN integration) with shared semantics
    - runnable AIE backend mapping (Dato-style streams/ports, PE grid)
    - future NPUs (new targets become new “arch instances”, not a new compiler fork)

### Out of scope (v10)

- Making Ascend/AIE backends runnable **in every dev environment** without the vendor toolchain (toolchain-gated); v10 still
  targets full functionality in the appropriate environments.
- A complete auto-scheduler / mapping search (Dato’s search) — we may add hooks but not full search.
- MLIR rewrite of compiler infrastructure.

## Guiding principles

1) **Artifact semantics are the spec**: if behavior is “implemented”, it must be implemented in generated artifacts.
2) **CSP correctness > throughput** (first): deadlocks must be diagnosable; bounded resources must be explicit.
3) **Policies are code, not flags**: dispatch/issue/scheduler behavior should live in named policy modules.
4) **Backends share infra**: avoid “copy/paste codegen”; invest in reusable C++ codegen AST + emitters.
5) **Emit-only is still structured**: even emit-only backends must produce reviewable, consistent artifacts.

## Workstreams (v10)

Supporting docs:
- `docs/future/v10_analysis.md` (WHY)
- `docs/future/v10_features.md` (WHAT)
- `docs/future/v10_spec.md` (semantics/API)
- `docs/future/v10_implementation.md` (HOW)

### W1) Backend/runtime architecture maturity

Deliverables:
- A single canonical runtime layering diagram in `docs/implementation.md`.
- A unified “artifact runtime ABI” story for:
  - task metadata + dependencies
  - runtime symbol/slot table (dynamic axes, predicates, config knobs)
  - scheduling policy hooks (dispatch, task_window, CSP)
- A runtime-level concurrency + flow-control model that supports:
  - **multi-ring bounded resources** (task ring + dep pools + heap/buffer arena + ready queues)
  - **stall-only backpressure** when any resource is exhausted
  - **flow-control statistics** (stall counts/time, high-water marks, current stall reason)
  - bounded queues (future: per-channel capacity)
  - deadlock detection hooks (CSP-aware)

Acceptance criteria:
- CPU-sim: examples continue to be fully self-validating (golden + runner) and pass.
- Runtime semantics are explicitly described and match implementation.

### W2) True CSP/CSPT semantics (cross-backend; CPU-sim first)

Semantics to finalize:
- What constitutes a process instance (static vs dynamic).
- How channels are represented in the artifact (IDs, endpoints).
- What “consume” means:
  - blocking semantics
  - termination semantics (end-of-stream)
- Time:
  - kernel cycles from PTO‑ISA are the only time source
  - channel latency is constant 0 cycles for v10 unless configured otherwise
  - channels are token-based with a small default logical capacity (1 token)

Implementation goals:
- CSP scheduling must not regress to Python execution paths.
- Add targeted “CSP deadlock” tests:
  - bounded task window + channel waits
  - helpful diagnostic output

### W3) Programmable dispatch/issue policies

Deliverables:
- A stable “policy registry” interface for codegen/runtime.
- CPU-sim:
  - keep `round_robin(num_aicpus=...)` as baseline
  - add at least one additional policy that is demonstrably different (e.g., affinity by axis, or work-stealing-lite)
- NPU (emit-only fallback):
  - preserve `dispatch` + `task_window` in emitted metadata
  - mark other scheduling knobs explicitly “unsupported / ignored” in emitted artifacts
  - in runnable NPU environments (Ascend/AIE), ensure the same semantics are enforced by the artifact runtime

### W4) Runtime predicates and tensor→scalar conversion

Deliverables:
- A clear API surface:
  - tensor reductions producing scalar(s)
  - scalar comparisons composing into ScalarExpr
  - explicit “materialize predicate” boundaries (when a tensor-derived scalar becomes usable for control)
- A single mechanism to feed runtime values:
  - slot table values can be updated without recompiling artifacts
  - artifacts read slots to choose branches / issue decisions

Risks to manage:
- Avoid hidden host-side “retrace” behavior; keep semantics explicit in artifacts.

### W5) AMD AIE / dataflow accelerator backend (runnable target; emit-only fallback)

Deliverables:
- New backend target `aie` (runnable target; emit-only fallback) producing:
  - a task graph representation
  - stream/channel edges (put/get style) and capacities (token-based, default logical capacity=1, latency=0)
  - layout/sharding annotations
  - a compile-time check that the stream graph is a DAG for the v10 runnable path (reject cyclic graphs explicitly until buffered channels exist)
- A documentation page describing:
    - mapping from PTO‑WSP IR to a Dato-like stream model
  - what is supported vs not supported in v10 for AIE

### W6) Multi-NPU architecture representation (enables runnable Ascend + AIE)

Deliverables:
- A backend-agnostic “architecture model” layer that is consumed by both:
  - codegen (lowering choices, buffer placement, ABI, launch configuration)
  - runtime (dispatch/issue policy, windowing/backpressure, cycle accounting)
- A minimal but explicit capability matrix per target:
  - supported exec units (vector/matrix), DMA style, barriers
  - supported channel/stream primitives (sync-only vs payload)
  - supported scheduling directives (enforced vs ignored)
- At least two concrete arch instances:
  - Ascend (v10 runnable in proper CANN env)
  - AMD AIE / dataflow (v10 runnable in appropriate env; emit-only fallback acceptable)

Acceptance criteria:
- The CSP + dispatch semantics are expressed once and mapped through the arch model (no per-backend semantic drift).
- Adding a new NPU target does not require copying the compiler; it requires adding a new arch instance + backend glue.

## Validation strategy (v10)

CPU-sim remains the correctness anchor:
- Every `examples/*/*_example.py` must:
  - generate artifacts
  - run CPU-sim
  - check outputs vs golden reference
  - check cycles/latency with tolerance-based checks (per-example documented)

Emit-only backends:
- Must generate artifacts that “look correct”:
  - compile host-side pieces where possible
  - ensure all metadata is present and consistent

Runnable NPU backends (Ascend/AIE):
- Must have a backend-specific validation suite runnable in the appropriate toolchain environment:
  - correctness vs golden reference (for supported examples)
  - timing/cycle reporting sanity (strict semantics; tolerance-based checks where hardware noise exists)
  - CSP correctness (no missed waits, no silent deadlocks)
  - for Ascend, support “remote device” execution via SSH (server+workspace+env in `.ASCEND_SERVER.toml`, not committed)

### pto-runtime targeting (v10 baseline)

For v10, “backend maturity” should primarily mean “aligned to pto-runtime”, not “ported from older runtime patterns”:

- local semantics backend: `pto-runtime` `a2a3sim` (AICPU/AICore scheduling in host threads)
- Ascend backend: `pto-runtime` `a2a3` (real device execution)

See:
- `docs/future/pto_runtime_analysis.md`
- `docs/future/pto_runtime_integration.md`

## Risks and open questions

- CSP + bounded resources introduces real deadlock risk; runtime must provide actionable diagnostics.
- Tensor→scalar conversion must avoid “magical” data-dependent execution paths; control must be explicit and reviewable.
- Multi-NPU backends (Ascend/AIE) are toolchain-gated; “emit-only fallback” must remain well-structured and spec-aligned.

## Additional feature candidates (v10 and beyond)

These are not all v10 “must-haves”, but they should be tracked as candidate work items as v10 is refined:

- **More mature backend choices**
  - Optionally adopt/port proven runtime structures (task ring windowing, threaded scheduler/worker split, dual-queue)
    where they demonstrably improve robustness and maintainability.
  - Separate “runtime core” from “backend glue” so new backends do not fork the scheduler.

- **Stream/layout type system expansion (Dato-aligned)**
  - Stream capacity typing (future: capacity > 1 and/or typed payload channels) and static deadlock checks where possible.
  - Layout effect tracking for collectives (e.g., pending allreduce effects).
  - A normalized layout algebra (composition/permute/tile) for both tensor-level and tile-level memory layouts.

- **Kernel programming surface growth**
  - Expand `pto.*` lowering coverage to PTO‑ISA calls where safe.
  - Expand `ptoisa.*` wrappers to cover more instructions and patterns.
  - Add meta-programming utilities for authoring PTO‑ISA kernels without “new core ops”.

- **Artifact lifecycle improvements**
  - Stable artifact format/versioning (explicit schema), introspection tooling, and forward compatibility.
  - Faster incremental rebuilds (fine-grained caching; avoid recompiling unchanged kernels).
  - Optional “fat artifact” packaging (CPU-sim + emit-only + metadata) for reproducible benchmarking.

- **Debugging and observability**
  - First-class task graph dumps + visualizers.
  - Cycle-level tracing that matches semantic time (PTO‑ISA cycles), plus deterministic replay.

- **Auto scheduling (hook-only in v10, search later)**
  - Expose hooks for tiling/cost models and mapping, even if global search is deferred.
  - AIE/Dato mapping constraints capture (ports, PE counts, stream bandwidth).
