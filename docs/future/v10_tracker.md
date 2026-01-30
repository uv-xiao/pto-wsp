# PTO‑RT v10 Tracker (draft)

This file is a checklist tracker for executing `docs/future/v10_plan.md`.

## W1) Backend/runtime architecture maturity

- [ ] Define and document the v10 artifact runtime ABI (tasks, deps, slots, policy hooks)
- [ ] Consolidate codegen build plumbing (deterministic artifact layout, logs, cache versioning policy)
- [ ] Implement bounded runtime resources (“multi-ring” flow control): task ring, tensormap pool, deplist pool, heap/buffer arena, ready queues
- [ ] Add flow-control stats + high-water marks (per bounded resource) and surface them via program stats/diagnostics
- [ ] Add runtime deadlock/diagnostic hooks suitable for CSP (CPU-sim; actionable stall/channel reports)
- [ ] Include scope/liveness state (scope depth, scope-held outputs) in deadlock diagnostics

## W2) CSP/CSPT semantics (cross-backend; CPU-sim first)

- [ ] Finalize CSP semantics doc (process instances, channel ops, termination) (upgrade v9 semantics to “complete”)
- [ ] Implement/verify channel ops are explicit (not tensor-inferred) in CPU-sim artifacts
- [ ] Implement logical channel capacity semantics (token-based; default capacity=1) and enforce it consistently in runtime/codegen
- [ ] Implement constant channel latency model (default 0 cycles; stall-only)
- [ ] Add CSP deadlock tests + diagnostics (bounded resources)
- [ ] Make CSP runnable on Ascend backend in CANN environment (no emit-only limitation in v10)
- [ ] Make CSP runnable on AIE backend in AIE environment (hardware/emulator)

## W3) Programmable dispatch/issue policies

- [ ] Define policy registry surface (IR → codegen/runtime policy modules) (upgrade v9 enforced subset)
- [ ] Keep baseline `round_robin` policy and add at least one additional CPU-sim policy
- [ ] Ensure `dispatch` + `task_window` are preserved in NPU emit-only artifacts
- [ ] Ensure non-core scheduling knobs are explicitly “unsupported/ignored” in emitted artifacts

## W4) Runtime predicates + tensor→scalar conversion

- [ ] Expand ScalarExpr coverage (easy missing ops)
- [ ] Add explicit tensor→scalar materialization API (reductions/comparisons)
- [ ] Plumb tensor-derived scalars into artifact control (via slots) without recompiling artifacts
- [ ] Add tests for data-dependent control (MoE-routing style toy workloads)

## W5) AMD AIE / dataflow accelerator backend (runnable target; emit-only fallback)

- [ ] Define `target="aie"` artifact format (task graph + streams + layouts)
- [ ] Implement AIE backend preserving dispatch + task_window + stream semantics (runnable in AIE env; emit-only fallback acceptable)
- [ ] Add compile-time stream-graph validation (DAG required for v10 runnable path; reject cyclic graphs explicitly)
- [ ] Add doc: PTO‑RT IR ↔ Dato-like model mapping and v10 support matrix

## W6) Multi-NPU architecture representation

- [ ] Define a backend-agnostic NPU architecture model (memory hierarchy, exec units, sync, launch model)
- [ ] Implement Ascend arch instance and wire it through codegen/runtime
- [ ] Implement AIE arch instance and wire it through codegen/runtime
- [ ] Add per-backend capability matrix (enforced vs ignored schedule + CSP support)

## Validation

- [ ] Ensure all examples remain self-validating (golden + runner + cycle tolerance)
- [ ] Add at least one CSP example and ensure it validates on CPU-sim
- [ ] Add a backend validation suite runnable on Ascend/CANN (correctness + cycles + CSP)
- [ ] Add a remote Ascend device testing harness driven by `.ASCEND_SERVER.toml` (ssh+rsync, build/run commands)
- [ ] Add a backend validation suite runnable on AIE (correctness + cycles + CSP)
