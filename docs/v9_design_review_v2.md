# PTO‑RT v9 Design & Implementation Expert Review (Repo snapshot: 2026‑01‑26)

## Scope & method
Reviewed:
- Design docs: `docs/spec.md`, `docs/analysis.md`, `docs/design/ir-design.md`, `docs/design/backend-arch.md` (+ cross-checks against `README.md`, `pyproject.toml`, and referenced code).
- Python API: `python/pto_wsp/` (frontend DSL, kernels, scheduling, IR bridge, type checking, passes, auto-scheduler).
- C++ core: `include/pto/rt/**`, `src/pto/rt/**`, and pybind bridge `src/python/pto_ir_bindings.cpp`.
- Examples: `examples/*/*`.

Limitations:
- I could not execute `pytest` in this environment because no writable temp directory is available (Python fails creating temporary files). This is an environment constraint, not necessarily a repo issue.

---

## Executive summary (critical)
PTO‑RT v9 has a strong *C++ foundation* (IR, parser/printer, task-graph runtime, backend registry) and a promising high-level *intent* (typed workload expressions + combinator scheduling + multi-backend). However, the project currently has **significant drift** between **(a) the spec/design docs**, **(b) the Python frontend**, and **(c) the C++ runtime/backends**.

The largest blockers to “v9 as described” are:
1. **Python→C++ integration is effectively broken by default packaging/import paths** and mismatched Workload/schedule representations.
2. **Python `Program` execution is a separate “toy runtime”** that ignores most scheduling primitives and does not connect to the C++ backends.
3. Several “headline” features in the spec (AIE backend, backend applicability validation, fully working CSP pipelines, extended schedule controls, error taxonomy) are **documented as present** but **not implemented or not wired**.

If you want a production-credible v9, the next step is not more features—it’s **consolidation and wiring**: pick a single compilation/execution path and make docs match reality.

---

# 1) Design document completeness & consistency

## 1.1 Major doc↔repo mismatches (high severity)

### Broken doc links / structure drift
- `README.md` links to `docs/ir-design.md` and `docs/backend-arch.md`, but the actual files are `docs/design/ir-design.md` and `docs/design/backend-arch.md`. This makes “Documentation” effectively dead on first click.

### Versioning contradictions
- `docs/spec.md` ends with “*Version: 9.3* … *Last Updated: 2026‑01‑25*”.
- Python package reports `__version__ = "0.9.0"` in `python/pto_wsp/__init__.py`, and `pyproject.toml` says `version = "0.9.0"`.
- This is confusing for users and makes it unclear whether “v9” is a design generation, an API version, or a release train.

### License contradictions
- `pyproject.toml` claims `license = {text = "Proprietary"}`.
- `README.md` says “License: [To be determined]”.
- This is a compliance/enterprise adoption blocker.

## 1.2 Spec completeness vs implemented surface area
`docs/spec.md` is detailed but frequently mixes **normative spec** with **aspirational roadmap** without clearly labeling “implemented / partially implemented / future”.

Concrete examples:
- Spec introduces `Metric`, `PoolBy`, `.issue(...)` API variants, backend applicability compile-time errors, and many schedule knobs that do not exist in Python or are not connected to the runtime.
- Spec’s **Error Handling** section references exception types (`CompileError`, `ExecutionError`, `ChannelClosed`, etc.) that do not exist in the Python package.
- Spec describes **multi-backend compilation** via `compile(target=...)`, but the default Python execution path does not honor `target` meaningfully.

## 1.3 `docs/design/backend-arch.md` vs actual C++ backend interface
`docs/design/backend-arch.md` describes a backend interface shaped around:
- `Backend::compile(workload, schedule)` and `Backend::compile(pipeline, schedule)`.

But `include/pto/rt/backend/backend.hpp` implements a **two-phase** model:
- `Backend::lower(module, options) -> LoweredPlan`
- `Backend::compile(plan, options) -> Program`
- plus `compile_module` convenience.

This is a major conceptual mismatch: the design doc is not describing the code you have.

## 1.4 `docs/design/ir-design.md` vs implementation
The IR design doc includes extensive sections on:
- pass infrastructure,
- IR rewriter,
- extension registry mechanisms.

Reality:
- C++ has a visitor (`include/pto/rt/ir/visitor.hpp`) and a type checker pass (`include/pto/rt/ir/type_check.hpp`) and parser/printer.
- Python has `python/pto_wsp/ir_passes.py`, but it references non-existent fields (e.g., `workload._name`) and is not robust.
- Net: IR passes/rewriter are *partially present conceptually* but not implemented coherently.

### Recommendation
Split docs into:
- **Spec (implemented)**: only what is callable and wired end-to-end.
- **Design (planned)**: explicitly marked roadmap items, with “not yet implemented” banners.

---

# 2) Python API design quality (consistency, type safety, ergonomics)

## 2.1 The single biggest API problem: duplicated “kernel systems”
There are overlapping kernel concepts across:
- `python/pto_wsp/kernel.py` (defines `JITKernel`, `KernelIR`, `tl`, tile language ops, compilation stubs),
- `python/pto_wsp/builder.py` (defines **another** `Kernel` wrapper and **another** `kernel()` decorator on top of `pto_wsp.kernel`),
- plus legacy `python/pto_wsp/npu.py`.

This duplication causes:
- unclear “source of truth” for JIT behavior,
- inconsistent compilation semantics,
- confusing imports (`from pto_wsp import kernel, tl, Tile, Scalar` looks simple but hides multiple implementations).

### Recommendation
Pick ONE kernel front-end:
- Either keep `python/pto_wsp/kernel.py` as canonical and delete/merge the duplicate kernel wrapper in `builder.py`,
- or make `builder.py` canonical and demote `kernel.py` to internal IR.

## 2.2 Type annotations are strings (not types)
In `python/pto_wsp/builder.py`, `In/Out/InOut/Constexpr` produce strings like `"In[Tensor]"`. The type checker in `python/pto_wsp/type_checker.py` then parses strings.

Problems:
- loses IDE/mypy benefits,
- makes annotations fragile to formatting,
- blocks richer typing (e.g., tile shapes, layout refinements, constexpr specialization keys).

### Recommendation
Use a real typing strategy:
- `typing.Annotated` for directionality and constexpr markers, or
- proper generic wrapper types (e.g., `class In(Generic[T])`) that preserve runtime structure,
- or `Protocol`/dataclass descriptors for kernel params.

## 2.3 Type checker has structural issues
In `python/pto_wsp/type_checker.py`:
- Kernel signature uses `func.__annotations__` but does not exclude `"return"`, so annotating returns can break arity checks.
- It only checks `Tensor` arguments; it does not validate tile shapes, scalar parameters, or layout constraints beyond a simplistic join.
- Tensor access checking exists in concept (`WorkloadBuilder.check_tensor_access`) but is not invoked for typical indexing patterns (e.g., `Q[b][h]`).

### Recommendation
- Normalize kernel signatures into a structured schema (`KernelParam(name, direction, kind, dtype, shape, layout, constexpr)`).
- Ensure builder hooks (tensor indexing, axis binding) actually call the type checker.
- Add first-class error objects with locations and actionable hints (the scaffolding is close, but wiring is missing).

## 2.4 Workload IR representation is too implicit (`_kind` + `_kwargs`)
`python/pto_wsp/workload.py` stores the IR as:
- `_kind: str` and `_kwargs: dict`,
- with many branches in enumerators/program compilers.

This is flexible but becomes fragile fast:
- no validation of required keys per kind,
- “silent ignore” patterns for unsupported kinds (`pipeline`, `consume`, etc.),
- lots of `Any` and implicit contracts.

### Recommendation
Convert to typed dataclasses for nodes (even if still dynamic):
- `TaskNode(kernel, params, resources)`
- `ParallelForNode(axis, var_name, body)`
- etc.
Then define a single visitor/enumerator and a single serializer.

## 2.5 Scheduling objects are not wired or serializable
- `python/pto_wsp/schedule.py` defines policy objects, but they are not consumed by the Python runtime beyond `dispatch` and `streams`.
- `Workload.stream_by()` and `Workload.timing()` exist, but `python/pto_wsp/program.py` ignores them in execution planning.
- The IR bridge (`python/pto_wsp/ir_bridge.py`) expects a nonexistent `workload._schedule` dict and also expects `_name`/`_params` that `Workload` does not define.

This makes “combinator scheduling” look present but not real end-to-end.

### Recommendation
Define a single canonical schedule representation, e.g.:
- `Schedule(dispatch=..., streams=..., stream_by=..., timing=..., task_graph=..., extended=...)`
store it as `workload._schedule: Schedule`, and ensure:
- Python runtime honors it (even if only partially),
- IR bridge can serialize it,
- C++ backends can interpret it or reject it with clear errors.

## 2.6 C++ bindings import path is inconsistent
- The package contains `python/pto_wsp/pto_ir_cpp.cpython-310-...so` (importable as `pto_wsp.pto_ir_cpp`),
- the build produces `build/pto_ir_cpp.cpython-310-...so` (importable as `pto_ir_cpp`),
- but `python/pto_wsp/ir_bridge.py` attempts `import pto_ir_cpp`, so **default usage that only adds `python/` to `sys.path` will not see bindings**.

This is a key reason “Python→C++ bridge” appears broken by default.

### Recommendation
Make bindings importable consistently:
- Either always package the extension as top-level `pto_ir_cpp`,
- or change Python bridge to import `pto_wsp.pto_ir_cpp` first and fall back to `pto_ir_cpp`.

---

# 3) Example quality (correctness, best practices, educational value)

## 3.1 Examples are often not runnable as written (API mismatches)
Concrete issues:
- `README.md` uses `tl.softmax(...)`, but `tl` has no `softmax` method (only primitives like `rowmax`, `rowsum`, etc.).
- `examples/llama/llama_example.py` uses `tl.slice_even`, `tl.slice_odd`, `tl.interleave`, which do not exist in `python/pto_wsp/kernel.py`.
- `examples/csp_pipeline/csp_pipeline_example.py` uses `Event(name="...")` but `Event` is an alias to `Channel` with default `depth=1`; rendezvous semantics described in docs require `depth=0`.

## 3.2 Examples generally don’t validate anything
Most examples allocate `Tensor(data=None, ...)` and never provide real buffers. The Python `Program` runtime also executes kernels only if manually registered; most examples don’t register implementations, so execution is effectively a no-op.

This harms educational value because it teaches API shape but not “how to get correct results”.

### Recommendation (high ROI)
Add one “golden path” example that:
- uses real `numpy` arrays,
- registers CPU kernel implementations (or uses the C++ CPU sim backend via pybind),
- asserts correctness vs a reference implementation,
- demonstrates tracing output + scheduling knobs.

## 3.3 CSP examples are conceptually good but not executed end-to-end
The CSP layer in Python builds IR nodes, but:
- `Workload.enumerate()` and `Program._enumerate_workload()` don’t actually expand `pipeline` / `consume` into tasks in a meaningful way.
- There is no lowering of CSP constructs into the C++ task graph runtime.

### Recommendation
Either:
- implement CSP lowering fully (channels/processes → task graph + dependencies),
or
- mark CSP as “design-only / not yet supported” and keep examples as “syntax demos” explicitly labeled.

---

# 4) Architecture coherence vs stated principles

## 4.1 “Typed workload expressions” are not true today (in Python)
Docs emphasize typed workloads and dependency typing (Independent/Sequential/etc.). In practice:
- Python workloads are dynamically typed and mostly validated by a string-parsing checker.
- Dynamic axes (DenseDyn/Ragged/Sparse) are modeled but not supported in C++ lowering (CPU/NPU lowerers only expand `DenseAxis`, not `DenseDynAxis`).

**Net:** The typing story exists primarily in design docs and partial checks, not enforced in compilation/execution.

## 4.2 Two runtime worlds: Python `Program` vs C++ backends
- Python `.compile()` returns `python/pto_wsp/program.py::Program`, which is a Python threadpool executor and does not use `pto_ir_cpp` backends.
- The C++ backends (`cpu_sim`, `ascend_npu`) compile IR modules (`ir::Module`) into `backend::Program`.

These two worlds are not integrated; users must choose one without guidance, and docs imply they’re unified when they aren’t.

### Recommendation (architectural decision required)
Pick a single default story:
- **Option A (recommended for production):** Python frontend always lowers to C++ IR and calls C++ backend programs when bindings exist; Python `Program` becomes a thin wrapper over C++ `Program`.
- **Option B (prototype mode):** Keep Python runtime as reference, but then docs must stop claiming multi-backend compilation from Python unless explicitly invoking the C++ path.

## 4.3 “Combinator scheduling” exists as syntax but not as semantics
Workload combinators (`dispatch/streams/stream_by/timing/task_graph/...`) exist, but:
- Python executor doesn’t respect most of them,
- IR bridge doesn’t serialize them correctly,
- C++ backends mostly take runtime config from `CompileOptions`, not IR schedule nodes.

So scheduling is currently:
- *syntax in Python*, and
- *runtime knobs in C++*,
but not one coherent model.

---

# 5) Industrial readiness (production systems perspective)

## 5.1 Error handling and observability
Python execution:
- `Program._execute_task()` catches kernel exceptions but largely suppresses them (records to trace only at FULL level, no propagation).
- Many unsupported IR kinds silently do nothing (pipeline/consume).

C++ execution:
- C++ side is cleaner (structured runtime, clear compile options), but Python doesn’t surface most of it.

### Recommendations
- Define a unified exception taxonomy and actually raise on execution failures by default (with opt-in “best effort” modes).
- Add structured logging hooks (or at least a debug mode that prints task IDs, kernel names, bindings).
- Ensure “unsupported primitive” is a hard error unless explicitly marked as “no-op for now”.

## 5.2 Testing and CI
You have a good set of unit tests under `tests/`, but the repo needs:
- CI that builds the extension and runs tests with a working temp dir,
- at least one “integration test” for Python frontend → C++ IR → CPU backend execution.

Also, several Python modules appear untested and likely broken (e.g., `python/pto_wsp/ir_passes.py` referencing `_name`).

## 5.3 Packaging and distribution
- Examples rely on `sys.path.insert(0, 'python')` instead of proper installation.
- Binding import path mismatch makes the C++ integration fragile.
- `package.json` exists but is effectively empty; unclear whether it is needed.

### Recommendation
Create a single “Getting started” path:
- `pip install -e .` works,
- `import pto_wsp` gives consistent behavior,
- C++ bindings available as `pto_ir_cpp` (or clearly documented alternative).

---

# 6) Research alignment (FlashInfer, Triton, TVM, etc.)

## 6.1 FlashInfer (plan/run, variable-length decode)
Strength:
- `examples/flashinfer_decode/flashinfer_decode_attention.py` captures the *concept* of plan/run and work descriptors.

Gaps vs state-of-art:
- No first-class IR support for “descriptor arrays” / segmented KV chunks.
- No runtime support for dynamic sequence lengths in C++ lowering.
- No integration with kernel specialization/selection based on tiers beyond Python-level logic.

### Recommendations
- Represent “work descriptors” as a first-class IR type (or a structured tensor type) and support iterating them efficiently (`select` over descriptor list).
- Add runtime expansion for `DenseDyn`/`Ragged` in the C++ CPU backend as a correctness baseline (then port to NPU).
- Add a kernel selection mechanism tied to `Constexpr` and layout constraints (tiered kernels should be a core feature, not only an example).

## 6.2 Triton-style kernel programming
Strength:
- `tl.*` SSA-like `Value`/`Op` IR is a reasonable starting point and resembles the “trace to IR” pattern.

Gaps vs Triton:
- Missing pointer/block-pointer model, program IDs, warp/thread semantics, and many core primitives needed to express real kernels efficiently.
- `Scalar[...]` parameters are not correctly modeled during tracing (they become tiles today in the `builder.Kernel` tracer).
- No auto-tuning or specialization cache keyed on constexpr/layout/dtypes in a real, enforceable way.

### Recommendations
- Add `tl.constexpr` semantics + specialization keying.
- Add memory pointer types and index math primitives (even if backend lowering is minimal initially).
- Integrate the kernel IR with the C++ NPU function IR (`ir::NPUFunction`) instead of generating ad-hoc strings in Python.

## 6.3 TVM / compiler ecosystem alignment
Strength:
- You have serious research notes under `docs/reference/` and `docs/research/`, and a real IR + parser/printer, which is “TVM-like” in the good sense.

Gaps:
- No robust pass pipeline (C++), no verified rewrite framework, and no cost model integration.
- Outer scheduling is not expressed as transformations on IR; it’s mostly runtime configuration today.

### Recommendations
- Implement a minimal C++ pass manager (even just “type check → canonicalize → lower”) and make it the compilation spine.
- Clarify IR levels (workload graph vs kernel IR) and how they compose (the design says this, but implementation doesn’t enforce it).

---

# Prioritized improvement roadmap (actionable)

## P0 (must do to make v9 believable)
1. Fix doc links and versioning: `README.md` doc links, unify “v9/0.9.0/9.3” story, resolve license text.
2. Unify C++ bindings import strategy so Python users can access them without adding `build/` manually.
3. Define `Workload` metadata consistently (`name`, params, schedule object) and fix `python/pto_wsp/ir_bridge.py` to match reality.
4. Decide the default execution path: either wire Python `.compile()` to C++ backend or clearly label Python `Program` as “toy/reference”.

## P1 (make the system cohesive)
1. Merge the duplicate kernel systems (`kernel.py` vs `builder.py`) and implement real typed annotations (no string parsing).
2. Make scheduling semantics real: `stream_by`, `timing`, `task_graph`, and extended knobs must affect execution or be removed from the public API.
3. Fix examples so at least 2–3 run end-to-end and validate correctness (one for workloads, one for kernels, one for CSP or plan/run).

## P2 (close the gap to research-grade systems)
1. Add dynamic axis runtime expansion in C++ CPU backend (`DenseDyn`, `Ragged`, `Sparse`) as the correctness baseline.
2. Implement real CSP lowering (or remove CSP claims until it exists).
3. Connect kernel IR → `ir::NPUFunction` → Ascend codegen, and make Ascend backend actually produce code from a workload module (today it returns empty generated code unless called manually).
4. Either implement the AMD AIE backend or remove it from “supported backends” documentation until it exists.

---

## Bottom line
The repo has a solid “compiler/runtime core” in C++ and a compelling vision, but the **Python frontend + docs currently over-promise** relative to what is implemented and wired. If you focus the next iteration on **integration, single-path execution, and truth-in-docs**, PTO‑RT v9 can become a credible foundation for real workloads and research experimentation.

If you want, I can follow up with a concrete “wiring plan” that maps Python `@workload/@kernel` → `ir::Module` → `backend::compile(...)` and identifies exactly what needs to change in `python/pto_wsp/workload.py`, `python/pto_wsp/ir_bridge.py`, and the pybind surface to make `workload.compile(target=...)` actually work.