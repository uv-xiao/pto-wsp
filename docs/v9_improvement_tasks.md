# PTO-RT v9 Improvement Tasks

**Generated:** 2026-01-26
**Source:** Comprehensive expert review (codex analysis)
**Status:** NEW - To be integrated into task_plan.md

---

## Executive Summary

The v9 implementation has **significant drift** between:
1. Design documents (spec.md, analysis.md)
2. Python frontend implementation
3. C++ runtime/backends

**Key finding:** The project is ~60% complete as a coherent system. The C++ foundation is solid, but Python frontend over-promises relative to what is wired end-to-end.

**Priority:** Consolidation and wiring before new features.

---

## Phase 13: Critical Integration Tasks (P0)

These must be done for v9 to be credible.

### Task 13.1: Fix Documentation Links and Versioning
**Priority:** P0 | **Effort:** Small

**Issues:**
- README.md links to `docs/ir-design.md` but file is at `docs/design/ir-design.md`
- README.md links to `docs/backend-arch.md` but file is at `docs/design/backend-arch.md`
- Version confusion: spec says "9.3", package says "0.9.0"
- License contradiction: pyproject.toml says "Proprietary", README says "[TBD]"

**Actions:**
- [ ] Fix README.md doc links to use `docs/design/` paths
- [ ] Unify version to "0.9.0" everywhere (or pick consistent scheme)
- [ ] Resolve license text (pick one)
- [ ] Add "Status: Prototype" banner to README

---

### Task 13.2: Unify C++ Bindings Import Path
**Priority:** P0 | **Effort:** Medium

**Issue:**
- Build produces `build/pto_ir_cpp.cpython-*.so` (top-level import)
- Python package has `python/pto_wsp/pto_ir_cpp.cpython-*.so` (submodule import)
- `ir_bridge.py` does `import pto_ir_cpp` which fails with default PYTHONPATH

**Actions:**
- [ ] Modify ir_bridge.py to try `from pto_wsp import pto_ir_cpp` first, then fallback
- [ ] Update CMakeLists.txt to install .so into python/pto_wsp/ consistently
- [ ] Add editable install test: `pip install -e . && python -c "from pto_wsp import pto_ir_cpp"`
- [ ] Document the import strategy in CLAUDE.md

---

### Task 13.3: Fix Workload Metadata for IR Bridge
**Priority:** P0 | **Effort:** Medium

**Issue:**
- `ir_bridge.py` expects `workload._name`, `workload._params`, `workload._schedule` dict
- `Workload` class doesn't define these attributes
- Bridge fails silently or with AttributeError

**Actions:**
- [ ] Add `_name: Optional[str]` to Workload class
- [ ] Add `_params: List[str]` to Workload class
- [ ] Replace `_schedule` dict with typed `Schedule` dataclass
- [ ] Update ir_bridge.py to match new Workload structure
- [ ] Add integration test: Python Workload → ir_bridge → C++ Module

---

### Task 13.4: Define Default Execution Path
**Priority:** P0 | **Effort:** Large (Architectural Decision)

**Issue:**
Two separate runtime worlds exist:
- Python `Program` (threadpool, ignores most scheduling)
- C++ `backend::Program` (full task graph runtime)

Docs imply they're unified when they aren't.

**Options:**
- **Option A (Production):** Python `.compile()` always lowers to C++ IR, calls C++ backend
- **Option B (Prototype):** Keep Python runtime as reference, document it clearly

**Actions:**
- [ ] Make architectural decision and document in analysis.md
- [ ] If Option A: Wire `Program.compile(target="cpu_sim")` to call C++ backend
- [ ] If Option B: Add warning/docstring that Python Program is "reference only"
- [ ] Update spec.md to reflect actual behavior

---

## Phase 14: System Cohesion Tasks (P1)

Make the system work as a coherent whole.

### Task 14.1: Merge Duplicate Kernel Systems
**Priority:** P1 | **Effort:** Large

**Issue:**
Three overlapping kernel implementations:
- `kernel.py`: JITKernel, KernelIR, tl primitives
- `builder.py`: Another Kernel wrapper on top
- `npu.py`: Legacy string-based builder

**Actions:**
- [ ] Choose canonical kernel location (recommend: keep kernel.py as internal, builder.py as public)
- [ ] Move JITKernel tracing logic into single location
- [ ] Delete/deprecate redundant code
- [ ] Ensure `@kernel` decorator has one clear implementation path
- [ ] Update all examples to use unified API

---

### Task 14.2: Implement Real Type Annotations
**Priority:** P1 | **Effort:** Large

**Issue:**
`In/Out/InOut/Constexpr` return strings like `"In[Tensor]"`. Type checker parses strings.

**Problems:**
- No IDE/mypy support
- Fragile to formatting
- Can't express rich types (shapes, layouts)

**Actions:**
- [ ] Replace string-based `In()` with `typing.Annotated[T, Direction.IN]` or generic class
- [ ] Create `KernelParam` dataclass: `(name, direction, kind, dtype, shape, layout, constexpr)`
- [ ] Update type_checker.py to work with structured types, not strings
- [ ] Ensure return annotations are properly excluded from param checking
- [ ] Add tests for tile shape validation

---

### Task 14.3: Wire Scheduling Semantics
**Priority:** P1 | **Effort:** Large

**Issue:**
Schedule combinators exist as syntax but don't affect execution:
- `stream_by()`, `timing()`, extended schedule knobs are ignored
- C++ backends use CompileOptions, not IR schedule nodes

**Actions:**
- [ ] Create canonical `Schedule` dataclass with all fields
- [ ] Store as `workload._schedule: Schedule`
- [ ] Update Python Program to honor: dispatch, streams, stream_by, timing
- [ ] Update IR bridge to serialize Schedule to C++ ScheduleDef
- [ ] Add test: schedule config affects task distribution

---

### Task 14.4: Create Golden Path Example with Validation
**Priority:** P1 | **Effort:** Medium

**Issue:**
All examples use `Tensor(data=None)` and don't validate correctness.

**Actions:**
- [ ] Create `examples/validated_matmul/` with:
  - Real numpy arrays as input
  - Registered CPU kernel that does actual computation
  - Assertion against numpy reference
  - Tracing output demonstration
  - Scheduling knob demonstration
- [ ] Add to CI as smoke test

---

### Task 14.5: Fix Example API Mismatches
**Priority:** P1 | **Effort:** Medium

**Issues found:**
- README uses `tl.softmax()` which doesn't exist
- llama_example uses `tl.slice_even`, `tl.slice_odd`, `tl.interleave` which don't exist
- csp_pipeline uses `Event(name=...)` but Event is Channel alias

**Actions:**
- [ ] Audit all examples against actual tl.* primitives
- [ ] Add missing primitives or rewrite examples
- [ ] Ensure every example runs without error
- [ ] Add `make test` to each example that validates basic execution

---

## Phase 15: Research-Grade Tasks (P2)

Close the gap to state-of-art systems.

### Task 15.1: Dynamic Axis Runtime Expansion
**Priority:** P2 | **Effort:** Large

**Issue:**
C++ lowerers only expand `DenseAxis`, not `DenseDynAxis`/`RaggedAxis`/`SparseAxis`.

**Actions:**
- [ ] Add DenseDynAxis handling in cpu_sim.cpp WorkloadLowerer
- [ ] Add RaggedAxis handling (iterate over per-row sizes)
- [ ] Add SparseAxis handling (iterate over indices)
- [ ] Test with variable-length sequences example

---

### Task 15.2: CSP Lowering or Removal
**Priority:** P2 | **Effort:** Large

**Issue:**
CSP primitives (Channel, Process, consume) exist but:
- `enumerate()` doesn't expand pipeline/consume
- No lowering to C++ task graph
- Examples are syntax demos only

**Decision needed:**
- **Option A:** Implement CSP lowering (processes → task graph with channel deps)
- **Option B:** Mark CSP as "design-only / future" and label examples accordingly

**Actions:**
- [ ] Make decision
- [ ] If Option A: Implement CSP → task graph lowering
- [ ] If Option B: Add "NOT YET IMPLEMENTED" warnings to CSP examples
- [ ] Update spec.md to reflect actual status

---

### Task 15.3: Connect Kernel IR to C++ NPU Backend
**Priority:** P2 | **Effort:** Large

**Issue:**
- Python generates kernel IR via tracing
- C++ has ir::NPUFunction representation
- These aren't connected - Ascend backend returns empty code unless manually invoked

**Actions:**
- [ ] Bridge Python KernelIR → C++ ir::NPUFunction
- [ ] Make `kernel.compile("ascend")` produce real code
- [ ] Test: workload with kernel → compile → non-empty generated code

---

### Task 15.4: Remove Unimplemented Backend Claims
**Priority:** P2 | **Effort:** Small

**Issue:**
Docs claim AMD AIE backend support, but it doesn't exist.

**Actions:**
- [ ] Remove AMD AIE from "supported backends" in docs
- [ ] Or add stub implementation with clear "not implemented" error
- [ ] Audit all docs for other unimplemented feature claims

---

## Phase 16: Documentation Truthfulness

### Task 16.1: Split Spec into Implemented vs Planned
**Priority:** P1 | **Effort:** Medium

**Issue:**
spec.md mixes normative spec with aspirational roadmap without labels.

**Actions:**
- [ ] Add "Status: Implemented/Partial/Planned" to each major section
- [ ] Move truly planned features to separate "Roadmap" section
- [ ] Ensure every "Implemented" feature has working example or test

---

### Task 16.2: Update Backend Architecture Doc
**Priority:** P1 | **Effort:** Medium

**Issue:**
`backend-arch.md` describes `Backend::compile(workload, schedule)` but code uses two-phase `lower() → compile()`.

**Actions:**
- [ ] Rewrite backend-arch.md to match actual C++ interface
- [ ] Document the two-phase compilation model
- [ ] Add diagram showing: Module → lower → LoweredPlan → compile → Program

---

### Task 16.3: Add Error Taxonomy
**Priority:** P2 | **Effort:** Medium

**Issue:**
Spec references exception types that don't exist: `CompileError`, `ExecutionError`, `ChannelClosed`

**Actions:**
- [ ] Create `python/pto_wsp/errors.py` with exception hierarchy
- [ ] Define: PtoError (base), CompileError, ExecutionError, TypeCheckError, ChannelError
- [ ] Raise actual exceptions instead of silent failures
- [ ] Add "unsupported primitive" as hard error (not silent no-op)

---

## Summary Table

| Phase | Task | Priority | Effort | Status |
|-------|------|----------|--------|--------|
| 13.1 | Fix doc links and versioning | P0 | Small | TODO |
| 13.2 | Unify C++ bindings import | P0 | Medium | TODO |
| 13.3 | Fix Workload metadata for IR bridge | P0 | Medium | TODO |
| 13.4 | Define default execution path | P0 | Large | TODO |
| 14.1 | Merge duplicate kernel systems | P1 | Large | TODO |
| 14.2 | Implement real type annotations | P1 | Large | TODO |
| 14.3 | Wire scheduling semantics | P1 | Large | TODO |
| 14.4 | Create golden path example | P1 | Medium | TODO |
| 14.5 | Fix example API mismatches | P1 | Medium | TODO |
| 15.1 | Dynamic axis runtime expansion | P2 | Large | TODO |
| 15.2 | CSP lowering or removal | P2 | Large | TODO |
| 15.3 | Connect kernel IR to C++ NPU | P2 | Large | TODO |
| 15.4 | Remove unimplemented backend claims | P2 | Small | TODO |
| 16.1 | Split spec into implemented/planned | P1 | Medium | TODO |
| 16.2 | Update backend architecture doc | P1 | Medium | TODO |
| 16.3 | Add error taxonomy | P2 | Medium | TODO |

---

## Next Steps

1. **Immediate:** Integrate Phase 13 (P0) tasks into task_plan.md
2. **This week:** Complete P0 tasks to make v9 credible
3. **Next iteration:** P1 tasks for system cohesion
4. **Future:** P2 tasks for research-grade capability

The focus should be **consolidation and wiring**, not new features.
