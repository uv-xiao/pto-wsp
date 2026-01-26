# PTO-RT v9 Design Review

**Review Date:** 2026-01-26
**Reviewers:** Claude Code (assisted by Codex analysis)
**Status:** Comprehensive Expert Review

---

## Executive Summary

PTO-RT v9 represents a significant design effort to create a typed workload expression system for Ascend NPU and other accelerators. The project has a solid architectural foundation with well-structured C++ IR and backend implementations. However, several gaps exist between documentation and implementation, and some design decisions need refinement for production readiness.

**Overall Assessment:** 70% complete - Core infrastructure is solid, but integration layers and advanced features need work.

---

## 1. Design Document Completeness

### 1.1 Documentation Inventory

| Document | Status | Issues |
|----------|--------|--------|
| `docs/spec.md` | Exists | Some features not implemented |
| `docs/analysis.md` | Exists | Good rationale, matches design intent |
| `docs/ir-design.md` | Exists | Pass Manager and IRRewriter not implemented |
| `docs/backend-arch.md` | Exists | Matches C++ implementation well |
| `docs/type-system.md` | Exists | Implementation uses string parsing, not typed |

### 1.2 Key Documentation Gaps

**Issue D1: README Import Path Mismatch**
- README shows: `from pto.rt import ...`
- Actual package: `from pto_wsp import ...`
- **Fix:** Update README to use correct import path

**Issue D2: IRRewriter/Pass Manager Not Implemented**
- `docs/design/ir-design.md` defines extensive pass infrastructure (lines 1705-1826)
- No corresponding implementation in `include/pto/rt/ir/`
- **Impact:** Missing optimization/transformation infrastructure
- **Fix:** Either implement or mark as "Future Work" in docs

**Issue D3: Extension Registry Undefined**
- Mentioned in IR design but not implemented
- **Impact:** No plugin/extension mechanism

---

## 2. API Design Quality

### 2.1 Python API Issues

**Issue A1: Type Annotations Return Strings, Not Types**
```python
# Current implementation (builder.py)
def In(inner_type: type) -> str:
    return f"In[{inner_type.__name__}]"
```
- Type annotations like `In[Tensor]` return strings
- Type checker parses these strings instead of inspecting types
- **Impact:** No compile-time type safety; IDE support is limited
- **Recommendation:** Implement proper generic types or use typing.Annotated

**Issue A2: Duplicate Layout Systems**
- `TensorLayout` in `types.py` - for tensor distribution
- `Layout` in `type_checker.py` - for type checking
- `Shard`/`Replicate` in `spatial.py` - for spatial layouts
- `TensorShard`/`TensorReplicate` in `types.py` - another variant
- **Impact:** User confusion, maintenance burden
- **Fix:** Consolidate into single layout system

**Issue A3: Workload.enumerate() Missing Handlers**
```python
# workload.py - _enumerate_recursive() handles:
# - "parallel_for"
# - "for_each"
# - "task"
# Missing handlers for:
# - "select" (documented in spec)
# - "cond" (documented in spec)
# - "pipeline" (CSP feature)
```
- **Impact:** `select()`/`cond()` primitives won't enumerate tasks correctly
- **Fix:** Implement missing handlers in `_enumerate_recursive()`

**Issue A4: Schedule Methods Return Copies Without Validation**
```python
def dispatch(self, policy: Any) -> Workload:
    """..."""
    new_workload = self.copy()
    new_workload._schedule["dispatch"] = policy
    return new_workload
```
- No validation that policies are compatible
- No type checking on policy objects
- **Recommendation:** Add validation and proper typing

### 2.2 Naming Inconsistencies

| Context | Name 1 | Name 2 | Issue |
|---------|--------|--------|-------|
| Kernel decorator | `@kernel` | `@jit_kernel` | Now unified (good) |
| Layout sharding | `Shard` | `TensorShard` | Duplicates |
| Sequence iteration | `P.seq()` | `for_each()` | Different APIs same concept |

### 2.3 Missing API Features

1. **No runtime kernel registration from Python**
   - `Kernel.compile()` generates code but doesn't register with runtime
   - CPU simulation backend has `KernelRegistry` but no Python bridge

2. **No tensor data binding**
   - `Tensor(data=None, ...)` - data is always None in examples
   - No mechanism to bind actual numpy arrays

3. **No execution feedback**
   - `program.execute()` returns `None`
   - No profiling data, no error details

---

## 3. Example Quality

### 3.1 Example Inventory

| Example | Pattern Demonstrated | Runs? | Issues |
|---------|---------------------|-------|--------|
| attention | Basic @workload + P | Yes | Good |
| bgemm | Batched GEMM | Yes | Good |
| softmax | Online algorithm | Yes | Good |
| llama | Full model | Yes | Large, good coverage |
| deepseek_v3 | MoE routing | Yes | Complex, good |
| deepseek_lightning_indexer | Tier selection | Yes | Good tier pattern |
| e2e_rmsnorm | RMSNorm | Yes | Good |
| flashinfer_decode | Plan-Run model | Yes | Good architecture demo |

### 3.2 Example Issues

**Issue E1: No Actual Computation**
- All examples define kernels with `tl.*` but no actual tensor data
- Examples demonstrate API, not correctness
- **Recommendation:** Add at least one example with numpy data and validation

**Issue E2: Inconsistent Kernel Style**
```python
# Some examples use full tile operations:
mm_result = tl.matmul(q_tile, k_cache)

# Others use element-wise style:
sq = tl.mul(x, x)
mean = tl.rowmean(sq)
```
- Need clearer guidelines on when to use which style

**Issue E3: Missing CSP Example**
- No example demonstrates `Channel`, `Process`, `consume`
- CSP is a major v9 feature but no end-to-end example
- **Fix:** Add `examples/csp_pipeline/` demonstrating producer-consumer

---

## 4. Architecture Coherence

### 4.1 Design Principles Adherence

| Principle | Status | Notes |
|-----------|--------|-------|
| Typed Workload Expressions | Partial | Python uses dynamic typing with string annotations |
| Combinator Scheduling | Good | `.dispatch().streams().compile()` works |
| CSP Primitives | Partial | IR exists, no backend lowering |
| Compute/Schedule Separation | Partial | Workload combines both (unlike Halide) |

### 4.2 Layer Integration Issues

**Issue AR1: Python → C++ IR Gap**
- Python `Workload` class builds Python-only structure
- No serialization to C++ `ir::Module`
- `pto_ir_cpp` pybind exists but not used by Python frontend
- **Impact:** Multi-backend compilation requires manual IR construction

**Issue AR2: Schedule Configuration Not Lowered**
- Python schedule combinators configure a dict
- C++ backend ignores Python schedule, uses its own defaults
- **Fix:** Bridge Python schedule to C++ `ir::ScheduleNode`

**Issue AR3: CSP Not Connected**
```
Python: Channel, Process, consume (csp.py)
   ↓ (no bridge)
C++ IR: ChannelNode, ProcessNode (csp.hpp)
   ↓ (no lowering)
Runtime: (missing)
```
- CSP primitives exist in both layers but aren't connected

### 4.3 C++ Architecture (Positive Notes)

The C++ side is well-designed:
- `graph/` - Task graph storage, ready queues, dependency tracking
- `backend/cpu_sim.cpp` - Full WorkloadLowerer with parallel_for expansion
- `backend/ascend_npu.cpp` - Code generation for NPU
- Type-safe IR nodes with proper memory management

---

## 5. Industrial Readiness

### 5.1 Error Handling

**Issue IR1: Silent Failures**
```python
def _enumerate_recursive(self, ...):
    # Unhandled workload kinds silently return []
    return []  # No warning or error
```

**Issue IR2: Type Errors Not Propagated**
- TypeChecker collects errors but they're not raised by default
- Must explicitly check `has_type_errors()`

**Issue IR3: No Runtime Error Context**
- C++ backend throws exceptions without Python context
- Stack traces don't show which task failed

### 5.2 Testing Coverage

| Component | Test File | Coverage |
|-----------|-----------|----------|
| Python Frontend | `test_python_frontend.py` | Good |
| C++ IR | `test_ir.cpp` | Good |
| C++ Backend | `test_backend.cpp` | Basic |
| Integration | (missing) | None |
| CSP Primitives | (missing) | None |
| Select/Cond | (missing) | None |

**Missing Test Areas:**
1. Python → C++ round-trip
2. CSP primitive execution
3. Dynamic axis handling
4. Error recovery
5. Multi-backend switching

### 5.3 Documentation for Users

- Examples have good docstrings
- API reference incomplete
- No tutorial or getting-started guide
- No troubleshooting section

---

## 6. Research Alignment

### 6.1 Comparison with State-of-Art

| System | Feature | PTO-RT Status |
|--------|---------|---------------|
| FlashInfer | Plan-Run model | Demonstrated in example, no runtime support |
| Triton | `@triton.jit` kernel | `@kernel` with `tl.*` is similar |
| TVM | Schedule primitives | Combinator API is less expressive |
| Halide | Compute/Schedule split | Partial - Workload combines both |
| JAX | JIT tracing | Similar tracing approach |

### 6.2 Novel Contributions

1. **Axis Type System** - Dense/DenseDyn/Ragged/Sparse is well-designed
2. **CSP Integration** - Novel for ML frameworks (if completed)
3. **Multi-Backend IR** - Clean separation of concerns

### 6.3 Missing Research Features

1. **Automatic Scheduling** - No auto-tuning
2. **Memory Planning** - No memory optimization pass
3. **Fusion** - No kernel fusion infrastructure
4. **Quantization** - No mixed-precision support

---

## 7. Recommendations and New Tasks

### 7.1 Critical (Must Fix)

| ID | Task | Priority |
|----|------|----------|
| C1 | Fix README import path (`pto.rt` → `pto_wsp`) | High |
| C2 | Implement `select`/`cond` handlers in Workload.enumerate() | High |
| C3 | Add Python → C++ IR serialization bridge | High |
| C4 | Create CSP end-to-end example with execution | High |

### 7.2 Important (Should Fix)

| ID | Task | Priority |
|----|------|----------|
| I1 | Consolidate layout systems (TensorLayout + Layout + Shard) | Medium |
| I2 | Add integration tests (Python → C++ → execution) | Medium |
| I3 | Implement schedule lowering to C++ ScheduleNode | Medium |
| I4 | Add example with actual tensor data and validation | Medium |
| I5 | Document deprecated vs current API patterns | Medium |

### 7.3 Nice to Have (Enhancement)

| ID | Task | Priority |
|----|------|----------|
| N1 | Implement IRRewriter/Pass Manager infrastructure | Low |
| N2 | Add profiling/tracing support in Program.execute() | Low |
| N3 | Create tutorial documentation | Low |
| N4 | Add automatic scheduling exploration | Low |

---

## 8. Conclusion

PTO-RT v9 has a solid foundation with well-designed C++ infrastructure and a clear vision for typed workload expressions. The main gaps are:

1. **Integration layers** between Python and C++ are incomplete
2. **CSP primitives** are declared but not connected
3. **Type safety** relies on string parsing rather than proper typing
4. **Test coverage** misses integration and advanced features

The project is suitable for research prototyping but needs the critical fixes above for production use. The design documents are comprehensive but should be updated to reflect implementation status.

---

## Appendix: Files Reviewed

### Python
- `python/pto_wsp/__init__.py`
- `python/pto_wsp/builder.py`
- `python/pto_wsp/kernel.py`
- `python/pto_wsp/workload.py`
- `python/pto_wsp/type_checker.py`
- `python/pto_wsp/types.py`
- `python/pto_wsp/csp.py`
- `python/pto_wsp/schedule.py`
- `python/pto_wsp/primitives.py`
- `python/pto_wsp/p_namespace.py`
- `python/pto_wsp/spatial.py`

### C++
- `include/pto/rt/ir/*.hpp`
- `include/pto/rt/graph/*.hpp`
- `include/pto/rt/backend/*.hpp`
- `src/pto/rt/ir/*.cpp`
- `src/pto/rt/backend/*.cpp`

### Tests
- `tests/test_python_frontend.py`
- `tests/test_ir.cpp`
- `tests/test_backend.cpp`

### Examples
- All 8 examples in `examples/`

### Documentation
- `docs/spec.md`
- `docs/analysis.md`
- `docs/design/ir-design.md`
- `docs/design/backend-arch.md`
- `README.md`
- `CLAUDE.md`
