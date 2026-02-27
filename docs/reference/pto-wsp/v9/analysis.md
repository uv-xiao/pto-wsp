# PTO-WSP v9: Design Analysis (As-Built)

> **Date:** 2026-01-28  
> **Purpose:** Explain *why* PTO‑RT looks the way it does in v9, and how the major features compose.  
> For a code-level “what runs where” guide, see `docs/implementation.md`. For the API surface, see `docs/spec.md`.

PTO‑RT is a runtime extension for dynamic LLM workloads, built around:

- **Typed workload expressions** (`@workload` + `P(...)`) for data-parallel and CSP pipeline-parallel graphs
- **A C++ IR** that is the single source of truth for compilation and codegen
- **Codegen-first execution**: backends emit artifacts that *execute the semantics* (not Python bootstrap)
- **PTO‑ISA kernels** as the kernel substrate (CPU simulation + NPU emission)

---

## 1) The “codegen-first” principle

The v9 non-negotiable is: once you call `compile(...)`, the behavior must be realized by the generated artifact.
Python is for authoring/bridging IR and providing runtime inputs (tensors, axis sizes, symbols).

CPU simulation path:

```
Python authoring (workload + kernels)
  -> build C++ ir::Module (pybind)
  -> C++ compile + codegen (emit C++ sources)
  -> build .so (cached)
  -> dlopen + execute (artifact semantics)
```

NPU path (in this environment it is “emit-only”):

```
Python authoring
  -> build C++ ir::Module
  -> emit host/aicpu/aicore source tree
  -> (device build/run requires Ascend/CANN)
```

Implication: features like `cond`, scheduling, CSP/CSPT, dynamic axes must be implemented in **C++ IR + emitted runtime**, not as a Python driver loop.

---

## 2) Two parallelism modes

### 2.1 Data-parallel workloads

`P(...)` expresses independent iteration, and `P.seq(...)` expresses explicit sequential dependencies.

```python
from pto_wsp import workload, P, kernel, In, Out, Tensor

@kernel
def op(x: In[Tensor], y: Out[Tensor]): ...

@workload
def w(batch):
    for b in P(batch):
        op[b](x=X[b], y=Y[b])
```

### 2.2 CSP pipeline-parallel workloads (CSPT)

PTO‑RT supports CSP-style pipelines via channels/processes. v9 treats pipeline time as **CSPT**:
each process maintains a local simulated time in **PTO‑ISA cycles**.

Timebase:

```
time_cycles := kernel_cycles + channel_latency_cycles + explicit_wait_cycles
```

Channel latency is modeled as a constant per channel (default `0` in v9).

---

## 3) Runtime predicates: ScalarExpr + tensor-driven slots

### 3.1 Why ScalarExpr exists

Dynamic workloads need runtime predicates (`cond`, schedule keys) that depend on:
- axis variables / task parameters
- runtime symbols (dynamic axes, ragged/sparse metadata)
- values derived from tensors (e.g., routing decisions)

v9 represents these predicates as a structured expression IR (**ScalarExpr**) evaluated inside artifacts.

### 3.2 Tensor → scalar conversion (“slots”)

Many decisions are data-dependent: kernels produce tensor values, but `cond`/dispatch needs scalars/bools.
v9 provides runtime **slots** (u64/i64/bool) as a bridge:

- Workload primitives can write slots:
  - `slot_set_u64(slot, value)` (tests/debug)
  - `slot_load_u64(slot, tensor_view, row=0, col=0)` (tensor→scalar)
- Predicates/schedule keys can read slots:
  - `slot_u64(i)` inside ScalarExpr

This enables data-dependent execution **without recompiling artifacts** and without a Python control loop.

---

## 4) Scheduling: what v9 enforces

Scheduling is programmable, but v9 enforces only the behavior-changing subset needed for the core deliverable:

- **enforced in CPU-sim artifacts:**
  - `dispatch(...)`
  - `task_window(..., mode=STALL)` (stall-only)
- **emitted/preserved for NPU artifacts:**
  - `dispatch(...)`
  - `task_window(..., mode=STALL)`
- **explicitly unsupported in v9 artifacts:**
  - advanced stream/timing controls beyond the above

Rationale: `dispatch` and `task_window` are the minimum schedule knobs that must influence artifact behavior for programmability.

---

## 5) Kernel authoring modes (3 modes, same artifact)

v9 supports three complementary ways to author kernels that all compile into the same artifact and share ABI + cycle reporting:

1) **`pto.*` IR-traced kernels** (`@kernel` + `pto.*` ops)  
   High-level kernel IR is traced in Python and lowered by C++ codegen to PTO‑ISA calls.

2) **`ptoisa.*` instruction-traced kernels** (`@ptoisa_kernel`)  
   A restricted Python DSL records PTO‑ISA tile instructions and emits a C++ kernel body calling `pto_wsp::ptoisa::...`.

3) **File-based custom C++ kernels** (`cpp_body_path` / `cpp_tu_path`)  
   Escape hatch for handwritten PTO‑ISA kernels in C++ files/snippets compiled into the artifact.

### Path A: TopK is not a primitive op

TopK-style logic is intentionally not exposed as `pto.topk`. Instead, write it using PTO‑ISA tile primitives via:
- `@ptoisa_kernel`, or
- `@kernel(cpp_body_path=...)` / `@kernel(cpp_tu_path=...)`, or
- a `pto.*` traced kernel lowered to PTO‑ISA calls.

This keeps the core op surface small while still allowing full functionality through kernels.

---

## 6) NPU backend “looks correct” (v9 deliverable in this env)

Without Ascend/CANN, v9’s NPU backend work is validated by:
- emitted host/AICPU/AICore source trees
- compile-only/source-inspection tests

Key v9 constraints:
- keep PTO‑RT schedule programmability: preserve `dispatch` + `task_window`
- perform on-device task expansion on AICPU conceptually (emitted AICPU expander)
- use PTO‑ISA to generate AICore kernel sources

For design/implementation notes, see `docs/design/npu-backend_support.md`.

---

## 7) Validation philosophy

The primary “works as shipped” signal is:

- all unit/integration tests pass (`pytest`, `ctest`)
- all examples under `examples/` are **self-validating**:
  - each example includes a golden reference implementation, PTO‑RT implementation, and a runner/checker
  - runner checks numeric correctness and tolerance-based cycle expectations

See `examples/README.md` for the full list and how to run them.

