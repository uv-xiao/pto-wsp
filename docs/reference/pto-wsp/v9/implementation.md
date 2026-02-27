# PTO-WSP v9: Implementation Guide (As-Built)

> **Date:** 2026-01-28  
> **Purpose:** Explain how the current codebase implements the feature catalog in `docs/features.md`, with concrete code entrypoints and an end-to-end view of the **codegen-first** execution model.

This document is “as-built” documentation. If `docs/features.md` or `docs/spec.md` conflict with the current code, treat this file as the source of truth.

---

## 0) Big Picture (What runs where)

PTO-WSP is split into:

- **Python frontend** (author workloads/kernels, build IR, provide runtime data)
- **C++ compilation + codegen** (typecheck/lower, emit C++ sources, build artifacts)
- **Generated artifacts** (CPU-sim shared library; NPU host/AICPU/AICore source tree)

### 0.1 CPU-sim “golden path” (end-to-end)

```
Python authoring
  @kernel + pto.* ops         @workload + P(...) loops
          |                             |
          +--------------+--------------+
                         |
                  workload_to_codegen_ir()
                  (python/pto_wsp/ir_bridge.py)
                         |
                         v
               C++ ir::Module (+ codegen metadata)
                         |
                         v
             pto_ir_cpp.compile_codegen(module, opts)
                 (src/python/pto_codegen.cpp)
                         |
         +---------------+----------------+
         |                                |
         v                                v
 emit workload_main.cpp + kernel_*.cpp   compile_sources_via_cmake()
   (C++ codegen)          (src/pto/wsp/codegen/cmake_compiler.cpp)
         |                                |
         +----------------+---------------+
                          |
                          v
                ~/.cache/pto_wsp/codegen/<name>_<key>.so
                          |
                          v
                  CodegenExecutable(dlopen)
                  (src/python/pto_codegen.cpp)
                          |
                          v
               Program.execute() / synchronize()
                (python/pto_wsp/program.py)
```

### 0.2 NPU “looks correct” codegen (compile-only in this env)

```
pto_ir_cpp.compile_codegen(target="ascend_npu")
          |
          v
emit source tree:
  host/runner_stub.cpp
  aicpu/expand.cpp        (on-device task expansion)
  aicore/dispatch.cpp     (kernel dispatch + timing)
  aicore/kernel_*.cpp     (PTO-ISA kernel sources)
  CMakeLists.txt          (device build gated)
          |
          v
Program.codegen_artifact_dir  (python/pto_wsp/program.py)
Program.execute() -> raises (codegen-only target)
```

---

## 1) Feature-to-Implementation Crosswalk (`docs/features.md`)

This section follows the numbering in `docs/features.md` and points to the implementation “hot paths”.

### 1. Axis Types (Dense/DenseDyn/Ragged/Sparse)

- **User API:** `python/pto_wsp/types.py`
- **Lowering to C++ IR:**
  - Workload params: `_convert_workload_def_codegen()` in `python/pto_wsp/ir_bridge.py`
  - Loop axes: `_convert_axis()` and `factory.create_*_axis(...)`
- **Runtime behavior (codegen-first):**
  - `DenseDyn` loop bounds come from `RuntimeContext.get_axis_size()` (ABI: `include/pto/wsp/codegen/abi/workload_abi.hpp`)
  - `Ragged`/`Sparse` use runtime **symbols** (see Section 4)
- **Validation:** `tests/test_codegen_dynamic_axes.py`, `tests/test_codegen_ragged_axis.py`, `tests/test_codegen_select_sparse.py`

### 2. Tensor and DType

- **User API:** `python/pto_wsp/types.py` (`Tensor`, `DType`, `Location`)
- **Codegen tensor binding model:**
  - Python `Tensor` views retain `base` + `index_exprs` for codegen (`python/pto_wsp/types.py`)
  - IR bridge emits `CodegenTensorArg` per kernel tensor param (base tensor id, view rank, index exprs) (`python/pto_wsp/ir_bridge.py`)
  - Generated code uses runtime **strides** to support non-contiguous views (kernel ABI: `include/pto/wsp/codegen/abi/kernel_abi.hpp`)

### 3–5. `@workload`, `@kernel`, `P` namespace

- **Workload tree construction:** `python/pto_wsp/builder.py`, `python/pto_wsp/p_namespace.py`, `python/pto_wsp/workload.py`
- **Kernel IR construction (typed, SSA-like):** `python/pto_wsp/kernel.py`
- **Bridge to C++ “codegen IR”:** `workload_to_codegen_ir()` (`python/pto_wsp/ir_bridge.py`)
  - Produces a C++ `ir::Module` with:
    - `module.tensors` table
    - `module.kernels` as `CodegenKernelDef` (typed ops, values, attrs)
    - `TaskNode` includes codegen bindings (axis args, scalar args, tensor args)

### 6. JIT Kernel API (`@kernel` / `@jit_kernel`, `pto.*`)

- **Implementation:** `python/pto_wsp/kernel.py`
  - `kernel.trace()` returns a `KernelIR` containing ops such as `Matmul`, `Store`, `RowSum`, `Exp`, etc.
- **Consumed by codegen-first path:** `python/pto_wsp/ir_bridge.py` extracts `KernelIR` and encodes it into C++ `CodegenKernelDef` attached to `ir::Module`.
- **C++ kernel codegen (PTO-ISA):** `emit_kernel_cpp()` in `src/python/pto_codegen.cpp`
  - Includes `pto/wsp/codegen/abi/ptoisa_bridge.hpp`, which provides wrappers for PTO-ISA ops used by generated kernels
  - CPU-sim timing uses PTO-ISA’s `pto::cpu_sim` counters (Section 5)

### 8–10. Scheduling APIs (`dispatch/streams/stream_by/timing/task_graph`)

- **User API exists:** `python/pto_wsp/workload.py`, `python/pto_wsp/schedule.py`
- **Codegen-first (v9) status:**
  - Schedule objects are captured on the Python `Workload` and converted into C++ `ScheduleDef` (`_convert_schedule_def()` in `python/pto_wsp/ir_bridge.py`).
  - The generated **CPU-sim artifact enforces v9 schedule semantics** inside the artifact:
    - `dispatch`: assigns each task to a worker lane (by policy + `ScalarExpr` key) and reports `Program.stats.total_cycles`
      as the **makespan** in **PTO-ISA cycles**.
    - `task_graph(window=TaskWindow(..., mode=STALL))`: enforces `task_window` (**stall-only**, `unit="tasks"`).
  - Unsupported directives are explicitly diagnosed in v9 artifacts (e.g. `stream_by`, `timing`, non-stall windows).
  - `Program.stats.task_count` is still computed from the compiled host-side plan (`python/pto_wsp/program.py`).

Entrypoint: schedule semantics are implemented in the emitted workload entrypoint
`emit_workload_cpp_from_ir_ast(...)` in `src/python/pto_codegen.cpp`.

### 11. CSP primitives (Channels/Process)

- **User API:** `python/pto_wsp/csp.py`
- **Codegen-first (v9) status:** Supported for CPU-sim when the workload root is a `Pipeline`.
  - Python → C++ IR lowering happens in `python/pto_wsp/ir_bridge.py` (channels/processes/send/consume/connect).
  - The CPU-sim artifact emits a CSPT runtime (channels + processes) and advances time **strictly** by:
    - PTO-ISA kernel-reported cycles
    - constant per-channel latency (default `0`, runtime symbol `__pto_wsp_channel_latency_cycles`)
    - explicit wait/consume semantics at channel operations
  - Validation: `tests/test_codegen_cspt_pipeline.py`, `examples/csp_pipeline/csp_pipeline_example.py`

### 12–14. Type system (Layout types, LinearLayout, TypeChecker)

- **Layout types (Dato-style facets):** `python/pto_wsp/types.py`
- **LinearLayout (F₂):** `python/pto_wsp/linear_layout.py`
- **Builder-time type checker:** `python/pto_wsp/type_checker.py` (used while constructing workloads)
- **C++ compile-time type checking:** `pto::wsp::ir::type_check(module)`
  - Invoked in the codegen entrypoint for CPU-sim and NPU codegen (`src/python/pto_codegen.cpp`)

### 15–17. C++ IR + backend/concurrency utilities

- **IR definitions:** `include/pto/wsp/ir/*` (workload nodes, axes, codegen attachments)
- **C++ codegen AST infra:** `include/pto/wsp/codegen/cpp_ast.hpp`, `include/pto/wsp/codegen/cpp_builder.hpp`, `src/pto/wsp/codegen/cpp_emitter.cpp`
- **Build + cache integration:** `include/pto/wsp/codegen/cmake_compiler.hpp`, `src/pto/wsp/codegen/cmake_compiler.cpp`
- **(Optional) legacy backends:** `include/pto/wsp/backend/*`, `src/pto/wsp/backend/*`  
  The v9 “golden path” uses `pto_ir_cpp.compile_codegen(...)` + generated artifacts (not the old CPUSimBackend registry model).

### 18. On-device task generation (NPU)

- **As implemented:** generated **AICPU expander** (`aicpu/expand.cpp`) that expands tasks from workload IR into `NpuTaskDesc[]`.
  - Emitted by `emit_aicpu_expand_cpp_from_ir(...)` (`src/python/pto_codegen.cpp`)
  - ABI: `include/pto/wsp/codegen/abi/npu_plan_abi.hpp`

The `docs/features.md` section mentions a “bytecode interpreter” style; PTO-WSP’s current v9 implementation uses **generated AICPU C++** instead of a generic bytecode VM, but the goal is the same: avoid host-side O(tasks) expansion and enable dynamic shapes.

---

## 2) CPU-sim Codegen-First Runtime: Step-by-step

### 2.1 Authoring (Python)

- Define kernels with typed signatures and `pto.*` primitives (`python/pto_wsp/kernel.py`)
- Define workloads using `@workload` and `P(...)` loops (`python/pto_wsp/builder.py`, `python/pto_wsp/p_namespace.py`)

### 2.2 Build C++ IR (Python → C++)

`Program._compile_codegen()` (`python/pto_wsp/program.py`) calls:

1. `workload_to_codegen_ir(...)` (`python/pto_wsp/ir_bridge.py`)
2. `pto_ir_cpp.compile_codegen(module, opts)` (pybind → `src/python/pto_codegen.cpp`)

### 2.3 Emit sources (C++)

The C++ binding compiles the C++ `ir::Module` into sources:

- `workload_main.cpp`: the workload entrypoint (`emit_workload_cpp_from_ir_ast(...)`)
- `kernel_<name>.cpp`: one kernel per `CodegenKernelDef` (`emit_kernel_cpp(...)`)

The workload entrypoint calls kernels using the ABI in:

- `include/pto/wsp/codegen/abi/workload_abi.hpp`
- `include/pto/wsp/codegen/abi/kernel_abi.hpp`

### 2.4 Build + cache the shared library (C++)

`compile_sources_via_cmake(...)` (`src/pto/wsp/codegen/cmake_compiler.cpp`) writes a small CMake project into:

- `~/.cache/pto_wsp/codegen/src_<name>_<key>/`
- `~/.cache/pto_wsp/codegen/build_<name>_<key>/`

And builds:

- `~/.cache/pto_wsp/codegen/<name>_<key>.so`

The cache key includes compiler/cmake versions plus ABI headers and PTO-ISA umbrella header to invalidate correctly when dependencies change.

### 2.5 Load + run

- `CodegenExecutable` (`src/python/pto_codegen.cpp`) `dlopen()`s the `.so` and resolves `<module>_main`.
- `Program.execute()` spawns a thread, collects numpy arrays, and calls:
  - `CodegenExecutable.run_with_symbols(arrays, axis_sizes, symbols_u64, symbols_ptr)`

---

## 3) Codegen ABI Surface (What generated code expects)

### 3.1 Workload ABI (`RuntimeContext`)

`include/pto/wsp/codegen/abi/workload_abi.hpp` defines:

- `get_axis_size(ctx, name)` for `DenseDyn`
- `get_symbol_u64(ctx, id)` / `get_symbol_ptr(ctx, id)` for runtime-bound values (Section 4)
- `get_tensor_ptr(ctx, tensor_id)` / `get_tensor_stride(ctx, tensor_id, dim)` for base tensors + strides

### 3.2 Kernel ABI (`KernelTaskDesc`)

`include/pto/wsp/codegen/abi/kernel_abi.hpp` defines:

- `task->args[]` (u64): loop indices, tail/mask dims, scalar params
- `task->tensor_ptrs[]`, `task->tensor_strides[]` (u64): tensor base pointers and 2D view strides

Generated kernels are “PTO-ISA kernels”: they use `pto_wsp::ptoisa::{TLOAD/TMATMUL/TSTORE/...}` wrappers and return a u64
**cycle report**.

### 3.3 Two kernel authoring paths (v9)

PTO-WSP supports two complementary ways to author kernels:

- `pto.*` IR-traced kernels (`@kernel`): PTO-WSP traces a high-level tile-language IR and emits PTO-ISA kernels.
- `ptoisa.*` instruction-traced kernels (`@ptoisa_kernel`): PTO-WSP traces a restricted Python authoring API and emits a C++ kernel body that calls `pto_wsp::ptoisa::...` wrappers directly.

`@kernel(cpp_src=...)` remains available as an escape hatch for manual C++ bodies.

---

## 4) Dynamic Axes + Runtime Symbols (no whole-artifact rebuild)

### 4.1 Stable symbol IDs

Both CPU-sim and NPU codegen use a stable symbol ID scheme:

```
symbol_id = FNV-1a_64("some_name")
```

- Python computes the same hash in `Program._fnv1a_64()` (`python/pto_wsp/program.py`)
- C++ uses the matching hash for code emission (`src/python/pto_codegen.cpp`)

### 4.2 Feeding values at runtime (Python API)

```
program.set_axis_sizes({"b": 4})        # DenseDyn loop bounds
program.set_symbol_u64("outer", 8)      # e.g., ragged outer size
program.set_symbol_ptr("lengths", arr)  # e.g., ragged lengths / CSR buffers
program.execute()
program.synchronize()
```

### 4.3 Where they are consumed

```
Program.set_symbol_*()
        |
        v
run_with_symbols(..., symbols_u64, symbols_ptr)
        |
        v
RuntimeContext.get_symbol_u64/get_symbol_ptr
        |
        v
generated workload / kernels read symbol values
        |
        v
same compiled .so works across runs without rebuild
```

Validated by:

- `tests/test_codegen_dynamic_symbols.py` (scalar symbols)
- `tests/test_codegen_ragged_axis.py` (ragged lengths/outer)
- `tests/test_codegen_select_sparse.py` (CSR indices/indptr)

---

## 5) Timing / Cycles (single source of truth)

### 5.1 CPU-sim: PTO-ISA cycle counter

Generated kernels reset and read PTO-ISA’s cpu-sim cycle counter:

```
pto::cpu_sim::reset_cycles();
... PTO-ISA ops ...
cycles += pto::cpu_sim::read_cycles();
return cycles;
```

The v9 codegen-first CPU-sim artifact uses **PTO-ISA kernel cycle reports as the only timebase**, and surfaces them as:

- **Non-CSP workloads**: `Program.stats.total_cycles` is the **makespan** computed by the artifact scheduler:
  - each task returns `kernel_cycles`
  - `dispatch` selects a worker lane
  - `task_window` (stall-only) may delay issue time
  - makespan = `max(worker_end_time)`
- **CSP/CSPT pipelines**: `Program.stats.total_cycles` is the **max process local time** (CSPT),
  derived from kernel cycles + channel latency + waits.

### 5.1.1 Wall-clock tracing vs cycle time

PTO‑RT also provides optional tracing (`Program.enable_tracing(...)`) that records **wall-clock** timestamps for debugging/profiling.
These trace durations are **not** the CSPT/scheduler “time” and must not be used as a substitute for `total_cycles`.

### 5.2 NPU: cycle report propagation into task descriptor

For NPU codegen, the AICore dispatcher records the kernel-returned cycle report into:

- `NpuTaskDesc.t_end` (with `t_begin` reserved for future real device timestamps)

ABI: `include/pto/wsp/codegen/abi/npu_plan_abi.hpp`

---

## 6) Tail / Partial Tiles (dynamic axes without recompilation)

When dynamic axes produce partial tiles (e.g., last tile not full), the generated workload appends per-tensor valid dimensions to the axis-args region:

```
task.args layout:
  [ user_axis_args...,
    (valid_row, valid_col) x tensor_param_count,
    scalar_args... ]

task.num_axis_args = len(user_axis_args) + 2 * tensor_param_count
```

The kernel reads those `(valid_row, valid_col)` pairs and uses them to:

- size `pto::GlobalTensor` shapes for TLOAD/TSTORE
- construct tiles with runtime-valid dims (PTO-ISA supports dynamic valid dims)

This avoids recompiling the artifact when the runtime size changes.

---

## 7) Where to look next (common entrypoints)

- “Golden path” example: `examples/validated_matmul/validated_matmul_example.py`
- Program compile/execute: `python/pto_wsp/program.py`
- IR bridge (Python → C++ module): `python/pto_wsp/ir_bridge.py`
- Codegen + loaders (C++ bindings): `src/python/pto_codegen.cpp`
- CMake compiler/cache: `src/pto/wsp/codegen/cmake_compiler.cpp`
- ABIs:
  - `include/pto/wsp/codegen/abi/kernel_abi.hpp`
  - `include/pto/wsp/codegen/abi/workload_abi.hpp`
  - `include/pto/wsp/codegen/abi/npu_plan_abi.hpp`

---

## Appendix A) Quick validation commands

```bash
# C++ unit tests
ctest --test-dir build

# Python tests
PYTHONPATH=python python -m pytest -q

# Run all examples (CPU-sim codegen path)
for f in examples/*/*_example.py; do
  PYTHONPATH=python python "$f"
done
```
