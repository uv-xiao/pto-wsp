# pto-runtime Integration (v10) Implementation Plan (updated: Phase 1 runnable + codegen-complete)

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Deliver a **complete** PTO‑WSP → pto-runtime integration path where PTO‑WSP:
1) performs **C++ codegen** that emits a **visible `host_build_graph` source tree** artifact, and then
2) **wraps pto-runtime tooling** (Python) to compile+run that artifact for:
   - `a2a3sim` (local simulation; must work end-to-end in CI), and
   - `a2a3` (real device; toolchain-gated but fully wired in code).

**Architecture (Option A):**
- `pto-runtime` is a pinned submodule at `3rdparty/pto-runtime` (single source of truth).
- PTO‑WSP C++ codegen emits a pto-runtime-shaped source tree:
  - `kernels/orchestration/*.cpp` (orchestration function that builds the runtime task graph)
  - `kernels/aiv/*.cpp` (incore kernels for `a2a3`)
  - `kernels/aiv_sim/*.cpp` (simulation kernels for `a2a3sim`)
  - `kernels/kernel_config.py` (declares orchestration + kernels, similar to pto-runtime examples)
- PTO‑WSP Python wraps pto-runtime:
  - `RuntimeBuilder(platform=...)` to build runtime libs (host + aicpu + aicore binaries)
  - `PTOCompiler.compile_orchestration(...)` to compile orchestration `.so`
  - `PTOCompiler.compile_incore(...)` to compile kernels; `elf_parser.extract_text_section(...)`
  - `bindings.{bind_host_binary,set_device,register_kernel,launch_runtime}`

**Tech Stack:** git submodule, C++ codegen, pto-runtime Python tooling, pytest, (optional) Ascend toolkit for `a2a3`.

---

## Execution context (repo policy)

This repo’s current preference is to **execute in this workspace and this conversation/session** (no worktrees, no separate
execution context). The “worktree” steps in Task 0 are historical; skip them. Even though this plan references the
`executing-plans` workflow style, implementation for this repo should proceed in-place in the current Codex CLI session.

## Constraints / invariants (do not regress)

- The **canonical Phase 1 artifact is the emitted source tree** (must be visible on disk).
- PTO‑WSP must be able to **compile+run** after emitting (not manual steps).
- `dispatch(policy)` meaning is **AICPU scheduler assignment** (multi-AICPU sharding). This is currently **not supported**
  by pto-runtime, and must stay a **documented gap** with **explicit TODO placeholders** in generated code.
- CSP requires **auto-generated orchestration logic** and is **not claimed** on the pto-runtime path until implemented.

## Supported subset for “Phase 1 runnable” (explicit)

To keep semantics honest while still delivering a runnable integration, Phase 1 initially supports:

- workload forms: `sequential`, `combine`, and **single-level** `parallel_for` expanded on host
- kernel shapes: contiguous 1D/2D dense tensors (flattened for execution)
- kernel ops: elementwise `Add/Sub/Mul/Div/Max/Min`, unary `Exp/Rsqrt`, and scalar broadcast
- no `Matmul` lowering yet (gap)
- no CSP (`Channel/Process/consume`) on the pto-runtime path yet (gap)
- no multi-AICPU dispatch mapping yet (gap; placeholder in generated orchestration)

If a workload/kernel exceeds this subset:
- `target="pto_runtime_a2a3sim"` / `target="pto_runtime_a2a3"` must raise a clear `CompileError` naming the first unsupported
  IR feature and link the user to `docs/future/pto_runtime_gaps.md`.

## Tracker (keep this updated during execution)

**Legend:** `[x] done` / `[ ] todo` / `[~] in progress`

- [x] Pin `pto-runtime` as submodule (`3rdparty/pto-runtime`)
- [x] Add pto-runtime import bridge (`python/pto_wsp/pto_runtime_bridge.py`)
- [x] Add emit-only codegen targets (`a2a3sim_codegen`, `a2a3_codegen`) emitting `host_build_graph` sources
- [x] Update v10 docs to include pto-runtime as a codegen target (cpu_sim / pto_runtime / aie)
- [x] Make Phase 1 **runnable** for `a2a3sim`: emit → build → compile → register kernels → execute → verify outputs
- [x] Make Phase 1 **runnable** for `a2a3`: same pipeline, toolchain-gated (`ASCEND_HOME_PATH`)
- [x] Upgrade codegen from “scaffold stubs” to “codegen-complete” for the supported subset
- [x] Add/refresh docs: supported subset + gaps, link each gap to a concrete TODO symbol in code
- [x] Keep `docs/future/v10_plan.md`, `docs/future/v10_tracker.md`, and `docs/future/v10_implementation.md` synchronized (as of 2026-02-03)

---

### (Bootstrap, historical) Task 0: Create an isolated worktree for integration work

**Files:**
- n/a (git worktree only)

**Step 1: Create a worktree**

Run: `git worktree add ../pto-rt-pto-runtime-integration -b feat/pto-runtime-integration`  
Expected: a new directory `../pto-rt-pto-runtime-integration` on branch `feat/pto-runtime-integration`.

**Step 2: Initialize submodules in the worktree**

Run: `cd ../pto-rt-pto-runtime-integration && git submodule update --init --recursive`  
Expected: submodules checked out (no missing gitlinks).

**Step 3: Confirm the worktree is clean**

Run: `git status --porcelain=v1`  
Expected: empty.

---

### Task 1: Add `pto-runtime` as a submodule (Option A)

**Files:**
- Modify: `.gitmodules`
- Create: `3rdparty/pto-runtime` (gitlink)
- Modify: `3rdparty/README.md`

**Step 1: Verify repo is clean enough to proceed**

Run: `git status --porcelain=v1`  
Expected: no *unexpected* deletions in `.codex/` or docs (if present, stop and reconcile before continuing).

**Step 2: Add the submodule**

Run: `git submodule add https://github.com/uv-xiao/pto-runtime 3rdparty/pto-runtime`  
Expected: `.gitmodules` gains a new entry for `3rdparty/pto-runtime`.

**Step 3: Initialize submodules**

Run: `git submodule update --init --recursive`  
Expected: `git submodule status` shows both `3rdparty/pto-isa` and `3rdparty/pto-runtime`.

**Step 4: Update docs for third-party deps**

Modify: `3rdparty/README.md` to include `pto-runtime` (setup + update instructions).

**Step 5: Commit**

```bash
git add .gitmodules 3rdparty/README.md 3rdparty/pto-runtime
git commit -m "build: add pto-runtime submodule"
```

---

### Task 2: Update v10 docs to specify *how* PTO‑WSP integrates with pto-runtime

**Files:**
- Modify: `docs/future/pto_runtime_integration.md`
- Modify: `docs/future/v10_pto_runtime_interface.md`
- Modify: `docs/future/v10_implementation.md`

**Step 1: Specify the integration method (codegen → pto-runtime source tree)**

Modify: `docs/future/pto_runtime_integration.md` to state explicitly:
- PTO‑WSP performs **C++ codegen** that emits a pto-runtime **host_build_graph** source tree:
  - orchestration C++ (`kernels/orchestration/*.cpp`)
  - kernel sources (`kernels/aiv/*.cpp` or equivalent)
  - a pto-runtime `kernel_config.py` describing orchestration + kernels
- Python is used to *invoke* pto-runtime tooling (builder/compiler), but not to “execute semantics”.

**Step 2: Align the interface contract wording**

Modify: `docs/future/v10_pto_runtime_interface.md` to clarify:
- Phase 1 entry payload is a pto-runtime **orchestration `.so`** built from generated C++ sources
- Phase 2 remains the true task-buffer/expander target
- `dispatch(policy)` intent is **AICPU scheduler assignment** (not `core_type`), and must be treated as a “gap” until runtime APIs exist

**Step 3: Update implementation plan doc to match the above**

Modify: `docs/future/v10_implementation.md` to add a short “pto-runtime codegen pipeline” subsection:
- where codegen lives (PTO‑WSP C++)
- where the generated tree goes (artifact dir)
- how python invokes pto-runtime builder (import story)

**Step 4: Commit**

```bash
git add docs/future/pto_runtime_integration.md docs/future/v10_pto_runtime_interface.md docs/future/v10_implementation.md
git commit -m "docs: specify pto-runtime integration method"
```

---

### Task 3: Document pto-runtime capability gaps PTO‑WSP depends on (and link to placeholders)

**Files:**
- Create: `docs/future/pto_runtime_gaps.md`
- Modify: `docs/future/pto_runtime_integration.md`
  - (follow-up edit to add links once placeholders exist)

**Step 1: Write the gaps doc**

Create: `docs/future/pto_runtime_gaps.md` with:
- dispatch → AICPU scheduler assignment (not `core_type`) support gap
- CSP channel semantics and deadlock diagnostics gap
- Phase 2 task-buffer / true `task_window` backpressure gap
- policy registry gap
- slots/symbol ABI + tensor→slot materialization gap
- package/manifest gap
- Python import ergonomics gap

**Step 2: Ensure integration notes point to the gaps doc**

Modify: `docs/future/pto_runtime_integration.md`:
- reference `docs/future/pto_runtime_gaps.md` in the “dispatch” and “CSP” sections as the source of truth for missing runtime support.

**Step 3: Commit**

```bash
git add docs/future/pto_runtime_gaps.md docs/future/pto_runtime_integration.md
git commit -m "docs: document pto-runtime integration gaps"
```

---

### Task 4: Add a Python-level import bridge for `pto-runtime` tooling (no execution yet)

**Files:**
- Create: `python/pto_wsp/pto_runtime_bridge.py`
- Test: `tests/test_pto_runtime_bridge_import.py`

**Step 1: Write a failing test (import-level)**

Create: `tests/test_pto_runtime_bridge_import.py`

```python
def test_pto_runtime_bridge_imports_runtime_builder():
    import pto_wsp.pto_runtime_bridge as b

    rb = b.import_runtime_builder()
    assert hasattr(rb, "RuntimeBuilder")
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_pto_runtime_bridge_import.py -q`  
Expected: FAIL (bridge doesn’t exist).

**Step 3: Implement minimal bridge**

Create: `python/pto_wsp/pto_runtime_bridge.py`:
- locate pto-runtime python module directory via:
  - env var `PTO_RUNTIME_PATH` (preferred override), else
  - repo-relative `3rdparty/pto-runtime`
- prepend `.../python` to `sys.path` (only for the duration of the import helper)
- expose `import_runtime_builder()` that imports `runtime_builder` and returns the module
- raise a clear `ImportError` message if not found

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_pto_runtime_bridge_import.py -q`  
Expected: PASS.

**Step 5: Commit**

```bash
git add python/pto_wsp/pto_runtime_bridge.py tests/test_pto_runtime_bridge_import.py
git commit -m "python: add pto-runtime import bridge"
```

---

### Task 5: Add CMake configuration hook for `pto-runtime` (no build integration yet)

**Files:**
- Modify: `CMakeLists.txt`
- Modify: `README.md` (optional)
- Test: n/a (configure-level check)

**Step 1: Add a cache variable**

Modify: `CMakeLists.txt` to introduce:
- `PTO_RUNTIME_PATH` (default: `${CMAKE_SOURCE_DIR}/3rdparty/pto-runtime`)
- a configure-time check that the path exists; if not, print a clear hint:
  - “Initialize submodules: git submodule update --init --recursive”

**Step 2: Verify configuration succeeds**

Run: `cmake -B build -DPTO_ISA_PATH=3rdparty/pto-isa -DPTO_RUNTIME_PATH=3rdparty/pto-runtime`  
Expected: configure succeeds; no compilation required yet.

**Step 3: Commit**

```bash
git add CMakeLists.txt
git commit -m "build: add PTO_RUNTIME_PATH cmake option"
```

---

### Task 6: Implement Phase 1 *emit-only* pto-runtime codegen target (`a2a3sim_codegen`)

**Files:**
- Create: `include/pto/wsp/codegen/pto_runtime_host_build_graph.hpp`
- Create: `src/pto/wsp/codegen/pto_runtime_host_build_graph.cpp`
- Modify: `src/python/pto_codegen.cpp`
- Test: `tests/test_pto_runtime_codegen_emit.py`

**Step 1: Write a failing test for the new target**

Create: `tests/test_pto_runtime_codegen_emit.py`

```python
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from pto_wsp import DType, In, Out, Scalar, Tensor, Tile, kernel, pto, workload


def test_a2a3sim_codegen_emits_pto_runtime_tree():
    x_data = np.arange(16, dtype=np.float32).reshape(4, 4)
    y_data = np.zeros((4, 4), dtype=np.float32)
    x = Tensor(data=x_data, shape=x_data.shape, dtype=DType.F32)
    y = Tensor(data=y_data, shape=y_data.shape, dtype=DType.F32)

    @kernel
    def scale(x_tile: In[Tile[4, 4, DType.F32]], y_tile: Out[Tile[4, 4, DType.F32]], s: Scalar[DType.F32]):
        out = pto.mul(x_tile, s)
        pto.store(y_tile, out)

    @workload
    def wl():
        scale(x_tile=x, y_tile=y, s=1.0)

    program = wl().named("pto_runtime_emit_smoke").compile(target="a2a3sim_codegen")
    assert program.using_cpp_backend
    assert program.codegen_artifact_dir

    artifact_dir = program.codegen_artifact_dir
    expected = [
        os.path.join("kernels", "kernel_config.py"),
        os.path.join("kernels", "orchestration", "pto_wsp_orch.cpp"),
    ]
    for rel in expected:
        assert os.path.exists(os.path.join(artifact_dir, rel))

    with open(os.path.join(artifact_dir, "kernels", "orchestration", "pto_wsp_orch.cpp"), "r", encoding="utf-8") as f:
        src = f.read()
    assert "TODO_PTO_RUNTIME_MULTI_AICPU_DISPATCH" in src
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_pto_runtime_codegen_emit.py -q`  
Expected: FAIL (unsupported target).

**Step 3: Add a dedicated pto-runtime host_build_graph emitter**

Create: `include/pto/wsp/codegen/pto_runtime_host_build_graph.hpp`

```cpp
#pragma once

#include "pto/wsp/backend/backend.hpp"
#include "pto/wsp/ir/ir.hpp"

#include <map>
#include <string>

namespace pto::wsp::codegen::pto_runtime {
std::map<std::string, std::string> emit_host_build_graph_sources(const ir::Module& module,
                                                                 const backend::CompileOptions& options);
}  // namespace pto::wsp::codegen::pto_runtime
```

Create: `src/pto/wsp/codegen/pto_runtime_host_build_graph.cpp` implementing:
- a minimal tree layout compatible with pto-runtime examples:
  - `kernels/kernel_config.py` (points to orchestration + kernel sources)
  - `kernels/orchestration/pto_wsp_orch.cpp` (function `build_pto_wsp_graph`)
- a clear placeholder function in the generated orchestration source:
  - marker string `TODO_PTO_RUNTIME_MULTI_AICPU_DISPATCH`
  - current behavior returns scheduler id `0` for all tasks

**Step 4: Wire a new codegen target in pybind compile_codegen**

Modify: `src/python/pto_codegen.cpp`:
- accept `target == "a2a3sim_codegen"` (and optionally `target == "a2a3_codegen"`)
- call `pto::wsp::codegen::pto_runtime::emit_host_build_graph_sources(...)`
- emit via `emit_sources_to_cache(..., suffix="a2a3sim")`
- return `can_execute=false` and `artifact_dir=...`

**Step 5: Run test to verify it passes**

Run: `python -m pytest tests/test_pto_runtime_codegen_emit.py -q`  
Expected: PASS.

**Step 6: Commit**

```bash
git add include/pto/wsp/codegen/pto_runtime_host_build_graph.hpp \
  src/pto/wsp/codegen/pto_runtime_host_build_graph.cpp \
  src/python/pto_codegen.cpp \
  tests/test_pto_runtime_codegen_emit.py
git commit -m "codegen: emit pto-runtime host_build_graph source tree (a2a3sim_codegen)"
```

---

### Task 7: Add placeholder hooks for unsupported multi-AICPU dispatch (and link gaps doc to code)

**Files:**
- Modify: `docs/future/pto_runtime_gaps.md`
- (Optional) Modify: `include/pto/wsp/codegen/pto_runtime_host_build_graph.hpp`

**Step 1: Add explicit references to placeholders**

Modify: `docs/future/pto_runtime_gaps.md` to reference:
- placeholder marker `TODO_PTO_RUNTIME_MULTI_AICPU_DISPATCH` in generated `kernels/orchestration/pto_wsp_orch.cpp`
- the PTO‑WSP emitter entrypoint `pto::wsp::codegen::pto_runtime::emit_host_build_graph_sources` as “where to implement”

**Step 2: Commit**

```bash
git add docs/future/pto_runtime_gaps.md
git commit -m "docs: link pto-runtime gaps to placeholder hooks"
```

---

### Task 8: Scaffold Phase 1 execution entrypoint (document-only, no runnable claim)

**Files:**
- Create: `docs/future/v10_pto_runtime_phase1_scaffold.md`

**Step 1: Write Phase 1 scaffold note**

Create: `docs/future/v10_pto_runtime_phase1_scaffold.md` describing:
- Phase 1 is a host-built graph orchestration `.so`
- what semantics it can/cannot claim (per `docs/future/v10_pto_runtime_interface.md`)
- how dispatch/CSP/task_window must be labeled in capability matrix until gaps close

**Step 2: Commit**

```bash
git add docs/future/v10_pto_runtime_phase1_scaffold.md
git commit -m "docs: scaffold Phase 1 pto-runtime integration note"
```

---

### Task 9: Baseline verification

**Step 1: Run existing tests**

Run: `python -m pytest tests/`  
Expected: PASS (or only pre-existing failures unrelated to this work).

**Step 2: (If C++ build is available) configure + build**

Run:
```bash
cmake -B build -DPTO_ISA_PATH=3rdparty/pto-isa -DPTO_RUNTIME_PATH=3rdparty/pto-runtime
cmake --build build
```
Expected: build succeeds.

---

### Task 10: Update v10 docs to reflect “complete codegen” + runnable pto-runtime targets

**Files:**
- Modify: `docs/future/pto_runtime_integration.md`
- Modify: `docs/future/v10_pto_runtime_interface.md`
- Modify: `docs/future/v10_implementation.md`
- Modify: `docs/future/v10_plan.md`
- Modify: `docs/future/v10_tracker.md`

**Step 1: Update wording (complete codegen)**

Update the docs above to explicitly define Phase 1 “complete integration” as:
- emitted source tree **plus**
- PTO‑WSP-wrapped build+run pipeline (**a2a3sim** runnable, **a2a3** wired/toolchain-gated)

**Step 2: Keep the supported subset + gaps honest**

Ensure the docs above:
- list the initial supported subset (same list as this plan)
- link unsupported features to `docs/future/pto_runtime_gaps.md`

**Step 3: Commit**

```bash
git add docs/future/pto_runtime_integration.md docs/future/v10_pto_runtime_interface.md \
  docs/future/v10_implementation.md docs/future/v10_plan.md docs/future/v10_tracker.md
git commit -m "docs(v10): define Phase 1 as codegen-complete + runnable pto-runtime targets"
```

---

### Task 11: Expand `pto_runtime_bridge` imports (runtime_builder + compiler + bindings + elf_parser)

**Files:**
- Modify: `python/pto_wsp/pto_runtime_bridge.py:55`
- Test: `tests/test_pto_runtime_bridge_import.py`

**Step 1: Write failing tests**

Extend `tests/test_pto_runtime_bridge_import.py`:

```python
def test_pto_runtime_bridge_imports_modules():
    import pto_wsp.pto_runtime_bridge as b

    assert hasattr(b.import_runtime_builder(), "RuntimeBuilder")
    assert hasattr(b.import_pto_compiler(), "PTOCompiler")
    assert hasattr(b.import_bindings(), "bind_host_binary")
    assert hasattr(b.import_elf_parser(), "extract_text_section")
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_pto_runtime_bridge_import.py -q`  
Expected: FAIL (`import_*` helpers missing).

**Step 3: Implement minimal helpers**

Update `python/pto_wsp/pto_runtime_bridge.py` to add:
- `import_pto_compiler() -> ModuleType`
- `import_bindings() -> ModuleType`
- `import_elf_parser() -> ModuleType`

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_pto_runtime_bridge_import.py -q`  
Expected: PASS.

**Step 5: Commit**

```bash
git add python/pto_wsp/pto_runtime_bridge.py tests/test_pto_runtime_bridge_import.py
git commit -m "python: expand pto-runtime bridge imports"
```

---

### Task 12: Define the Phase 1 orchestration argument ABI (Python helper + doc)

**Files:**
- Create: `python/pto_wsp/pto_runtime_abi.py`
- Test: `tests/test_pto_runtime_abi.py`
- Modify: `docs/future/v10_pto_runtime_interface.md`

**Step 1: Write failing test**

Create `tests/test_pto_runtime_abi.py`:

```python
import numpy as np

from pto_wsp.pto_runtime_abi import build_orch_func_args


def test_build_orch_func_args_layout():
    a = np.zeros((4, 4), dtype=np.float32)
    b = np.zeros((4, 4), dtype=np.float32)

    args = build_orch_func_args([a, b])
    assert args == [int(a.ctypes.data), a.nbytes, int(b.ctypes.data), b.nbytes]
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_pto_runtime_abi.py -q`  
Expected: FAIL (module missing).

**Step 3: Implement ABI helper**

Create `python/pto_wsp/pto_runtime_abi.py`:

```python
from __future__ import annotations

from typing import Iterable

import numpy as np


def build_orch_func_args(arrays: Iterable[np.ndarray]) -> list[int]:
    out: list[int] = []
    for a in arrays:
        if not isinstance(a, np.ndarray):
            raise TypeError(f"expected numpy.ndarray, got {type(a)}")
        out.append(int(a.ctypes.data))
        out.append(int(a.nbytes))
    return out
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_pto_runtime_abi.py -q`  
Expected: PASS.

**Step 5: Document ABI in interface doc**

Update `docs/future/v10_pto_runtime_interface.md` to define:
- Phase 1 orchestration arg ABI: `[ptr0, nbytes0, ptr1, nbytes1, ...]`

**Step 6: Commit**

```bash
git add python/pto_wsp/pto_runtime_abi.py tests/test_pto_runtime_abi.py docs/future/v10_pto_runtime_interface.md
git commit -m "python: define Phase 1 pto-runtime orchestration ABI"
```

---

### Task 13: Emit platform-correct kernel sources (`aiv_sim` vs `aiv`) and platform-selecting `kernel_config.py`

**Files:**
- Modify: `src/pto/wsp/codegen/pto_runtime_host_build_graph.cpp:1`
- Test: `tests/test_pto_runtime_codegen_emit.py`

**Step 1: Write a failing test for sim kernel sources**

Extend `tests/test_pto_runtime_codegen_emit.py` to assert:
- for `a2a3sim_codegen` we emit under `kernels/aiv_sim/`
- sim kernel sources have no PTO-ISA includes and export `extern "C" void kernel_<name>(int64_t* args)`

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_pto_runtime_codegen_emit.py -q`  
Expected: FAIL (kernels currently emitted under `kernels/aiv/` and include PTO headers).

**Step 3: Implement platform split**

Update `src/pto/wsp/codegen/pto_runtime_host_build_graph.cpp`:
- `a2a3sim_codegen` → emit `kernels/aiv_sim/kernel_<name>.cpp` (plain C++ loop kernel)
- `a2a3_codegen` → emit `kernels/aiv/kernel_<name>.cpp` (incore kernel signature; Phase 1 body can be loop-based)

**Step 4: Update `kernel_config.py` to select `KERNELS` by platform**

Emit `kernel_config.py` so it chooses sources based on `PTO_RUNTIME_PLATFORM` env var:
- `"a2a3sim"` → `kernels/aiv_sim/*.cpp`
- otherwise → `kernels/aiv/*.cpp`

**Step 5: Run test**

Run: `python -m pytest tests/test_pto_runtime_codegen_emit.py -q`  
Expected: PASS.

**Step 6: Commit**

```bash
git add src/pto/wsp/codegen/pto_runtime_host_build_graph.cpp tests/test_pto_runtime_codegen_emit.py
git commit -m "codegen: emit pto-runtime sim/incore kernels and platform-selecting kernel_config"
```

---

### Task 14: Upgrade orchestration codegen to build a runnable task graph (supported subset only)

**Files:**
- Modify: `src/pto/wsp/codegen/pto_runtime_host_build_graph.cpp:1`
- Test: `tests/test_pto_runtime_codegen_emit.py`
- Modify: `docs/future/pto_runtime_gaps.md`

**Step 1: Write failing test for “non-stub orchestration”**

Extend `tests/test_pto_runtime_codegen_emit.py` to assert the orchestration contains `runtime->add_task`.

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_pto_runtime_codegen_emit.py -q`  
Expected: FAIL (stub orchestration does not call `add_task`).

**Step 3: Implement minimal lowering**

Update orchestration emission so generated `build_pto_wsp_graph(...)`:
- decodes Phase 1 ABI `[host_ptr_i, nbytes_i]` for every `module.tensors[i]`
- allocates `dev_ptrs[i] = runtime->host_api.device_malloc(nbytes_i)`
- `copy_to_device(dev_ptrs[i], host_ptr_i, nbytes_i)` for all tensors
- records all tensors for copy-back: `runtime->record_tensor_pair(host_ptr_i, dev_ptrs[i], nbytes_i)`
- walks the workload body (supported subset) and emits `runtime->add_task(...)` calls:
  - `func_id`: kernel index in `module.kernels`
  - `core_type`: 1 (AIV) for now
  - task args ABI: `[src_ptr, dst_ptr, scalar_u64, size_elems]` (or similar) for supported kernels
- adds conservative deps (sequential chaining) for correctness

Keep the explicit multi-AICPU placeholder marker:
- `TODO_PTO_RUNTIME_MULTI_AICPU_DISPATCH`

**Step 4: Update gaps doc to link the placeholder**

Update `docs/future/pto_runtime_gaps.md` to reference the exact placeholder symbol in generated orchestration.

**Step 5: Run test**

Run: `python -m pytest tests/test_pto_runtime_codegen_emit.py -q`  
Expected: PASS.

**Step 6: Commit**

```bash
git add src/pto/wsp/codegen/pto_runtime_host_build_graph.cpp tests/test_pto_runtime_codegen_emit.py docs/future/pto_runtime_gaps.md
git commit -m "codegen: generate runnable pto-runtime orchestration (supported subset)"
```

---

### Task 15: Add PTO‑WSP wrapper runner for pto-runtime (a2a3sim runnable, a2a3 wired)

**Files:**
- Create: `python/pto_wsp/pto_runtime_runner.py`
- Modify: `python/pto_wsp/program.py:437`

**Step 1: Implement runner**

Create `python/pto_wsp/pto_runtime_runner.py` with:
- `run_host_build_graph(artifact_dir: str, platform: str, tensors: list["Tensor"], device_id: int = 0) -> None`
  - ensures `Tensor.data` is a `numpy.ndarray` (allocate zeros for missing outputs)
  - sets `PTO_RUNTIME_PLATFORM` env var for `kernel_config.py` selection
  - imports pto-runtime modules via `pto_wsp.pto_runtime_bridge`
  - imports generated `kernels/kernel_config.py` from `artifact_dir`
  - builds runtime binaries via `RuntimeBuilder(platform=platform).build("host_build_graph")`
  - binds runtime, sets device, compiles orchestration, compiles/registers kernels, launches runtime, finalizes
  - for `platform=="a2a3"` passes `pto_isa_root=<repo>/3rdparty/pto-isa` to `compile_incore`

**Step 2: Wire Program target mapping**

Update `python/pto_wsp/program.py:437`:
- if `self.target == "pto_runtime_a2a3sim"`, compile codegen with `opts.target="a2a3sim_codegen"`
- if `self.target == "pto_runtime_a2a3"`, compile codegen with `opts.target="a2a3_codegen"`

Update `python/pto_wsp/program.py:670`:
- execute via `pto_wsp.pto_runtime_runner.run_host_build_graph(...)` when target is pto-runtime runnable

**Step 3: No commit yet** (commit after end-to-end tests exist in Task 16/17).

---

### Task 16: Add end-to-end test for `target="pto_runtime_a2a3sim"` (must pass in CI)

**Files:**
- Test: `tests/test_pto_runtime_a2a3sim_e2e.py`

**Step 1: Write failing test**

Create `tests/test_pto_runtime_a2a3sim_e2e.py` (same shape as the existing emit test, but runs):

```python
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from pto_wsp import DType, In, Out, Scalar, Tensor, Tile, kernel, pto, workload


def test_pto_runtime_a2a3sim_scale_e2e():
    x_data = np.arange(16, dtype=np.float32).reshape(4, 4)
    y_data = np.zeros((4, 4), dtype=np.float32)
    x = Tensor(data=x_data, shape=x_data.shape, dtype=DType.F32)
    y = Tensor(data=y_data, shape=y_data.shape, dtype=DType.F32)

    @kernel
    def scale(x_tile: In[Tile[4, 4, DType.F32]], y_tile: Out[Tile[4, 4, DType.F32]], s: Scalar[DType.F32]):
        out = pto.mul(x_tile, s)
        pto.store(y_tile, out)

    @workload
    def wl():
        scale(x_tile=x, y_tile=y, s=2.0)

    program = wl().named("pto_runtime_scale_e2e").compile(target="pto_runtime_a2a3sim")
    program.execute()
    program.synchronize()
    assert np.allclose(y.data, x.data * 2.0)
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_pto_runtime_a2a3sim_e2e.py -q`  
Expected: FAIL until runner + codegen are runnable.

**Step 3: Iterate implementation until it passes**

**Step 4: Commit**

```bash
git add python/pto_wsp/pto_runtime_runner.py python/pto_wsp/program.py tests/test_pto_runtime_a2a3sim_e2e.py
git commit -m "pto-runtime: runnable a2a3sim target (emit + build + run)"
```

---

### Task 17: Add toolchain-gated end-to-end test for `target="pto_runtime_a2a3"`

**Files:**
- Test: `tests/test_pto_runtime_a2a3_e2e.py`

**Step 1: Create gated test**

Create `tests/test_pto_runtime_a2a3_e2e.py`:

```python
import os
import pytest

ASCEND_HOME_PATH = os.environ.get("ASCEND_HOME_PATH")


@pytest.mark.skipif(not ASCEND_HOME_PATH, reason="ASCEND_HOME_PATH not set (Ascend toolkit not available)")
def test_pto_runtime_a2a3_smoke():
    # Same workload as a2a3sim test, but target="pto_runtime_a2a3"
    # Assert output correctness.
    ...
```

**Step 2: Run test**

Run: `python -m pytest tests/test_pto_runtime_a2a3_e2e.py -q`  
Expected: SKIP in environments without Ascend toolkit.

**Step 3: Commit**

```bash
git add tests/test_pto_runtime_a2a3_e2e.py
git commit -m "pto-runtime: add gated a2a3 end-to-end smoke test"
```

---

### Task 18: Update v10 tracker + implementation snapshot as runnable milestones land

**Files:**
- Modify: `docs/future/v10_tracker.md`
- Modify: `docs/future/v10_implementation.md`

**Step 1: After Task 16**

Update:
- `docs/future/v10_tracker.md`: check off Phase 1 runnable a2a3sim
- `docs/future/v10_implementation.md`: add a dated bullet that Phase 1 is runnable for a2a3sim and list the main code entrypoints

**Step 2: After Task 17 (when a2a3 is wired)**

Update:
- `docs/future/v10_tracker.md`: add a note for a2a3 wiring + toolchain gating
- `docs/future/v10_implementation.md`: reflect any toolchain/dev-environment requirements

**Step 3: Commit**

```bash
git add docs/future/v10_tracker.md docs/future/v10_implementation.md
git commit -m "docs(v10): update progress snapshot for pto-runtime runnable integration"
```
