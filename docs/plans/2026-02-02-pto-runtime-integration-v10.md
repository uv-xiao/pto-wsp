# pto-runtime Integration (v10) Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Start semantically-honest PTO‑WSP v10 integration with `pto-runtime` by pinning it as a submodule and scaffolding
the build/docs plumbing needed for Phase 1 (host-built orchestration) and Phase 2 (task-buffer) work.

**Architecture:** Keep `pto-runtime` as a pinned submodule under `3rdparty/pto-runtime`, preserve `references/pto-runtime`
as a read-only mirror/reference, and introduce a minimal “bridge” layer in PTO‑WSP that (a) locates `pto-runtime`, (b)
exposes Python import ergonomics, and (c) documents gaps/capability boundaries before any runnable integration claims.

**Tech Stack:** git submodules, CMake (PTO‑WSP), Python packaging/import helpers, pytest (import-level tests).

---

### Task 0: Create an isolated worktree for integration work

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
