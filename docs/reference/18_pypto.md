# Reference Notes: PyPTO (for PTO‑WSP v10)

This document summarizes what PTO‑WSP v10 should learn from the `pypto` project’s **Python-driven compilation pipeline** and
its **artifact packaging** patterns.

Local clone (gitignored): `references/pypto/`.

## 1) What PyPTO supports (high-level)

From repository structure and public docs:

- **Tile-based programming model** with multi-level IRs (Tensor → Tile → Block → Execution) and explicit pass pipelines.
- **Python-driven compilation** over a C++ core (bindings via a C++ extension; `pypto_core`).
- **Backend selection** is explicit and affects both pass selection and codegen:
  - `BackendType.PTO`: emit PTO virtual instruction code (`output.pto`)
  - `BackendType.CCE`: emit a C++ artifact tree (kernels + orchestration + config)
- **Artifact-first mindset**: compilation produces a unified output directory containing IR dumps and codegen outputs.

Pointers:
- `references/pypto/README.md`
- `references/pypto/python/pypto/pypto_core/backend.pyi`

## 2) Compilation driver shape (Python)

PyPTO’s high-level compile API is “pipeline complete”:

- set a global backend type (idempotent),
- choose a pass strategy,
- optionally dump IR snapshots per pass,
- emit artifacts into an output directory.

Pointer:
- `references/pypto/python/pypto/ir/compile.py`

Notable pattern for v10:
- the output directory is treated as the **unified artifact root** for all compiler outputs (IR dumps + codegen outputs).

### 2.1 Pass pipeline patterns

PyPTO exposes a small pass-manager abstraction with named strategies (example: `Default`, `PTOAS`) that map to concrete pass
pipelines.

Pointer:
- `references/pypto/python/pypto/ir/pass_manager.py`

Example pass names present in the default strategy:
- SSA conversion, call-expression flattening, verifier, memref init, basic memory reuse, sync insertion, alloc insertion.

Takeaway for PTO‑WSP v10:
- keep a first-class pass manager boundary,
- make “dump before/after pass” a standard debugging affordance,
- allow backend-aware pass strategies without entangling them with the emitter’s file layout decisions.

## 3) Artifact packaging pattern (CCE codegen)

PyPTO’s CCE codegen emits a structure that is directly relevant to PTO‑WSP v10 backend packaging goals:

- split “kernel functions” vs “orchestration function”
- emit kernels into per-executor subdirectories (vector vs matrix style)
- emit orchestration code separately
- generate a `kernel_config.py` describing:
  - orchestration source + entry symbol
  - a list of kernels with stable `func_id` and `core_type`
  - runtime configuration knobs (threads, block dims, etc.)

Pointer:
- `references/pypto/src/codegen/cce/cce_codegen.cpp`

Concrete shape (conceptual):

```
<artifact_root>/
  kernels/
    aiv/
      <kernel>.cpp
    aic/
      <kernel>.cpp
  orchestration/
    <orch>.cpp
  kernel_config.py
```

Notable for v10:
- `kernel_config.py` plays the role of a **manifest/config** tying together orchestration and kernels, including runtime
  config and `func_id` mapping. This aligns closely with PTO‑WSP’s `pto_runtime_*` packaging needs (Phase 1 host_build_graph
  trees and Phase 2 versioned packages).

## 4) PTO backend output (PTO “assembly” artifact)

When targeting `BackendType.PTO`, PyPTO emits an `output.pto` file rather than a C++ tree. The key design lesson is not the
format itself, but the “one compilation → one artifact root” discipline:

- optimization passes run first,
- codegen emits a single backend-specific artifact (here: `.pto`),
- all intermediate pass dumps live alongside the final output in the same output directory.

Pointer:
- `references/pypto/python/pypto/ir/compile.py`

## 4) Takeaways for PTO‑WSP v10

- Keep compilation **Python-driven** but keep semantics in artifacts/runtime.
- Treat the emitted directory/tree as a **first-class visible artifact** (reviewable, reproducible).
- Make “packaging” a backend responsibility, but require a stable manifest/config contract:
  - kernel registry (`kernel_id/func_id` + executor type)
  - orchestration entrypoint identity
  - runtime configuration and capability declarations
- Preserve a clean pass pipeline boundary: optimization passes should be backend-aware but not entangled with codegen output
  formatting decisions.

Concrete “follow this” guideline for PTO‑WSP v10:
- For `pto_runtime_*` targets, treat `kernels/kernel_config.py` (or a future manifest) as the stable bridge between PTO‑WSP
  codegen and pto-runtime build/run tooling, in the same spirit as PyPTO’s CCE packaging.
