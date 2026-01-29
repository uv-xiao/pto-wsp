# PTO-RT Release Readiness Review (v9 design / 0.1.0 impl)

> **Date:** 2026-01-29  
> **Scope:** code + docs in this repo (CPU-sim runnable; NPU emit-only in this environment)

This document lists **release-prep problems** found via:

- Repo-wide scans (`TODO/FIXME`, “unsupported”, “deprecated”, etc.)
- Cross-checking docs (`docs/spec.md`, `docs/features.md`, `docs/analysis.md`, `docs/tutorial.md`, `README.md`) against implementation
- Running validation:
  - `ctest --test-dir build` (PASS)
  - `python -m pytest -q` (PASS: 203 passed, 43 skipped)
  - `for f in examples/*/*_example.py; do PYTHONPATH=python python "$f"; done` (PASS: 11/11)

The goal is to converge on a release-quality baseline where:

1) docs don’t promise behavior that is not implemented,  
2) partial/experimental features are explicitly labeled,  
3) build/licensing is coherent, and  
4) the “golden path” is stable and quiet (no noisy warnings by default).

Revalidation (2026-01-29):
- Re-ran all validations above; all still PASS.

---

## A) Release blockers (P0)

### A1) License mismatch: repo is MIT, but C++ sources declare Apache-2.0

- Repo license: `LICENSE` is **MIT**, and `pyproject.toml` declares `license = {text = "MIT"}`.
- Previously, many C++ headers/sources declared `// SPDX-License-Identifier: Apache-2.0`, e.g.:
  - `src/pto/rt/codegen/cmake_compiler.cpp:1`
  - `src/python/pto_codegen.cpp:2`
  - many files under `include/pto/rt/**` and `src/pto/rt/**`

Why this blocks release:
- Users cannot know which license actually applies.
- SPDX tags are commonly treated as authoritative by tooling.

Fix direction:
- Choose one license as canonical (current intent appears to be **MIT**).
- Update/replace SPDX tags to match the canonical license, or remove per-file SPDX tags entirely.

Status (2026-01-28):
- Resolved by switching all repo-local `SPDX-License-Identifier` tags to `MIT` to match `LICENSE` and `pyproject.toml`.

---

### A2) README Quick Start uses nonexistent API (`pto.softmax`) and misleading schedule calls

In `README.md`, the Quick Start example uses:
- `pto.softmax(scores)` (no `pto.softmax` primitive exists in `python/pto_wsp/kernel.py`)
- schedule calls that imply full runtime enforcement:
  - `.streams(2)` and `.timing(TimingPolicy.immediate)` (v9 artifacts explicitly ignore most knobs beyond `dispatch` + `task_window(stall)`)

Why this blocks release:
- First-run experience will fail immediately or mislead.

Fix direction:
- Replace Quick Start with a minimal, known-good snippet (e.g., add/mul/rmsnorm-like) that uses only supported primitives.
- Explicitly show `compile(target="cpu_sim")` and `program.execute(); program.synchronize()`.
- If `.streams/.timing` remain API-only, either remove them from Quick Start or label as “currently ignored in v9 artifacts”.

Status (2026-01-28):
- Resolved by rewriting `README.md` Quick Start to a runnable CPU-sim snippet using supported `pto.*` ops and the v9 enforced schedule subset.

---

### A3) `docs/spec.md` execution model is out of date (contradicts code + other docs)

`docs/spec.md` still states:
- Python threadpool fallback exists (e.g. `docs/spec.md:790`)
- on-device generation is a **bytecode interpreter** path (e.g. `docs/spec.md:849`)

But implementation is codegen-first:
- `python/pto_wsp/program.py:10` says Python fallback execution has been removed.
- NPU emission in v9 uses emitted C++ expand code (AICPU expander), not a bytecode interpreter in the current implementation.

Why this blocks release:
- `docs/spec.md` is marketed as “the current specification”; it must match reality or explicitly mark future work.

Fix direction:
- Rewrite `docs/spec.md` “Execution Path” to match `docs/implementation.md` (codegen-first artifact).
- Move “bytecode interpreter” material to `docs/design/on-device-task-gen.md` as future design (or update the design doc to match the emitted-expander approach).

Status (2026-01-28):
- Resolved by rewriting `docs/spec.md` sections 4.8 and 5.x to match the codegen-first implementation and by marking bytecode as design-only.

---

### A4) Internal docstring contradiction in `python/pto_wsp/program.py`

`python/pto_wsp/program.py` contains both:
- “Python fallback execution has been removed.” (module docstring)
- “Fallback (Development): If C++ bindings unavailable, falls back to Python threadpool executor” (class docstring)

Why this blocks release:
- even if docs are fixed, the in-code docs remain misleading.

Fix direction:
- Remove the fallback claims from the `Program` class docstring (or re-introduce a real fallback intentionally).

Status (2026-01-28):
- Resolved by updating the `Program` docstring to describe only the codegen-first pipeline (no Python fallback executor).

---

## B) High priority (P1)

### B1) `docs/spec.md` schedule examples contradict the “layout is not schedule” rule

`docs/spec.md` says layout is a type-level refinement, but still shows:
- `schedule.layout(...)` usage in examples (e.g. `docs/spec.md:498`)

In code:
- `Workload.layout()` exists but is explicitly marked deprecated: `python/pto_wsp/workload.py:515`

Fix direction:
- Remove `schedule.layout(...)` examples from `docs/spec.md`.
- Keep layout examples as `Tensor.with_layout(...)` / `relayout(...)` instead.

Status (2026-01-28):
- Resolved by removing `schedule.layout(...)` from `docs/spec.md` schedule examples and adding an explicit note to use `TensorLayout` + `relayout`.

---

### B2) “Unsupported schedule directives” behavior should be a deliberate product choice

Current CPU-sim artifact behavior (codegen) is:
- ignore unsupported schedule directives and print a one-time stderr diagnostic
  - `src/python/pto_codegen.cpp` emits: `"[pto-wsp] v9: unsupported schedule directives ignored:"`

But `docs/spec.md` currently claims compile-time errors for unsupported primitives (see `docs/spec.md` around “Errors at compile time…”).

Fix direction (pick one and align everything):
- Option 1 (strict): fail compilation if unsupported schedule directives are present.
- Option 2 (lenient): keep diagnostic-only, but document clearly in spec + features + README.

Status (2026-01-28):
- Chosen policy: **lenient diagnostic-only** in v9 artifacts (unsupported directives are ignored and reported once at runtime / annotated in NPU emission).

---

### B3) Noisy compile warnings from PTO-ISA headers during codegen builds

When compiling generated artifacts (codegen cache builds), compilation currently produces warnings such as:
- `warning: 'get_temporary_buffer<...>' is deprecated [-Wdeprecated-declarations]`
  - from `3rdparty/pto-isa/include/pto/cpu/TSort32.hpp` (used by TSORT32 path)

Fix direction:
- Prefer suppressing these warnings in the codegen build (e.g., add `-Wno-deprecated-declarations` for the generated-code compilation target) so example runs are quiet by default.
- If we cannot change flags, document the warning as expected when TSORT32 is used.

Status (2026-01-28):
- Resolved by suppressing common PTO-ISA warning classes in the generated-code CMake target and bumping the codegen cache version.

---

### B4) `docs/design/on-device-task-gen.md` likely describes a different implementation than what exists

The design doc describes a bytecode interpreter-based AICPU runtime. Current v9 implementation emits C++ expander code for NPU artifact generation.

Fix direction:
- Decide which approach is canonical for v9:
  - keep the emitted-expander approach (update the design doc), or
  - implement bytecode + interpreter (align code to doc).

Status (2026-01-28):
- Resolved by updating `docs/design/on-device-task-gen.md` to clearly label the bytecode interpreter as design exploration and to describe the current emitted-expander approach.

---

### B5) “Partial” features should be labeled consistently across docs

Examples:
- `docs/features.md` marks schedule/CSP as partial and explains what is enforced.
- `README.md` and `docs/spec.md` still read like full enforcement.

Fix direction:
- Adopt a single “status vocabulary” across docs:
  - **Enforced** (artifact semantics implemented)
  - **API-only** (recorded/serialized but ignored)
  - **Emit-only** (NPU artifact output, not runnable here)
  - **Deprecated**

Status (2026-01-28):
- Resolved by adding a shared status vocabulary to `README.md`, `docs/spec.md`, and `docs/features.md`.

---

### B6) `docs/spec.md` Layout section used non-existent names (`Layout`, `Shard`, `Replicate`)

`docs/spec.md` had an internal inconsistency:
- Module import list correctly documents `TensorLayout`, `TensorShard`, `TensorReplicate`.
- But the Layout section used `from pto_wsp import Layout, Shard, Replicate, MemLayout` and pseudo-constructors that do not match
  the real API (`TensorLayout.default(...)`, `TensorLayout.sharded(...)`, or `TensorLayout((...,), mem=...)`).

Why this matters for release:
- Readers copy/pasting from `docs/spec.md` will hit runtime import errors.

Fix direction:
- Update the Layout section to show the real `TensorLayout` API surface and avoid pseudo classes/constructors.

Status (2026-01-29):
- Resolved by rewriting `docs/spec.md` Layout examples to use `TensorLayout`, `TensorShard`, `TensorReplicate`, and `MemLayout`.

---

### B7) Legacy Claude workspace artifacts should not ship as canonical guidance

Repo had legacy Claude Code workspace artifacts:
- `.claude/skills/**`
- `CLAUDE.md` containing stale examples / references (including removed docs paths).

Why this matters for release:
- Confusing duplicate “agent guidance” sources increases doc drift risk.
- Stale agent docs can reintroduce removed APIs in examples.

Fix direction:
- Treat `AGENTS.md` + `.codex/` as canonical.
- Remove `.claude/**`.
- Keep `CLAUDE.md` minimal (compat-only pointer) or remove it entirely.

Status (2026-01-29):
- Resolved by deleting `.claude/**`, adding `.claude/` to `.gitignore`, and truncating `CLAUDE.md` to a minimal pointer.

---

## C) Robustness / maintainability risks (P2)

### C1) Codegen compiler shelling-out uses `system()` with string commands

`src/pto/rt/codegen/cmake_compiler.cpp` uses:
- `std::system(cmd.c_str())` and string-concatenated commands

Risks:
- quoting issues for paths with spaces
- harder to capture logs for debugging

Fix direction:
- Use a structured process runner (argv vector) or at least robust quoting and log capture to a known file in the build dir.

Status (2026-01-28):
- Improved robustness by adding shell quoting for key arguments and capturing CMake configure/build output to `configure.log` / `build.log` under the generated build directory.

---

### C2) Schedule/time semantics: wall-clock tracing vs cycle time can confuse users

`Program` exposes wall-clock trace tooling, while v9 semantic time is in PTO-ISA cycles.

Fix direction:
- Document clearly:
  - `Program.stats.total_cycles` is the canonical “time” for CPU-sim semantics
  - tracing is for debugging/perf tooling only

Status (2026-01-28):
- Resolved by adding explicit notes to `docs/tutorial.md` and `docs/implementation.md` distinguishing cycle time from wall-clock trace time.

---

### C3) Deprecation surface still exists in code

Example: string-based task-like paths are referenced as “deprecated” in code/doc research. For release, decide:
- keep deprecated paths (with clear docs and tests), or
- remove them to reduce surface area.

Status (2026-01-28):
- Completed a minimal audit: kept the deprecation surface but documented it explicitly in `docs/spec.md` (Section 5.4) and verified tests cover key deprecations (e.g., `Workload.layout()` warnings).

---

## Fix plan (checklist)

This plan is ordered: **P0 first**, then P1, then P2.

### P0 — must complete before release tag

- [x] Decide canonical repo license (expected: MIT).
- [x] Replace/remove all `Apache-2.0` SPDX identifiers to match the canonical license.
- [x] Fix `README.md` Quick Start to use only implemented primitives (no `pto.softmax`) and to reflect v9 enforcement limits.
- [x] Rewrite `docs/spec.md` execution model (remove Python fallback + bytecode claims unless implemented; align with `docs/implementation.md`).
- [x] Fix `python/pto_wsp/program.py` docstrings to match actual behavior (no fallback claims unless implemented).

### P1 — release-quality polish

- [x] Remove `schedule.layout(...)` examples from `docs/spec.md` (layout is type-level; `Workload.layout()` is deprecated).
- [x] Decide strict vs lenient handling of unsupported schedule directives; align code + docs accordingly.
- [x] Silence PTO-ISA deprecation warnings in codegen builds (or document as expected).
- [x] Align `docs/design/on-device-task-gen.md` with current NPU emission approach (or implement the bytecode interpreter path).
- [x] Standardize doc “feature status” vocabulary across `README.md`, `docs/spec.md`, `docs/features.md`.
- [x] Fix `docs/spec.md` Layout section to match real `TensorLayout` API names/constructors.
- [x] Remove legacy `.claude/**` and minimize `CLAUDE.md` to avoid stale guidance.

### P2 — long-term robustness

- [x] Improve codegen build command execution (quoting, log capture, better error messages).
- [x] Clarify “cycle time vs wall-clock trace time” in `docs/tutorial.md` and `docs/implementation.md`.
- [x] Audit and reduce deprecated/legacy API paths (either document+test or remove).
