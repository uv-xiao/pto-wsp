---
name: e2e-example
description: List, run, and scaffold PTO-WSP examples under `examples/` with a validation checklist (imports, codegen artifacts, synchronize, output checks, cycles/timing).
---

# e2e-example

Manage and validate PTO-WSP examples with a simple subcommand-style flow:

- `list`: show available examples
- `run <name>`: run and validate an example
- `add <name>`: scaffold a new example

## List examples

1. Find example directories:
   ```bash
   ls -d examples/*/
   ```
2. For each example, report:
   - Example name
   - Main script file
   - Has `README.md` (yes/no)
   - Has `Makefile` (yes/no)
   - Quick validation status (runs without error)

## Run and validate example

1. Locate the example:
   ```bash
   ls examples/<name>/*.py
   ```
2. Execute (must build + run codegen artifacts; Python fallback is not acceptable for v9):
   ```bash
   PYTHONPATH=python python examples/<name>/<script>.py 2>&1
   ```
3. Check output for:
   - Import errors (`ImportError`, `ModuleNotFoundError`)
   - Deprecation warnings (treat `npu()` deprecation as error)
   - Evidence codegen actually executed (not a stub):
     - a built/loaded artifact (e.g. `.so` in `~/.cache/pto_wsp/codegen/`, or a logged build-id), AND
     - the example calls `program.synchronize()` after `program.execute()`
     - the example uses the **codegen-first** path (`.compile(target="cpu_sim")` or emits NPU sources via `.compile(target="ascend_npu")`), not legacy interpreters
   - Correctness signals (when provided by the example):
     - `Status: PASS` (preferred), OR `np.testing.assert_allclose` with no exception
   - Cycle/timing signals (must be present for “golden path” examples):
     - a printed cycle count / timing summary derived from PTO‑ISA kernel reports (CPU sim or NPU timestamps), and it should be non-zero for non-trivial workloads
   - No substitute harness:
     - any shared “CI helper” that prints `Status: PASS` / `Total cycles:` without validating the example’s real workload is a v9 blocker
4. Source-level checks (read the example `.py`):
   - `@jit_kernel` defined but not used by workload (warning)
   - Empty `@kernel` stubs or CPU stubs (`pass`) (warning)
   - Any `program.register_kernel(...)` usage (ERROR; legacy path)
5. Artifact check (optional but useful):
   - Confirm a `.so` was produced/loaded (cache-based builds are OK). Typical cache location:
     - `~/.cache/pto_wsp/codegen/`

## Add new example

Create `examples/<name>/` with:
- `examples/<name>/<name>_example.py`
- `examples/<name>/Makefile`
- `examples/<name>/README.md`

Use the existing examples as style references.

## Severity conventions

- ERROR: script fails to execute, import errors, `npu()` deprecation warning
- WARNING: JIT kernel not connected, empty stubs, no output validation
- INFO: documented/acceptable deprecations (`batch_deps()`, `pipeline_depth()`)
