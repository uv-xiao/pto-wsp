# PTO-RT Documentation

This directory contains the main documentation set for PTO Workload-Schedule Programming (PTO-WSP) framework (PTO‑RT).

## Start here

- `docs/tutorial.md` — hands-on walkthrough (CPU-sim first; NPU emit-only here)
- `docs/features.md` — feature catalog + status (what is enforced in v9 vs API-only)
- `docs/spec.md` — API spec (Python surface + semantics notes)
- `docs/implementation.md` — as-built guide (entrypoints, IR bridge, codegen, runtime)
- `docs/analysis.md` — design analysis (why the architecture/semantics look this way)
- `docs/release_review.md` — release readiness review + fix checklist

## Supporting material

- `docs/design/` — detailed design docs (IR, backend architecture, NPU notes, codegen AST)
- `docs/reference/` — external references and extracted notes
- `docs/research/` — older working docs and explorations
- `docs/archive/` — historical specs/analysis

## Examples

All runnable, self-validating examples live under `examples/` (not under `docs/`).
Each example includes:
- a golden reference (`golden.py`)
- a PTO‑RT implementation (`pto_wsp_impl.py`)
- a runner/checker (`*_example.py`)

Run the full suite:

```bash
for f in examples/*/*_example.py; do
  PYTHONPATH=python python "$f"
done
```
