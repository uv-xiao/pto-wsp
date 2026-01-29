# CLAUDE.md (legacy)

This repository’s canonical agent guidance is in `AGENTS.md` (Codex CLI).
This file is kept only for minimal compatibility with Claude Code.

## Start here

- Canonical docs index: `docs/README.md`
- Spec: `docs/spec.md`
- Tutorial: `docs/tutorial.md`
- Implementation (as-built): `docs/implementation.md`

## Build & test

```bash
cmake -B build -DPTO_ISA_PATH=3rdparty/pto-isa
cmake --build build
ctest --test-dir build
PYTHONPATH=python python -m pytest -q
```
