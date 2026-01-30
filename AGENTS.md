# AGENTS.md

This file provides guidance to Codex CLI when working with code in this repository.

## Project Overview

PTO Workload-Schedule Programming (PTO-WSP) framework (pto-wsp) enables dynamic LLM workloads on Ascend NPU and other accelerators with typed workload expressions and two parallelism modes:
- **Data-parallel**: `parallel_for`, `for_each`, `select`, `cond`
- **Pipeline-parallel**: CSP with `Channel`, `Process`, `consume`

Key characteristics:
- Python frontend with declarative workload definition
- C++ IR layer for multi-backend targeting
- Combinator-style scheduling: `workload.dispatch(...).streams(...).compile()`
- `pto-isa` is a 3rd party dependency

## Build & Test

```bash
# Configure with pto-isa path
cmake -B build -DPTO_ISA_PATH=../pto-isa

# Build (also copies C++ bindings to python/pto_wsp/)
cmake --build build

# Run C++/CMake tests
ctest --test-dir build

# Python tests
python -m pytest tests/

# Install in development mode
pip install -e .
```

## C++ Bindings Import Strategy

The `pto_ir_cpp` C++ bindings module is available via two import paths:
1. After `pip install -e .`: `from pto_wsp import pto_ir_cpp`
2. During development: the build copies the `.so` to `python/pto_wsp/`

`python/pto_wsp/ir_bridge.py` tries the submodule import first, then falls back to top-level import.

## Repo Layout (high-level)

```
pto-wsp/
  docs/                 # design docs + reference notes
  python/pto_wsp/        # Python frontend
  include/pto/wsp/       # C++ headers
  src/pto/wsp/           # C++ implementation
  tests/                # unit/integration tests
  examples/             # examples
  3rdparty/             # external dependencies
```

## Skills

A skill is a set of local instructions stored in a `SKILL.md` file.

### Available (repo-local) skills
- planning-with-files: Manus-style file-based planning (file: `./.codex/skills/planning-with-files/SKILL.md`)
- e2e-example: List/run/add examples with validation checklist (file: `./.codex/skills/e2e-example/SKILL.md`)
- codex: Notes on using `codex exec` for scripted analysis (file: `./.codex/skills/codex/SKILL.md`)

### How to use skills
- Trigger rules: If the user names a skill (with `$SkillName` or plain text) OR the task clearly matches a skill’s description, use that skill for that turn.
- Progressive disclosure: Open the relevant `SKILL.md` first; only read referenced files as needed.
- Prefer scripts/templates shipped with the skill over retyping.
