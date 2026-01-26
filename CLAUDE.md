# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PTO Workload-Schedule Programming (PTO-WSP) framework (pto-wsp) enables dynamic LLM workloads on Ascend NPU and other accelerators with typed workload expressions and two parallelism modes:
- **Data-parallel**: `parallel_for`, `for_each`, `select`, `cond`
- **Pipeline-parallel**: CSP with `Channel`, `Process`, `consume`

**Key characteristics:**
- Python frontend with declarative workload definition
- C++ IR layer for multi-backend targeting
- Combinator-style scheduling: `workload.dispatch(...).streams(...).compile()`
- pto-isa is a 3rd party dependency

## Build & Test Commands

```bash
# Configure with pto-isa path
cmake -B build -DPTO_ISA_PATH=../pto-isa

# Build
cmake --build build

# Run tests
ctest --test-dir build

# Python tests
python -m pytest tests/
```

## Architecture

```
pto-wsp/
├── docs/                    # Design documents
│   ├── analysis.md          # Design rationale (WHY)
│   ├── spec.md              # API specification (WHAT)
│   ├── features.md          # Feature catalog with links
│   ├── task_plan.md         # Implementation plan
│   ├── reference/           # Reference analysis (external research)
│   │   └── 01_xxx.md - 16_xxx.md  # Numbered research notes
│   ├── research/            # Intermediate working analysis
│   │   └── *_research.md    # Design explorations
│   └── design/              # Detailed design documents
│       ├── backend-arch.md  # Backend architecture
│       ├── ir-design.md     # IR system design
│       ├── npu-design.md    # NPU programming
│       └── type-system.md   # Type system
├── python/pto_wsp/           # Python frontend
├── include/pto/rt/          # C++ headers
│   ├── ir/                  # IR node types
│   ├── graph/               # Task graph infrastructure
│   └── backend/             # Backend implementations
├── src/pto/rt/              # C++ implementation files
│   ├── ir/                  # parser.cpp, type_check.cpp
│   ├── graph/               # runtime.cpp
│   └── backend/             # cpu_sim.cpp, ascend_npu.cpp
├── tests/                   # Unit tests
├── examples/                # Usage examples
└── 3rdparty/                # External dependencies
```

## Key Design Concepts

### Typed Workload Expressions

```python
# Declarative style (not decorators)
attention = parallel_for(batch, lambda b:
    parallel_for(heads, lambda h:
        task("attn", [b, h], [Q[b][h], K[b], V[b], O[b][h]])))

# Type: Workload[DenseDyn × Dense[8], AttnTask, Independent]
```

### Combinator-Style Schedule

```python
# Each method returns new schedule, enabling type-safe chaining
program = (attention
    .dispatch(DispatchPolicy.affinity(lambda t: t.batch))
    .streams(2)
    .stream_by(lambda t: t.head % 2)
    .timing(TimingPolicy.immediate)
    .compile())
```

### CSP Primitives

```python
# Process bodies use declarative primitives
loader = (process("loader")
    .produces(channel)
    .body(for_each(tiles, lambda i:
        send(channel, task("load", [i], [...])))))

# consume replaces while-recv
computer = (process("computer")
    .consumes(input_ch)
    .produces(output_ch)
    .body(consume(input_ch, lambda t:
        send(output_ch, task("compute", [t], [...])))))
```

## Namespace

- Python: `pto.rt`
- C++ IR: `pto::wsp::ir`
- C++ Backend: `pto::wsp::backend`

## Design Evolution

The design evolves through a structured, research-driven process:

1. **Research Phase**: Study relevant systems and papers → `docs/reference/` (numbered notes: `01_flashinfer.md`, `02_gpu_patterns.md`, etc.)
2. **Working Analysis**: Design explorations → `docs/research/` (e.g., `jit_design_research.md`)
3. **Design Synthesis**: Create design documents → `docs/analysis.md`, `docs/spec.md`
4. **Detailed Design**: Technical specifications → `docs/design/` (backend-arch.md, ir-design.md, etc.)
5. **Implementation**: Build with task tracking → `docs/task_plan.md`
6. **Iteration**: Archive old versions → `docs/archive/v1/` through `v8/`

**Key files for tracking evolution:**
- `docs/progress.md` — Session log with decisions and status
- `docs/findings.md` — Research findings and key decisions
- `docs/archive/` — Previous design versions (v1-v8)

**Current version:** v9 (typed workload expressions, two parallelism modes, declarative CSP)

## Documentation

### Document Organization

| Directory | Purpose | When to use |
|-----------|---------|-------------|
| `docs/` | Core documents | analysis.md (WHY), spec.md (WHAT), features.md (catalog) |
| `docs/reference/` | Reference analysis | External research, numbered notes (01_xxx.md - 16_xxx.md) |
| `docs/research/` | Working analysis | Intermediate research, design explorations |
| `docs/design/` | Detailed design | Backend, IR, NPU, Type system specifications |
| `docs/archive/` | Old versions | v1-v8 designs |

### Key Documents

| Document | Description |
|----------|-------------|
| `docs/analysis.md` | v9 design rationale (WHY) |
| `docs/spec.md` | API specification (WHAT) |
| `docs/features.md` | Feature catalog with code links |
| `docs/design/ir-design.md` | C++ IR architecture |
| `docs/design/backend-arch.md` | Backend specifications |
| `docs/task_plan.md` | Implementation plan |
| `docs/progress.md` | Progress log |

## Requirements

- Python >= 3.10
- CMake >= 3.16
- C++23 compiler: GCC 14+ / Clang 16+
- pto-isa (3rd party dependency)

## Related Projects

- **pto-isa**: PTO Tile Library (../pto-isa/)
