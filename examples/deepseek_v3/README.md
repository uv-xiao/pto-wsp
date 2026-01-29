# Toy DeepSeek-V3 MoE Block (Validated)

## Overview

This is a **small validated** Mixture-of-Experts (MoE) building block suitable for v9 CPU-sim:

- Two expert linears: `e0 = X @ W0`, `e1 = X @ W1`
- Per-token gate: `Y = g*e0 + (1-g)*e1`

It is **not** a full DeepSeek-V3.2 system (MLA, sparse routing, gather/scatter).

## v9 Features Demonstrated

- `@kernel` + PTO‑ISA primitives (`pto.matmul`, `pto.add`, `pto.mul`, `pto.store`)
- `@workload` + tiled sequence loop (`P.seq_tiles`)
- Codegen-first CPU simulation (C++ IR → codegen → build `.so` → execute)
- Numerical validation vs NumPy + non-zero cycle reporting (tolerance-based)

## Prerequisites

- Python >= 3.10
- pto-wsp package (from project root's `python/` directory)

## How to Run

```bash
# From project root
PYTHONPATH=python python examples/deepseek_v3/deepseek_v3_example.py

# Or using Makefile
cd examples/deepseek_v3
make run
```

## Expected Behavior

The example prints:

- `deepseek_v3: PASS`
- `deepseek_v3: total_cycles=<non-zero>`

## Checking Rules

### Pass Criteria

- [x] Exit code is 0
- [x] No Python exceptions raised
- [x] "Execution complete!" message printed
- [x] "Example completed successfully!" message printed
- [x] Expert FFN (sparse) = 2048 tasks
- [x] Window size = 16384
- [x] Trace policy shows "cycles"
- [x] No `npu()` deprecation warnings

### Expected Deprecation Warnings

The following deprecation warnings are **expected** (R5 extensions demo):
- `batch_deps() is deprecated (L10)`
- `pipeline_depth() is deprecated (L10)`

### Behavior Checking

Use `/codex` to verify example behavior:
```bash
codex exec "Run examples/deepseek_v3/deepseek_v3_example.py and verify:
1. Output matches expected behavior in README
2. All 3 JIT kernels defined correctly
3. MoE configuration shows 256 experts, top-8 selection
4. Only batch_deps/pipeline_depth warnings appear (not npu() warnings)
5. CPU simulation executes without errors"
```

### Fail Indicators

- ImportError or ModuleNotFoundError
- AttributeError (API mismatch)
- RuntimeError during execution
- Missing MoE configuration output
- DeprecationWarning for npu() usage

## Troubleshooting

**ModuleNotFoundError: No module named 'pto_wsp'**
- Run from project root directory
- Requires pip install -e . from project root

**Deprecation warnings**
- `batch_deps()` and `pipeline_depth()` warnings are expected
- These R5 extensions are retained for demonstration purposes
- For new code, use `.task_graph(deps=..., window=...)` instead
