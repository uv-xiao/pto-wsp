# RMSNorm (Validated)

## Overview

This example validates RMSNorm end-to-end on CPU-sim artifacts.

## v9 Features Demonstrated

- `@kernel` + PTO‑ISA primitives (`pto.mul`, `pto.rowmean`, `pto.rsqrt`, `pto.store`)
- `@workload` + tiled sequence loop
- Codegen-first CPU simulation (C++ IR → codegen → build `.so` → execute)
- Numerical validation vs NumPy + non-zero cycle reporting (tolerance-based)

## Prerequisites

- Python >= 3.10
- pto-wsp package (from project root's `python/` directory)

## How to Run

```bash
# From project root
PYTHONPATH=python python examples/e2e_rmsnorm/e2e_rmsnorm_example.py

# Or using Makefile
cd examples/e2e_rmsnorm
make run
```

## Expected Behavior

The example prints:

- `e2e_rmsnorm: PASS`
- `e2e_rmsnorm: total_cycles=<non-zero>`

## Checking Rules

### Pass Criteria

- [x] Exit code is 0
- [x] No Python exceptions raised
- [x] All 7 stages complete successfully
- [x] Total tasks = 64 (4 × 16)
- [x] "Execution complete!" message printed
- [x] "Example completed successfully!" message printed
- [x] No `npu()` deprecation warnings

### Behavior Checking

Use `/codex` to verify example behavior:
```bash
codex exec "Run examples/e2e_rmsnorm/e2e_rmsnorm_example.py and verify:
1. All 7 stages of the pipeline are shown
2. JIT kernel uses tl.* primitives (tl.mul, tl.rowmean, tl.rsqrt, tl.store)
3. Task enumeration shows 64 tasks
4. No deprecated API warnings (npu() should not appear)
5. Execution statistics are reported"
```

### Fail Indicators

- ImportError or ModuleNotFoundError
- AttributeError (API mismatch)
- RuntimeError during execution
- Missing stages in output
- Task count != 64
- DeprecationWarning for npu() usage

## Troubleshooting

**ModuleNotFoundError: No module named 'pto_wsp'**
- Run from project root directory
- Requires pip install -e . from project root

**Stage X not showing**
- Verify Python version >= 3.10
- Check pto_wsp module is up-to-date
