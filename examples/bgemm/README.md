# Batched GEMM (bgemm) Example

## Overview

This example computes `C[b] = A[b] @ B[b]` and validates against NumPy.
It demonstrates:

- `@kernel` with PTO‑ISA primitives (`pto.matmul`, `pto.add`, `pto.store`)
- `@workload` with parallel axes and `P.seq()` for sequential K accumulation
- Codegen-first CPU simulation (C++ IR → codegen → build `.so` → execute)

## v9 Features Demonstrated

- Correctness validation vs NumPy
- Cycle reporting (non-zero; tolerance-based policy)

## Prerequisites

- Python >= 3.10
- pto-wsp package (from project root's `python/` directory)

## How to Run

```bash
# From project root
PYTHONPATH=python python examples/bgemm/bgemm_example.py

# Or using Makefile
cd examples/bgemm
make run
```

## Expected Behavior

The example prints:

- `bgemm: PASS`
- `bgemm: total_cycles=<non-zero>`

## Checking Rules

### Pass Criteria

- [x] Exit code is 0
- [x] No Python exceptions raised
- [x] "Execution complete!" message printed
- [x] "Example completed successfully!" message printed
- [x] Total tiles = 4096
- [x] Deps mode shows "infer_tensor_map_exact"
- [x] No `npu()` deprecation warnings

### Behavior Checking

Use `/codex` to verify example behavior:
```bash
codex exec "PYTHONPATH=python python examples/bgemm/bgemm_example.py"
```

### Fail Indicators

- ImportError or ModuleNotFoundError
- AttributeError (API mismatch)
- RuntimeError during execution
- Total tiles != 4096
- Missing task graph configuration output
- DeprecationWarning for npu() usage

## Troubleshooting

**ModuleNotFoundError: No module named 'pto_wsp'**
- Run from project root directory
- Requires pip install -e . from project root

**Task count mismatch**
- Verify tile dimensions divide evenly into matrix dimensions
- Check batch size configuration
