# Tiled Softmax (Validated)

## Overview

This example computes:

`y = softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))`

and validates the result against a NumPy reference.

## v9 Features Demonstrated

- `@kernel` + PTO‑ISA primitives:
  - `update_rowmax`: running row max over vocab tiles
  - `exp_and_accumulate`: `exp(x-max)` and running row sum
  - `rescale`: final normalization
- `@workload` + `P.seq()` to express sequential passes over vocab tiles
- Codegen-first CPU simulation (C++ IR → codegen → build `.so` → execute)
- Numerical validation vs NumPy + non-zero cycle reporting (tolerance-based)

## Prerequisites

- Python >= 3.10
- pto-wsp package (from project root's `python/` directory)

## How to Run

```bash
# From project root
PYTHONPATH=python python examples/softmax/softmax_example.py

# Or using Makefile
cd examples/softmax
make run
```

## Expected Behavior

The example prints:

- `softmax: PASS`
- `softmax: total_cycles=<non-zero>`

## Checking Rules

### Pass Criteria

- [x] Exit code is 0
- [x] No Python exceptions raised
- [x] "Execution complete!" message printed
- [x] "Example completed successfully!" message printed
- [x] Online tiles = 7936
- [x] Rescale tiles = 7936
- [x] Ready policy shows "fifo"
- [x] No `npu()` deprecation warnings

### Behavior Checking

Use `/codex` to verify example behavior:
```bash
codex exec "PYTHONPATH=python python examples/softmax/softmax_example.py"
```

### Fail Indicators

- ImportError or ModuleNotFoundError
- AttributeError (API mismatch)
- RuntimeError during execution
- Validation mismatch vs NumPy
- Missing JIT kernel definitions
- DeprecationWarning for npu() usage

## Troubleshooting

**ModuleNotFoundError: No module named 'pto_wsp'**
- Run from project root directory
- Requires pip install -e . from project root

**Numerical issues in online softmax**
- The online algorithm requires careful ordering of operations
- Sequential vocab tile iteration ensures correct running max/sum accumulation
