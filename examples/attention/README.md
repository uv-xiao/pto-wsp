# Tiled Attention (Validated)

## Overview

This example computes a **per-tile** scaled dot-product attention:

`O = softmax(Q K^T / sqrt(d)) V`

and validates against a NumPy reference for each sequence tile (no cross-tile attention).

## v9 Features Demonstrated

- `@kernel` + PTO‑ISA primitives (`pto.matmul`, `pto.rowmax`, `pto.exp`, `pto.div`, `pto.store`)
- `@workload` + `P(batch, heads)` parallel grid + tiled sequence loop
- Codegen-first CPU simulation (C++ IR → codegen → build `.so` → execute)
- Numerical validation vs NumPy + non-zero cycle reporting (tolerance-based)

## Prerequisites

- Python >= 3.10
- pto-wsp package (from project root's `python/` directory)

## How to Run

```bash
# From project root
PYTHONPATH=python python examples/attention/attention_example.py

# Or using Makefile
cd examples/attention
make run
```

## Expected Behavior

The example prints:

- `attention: PASS`
- `attention: total_cycles=<non-zero>`

## Checking Rules

### Pass Criteria

- [x] Exit code is 0
- [x] No Python exceptions raised
- [x] "Execution complete!" message printed
- [x] "Example completed successfully!" message printed
- [x] Total tasks = 32 (4 × 8)
- [x] No `npu()` deprecation warnings

### Behavior Checking

Use `/codex` to verify example behavior:
```bash
codex exec "PYTHONPATH=python python examples/attention/attention_example.py"
```

### Fail Indicators

- ImportError or ModuleNotFoundError
- AttributeError (API mismatch)
- RuntimeError during execution
- Validation mismatch vs NumPy

## Troubleshooting

**ModuleNotFoundError: No module named 'pto_wsp'**
- Run from project root directory
- Requires pip install -e . from project root

**AttributeError: module has no attribute**
- Check pto_wsp version matches v9 API
- Verify imports match current module structure
