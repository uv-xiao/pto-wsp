# FlashInfer Decode Attention (Validated)

## Overview

This example is a **small validated** decode-attention workload:

- `out = softmax(q k^T / sqrt(d)) v`
- Codegen-first CPU simulation (C++ IR → codegen → build `.so` → execute)
- Numerical validation vs NumPy + non-zero cycle reporting (tolerance-based)

This example is intentionally self-contained and fully validated via `flashinfer_decode_example.py`.

## v9 Features Demonstrated

- `@kernel` + PTO‑ISA primitives (`pto.matmul`, `pto.rowmax`, `pto.exp`, `pto.div`, `pto.store`)
- `@workload` as a single-task program (decode token)
- Validated output vs NumPy + cycle reporting

## Prerequisites

- Python >= 3.10
- pto-wsp package (from project root's `python/` directory)

## How to Run

```bash
# From project root
PYTHONPATH=python python examples/flashinfer_decode/flashinfer_decode_example.py

# Or using Makefile
cd examples/flashinfer_decode
make run
```

## Expected Behavior

The example prints:

- `flashinfer_decode: PASS`
- `flashinfer_decode: total_cycles=<non-zero>`

## Checking Rules

### Pass Criteria

- [x] Exit code is 0
- [x] No Python exceptions raised
- [x] Planning phase completes with work descriptors
- [x] Tier distribution shows all 4 tiers
- [x] "Execution complete!" message printed
- [x] Comparison table displayed
- [x] No `npu()` deprecation warnings

### Behavior Checking

Use `/codex` to verify example behavior:
```bash
codex exec "PYTHONPATH=python python examples/flashinfer_decode/flashinfer_decode_example.py"
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

**Chunk size calculation issues**
- Binary search finds optimal chunk size
- Default target is 512 work units
- Adjust sequence lengths if needed
