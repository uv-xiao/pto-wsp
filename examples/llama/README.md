# Toy LLaMA-Style Block (Validated)

## Overview

This is a small validated “transformer block”-style example suitable for v9 CPU-sim:

- Per-tile attention: `Attn = softmax(X X^T / sqrt(d)) X`
- Residual + RMSNorm
- 2-layer MLP with ReLU

It is intentionally **not** a full LLaMA-7B (no RoPE, no multi-head, no SwiGLU).

## v9 Features Demonstrated

- `@kernel` + PTO‑ISA primitives (matmul/reductions/exp/etc.)
- `@workload` + tiled sequence loop
- Codegen-first CPU simulation (C++ IR → codegen → build `.so` → execute)
- Numerical validation vs NumPy + non-zero cycle reporting (tolerance-based)

## Prerequisites

- Python >= 3.10
- pto-wsp package (from project root's `python/` directory)

## How to Run

```bash
# From project root
PYTHONPATH=python python examples/llama/llama_example.py

# Or using Makefile
cd examples/llama
make run
```

## Expected Behavior

The example prints:

- `llama: PASS`
- `llama: total_cycles=<non-zero>`

## Checking Rules

### Pass Criteria

- [x] Exit code is 0
- [x] No Python exceptions raised
- [x] "Execution complete!" message printed
- [x] "Example completed successfully!" message printed
- [x] Total tasks = 928
- [x] Deps mode shows "hybrid"
- [x] Pools shows "by_exec_unit"
- [x] Ready policy shows "work_steal"
- [x] No `npu()` deprecation warnings

### Behavior Checking

Use `/codex` to verify example behavior:
```bash
codex exec "Run examples/llama/llama_example.py and verify:
1. Output matches expected behavior in README
2. All 4 JIT kernels defined correctly
3. Task graph schedule shows hybrid deps, by_exec_unit pools, work_steal ready
4. No deprecated API warnings (npu() should not appear)
5. CPU simulation executes without errors"
```

### Fail Indicators

- ImportError or ModuleNotFoundError
- AttributeError (API mismatch)
- RuntimeError during execution
- Total tasks != 928
- Missing JIT kernel definitions
- DeprecationWarning for npu() usage

## Troubleshooting

**ModuleNotFoundError: No module named 'pto_wsp'**
- Run from project root directory
- Requires pip install -e . from project root

**Task count mismatch**
- Verify tile size divides evenly into dimensions
- Check batch size and sequence length configuration
