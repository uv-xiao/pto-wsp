# LLaMA Transformer Layer Example

## Overview

This example demonstrates a complete LLaMA transformer layer using PTO-RT v9:
- RMSNorm (pre-attention and pre-MLP)
- Multi-head attention with RoPE (Rotary Position Embedding)
- SwiGLU MLP (gate projection, up projection, SiLU, down projection)
- Residual connections

Based on LLaMA-7B architecture parameters.

## v9 Features Demonstrated

- `@jit_kernel` decorator with `tl.*` primitives for multiple kernels:
  - `rmsnorm_tile_jit`: RMSNorm computation
  - `linear_tile_jit`: Linear projection (cube operations)
  - `silu_tile_jit`: SiLU activation
  - `attention_tile_jit`: Scaled dot-product attention
- `@workload` decorator with `P` namespace
- Hybrid dependency inference (`Deps.hybrid()`)
- Pool separation by execution unit (`Pools.by_exec_unit()`)
- Work-stealing ready policy
- Start threshold configuration

## Prerequisites

- Python >= 3.10
- pto-wsp package (from project root's `python/` directory)

## How to Run

```bash
# From project root
python examples/llama/llama_example.py

# Or using Makefile
cd examples/llama
make run
```

## Expected Behavior

### Successful Execution

- Program completes without errors
- Outputs LLaMA-7B style configuration
- Shows 4 JIT kernel definitions with appropriate tl.* operations
- Estimates 928 total tasks across all operations
- Task graph schedule uses hybrid dependencies and work-stealing
- Execution completes with "Example completed successfully!"

### Expected Output Sample

```
============================================================
PTO-RT v9 LLaMA Transformer Layer Example
============================================================

Configuration (LLaMA-7B style):
  Batch size: 1
  Sequence length: 512
  Hidden dimension: 4096
  Number of heads: 32
  Head dimension: 128
  MLP dimension: 11008
  Tile size: 32

Estimated tasks:
  RMSNorm: 32
  Linear: 112
  RoPE: 32
  Attention: 32
  SiLU: 344
  Mul: 344
  Add: 32
  Total: 928

JIT Kernels (@jit_kernel + tl.*):
  rmsnorm_tile_jit: tl.mul, tl.rowmean, tl.add, tl.rsqrt
  linear_tile_jit (cube): tl.matmul
  silu_tile_jit: tl.neg, tl.exp, tl.add, tl.div
  attention_tile_jit: tl.matmul, tl.rsqrt, tl.rowmax, tl.exp, tl.rowsum, tl.div

Building workload...
  Workload kind: combine

Applying task graph schedule...
  Deps: hybrid
  Pools: by_exec_unit (vector/cube separation)
  Ready: work_steal
  Start threshold: 50

Compiling program...
  Program type: Program

Executing with CPU simulation...
Execution complete!

Example completed successfully!
============================================================
```

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
- Or add `sys.path.insert(0, 'python')` at script start

**Task count mismatch**
- Verify tile size divides evenly into dimensions
- Check batch size and sequence length configuration
