# Online Softmax Example

## Overview

This example demonstrates online softmax computation using PTO-RT v9:
- y = softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
- Uses online algorithm to avoid materializing full intermediate
- Two-phase computation: online max/sum accumulation + final rescaling

## v9 Features Demonstrated

- `@jit_kernel` decorator with `tl.*` primitives for online softmax
- Two JIT kernels: `online_softmax_tile_jit` and `softmax_rescale_jit`
- `@workload` decorator with `P` namespace
- `P.seq()` for sequential vocabulary tile iteration (phase 1)
- `P()` for parallel rescaling (phase 2)
- Task graph scheduling with FIFO ready policy
- Tensor map exact dependency inference

## Prerequisites

- Python >= 3.10
- pto-wsp package (from project root's `python/` directory)

## How to Run

```bash
# From project root
python examples/softmax/softmax_example.py

# Or using Makefile
cd examples/softmax
make run
```

## Expected Behavior

### Successful Execution

- Program completes without errors
- Outputs configuration (batch=4, seq=2048, vocab=32000)
- Shows two JIT kernel definitions with tl.* operations
- Reports 7936 online tiles and 7936 rescale tiles
- Task graph schedule uses FIFO ready policy
- Execution completes with "Example completed successfully!"

### Expected Output Sample

```
============================================================
PTO-RT v9 Online Softmax Example
============================================================

Configuration:
  Batch size: 4
  Sequence length: 2048
  Vocabulary size: 32000
  Tile size: 32 x 1024
  Online tiles: 7936
  Rescale tiles: 7936

JIT Kernels (@jit_kernel + tl.*):
  online_softmax_tile_jit:
    Input x: Tile[32, 1024, F16]
    Output y: Tile[32, 1024, F16]
    Running max: Tile[32, 1, F32]
    Running sum: Tile[32, 1, F32]
    Operations: tl.rowmax, tl.max, tl.sub, tl.exp, tl.rowsum, tl.add
  softmax_rescale_jit:
    Operations: tl.rsqrt, tl.mul, tl.store

Building workload...
  Workload kind: combine

Applying task graph schedule...
  Deps: infer_tensor_map_exact
  Window: 4096 tasks
  Ready: fifo

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
- [x] Online tiles = 7936
- [x] Rescale tiles = 7936
- [x] Ready policy shows "fifo"
- [x] No `npu()` deprecation warnings

### Behavior Checking

Use `/codex` to verify example behavior:
```bash
codex exec "Run examples/softmax/softmax_example.py and verify:
1. Output matches expected behavior in README
2. Two JIT kernels defined (online_softmax_tile_jit, softmax_rescale_jit)
3. Task graph schedule uses FIFO ready policy
4. No deprecated API warnings (npu() should not appear)
5. CPU simulation executes without errors"
```

### Fail Indicators

- ImportError or ModuleNotFoundError
- AttributeError (API mismatch)
- RuntimeError during execution
- Tile counts don't match expected values
- Missing JIT kernel definitions
- DeprecationWarning for npu() usage

## Troubleshooting

**ModuleNotFoundError: No module named 'pto_wsp'**
- Run from project root directory
- Or add `sys.path.insert(0, 'python')` at script start

**Numerical issues in online softmax**
- The online algorithm requires careful ordering of operations
- Sequential vocab tile iteration ensures correct running max/sum accumulation
