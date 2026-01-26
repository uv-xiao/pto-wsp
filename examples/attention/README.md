# Multi-Head Attention Example

## Overview

This example demonstrates multi-head attention computation using PTO-RT v9:
- Scaled dot-product attention: O = softmax(Q @ K^T / sqrt(d)) @ V
- Parallel iteration over batch and heads dimensions
- Tiled computation with online softmax for numerical stability

## v9 Features Demonstrated

- `@jit_kernel` decorator with `tl.*` primitives for attention computation
- `@workload` decorator with `P` namespace for parallel iteration
- `@kernel` stub for workload task definition
- Stream-based scheduling with `.streams()` and `.stream_by()`
- Dispatch policy with `.dispatch(DispatchPolicy.affinity(...))`
- Type checking with `TypeChecker` and layout join operations

## Prerequisites

- Python >= 3.10
- pto-wsp package (from project root's `python/` directory)

## How to Run

```bash
# From project root
python examples/attention/attention_example.py

# Or using Makefile
cd examples/attention
make run
```

## Expected Behavior

### Successful Execution

- Program completes without errors
- Outputs configuration summary (batch=4, heads=8, seq=512, dim=64)
- Shows JIT kernel information with tl.* operations
- Reports 32 total tasks (4 batches × 8 heads)
- Execution completes with "Example completed successfully!"

### Expected Output Sample

```
============================================================
PTO-RT v9 Multi-Head Attention Example
============================================================

Configuration:
  Batch size: 4
  Num heads:  8
  Seq length: 512
  Head dim:   64
  Total tasks: 32

JIT Kernel (@jit_kernel + tl.*):
  Name: attention_tile_jit
  Q tile: Tile[32, 64, F16]
  K tile: Tile[32, 64, F16]
  V tile: Tile[32, 64, F16]
  Operations: tl.matmul, tl.rsqrt, tl.rowmax, tl.exp, tl.div

Building workload...
  Workload type: Workload

Applying schedule...
  Scheduled type: Workload
  Streams: 4

Type checking...
  Layout join R ⊔ S(0): Layout(S(0), R, R, R)

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
- [x] Total tasks = 32 (4 × 8)
- [x] No `npu()` deprecation warnings

### Behavior Checking

Use `/codex` to verify example behavior:
```bash
codex exec "Run examples/attention/attention_example.py and verify:
1. Output matches expected behavior in README
2. All v9 features work correctly (@jit_kernel, @workload, tl.*)
3. No deprecated API warnings (npu() should not appear)
4. Task count shows 32 tasks
5. CPU simulation executes without errors"
```

### Fail Indicators

- ImportError or ModuleNotFoundError
- AttributeError (API mismatch)
- RuntimeError during execution
- Task count != 32
- Missing "Execution complete!" message
- DeprecationWarning for npu() usage

## Troubleshooting

**ModuleNotFoundError: No module named 'pto_wsp'**
- Run from project root directory
- Or add `sys.path.insert(0, 'python')` at script start

**AttributeError: module has no attribute**
- Check pto_wsp version matches v9 API
- Verify imports match current module structure
