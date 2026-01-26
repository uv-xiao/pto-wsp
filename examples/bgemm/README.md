# Batched GEMM (bgemm) Example

## Overview

This example demonstrates batched general matrix multiplication using PTO-RT v9:
- C[b] = A[b] @ B[b] for b in 0..batch_size
- 3D tiling over batch, M, N, and K dimensions
- Sequential K-accumulation for numerical correctness

## v9 Features Demonstrated

- `@jit_kernel` decorator with `tl.*` primitives for matrix multiplication
- `@workload` decorator with `P` namespace for parallel/sequential iteration
- `P.seq()` for sequential K-accumulation
- Task graph scheduling with `.task_graph()`
- Dependency inference with `Deps.infer_tensor_map_exact()`
- Work-stealing ready policy with `ReadyPolicy.work_steal()`
- Window management with `TaskWindow`

## Prerequisites

- Python >= 3.10
- pto-wsp package (from project root's `python/` directory)

## How to Run

```bash
# From project root
python examples/bgemm/bgemm_example.py

# Or using Makefile
cd examples/bgemm
make run
```

## Expected Behavior

### Successful Execution

- Program completes without errors
- Outputs configuration summary (batch=4, M=512, N=512, K=256)
- Shows JIT kernel information with tl.matmul, tl.add, tl.store
- Reports 4096 total tiles (4 × 16 × 16 × 4)
- Demonstrates both task graph and stream scheduling
- Execution completes with "Example completed successfully!"

### Expected Output Sample

```
============================================================
PTO-RT v9 Batched GEMM (bgemm) Example
============================================================

Configuration:
  Batch size: 4
  Matrix dims: [512x256] @ [256x512] = [512x512]
  Tile size: 32x32x64
  Total tiles: 4096

JIT Kernel (@jit_kernel + tl.*):
  Name: gemm_tile_jit
  Input A: Tile[32, 64, F16]
  Input B: Tile[64, 32, F16]
  Output C: Tile[32, 32, F32]
  Operations: tl.matmul, tl.add, tl.store

Building workload...
  Workload kind: parallel_for

Applying task graph schedule (R9)...
  Deps mode: infer_tensor_map_exact
  Window size: 8192
  Pools: by_exec_unit
  Ready policy: work_steal

Applying stream schedule (alternative)...
  Streams: 4

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
- [x] Total tiles = 4096
- [x] Deps mode shows "infer_tensor_map_exact"
- [x] No `npu()` deprecation warnings

### Behavior Checking

Use `/codex` to verify example behavior:
```bash
codex exec "Run examples/bgemm/bgemm_example.py and verify:
1. Output matches expected behavior in README
2. All v9 features work correctly (@jit_kernel, @workload, tl.*)
3. Task graph schedule shows correct configuration
4. No deprecated API warnings (npu() should not appear)
5. CPU simulation executes without errors"
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
- Or add `sys.path.insert(0, 'python')` at script start

**Task count mismatch**
- Verify tile dimensions divide evenly into matrix dimensions
- Check batch size configuration
