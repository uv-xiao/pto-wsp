# End-to-End RMSNorm Pipeline Example

## Overview

This example demonstrates the complete PTO-RT v9 pipeline from Python definition to execution:
1. JIT Kernel Definition with `tl.*` primitives
2. Kernel IR generation (lazy tracing)
3. Workload Definition with `@workload` + `P`
4. Schedule configuration with combinator API
5. Task Enumeration
6. Compilation to Program
7. CPU Simulation Execution

This is the **canonical end-to-end example** showing all 7 stages of the PTO-RT pipeline.

## v9 Features Demonstrated

- `@jit_kernel` decorator with `tl.*` primitives:
  - `tl.mul`, `tl.rowmean`, `tl.add`, `tl.rsqrt`, `tl.store`
- `@workload` decorator with `P` namespace
- `Tile[H, W, DType]` and `Scalar[DType]` type annotations
- Stream-based scheduling with `.streams()` and `.stream_by()`
- Round-robin dispatch policy
- Immediate timing policy
- Task enumeration with `workload.enumerate()`
- Compilation and CPU simulation execution
- Execution statistics (compile time, execute time)

## Prerequisites

- Python >= 3.10
- pto-wsp package (from project root's `python/` directory)

## How to Run

```bash
# From project root
python examples/e2e_rmsnorm/e2e_rmsnorm_example.py

# Or using Makefile
cd examples/e2e_rmsnorm
make run
```

## Expected Behavior

### Successful Execution

- Program shows all 7 stages of the pipeline
- Stage 1: JIT Kernel with typed annotations
- Stage 2: Kernel IR (built lazily)
- Stage 3: Workload with 64 tasks (4 batch × 16 tiles)
- Stage 4: Schedule with streams and timing
- Stage 5: Task enumeration showing first 5 tasks
- Stage 6: Compilation to Program
- Stage 7: CPU simulation execution with timing

### Expected Output Sample

```
======================================================================
PTO-RT v9 End-to-End Example: RMSNorm
======================================================================

[Stage 1] JIT Kernel Definition with tl.* Primitives
--------------------------------------------------
Kernel defined: rmsnorm_kernel
  Input:  x[32, 128] F16
  Output: out[32, 128] F16
  Scalar: eps = 1e-6

[Stage 2] Kernel IR (from @jit_kernel tracing)
--------------------------------------------------
(Kernel IR is built lazily on first call)

[Stage 3] Workload Definition (@workload + P)
--------------------------------------------------
Workload: rmsnorm_workload
  Batch axis: DenseDyn(4)
  Tile axis:  Dense[16]
  Total tasks: 4 x 16 = 64

[Stage 4] Schedule with Combinator API
--------------------------------------------------
Schedule configuration:
  .dispatch(DispatchPolicy.round_robin())
  .streams(2)
  .stream_by(lambda t: t.get('b') % 2)
  .timing(TimingPolicy.immediate)

[Stage 5] Task Enumeration
--------------------------------------------------
Enumerated 64 tasks:
  Task 0: kernel=rmsnorm, params=[0, 0]
  Task 1: kernel=rmsnorm, params=[0, 1]
  ...

[Stage 6] Compilation
--------------------------------------------------
Program compiled successfully
  Type: Program

[Stage 7] CPU Simulation Execution
--------------------------------------------------
Executing with CPU simulation backend...
Execution complete!

Execution Statistics:
  Tasks executed: N/A
  Compile time:   X.XX ms
  Execute time:   X.XX ms

======================================================================
End-to-End Pipeline Summary
======================================================================
...
Example completed successfully!
```

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
- Or add `sys.path.insert(0, 'python')` at script start

**Stage X not showing**
- Verify Python version >= 3.10
- Check pto_wsp module is up-to-date
