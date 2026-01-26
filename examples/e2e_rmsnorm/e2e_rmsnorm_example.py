#!/usr/bin/env python3
"""
PTO-RT v9 End-to-End Example: RMSNorm with Full Pipeline Visibility

This example demonstrates the complete pipeline from Python to backend code:
  1. Python kernel definition with @kernel + tl.* primitives
  2. Kernel IR tracing (kernel.trace())
  3. Workload definition with axis binding
  4. Schedule with combinator API
  5. CPU simulation execution (via pto-isa backend)

Architecture:
- @kernel defines the kernel with tl.* primitives (Triton-style)
- Kernel is used directly in @workload with axis binding
- CPU simulation uses pto-isa's built-in CPU backend (-D__CPU_SIM)
- No separate stub or CPU implementation needed

Usage:
    python examples/e2e_rmsnorm/e2e_rmsnorm_example.py
"""

import sys
sys.path.insert(0, 'python')

from pto_wsp import (
    kernel, tl,
    In, Out, Tile, Scalar,
    workload, P,
    Dense, DenseDyn, Tensor, DType,
    DispatchPolicy, TimingPolicy,
)

# DType shortcuts
F16 = DType.F16
F32 = DType.F32

# ============================================================
# Configuration
# ============================================================

BATCH_SIZE = 4
SEQ_LEN = 512
HIDDEN_DIM = 128
TILE_SIZE = 32  # Process 32 elements at a time

print("=" * 70)
print("PTO-RT v9 End-to-End Example: RMSNorm")
print("=" * 70)

# ============================================================
# STAGE 1: Kernel Definition with @kernel + tl.* Primitives
# ============================================================

print("\n[Stage 1] Kernel Definition (@kernel + tl.* Primitives)")
print("-" * 50)

@kernel
def rmsnorm_kernel(
    x: In[Tile[TILE_SIZE, HIDDEN_DIM, F16]],
    out: Out[Tile[TILE_SIZE, HIDDEN_DIM, F16]],
    eps: Scalar[F32] = 1e-6
):
    """RMS normalization using Triton-style tl.* primitives.

    Uses tl.* primitives that map to PTO-ISA tile operations:
    - tl.mul -> TMUL (Vector unit multiply)
    - tl.rowmean -> TREDUCE (Reduction unit)
    - tl.rsqrt -> TRSQRT (Vector unit)
    - tl.store -> TSTORE (Data movement)

    When compiled with -D__CPU_SIM, these execute on CPU.
    When compiled for NPU, these generate Ascend kernel code.

    Formula: out = x * rsqrt(mean(x^2) + eps)
    """
    # Square input elements
    sq = tl.mul(x, x)  # sq: Value, not a string!

    # Compute row-wise mean of squared values
    mean_sq = tl.rowmean(sq)  # mean_sq: Value

    # Add epsilon and compute reciprocal square root
    rsqrt_val = tl.rsqrt(mean_sq)  # Note: eps added internally

    # Scale input by rsqrt
    result = tl.mul(x, rsqrt_val)  # result: Value

    # Store result
    tl.store(out, result)

print(f"Kernel defined: {rmsnorm_kernel.name}")
print(f"  Input:  x[{TILE_SIZE}, {HIDDEN_DIM}] F16")
print(f"  Output: out[{TILE_SIZE}, {HIDDEN_DIM}] F16")
print(f"  Scalar: eps = 1e-6")
print(f"  Operations: tl.mul, tl.rowmean, tl.rsqrt, tl.store")

# ============================================================
# STAGE 2: Show Kernel IR (traced from tl.* primitives)
# ============================================================

print("\n[Stage 2] Kernel IR (traced from tl.* primitives)")
print("-" * 50)

# Trace the kernel to build KernelIR
ir = rmsnorm_kernel.trace()
print(f"  Parameters: {len(ir.params)}")
print(f"  Operations: {len(ir.ops)}")
for op in ir.ops:
    print(f"    {op}")

# ============================================================
# STAGE 3: Workload Definition with @workload + P namespace
# ============================================================

print("\n[Stage 3] Workload Definition (@workload + P)")
print("-" * 50)

# Axis types
batch = DenseDyn(BATCH_SIZE)
seq_tiles = Dense[SEQ_LEN // TILE_SIZE]()

# Tensors
X = Tensor(data=None, shape=(BATCH_SIZE, SEQ_LEN, HIDDEN_DIM), dtype=DType.F16)
Y = Tensor(data=None, shape=(BATCH_SIZE, SEQ_LEN, HIDDEN_DIM), dtype=DType.F16)

@workload
def rmsnorm_workload():
    """Apply RMSNorm to each batch and tile position.

    Type: Workload[DenseDyn x Dense[16], RMSNormTask, Independent]

    Uses the unified @kernel directly with axis binding.
    The kernel contains tl.* primitives that execute via pto-isa backend.
    """
    for b in P(batch):
        for t in P(seq_tiles):
            rmsnorm_kernel[b, t](x=X, out=Y)

print(f"Workload: rmsnorm_workload")
print(f"  Batch axis: DenseDyn({BATCH_SIZE})")
print(f"  Tile axis:  Dense[{SEQ_LEN // TILE_SIZE}]")
print(f"  Total tasks: {BATCH_SIZE} x {SEQ_LEN // TILE_SIZE} = {BATCH_SIZE * (SEQ_LEN // TILE_SIZE)}")

# ============================================================
# STAGE 4: Schedule with Combinator API
# ============================================================

print("\n[Stage 4] Schedule with Combinator API")
print("-" * 50)

# Create scheduled workload
scheduled = (rmsnorm_workload()
    .dispatch(DispatchPolicy.round_robin(num_aicpus=2))
    .streams(2)
    .stream_by(lambda t: t.get("b") % 2)  # Batch-based stream assignment
    .timing(TimingPolicy.immediate))

print("Schedule configuration:")
print("  .dispatch(DispatchPolicy.round_robin())")
print("  .streams(2)")
print("  .stream_by(lambda t: t.get('b') % 2)")
print("  .timing(TimingPolicy.immediate)")

# ============================================================
# STAGE 5: Task Enumeration
# ============================================================

print("\n[Stage 5] Task Enumeration")
print("-" * 50)

# Enumerate tasks from workload
if hasattr(scheduled, 'enumerate'):
    tasks = list(scheduled.enumerate())
    print(f"Enumerated {len(tasks)} tasks:")
    for i, task in enumerate(tasks[:5]):  # Show first 5
        print(f"  Task {i}: kernel={task.kernel}, params={task.params}")
    if len(tasks) > 5:
        print(f"  ... and {len(tasks) - 5} more tasks")
else:
    print("(Task enumeration requires compiled program)")

# ============================================================
# STAGE 6: Compilation
# ============================================================

print("\n[Stage 6] Compilation")
print("-" * 50)

program = scheduled.compile()
print(f"Program compiled successfully")
print(f"  Type: {type(program).__name__}")

# ============================================================
# STAGE 7: CPU Simulation Execution (via pto-isa backend)
# ============================================================

print("\n[Stage 7] CPU Simulation Execution (pto-isa backend)")
print("-" * 50)

# No need to register CPU implementation - pto-isa backend handles it
# The @kernel with tl.* primitives compiles to pto-isa code that can
# run in CPU simulation mode (-D__CPU_SIM)

print("Executing with CPU simulation backend...")
program.execute()
print("Execution complete!")

# Get statistics
if hasattr(program, 'stats'):
    stats = program.stats
    if callable(stats):
        stats = stats()
    print(f"\nExecution Statistics:")
    print(f"  Tasks executed: {stats.num_tasks if hasattr(stats, 'num_tasks') else 'N/A'}")
    print(f"  Compile time:   {stats.compile_time_ms if hasattr(stats, 'compile_time_ms') else 'N/A'} ms")
    print(f"  Execute time:   {stats.execute_time_ms if hasattr(stats, 'execute_time_ms') else 'N/A'} ms")

# ============================================================
# STAGE 8: Show Generated Backend Code
# ============================================================

print("\n[Stage 8] Generated Backend Code")
print("-" * 50)

# Show generated Ascend code (for NPU target)
print("Generated Ascend code (for NPU target):")
compiled = rmsnorm_kernel.compile(target="ascend")
print(compiled.code[:500] + "..." if len(compiled.code) > 500 else compiled.code)

# ============================================================
# Summary
# ============================================================

print("\n" + "=" * 70)
print("End-to-End Pipeline Summary")
print("=" * 70)
print("""
1. Unified @kernel (with tl.* primitives):
   - Single @kernel decorator for kernel definition
   - Triton-style primitives: tl.mul, tl.rowmean, tl.rsqrt, tl.store
   - Maps to PTO-ISA tile operations
   - CPU simulation via -D__CPU_SIM flag
   - No separate CPU implementation needed

2. Workload (@workload + P):
   - Declarative task generation
   - Type: Workload[Axes, Task, Deps]
   - kernel[axes](...) used directly in workload
   - P() for parallel, P.seq() for sequential

3. Schedule (combinator API):
   - .dispatch() - task routing
   - .streams() - concurrent execution
   - .timing() - issue policy
   - Type-safe method chaining

4. Compilation:
   - workload.compile() -> Program
   - Backend-specific lowering via pto-isa

5. Execution:
   - program.execute() - run tasks
   - CPU simulation or NPU execution
   - No kernel registration required
""")

print("Example completed successfully!")
