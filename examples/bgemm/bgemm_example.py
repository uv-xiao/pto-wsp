#!/usr/bin/env python3
"""
PTO-RT v9 End-to-End Example: Batched GEMM (bgemm)

This example demonstrates batched general matrix multiplication:
  C[b] = A[b] @ B[b]  for b in 0..batch_size

Architecture:
- @kernel defines the kernel with tl.* primitives (Triton-style)
- Kernel is used directly in @workload with axis binding
- CPU simulation uses pto-isa's built-in CPU backend (-D__CPU_SIM)
- No separate stub or CPU implementation needed

Usage:
    python examples/bgemm/bgemm_example.py
"""

import sys
sys.path.insert(0, 'python')

from pto_wsp import (
    kernel, tl,
    In, Out, InOut, Tile,
    workload, P,
    Dense, Tensor, DType,
    DispatchPolicy, TimingPolicy,
    Deps, ReadyPolicy, StartPolicy, Pools, TaskWindow, WindowMode,
)

# DType shortcuts
F16 = DType.F16
F32 = DType.F32

# ============================================================
# Configuration
# ============================================================

BATCH_SIZE = 4
M = 512
N = 512
K = 256
TILE_M = 32
TILE_N = 32
TILE_K = 64

# ============================================================
# Axis Types
# ============================================================

batch = Dense[BATCH_SIZE]()
tile_m = Dense[M // TILE_M]()
tile_n = Dense[N // TILE_N]()
tile_k = Dense[K // TILE_K]()

# ============================================================
# Tensor Declarations
# ============================================================

A = Tensor(data=None, shape=(BATCH_SIZE, M, K), dtype=F16)
B = Tensor(data=None, shape=(BATCH_SIZE, K, N), dtype=F16)
C = Tensor(data=None, shape=(BATCH_SIZE, M, N), dtype=F16)

# ============================================================
# Kernel Definition (unified @kernel with tl.* primitives)
# ============================================================

@kernel
def gemm_tile(
    a_tile: In[Tile[TILE_M, TILE_K, F16]],
    b_tile: In[Tile[TILE_K, TILE_N, F16]],
    c_tile: InOut[Tile[TILE_M, TILE_N, F32]],
):
    """Tiled GEMM kernel: C_tile += A_tile @ B_tile

    Uses tl.* primitives that map to PTO-ISA tile operations:
    - tl.matmul -> TMATMUL (Cube unit matmul)
    - tl.add -> TADD (Vector unit add)
    - tl.store -> TSTORE (Data movement)

    When compiled with -D__CPU_SIM, these execute on CPU.
    When compiled for NPU, these generate Ascend kernel code.
    """
    # Matrix multiply with accumulation
    result = tl.matmul(a_tile, b_tile)

    # Add to existing C tile (accumulate)
    c_acc = tl.add(c_tile, result)

    # Store back
    tl.store(c_tile, c_acc)


# ============================================================
# Workload Definition
# ============================================================

@workload
def bgemm_workload():
    """Batched GEMM with 3D tiling.

    Type: Workload[Dense[4] x Dense[16] x Dense[16] x Dense[4], GemmTask, Sequential_K]

    Outer loop over batch dimension (parallel).
    Inner loops over M and N tiles (parallel).
    K-accumulation loop (sequential).
    """
    # Parallel over batch and output tiles
    for b in P(batch):
        for m, n in P(tile_m, tile_n):
            # Sequential accumulation along K dimension
            for k in P.seq(tile_k):
                gemm_tile[b, m, n, k](
                    a_tile=A[b],
                    b_tile=B[b],
                    c_tile=C[b],
                )


# ============================================================
# Task Graph Schedule (R9)
# ============================================================

def create_task_graph_schedule(wl):
    """Use task graph execution (pto-isa-lh compatible)."""
    return (wl
        .dispatch(DispatchPolicy.round_robin(num_aicpus=4))
        .task_graph(
            deps=Deps.infer_tensor_map_exact(),
            window=TaskWindow(8192, "tasks", WindowMode.STALL),
            pools=Pools.by_exec_unit(),
            ready=ReadyPolicy.work_steal(),
            start=StartPolicy.threshold(100),
        ))


# ============================================================
# Stream Schedule (Alternative)
# ============================================================

def create_stream_schedule(wl):
    """Use stream-based execution."""
    return (wl
        .dispatch(DispatchPolicy.affinity(lambda t: t.get("b")))
        .streams(BATCH_SIZE)
        .stream_by(lambda t: t.get("b"))  # One stream per batch
        .timing(TimingPolicy.immediate))


# ============================================================
# Main Entry Point
# ============================================================

def main():
    print("=" * 60)
    print("PTO-RT v9 Batched GEMM (bgemm) Example")
    print("=" * 60)
    print()

    # Configuration
    print("Configuration:")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Matrix dims: [{M}x{K}] @ [{K}x{N}] = [{M}x{N}]")
    print(f"  Tile size: {TILE_M}x{TILE_N}x{TILE_K}")
    total_tiles = BATCH_SIZE * (M // TILE_M) * (N // TILE_N) * (K // TILE_K)
    print(f"  Total tiles: {total_tiles}")
    print()

    # Kernel info
    print("Kernel (@kernel with tl.* primitives):")
    print(f"  Name: {gemm_tile.name}")
    print(f"  Input A: Tile[{TILE_M}, {TILE_K}, F16]")
    print(f"  Input B: Tile[{TILE_K}, {TILE_N}, F16]")
    print(f"  Output C: Tile[{TILE_M}, {TILE_N}, F32]")
    print(f"  Operations: tl.matmul, tl.add, tl.store")
    print()

    # Trace kernel to show IR
    print("Kernel IR (traced from tl.* primitives):")
    ir = gemm_tile.trace()
    print(f"  Parameters: {len(ir.params)}")
    print(f"  Operations: {len(ir.ops)}")
    for op in ir.ops:
        print(f"    {op}")
    print()

    # Build workload
    print("Building workload...")
    wl = bgemm_workload()
    print(f"  Workload kind: {wl._kind}")
    print()

    # Task graph schedule (R9)
    print("Applying task graph schedule (R9)...")
    task_graph_sched = create_task_graph_schedule(wl)
    cfg = task_graph_sched._task_graph_config
    print(f"  Deps mode: {cfg.deps.mode.value}")
    print(f"  Window size: {cfg.window.size}")
    print(f"  Pools: {cfg.pools._kind}")
    print(f"  Ready policy: {cfg.ready._kind}")
    print()

    # Stream schedule (alternative)
    print("Applying stream schedule (alternative)...")
    stream_sched = create_stream_schedule(wl)
    print(f"  Streams: {stream_sched._stream_count}")
    print()

    # Compile and execute
    print("Compiling program...")
    program = task_graph_sched.compile()
    print(f"  Program type: {type(program).__name__}")
    print()

    # Execute with CPU simulation (via pto-isa backend)
    print("Executing with CPU simulation (pto-isa backend)...")
    program.execute()
    print("Execution complete!")
    print()

    # Show generated Ascend code
    print("Generated Ascend code (for NPU target):")
    compiled = gemm_tile.compile(target="ascend")
    print(compiled.code[:500] + "..." if len(compiled.code) > 500 else compiled.code)
    print()

    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
