#!/usr/bin/env python3
"""
PTO-RT v9 End-to-End Example: Online Softmax

This example demonstrates online softmax with tiled computation:
  y = softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))

Uses the online algorithm to avoid materializing the full intermediate.

Architecture:
- @kernel defines the kernel with tl.* primitives (Triton-style)
- Kernel is used directly in @workload with axis binding
- CPU simulation uses pto-isa's built-in CPU backend (-D__CPU_SIM)
- No separate stub or CPU implementation needed

Usage:
    python examples/softmax/softmax_example.py
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
SEQ_LEN = 2048
VOCAB_SIZE = 32000
TILE_ROW = 32
TILE_COL = 1024  # Process multiple columns per tile

# ============================================================
# Axis Types
# ============================================================

batch = Dense[BATCH_SIZE]()
seq_tiles = Dense[SEQ_LEN // TILE_ROW]()
vocab_tiles = Dense[VOCAB_SIZE // TILE_COL]()

# ============================================================
# Tensor Declarations
# ============================================================

# Input logits [batch, seq, vocab]
logits = Tensor(
    data=None,
    shape=(BATCH_SIZE, SEQ_LEN, VOCAB_SIZE),
    dtype=F16
)

# Output probabilities [batch, seq, vocab]
probs = Tensor(
    data=None,
    shape=(BATCH_SIZE, SEQ_LEN, VOCAB_SIZE),
    dtype=F16
)

# Intermediate: running max per row [batch, seq]
running_max = Tensor(
    data=None,
    shape=(BATCH_SIZE, SEQ_LEN),
    dtype=F32
)

# Intermediate: running sum per row [batch, seq]
running_sum = Tensor(
    data=None,
    shape=(BATCH_SIZE, SEQ_LEN),
    dtype=F32
)

# ============================================================
# Kernel Definitions (unified @kernel with tl.* primitives)
# ============================================================

@kernel
def online_softmax_tile(
    x_tile: In[Tile[TILE_ROW, TILE_COL, F16]],
    y_tile: Out[Tile[TILE_ROW, TILE_COL, F16]],
    m_prev: In[Tile[TILE_ROW, 1, F32]],
    m_new: Out[Tile[TILE_ROW, 1, F32]],
    l_prev: In[Tile[TILE_ROW, 1, F32]],
    l_new: Out[Tile[TILE_ROW, 1, F32]],
):
    """Online softmax tile kernel using tl.* primitives.

    Updates running max and sum while computing partial output.
    Final pass rescales output.

    Uses tl.* primitives that map to PTO-ISA tile operations:
    - tl.rowmax -> TREDUCE_MAX (Vector unit reduction)
    - tl.max -> TMAX (Vector unit max)
    - tl.sub -> TSUB (Vector unit subtract)
    - tl.exp -> TEXP (Vector unit exp)
    - tl.rowsum -> TREDUCE_SUM (Vector unit reduction)
    - tl.add -> TADD (Vector unit add)
    - tl.store -> TSTORE (Data movement)

    When compiled with -D__CPU_SIM, these execute on CPU.
    When compiled for NPU, these generate Ascend kernel code.
    """
    # Compute row max of current tile
    m_tile = tl.rowmax(x_tile)

    # New running max: max(m_prev, m_tile)
    m_combined = tl.max(m_prev, m_tile)
    tl.store(m_new, m_combined)

    # exp(x - m_new): shift by max for numerical stability
    x_shifted = tl.sub(x_tile, m_combined)
    exp_x = tl.exp(x_shifted)

    # Update running sum
    s_tile = tl.rowsum(exp_x)
    l_combined = tl.add(l_prev, s_tile)
    tl.store(l_new, l_combined)

    # Store exp values (will need rescaling in final pass)
    tl.store(y_tile, exp_x)


@kernel
def softmax_rescale(
    y_tile: InOut[Tile[TILE_ROW, TILE_COL, F16]],
    l_tile: In[Tile[TILE_ROW, 1, F32]],
):
    """Final rescaling by 1/sum using tl.* primitives.

    Uses tl.* primitives that map to PTO-ISA tile operations:
    - tl.rsqrt -> TRSQRT (Vector unit reciprocal sqrt)
    - tl.mul -> TMUL (Vector unit multiply)
    - tl.store -> TSTORE (Data movement)

    When compiled with -D__CPU_SIM, these execute on CPU.
    When compiled for NPU, these generate Ascend kernel code.
    """
    # Compute 1/l (reciprocal)
    inv_l = tl.rsqrt(l_tile)
    inv_l = tl.mul(inv_l, inv_l)  # rsqrt^2 = 1/l (simplified)

    # Scale y by 1/sum
    y_scaled = tl.mul(y_tile, inv_l)

    # Store rescaled output
    tl.store(y_tile, y_scaled)


# ============================================================
# Workload Definition
# ============================================================

@workload
def softmax_workload():
    """Online softmax workload.

    Type: Workload[Dense[4] x Dense[64] x Dense[31], SoftmaxTask, Sequential_V]

    Phase 1: Sequential pass to compute max and sum (along vocab tiles)
    Phase 2: Parallel rescaling
    """
    # Phase 1: Online computation (sequential along vocab tiles)
    for b in P(batch):
        for s in P(seq_tiles):
            for v in P.seq(vocab_tiles):
                online_softmax_tile[b, s, v](
                    x_tile=logits[b],
                    y_tile=probs[b],
                    m_prev=running_max[b],
                    m_new=running_max[b],
                    l_prev=running_sum[b],
                    l_new=running_sum[b],
                )

    # Phase 2: Rescaling (parallel)
    for b in P(batch):
        for s in P(seq_tiles):
            for v in P(vocab_tiles):
                softmax_rescale[b, s, v](
                    y_tile=probs[b],
                    l_tile=running_sum[b],
                )


# ============================================================
# Task Graph Schedule (R9)
# ============================================================

def create_schedule(wl):
    """Task graph schedule with dependency inference."""
    return (wl
        .dispatch(DispatchPolicy.round_robin(4))
        .task_graph(
            # Use tensor map exact for RAW dependencies
            deps=Deps.infer_tensor_map_exact(),
            window=TaskWindow(4096, "tasks", WindowMode.STALL),
            pools=Pools.single(),  # All vector ops
            ready=ReadyPolicy.fifo(),
            start=StartPolicy.after_orchestration(),
        ))


# ============================================================
# Main Entry Point
# ============================================================

def main():
    print("=" * 60)
    print("PTO-RT v9 Online Softmax Example")
    print("=" * 60)
    print()

    # Configuration
    print("Configuration:")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Sequence length: {SEQ_LEN}")
    print(f"  Vocabulary size: {VOCAB_SIZE}")
    print(f"  Tile size: {TILE_ROW} x {TILE_COL}")
    n_online_tiles = BATCH_SIZE * (SEQ_LEN // TILE_ROW) * (VOCAB_SIZE // TILE_COL)
    n_rescale_tiles = n_online_tiles
    print(f"  Online tiles: {n_online_tiles}")
    print(f"  Rescale tiles: {n_rescale_tiles}")
    print()

    # Kernel info
    print("Kernels (@kernel with tl.* primitives):")
    print(f"  online_softmax_tile:")
    print(f"    Name: {online_softmax_tile.name}")
    print(f"    Input x: Tile[{TILE_ROW}, {TILE_COL}, F16]")
    print(f"    Output y: Tile[{TILE_ROW}, {TILE_COL}, F16]")
    print(f"    Running max: Tile[{TILE_ROW}, 1, F32]")
    print(f"    Running sum: Tile[{TILE_ROW}, 1, F32]")
    print(f"    Operations: tl.rowmax, tl.max, tl.sub, tl.exp, tl.rowsum, tl.add, tl.store")
    print(f"  softmax_rescale:")
    print(f"    Name: {softmax_rescale.name}")
    print(f"    Operations: tl.rsqrt, tl.mul, tl.store")
    print()

    # Trace kernel to show IR
    print("Kernel IR (traced from tl.* primitives):")
    ir = online_softmax_tile.trace()
    print(f"  Parameters: {len(ir.params)}")
    print(f"  Operations: {len(ir.ops)}")
    for op in ir.ops:
        print(f"    {op}")
    print()

    # Build workload
    print("Building workload...")
    wl = softmax_workload()
    print(f"  Workload kind: {wl._kind}")
    print()

    # Apply schedule
    print("Applying task graph schedule...")
    scheduled = create_schedule(wl)
    cfg = scheduled._task_graph_config
    print(f"  Deps: {cfg.deps.mode.value}")
    print(f"  Window: {cfg.window.size} {cfg.window.unit}")
    print(f"  Ready: {cfg.ready._kind}")
    print()

    # Compile and execute
    print("Compiling program...")
    program = scheduled.compile()
    print(f"  Program type: {type(program).__name__}")
    print()

    # Execute with CPU simulation (via pto-isa backend)
    print("Executing with CPU simulation (pto-isa backend)...")
    program.execute()
    print("Execution complete!")
    print()

    # Show generated Ascend code
    print("Generated Ascend code (for NPU target):")
    compiled = online_softmax_tile.compile(target="ascend")
    print(compiled.code[:500] + "..." if len(compiled.code) > 500 else compiled.code)
    print()

    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
