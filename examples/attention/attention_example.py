#!/usr/bin/env python3
"""
PTO-RT v9 End-to-End Example: Multi-Head Attention

This example demonstrates the complete Python frontend workflow:
1. Define axis types and tensors
2. Define kernels with @kernel using tl.* primitives
3. Define workload with @workload decorator and P namespace
4. Apply schedule combinators
5. Compile and execute with CPU simulation (via pto-isa)

Usage:
    python examples/attention/attention_example.py
"""

import sys
sys.path.insert(0, 'python')

from pto_wsp import (
    kernel, tl,
    In, Out, Tile,
    workload, P,
    Dense, Tensor, DType,
    DispatchPolicy, TimingPolicy,
    Deps, ReadyPolicy, StartPolicy, Pools,
    TypeChecker, Layout, layout_join, LayoutCompatibilityError,
)

# DType shortcuts
F16 = DType.F16
F32 = DType.F32

# ============================================================
# Configuration
# ============================================================

BATCH_SIZE = 4
NUM_HEADS = 8
SEQ_LEN = 512
HEAD_DIM = 64
TILE_SEQ = 32  # Tile size for sequence dimension

# ============================================================
# Axis Types
# ============================================================

batch = Dense[BATCH_SIZE]()
heads = Dense[NUM_HEADS]()
seq = Dense[SEQ_LEN]()
dim = Dense[HEAD_DIM]()

# ============================================================
# Tensor Declarations
# ============================================================

Q = Tensor(data=None, shape=(BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=F16)
K = Tensor(data=None, shape=(BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=F16)
V = Tensor(data=None, shape=(BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=F16)
O = Tensor(data=None, shape=(BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=F16)

# ============================================================
# Kernel Definition (unified @kernel with tl.* primitives)
# ============================================================

@kernel
def attention_kernel(
    q_tile: In[Tile[TILE_SEQ, HEAD_DIM, F16]],
    k_tile: In[Tile[TILE_SEQ, HEAD_DIM, F16]],
    v_tile: In[Tile[TILE_SEQ, HEAD_DIM, F16]],
    out_tile: Out[Tile[TILE_SEQ, HEAD_DIM, F16]],
):
    """Tiled attention kernel: O = softmax(Q @ K^T / sqrt(d)) @ V

    Uses tl.* primitives that map to PTO-ISA operations:
    - tl.matmul -> TMATMUL (Cube unit)
    - tl.rowmax/rowsum -> TREDUCE (Vector unit)
    - tl.exp/div -> TEXP/TDIV (Vector unit)

    Implements scaled dot-product attention with online softmax.
    """
    # Compute Q @ K^T (attention scores)
    qk = tl.matmul(q_tile, k_tile)  # [TILE_SEQ, TILE_SEQ]

    # Scale by 1/sqrt(d) - use mul with constant
    scale_val = tl.alloc((1, 1), dtype=F32)  # Scalar-like tile
    qk_scaled = tl.mul(qk, scale_val)

    # Online softmax
    max_score = tl.rowmax(qk_scaled)
    qk_shifted = tl.sub(qk_scaled, max_score)
    exp_scores = tl.exp(qk_shifted)
    sum_exp = tl.rowsum(exp_scores)
    attn_weights = tl.div(exp_scores, sum_exp)

    # Weighted sum: O = attn_weights @ V
    result = tl.matmul(attn_weights, v_tile)

    # Store output
    tl.store(out_tile, result)


# ============================================================
# Workload Definition
# ============================================================

@workload
def attention_workload():
    """Multi-head attention workload.

    Type: Workload[Dense[4] x Dense[8], AttentionTask, Independent]

    Parallel iteration over batch and heads dimensions.
    Each iteration calls the attention kernel for one (batch, head) pair.
    """
    for b, h in P(batch, heads):
        attention_kernel[b, h](
            q_tile=Q[b][h],
            k_tile=K[b][h],
            v_tile=V[b][h],
            out_tile=O[b][h],
        )


# ============================================================
# Schedule Definition
# ============================================================

def create_schedule(wl):
    """Apply schedule combinators to the workload."""
    return (wl
        .dispatch(DispatchPolicy.affinity(lambda t: t.get("b")))
        .streams(4)
        .stream_by(lambda t: t.get("h") % 4)
        .timing(TimingPolicy.immediate))


# ============================================================
# Main Entry Point
# ============================================================

def main():
    print("=" * 60)
    print("PTO-RT v9 Multi-Head Attention Example")
    print("=" * 60)
    print()

    # Show configuration
    print("Configuration:")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Num heads:  {NUM_HEADS}")
    print(f"  Seq length: {SEQ_LEN}")
    print(f"  Head dim:   {HEAD_DIM}")
    print(f"  Total tasks: {BATCH_SIZE * NUM_HEADS}")
    print()

    # Kernel info
    print("Kernel (@kernel with tl.* primitives):")
    print(f"  Name: {attention_kernel.name}")
    print(f"  Q tile: Tile[{TILE_SEQ}, {HEAD_DIM}, F16]")
    print(f"  K tile: Tile[{TILE_SEQ}, {HEAD_DIM}, F16]")
    print(f"  V tile: Tile[{TILE_SEQ}, {HEAD_DIM}, F16]")
    print(f"  Operations: tl.matmul, tl.rowmax, tl.exp, tl.div")
    print()

    # Trace kernel to show IR
    print("Kernel IR (traced from tl.* primitives):")
    ir = attention_kernel.trace()
    print(f"  Parameters: {len(ir.params)}")
    print(f"  Operations: {len(ir.ops)}")
    for op in ir.ops:
        print(f"    {op}")
    print()

    # Build workload
    print("Building workload...")
    wl = attention_workload()
    print(f"  Workload type: {type(wl).__name__}")
    print()

    # Apply schedule
    print("Applying schedule...")
    scheduled = create_schedule(wl)
    print(f"  Scheduled type: {type(scheduled).__name__}")
    print(f"  Streams: {scheduled._stream_count}")
    print()

    # Type checking example
    print("Type checking...")
    checker = TypeChecker()

    layout_r = Layout.default(4)  # All replicated
    layout_s = Layout.sharded(0, 4)  # Sharded on dim 0

    try:
        joined = layout_join(layout_r, layout_s)
        print(f"  Layout join R ⊔ S(0): {joined}")
    except LayoutCompatibilityError as e:
        print(f"  Layout error: {e}")
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

    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
