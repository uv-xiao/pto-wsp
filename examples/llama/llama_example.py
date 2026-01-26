#!/usr/bin/env python3
"""
PTO-RT v9 End-to-End Example: LLaMA-style Transformer

This example demonstrates a complete LLaMA transformer layer:
- RMSNorm
- Multi-head attention with RoPE
- MLP (SwiGLU)
- Residual connections

Architecture:
- @kernel defines the kernel with tl.* primitives (Triton-style)
- Kernel is used directly in @workload with axis binding
- CPU simulation uses pto-isa's built-in CPU backend (-D__CPU_SIM)
- No separate stub or CPU implementation needed

Usage:
    python examples/llama/llama_example.py
"""

import sys
sys.path.insert(0, 'python')

from pto_wsp import (
    kernel, tl,
    In, Out, InOut, Tile, Scalar,
    workload, P,
    Dense, Tensor, DType,
    DispatchPolicy, TimingPolicy,
    Deps, DepsMode, ReadyPolicy, StartPolicy, TracePolicy, Pools, TaskWindow, WindowMode,
    TensorLayout,
)

# DType shortcuts
F16 = DType.F16
F32 = DType.F32

# ============================================================
# Configuration (LLaMA-7B style)
# ============================================================

BATCH_SIZE = 1
SEQ_LEN = 512
HIDDEN_DIM = 4096
NUM_HEADS = 32
HEAD_DIM = HIDDEN_DIM // NUM_HEADS  # 128
MLP_DIM = 11008  # LLaMA's MLP dimension
NUM_LAYERS = 1  # Single layer for example
TILE_SIZE = 32

# ============================================================
# Axis Types
# ============================================================

batch = Dense[BATCH_SIZE]()
seq = Dense[SEQ_LEN]()
heads = Dense[NUM_HEADS]()
seq_tiles = Dense[SEQ_LEN // TILE_SIZE]()
hidden_tiles = Dense[HIDDEN_DIM // TILE_SIZE]()
mlp_tiles = Dense[MLP_DIM // TILE_SIZE]()

# ============================================================
# Tensor Declarations with Layout (R10)
# ============================================================

# Input/Output hidden states [batch, seq, hidden]
hidden = Tensor(
    data=None,
    shape=(BATCH_SIZE, SEQ_LEN, HIDDEN_DIM),
    dtype=F16,
    layout=TensorLayout.default(3)
)

# Attention tensors
q_proj = Tensor(data=None, shape=(BATCH_SIZE, SEQ_LEN, HIDDEN_DIM), dtype=F16)
k_proj = Tensor(data=None, shape=(BATCH_SIZE, SEQ_LEN, HIDDEN_DIM), dtype=F16)
v_proj = Tensor(data=None, shape=(BATCH_SIZE, SEQ_LEN, HIDDEN_DIM), dtype=F16)
attn_out = Tensor(data=None, shape=(BATCH_SIZE, SEQ_LEN, HIDDEN_DIM), dtype=F16)

# MLP tensors
gate_proj = Tensor(data=None, shape=(BATCH_SIZE, SEQ_LEN, MLP_DIM), dtype=F16)
up_proj = Tensor(data=None, shape=(BATCH_SIZE, SEQ_LEN, MLP_DIM), dtype=F16)
mlp_out = Tensor(data=None, shape=(BATCH_SIZE, SEQ_LEN, HIDDEN_DIM), dtype=F16)

# Weights (would be actual data in real impl)
wq = Tensor(data=None, shape=(HIDDEN_DIM, HIDDEN_DIM), dtype=F16)
wk = Tensor(data=None, shape=(HIDDEN_DIM, HIDDEN_DIM), dtype=F16)
wv = Tensor(data=None, shape=(HIDDEN_DIM, HIDDEN_DIM), dtype=F16)
wo = Tensor(data=None, shape=(HIDDEN_DIM, HIDDEN_DIM), dtype=F16)
w_gate = Tensor(data=None, shape=(HIDDEN_DIM, MLP_DIM), dtype=F16)
w_up = Tensor(data=None, shape=(HIDDEN_DIM, MLP_DIM), dtype=F16)
w_down = Tensor(data=None, shape=(MLP_DIM, HIDDEN_DIM), dtype=F16)

# Position tensor for RoPE
positions = Tensor(data=None, shape=(SEQ_LEN,), dtype=F32)

# ============================================================
# Kernel Definitions (unified @kernel with tl.* primitives)
# ============================================================

@kernel
def rmsnorm_tile(
    x_tile: In[Tile[TILE_SIZE, TILE_SIZE, F16]],
    y_tile: Out[Tile[TILE_SIZE, TILE_SIZE, F16]],
    eps: Scalar[F32],
):
    """RMSNorm tile kernel: y = x / sqrt(mean(x^2) + eps)

    Uses tl.* primitives that map to PTO-ISA tile operations:
    - tl.mul -> TMUL (Vector unit multiply)
    - tl.rowmean -> TROWMEAN (Vector unit reduction)
    - tl.add -> TADD (Vector unit add)
    - tl.rsqrt -> TRSQRT (Vector unit reciprocal sqrt)
    - tl.store -> TSTORE (Data movement)

    When compiled with -D__CPU_SIM, these execute on CPU.
    When compiled for NPU, these generate Ascend kernel code.
    """
    # Compute x^2
    sq = tl.mul(x_tile, x_tile)

    # Mean of squared values
    mean_sq = tl.rowmean(sq)

    # Add epsilon and compute rsqrt
    mean_eps = tl.add(mean_sq, eps)
    scale = tl.rsqrt(mean_eps)

    # Scale input
    result = tl.mul(x_tile, scale)

    tl.store(y_tile, result)


@kernel
def linear_tile(
    x_tile: In[Tile[TILE_SIZE, TILE_SIZE, F16]],
    w_tile: In[Tile[TILE_SIZE, TILE_SIZE, F16]],
    y_tile: Out[Tile[TILE_SIZE, TILE_SIZE, F32]],
):
    """Linear projection tile kernel: y = x @ w

    Uses tl.* primitives:
    - tl.matmul -> TMATMUL (Cube unit matmul)
    - tl.store -> TSTORE (Data movement)
    """
    result = tl.matmul(x_tile, w_tile)
    tl.store(y_tile, result)


@kernel
def rope_tile(
    x_tile: InOut[Tile[TILE_SIZE, HEAD_DIM, F16]],
    pos_tile: In[Tile[TILE_SIZE, F32]],
):
    """Rotary Position Embedding tile kernel.

    Uses tl.* primitives:
    - tl.sin, tl.cos -> TSIN, TCOS (Vector unit)
    - tl.mul, tl.add, tl.sub -> TMUL, TADD, TSUB (Vector unit)
    - tl.store -> TSTORE (Data movement)

    Applies RoPE: x' = x * cos(pos) + rotate(x) * sin(pos)
    """
    # Compute sin/cos of positions
    cos_pos = tl.cos(pos_tile)
    sin_pos = tl.sin(pos_tile)

    # Split x into even/odd pairs for rotation
    x_even = tl.slice_even(x_tile)
    x_odd = tl.slice_odd(x_tile)

    # Apply rotation: [x_even, x_odd] -> [x_even * cos - x_odd * sin, x_odd * cos + x_even * sin]
    new_even = tl.sub(tl.mul(x_even, cos_pos), tl.mul(x_odd, sin_pos))
    new_odd = tl.add(tl.mul(x_odd, cos_pos), tl.mul(x_even, sin_pos))

    # Interleave back
    result = tl.interleave(new_even, new_odd)

    tl.store(x_tile, result)


@kernel
def attention_tile(
    q_tile: In[Tile[TILE_SIZE, HEAD_DIM, F16]],
    k_tile: In[Tile[TILE_SIZE, HEAD_DIM, F16]],
    v_tile: In[Tile[TILE_SIZE, HEAD_DIM, F16]],
    o_tile: Out[Tile[TILE_SIZE, HEAD_DIM, F16]],
):
    """Attention tile kernel: O = softmax(Q @ K^T / sqrt(d)) @ V

    Uses tl.* primitives:
    - tl.matmul -> TMATMUL (Cube unit)
    - tl.rsqrt -> TRSQRT (Vector unit)
    - tl.rowmax -> TROWMAX (Vector unit reduction)
    - tl.exp -> TEXP (Vector unit)
    - tl.rowsum -> TROWSUM (Vector unit reduction)
    - tl.div -> TDIV (Vector unit)
    - tl.store -> TSTORE (Data movement)
    """
    # Q @ K^T
    qk = tl.matmul(q_tile, k_tile)

    # Scale by 1/sqrt(d)
    scale = tl.rsqrt(tl.constant(HEAD_DIM, F32))
    qk_scaled = tl.mul(qk, scale)

    # Softmax
    max_score = tl.rowmax(qk_scaled)
    qk_shifted = tl.sub(qk_scaled, max_score)
    exp_scores = tl.exp(qk_shifted)
    sum_exp = tl.rowsum(exp_scores)
    attn_weights = tl.div(exp_scores, sum_exp)

    # Weighted sum
    result = tl.matmul(attn_weights, v_tile)

    tl.store(o_tile, result)


@kernel
def silu_tile(
    x_tile: InOut[Tile[TILE_SIZE, TILE_SIZE, F16]],
):
    """SiLU activation tile kernel: SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))

    Uses tl.* primitives:
    - tl.neg -> TNEG (Vector unit)
    - tl.exp -> TEXP (Vector unit)
    - tl.add -> TADD (Vector unit)
    - tl.div -> TDIV (Vector unit)
    - tl.store -> TSTORE (Data movement)
    """
    # Compute -x
    neg_x = tl.neg(x_tile)

    # Compute exp(-x)
    exp_neg = tl.exp(neg_x)

    # Compute 1 + exp(-x)
    one = tl.constant(1.0, F32)
    denom = tl.add(one, exp_neg)

    # Compute x / (1 + exp(-x))
    result = tl.div(x_tile, denom)

    tl.store(x_tile, result)


@kernel
def mul_tile(
    a_tile: In[Tile[TILE_SIZE, TILE_SIZE, F16]],
    b_tile: In[Tile[TILE_SIZE, TILE_SIZE, F16]],
    c_tile: Out[Tile[TILE_SIZE, TILE_SIZE, F16]],
):
    """Element-wise multiply tile kernel: c = a * b

    Uses tl.* primitives:
    - tl.mul -> TMUL (Vector unit)
    - tl.store -> TSTORE (Data movement)
    """
    result = tl.mul(a_tile, b_tile)
    tl.store(c_tile, result)


@kernel
def add_tile(
    a_tile: In[Tile[TILE_SIZE, TILE_SIZE, F16]],
    b_tile: In[Tile[TILE_SIZE, TILE_SIZE, F16]],
    c_tile: Out[Tile[TILE_SIZE, TILE_SIZE, F16]],
):
    """Element-wise add tile kernel: c = a + b

    Uses tl.* primitives:
    - tl.add -> TADD (Vector unit)
    - tl.store -> TSTORE (Data movement)
    """
    result = tl.add(a_tile, b_tile)
    tl.store(c_tile, result)


# ============================================================
# Workload Definition
# ============================================================

@workload
def llama_layer_workload():
    """Single LLaMA transformer layer.

    Type: Workload[Dense[1] x Dense[16] x Dense[32], TransformerTasks, Hybrid]

    1. RMSNorm (pre-attention)
    2. QKV projection
    3. RoPE on Q, K
    4. Multi-head attention
    5. Output projection + residual
    6. RMSNorm (pre-MLP)
    7. SwiGLU MLP
    8. Down projection + residual
    """
    # === Pre-attention RMSNorm ===
    for b in P(batch):
        for s in P(seq_tiles):
            rmsnorm_tile[b, s](x_tile=hidden[b], y_tile=hidden[b], eps=1e-6)

    # === QKV Projections (parallel) ===
    for b in P(batch):
        for s in P(seq_tiles):
            linear_tile[b, s, "q"](x_tile=hidden[b], w_tile=wq, y_tile=q_proj[b])
            linear_tile[b, s, "k"](x_tile=hidden[b], w_tile=wk, y_tile=k_proj[b])
            linear_tile[b, s, "v"](x_tile=hidden[b], w_tile=wv, y_tile=v_proj[b])

    # === RoPE on Q and K ===
    for b in P(batch):
        for s in P(seq_tiles):
            rope_tile[b, s, "q"](x_tile=q_proj[b], pos_tile=positions)
            rope_tile[b, s, "k"](x_tile=k_proj[b], pos_tile=positions)

    # === Multi-head Attention ===
    for b in P(batch):
        for h in P(heads):
            attention_tile[b, h](
                q_tile=q_proj[b], k_tile=k_proj[b], v_tile=v_proj[b], o_tile=attn_out[b]
            )

    # === Output projection + residual ===
    for b in P(batch):
        for s in P(seq_tiles):
            linear_tile[b, s, "o"](x_tile=attn_out[b], w_tile=wo, y_tile=attn_out[b])
            add_tile[b, s](a_tile=hidden[b], b_tile=attn_out[b], c_tile=hidden[b])

    # === Pre-MLP RMSNorm ===
    for b in P(batch):
        for s in P(seq_tiles):
            rmsnorm_tile[b, s, "mlp"](x_tile=hidden[b], y_tile=hidden[b], eps=1e-6)

    # === SwiGLU MLP ===
    # Gate and up projections
    for b in P(batch):
        for s in P(seq_tiles):
            linear_tile[b, s, "gate"](x_tile=hidden[b], w_tile=w_gate, y_tile=gate_proj[b])
            linear_tile[b, s, "up"](x_tile=hidden[b], w_tile=w_up, y_tile=up_proj[b])

    # SiLU on gate
    for b in P(batch):
        for s in P(mlp_tiles):
            silu_tile[b, s](x_tile=gate_proj[b])

    # gate * up
    for b in P(batch):
        for s in P(mlp_tiles):
            mul_tile[b, s](a_tile=gate_proj[b], b_tile=up_proj[b], c_tile=gate_proj[b])

    # Down projection + residual
    for b in P(batch):
        for s in P(seq_tiles):
            linear_tile[b, s, "down"](x_tile=gate_proj[b], w_tile=w_down, y_tile=mlp_out[b])
            add_tile[b, s, "res"](a_tile=hidden[b], b_tile=mlp_out[b], c_tile=hidden[b])


# ============================================================
# Schedule Definition
# ============================================================

def create_schedule(wl):
    """Task graph schedule with hybrid dependencies."""
    return (wl
        .dispatch(DispatchPolicy.work_steal())
        .task_graph(
            # Hybrid: inferred tensor deps + structural deps
            deps=Deps.hybrid(
                infer=DepsMode.INFER_TENSOR_MAP_EXACT,
                explicit=True
            ),
            window=TaskWindow(8192, "tasks", WindowMode.STALL),
            pools=Pools.by_exec_unit(),  # Separate vector/cube queues
            ready=ReadyPolicy.work_steal(),
            start=StartPolicy.threshold(50),
            trace=TracePolicy.none(),
        ))


# ============================================================
# Main Entry Point
# ============================================================

def main():
    print("=" * 60)
    print("PTO-RT v9 LLaMA Transformer Layer Example")
    print("=" * 60)
    print()

    # Configuration
    print("Configuration (LLaMA-7B style):")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Sequence length: {SEQ_LEN}")
    print(f"  Hidden dimension: {HIDDEN_DIM}")
    print(f"  Number of heads: {NUM_HEADS}")
    print(f"  Head dimension: {HEAD_DIM}")
    print(f"  MLP dimension: {MLP_DIM}")
    print(f"  Tile size: {TILE_SIZE}")
    print()

    # Estimate task count
    n_rmsnorm = BATCH_SIZE * (SEQ_LEN // TILE_SIZE) * 2  # Pre-attn + pre-MLP
    n_linear = BATCH_SIZE * (SEQ_LEN // TILE_SIZE) * 7   # Q,K,V,O,gate,up,down
    n_rope = BATCH_SIZE * (SEQ_LEN // TILE_SIZE) * 2     # Q and K
    n_attn = BATCH_SIZE * NUM_HEADS
    n_silu = BATCH_SIZE * (MLP_DIM // TILE_SIZE)
    n_mul = BATCH_SIZE * (MLP_DIM // TILE_SIZE)
    n_add = BATCH_SIZE * (SEQ_LEN // TILE_SIZE) * 2      # Two residuals
    total = n_rmsnorm + n_linear + n_rope + n_attn + n_silu + n_mul + n_add
    print(f"Estimated tasks:")
    print(f"  RMSNorm: {n_rmsnorm}")
    print(f"  Linear: {n_linear}")
    print(f"  RoPE: {n_rope}")
    print(f"  Attention: {n_attn}")
    print(f"  SiLU: {n_silu}")
    print(f"  Mul: {n_mul}")
    print(f"  Add: {n_add}")
    print(f"  Total: {total}")
    print()

    # Kernel info
    print("Kernels (@kernel with tl.* primitives):")
    print(f"  rmsnorm_tile: tl.mul, tl.rowmean, tl.add, tl.rsqrt, tl.store")
    print(f"  linear_tile (cube): tl.matmul, tl.store")
    print(f"  rope_tile: tl.sin, tl.cos, tl.mul, tl.sub, tl.add, tl.store")
    print(f"  attention_tile: tl.matmul, tl.rsqrt, tl.rowmax, tl.exp, tl.rowsum, tl.div, tl.store")
    print(f"  silu_tile: tl.neg, tl.exp, tl.add, tl.div, tl.store")
    print(f"  mul_tile: tl.mul, tl.store")
    print(f"  add_tile: tl.add, tl.store")
    print()

    # Trace a kernel to show IR
    print("Kernel IR (traced from tl.* primitives):")
    ir = rmsnorm_tile.trace()
    print(f"  Parameters: {len(ir.params)}")
    print(f"  Operations: {len(ir.ops)}")
    for op in ir.ops:
        print(f"    {op}")
    print()

    # Build workload
    print("Building workload...")
    wl = llama_layer_workload()
    print(f"  Workload kind: {wl._kind}")
    print()

    # Apply schedule
    print("Applying task graph schedule...")
    scheduled = create_schedule(wl)
    cfg = scheduled._task_graph_config
    print(f"  Deps: {cfg.deps.mode.value}")
    print(f"  Pools: {cfg.pools._kind} (vector/cube separation)")
    print(f"  Ready: {cfg.ready._kind}")
    print(f"  Start threshold: {cfg.start._kwargs.get('n', 'N/A')}")
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

    # Show generated Ascend code for one kernel
    print("Generated Ascend code for rmsnorm_tile (for NPU target):")
    compiled = rmsnorm_tile.compile(target="ascend")
    print(compiled.code[:500] + "..." if len(compiled.code) > 500 else compiled.code)
    print()

    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
