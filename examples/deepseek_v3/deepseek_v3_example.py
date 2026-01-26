#!/usr/bin/env python3
"""
PTO-RT v9 End-to-End Example: DeepSeek-V3.2 Mixture of Experts (MoE)

This example demonstrates DeepSeek-V3.2-exp architecture features:
- Multi-head Latent Attention (MLA)
- DeepSeekMoE with auxiliary-loss-free load balancing
- Sparse expert routing

Architecture:
- @kernel defines kernels with tl.* primitives (Triton-style)
- Kernels are used directly in @workload with axis binding
- CPU simulation uses pto-isa's built-in CPU backend (-D__CPU_SIM)
- No separate stub or CPU implementation needed

Reference: DeepSeek-V3 Technical Report (2024)

Usage:
    python examples/deepseek_v3/deepseek_v3_example.py
"""

import sys
sys.path.insert(0, 'python')

from pto_wsp import (
    kernel, tl,
    In, Out, InOut, Tile, Scalar, Constexpr,
    workload, P,
    Dense, DenseDyn, Tensor, DType,
    DispatchPolicy, TimingPolicy,
    Deps, DepsMode, ReadyPolicy, StartPolicy, TracePolicy, Pools, TaskWindow, WindowMode,
)

# DType shortcuts
F16 = DType.F16
F32 = DType.F32
I32 = DType.I32

# ============================================================
# Configuration (DeepSeek-V3 style)
# ============================================================

BATCH_SIZE = 1
SEQ_LEN = 256
HIDDEN_DIM = 4096

# MLA (Multi-head Latent Attention)
NUM_HEADS = 32
HEAD_DIM = 128
KV_LORA_RANK = 512  # Low-rank compression for KV

# MoE Configuration
NUM_EXPERTS = 256  # Total experts
TOP_K = 8          # Experts per token
EXPERT_DIM = 1536  # FFN dimension per expert
SHARED_EXPERTS = 2 # Number of shared experts

TILE_SIZE = 32

# ============================================================
# Axis Types
# ============================================================

batch = Dense[BATCH_SIZE]()
seq = Dense[SEQ_LEN]()
heads = Dense[NUM_HEADS]()
seq_tiles = Dense[SEQ_LEN // TILE_SIZE]()

# Expert axes
experts = Dense[NUM_EXPERTS]()
shared_exp = Dense[SHARED_EXPERTS]()

# Sparse routing (dynamic per batch)
# Represents which expert handles which tokens
# This would be Sparse axis in real implementation
expert_tokens = DenseDyn(BATCH_SIZE * SEQ_LEN * TOP_K)

# ============================================================
# Tensor Declarations
# ============================================================

# Input hidden states
hidden = Tensor(data=None, shape=(BATCH_SIZE, SEQ_LEN, HIDDEN_DIM), dtype=F16)

# MLA tensors
# Compressed KV latent
kv_compressed = Tensor(data=None, shape=(BATCH_SIZE, SEQ_LEN, KV_LORA_RANK), dtype=F16)
q_rope = Tensor(data=None, shape=(BATCH_SIZE, SEQ_LEN, NUM_HEADS * HEAD_DIM), dtype=F16)
k_rope = Tensor(data=None, shape=(BATCH_SIZE, SEQ_LEN, NUM_HEADS * HEAD_DIM), dtype=F16)
attn_out = Tensor(data=None, shape=(BATCH_SIZE, SEQ_LEN, HIDDEN_DIM), dtype=F16)

# MoE tensors
router_logits = Tensor(data=None, shape=(BATCH_SIZE, SEQ_LEN, NUM_EXPERTS), dtype=F32)
expert_weights = Tensor(data=None, shape=(BATCH_SIZE, SEQ_LEN, TOP_K), dtype=F32)
expert_indices = Tensor(data=None, shape=(BATCH_SIZE, SEQ_LEN, TOP_K), dtype=I32)

# Expert outputs (sparse)
expert_out = Tensor(data=None, shape=(BATCH_SIZE, SEQ_LEN, HIDDEN_DIM), dtype=F16)
shared_out = Tensor(data=None, shape=(BATCH_SIZE, SEQ_LEN, HIDDEN_DIM), dtype=F16)
moe_out = Tensor(data=None, shape=(BATCH_SIZE, SEQ_LEN, HIDDEN_DIM), dtype=F16)

# Weight tensors for kernels
w_kv_compress = Tensor(data=None, shape=(HIDDEN_DIM, KV_LORA_RANK), dtype=F16)
w_router = Tensor(data=None, shape=(HIDDEN_DIM, NUM_EXPERTS), dtype=F16)
w_gate = Tensor(data=None, shape=(HIDDEN_DIM, EXPERT_DIM), dtype=F16)
w_up = Tensor(data=None, shape=(HIDDEN_DIM, EXPERT_DIM), dtype=F16)
w_down = Tensor(data=None, shape=(EXPERT_DIM, HIDDEN_DIM), dtype=F16)

# ============================================================
# Kernel Definitions (unified @kernel with tl.* primitives)
# ============================================================

@kernel
def rmsnorm_kernel(
    x_tile: In[Tile[TILE_SIZE, HIDDEN_DIM, F16]],
    y_tile: Out[Tile[TILE_SIZE, HIDDEN_DIM, F16]],
):
    """RMSNorm kernel using tl.* primitives.

    Uses tl.* primitives that map to PTO-ISA tile operations.
    When compiled with -D__CPU_SIM, these execute on CPU.
    When compiled for NPU, these generate Ascend kernel code.
    """
    # Compute squared values
    x_sq = tl.mul(x_tile, x_tile)

    # Mean of squared values (along hidden dim)
    mean_sq = tl.rowmean(x_sq)

    # Add epsilon for numerical stability
    eps = tl.constant(1e-6, F32)
    mean_sq_eps = tl.add(mean_sq, eps)

    # Compute reciprocal sqrt
    rms_inv = tl.rsqrt(mean_sq_eps)

    # Normalize
    result = tl.mul(x_tile, rms_inv)
    tl.store(y_tile, result)


@kernel
def kv_compression_kernel(
    h_tile: In[Tile[TILE_SIZE, HIDDEN_DIM, F16]],
    w_compress: In[Tile[HIDDEN_DIM, KV_LORA_RANK, F16]],
    kv_tile: Out[Tile[TILE_SIZE, KV_LORA_RANK, F16]],
):
    """Compress hidden states to low-rank KV representation.

    KV compression reduces memory bandwidth for attention.
    """
    result = tl.matmul(h_tile, w_compress)
    tl.store(kv_tile, result)


@kernel
def mla_attention_kernel(
    q_tile: In[Tile[TILE_SIZE, HEAD_DIM, F16]],
    kv_tile: In[Tile[TILE_SIZE, KV_LORA_RANK, F16]],
    o_tile: Out[Tile[TILE_SIZE, HEAD_DIM, F16]],
):
    """Multi-head Latent Attention with absorbed weights using tl.* primitives.

    MLA compresses KV to a low-rank latent space.
    Simplified version: compute attention with compressed KV.
    In real implementation, would decompress and apply absorbed weights.
    """
    qk = tl.matmul(q_tile, kv_tile)

    scale = tl.rsqrt(tl.constant(HEAD_DIM, F32))
    qk_scaled = tl.mul(qk, scale)

    max_score = tl.rowmax(qk_scaled)
    qk_shifted = tl.sub(qk_scaled, max_score)
    exp_scores = tl.exp(qk_shifted)
    sum_exp = tl.rowsum(exp_scores)
    attn_weights = tl.div(exp_scores, sum_exp)

    result = tl.matmul(attn_weights, kv_tile)
    tl.store(o_tile, result)


@kernel
def router_kernel(
    h_tile: In[Tile[TILE_SIZE, HIDDEN_DIM, F16]],
    w_route: In[Tile[HIDDEN_DIM, NUM_EXPERTS, F16]],
    logits_tile: Out[Tile[TILE_SIZE, NUM_EXPERTS, F32]],
):
    """Router kernel for expert selection using tl.* primitives."""
    result = tl.matmul(h_tile, w_route)
    tl.store(logits_tile, result)


@kernel
def topk_kernel(
    logits_tile: In[Tile[TILE_SIZE, NUM_EXPERTS, F32]],
    weights_tile: Out[Tile[TILE_SIZE, TOP_K, F32]],
    indices_tile: Out[Tile[TILE_SIZE, TOP_K, I32]],
):
    """Select top-K experts per token.

    Computes softmax over top-K logits for expert weighting.
    """
    # Find top-K indices and values
    topk_vals, topk_idx = tl.topk(logits_tile, TOP_K)

    # Softmax over top-K values for weights
    max_val = tl.rowmax(topk_vals)
    shifted = tl.sub(topk_vals, max_val)
    exp_vals = tl.exp(shifted)
    sum_exp = tl.rowsum(exp_vals)
    weights = tl.div(exp_vals, sum_exp)

    tl.store(weights_tile, weights)
    tl.store(indices_tile, topk_idx)


@kernel
def expert_ffn_kernel(
    x_tile: In[Tile[TILE_SIZE, HIDDEN_DIM, F16]],
    w_gate: In[Tile[HIDDEN_DIM, EXPERT_DIM, F16]],
    w_up: In[Tile[HIDDEN_DIM, EXPERT_DIM, F16]],
    w_down: In[Tile[EXPERT_DIM, HIDDEN_DIM, F16]],
    y_tile: Out[Tile[TILE_SIZE, HIDDEN_DIM, F16]],
):
    """Expert FFN with SwiGLU using tl.* primitives.

    SwiGLU: y = (x @ W_gate * SiLU(x @ W_up)) @ W_down
    """
    # Gate projection
    gate = tl.matmul(x_tile, w_gate)

    # Up projection
    up = tl.matmul(x_tile, w_up)

    # SiLU on up: up * sigmoid(up)
    neg_up = tl.neg(up)
    exp_neg = tl.exp(neg_up)
    one = tl.constant(1.0, F32)
    denom = tl.add(one, exp_neg)
    silu_up = tl.div(up, denom)

    # Gate * SiLU(up)
    combined = tl.mul(gate, silu_up)

    # Down projection
    result = tl.matmul(combined, w_down)

    tl.store(y_tile, result)


@kernel
def shared_expert_kernel(
    x_tile: In[Tile[TILE_SIZE, HIDDEN_DIM, F16]],
    w_gate: In[Tile[HIDDEN_DIM, EXPERT_DIM, F16]],
    w_up: In[Tile[HIDDEN_DIM, EXPERT_DIM, F16]],
    w_down: In[Tile[EXPERT_DIM, HIDDEN_DIM, F16]],
    y_tile: Out[Tile[TILE_SIZE, HIDDEN_DIM, F16]],
):
    """Shared expert FFN (always activated).

    Same SwiGLU structure as expert_ffn_kernel.
    """
    gate = tl.matmul(x_tile, w_gate)
    up = tl.matmul(x_tile, w_up)

    neg_up = tl.neg(up)
    exp_neg = tl.exp(neg_up)
    one = tl.constant(1.0, F32)
    denom = tl.add(one, exp_neg)
    silu_up = tl.div(up, denom)

    combined = tl.mul(gate, silu_up)
    result = tl.matmul(combined, w_down)

    tl.store(y_tile, result)


@kernel
def combine_expert_outputs_kernel(
    expert_tile: In[Tile[TILE_SIZE, HIDDEN_DIM, F16]],
    weights_tile: In[Tile[TILE_SIZE, TOP_K, F32]],
    combined_tile: InOut[Tile[TILE_SIZE, HIDDEN_DIM, F16]],
):
    """Combine weighted expert outputs.

    Accumulates weighted expert output into combined buffer.
    """
    # Scale expert output by weight (simplified: uses first weight)
    weight = tl.broadcast(weights_tile, (TILE_SIZE, HIDDEN_DIM))
    weighted = tl.mul(expert_tile, weight)

    # Accumulate
    result = tl.add(combined_tile, weighted)
    tl.store(combined_tile, result)


@kernel
def add_kernel(
    a_tile: In[Tile[TILE_SIZE, HIDDEN_DIM, F16]],
    b_tile: In[Tile[TILE_SIZE, HIDDEN_DIM, F16]],
    c_tile: Out[Tile[TILE_SIZE, HIDDEN_DIM, F16]],
):
    """Element-wise addition for residual connections."""
    result = tl.add(a_tile, b_tile)
    tl.store(c_tile, result)


# ============================================================
# Workload Definition
# ============================================================

@workload
def deepseek_v3_layer_workload():
    """DeepSeek-V3 transformer layer with MLA and MoE.

    Structure:
    1. Pre-norm
    2. MLA (Multi-head Latent Attention)
    3. Post-attention residual
    4. Pre-MoE norm
    5. Router (expert selection)
    6. Expert computation (sparse)
    7. Shared experts (dense)
    8. Combine outputs
    9. Post-MoE residual

    Kernels use tl.* primitives directly.
    CPU simulation via pto-isa backend, NPU via Ascend codegen.
    """

    # === Pre-attention RMSNorm ===
    for b in P(batch):
        for s in P(seq_tiles):
            rmsnorm_kernel[b, s](x_tile=hidden[b], y_tile=hidden[b])

    # === MLA: KV Compression ===
    # Compress hidden states to low-rank KV
    for b in P(batch):
        for s in P(seq_tiles):
            kv_compression_kernel[b, s](
                h_tile=hidden[b],
                w_compress=w_kv_compress,
                kv_tile=kv_compressed[b]
            )

    # === MLA: Attention with latent KV ===
    for b in P(batch):
        for h in P(heads):
            mla_attention_kernel[b, h](
                q_tile=q_rope[b],
                kv_tile=kv_compressed[b],
                o_tile=attn_out[b]
            )

    # === Post-attention residual ===
    for b in P(batch):
        for s in P(seq_tiles):
            add_kernel[b, s, "attn"](a_tile=hidden[b], b_tile=attn_out[b], c_tile=hidden[b])

    # === Pre-MoE RMSNorm ===
    for b in P(batch):
        for s in P(seq_tiles):
            rmsnorm_kernel[b, s, "moe"](x_tile=hidden[b], y_tile=hidden[b])

    # === Router: Expert Selection ===
    for b in P(batch):
        for s in P(seq_tiles):
            router_kernel[b, s](
                h_tile=hidden[b],
                w_route=w_router,
                logits_tile=router_logits[b]
            )
            topk_kernel[b, s](
                logits_tile=router_logits[b],
                weights_tile=expert_weights[b],
                indices_tile=expert_indices[b]
            )

    # === Sparse Expert Computation ===
    # In practice, this would use sparse iteration based on routing
    # Here we show the conceptual structure
    for b in P(batch):
        # Iterate over selected expert assignments (sparse)
        for e in P.sel(expert_tokens):
            expert_ffn_kernel[b, e](
                x_tile=hidden[b],
                w_gate=w_gate,
                w_up=w_up,
                w_down=w_down,
                y_tile=expert_out[b]
            )

    # === Shared Experts (always activated) ===
    for b in P(batch):
        for s in P(seq_tiles):
            for e in P(shared_exp):
                shared_expert_kernel[b, s, e](
                    x_tile=hidden[b],
                    w_gate=w_gate,
                    w_up=w_up,
                    w_down=w_down,
                    y_tile=shared_out[b]
                )

    # === Combine Expert Outputs ===
    for b in P(batch):
        for s in P(seq_tiles):
            # Weighted sum of selected expert outputs
            combine_expert_outputs_kernel[b, s](
                expert_tile=expert_out[b],
                weights_tile=expert_weights[b],
                combined_tile=moe_out[b]
            )
            # Add shared expert outputs
            add_kernel[b, s, "shared"](a_tile=moe_out[b], b_tile=shared_out[b], c_tile=moe_out[b])

    # === Post-MoE residual ===
    for b in P(batch):
        for s in P(seq_tiles):
            add_kernel[b, s, "moe"](a_tile=hidden[b], b_tile=moe_out[b], c_tile=hidden[b])


# ============================================================
# Schedule Definition
# ============================================================

def create_schedule(wl):
    """Task graph schedule optimized for MoE sparsity."""
    return (wl
        .dispatch(DispatchPolicy.work_steal())
        .task_graph(
            # Hybrid for complex dependencies
            deps=Deps.hybrid(
                infer=DepsMode.INFER_TENSOR_MAP_EXACT,
                explicit=True
            ),
            # Large window for many experts
            window=TaskWindow(16384, "tasks", WindowMode.STALL),
            # Separate queues for vector (routing) and cube (FFN)
            pools=Pools.by_exec_unit(),
            ready=ReadyPolicy.work_steal(),
            # Start early for pipelined execution
            start=StartPolicy.threshold(100),
            # Enable tracing for profiling
            trace=TracePolicy.cycles(),
        ))


# ============================================================
# Extended Schedule with Batch Dependencies (R5)
# ============================================================

def create_advanced_schedule(wl):
    """Advanced schedule with extended primitives."""
    return (wl
        .dispatch_threshold(
            thresholds=[256, 1024, 4096],
            policies={
                256: DispatchPolicy.round_robin(1),
                1024: DispatchPolicy.round_robin(4),
                4096: DispatchPolicy.work_steal(),
            }
        )
        .task_graph(
            deps=Deps.hybrid(),
            window=TaskWindow(16384, "tasks", WindowMode.STALL),
            pools=Pools.by_exec_unit(),
            ready=ReadyPolicy.work_steal(),
        )
        .batch_deps(128, range_compression=True)
        .pipeline_depth(3, scope="per_pool"))


# ============================================================
# Main Entry Point
# ============================================================

def main():
    print("=" * 60)
    print("PTO-RT v9 DeepSeek-V3.2 MoE Example")
    print("=" * 60)
    print()

    # Configuration
    print("Configuration (DeepSeek-V3 style):")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Sequence length: {SEQ_LEN}")
    print(f"  Hidden dimension: {HIDDEN_DIM}")
    print()
    print("MLA Configuration:")
    print(f"  Num heads: {NUM_HEADS}")
    print(f"  Head dim: {HEAD_DIM}")
    print(f"  KV LoRA rank: {KV_LORA_RANK}")
    print()
    print("MoE Configuration:")
    print(f"  Total experts: {NUM_EXPERTS}")
    print(f"  Active experts (top-K): {TOP_K}")
    print(f"  Shared experts: {SHARED_EXPERTS}")
    print(f"  Expert FFN dim: {EXPERT_DIM}")
    print()

    # Task estimation
    n_norm = BATCH_SIZE * (SEQ_LEN // TILE_SIZE) * 2
    n_mla = BATCH_SIZE * NUM_HEADS + BATCH_SIZE * (SEQ_LEN // TILE_SIZE)
    n_router = BATCH_SIZE * (SEQ_LEN // TILE_SIZE) * 2
    n_experts = BATCH_SIZE * SEQ_LEN * TOP_K  # Sparse
    n_shared = BATCH_SIZE * (SEQ_LEN // TILE_SIZE) * SHARED_EXPERTS
    n_combine = BATCH_SIZE * (SEQ_LEN // TILE_SIZE) * 2
    n_residual = BATCH_SIZE * (SEQ_LEN // TILE_SIZE) * 2
    print("Task estimation:")
    print(f"  RMSNorm: {n_norm}")
    print(f"  MLA: {n_mla}")
    print(f"  Router: {n_router}")
    print(f"  Expert FFN (sparse): {n_experts}")
    print(f"  Shared experts: {n_shared}")
    print(f"  Combine + residual: {n_combine + n_residual}")
    print()

    # Kernel info
    print("Kernels (@kernel with tl.* primitives):")
    print(f"  rmsnorm_kernel: tl.mul, tl.rowmean, tl.rsqrt")
    print(f"  kv_compression_kernel: tl.matmul")
    print(f"  mla_attention_kernel: tl.matmul, tl.rsqrt, tl.rowmax, tl.exp, tl.rowsum, tl.div")
    print(f"  router_kernel: tl.matmul")
    print(f"  topk_kernel: tl.topk, tl.rowmax, tl.exp, tl.rowsum, tl.div")
    print(f"  expert_ffn_kernel (cube): tl.matmul, tl.neg, tl.exp, tl.div, tl.mul")
    print(f"  shared_expert_kernel (cube): tl.matmul, tl.neg, tl.exp, tl.div, tl.mul")
    print(f"  combine_expert_outputs_kernel: tl.broadcast, tl.mul, tl.add")
    print(f"  add_kernel: tl.add")
    print()

    # Build workload
    print("Building workload...")
    wl = deepseek_v3_layer_workload()
    print(f"  Workload kind: {wl._kind}")
    print()

    # Basic schedule
    print("Basic task graph schedule:")
    scheduled = create_schedule(wl)
    cfg = scheduled._task_graph_config
    print(f"  Deps: {cfg.deps.mode.value}")
    print(f"  Window: {cfg.window.size}")
    print(f"  Pools: {cfg.pools._kind}")
    print(f"  Trace: {cfg.trace._kind}")
    print()

    # Advanced schedule
    print("Advanced schedule (with R5 extensions):")
    advanced = create_advanced_schedule(wl)
    print(f"  Dispatch threshold: {advanced._dispatch_threshold.thresholds}")
    print(f"  Batch deps: threshold={advanced._batch_deps.threshold}")
    print(f"  Pipeline depth: {advanced._pipeline_depth.depth}")
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

    # Show kernel trace info
    print("Kernel IR (example: expert_ffn_kernel):")
    ir = expert_ffn_kernel.trace()
    print(f"  Parameters: {len(ir.params)}")
    print(f"  Operations: {len(ir.ops)}")
    for op in ir.ops[:5]:  # Show first 5 ops
        print(f"    {op}")
    if len(ir.ops) > 5:
        print(f"    ... ({len(ir.ops) - 5} more operations)")
    print()

    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
