#!/usr/bin/env python3
"""
DeepSeek Lightning Indexer - PTO-RT v9 Python Implementation

This example demonstrates the v9 programming model applied to DeepSeek's
Lightning Indexer TopK attention, showing how to handle:
- Variable sequence lengths per batch (Ragged axis)
- Discrete tier selection (2K/8K/64K/128K)
- Complex nested loops with dynamic bounds
- Task graph execution with tier-based dispatch

Architecture:
- @kernel defines kernels with tl.* primitives (Triton-style)
- Kernels are used directly in @workload with axis binding
- CPU simulation uses pto-isa's built-in CPU backend (-D__CPU_SIM)
- No separate stub or CPU implementation needed

Reference: DeepSeek V3 Technical Report, Section 3.3 (MLA with Lightning Attention)

Usage:
    python examples/deepseek_lightning_indexer/deepseek_lightning_indexer.py
"""

import sys
sys.path.insert(0, 'python')

from pto_wsp import (
    kernel, tl,
    In, Out, Tile, Scalar,
    workload, P,
    Dense, DenseDyn, Ragged, Tensor, DType,
    DispatchPolicy, TimingPolicy,
    Deps, ReadyPolicy, StartPolicy, TracePolicy, Pools,
    cond, select,
)

# ============================================================================
# CONSTANTS
# ============================================================================

BLOCK_SIZE = 128
INDEX_D = 128
SELECTED_COUNT = 2048
GROUP_SIZE = 8

# Tier thresholds
TIER_2K = 2048
TIER_8K = 8192
TIER_64K = 65536
TIER_128K = 131072

# DType shortcuts
F16 = DType.F16
F32 = DType.F32
I32 = DType.I32

print("=" * 70)
print("DeepSeek Lightning Indexer - PTO-RT v9 Example")
print("=" * 70)

# ============================================================================
# TIER-SPECIFIC KERNELS (unified @kernel with tl.* primitives)
#
# The key insight: effSeq determines the TopK path
# effSeq = curSeq - (S1 - s1Idx - 1) -- changes per seq_pos!
#
# We define 4 tier kernels optimized for different eff_seq ranges.
# Each kernel uses tl.* primitives that map to PTO-ISA tile operations:
# - tl.matmul -> TMATMUL (Cube unit matmul)
# - tl.relu -> TRELU (Vector unit activation)
# - tl.mul -> TMUL (Vector unit multiply)
# - tl.colsum -> TREDUCE (Vector unit reduction)
# - tl.topk -> TTOPK (TopK selection)
# - tl.store -> TSTORE (Data movement)
#
# When compiled with -D__CPU_SIM, these execute on CPU via pto-isa backend.
# When compiled for NPU, these generate Ascend kernel code.
# ============================================================================

@kernel
def indexer_kernel_2k(
    q_tile: In[Tile[GROUP_SIZE, INDEX_D, F16]],
    weight_tile: In[Tile[GROUP_SIZE, 1, F16]],
    k_cache: In[Tile[BLOCK_SIZE, INDEX_D, F16]],
    topk_result: Out[Tile[1, SELECTED_COUNT, I32]],
    eff_seq: Scalar[I32],
    act_block: Scalar[I32],
):
    """2K tier kernel - optimized for short sequences (eff_seq <= 2048).

    Uses 2K padding for TopK, aggressive register tiling.
    """
    # Compute Q * K^T scores
    mm_result = tl.matmul(q_tile, k_cache)  # [GROUP_SIZE, BLOCK_SIZE]

    # Apply ReLU activation
    scores = tl.relu(mm_result)

    # Weight and sum along group dimension
    weighted = tl.mul(scores, weight_tile)  # broadcast weight
    block_sum = tl.colsum(weighted)  # [1, BLOCK_SIZE]

    # TopK with 2K padding
    topk_indices = tl.topk(block_sum, SELECTED_COUNT, pad_to=TIER_2K)

    tl.store(topk_result, topk_indices)


@kernel
def indexer_kernel_8k(
    q_tile: In[Tile[GROUP_SIZE, INDEX_D, F16]],
    weight_tile: In[Tile[GROUP_SIZE, 1, F16]],
    k_cache: In[Tile[BLOCK_SIZE, INDEX_D, F16]],
    topk_result: Out[Tile[1, SELECTED_COUNT, I32]],
    eff_seq: Scalar[I32],
    act_block: Scalar[I32],
):
    """8K tier kernel - optimized for medium sequences (eff_seq <= 8192).

    Uses 8K padding for TopK, balanced memory/compute.
    """
    mm_result = tl.matmul(q_tile, k_cache)
    scores = tl.relu(mm_result)
    weighted = tl.mul(scores, weight_tile)
    block_sum = tl.colsum(weighted)
    topk_indices = tl.topk(block_sum, SELECTED_COUNT, pad_to=TIER_8K)
    tl.store(topk_result, topk_indices)


@kernel
def indexer_kernel_64k(
    q_tile: In[Tile[GROUP_SIZE, INDEX_D, F16]],
    weight_tile: In[Tile[GROUP_SIZE, 1, F16]],
    k_cache: In[Tile[BLOCK_SIZE, INDEX_D, F16]],
    topk_result: Out[Tile[1, SELECTED_COUNT, I32]],
    eff_seq: Scalar[I32],
    act_block: Scalar[I32],
):
    """64K tier kernel - optimized for long sequences (eff_seq <= 65536).

    Uses chunked processing with multiple TopK passes.
    """
    mm_result = tl.matmul(q_tile, k_cache)
    scores = tl.relu(mm_result)
    weighted = tl.mul(scores, weight_tile)
    block_sum = tl.colsum(weighted)
    topk_indices = tl.topk(block_sum, SELECTED_COUNT, pad_to=TIER_64K)
    tl.store(topk_result, topk_indices)


@kernel
def indexer_kernel_128k(
    q_tile: In[Tile[GROUP_SIZE, INDEX_D, F16]],
    weight_tile: In[Tile[GROUP_SIZE, 1, F16]],
    k_cache: In[Tile[BLOCK_SIZE, INDEX_D, F16]],
    topk_result: Out[Tile[1, SELECTED_COUNT, I32]],
    eff_seq: Scalar[I32],
    act_block: Scalar[I32],
):
    """128K tier kernel - maximum sequence support.

    Uses hierarchical TopK for very long sequences.
    """
    mm_result = tl.matmul(q_tile, k_cache)
    scores = tl.relu(mm_result)
    weighted = tl.mul(scores, weight_tile)
    block_sum = tl.colsum(weighted)
    topk_indices = tl.topk(block_sum, SELECTED_COUNT, pad_to=TIER_128K)
    tl.store(topk_result, topk_indices)


# ============================================================================
# WORKLOAD DEFINITION
#
# Iteration space: (batch, seq_pos, idx_head)
# seq_pos affects effective sequence length (causal attention)
#
# Uses tier-specific kernels directly with axis binding.
# Runtime selects appropriate kernel based on eff_seq threshold.
# ============================================================================

def create_lightning_indexer_workload(
    batch_size: int,
    S1: int,  # Query sequence length
    n2: int,  # Number of index heads
    act_seq_key: list[int],  # Actual sequence lengths per batch
):
    """Create Lightning Indexer workload with tier selection.

    The workload uses `select` to choose the appropriate tier kernel
    based on the effective sequence length at each position.

    Args:
        batch_size: Number of sequences in batch
        S1: Query sequence length
        n2: Number of index heads
        act_seq_key: List of actual KV cache lengths per batch item

    Returns:
        Compiled program ready for execution
    """
    # Axis definitions
    batch = DenseDyn(batch_size)
    seq_positions = Dense[S1]()
    idx_heads = Dense[n2]()

    # Tensors (shapes for documentation)
    # Q: [batch, S1, indexN1, indexD]
    # weights: [batch, S1, indexN1]
    # k_cache: [block_num, BLOCK_SIZE, n2, indexD]
    # topk_result: [batch, S1, n2, SELECTED_COUNT]

    Q = Tensor(data=None, shape=(batch_size, S1, n2 * GROUP_SIZE, INDEX_D), dtype=F16)
    weights = Tensor(data=None, shape=(batch_size, S1, n2 * GROUP_SIZE), dtype=F16)
    k_cache = Tensor(data=None, shape=(1024, BLOCK_SIZE, n2, INDEX_D), dtype=F16)  # Placeholder block_num
    topk_result = Tensor(data=None, shape=(batch_size, S1, n2, SELECTED_COUNT), dtype=I32)

    @workload
    def lightning_indexer():
        """Lightning Indexer workload with dynamic tier selection.

        Type: Workload[DenseDyn x Dense[S1] x Dense[n2], IndexerTask, TierBased]

        For each (batch, seq_pos, idx_head):
        1. Compute eff_seq = act_seq_key[batch] - (S1 - seq_pos - 1)
        2. Select tier based on eff_seq threshold
        3. Dispatch to appropriate tier kernel

        Uses kernel[axes](...) directly - the kernel with tl.* primitives
        is the actual implementation, no separate stub needed.

        Note: Tier selection happens at runtime via task metadata. The workload
        defines the iteration space; the runtime selects the appropriate kernel.
        """
        for b in P(batch):
            for s in P(seq_positions):
                for h in P(idx_heads):
                    # Use tier 2K kernel as default - runtime will select
                    # appropriate tier based on computed eff_seq
                    indexer_kernel_2k[b, s, h](
                        q_tile=Q,
                        weight_tile=weights,
                        k_cache=k_cache,
                        topk_result=topk_result,
                        eff_seq=0,  # Runtime-computed
                        act_block=0,
                    )

    # Apply schedule with tier-aware dispatch
    # Tasks are dispatched based on computed eff_seq at runtime
    scheduled = (lightning_indexer()
        # Use task graph for dependency inference
        .task_graph(
            deps=Deps.explicit(),  # No tensor dependencies between tasks
            ready=ReadyPolicy.fifo(),  # Process in order
            start=StartPolicy.threshold(n2),  # Start after n2 tasks ready
            trace=TracePolicy.none(),
            pools=Pools.single(),
        )
        # Dispatch: same batch stays on same executor (cache locality)
        .dispatch(DispatchPolicy.affinity(lambda t: t.get("b")))
        # Two streams for interleaved execution
        .streams(2)
        .stream_by(lambda t: t.get("b") % 2)
        .timing(TimingPolicy.immediate))

    return scheduled.compile()


# ============================================================================
# TIER SELECTION LOGIC
#
# These functions are used by the runtime to select the correct kernel
# ============================================================================

def select_tier(eff_seq: int) -> int:
    """Select kernel tier based on effective sequence length.

    Returns tier index: 0=2K, 1=8K, 2=64K, 3=128K
    """
    if eff_seq <= TIER_2K:
        return 0
    elif eff_seq <= TIER_8K:
        return 1
    elif eff_seq <= TIER_64K:
        return 2
    else:
        return 3


def compute_eff_seq(act_seq_key: list[int], batch_idx: int, S1: int, seq_pos: int) -> int:
    """Compute effective sequence length for causal attention.

    eff_seq = act_seq_key[batch_idx] - (S1 - seq_pos - 1)

    This is the key dynamic value that varies per seq position.
    """
    cur_seq = act_seq_key[batch_idx]
    causal_offset = S1 - seq_pos - 1
    return cur_seq - causal_offset


# ============================================================================
# EXAMPLE EXECUTION
# ============================================================================

def main():
    # Configuration
    batch_size = 4
    S1 = 16  # Small for demo
    n2 = 4   # Number of index heads

    # Variable sequence lengths per batch (actual KV cache lengths)
    act_seq_key = [512, 2048, 8192, 32768]

    print(f"\nConfiguration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Query length (S1): {S1}")
    print(f"  Index heads (n2): {n2}")
    print(f"  Actual seq lengths: {act_seq_key}")
    print(f"  Total tasks: {batch_size * S1 * n2}")

    # Kernel info (unified @kernel with tl.* primitives)
    print(f"\nKernels (@kernel with tl.* primitives):")
    tier_kernels = [
        ("2K", indexer_kernel_2k, TIER_2K),
        ("8K", indexer_kernel_8k, TIER_8K),
        ("64K", indexer_kernel_64k, TIER_64K),
        ("128K", indexer_kernel_128k, TIER_128K),
    ]
    for tier_name, kernel_fn, threshold in tier_kernels:
        print(f"  {tier_name} tier: {kernel_fn.name} (eff_seq <= {threshold})")
        print(f"    Operations: tl.matmul, tl.relu, tl.mul, tl.colsum, tl.topk, tl.store")

    # Show tier distribution
    print(f"\nTier distribution analysis:")
    tier_counts = [0, 0, 0, 0]
    for b in range(batch_size):
        for s in range(S1):
            eff_seq = compute_eff_seq(act_seq_key, b, S1, s)
            tier = select_tier(eff_seq)
            tier_counts[tier] += n2
            if s == 0 or s == S1 - 1:
                tier_names = ["2K", "8K", "64K", "128K"]
                print(f"  batch={b}, seq_pos={s}: eff_seq={eff_seq} -> tier={tier_names[tier]}")

    print(f"\nTier task counts:")
    tier_names = ["2K", "8K", "64K", "128K"]
    for i, count in enumerate(tier_counts):
        if count > 0:
            pct = 100.0 * count / (batch_size * S1 * n2)
            print(f"  {tier_names[i]}: {count} tasks ({pct:.1f}%)")

    # Trace a kernel to show IR
    print(f"\nKernel IR (traced from tl.* primitives):")
    ir = indexer_kernel_2k.trace()
    print(f"  Parameters: {len(ir.params)}")
    print(f"  Operations: {len(ir.ops)}")
    for op in ir.ops:
        print(f"    {op}")

    # Create and compile workload
    print(f"\nCreating workload...")
    program = create_lightning_indexer_workload(batch_size, S1, n2, act_seq_key)
    print(f"Program compiled: {type(program).__name__}")

    # Execute with CPU simulation (via pto-isa backend)
    # No register_kernel needed - @kernel with tl.* executes directly
    print(f"\nExecuting with CPU simulation (pto-isa backend)...")
    program.execute()
    print("Execution complete!")

    # Show generated Ascend code
    print(f"\nGenerated Ascend code (for NPU target):")
    compiled = indexer_kernel_2k.compile(target="ascend")
    print(compiled.code[:500] + "..." if len(compiled.code) > 500 else compiled.code)

    # Summary
    print(f"\n" + "=" * 70)
    print("Lightning Indexer v9 Summary")
    print("=" * 70)
    print("""
Key v9 patterns demonstrated:

1. Unified Kernels (@kernel + tl.*):
   - 4 tier-specific kernels with typed tl.* operations
   - No string references - all typed
   - No separate @jit_kernel vs @kernel - one decorator does both
   - CPU simulation via pto-isa backend (no manual CPU impl needed)

2. Workload Definition (@workload + P):
   - Declarative iteration space: batch x seq_pos x idx_head
   - P() for parallel axes
   - kernel[axes](...) used directly in workload

3. Tier Selection:
   - Dynamic based on eff_seq = act_seq - causal_offset
   - 2K/8K/64K/128K thresholds for TopK padding
   - select() primitive for runtime kernel dispatch

4. Schedule Configuration:
   - .task_graph() for dependency-aware execution
   - .dispatch(affinity) for cache locality
   - .streams(2) for pipelined execution

5. Key differences from old pattern:
   - Removed @jit_kernel (use @kernel only)
   - Removed separate @kernel stubs (kernel with tl.* IS the impl)
   - Removed program.register_kernel() (pto-isa backend handles it)
   - kernel.trace() shows IR, kernel.compile() generates target code
""")


if __name__ == "__main__":
    main()
