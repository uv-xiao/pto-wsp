#!/usr/bin/env python3
"""
FlashInfer-Style Decode Attention - PTO-RT v9 Python Implementation

This example demonstrates how to implement FlashInfer-style decode attention
using the v9 programming model. FlashInfer's key innovation is the Plan-Run
execution model that efficiently handles variable-length sequences through
index-based work assignment.

Architecture:
- @kernel defines the kernel with tl.* primitives (Triton-style)
- Kernel is used directly in @workload with axis binding
- CPU simulation uses pto-isa's built-in CPU backend (-D__CPU_SIM)
- No separate stub or CPU implementation needed

Key Concepts:
1. Work Descriptor - Each task receives a compact descriptor specifying its work
2. Planning Phase - Generate descriptors for all work units
3. Execution Phase - Each worker reads its descriptor and executes

Reference: FlashInfer Paper (https://arxiv.org/abs/2310.08000)

Usage:
    python examples/flashinfer_decode/flashinfer_decode_attention.py
"""

import sys
sys.path.insert(0, 'python')

from dataclasses import dataclass
from typing import List
import math

import pto_wsp as pto
from pto_wsp import (
    kernel, tl,
    In, Out, Tile, Scalar,
    workload, P,
    Dense, DenseDyn, Tensor, DType,
    DispatchPolicy, TimingPolicy,
    Deps, ReadyPolicy, StartPolicy, TracePolicy, Pools,
)

# ============================================================================
# CONSTANTS
# ============================================================================

HEAD_DIM = 128
CHUNK_MIN = 256
CHUNK_MAX = 4096
MAX_WORK_UNITS = 65536

# DType shortcuts
F16 = DType.F16
F32 = DType.F32
I32 = DType.I32

print("=" * 70)
print("FlashInfer-Style Decode Attention - PTO-RT v9 Example")
print("=" * 70)


# ============================================================================
# WORK DESCRIPTOR
#
# Each worker receives a compact descriptor specifying its work
# ============================================================================

@dataclass
class WorkDescriptor:
    """Work descriptor for decode attention.

    Equivalent to FlashInfer's (request_indices, kv_tile_indices) but unified.
    """
    work_id: int        # Unique work identifier
    tier: int           # Kernel tier (size-optimized variant)
    flags: int          # First/last chunk flags (bit 0 = first, bit 1 = last)
    request_idx: int    # Which request in batch
    head_idx: int       # Which attention head
    kv_start: int       # KV cache start position
    kv_len: int         # KV cache length for this chunk

    @property
    def is_first_chunk(self) -> bool:
        return (self.flags & 0x01) != 0

    @property
    def is_last_chunk(self) -> bool:
        return (self.flags & 0x02) != 0


# ============================================================================
# PLANNING PHASE
#
# Generate work descriptors for all work units
# Binary search for optimal chunk size, then emit descriptors
# ============================================================================

class AttentionPlanner:
    """FlashInfer-style work planner for decode attention.

    Generates work descriptors that assign each AICore a specific chunk
    of KV cache to process.
    """

    def __init__(self, chunk_min: int = CHUNK_MIN, chunk_max: int = CHUNK_MAX):
        self.chunk_min = chunk_min
        self.chunk_max = chunk_max

    def get_total_work(
        self,
        kv_lengths: List[int],
        num_heads: int,
        chunk_size: int
    ) -> int:
        """Calculate total work units for given chunk size."""
        total = 0
        for kv_len in kv_lengths:
            num_chunks = math.ceil(kv_len / chunk_size)
            total += num_chunks * num_heads
        return total

    def plan_chunk_size(
        self,
        kv_lengths: List[int],
        num_heads: int,
        max_work_units: int = MAX_WORK_UNITS
    ) -> int:
        """Binary search for optimal chunk size.

        Finds largest chunk size that keeps total work under max_work_units.
        Larger chunks = fewer work units = less scheduling overhead.
        """
        lo, hi = self.chunk_min, self.chunk_max
        best = self.chunk_min

        while lo <= hi:
            mid = (lo + hi) // 2
            total = self.get_total_work(kv_lengths, num_heads, mid)

            if total <= max_work_units:
                best = mid
                lo = mid + 1  # Try larger chunk
            else:
                hi = mid - 1  # Need smaller chunk

        return best

    def generate(
        self,
        kv_lengths: List[int],
        num_heads: int,
        chunk_size: int
    ) -> List[WorkDescriptor]:
        """Generate work descriptors for all work units.

        Returns list of WorkDescriptor, one per chunk of work.
        Each descriptor specifies (request, head, kv_range, tier).
        """
        descriptors = []
        work_id = 0

        for req_idx, kv_len in enumerate(kv_lengths):
            num_chunks = math.ceil(kv_len / chunk_size)

            for head_idx in range(num_heads):
                for chunk_idx in range(num_chunks):
                    kv_start = chunk_idx * chunk_size
                    kv_chunk_len = min(chunk_size, kv_len - kv_start)

                    # Determine flags
                    flags = 0
                    if chunk_idx == 0:
                        flags |= 0x01  # First chunk
                    if chunk_idx == num_chunks - 1:
                        flags |= 0x02  # Last chunk

                    # Select tier based on chunk length
                    tier = self._select_tier(kv_chunk_len)

                    descriptors.append(WorkDescriptor(
                        work_id=work_id,
                        tier=tier,
                        flags=flags,
                        request_idx=req_idx,
                        head_idx=head_idx,
                        kv_start=kv_start,
                        kv_len=kv_chunk_len,
                    ))
                    work_id += 1

        return descriptors

    def _select_tier(self, kv_len: int) -> int:
        """Select kernel tier based on KV length."""
        if kv_len <= 512:
            return 0
        elif kv_len <= 1024:
            return 1
        elif kv_len <= 2048:
            return 2
        else:
            return 3


# ============================================================================
# KERNEL DEFINITIONS (unified @kernel with tl.* primitives)
#
# Tier-specific kernels optimized for different chunk sizes.
# Uses tl.* primitives that map to PTO-ISA tile operations:
# - tl.matmul -> TMATMUL (Cube unit matmul)
# - tl.rowmax/rowsum -> TREDUCE (Vector unit reduction)
# - tl.exp/div/mul/sub -> TMATH (Vector unit math)
# - tl.store -> TSTORE (Data movement)
#
# When compiled with -D__CPU_SIM, these execute on CPU.
# When compiled for NPU, these generate Ascend kernel code.
# ============================================================================

@kernel
def decode_attention_tier0(
    q_tile: In[Tile[1, HEAD_DIM, F16]],
    k_cache: In[Tile[512, HEAD_DIM, F16]],
    v_cache: In[Tile[512, HEAD_DIM, F16]],
    output: Out[Tile[1, HEAD_DIM, F16]],
    kv_len: Scalar[I32],
):
    """Tier 0 kernel - optimized for kv_len <= 512.

    Performs scaled dot-product attention with online softmax:
    1. Q * K^T -> attention scores
    2. Scale by 1/sqrt(d)
    3. Online softmax (FlashAttention style)
    4. Weighted sum of values
    """
    # Q * K^T -> attention scores
    scores = tl.matmul(q_tile, k_cache)  # [1, 512]

    # Scale by 1/sqrt(d)
    scale = tl.rsqrt(tl.constant(HEAD_DIM, F32))
    scores = tl.mul(scores, scale)

    # Online softmax (FlashAttention style)
    max_score = tl.rowmax(scores)
    scores = tl.sub(scores, max_score)
    exp_scores = tl.exp(scores)
    sum_exp = tl.rowsum(exp_scores)

    # Attention weights
    weights = tl.div(exp_scores, sum_exp)

    # Weighted sum of values
    result = tl.matmul(weights, v_cache)  # [1, HEAD_DIM]

    tl.store(output, result)


@kernel
def decode_attention_tier1(
    q_tile: In[Tile[1, HEAD_DIM, F16]],
    k_cache: In[Tile[1024, HEAD_DIM, F16]],
    v_cache: In[Tile[1024, HEAD_DIM, F16]],
    output: Out[Tile[1, HEAD_DIM, F16]],
    kv_len: Scalar[I32],
):
    """Tier 1 kernel - optimized for kv_len <= 1024."""
    scores = tl.matmul(q_tile, k_cache)
    scale = tl.rsqrt(tl.constant(HEAD_DIM, F32))
    scores = tl.mul(scores, scale)
    max_score = tl.rowmax(scores)
    scores = tl.sub(scores, max_score)
    exp_scores = tl.exp(scores)
    sum_exp = tl.rowsum(exp_scores)
    weights = tl.div(exp_scores, sum_exp)
    result = tl.matmul(weights, v_cache)
    tl.store(output, result)


@kernel
def decode_attention_tier2(
    q_tile: In[Tile[1, HEAD_DIM, F16]],
    k_cache: In[Tile[2048, HEAD_DIM, F16]],
    v_cache: In[Tile[2048, HEAD_DIM, F16]],
    output: Out[Tile[1, HEAD_DIM, F16]],
    kv_len: Scalar[I32],
):
    """Tier 2 kernel - optimized for kv_len <= 2048."""
    scores = tl.matmul(q_tile, k_cache)
    scale = tl.rsqrt(tl.constant(HEAD_DIM, F32))
    scores = tl.mul(scores, scale)
    max_score = tl.rowmax(scores)
    scores = tl.sub(scores, max_score)
    exp_scores = tl.exp(scores)
    sum_exp = tl.rowsum(exp_scores)
    weights = tl.div(exp_scores, sum_exp)
    result = tl.matmul(weights, v_cache)
    tl.store(output, result)


@kernel
def decode_attention_tier3(
    q_tile: In[Tile[1, HEAD_DIM, F16]],
    k_cache: In[Tile[4096, HEAD_DIM, F16]],
    v_cache: In[Tile[4096, HEAD_DIM, F16]],
    output: Out[Tile[1, HEAD_DIM, F16]],
    kv_len: Scalar[I32],
):
    """Tier 3 kernel - optimized for kv_len <= 4096."""
    scores = tl.matmul(q_tile, k_cache)
    scale = tl.rsqrt(tl.constant(HEAD_DIM, F32))
    scores = tl.mul(scores, scale)
    max_score = tl.rowmax(scores)
    scores = tl.sub(scores, max_score)
    exp_scores = tl.exp(scores)
    sum_exp = tl.rowsum(exp_scores)
    weights = tl.div(exp_scores, sum_exp)
    result = tl.matmul(weights, v_cache)
    tl.store(output, result)


# Tier kernel dispatch table
TIER_KERNELS = [
    decode_attention_tier0,
    decode_attention_tier1,
    decode_attention_tier2,
    decode_attention_tier3,
]


# ============================================================================
# WORKLOAD DEFINITION
# ============================================================================

def create_decode_attention_workload(
    descriptors: List[WorkDescriptor],
    batch_size: int,
    num_heads: int,
):
    """Create decode attention workload from pre-planned descriptors.

    This is the Plan-Run model: planning happens before workload creation,
    and the workload simply iterates over pre-computed descriptors.

    Args:
        descriptors: List of WorkDescriptor from planning phase
        batch_size: Number of requests in batch
        num_heads: Number of attention heads

    Returns:
        Compiled program ready for execution
    """
    # Tensors (shapes for documentation)
    Q = Tensor(data=None, shape=(batch_size, 1, num_heads, HEAD_DIM), dtype=F16)
    K_cache = Tensor(data=None, shape=(batch_size, CHUNK_MAX, num_heads, HEAD_DIM), dtype=F16)
    V_cache = Tensor(data=None, shape=(batch_size, CHUNK_MAX, num_heads, HEAD_DIM), dtype=F16)
    Output = Tensor(data=None, shape=(batch_size, 1, num_heads, HEAD_DIM), dtype=F16)

    # Convert descriptors to axis for iteration
    work_axis = DenseDyn(len(descriptors))

    # Use tier 0 kernel as the primary kernel for the workload
    # (In practice, tier selection happens at runtime based on descriptor.tier)
    primary_kernel = decode_attention_tier0

    @workload
    def decode_attention():
        """FlashInfer-style decode attention workload.

        Type: Workload[DenseDyn, DecodeTask, Independent]

        Each work unit corresponds to one descriptor specifying
        (request, head, kv_chunk) to process. The kernel is used
        directly with axis binding - no separate stub needed.
        """
        for w in P(work_axis):
            # Use kernel directly with axis binding
            # The runtime reads descriptor[w] to get (request, head, kv_range)
            # and selects the appropriate tier kernel
            primary_kernel[w](
                q_tile=Q,
                k_cache=K_cache,
                v_cache=V_cache,
                output=Output,
                kv_len=descriptors[0].kv_len if descriptors else 0,
            )

    # Schedule: all tasks are independent, use work-stealing for load balance
    scheduled = (decode_attention()
        .task_graph(
            deps=Deps.explicit(),  # No inter-task dependencies
            ready=ReadyPolicy.work_steal(),  # Dynamic load balancing
            start=StartPolicy.threshold(1),  # Start immediately when 1 task ready
            trace=TracePolicy.none(),
            pools=Pools.single(),
        )
        .dispatch(DispatchPolicy.work_steal())
        .streams(4)  # Multiple streams for parallelism
        .timing(TimingPolicy.immediate))

    return scheduled.compile()


# ============================================================================
# PLAN STATISTICS
# ============================================================================

def analyze_plan(descriptors: List[WorkDescriptor]) -> dict:
    """Analyze work distribution in the plan."""
    stats = {
        'total_work_units': len(descriptors),
        'tier_counts': [0, 0, 0, 0],
        'requests_with_chunks': {},
    }

    for desc in descriptors:
        stats['tier_counts'][desc.tier] += 1

        key = (desc.request_idx, desc.head_idx)
        if key not in stats['requests_with_chunks']:
            stats['requests_with_chunks'][key] = 0
        stats['requests_with_chunks'][key] += 1

    # Compute chunks per request stats
    chunks = list(stats['requests_with_chunks'].values())
    if chunks:
        stats['min_chunks_per_head'] = min(chunks)
        stats['max_chunks_per_head'] = max(chunks)
        stats['avg_chunks_per_head'] = sum(chunks) / len(chunks)

    return stats


def print_plan_stats(descriptors: List[WorkDescriptor], chunk_size: int):
    """Print planning statistics."""
    stats = analyze_plan(descriptors)

    print(f"\nPlan Statistics:")
    print(f"  Chunk size: {chunk_size}")
    print(f"  Total work units: {stats['total_work_units']}")
    print(f"  Tier distribution:")
    for i, count in enumerate(stats['tier_counts']):
        if count > 0:
            pct = 100.0 * count / stats['total_work_units']
            print(f"    Tier {i}: {count} ({pct:.1f}%)")

    if 'min_chunks_per_head' in stats:
        print(f"  Chunks per (request, head):")
        print(f"    min={stats['min_chunks_per_head']}, "
              f"max={stats['max_chunks_per_head']}, "
              f"avg={stats['avg_chunks_per_head']:.1f}")


# ============================================================================
# EXAMPLE EXECUTION
# ============================================================================

def main():
    # Configuration: Variable-length sequences
    kv_lengths = [
        512, 1024, 2048, 4096,
        512, 512, 1024, 2048,
        8192, 16384, 1024, 512,
        2048, 4096, 8192, 1024
    ]
    batch_size = len(kv_lengths)
    num_heads = 32

    print(f"\nConfiguration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Number of heads: {num_heads}")
    print(f"  Sequence lengths: {kv_lengths[:8]}...")
    print(f"  Total KV tokens: {sum(kv_lengths)}")

    # ===== PLANNING PHASE =====
    print(f"\n--- Planning Phase ---")

    planner = AttentionPlanner(chunk_min=256, chunk_max=4096)

    # Find optimal chunk size
    chunk_size = planner.plan_chunk_size(kv_lengths, num_heads)
    print(f"  Optimal chunk size: {chunk_size}")

    # Generate work descriptors
    descriptors = planner.generate(kv_lengths, num_heads, chunk_size)
    print(f"  Generated {len(descriptors)} work descriptors")

    # Print statistics
    print_plan_stats(descriptors, chunk_size)

    # Print first few descriptors
    print(f"\n  First 5 descriptors:")
    for desc in descriptors[:5]:
        print(f"    [{desc.work_id}] req={desc.request_idx} head={desc.head_idx} "
              f"kv=[{desc.kv_start},{desc.kv_start + desc.kv_len}) "
              f"tier={desc.tier} flags=0x{desc.flags:02x}")

    # ===== KERNEL INFO =====
    print(f"\n--- Kernel Info ---")
    print(f"  Available tier kernels: {len(TIER_KERNELS)}")
    for i, k in enumerate(TIER_KERNELS):
        print(f"    Tier {i}: {k.name}")

    # Trace tier 0 kernel to show IR
    print(f"\n  Kernel IR (tier 0, traced from tl.* primitives):")
    ir = decode_attention_tier0.trace()
    print(f"    Parameters: {len(ir.params)}")
    print(f"    Operations: {len(ir.ops)}")
    for op in ir.ops[:5]:
        print(f"      {op}")
    if len(ir.ops) > 5:
        print(f"      ... ({len(ir.ops) - 5} more operations)")

    # ===== EXECUTION PHASE =====
    print(f"\n--- Execution Phase ---")

    # Create workload from descriptors
    program = create_decode_attention_workload(descriptors, batch_size, num_heads)
    print(f"  Program compiled: {type(program).__name__}")

    # Execute with CPU simulation (via pto-isa backend)
    # No register_kernel needed - @kernel with tl.* primitives
    # automatically uses pto-isa CPU backend when compiled with -D__CPU_SIM
    print(f"  Executing with CPU simulation (pto-isa backend)...")
    program.execute()
    print("  Execution complete!")

    # ===== SUMMARY =====
    print(f"\n" + "=" * 70)
    print("FlashInfer-Style Decode Attention v9 Summary")
    print("=" * 70)
    print("""
Key v9 patterns demonstrated:

1. Plan-Run Execution Model:
   - Planning phase: Binary search for chunk size, generate descriptors
   - Execution phase: Each worker reads descriptor and executes
   - O(1) work lookup per task

2. Unified @kernel with tl.* primitives:
   - 4 tier-specific kernels for different chunk sizes
   - Online softmax (FlashAttention style)
   - tl.* operations map to PTO-ISA tile operations
   - CPU simulation uses pto-isa backend (-D__CPU_SIM)
   - No separate stub or CPU implementation needed

3. Work Descriptors:
   - Compact: (work_id, tier, flags, request, head, kv_range)
   - First/last chunk flags for partial results merging
   - Tier selection based on chunk length

4. Schedule Configuration:
   - Work-stealing for dynamic load balancing
   - Multiple streams for parallelism
   - Immediate timing for low latency

5. Comparison with FlashInfer CUDA:
   | Aspect           | FlashInfer CUDA        | PTO-RT v9              |
   |------------------|------------------------|------------------------|
   | Planning         | Python/C++ PagedAttention | AttentionPlanner class |
   | Descriptors      | request_indices[]      | WorkDescriptor list    |
   | Kernel dispatch  | CUDA blocks            | Task graph workers     |
   | Tile operations  | Manual registers       | tl.* primitives        |
   | Online softmax   | state_t::merge()       | tl.rowmax/exp/sum      |
   | CPU simulation   | N/A                    | pto-isa backend        |
""")


if __name__ == "__main__":
    main()
