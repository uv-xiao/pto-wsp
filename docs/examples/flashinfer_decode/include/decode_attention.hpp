/**
 * FlashInfer-Style Decode Attention Kernel
 *
 * This header defines the decode attention kernel interface using the
 * PTO-ISA runtime extension. It demonstrates:
 * - Multi-tier kernel definition
 * - Descriptor-based work assignment
 * - Online softmax with state merging
 */

#ifndef DECODE_ATTENTION_HPP
#define DECODE_ATTENTION_HPP

#include "runtime.hpp"
#include <pto/pto-inst.hpp>

namespace flashinfer {

using namespace pto;
using namespace pto::runtime;

// =============================================================================
// Kernel Configuration
// =============================================================================

// Tile sizes for each tier (optimized for different sequence lengths)
template <int Tier>
struct TierConfig;

template <>
struct TierConfig<0> {  // Tier 0: 1-1024 tokens
    static constexpr int TILE_KV = 256;
    static constexpr int PREFETCH_DEPTH = 2;
};

template <>
struct TierConfig<1> {  // Tier 1: 1025-4096 tokens
    static constexpr int TILE_KV = 512;
    static constexpr int PREFETCH_DEPTH = 3;
};

template <>
struct TierConfig<2> {  // Tier 2: 4097-16384 tokens
    static constexpr int TILE_KV = 1024;
    static constexpr int PREFETCH_DEPTH = 4;
};

template <>
struct TierConfig<3> {  // Tier 3: 16385-131072 tokens
    static constexpr int TILE_KV = 2048;
    static constexpr int PREFETCH_DEPTH = 4;
};

// =============================================================================
// Online Softmax State
// =============================================================================

template <int VecSize>
struct SoftmaxState {
    float max_val;
    float sum_val;
    float output[VecSize];

    void init() {
        max_val = -1e30f;
        sum_val = 0.0f;
        for (int i = 0; i < VecSize; i++) {
            output[i] = 0.0f;
        }
    }

    void merge(const SoftmaxState& other) {
        float new_max = (max_val > other.max_val) ? max_val : other.max_val;
        float scale_self = expf(max_val - new_max);
        float scale_other = expf(other.max_val - new_max);

        for (int i = 0; i < VecSize; i++) {
            output[i] = output[i] * scale_self + other.output[i] * scale_other;
        }
        sum_val = sum_val * scale_self + other.sum_val * scale_other;
        max_val = new_max;
    }

    void normalize() {
        float inv_sum = 1.0f / sum_val;
        for (int i = 0; i < VecSize; i++) {
            output[i] *= inv_sum;
        }
    }
};

// =============================================================================
// Kernel Implementation (Template for each tier)
// =============================================================================

template <int Tier>
AICORE void decode_attention_kernel_impl(
    const WorkDescriptor& desc,
    __gm__ half* Q,
    __gm__ half* K_cache,
    __gm__ half* V_cache,
    __gm__ half* Output,
    int num_heads,
    int head_dim,
    int max_seq_len
) {
    using TC = TierConfig<Tier>;

    // Extract work parameters from descriptor
    const uint32_t req_idx = params::Attention::request_idx(desc);
    const uint32_t head_idx = params::Attention::head_idx(desc);
    const uint32_t kv_start = params::Attention::kv_start(desc);
    const uint32_t kv_len = params::Attention::kv_len(desc);
    const bool is_first = (desc.flags & WorkDescriptor::FLAG_FIRST) != 0;
    const bool is_last = (desc.flags & WorkDescriptor::FLAG_LAST) != 0;

    // Tile type definitions
    using QTile = VecTile<half, 1, 128>;
    using KTile = MatTile<half, TC::TILE_KV, 128>;
    using VTile = MatTile<half, TC::TILE_KV, 128>;
    using ScoreTile = VecTile<float, 1, TC::TILE_KV>;
    using AccTile = VecTile<float, 1, 128>;

    // Allocate tiles
    QTile q_tile;
    KTile k_tile;
    VTile v_tile;
    ScoreTile score_tile;
    AccTile acc_tile;

    // Online softmax state
    VecTile<float, 1, 1> row_max, row_sum;

    // Scaling factor: 1/sqrt(head_dim)
    constexpr float scale = 0.0883883476f;  // 1/sqrt(128)

    // Load query vector
    auto q_offset = (req_idx * num_heads + head_idx) * head_dim;
    // Using GlobalTensor for strided access
    using QGlobal = GlobalTensor<half, Shape<1, 1, 1, 1, 128>, Stride<1, 1, 1, 1, 1>, Layout::ND>;
    QGlobal q_global(Q + q_offset);
    TLOAD(q_tile, q_global);

    // Initialize state for first chunk
    if (is_first) {
        TEXPANDS(row_max, -1e30f);
        TEXPANDS(row_sum, 0.0f);
        TEXPANDS(acc_tile, 0.0f);
    }

    // Process KV cache in tiles
    for (uint32_t pos = 0; pos < kv_len; pos += TC::TILE_KV) {
        int tile_len = (kv_len - pos < TC::TILE_KV) ? (kv_len - pos) : TC::TILE_KV;

        // Set valid region for partial tiles
        k_tile.SetValidRegion(tile_len, head_dim);
        v_tile.SetValidRegion(tile_len, head_dim);

        // Calculate KV cache offset
        // Layout: [batch, seq, head, dim]
        auto kv_offset = (req_idx * max_seq_len * num_heads +
                         (kv_start + pos) * num_heads +
                         head_idx) * head_dim;

        // Using GlobalTensor for KV access
        using KVGlobal = GlobalTensor<half,
            Shape<1, 1, 1, TC::TILE_KV, 128>,
            Stride<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC, 1>,
            Layout::ND>;

        KVGlobal k_global(K_cache + kv_offset);
        KVGlobal v_global(V_cache + kv_offset);

        TLOAD(k_tile, k_global);
        TLOAD(v_tile, v_global);

        // Compute attention scores: Q @ K^T
        // score = Q * K^T (simplified - actual impl uses TMATMUL)
        // TMATMUL(score_tile, q_tile, TTRANS(k_tile));

        // Scale scores
        TMULS(score_tile, score_tile, scale);

        // Online softmax update
        VecTile<float, 1, 1> local_max, local_sum;
        VecTile<float, 1, TC::TILE_KV> score_exp;

        // max_new = max(max_old, max(scores))
        TROWMAX(local_max, score_tile);
        TMAX(local_max, local_max, row_max);

        // Rescale old accumulator
        VecTile<float, 1, 1> scale_old;
        TSUB(scale_old, row_max, local_max);
        TEXP(scale_old, scale_old);
        TMUL(acc_tile, acc_tile, scale_old);
        TMUL(row_sum, row_sum, scale_old);

        // Compute exp(scores - max_new)
        TROWEXPANDSUB(score_exp, score_tile, local_max);
        TEXP(score_exp, score_exp);

        // Update sum
        TROWSUM(local_sum, score_exp);
        TADD(row_sum, row_sum, local_sum);

        // Accumulate: acc += exp_scores @ V
        // TMATMUL_ACC(acc_tile, acc_tile, score_exp, v_tile);

        // Update max
        row_max = local_max;
    }

    // Store output if last chunk
    if (is_last) {
        // Normalize by sum
        TROWEXPANDDIV(acc_tile, acc_tile, row_sum);

        // Convert to half and store
        VecTile<half, 1, 128> out_tile;
        TCVT(out_tile, acc_tile, RoundMode::CAST_ROUND);

        auto out_offset = (req_idx * num_heads + head_idx) * head_dim;
        using OutGlobal = GlobalTensor<half, Shape<1, 1, 1, 1, 128>, Stride<1, 1, 1, 1, 1>, Layout::ND>;
        OutGlobal out_global(Output + out_offset);
        TSTORE(out_global, out_tile);
    }
}

// =============================================================================
// Kernel Dispatch
// =============================================================================

// Dispatch table type
using KernelFnPtr = void (*)(
    const WorkDescriptor&,
    __gm__ half*,
    __gm__ half*,
    __gm__ half*,
    __gm__ half*,
    int, int, int
);

// Static dispatch table
inline constexpr KernelFnPtr dispatch_table[] = {
    &decode_attention_kernel_impl<0>,
    &decode_attention_kernel_impl<1>,
    &decode_attention_kernel_impl<2>,
    &decode_attention_kernel_impl<3>,
};

// Entry point
__global__ __aicore__ void decode_attention_entry(
    __gm__ WorkDescriptor* descs,
    __gm__ half* Q,
    __gm__ half* K_cache,
    __gm__ half* V_cache,
    __gm__ half* Output,
    int num_heads,
    int head_dim,
    int max_seq_len
) {
    // Read descriptor for this work unit
    WorkDescriptor desc = descs[get_block_idx()];

    // Dispatch to tier-specific kernel
    dispatch_table[desc.tier](
        desc, Q, K_cache, V_cache, Output,
        num_heads, head_dim, max_seq_len
    );
}

}  // namespace flashinfer

#endif  // DECODE_ATTENTION_HPP
