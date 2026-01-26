/**
 * Host-Side Planning for Decode Attention
 *
 * This file demonstrates how to use the WorkPlanner to generate
 * work descriptors for FlashInfer-style decode attention.
 */

#include "../include/runtime.hpp"
#include <vector>
#include <cstdio>

using namespace pto::runtime;

// =============================================================================
// Planning Example
// =============================================================================

struct DecodeAttentionPlan {
    std::vector<WorkDescriptor> descriptors;
    int work_count;
    int chunk_size;
};

DecodeAttentionPlan plan_decode_attention(
    const int* kv_lengths,
    int batch_size,
    int num_heads,
    int max_work_units = 65536
) {
    DecodeAttentionPlan plan;

    // Configure planner
    PlanConfig config;
    config.chunk_min = 256;
    config.chunk_max = 4096;
    config.max_work_units = max_work_units;

    AttentionPlanner planner(config);

    // Binary search for optimal chunk size
    plan.chunk_size = planner.plan_chunk_size(kv_lengths, batch_size, num_heads);

    // Calculate buffer size needed
    int estimated_work = planner.get_total_work(
        kv_lengths, batch_size, num_heads, plan.chunk_size
    );

    // Allocate descriptor buffer
    plan.descriptors.resize(estimated_work);

    // Generate descriptors
    PlanResult result = planner.generate(
        kv_lengths,
        batch_size,
        num_heads,
        plan.chunk_size,
        plan.descriptors.data(),
        static_cast<int>(plan.descriptors.size()),
        &plan.work_count
    );

    if (result != PlanResult::OK) {
        fprintf(stderr, "Planning failed with error: %d\n", static_cast<int>(result));
        plan.work_count = 0;
    }

    // Resize to actual count
    plan.descriptors.resize(plan.work_count);

    return plan;
}

// =============================================================================
// Statistics and Debugging
// =============================================================================

struct PlanStats {
    int total_work_units;
    int tier_counts[4];
    int max_chunks_per_request;
    int min_chunks_per_request;
};

PlanStats analyze_plan(const DecodeAttentionPlan& plan) {
    PlanStats stats = {};
    stats.total_work_units = plan.work_count;
    stats.max_chunks_per_request = 0;
    stats.min_chunks_per_request = INT32_MAX;

    int current_request = -1;
    int chunks_in_request = 0;

    for (const auto& desc : plan.descriptors) {
        // Count tiers
        if (desc.tier < 4) {
            stats.tier_counts[desc.tier]++;
        }

        // Track chunks per request
        int req = params::Attention::request_idx(desc);
        if (req != current_request) {
            if (current_request >= 0) {
                stats.max_chunks_per_request = std::max(
                    stats.max_chunks_per_request, chunks_in_request
                );
                stats.min_chunks_per_request = std::min(
                    stats.min_chunks_per_request, chunks_in_request
                );
            }
            current_request = req;
            chunks_in_request = 1;
        } else {
            chunks_in_request++;
        }
    }

    // Handle last request
    if (current_request >= 0) {
        stats.max_chunks_per_request = std::max(
            stats.max_chunks_per_request, chunks_in_request
        );
        stats.min_chunks_per_request = std::min(
            stats.min_chunks_per_request, chunks_in_request
        );
    }

    return stats;
}

void print_plan_stats(const DecodeAttentionPlan& plan) {
    PlanStats stats = analyze_plan(plan);

    printf("Plan Statistics:\n");
    printf("  Chunk size: %d\n", plan.chunk_size);
    printf("  Total work units: %d\n", stats.total_work_units);
    printf("  Tier distribution:\n");
    for (int i = 0; i < 4; i++) {
        if (stats.tier_counts[i] > 0) {
            printf("    Tier %d: %d (%.1f%%)\n",
                   i, stats.tier_counts[i],
                   100.0 * stats.tier_counts[i] / stats.total_work_units);
        }
    }
    printf("  Chunks per request: min=%d, max=%d\n",
           stats.min_chunks_per_request, stats.max_chunks_per_request);
}

// =============================================================================
// Example Usage
// =============================================================================

#ifdef HOST_PLANNING_STANDALONE

int main() {
    // Example: Variable-length sequences
    std::vector<int> kv_lengths = {
        512, 1024, 2048, 4096,
        512, 512, 1024, 2048,
        8192, 16384, 1024, 512,
        2048, 4096, 8192, 1024
    };

    int batch_size = static_cast<int>(kv_lengths.size());
    int num_heads = 128;

    printf("FlashInfer-Style Decode Attention Planning\n");
    printf("==========================================\n");
    printf("Batch size: %d\n", batch_size);
    printf("Number of heads: %d\n", num_heads);
    printf("Sequence lengths: ");
    for (int i = 0; i < std::min(8, batch_size); i++) {
        printf("%d ", kv_lengths[i]);
    }
    if (batch_size > 8) printf("...");
    printf("\n\n");

    // Plan the workload
    auto plan = plan_decode_attention(
        kv_lengths.data(),
        batch_size,
        num_heads
    );

    // Print statistics
    print_plan_stats(plan);

    // Print first few descriptors
    printf("\nFirst 5 descriptors:\n");
    for (int i = 0; i < std::min(5, plan.work_count); i++) {
        const auto& d = plan.descriptors[i];
        printf("  [%d] req=%d head=%d kv=[%d,%d) tier=%d flags=0x%02x\n",
               d.work_id,
               params::Attention::request_idx(d),
               params::Attention::head_idx(d),
               params::Attention::kv_start(d),
               params::Attention::kv_end(d),
               d.tier,
               d.flags);
    }

    return 0;
}

#endif  // HOST_PLANNING_STANDALONE
