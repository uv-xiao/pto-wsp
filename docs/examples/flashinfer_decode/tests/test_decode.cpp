/**
 * Unit Tests for FlashInfer-Style Decode Attention
 *
 * Tests the planning and descriptor generation components.
 * Note: Device kernel tests require NPU hardware.
 */

#include "../include/runtime.hpp"
#include <cassert>
#include <vector>
#include <cstdio>
#include <cstring>

using namespace pto::runtime;

// =============================================================================
// Test Utilities
// =============================================================================

#define TEST_ASSERT(cond, msg) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAILED: %s\n  %s:%d\n", msg, __FILE__, __LINE__); \
        return false; \
    } \
} while(0)

#define RUN_TEST(test_fn) do { \
    printf("Running %s... ", #test_fn); \
    if (test_fn()) { \
        printf("PASSED\n"); \
        passed++; \
    } else { \
        failed++; \
    } \
} while(0)

// =============================================================================
// Dim Tests
// =============================================================================

bool test_static_dim() {
    Dim<128> d;
    TEST_ASSERT(d.get_size() == 128, "Static dim should return compile-time size");
    TEST_ASSERT(!Dim<128>::is_dynamic, "Static dim should not be dynamic");
    return true;
}

bool test_dynamic_dim() {
    Dim<DYNAMIC> d(256);
    TEST_ASSERT(d.get_size() == 256, "Dynamic dim should return runtime size");
    TEST_ASSERT(Dim<DYNAMIC>::is_dynamic, "Dynamic dim should be dynamic");
    return true;
}

// =============================================================================
// IterationSpace Tests
// =============================================================================

bool test_iteration_space_static() {
    IterationSpace<Dim<2>, Dim<3>, Dim<4>> space;
    TEST_ASSERT(space.total_work() == 24, "Total work should be 2*3*4=24");
    return true;
}

bool test_iteration_space_dynamic() {
    IterationSpace<Dim<DYNAMIC>, Dim<128>, Dim<DYNAMIC>> space;
    space.set_dim<0>(4);
    space.set_dim<2>(8);
    TEST_ASSERT(space.total_work() == 4 * 128 * 8, "Total work should be 4*128*8");
    return true;
}

bool test_iteration_space_coords() {
    IterationSpace<Dim<2>, Dim<3>, Dim<4>> space;

    int coords[3];
    space.index_to_coords(0, coords);
    TEST_ASSERT(coords[0] == 0 && coords[1] == 0 && coords[2] == 0,
               "Index 0 should map to (0,0,0)");

    space.index_to_coords(1, coords);
    TEST_ASSERT(coords[0] == 1 && coords[1] == 0 && coords[2] == 0,
               "Index 1 should map to (1,0,0)");

    space.index_to_coords(5, coords);  // 5 = 1 + 2*2 = (1, 2, 0)
    TEST_ASSERT(coords[0] == 1 && coords[1] == 2 && coords[2] == 0,
               "Index 5 should map to (1,2,0)");

    return true;
}

// =============================================================================
// WorkDescriptor Tests
// =============================================================================

bool test_work_descriptor_size() {
    TEST_ASSERT(sizeof(WorkDescriptor) == 24, "WorkDescriptor should be 24 bytes");
    TEST_ASSERT(alignof(WorkDescriptor) == 8, "WorkDescriptor should be 8-byte aligned");
    return true;
}

bool test_work_descriptor_params() {
    WorkDescriptor d;
    params::Attention::set(d, 10, 20, 1000, 500);

    TEST_ASSERT(params::Attention::request_idx(d) == 10, "request_idx mismatch");
    TEST_ASSERT(params::Attention::head_idx(d) == 20, "head_idx mismatch");
    TEST_ASSERT(params::Attention::kv_start(d) == 1000, "kv_start mismatch");
    TEST_ASSERT(params::Attention::kv_len(d) == 500, "kv_len mismatch");
    TEST_ASSERT(params::Attention::kv_end(d) == 1500, "kv_end mismatch");

    return true;
}

// =============================================================================
// TierConfig Tests
// =============================================================================

bool test_tier_selection() {
    TEST_ASSERT(DecodeAttentionTiers::select_tier(100) == 0, "100 should be tier 0");
    TEST_ASSERT(DecodeAttentionTiers::select_tier(1024) == 0, "1024 should be tier 0");
    TEST_ASSERT(DecodeAttentionTiers::select_tier(1025) == 1, "1025 should be tier 1");
    TEST_ASSERT(DecodeAttentionTiers::select_tier(4096) == 1, "4096 should be tier 1");
    TEST_ASSERT(DecodeAttentionTiers::select_tier(4097) == 2, "4097 should be tier 2");
    TEST_ASSERT(DecodeAttentionTiers::select_tier(16384) == 2, "16384 should be tier 2");
    TEST_ASSERT(DecodeAttentionTiers::select_tier(16385) == 3, "16385 should be tier 3");
    TEST_ASSERT(DecodeAttentionTiers::select_tier(131072) == 3, "131072 should be tier 3");
    TEST_ASSERT(DecodeAttentionTiers::select_tier(200000) == -1, "200000 should be unsupported");

    return true;
}

// =============================================================================
// WorkPlanner Tests
// =============================================================================

bool test_planner_uniform_length() {
    int kv_lengths[] = {1024, 1024, 1024, 1024};
    int batch_size = 4;
    int num_heads = 2;

    AttentionPlanner planner(PlanConfig{.chunk_min = 512, .chunk_max = 1024});

    int chunk_size = planner.plan_chunk_size(kv_lengths, batch_size, num_heads);
    TEST_ASSERT(chunk_size >= 512 && chunk_size <= 1024, "Chunk size out of range");

    std::vector<WorkDescriptor> descs(1000);
    int work_count;
    auto result = planner.generate(
        kv_lengths, batch_size, num_heads, chunk_size,
        descs.data(), 1000, &work_count
    );

    TEST_ASSERT(result == PlanResult::OK, "Planning should succeed");
    TEST_ASSERT(work_count > 0, "Should have work units");

    // Verify all descriptors have tier 0 (seq_len = 1024)
    for (int i = 0; i < work_count; i++) {
        TEST_ASSERT(descs[i].tier == 0, "All descriptors should be tier 0");
    }

    return true;
}

bool test_planner_variable_length() {
    int kv_lengths[] = {512, 2048, 8192, 1024};
    int batch_size = 4;
    int num_heads = 1;

    AttentionPlanner planner;

    int chunk_size = planner.plan_chunk_size(kv_lengths, batch_size, num_heads);

    std::vector<WorkDescriptor> descs(1000);
    int work_count;
    auto result = planner.generate(
        kv_lengths, batch_size, num_heads, chunk_size,
        descs.data(), 1000, &work_count
    );

    TEST_ASSERT(result == PlanResult::OK, "Planning should succeed");

    // Verify tier assignment
    bool found_tier_0 = false, found_tier_1 = false, found_tier_2 = false;
    for (int i = 0; i < work_count; i++) {
        if (descs[i].tier == 0) found_tier_0 = true;
        if (descs[i].tier == 1) found_tier_1 = true;
        if (descs[i].tier == 2) found_tier_2 = true;
    }
    TEST_ASSERT(found_tier_0, "Should have tier 0 work (512, 1024)");
    TEST_ASSERT(found_tier_1, "Should have tier 1 work (2048)");
    TEST_ASSERT(found_tier_2, "Should have tier 2 work (8192)");

    return true;
}

bool test_planner_first_last_flags() {
    int kv_lengths[] = {2048};  // Will be split into multiple chunks
    int batch_size = 1;
    int num_heads = 1;
    int chunk_size = 512;  // Force 4 chunks

    AttentionPlanner planner;

    std::vector<WorkDescriptor> descs(100);
    int work_count;
    auto result = planner.generate(
        kv_lengths, batch_size, num_heads, chunk_size,
        descs.data(), 100, &work_count
    );

    TEST_ASSERT(result == PlanResult::OK, "Planning should succeed");
    TEST_ASSERT(work_count == 4, "Should have 4 chunks");

    TEST_ASSERT(descs[0].flags & WorkDescriptor::FLAG_FIRST, "First chunk should have FIRST flag");
    TEST_ASSERT(!(descs[0].flags & WorkDescriptor::FLAG_LAST), "First chunk should not have LAST flag");

    TEST_ASSERT(!(descs[1].flags & WorkDescriptor::FLAG_FIRST), "Middle chunk should not have FIRST flag");
    TEST_ASSERT(!(descs[1].flags & WorkDescriptor::FLAG_LAST), "Middle chunk should not have LAST flag");

    TEST_ASSERT(!(descs[3].flags & WorkDescriptor::FLAG_FIRST), "Last chunk should not have FIRST flag");
    TEST_ASSERT(descs[3].flags & WorkDescriptor::FLAG_LAST, "Last chunk should have LAST flag");

    return true;
}

bool test_planner_buffer_overflow() {
    int kv_lengths[] = {4096, 4096, 4096, 4096};
    int batch_size = 4;
    int num_heads = 128;
    int chunk_size = 256;  // This will create many work units

    AttentionPlanner planner;

    std::vector<WorkDescriptor> descs(10);  // Too small
    int work_count;
    auto result = planner.generate(
        kv_lengths, batch_size, num_heads, chunk_size,
        descs.data(), 10, &work_count
    );

    TEST_ASSERT(result == PlanResult::BUFFER_OVERFLOW, "Should report buffer overflow");

    return true;
}

// =============================================================================
// Main
// =============================================================================

int main() {
    printf("FlashInfer Decode Attention Tests\n");
    printf("==================================\n\n");

    int passed = 0;
    int failed = 0;

    // Dim tests
    RUN_TEST(test_static_dim);
    RUN_TEST(test_dynamic_dim);

    // IterationSpace tests
    RUN_TEST(test_iteration_space_static);
    RUN_TEST(test_iteration_space_dynamic);
    RUN_TEST(test_iteration_space_coords);

    // WorkDescriptor tests
    RUN_TEST(test_work_descriptor_size);
    RUN_TEST(test_work_descriptor_params);

    // TierConfig tests
    RUN_TEST(test_tier_selection);

    // WorkPlanner tests
    RUN_TEST(test_planner_uniform_length);
    RUN_TEST(test_planner_variable_length);
    RUN_TEST(test_planner_first_last_flags);
    RUN_TEST(test_planner_buffer_overflow);

    printf("\n==================================\n");
    printf("Results: %d passed, %d failed\n", passed, failed);

    return failed > 0 ? 1 : 0;
}
