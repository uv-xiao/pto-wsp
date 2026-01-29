// PTO Workload-Schedule Programming (PTO-WSP) framework v9 - Task Graph Infrastructure Unit Tests
// Copyright (c) 2026 PTO Project
// SPDX-License-Identifier: MIT

#include "pto/rt/graph/graph.hpp"
#include <iostream>
#include <cassert>
#include <thread>
#include <atomic>
#include <chrono>

using namespace pto::wsp::graph;

// Test helper
#define TEST(name) void test_##name(); \
    static bool registered_##name = (tests.push_back({#name, test_##name}), true); \
    void test_##name()

std::vector<std::pair<const char*, void(*)()>> tests;

// ============================================================
// TensorRegion2D Tests
// ============================================================

TEST(tensor_region_overlap) {
    TensorRegion2D r1{0x1000, 0, 0, 32, 32};
    TensorRegion2D r2{0x1000, 16, 16, 32, 32};  // Overlaps with r1
    TensorRegion2D r3{0x1000, 64, 64, 32, 32};  // No overlap
    TensorRegion2D r4{0x2000, 0, 0, 32, 32};    // Different base

    assert(r1.overlaps(r2) && "r1 and r2 should overlap");
    assert(!r1.overlaps(r3) && "r1 and r3 should not overlap");
    assert(!r1.overlaps(r4) && "r1 and r4 have different base");

    std::cout << "  TensorRegion2D overlap tests passed\n";
}

TEST(tensor_region_contains) {
    TensorRegion2D outer{0x1000, 0, 0, 64, 64};
    TensorRegion2D inner{0x1000, 16, 16, 16, 16};  // Inside outer
    TensorRegion2D partial{0x1000, 32, 32, 64, 64};  // Partially outside

    assert(outer.contains(inner) && "outer should contain inner");
    assert(!outer.contains(partial) && "outer should not contain partial");
    assert(!inner.contains(outer) && "inner should not contain outer");

    std::cout << "  TensorRegion2D contains tests passed\n";
}

// ============================================================
// KernelBundle Tests
// ============================================================

TEST(kernel_bundle_register) {
    KernelBundle bundle;

    KernelInfo attn_info{"attn_kernel", "_attn_kernel_v1", 2, 4,
                         ExecDomain::AscendAICore, ExecPool::Cube};
    KernelInfo mlp_info{"mlp_kernel", "_mlp_kernel_v1", 3, 2,
                        ExecDomain::AscendAICore, ExecPool::Vector};

    auto attn_id = bundle.register_kernel(attn_info);
    auto mlp_id = bundle.register_kernel(mlp_info);

    assert(attn_id == 0);
    assert(mlp_id == 1);
    assert(bundle.size() == 2);

    auto found_attn = bundle.find_kernel("attn_kernel");
    assert(found_attn.has_value() && *found_attn == attn_id);

    auto not_found = bundle.find_kernel("nonexistent");
    assert(!not_found.has_value());

    std::cout << "  KernelBundle register tests passed\n";
}

// ============================================================
// TaskGraphStorage Tests
// ============================================================

TEST(task_graph_storage_basic) {
    TaskGraphStorage storage;

    // Create some tasks
    TaskNodePod task1{};
    task1.kernel = 0;
    task1.domain = ExecDomain::HostCPU;
    task1.pool = ExecPool::Any;
    task1.fanin = 0;

    TaskNodePod task2{};
    task2.kernel = 0;
    task2.domain = ExecDomain::HostCPU;
    task2.pool = ExecPool::Any;
    task2.fanin = 0;

    auto id1 = storage.add_task(task1);
    auto id2 = storage.add_task(task2);

    assert(id1 == 0);
    assert(id2 == 1);
    assert(storage.num_tasks() == 2);

    // Add edge: task1 -> task2
    storage.increment_fanin(id2);
    storage.set_fanout(id1, {id2});
    storage.finalize();

    // Check ready tasks (only task1 should be ready)
    assert(storage.ready_tasks().size() == 1);
    assert(storage.ready_tasks()[0] == id1);

    // Check fanout
    auto fanout = storage.get_fanout(id1);
    assert(fanout.size() == 1 && fanout[0] == id2);

    std::cout << "  TaskGraphStorage basic tests passed\n";
}

TEST(task_graph_builder) {
    TaskGraphStorage storage;
    TaskGraphBuilder builder(storage);

    // Build a simple diamond graph: t0 -> (t1, t2) -> t3
    auto t0 = builder.begin_task(0)
        .set_domain(ExecDomain::HostCPU)
        .add_arg(42)
        .submit();

    auto t1 = builder.begin_task(0).set_pool(ExecPool::Vector).submit();
    auto t2 = builder.begin_task(0).set_pool(ExecPool::Cube).submit();
    auto t3 = builder.begin_task(0).submit();

    builder.add_dependency(t0, t1);
    builder.add_dependency(t0, t2);
    builder.add_dependency(t1, t3);
    builder.add_dependency(t2, t3);
    builder.finalize();

    assert(storage.num_tasks() == 4);
    assert(storage.ready_tasks().size() == 1);
    assert(storage.ready_tasks()[0] == t0);

    // Check fanin counts
    assert(storage.get_task(t0).fanin == 0);
    assert(storage.get_task(t1).fanin == 1);
    assert(storage.get_task(t2).fanin == 1);
    assert(storage.get_task(t3).fanin == 2);

    std::cout << "  TaskGraphBuilder tests passed\n";
}

// ============================================================
// TensorMap Tests
// ============================================================

TEST(fixed_tensor_map_basic) {
    FixedTensorMap map;

    TensorRegion2D r1{0x1000, 0, 0, 32, 32};
    TensorRegion2D r2{0x1000, 32, 0, 32, 32};
    TensorRegion2D r3{0x2000, 0, 0, 64, 64};

    map.insert_output(r1, ProducerRef{0, 0});
    map.insert_output(r2, ProducerRef{1, 0});
    map.insert_output(r3, ProducerRef{2, 0});

    assert(map.size() == 3);

    // Find exact matches
    auto p1 = map.find_producer(r1);
    assert(p1.has_value() && p1->producer == 0);

    auto p2 = map.find_producer(r2);
    assert(p2.has_value() && p2->producer == 1);

    // Find overlapping region
    TensorRegion2D overlap{0x1000, 16, 16, 32, 32};  // Overlaps r1 and r2
    auto p_overlap = map.find_producer(overlap);
    assert(p_overlap.has_value());  // Should find something (latest producer)

    std::cout << "  FixedTensorMap basic tests passed\n";
}

TEST(tensor_map_gc) {
    DynamicTensorMap map;

    TensorRegion2D r1{0x1000, 0, 0, 32, 32};
    TensorRegion2D r2{0x1000, 32, 0, 32, 32};

    map.insert_output(r1, ProducerRef{0, 0});
    map.insert_output(r2, ProducerRef{5, 0});

    assert(map.size() == 2);

    // GC entries before task 3
    map.gc_before(3);

    // r1 (producer 0) should be removed, r2 (producer 5) should remain
    assert(map.size() == 1);

    auto p1 = map.find_producer(r1);
    assert(!p1.has_value());

    auto p2 = map.find_producer(r2);
    assert(p2.has_value() && p2->producer == 5);

    std::cout << "  TensorMap GC tests passed\n";
}

TEST(dependency_analyzer) {
    DynamicTensorMap map;
    DependencyAnalyzer analyzer(map);

    // Create a producer task
    TaskNodePod producer{};
    producer.id = 0;
    producer.num_io = 1;
    producer.io[0] = TaskIO{{0x1000, 0, 0, 32, 32}, 1, {}};  // Output

    analyzer.register_outputs(producer);

    // Create a consumer task
    TaskNodePod consumer{};
    consumer.id = 1;
    consumer.num_io = 1;
    consumer.io[0] = TaskIO{{0x1000, 0, 0, 32, 32}, 0, {}};  // Input (same region)

    auto deps = analyzer.analyze_dependencies(consumer);

    assert(deps.size() == 1);
    assert(deps[0] == 0);  // Depends on producer

    std::cout << "  DependencyAnalyzer tests passed\n";
}

// ============================================================
// ReadyQueue Tests
// ============================================================

TEST(ready_queue_mpmc_basic) {
    ReadyQueueMPMC queue;

    queue.push(1);
    queue.push(2);
    queue.push(3);

    assert(queue.size() == 3);
    assert(!queue.empty());

    auto t1 = queue.try_pop();
    assert(t1.has_value() && *t1 == 1);

    auto t2 = queue.try_pop();
    assert(t2.has_value() && *t2 == 2);

    auto t3 = queue.try_pop();
    assert(t3.has_value() && *t3 == 3);

    assert(queue.empty());

    auto t4 = queue.try_pop();
    assert(!t4.has_value());

    std::cout << "  ReadyQueueMPMC basic tests passed\n";
}

TEST(ready_queue_spsc_basic) {
    ReadyQueueSPSC queue;

    queue.push(10);
    queue.push(20);

    assert(queue.size() == 2);

    auto t1 = queue.try_pop();
    assert(t1.has_value() && *t1 == 10);

    auto t2 = queue.try_pop();
    assert(t2.has_value() && *t2 == 20);

    assert(queue.empty());

    std::cout << "  ReadyQueueSPSC basic tests passed\n";
}

TEST(dual_queue_set) {
    DualQueueSet queues;

    queues.push(ExecPool::Vector, 1);
    queues.push(ExecPool::Cube, 2);
    queues.push(ExecPool::Vector, 3);

    auto v1 = queues.try_pop(ExecPool::Vector);
    assert(v1.has_value() && *v1 == 1);

    auto c1 = queues.try_pop(ExecPool::Cube);
    assert(c1.has_value() && *c1 == 2);

    auto any = queues.try_pop_any();
    assert(any.has_value() && *any == 3);  // From vector queue

    assert(queues.all_empty());

    std::cout << "  DualQueueSet tests passed\n";
}

// ============================================================
// Runtime State Tests
// ============================================================

TEST(window_state_basic) {
    WindowState window(3, WindowMode::Stall);

    assert(window.has_capacity());
    assert(window.try_enter());
    assert(window.try_enter());
    assert(window.try_enter());
    assert(!window.try_enter());  // Full

    window.exit();
    assert(window.try_enter());  // Space available again

    std::cout << "  WindowState basic tests passed\n";
}

TEST(issue_gate_basic) {
    IssueGate gate(2);

    assert(gate.try_acquire());
    assert(gate.try_acquire());
    assert(!gate.try_acquire());  // At depth limit

    assert(gate.in_flight() == 2);

    gate.release();
    assert(gate.in_flight() == 1);
    assert(gate.try_acquire());

    std::cout << "  IssueGate basic tests passed\n";
}

TEST(dep_batcher_basic) {
    DepBatcher batcher(3);  // Threshold of 3

    batcher.add_pending(0, 1);
    batcher.add_pending(0, 2);
    assert(!batcher.needs_flush());

    batcher.add_pending(1, 3);
    assert(batcher.needs_flush());

    auto pending = batcher.flush();
    assert(pending.size() == 3);
    assert(!batcher.needs_flush());

    std::cout << "  DepBatcher basic tests passed\n";
}

// ============================================================
// Integration Test
// ============================================================

TEST(runtime_integration) {
    TaskGraphStorage storage;
    TaskGraphBuilder builder(storage);

    // Build a chain: t0 -> t1 -> t2
    auto t0 = builder.begin_task(0)
        .add_output({0x1000, 0, 0, 32, 32})
        .submit();

    auto t1 = builder.begin_task(0)
        .add_input({0x1000, 0, 0, 32, 32})
        .add_output({0x2000, 0, 0, 32, 32})
        .submit();

    auto t2 = builder.begin_task(0)
        .add_input({0x2000, 0, 0, 32, 32})
        .submit();

    builder.add_dependency(t0, t1);
    builder.add_dependency(t1, t2);
    builder.finalize();

    // Create runtime
    TaskGraphRuntime runtime(storage);
    runtime.initialize();

    // Initially only t0 should be ready
    auto ready = runtime.try_get_ready();
    assert(ready.has_value() && *ready == t0);

    // Complete t0
    runtime.task_complete(t0);

    // Now t1 should be ready
    ready = runtime.try_get_ready();
    assert(ready.has_value() && *ready == t1);

    // Complete t1
    runtime.task_complete(t1);

    // Now t2 should be ready
    ready = runtime.try_get_ready();
    assert(ready.has_value() && *ready == t2);

    // Complete t2
    runtime.task_complete(t2);

    // All done
    assert(runtime.all_complete());

    std::cout << "  Runtime integration test passed\n";
}

// ============================================================
// Concurrent Test
// ============================================================

TEST(concurrent_ready_queue) {
    ReadyQueueMPMC queue;
    const int NUM_PRODUCERS = 4;
    const int NUM_TASKS_PER_PRODUCER = 100;
    std::atomic<int> consumed{0};

    // Producer threads
    std::vector<std::thread> producers;
    for (int p = 0; p < NUM_PRODUCERS; ++p) {
        producers.emplace_back([&queue, p]() {
            for (int i = 0; i < NUM_TASKS_PER_PRODUCER; ++i) {
                queue.push(p * NUM_TASKS_PER_PRODUCER + i);
            }
        });
    }

    // Consumer threads
    std::vector<std::thread> consumers;
    for (int c = 0; c < 2; ++c) {
        consumers.emplace_back([&queue, &consumed]() {
            while (true) {
                auto tid = queue.try_pop();
                if (tid) {
                    consumed.fetch_add(1);
                } else {
                    // Check if we're done
                    if (consumed.load() >= NUM_PRODUCERS * NUM_TASKS_PER_PRODUCER) {
                        break;
                    }
                    std::this_thread::yield();
                }
            }
        });
    }

    // Wait for all threads
    for (auto& t : producers) t.join();

    // Give consumers time to finish
    auto start = std::chrono::steady_clock::now();
    while (consumed.load() < NUM_PRODUCERS * NUM_TASKS_PER_PRODUCER) {
        if (std::chrono::steady_clock::now() - start > std::chrono::seconds(5)) {
            assert(false && "Timeout waiting for consumers");
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    queue.shutdown();
    for (auto& t : consumers) t.join();

    assert(consumed.load() == NUM_PRODUCERS * NUM_TASKS_PER_PRODUCER);

    std::cout << "  Concurrent ready queue test passed (" << consumed.load() << " tasks)\n";
}

// ============================================================
// Main
// ============================================================

int main() {
    std::cout << "\n=== PTO-RT Task Graph Infrastructure Tests ===\n\n";

    int passed = 0;
    int failed = 0;

    for (const auto& [name, func] : tests) {
        std::cout << "Running " << name << "...\n";
        try {
            func();
            passed++;
        } catch (const std::exception& e) {
            std::cout << "  FAILED: " << e.what() << "\n";
            failed++;
        } catch (...) {
            std::cout << "  FAILED: Unknown exception\n";
            failed++;
        }
    }

    std::cout << "\n=== Results: " << passed << " passed, " << failed << " failed ===\n";
    return failed > 0 ? 1 : 0;
}
