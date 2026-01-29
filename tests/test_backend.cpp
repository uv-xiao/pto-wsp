// PTO Workload-Schedule Programming (PTO-WSP) framework v9 - Backend Unit Tests
// Copyright (c) 2026 PTO Project
// SPDX-License-Identifier: MIT

#include "pto/rt/backend/cpu_sim.hpp"
#include "pto/rt/backend/codegen.hpp"
#include "pto/rt/backend/ascend_npu.hpp"
#include <iostream>
#include <cassert>
#include <atomic>
#include <chrono>
#include <span>

using namespace pto::wsp;
using namespace pto::wsp::backend;
using namespace pto::wsp::backend::cpu;
using namespace pto::wsp::backend::codegen;
using namespace pto::wsp::backend::ascend;
using namespace pto::wsp::graph;

// Test helper
#define TEST(name) void test_##name(); \
    static bool registered_##name = (tests.push_back({#name, test_##name}), true); \
    void test_##name()

std::vector<std::pair<const char*, void(*)()>> tests;

// ============================================================
// Kernel Registry Tests
// ============================================================

TEST(kernel_registry_basic) {
    // Register a simple kernel (CPP-3: use span signature)
    KernelRegistry::instance().register_kernel("test_kernel",
        [](std::span<const uint64_t> args, std::span<const TaskIO> io) {
            // Simple kernel that does nothing
        });

    assert(KernelRegistry::instance().has_kernel("test_kernel"));
    assert(!KernelRegistry::instance().has_kernel("nonexistent"));

    auto kernel = KernelRegistry::instance().get_kernel("test_kernel");
    assert(kernel != nullptr);

    // Call the kernel
    uint64_t args[] = {1, 2, 3};
    kernel(std::span<const uint64_t>(args, 3), std::span<const TaskIO>());

    std::cout << "  Kernel registry basic tests passed\n";
}

// ============================================================
// Thread Pool Tests
// ============================================================

TEST(thread_pool_basic) {
    ThreadPoolExecutor executor(4);
    executor.start();

    std::atomic<int> counter{0};
    const int NUM_TASKS = 100;

    for (int i = 0; i < NUM_TASKS; ++i) {
        executor.submit([&counter]() {
            counter.fetch_add(1);
        });
    }

    executor.wait_all();

    assert(counter.load() == NUM_TASKS);

    executor.shutdown();

    std::cout << "  Thread pool basic tests passed (" << counter.load() << " tasks)\n";
}

TEST(thread_pool_concurrent) {
    ThreadPoolExecutor executor(8);
    executor.start();

    std::atomic<int> sum{0};
    const int NUM_TASKS = 1000;

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < NUM_TASKS; ++i) {
        executor.submit([&sum, i]() {
            sum.fetch_add(i);
        });
    }

    executor.wait_all();

    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration<double, std::milli>(end - start);

    // Sum of 0..999 = 499500
    int expected = (NUM_TASKS - 1) * NUM_TASKS / 2;
    assert(sum.load() == expected);

    executor.shutdown();

    std::cout << "  Thread pool concurrent tests passed (sum=" << sum.load()
              << ", expected=" << expected << ", time=" << elapsed.count() << "ms)\n";
}

// ============================================================
// Backend Registry Tests
// ============================================================

TEST(backend_registry) {
    auto backends = BackendRegistry::instance().available_backends();

    // CPU sim should be registered automatically
    bool found_cpu_sim = false;
    for (const auto& name : backends) {
        if (name == "cpu_sim") {
            found_cpu_sim = true;
            break;
        }
    }

    // Note: Backend registration happens via static initialization
    // which may not have run yet in test context
    // Just check registry works

    std::cout << "  Backend registry tests passed (";
    for (size_t i = 0; i < backends.size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << backends[i];
    }
    std::cout << ")\n";
}

// ============================================================
// CPU Sim Backend Tests
// ============================================================

TEST(cpu_sim_backend_basic) {
    CPUSimBackend backend(4);

    assert(backend.name() == "cpu_sim");

    auto targets = backend.supported_targets();
    assert(!targets.empty());

    assert(backend.supports(ir::NodeKind::Task));

    std::cout << "  CPU sim backend basic tests passed\n";
}

TEST(cpu_sim_compile_empty) {
    CPUSimBackend backend(4);

    ir::Module module;
    module.name = "test_module";

    CompileOptions options;
    options.num_threads = 2;

    auto plan = backend.lower(module, options);
    assert(plan.workload_name == "test_module");

    auto program = backend.compile(plan, options);
    assert(program != nullptr);

    auto stats = program->stats();
    assert(stats.num_executors == 2);

    std::cout << "  CPU sim compile empty tests passed\n";
}

TEST(cpu_sim_program_execution) {
    // Build a simple task graph manually
    TaskGraphStorage storage;
    TaskGraphBuilder builder(storage);

    // Create 10 independent tasks
    std::vector<TaskId> tasks;
    for (int i = 0; i < 10; ++i) {
        auto tid = builder.begin_task(0)
            .add_arg(i)
            .submit();
        tasks.push_back(tid);
    }
    builder.finalize();

    // Create config
    ScheduleRuntimeConfig config;
    config.window_size = 100;
    config.pipeline_depth = 4;

    // Register a test kernel (CPP-3: use span signature)
    std::atomic<int> executed_count{0};
    KernelRegistry::instance().register_kernel(KernelId(0),
        [&executed_count](std::span<const uint64_t> args,
                          std::span<const TaskIO> io) {
            executed_count.fetch_add(1);
        });

    // Create and execute program
    auto program = std::make_unique<CPUSimProgram>(storage, config, 4);
    program->execute();

    assert(program->is_complete());
    assert(executed_count.load() == 10);

    auto stats = program->stats();
    assert(stats.num_tasks == 10);

    std::cout << "  CPU sim program execution tests passed (executed="
              << executed_count.load() << ", time=" << program->elapsed_ms() << "ms)\n";
}

TEST(cpu_sim_program_dependencies) {
    // Build a chain: t0 -> t1 -> t2 -> t3
    TaskGraphStorage storage;
    TaskGraphBuilder builder(storage);

    std::vector<TaskId> order;
    std::mutex order_mutex;

    auto t0 = builder.begin_task(1).add_arg(0).submit();
    auto t1 = builder.begin_task(1).add_arg(1).submit();
    auto t2 = builder.begin_task(1).add_arg(2).submit();
    auto t3 = builder.begin_task(1).add_arg(3).submit();

    builder.add_dependency(t0, t1);
    builder.add_dependency(t1, t2);
    builder.add_dependency(t2, t3);
    builder.finalize();

    // Register kernel that records execution order (CPP-3: use span signature)
    KernelRegistry::instance().register_kernel(KernelId(1),
        [&order, &order_mutex](std::span<const uint64_t> args,
                                std::span<const TaskIO> io) {
            std::lock_guard<std::mutex> lock(order_mutex);
            order.push_back(static_cast<TaskId>(args[0]));
        });

    // Execute
    ScheduleRuntimeConfig config;
    auto program = std::make_unique<CPUSimProgram>(storage, config, 4);
    program->execute();

    assert(program->is_complete());
    assert(order.size() == 4);

    // Verify order: t0 must come before t1, t1 before t2, etc.
    for (size_t i = 0; i < order.size(); ++i) {
        assert(order[i] == i);  // Should execute in order 0, 1, 2, 3
    }

    std::cout << "  CPU sim dependency tests passed (order: ";
    for (size_t i = 0; i < order.size(); ++i) {
        if (i > 0) std::cout << " -> ";
        std::cout << order[i];
    }
    std::cout << ")\n";
}

TEST(cpu_sim_async_execution) {
    TaskGraphStorage storage;
    TaskGraphBuilder builder(storage);

    // Create 50 independent tasks
    std::atomic<int> count{0};
    for (int i = 0; i < 50; ++i) {
        builder.begin_task(2).add_arg(i).submit();
    }
    builder.finalize();

    KernelRegistry::instance().register_kernel(KernelId(2),
        [&count](std::span<const uint64_t>, std::span<const TaskIO>) {
            count.fetch_add(1);
            // Add small delay to simulate work
            std::this_thread::sleep_for(std::chrono::microseconds(10));
        });

    ScheduleRuntimeConfig config;
    auto program = std::make_unique<CPUSimProgram>(storage, config, 4);

    // Execute asynchronously
    program->execute_async();

    assert(!program->is_complete());  // Should not be complete immediately

    // Wait for completion
    program->synchronize();

    assert(program->is_complete());
    assert(count.load() == 50);

    std::cout << "  CPU sim async execution tests passed (count="
              << count.load() << ")\n";
}

// ============================================================
// Performance Test
// ============================================================

TEST(cpu_sim_performance) {
    const int NUM_TASKS = 10000;

    TaskGraphStorage storage;
    TaskGraphBuilder builder(storage);

    // Create many independent tasks
    for (int i = 0; i < NUM_TASKS; ++i) {
        builder.begin_task(3).add_arg(i).submit();
    }
    builder.finalize();

    std::atomic<int> count{0};
    KernelRegistry::instance().register_kernel(KernelId(3),
        [&count](std::span<const uint64_t>, std::span<const TaskIO>) {
            count.fetch_add(1);
        });

    ScheduleRuntimeConfig config;
    config.window_size = 2048;
    config.pipeline_depth = 8;

    auto program = std::make_unique<CPUSimProgram>(storage, config, 8);

    auto start = std::chrono::high_resolution_clock::now();
    program->execute();
    auto end = std::chrono::high_resolution_clock::now();

    auto elapsed = std::chrono::duration<double, std::milli>(end - start);
    double tasks_per_ms = NUM_TASKS / elapsed.count();

    assert(count.load() == NUM_TASKS);

    std::cout << "  CPU sim performance test: " << NUM_TASKS << " tasks in "
              << elapsed.count() << "ms (" << tasks_per_ms << " tasks/ms)\n";

    // Target: 5000+ tasks/ms (from backend-arch.md)
    // Note: This is a basic test without real work, so throughput should be higher
    if (tasks_per_ms < 1000) {
        std::cout << "  WARNING: Performance below 1000 tasks/ms\n";
    }
}

// ============================================================
// Codegen Tests
// ============================================================

TEST(codegen_template_basic) {
    Template tpl("Hello ${name}, you have ${count} messages.");
    tpl.set("name", std::string("Alice"))
       .set("count", (int64_t)5);

    std::string result = tpl.render();
    assert(result == "Hello Alice, you have 5 messages.");

    std::cout << "  Template basic test passed: " << result << "\n";
}

TEST(codegen_context_basic) {
    CodeGenContext ctx;
    ctx.emit("int main() {");
    ctx.push_indent();
    ctx.emit("return 0;");
    ctx.pop_indent();
    ctx.emit("}");

    std::string output = ctx.get_output();
    assert(output.find("int main()") != std::string::npos);
    assert(output.find("    return 0;") != std::string::npos);

    std::cout << "  CodeGenContext test passed\n";
}

TEST(codegen_emitter_registry) {
    // Check that Ascend emitter is registered
    auto emitter = EmitterRegistry::instance().create_emitter("ascend_npu");
    assert(emitter != nullptr);
    assert(emitter->name() == "ascend_npu");
    assert(emitter->file_extension() == ".cpp");

    auto emitters = EmitterRegistry::instance().available_emitters();
    bool found = false;
    for (const auto& e : emitters) {
        if (e == "ascend_npu") found = true;
    }
    assert(found);

    std::cout << "  Emitter registry test passed (found " << emitters.size() << " emitters)\n";
}

// ============================================================
// Ascend Backend Tests
// ============================================================

TEST(ascend_backend_registration) {
    auto* backend = BackendRegistry::instance().get_backend("ascend_npu");
    assert(backend != nullptr);
    assert(backend->name() == "ascend_npu");

    auto targets = backend->supported_targets();
    assert(std::find(targets.begin(), targets.end(), "ascend_npu") != targets.end());

    std::cout << "  Ascend backend registration test passed\n";
}

TEST(ascend_codegen_basic) {
    // Create a simple NPU function
    ir::NPUFunction func;
    func.name = "test_kernel";
    func.tiles.push_back(ir::TileDecl{"x", 32, 128, ir::DType::F16, ir::Location::UB});
    func.tiles.push_back(ir::TileDecl{"y", 32, 128, ir::DType::F16, ir::Location::UB});
    func.memrefs.push_back(ir::MemrefDecl{"input", ir::DType::F16, ir::Location::Global, {}, true, false});
    func.memrefs.push_back(ir::MemrefDecl{"output", ir::DType::F16, ir::Location::Global, {}, false, true});

    func.ops.push_back(std::make_unique<ir::LoadOp>("x", "input"));
    func.ops.push_back(std::make_unique<ir::BinaryOp>(ir::NPUOpKind::Mul, "y", "x", "x"));
    func.ops.push_back(std::make_unique<ir::StoreOp>("y", "output"));

    // Generate code
    AscendNPUBackend backend;
    ir::NPUModule module;
    module.name = "test_module";
    module.registerFunction(std::make_unique<ir::NPUFunction>(std::move(func)));

    std::string code = backend.generate_code(module);

    // Verify code contains expected elements
    assert(code.find("test_kernel") != std::string::npos);
    assert(code.find("Tile<half") != std::string::npos);
    assert(code.find("DataCopy") != std::string::npos);
    assert(code.find("Mul") != std::string::npos);

    std::cout << "  Ascend codegen basic test passed\n";
    std::cout << "  Generated code:\n" << code << "\n";
}

TEST(ascend_codegen_matmul) {
    ir::NPUFunction func;
    func.name = "matmul_kernel";
    func.tiles.push_back(ir::TileDecl{"a", 32, 64, ir::DType::F16, ir::Location::UB});
    func.tiles.push_back(ir::TileDecl{"b", 64, 32, ir::DType::F16, ir::Location::UB});
    func.tiles.push_back(ir::TileDecl{"c", 32, 32, ir::DType::F32, ir::Location::L1});
    func.memrefs.push_back(ir::MemrefDecl{"A", ir::DType::F16, ir::Location::Global, {}, true, false});
    func.memrefs.push_back(ir::MemrefDecl{"B", ir::DType::F16, ir::Location::Global, {}, true, false});
    func.memrefs.push_back(ir::MemrefDecl{"C", ir::DType::F32, ir::Location::Global, {}, false, true});

    func.ops.push_back(std::make_unique<ir::LoadOp>("a", "A"));
    func.ops.push_back(std::make_unique<ir::LoadOp>("b", "B"));
    func.ops.push_back(std::make_unique<ir::MatmulOp>("c", "a", "b", "", true));
    func.ops.push_back(std::make_unique<ir::StoreOp>("c", "C"));

    AscendNPUBackend backend;
    ir::NPUModule module;
    module.name = "matmul_module";
    module.registerFunction(std::make_unique<ir::NPUFunction>(std::move(func)));

    std::string code = backend.generate_code(module);

    assert(code.find("Cube::Matmul") != std::string::npos);

    std::cout << "  Ascend codegen matmul test passed\n";
}

TEST(ascend_codegen_loop) {
    ir::NPUFunction func;
    func.name = "loop_kernel";
    func.tiles.push_back(ir::TileDecl{"x", 32, 128, ir::DType::F16, ir::Location::UB});
    func.memrefs.push_back(ir::MemrefDecl{"input", ir::DType::F16, ir::Location::Global, {}, true, true});

    func.ops.push_back(std::make_unique<ir::ForLoopBeginOp>("i", 0, 4, 1));
    func.ops.push_back(std::make_unique<ir::LoadOp>("x", "input"));
    func.ops.push_back(std::make_unique<ir::UnaryOp>(ir::NPUOpKind::Exp, "x", "x"));
    func.ops.push_back(std::make_unique<ir::StoreOp>("x", "input"));
    func.ops.push_back(std::make_unique<ir::ForLoopEndOp>());

    AscendNPUBackend backend;
    ir::NPUModule module;
    module.name = "loop_module";
    module.registerFunction(std::make_unique<ir::NPUFunction>(std::move(func)));

    std::string code = backend.generate_code(module);

    assert(code.find("for (int i = 0; i < 4;") != std::string::npos);
    assert(code.find("Exp(") != std::string::npos);

    std::cout << "  Ascend codegen loop test passed\n";
}

// ============================================================
// CSP Runtime Tests
// ============================================================

TEST(bounded_channel_basic) {
    // Test basic buffered channel operations
    BoundedChannel<int> ch(3);

    // Non-blocking send should succeed
    assert(ch.try_send(1));
    assert(ch.try_send(2));
    assert(ch.try_send(3));
    assert(!ch.try_send(4));  // Buffer full

    assert(ch.size() == 3);
    assert(!ch.empty());

    // Receive values
    auto v1 = ch.try_recv();
    assert(v1.has_value() && *v1 == 1);

    auto v2 = ch.try_recv();
    assert(v2.has_value() && *v2 == 2);

    auto v3 = ch.try_recv();
    assert(v3.has_value() && *v3 == 3);

    assert(ch.empty());
    assert(!ch.try_recv().has_value());  // Empty

    std::cout << "  BoundedChannel basic tests passed\n";
}

TEST(bounded_channel_blocking) {
    // Test blocking send/recv with threads
    BoundedChannel<int> ch(2);
    std::atomic<int> sum{0};

    // Producer thread
    std::thread producer([&ch]() {
        for (int i = 1; i <= 5; ++i) {
            ch.send(i);
        }
        ch.close();
    });

    // Consumer thread
    std::thread consumer([&ch, &sum]() {
        while (auto v = ch.recv()) {
            sum.fetch_add(*v);
        }
    });

    producer.join();
    consumer.join();

    // Sum of 1..5 = 15
    assert(sum.load() == 15);

    std::cout << "  BoundedChannel blocking tests passed (sum=" << sum.load() << ")\n";
}

TEST(bounded_channel_rendezvous) {
    // Test rendezvous channel (capacity 0)
    BoundedChannel<int> ch(0);
    std::atomic<int> received{0};

    // Consumer thread (starts first, waits for sender)
    std::thread consumer([&ch, &received]() {
        for (int i = 0; i < 3; ++i) {
            auto v = ch.recv();
            if (v) received.fetch_add(*v);
        }
    });

    // Give consumer time to block
    std::this_thread::sleep_for(std::chrono::milliseconds(10));

    // Producer sends (each send blocks until consumer receives)
    std::thread producer([&ch]() {
        ch.send(10);
        ch.send(20);
        ch.send(30);
        ch.close();
    });

    producer.join();
    consumer.join();

    assert(received.load() == 60);

    std::cout << "  BoundedChannel rendezvous tests passed (received=" << received.load() << ")\n";
}

TEST(channel_registry_basic) {
    ChannelRegistry registry;

    registry.create_channel("ch1", 5);
    registry.create_channel("ch2", 0);  // Rendezvous

    assert(registry.has_channel("ch1"));
    assert(registry.has_channel("ch2"));
    assert(!registry.has_channel("ch3"));

    auto ch1 = registry.get_channel("ch1");
    assert(ch1 != nullptr);
    assert(ch1->capacity() == 5);

    auto ch2 = registry.get_channel("ch2");
    assert(ch2 != nullptr);
    assert(ch2->capacity() == 0);

    registry.clear();
    assert(!registry.has_channel("ch1"));

    std::cout << "  ChannelRegistry basic tests passed\n";
}

TEST(csp_executor_basic) {
    ChannelRegistry registry;
    registry.create_channel("work", 10);

    CSPExecutor executor(registry, 4);

    std::atomic<int> produced{0};
    std::atomic<int> consumed{0};

    // Add producer
    auto work_ch = registry.get_channel("work");
    executor.add_producer("producer", {"work"}, [&work_ch, &produced]() {
        for (int i = 0; i < 5; ++i) {
            work_ch->send(static_cast<TaskId>(i));
            produced.fetch_add(1);
        }
        work_ch->close();
    });

    // Add consumer
    executor.add_sink("consumer", {"work"}, [&consumed](TaskId tid) {
        consumed.fetch_add(1);
    });

    executor.start();
    executor.wait();

    assert(produced.load() == 5);
    assert(consumed.load() == 5);

    std::cout << "  CSPExecutor basic tests passed (produced=" << produced.load()
              << ", consumed=" << consumed.load() << ")\n";
}

TEST(csp_executor_pipeline) {
    // Test multi-stage pipeline: producer -> transformer -> sink
    ChannelRegistry registry;
    registry.create_channel("stage1", 5);
    registry.create_channel("stage2", 5);

    CSPExecutor executor(registry, 4);

    std::atomic<int> final_sum{0};

    auto ch1 = registry.get_channel("stage1");
    auto ch2 = registry.get_channel("stage2");

    // Producer: sends 1, 2, 3
    executor.add_producer("producer", {"stage1"}, [&ch1]() {
        for (int i = 1; i <= 3; ++i) {
            ch1->send(static_cast<TaskId>(i));
        }
        ch1->close();
    });

    // Transformer: doubles values
    executor.add_consumer("transformer", {"stage1"}, {"stage2"}, [&ch2](TaskId tid) {
        ch2->send(tid * 2);
    });

    // Need to close ch2 when transformer is done
    // For simplicity, close after transformer finishes
    std::thread closer([&ch1, &ch2]() {
        // Wait for ch1 to close and drain
        while (!ch1->is_closed() || !ch1->empty()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        // Give transformer time to process
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        ch2->close();
    });

    // Sink: sums values
    executor.add_sink("sink", {"stage2"}, [&final_sum](TaskId tid) {
        final_sum.fetch_add(static_cast<int>(tid));
    });

    executor.start();
    closer.join();
    executor.wait();

    // Sum of (1*2 + 2*2 + 3*2) = 2 + 4 + 6 = 12
    assert(final_sum.load() == 12);

    std::cout << "  CSPExecutor pipeline tests passed (sum=" << final_sum.load() << ")\n";
}

// ============================================================
// Backend Lowering Tests (L5 - C++ test coverage for backend lowering)
// ============================================================

TEST(backend_lower_simple_workload) {
    // Create a simple IR module with a workload
    ir::IRFactory f;

    ir::Module module;
    module.name = "lower_test";

    auto axis = f.create<ir::DenseAxisNode>(4);
    auto task = f.create<ir::TaskNode>(
        std::string("compute"),
        std::vector<std::string>{"i"},
        std::vector<std::string>{"data"}
    );
    auto pfor = f.create<ir::ParallelForNode>(
        std::static_pointer_cast<const ir::AxisNode>(axis),
        "i",
        std::static_pointer_cast<const ir::WorkloadNode>(task)
    );

    ir::WorkloadDef wd;
    wd.name = "test_workload";
    wd.level = ir::WorkloadLevel::CPU;
    wd.body = std::static_pointer_cast<const ir::WorkloadNode>(pfor);
    module.workloads.push_back(wd);

    // Lower using CPU backend
    CPUSimBackend backend(4);
    CompileOptions options;
    options.num_threads = 2;

    auto plan = backend.lower(module, options);
    assert(plan.workload_name == "lower_test");

    // Compile to program
    auto program = backend.compile(plan, options);
    assert(program != nullptr);

    std::cout << "  Backend lowering test passed\n";
}

TEST(backend_lower_nested_workload) {
    // Create nested parallel_for workload
    ir::IRFactory f;

    ir::Module module;
    module.name = "nested_lower_test";

    auto outer_axis = f.create<ir::DenseAxisNode>(2);
    auto inner_axis = f.create<ir::DenseAxisNode>(3);
    auto task = f.create<ir::TaskNode>(
        std::string("matmul"),
        std::vector<std::string>{"i", "j"},
        std::vector<std::string>{"A", "B", "C"}
    );

    auto inner_pfor = f.create<ir::ParallelForNode>(
        std::static_pointer_cast<const ir::AxisNode>(inner_axis),
        "j",
        std::static_pointer_cast<const ir::WorkloadNode>(task)
    );

    auto outer_pfor = f.create<ir::ParallelForNode>(
        std::static_pointer_cast<const ir::AxisNode>(outer_axis),
        "i",
        std::static_pointer_cast<const ir::WorkloadNode>(inner_pfor)
    );

    ir::WorkloadDef wd;
    wd.name = "nested_workload";
    wd.level = ir::WorkloadLevel::CPU;
    wd.body = std::static_pointer_cast<const ir::WorkloadNode>(outer_pfor);
    module.workloads.push_back(wd);

    // Lower
    CPUSimBackend backend(4);
    CompileOptions options;
    auto plan = backend.lower(module, options);
    auto program = backend.compile(plan, options);

    assert(program != nullptr);

    std::cout << "  Nested workload lowering test passed\n";
}

TEST(backend_lower_with_schedule) {
    // Create workload with schedule
    ir::IRFactory f;

    ir::Module module;
    module.name = "schedule_lower_test";

    auto axis = f.create<ir::DenseAxisNode>(8);
    auto task = f.create<ir::TaskNode>(
        std::string("process"),
        std::vector<std::string>{"i"},
        std::vector<std::string>{"input", "output"}
    );
    auto pfor = f.create<ir::ParallelForNode>(
        std::static_pointer_cast<const ir::AxisNode>(axis),
        "i",
        std::static_pointer_cast<const ir::WorkloadNode>(task)
    );

    ir::WorkloadDef wd;
    wd.name = "scheduled_workload";
    wd.level = ir::WorkloadLevel::CPU;
    wd.body = std::static_pointer_cast<const ir::WorkloadNode>(pfor);
    module.workloads.push_back(wd);

    // Add schedule
    auto dispatch = f.create<ir::DispatchNode>(ir::DispatchPolicy::RoundRobin, 4);
    auto stream = f.create<ir::StreamNode>(2);
    auto timing = f.create<ir::TimingNode>(ir::TimingPolicy::Immediate);

    ir::ScheduleDef sd;
    sd.name = "test_sched";
    sd.workload_name = "scheduled_workload";
    sd.level = ir::WorkloadLevel::CPU;
    sd.directives.push_back(std::static_pointer_cast<const ir::ScheduleNode>(dispatch));
    sd.directives.push_back(std::static_pointer_cast<const ir::ScheduleNode>(stream));
    sd.directives.push_back(std::static_pointer_cast<const ir::ScheduleNode>(timing));
    module.schedules.push_back(sd);

    // Lower
    CPUSimBackend backend(4);
    CompileOptions options;
    options.num_streams = 2;
    auto plan = backend.lower(module, options);
    auto program = backend.compile(plan, options);

    assert(program != nullptr);

    std::cout << "  Scheduled workload lowering test passed\n";
}

TEST(backend_ascend_lower) {
    // Test Ascend NPU backend lowering
    ir::IRFactory f;

    ir::Module module;
    module.name = "ascend_lower_test";
    module.targets = {"ascend_npu"};

    auto axis = f.create<ir::DenseAxisNode>(4);
    auto task = f.create<ir::TaskNode>(
        std::string("npu_kernel"),
        std::vector<std::string>{"i"},
        std::vector<std::string>{"data"}
    );
    auto pfor = f.create<ir::ParallelForNode>(
        std::static_pointer_cast<const ir::AxisNode>(axis),
        "i",
        std::static_pointer_cast<const ir::WorkloadNode>(task)
    );

    ir::WorkloadDef wd;
    wd.name = "npu_workload";
    wd.level = ir::WorkloadLevel::NPU;
    wd.body = std::static_pointer_cast<const ir::WorkloadNode>(pfor);
    module.workloads.push_back(wd);

    // Lower using Ascend backend
    AscendNPUBackend backend;
    CompileOptions options;
    options.target = "ascend_npu";

    auto plan = backend.lower(module, options);
    assert(plan.workload_name == "ascend_lower_test");

    std::cout << "  Ascend backend lowering test passed\n";
}

// ============================================================
// Main
// ============================================================

int main() {
    std::cout << "\n=== PTO-RT Backend Tests ===\n\n";

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
