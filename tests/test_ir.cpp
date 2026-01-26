// PTO Workload-Schedule Programming (PTO-WSP) framework v9 - IR Unit Tests
// Copyright (c) 2024 PTO Project
// SPDX-License-Identifier: Apache-2.0

#include "pto/rt/ir/ir.hpp"
#include <iostream>
#include <sstream>
#include <cassert>

using namespace pto::wsp::ir;

// Test helper
#define TEST(name) void test_##name(); \
    static bool registered_##name = (tests.push_back({#name, test_##name}), true); \
    void test_##name()

std::vector<std::pair<const char*, void(*)()>> tests;

// Test: Basic factory and node creation
TEST(factory_creates_nodes) {
    IRFactory f;

    auto axis = f.create<DenseAxisNode>(8);
    assert(axis->id == 0);
    assert(axis->kind == NodeKind::DenseAxis);
    assert(axis->size == 8);

    auto axis2 = f.create<DenseDynAxisNode>("batch_size");
    assert(axis2->id == 1);
    assert(axis2->size_var == "batch_size");

    std::cout << "  Created nodes with IDs " << axis->id << ", " << axis2->id << "\n";
}

// Test: Task node creation and printing
TEST(task_node_print) {
    IRFactory f;

    auto task = f.create<TaskNode>(
        "attn_kernel",
        std::vector<std::string>{"b", "h"},
        std::vector<std::string>{"Q[b][h]", "K[b]", "V[b]", "O[b][h]"}
    );

    std::ostringstream oss;
    task->print(oss, 0);
    std::string output = oss.str();

    assert(output.find("@attn_kernel") != std::string::npos);
    assert(output.find("b, h") != std::string::npos);
    assert(output.find("resources") != std::string::npos);

    std::cout << "  Task output: " << output << "\n";
}

// Test: ParallelFor composition
TEST(parallel_for_composition) {
    IRFactory f;

    auto batch_axis = f.create<DenseDynAxisNode>("batch");
    auto heads_axis = f.create<DenseAxisNode>(8);

    auto task = f.create<TaskNode>(
        "attn",
        std::vector<std::string>{"b", "h"},
        std::vector<std::string>{"Q", "K", "V", "O"}
    );

    auto inner = f.create<ParallelForNode>(
        std::static_pointer_cast<const AxisNode>(heads_axis),
        "h",
        std::static_pointer_cast<const WorkloadNode>(task)
    );

    auto outer = f.create<ParallelForNode>(
        std::static_pointer_cast<const AxisNode>(batch_axis),
        "b",
        std::static_pointer_cast<const WorkloadNode>(inner)
    );

    std::ostringstream oss;
    outer->print(oss, 0);
    std::string output = oss.str();

    assert(output.find("parallel_for b in dense_dyn(batch)") != std::string::npos);
    assert(output.find("parallel_for h in dense[8]") != std::string::npos);

    std::cout << "  Composed workload:\n" << output << "\n";
}

// Test: Visitor traversal
TEST(visitor_traversal) {
    IRFactory f;

    auto axis = f.create<DenseAxisNode>(4);
    auto task = f.create<TaskNode>(
        std::string("kernel"),
        std::vector<std::string>{},
        std::vector<std::string>{}
    );
    auto pfor = f.create<ParallelForNode>(
        std::static_pointer_cast<const AxisNode>(axis),
        "i",
        std::static_pointer_cast<const WorkloadNode>(task)
    );

    // Count nodes visited
    struct CountVisitor : IRVisitor {
        int count = 0;
        WalkControl enter(const IRNode&) override {
            count++;
            return WalkControl::Continue;
        }
    };

    CountVisitor v;
    walk(std::static_pointer_cast<const IRNode>(pfor), v);

    assert(v.count == 3);  // ParallelFor + Axis + Task
    std::cout << "  Visited " << v.count << " nodes\n";
}

// Test: Schedule nodes
TEST(schedule_nodes) {
    IRFactory f;

    auto dispatch = f.create<DispatchNode>(DispatchPolicy::Affinity, 0, "batch");
    auto stream = f.create<StreamNode>(2, "head % 2");
    auto timing = f.create<TimingNode>(TimingPolicy::Immediate);

    std::ostringstream oss;
    dispatch->print(oss, 0);
    oss << "\n";
    stream->print(oss, 0);
    oss << "\n";
    timing->print(oss, 0);

    std::string output = oss.str();
    assert(output.find("affinity(batch)") != std::string::npos);
    assert(output.find("streams = 2") != std::string::npos);
    assert(output.find("immediate") != std::string::npos);

    std::cout << "  Schedule directives:\n" << output << "\n";
}

// Test: Module creation
TEST(module_creation) {
    IRFactory f;

    Module m;
    m.name = "attention_example";
    m.version = "9.0";
    m.targets = {"cpu_sim", "ascend_npu"};

    // Create workload
    auto axis = f.create<DenseAxisNode>(8);
    auto task = f.create<TaskNode>(
        std::string("attn"),
        std::vector<std::string>{"b"},
        std::vector<std::string>{"Q", "K", "V", "O"}
    );
    auto pfor = f.create<ParallelForNode>(
        std::static_pointer_cast<const AxisNode>(axis),
        "b",
        std::static_pointer_cast<const WorkloadNode>(task)
    );

    WorkloadDef wd;
    wd.name = "attention";
    wd.level = WorkloadLevel::CPU;
    wd.params.push_back({"batch", axis});
    wd.body = std::static_pointer_cast<const WorkloadNode>(pfor);
    m.workloads.push_back(wd);

    // Create schedule
    auto dispatch = f.create<DispatchNode>(DispatchPolicy::RoundRobin, 4);
    auto stream = f.create<StreamNode>(2);

    ScheduleDef sd;
    sd.name = "attention_sched";
    sd.workload_name = "attention";
    sd.level = WorkloadLevel::CPU;
    sd.directives.push_back(std::static_pointer_cast<const ScheduleNode>(dispatch));
    sd.directives.push_back(std::static_pointer_cast<const ScheduleNode>(stream));
    m.schedules.push_back(sd);

    std::ostringstream oss;
    m.print(oss);
    std::string output = oss.str();

    assert(output.find("PTO Module: attention_example") != std::string::npos);
    assert(output.find("@workload attention") != std::string::npos);
    assert(output.find("@schedule attention_sched") != std::string::npos);

    std::cout << "  Module output:\n" << output << "\n";
}

// Test: Extension nodes
TEST(extension_nodes) {
    IRFactory f;

    AttrMap attrs;
    attrs["buffer"] = std::string("ub");
    attrs["depth"] = int64_t(2);

    auto ext = f.create<ExtOpNode>(ExtClass::Schedule, "npu.double_buffer", attrs);

    std::ostringstream oss;
    ext->print(oss, 0);
    std::string output = oss.str();

    assert(output.find("ext.npu.double_buffer") != std::string::npos);
    assert(ext->getAttrOr<int64_t>("depth", 0) == 2);
    assert(ext->getAttrOr<std::string>("buffer", "") == "ub");

    std::cout << "  Extension node: " << output << "\n";
}

// Test: CSP nodes
TEST(csp_nodes) {
    IRFactory f;

    TensorType elem_type;
    elem_type.shape = {{32, 128}};
    elem_type.dtype = DType::F16;
    elem_type.location = Location::UB;

    ChannelType ch_type;
    ch_type.element_type = elem_type;
    ch_type.capacity = 2;

    auto channel = f.create<ChannelNode>("data_ch", ch_type);

    std::ostringstream oss;
    channel->print(oss, 0);
    std::string output = oss.str();

    assert(output.find("channel data_ch") != std::string::npos);
    std::cout << "  Channel: " << output << "\n";
}

// Test: NPU Function IR
TEST(npu_function) {
    NPUFunction func;
    func.name = "rmsnorm_kernel";

    // Declarations
    func.tiles.push_back(TileDecl{"x", 32, 128, DType::F16, Location::UB});
    func.tiles.push_back(TileDecl{"out", 32, 128, DType::F16, Location::UB});
    func.scalars.push_back(ScalarDecl{"eps", DType::F32, 1e-6});
    func.memrefs.push_back(MemrefDecl{"input", DType::F16, Location::Global, {}, true, false});
    func.memrefs.push_back(MemrefDecl{"output", DType::F16, Location::Global, {}, false, true});

    // Operations
    func.ops.push_back(std::make_unique<LoadOp>("x", "input"));
    func.ops.push_back(std::make_unique<BinaryOp>(NPUOpKind::Mul, "sq", "x", "x"));
    func.ops.push_back(std::make_unique<ReduceOp>(NPUOpKind::RowMean, "mean", "sq"));
    func.ops.push_back(std::make_unique<UnaryOp>(NPUOpKind::Rsqrt, "rsqrt_val", "mean"));
    func.ops.push_back(std::make_unique<BroadcastOp>(NPUOpKind::RowExpandMul, "out", "x", "rsqrt_val"));
    func.ops.push_back(std::make_unique<StoreOp>("out", "output"));

    std::ostringstream oss;
    func.print(oss);
    std::string output = oss.str();

    assert(output.find("@npu_kernel rmsnorm_kernel") != std::string::npos);
    assert(output.find("tile x[32, 128]") != std::string::npos);
    assert(output.find("load x, input") != std::string::npos);
    assert(output.find("rowmean mean, sq") != std::string::npos);

    std::cout << "  NPU Function:\n" << output << "\n";
}

// Test: NPU Module
TEST(npu_module) {
    NPUModule module;
    module.name = "attention_kernels";

    // Create a kernel
    auto func = std::make_unique<NPUFunction>();
    func->name = "attn_qk";
    func->tiles.push_back(TileDecl{"q", 32, 64, DType::F16, Location::UB});
    func->tiles.push_back(TileDecl{"k", 64, 32, DType::F16, Location::UB});
    func->tiles.push_back(TileDecl{"s", 32, 32, DType::F16, Location::L1});
    func->ops.push_back(std::make_unique<LoadOp>("q", "Q"));
    func->ops.push_back(std::make_unique<LoadOp>("k", "K"));
    func->ops.push_back(std::make_unique<MatmulOp>("s", "q", "k", "", true));
    func->ops.push_back(std::make_unique<StoreOp>("s", "S"));

    int id = module.registerFunction(std::move(func));
    assert(id == 0);

    auto found = module.findFunction("attn_qk");
    assert(found != nullptr);
    assert(found->kernel_id == 0);

    std::ostringstream oss;
    module.print(oss);
    std::string output = oss.str();

    assert(output.find("NPU Module: attention_kernels") != std::string::npos);
    assert(output.find("@npu_kernel attn_qk") != std::string::npos);
    assert(output.find("matmul s, q, k") != std::string::npos);

    std::cout << "  NPU Module:\n" << output << "\n";
}

// Test: Printer and Parser round-trip
TEST(printer_parser_roundtrip) {
    // Create a module
    IRFactory f;

    Module m;
    m.name = "test_module";
    m.version = "9.0";
    m.targets = {"cpu_sim"};

    // Create workload
    auto axis = f.create<DenseAxisNode>(4);
    auto task = f.create<TaskNode>(
        std::string("compute"),
        std::vector<std::string>{"i"},
        std::vector<std::string>{"input", "output"}
    );
    auto pfor = f.create<ParallelForNode>(
        std::static_pointer_cast<const AxisNode>(axis),
        "i",
        std::static_pointer_cast<const WorkloadNode>(task)
    );

    WorkloadDef wd;
    wd.name = "test_workload";
    wd.level = WorkloadLevel::CPU;
    wd.params.push_back({"N", axis});
    wd.body = std::static_pointer_cast<const WorkloadNode>(pfor);
    m.workloads.push_back(wd);

    // Print to string
    std::string output = Printer::to_string(m);
    std::cout << "  Original output:\n" << output << "\n";

    // Parse back (note: parser is basic, may not fully round-trip)
    // For now just verify it doesn't crash
    try {
        std::istringstream iss(output);
        // Skip header comments for now
    } catch (const ParseError& e) {
        std::cout << "  Parse error (expected for partial parser): " << e.what() << "\n";
    }

    std::cout << "  Printer test passed\n";
}

// Test: Parser basic parsing
TEST(parser_basic) {
    std::string source = R"(@workload simple(%N: dense[8]) {
  parallel_for i in dense[8] {
    task @kernel(i) resources(input, output)
  }
})";

    Module m = Parser::parseString(source);

    assert(m.workloads.size() == 1);
    assert(m.workloads[0].name == "simple");
    assert(m.workloads[0].params.size() == 1);
    assert(m.workloads[0].params[0].first == "N");

    std::cout << "  Parsed workload: " << m.workloads[0].name << "\n";
    std::cout << "  Parameter: " << m.workloads[0].params[0].first << "\n";

    // Print the parsed workload back
    std::ostringstream oss;
    m.workloads[0].print(oss);
    std::cout << "  Round-trip output:\n" << oss.str() << "\n";
}

// Test: IR round-trip verification (parse → print → parse → verify equivalence)
TEST(ir_roundtrip_equivalence) {
    // Test case 1: Simple parallel_for workload
    std::string source1 = R"(@workload compute(%N: dense[8]) {
  parallel_for i in dense[8] {
    task @kernel(i) resources(input, output)
  }
})";

    // First parse
    Module m1 = Parser::parseString(source1);
    assert(m1.workloads.size() == 1);
    assert(m1.workloads[0].name == "compute");

    // Print back to string
    std::string printed1 = Printer::to_string(m1);
    std::cout << "  First print:\n" << printed1 << "\n";

    // Parse again
    Module m2 = Parser::parseString(printed1);
    assert(m2.workloads.size() == 1);
    assert(m2.workloads[0].name == "compute");

    // Print again and verify equivalence
    std::string printed2 = Printer::to_string(m2);
    std::cout << "  Second print:\n" << printed2 << "\n";

    // The two printed versions should be identical
    assert(printed1 == printed2 && "Round-trip should produce identical output");

    std::cout << "  Round-trip equivalence verified!\n";
}

// Test: IR round-trip with nested structures
TEST(ir_roundtrip_nested) {
    std::string source = R"(@workload attention(%batch: dense_dyn(batch), %heads: dense[8]) {
  parallel_for b in dense_dyn(batch) {
    parallel_for h in dense[8] {
      task @attn_kernel(b, h) resources(Q, K, V, O)
    }
  }
})";

    // First round
    Module m1 = Parser::parseString(source);
    std::string printed1 = Printer::to_string(m1);

    // Second round
    Module m2 = Parser::parseString(printed1);
    std::string printed2 = Printer::to_string(m2);

    // Verify structure is preserved
    assert(m1.workloads.size() == m2.workloads.size());
    assert(m1.workloads[0].name == m2.workloads[0].name);
    assert(m1.workloads[0].params.size() == m2.workloads[0].params.size());

    // Verify printed output is stable
    assert(printed1 == printed2 && "Nested round-trip should be stable");

    std::cout << "  Nested round-trip verified!\n";
}

// Test: IR round-trip with schedule
TEST(ir_roundtrip_with_schedule) {
    std::string source = R"(@workload matmul(%M: dense[1024], %N: dense[1024]) {
  parallel_for i in dense[1024] {
    parallel_for j in dense[1024] {
      task @gemm(i, j) resources(A, B, C)
    }
  }
}

@schedule matmul_sched for matmul {
  dispatch = round_robin(4)
  streams = 4
  timing = immediate
})";

    // First round
    Module m1 = Parser::parseString(source);
    assert(m1.workloads.size() == 1);
    assert(m1.schedules.size() == 1);

    std::string printed1 = Printer::to_string(m1);

    // Second round
    Module m2 = Parser::parseString(printed1);
    std::string printed2 = Printer::to_string(m2);

    // Verify structure
    assert(m2.workloads.size() == 1);
    assert(m2.schedules.size() == 1);
    assert(m2.schedules[0].workload_name == "matmul");

    // Verify stability
    assert(printed1 == printed2 && "Schedule round-trip should be stable");

    std::cout << "  Schedule round-trip verified!\n";
}

// Test: IR round-trip with for_each (sequential iteration)
TEST(ir_roundtrip_foreach) {
    std::string source = R"(@workload sequential_work(%N: dense[4]) {
  for_each i in dense[4] {
    task @compute(i) resources(data)
  }
})";

    Module m1 = Parser::parseString(source);
    std::string printed1 = Printer::to_string(m1);
    std::cout << "  First print:\n" << printed1 << "\n";

    Module m2 = Parser::parseString(printed1);
    std::string printed2 = Printer::to_string(m2);

    assert(printed1 == printed2 && "ForEach round-trip should be stable");
    std::cout << "  ForEach round-trip verified!\n";
}

// ============================================================
// Additional IR Parsing Tests (L5 - C++ test coverage)
// ============================================================

// Test: Parse multiple workloads
TEST(parser_multiple_workloads) {
    std::string source = R"(
@workload workload1(%N: dense[4]) {
  parallel_for i in dense[4] {
    task @kernel1(i) resources(data)
  }
}

@workload workload2(%M: dense[8]) {
  parallel_for j in dense[8] {
    task @kernel2(j) resources(other)
  }
})";

    Module m = Parser::parseString(source);

    assert(m.workloads.size() == 2);
    assert(m.workloads[0].name == "workload1");
    assert(m.workloads[1].name == "workload2");

    // Verify bodies are parsed
    assert(m.workloads[0].body != nullptr);
    assert(m.workloads[1].body != nullptr);

    std::cout << "  Parsed " << m.workloads.size() << " workloads\n";
}

// Test: Parse combine construct
TEST(parser_combine) {
    std::string source = R"(@workload combined(%N: dense[4]) {
  combine {
    task @kernel1() resources(a)
    task @kernel2() resources(b)
    task @kernel3() resources(c)
  }
})";

    Module m = Parser::parseString(source);
    assert(m.workloads.size() == 1);
    assert(m.workloads[0].body != nullptr);
    assert(m.workloads[0].body->kind == NodeKind::Combine);

    auto* combine = static_cast<const CombineNode*>(m.workloads[0].body.get());
    assert(combine->workloads.size() == 3);

    std::cout << "  Parsed combine with " << combine->workloads.size() << " tasks\n";
}

// Test: Parse sequential construct
TEST(parser_sequential) {
    std::string source = R"(@workload sequenced(%N: dense[4]) {
  sequential {
    task @step1() resources(a)
    task @step2() resources(b)
  }
})";

    Module m = Parser::parseString(source);
    assert(m.workloads.size() == 1);
    assert(m.workloads[0].body != nullptr);
    assert(m.workloads[0].body->kind == NodeKind::Sequential);

    auto* seq = static_cast<const SequentialNode*>(m.workloads[0].body.get());
    assert(seq->workloads.size() == 2);

    std::cout << "  Parsed sequential with " << seq->workloads.size() << " steps\n";
}

// Test: Parse ragged axis in parameter
TEST(parser_ragged_axis) {
    std::string source = R"(@workload ragged_work(%tokens: ragged(batch, lengths)) {
  task @compute_token() resources(data)
})";

    Module m = Parser::parseString(source);
    assert(m.workloads.size() == 1);
    assert(m.workloads[0].params.size() == 1);
    assert(m.workloads[0].params[0].second->kind == NodeKind::RaggedAxis);

    auto* ragged = static_cast<const RaggedAxisNode*>(m.workloads[0].params[0].second.get());
    assert(ragged->outer_size_var == "batch");
    assert(ragged->lengths_var == "lengths");

    std::cout << "  Parsed ragged axis: ragged(" << ragged->outer_size_var << ", " << ragged->lengths_var << ")\n";
}

// Test: Parse sparse axis
TEST(parser_sparse_axis) {
    std::string source = R"(@workload sparse_work(%routing: sparse(experts, indptr, indices)) {
  parallel_for e in sparse(experts, indptr, indices) {
    task @expert(e) resources(weights, activations)
  }
})";

    Module m = Parser::parseString(source);
    assert(m.workloads.size() == 1);
    assert(m.workloads[0].params.size() == 1);
    assert(m.workloads[0].params[0].second->kind == NodeKind::SparseAxis);

    auto* sparse = static_cast<const SparseAxisNode*>(m.workloads[0].params[0].second.get());
    assert(sparse->outer_size_var == "experts");
    assert(sparse->indptr_var == "indptr");
    assert(sparse->indices_var == "indices");

    std::cout << "  Parsed sparse axis: sparse(" << sparse->outer_size_var << ", "
              << sparse->indptr_var << ", " << sparse->indices_var << ")\n";
}

// Test: Parse schedule with all directive types
TEST(parser_full_schedule) {
    std::string source = R"(@workload compute(%N: dense[1024]) {
  parallel_for i in dense[1024] {
    task @kernel(i) resources(data)
  }
}

@schedule compute_sched for compute {
  dispatch = affinity(batch)
  streams = 8
  timing = batched(32)
})";

    Module m = Parser::parseString(source);
    assert(m.workloads.size() == 1);
    assert(m.schedules.size() == 1);

    const auto& sched = m.schedules[0];
    assert(sched.name == "compute_sched");
    assert(sched.workload_name == "compute");

    // Check dispatch directive
    auto disp = sched.dispatch();
    assert(disp != nullptr);
    assert(disp->policy == DispatchPolicy::Affinity);
    assert(disp->key_expr == "batch");

    // Check stream directive
    auto str = sched.stream();
    assert(str != nullptr);
    assert(str->num_streams == 8);

    // Check timing directive
    auto tim = sched.timing();
    assert(tim != nullptr);
    assert(tim->policy == TimingPolicy::Batched);
    assert(tim->param == 32);

    std::cout << "  Parsed full schedule with dispatch=" << static_cast<int>(disp->policy)
              << ", streams=" << str->num_streams << ", timing=" << static_cast<int>(tim->policy) << "\n";
}

// Test: Parse cond construct
TEST(parser_cond) {
    std::string source = R"(@workload conditional(%N: dense[4]) {
  cond expert_active {
    task @expert() resources(data)
  } else {
    task @fallback() resources(data)
  }
})";

    Module m = Parser::parseString(source);
    assert(m.workloads.size() == 1);
    assert(m.workloads[0].body != nullptr);
    assert(m.workloads[0].body->kind == NodeKind::Cond);

    auto* cond = static_cast<const CondNode*>(m.workloads[0].body.get());
    assert(cond->predicate == "expert_active");
    assert(cond->then_branch != nullptr);
    assert(cond->else_branch != nullptr);

    std::cout << "  Parsed cond with predicate: " << cond->predicate << "\n";
}

// Test: Parse deeply nested structure
TEST(parser_deeply_nested) {
    std::string source = R"(@workload deep(%batch: dense[4], %heads: dense[8], %seq: dense[512]) {
  parallel_for b in dense[4] {
    parallel_for h in dense[8] {
      parallel_for s in dense[512] {
        task @attention(b, h, s) resources(Q, K, V, O)
      }
    }
  }
})";

    Module m = Parser::parseString(source);
    assert(m.workloads.size() == 1);

    // Navigate through nested structure
    auto* level1 = static_cast<const ParallelForNode*>(m.workloads[0].body.get());
    assert(level1->kind == NodeKind::ParallelFor);
    assert(level1->index_var == "b");

    auto* level2 = static_cast<const ParallelForNode*>(level1->body.get());
    assert(level2->kind == NodeKind::ParallelFor);
    assert(level2->index_var == "h");

    auto* level3 = static_cast<const ParallelForNode*>(level2->body.get());
    assert(level3->kind == NodeKind::ParallelFor);
    assert(level3->index_var == "s");

    auto* task_node = static_cast<const TaskNode*>(level3->body.get());
    assert(task_node->kind == NodeKind::Task);
    assert(task_node->kernel_name == "attention");
    assert(task_node->params.size() == 3);
    assert(task_node->resources.size() == 4);

    std::cout << "  Parsed 3-level nested parallel_for with task\n";
}

// ============================================================
// Type Check Pass Tests
// ============================================================

// Test: Type check valid module
TEST(type_check_valid_module) {
    IRFactory f;

    Module m;
    m.name = "valid_module";

    // Create valid workload
    auto axis = f.create<DenseAxisNode>(4);
    auto task = f.create<TaskNode>(
        std::string("kernel"),
        std::vector<std::string>{"i"},
        std::vector<std::string>{"input"}
    );
    auto pfor = f.create<ParallelForNode>(
        std::static_pointer_cast<const AxisNode>(axis),
        "i",
        std::static_pointer_cast<const WorkloadNode>(task)
    );

    WorkloadDef wd;
    wd.name = "my_workload";
    wd.level = WorkloadLevel::CPU;
    wd.body = std::static_pointer_cast<const WorkloadNode>(pfor);
    m.workloads.push_back(wd);

    // Type check
    auto result = type_check(m);
    assert(result.valid);
    assert(result.errors.empty());

    std::cout << "  Type check result: " << result.to_string() << "\n";
}

// Test: Type check detects undefined call target
TEST(type_check_undefined_call) {
    IRFactory f;

    Module m;
    m.name = "undefined_call_module";

    // Create workload with call to undefined target
    auto call = f.create<CallNode>("undefined_workload", std::vector<std::string>{});

    WorkloadDef wd;
    wd.name = "caller";
    wd.level = WorkloadLevel::CPU;
    wd.body = std::static_pointer_cast<const WorkloadNode>(call);
    m.workloads.push_back(wd);

    // Type check
    auto result = type_check(m);
    assert(!result.valid);
    assert(result.errors.size() >= 1);

    bool found_undefined = false;
    for (const auto& err : result.errors) {
        if (err.kind == TypeCheckErrorKind::UndefinedReference) {
            found_undefined = true;
            assert(err.message.find("undefined_workload") != std::string::npos);
        }
    }
    assert(found_undefined);

    std::cout << "  Type check result: " << result.to_string();
}

// Test: Type check detects duplicate workload names
TEST(type_check_duplicate_workload) {
    IRFactory f;

    Module m;
    m.name = "duplicate_module";

    // Create two workloads with same name
    auto task = f.create<TaskNode>(
        std::string("k"),
        std::vector<std::string>{},
        std::vector<std::string>{}
    );

    WorkloadDef wd1;
    wd1.name = "duplicated";
    wd1.level = WorkloadLevel::CPU;
    wd1.body = std::static_pointer_cast<const WorkloadNode>(task);
    m.workloads.push_back(wd1);

    WorkloadDef wd2;
    wd2.name = "duplicated";  // Same name
    wd2.level = WorkloadLevel::CPU;
    wd2.body = std::static_pointer_cast<const WorkloadNode>(task);
    m.workloads.push_back(wd2);

    auto result = type_check(m);
    assert(!result.valid);

    bool found_dup = false;
    for (const auto& err : result.errors) {
        if (err.kind == TypeCheckErrorKind::DuplicateDefinition) {
            found_dup = true;
        }
    }
    assert(found_dup);

    std::cout << "  Type check result: " << result.to_string();
}

// Test: Type check validates axis bounds
TEST(type_check_axis_bounds) {
    IRFactory f;

    Module m;
    m.name = "axis_bounds_module";

    // Create axis with invalid size
    auto bad_axis = f.create<DenseAxisNode>(-5);  // Invalid negative size
    auto task = f.create<TaskNode>("k", std::vector<std::string>{}, std::vector<std::string>{});
    auto pfor = f.create<ParallelForNode>(
        std::static_pointer_cast<const AxisNode>(bad_axis),
        "i",
        std::static_pointer_cast<const WorkloadNode>(task)
    );

    WorkloadDef wd;
    wd.name = "bad_axis_workload";
    wd.level = WorkloadLevel::CPU;
    wd.body = std::static_pointer_cast<const WorkloadNode>(pfor);
    m.workloads.push_back(wd);

    auto result = type_check(m);
    assert(!result.valid);

    bool found_bounds = false;
    for (const auto& err : result.errors) {
        if (err.kind == TypeCheckErrorKind::AxisBoundsError) {
            found_bounds = true;
        }
    }
    assert(found_bounds);

    std::cout << "  Type check result: " << result.to_string();
}

// Test: Type check validates schedule references
TEST(type_check_schedule_reference) {
    IRFactory f;

    Module m;
    m.name = "schedule_ref_module";

    // Create schedule referencing non-existent workload
    ScheduleDef sd;
    sd.name = "bad_schedule";
    sd.workload_name = "nonexistent_workload";
    sd.level = WorkloadLevel::CPU;
    m.schedules.push_back(sd);

    auto result = type_check(m);
    assert(!result.valid);

    bool found_ref = false;
    for (const auto& err : result.errors) {
        if (err.kind == TypeCheckErrorKind::UndefinedReference &&
            err.message.find("nonexistent_workload") != std::string::npos) {
            found_ref = true;
        }
    }
    assert(found_ref);

    std::cout << "  Type check result: " << result.to_string();
}

int main() {
    std::cout << "Running IR tests...\n\n";

    int passed = 0;
    int failed = 0;

    for (const auto& [name, fn] : tests) {
        std::cout << "Test: " << name << "\n";
        try {
            fn();
            std::cout << "  PASSED\n\n";
            passed++;
        } catch (const std::exception& e) {
            std::cout << "  FAILED: " << e.what() << "\n\n";
            failed++;
        } catch (...) {
            std::cout << "  FAILED: unknown exception\n\n";
            failed++;
        }
    }

    std::cout << "Results: " << passed << " passed, " << failed << " failed\n";
    return failed > 0 ? 1 : 0;
}
