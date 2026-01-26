// PTO Workload-Schedule Programming (PTO-WSP) framework v9 - Python Bindings for C++ IR
// Copyright (c) 2026 PTO Project
// SPDX-License-Identifier: Apache-2.0

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

#include "pto/rt/ir/ir.hpp"
#include "pto/rt/graph/graph.hpp"
#include "pto/rt/backend/backend.hpp"
#include "pto/rt/backend/cpu_sim.hpp"
#include "pto/rt/backend/ascend_npu.hpp"

namespace py = pybind11;
using namespace pto::wsp;

// ============================================================
// IR Bindings
// ============================================================

void bind_ir(py::module_& m) {
    // NodeKind enum
    py::enum_<ir::NodeKind>(m, "NodeKind")
        .value("DenseAxis", ir::NodeKind::DenseAxis)
        .value("DenseDynAxis", ir::NodeKind::DenseDynAxis)
        .value("RaggedAxis", ir::NodeKind::RaggedAxis)
        .value("SparseAxis", ir::NodeKind::SparseAxis)
        .value("Task", ir::NodeKind::Task)
        .value("ParallelFor", ir::NodeKind::ParallelFor)
        .value("ForEach", ir::NodeKind::ForEach)
        .value("Select", ir::NodeKind::Select)
        .value("Cond", ir::NodeKind::Cond)
        .value("Combine", ir::NodeKind::Combine)
        .value("Sequential", ir::NodeKind::Sequential)
        .value("Call", ir::NodeKind::Call)
        .value("Channel", ir::NodeKind::Channel)
        .value("Process", ir::NodeKind::Process)
        .value("Send", ir::NodeKind::Send)
        .value("Consume", ir::NodeKind::Consume)
        .value("Pipeline", ir::NodeKind::Pipeline)
        .value("Dispatch", ir::NodeKind::Dispatch)
        .value("Stream", ir::NodeKind::Stream)
        .value("Timing", ir::NodeKind::Timing)
        .value("SpatialMap", ir::NodeKind::SpatialMap)
        .value("Layout", ir::NodeKind::Layout)
        .value("DispatchThreshold", ir::NodeKind::DispatchThreshold)
        .value("PipelineDepth", ir::NodeKind::PipelineDepth)
        .value("TaskWindow", ir::NodeKind::TaskWindow)
        .value("BatchDeps", ir::NodeKind::BatchDeps)
        .value("Ext", ir::NodeKind::Ext);

    // WorkloadLevel enum
    py::enum_<ir::WorkloadLevel>(m, "WorkloadLevel")
        .value("CPU", ir::WorkloadLevel::CPU)
        .value("NPU", ir::WorkloadLevel::NPU)
        .value("Any", ir::WorkloadLevel::Any);

    // DType enum
    py::enum_<ir::DType>(m, "DType")
        .value("F16", ir::DType::F16)
        .value("BF16", ir::DType::BF16)
        .value("F32", ir::DType::F32)
        .value("F64", ir::DType::F64)
        .value("I8", ir::DType::I8)
        .value("I16", ir::DType::I16)
        .value("I32", ir::DType::I32)
        .value("I64", ir::DType::I64)
        .value("U8", ir::DType::U8)
        .value("U16", ir::DType::U16)
        .value("U32", ir::DType::U32)
        .value("U64", ir::DType::U64)
        .value("Bool", ir::DType::Bool);

    // Location enum
    py::enum_<ir::Location>(m, "Location")
        .value("Global", ir::Location::Global)
        .value("L2", ir::Location::L2)
        .value("UB", ir::Location::UB)
        .value("L1", ir::Location::L1);

    // DispatchPolicy enum
    py::enum_<ir::DispatchPolicy>(m, "DispatchPolicy")
        .value("RoundRobin", ir::DispatchPolicy::RoundRobin)
        .value("Affinity", ir::DispatchPolicy::Affinity)
        .value("Hash", ir::DispatchPolicy::Hash)
        .value("WorkSteal", ir::DispatchPolicy::WorkSteal)
        .value("Custom", ir::DispatchPolicy::Custom);

    // TimingPolicy enum
    py::enum_<ir::TimingPolicy>(m, "TimingPolicy")
        .value("Immediate", ir::TimingPolicy::Immediate)
        .value("Batched", ir::TimingPolicy::Batched)
        .value("Interleaved", ir::TimingPolicy::Interleaved)
        .value("RateLimited", ir::TimingPolicy::RateLimited);

    // TaskWindowOverflowPolicy enum
    py::enum_<ir::TaskWindowOverflowPolicy>(m, "TaskWindowOverflowPolicy")
        .value("Stall", ir::TaskWindowOverflowPolicy::Stall)
        .value("Abort", ir::TaskWindowOverflowPolicy::Abort)
        .value("Benchmark", ir::TaskWindowOverflowPolicy::Benchmark);

    // Shape struct
    py::class_<ir::Shape>(m, "Shape")
        .def(py::init<>())
        .def(py::init([](const std::vector<int64_t>& dims) {
            ir::Shape s;
            s.dims = dims;
            return s;
        }))
        .def_readwrite("dims", &ir::Shape::dims)
        .def("is_static", &ir::Shape::is_static)
        .def("numel", &ir::Shape::numel)
        .def("rank", &ir::Shape::rank);

    // TensorType struct
    py::class_<ir::TensorType>(m, "TensorType")
        .def(py::init<>())
        .def_readwrite("shape", &ir::TensorType::shape)
        .def_readwrite("dtype", &ir::TensorType::dtype)
        .def_readwrite("location", &ir::TensorType::location)
        .def("to_string", &ir::TensorType::to_string);

    // ChannelType struct
    py::class_<ir::ChannelType>(m, "ChannelType")
        .def(py::init<>())
        .def_readwrite("element_type", &ir::ChannelType::element_type)
        .def_readwrite("capacity", &ir::ChannelType::capacity)
        .def("to_string", &ir::ChannelType::to_string);

    // IRNode base class (using shared_ptr for proper sharing)
    py::class_<ir::IRNode, std::shared_ptr<ir::IRNode>>(m, "IRNode")
        .def_readonly("id", &ir::IRNode::id)
        .def_readonly("kind", &ir::IRNode::kind)
        .def_readonly("level", &ir::IRNode::level)
        .def("print", [](const ir::IRNode& self) {
            std::ostringstream oss;
            self.print(oss, 0);
            return oss.str();
        });

    // AxisNode hierarchy
    py::class_<ir::AxisNode, ir::IRNode, std::shared_ptr<ir::AxisNode>>(m, "AxisNode");

    py::class_<ir::DenseAxisNode, ir::AxisNode, std::shared_ptr<ir::DenseAxisNode>>(m, "DenseAxisNode")
        .def_readonly("size", &ir::DenseAxisNode::size);

    py::class_<ir::DenseDynAxisNode, ir::AxisNode, std::shared_ptr<ir::DenseDynAxisNode>>(m, "DenseDynAxisNode")
        .def_readonly("size_var", &ir::DenseDynAxisNode::size_var);

    py::class_<ir::RaggedAxisNode, ir::AxisNode, std::shared_ptr<ir::RaggedAxisNode>>(m, "RaggedAxisNode")
        .def_readonly("outer_size_var", &ir::RaggedAxisNode::outer_size_var)
        .def_readonly("lengths_var", &ir::RaggedAxisNode::lengths_var);

    py::class_<ir::SparseAxisNode, ir::AxisNode, std::shared_ptr<ir::SparseAxisNode>>(m, "SparseAxisNode")
        .def_readonly("outer_size_var", &ir::SparseAxisNode::outer_size_var)
        .def_readonly("indptr_var", &ir::SparseAxisNode::indptr_var)
        .def_readonly("indices_var", &ir::SparseAxisNode::indices_var);

    // WorkloadNode hierarchy
    py::class_<ir::WorkloadNode, ir::IRNode, std::shared_ptr<ir::WorkloadNode>>(m, "WorkloadNode");

    py::class_<ir::TaskNode, ir::WorkloadNode, std::shared_ptr<ir::TaskNode>>(m, "TaskNode")
        .def_readonly("kernel_name", &ir::TaskNode::kernel_name)
        .def_readonly("params", &ir::TaskNode::params)
        .def_readonly("resources", &ir::TaskNode::resources);

    py::class_<ir::ParallelForNode, ir::WorkloadNode, std::shared_ptr<ir::ParallelForNode>>(m, "ParallelForNode")
        .def_readonly("axis", &ir::ParallelForNode::axis)
        .def_readonly("index_var", &ir::ParallelForNode::index_var)
        .def_readonly("body", &ir::ParallelForNode::body);

    py::class_<ir::ForEachNode, ir::WorkloadNode, std::shared_ptr<ir::ForEachNode>>(m, "ForEachNode")
        .def_readonly("axis", &ir::ForEachNode::axis)
        .def_readonly("index_var", &ir::ForEachNode::index_var)
        .def_readonly("body", &ir::ForEachNode::body);

    py::class_<ir::SelectNode, ir::WorkloadNode, std::shared_ptr<ir::SelectNode>>(m, "SelectNode")
        .def_readonly("sparse", &ir::SelectNode::sparse)
        .def_readonly("index_var", &ir::SelectNode::index_var)
        .def_readonly("body", &ir::SelectNode::body);

    py::class_<ir::CondNode, ir::WorkloadNode, std::shared_ptr<ir::CondNode>>(m, "CondNode")
        .def_readonly("predicate", &ir::CondNode::predicate)
        .def_readonly("then_branch", &ir::CondNode::then_branch)
        .def_readonly("else_branch", &ir::CondNode::else_branch);

    py::class_<ir::CombineNode, ir::WorkloadNode, std::shared_ptr<ir::CombineNode>>(m, "CombineNode")
        .def_readonly("workloads", &ir::CombineNode::workloads);

    py::class_<ir::SequentialNode, ir::WorkloadNode, std::shared_ptr<ir::SequentialNode>>(m, "SequentialNode")
        .def_readonly("workloads", &ir::SequentialNode::workloads);

    py::class_<ir::CallNode, ir::WorkloadNode, std::shared_ptr<ir::CallNode>>(m, "CallNode")
        .def_readonly("target_name", &ir::CallNode::target_name)
        .def_readonly("args", &ir::CallNode::args);

    // ScheduleNode hierarchy
    py::class_<ir::ScheduleNode, ir::IRNode, std::shared_ptr<ir::ScheduleNode>>(m, "ScheduleNode");

    py::class_<ir::DispatchNode, ir::ScheduleNode, std::shared_ptr<ir::DispatchNode>>(m, "DispatchNode")
        .def_readonly("policy", &ir::DispatchNode::policy)
        .def_readonly("num_targets", &ir::DispatchNode::num_targets)
        .def_readonly("key_expr", &ir::DispatchNode::key_expr);

    py::class_<ir::StreamNode, ir::ScheduleNode, std::shared_ptr<ir::StreamNode>>(m, "StreamNode")
        .def_readonly("num_streams", &ir::StreamNode::num_streams)
        .def_readonly("key_expr", &ir::StreamNode::key_expr);

    py::class_<ir::TimingNode, ir::ScheduleNode, std::shared_ptr<ir::TimingNode>>(m, "TimingNode")
        .def_readonly("policy", &ir::TimingNode::policy)
        .def_readonly("param", &ir::TimingNode::param);

    py::class_<ir::SpatialMapNode, ir::ScheduleNode, std::shared_ptr<ir::SpatialMapNode>>(m, "SpatialMapNode")
        .def_readonly("grid", &ir::SpatialMapNode::grid);

    // LayoutDimKind enum
    py::enum_<ir::LayoutDimKind>(m, "LayoutDimKind")
        .value("Shard", ir::LayoutDimKind::Shard)
        .value("Replicate", ir::LayoutDimKind::Replicate);

    // LayoutDim struct
    py::class_<ir::LayoutDim>(m, "LayoutDim")
        .def(py::init<>())
        .def_readwrite("kind", &ir::LayoutDim::kind)
        .def_readwrite("mesh_axis", &ir::LayoutDim::mesh_axis)
        .def("to_string", &ir::LayoutDim::to_string);

    py::class_<ir::LayoutNode, ir::ScheduleNode, std::shared_ptr<ir::LayoutNode>>(m, "LayoutNode")
        .def_readonly("tensor_name", &ir::LayoutNode::tensor_name)
        .def_readonly("dims", &ir::LayoutNode::dims);

    // Extended Schedule nodes
    py::class_<ir::DispatchThresholdNode, ir::ScheduleNode, std::shared_ptr<ir::DispatchThresholdNode>>(m, "DispatchThresholdNode")
        .def_readonly("thresholds", &ir::DispatchThresholdNode::thresholds)
        .def_readonly("policies", &ir::DispatchThresholdNode::policies);

    py::class_<ir::PipelineDepthNode, ir::ScheduleNode, std::shared_ptr<ir::PipelineDepthNode>>(m, "PipelineDepthNode")
        .def_readonly("depth", &ir::PipelineDepthNode::depth);

    py::class_<ir::TaskWindowNode, ir::ScheduleNode, std::shared_ptr<ir::TaskWindowNode>>(m, "TaskWindowNode")
        .def_readonly("size", &ir::TaskWindowNode::size)
        .def_readonly("unit", &ir::TaskWindowNode::unit)
        .def_readonly("overflow", &ir::TaskWindowNode::overflow);

    py::class_<ir::BatchDepsNode, ir::ScheduleNode, std::shared_ptr<ir::BatchDepsNode>>(m, "BatchDepsNode")
        .def_readonly("threshold", &ir::BatchDepsNode::threshold)
        .def_readonly("range_compression", &ir::BatchDepsNode::range_compression);

    // WorkloadDef
    py::class_<ir::WorkloadDef>(m, "WorkloadDef")
        .def(py::init<>())
        .def_readwrite("name", &ir::WorkloadDef::name)
        .def_readwrite("level", &ir::WorkloadDef::level)
        .def_readwrite("params", &ir::WorkloadDef::params)
        .def_readwrite("body", &ir::WorkloadDef::body)
        .def("print", [](const ir::WorkloadDef& self) {
            std::ostringstream oss;
            self.print(oss);
            return oss.str();
        });

    // ScheduleDef
    py::class_<ir::ScheduleDef>(m, "ScheduleDef")
        .def(py::init<>())
        .def_readwrite("name", &ir::ScheduleDef::name)
        .def_readwrite("workload_name", &ir::ScheduleDef::workload_name)
        .def_readwrite("level", &ir::ScheduleDef::level)
        .def_readwrite("directives", &ir::ScheduleDef::directives)
        .def("dispatch", &ir::ScheduleDef::dispatch)
        .def("stream", &ir::ScheduleDef::stream)
        .def("timing", &ir::ScheduleDef::timing)
        .def("print", [](const ir::ScheduleDef& self) {
            std::ostringstream oss;
            self.print(oss);
            return oss.str();
        });

    // Module
    py::class_<ir::Module>(m, "Module")
        .def(py::init<>())
        .def_readwrite("name", &ir::Module::name)
        .def_readwrite("version", &ir::Module::version)
        .def_readwrite("targets", &ir::Module::targets)
        .def_readwrite("workloads", &ir::Module::workloads)
        .def_readwrite("schedules", &ir::Module::schedules)
        .def("find_workload", &ir::Module::findWorkload, py::return_value_policy::reference)
        .def("find_schedule", &ir::Module::findSchedule, py::return_value_policy::reference)
        .def("find_schedule_for_workload", &ir::Module::findScheduleForWorkload, py::return_value_policy::reference)
        .def("print", [](const ir::Module& self) {
            std::ostringstream oss;
            self.print(oss);
            return oss.str();
        });

    // IRFactory
    py::class_<ir::IRFactory>(m, "IRFactory")
        .def(py::init<>())
        .def("reset", &ir::IRFactory::reset)
        .def("create_dense_axis", [](ir::IRFactory& f, int64_t size) {
            return f.create<ir::DenseAxisNode>(size);
        })
        .def("create_dense_dyn_axis", [](ir::IRFactory& f, const std::string& var) {
            return f.create<ir::DenseDynAxisNode>(var);
        })
        .def("create_ragged_axis", [](ir::IRFactory& f, const std::string& outer, const std::string& lengths) {
            return f.create<ir::RaggedAxisNode>(outer, lengths);
        })
        .def("create_sparse_axis", [](ir::IRFactory& f, const std::string& outer,
                                      const std::string& indptr, const std::string& indices) {
            return f.create<ir::SparseAxisNode>(outer, indptr, indices);
        })
        .def("create_task", [](ir::IRFactory& f, const std::string& kernel,
                               const std::vector<std::string>& params,
                               const std::vector<std::string>& resources) {
            return f.create<ir::TaskNode>(kernel, params, resources);
        })
        .def("create_parallel_for", [](ir::IRFactory& f,
                                       ir::IRPtr<ir::AxisNode> axis,
                                       const std::string& var,
                                       ir::IRPtr<ir::WorkloadNode> body) {
            return f.create<ir::ParallelForNode>(axis, var, body);
        })
        .def("create_for_each", [](ir::IRFactory& f,
                                   ir::IRPtr<ir::AxisNode> axis,
                                   const std::string& var,
                                   ir::IRPtr<ir::WorkloadNode> body) {
            return f.create<ir::ForEachNode>(axis, var, body);
        })
        .def("create_select", [](ir::IRFactory& f,
                                 ir::IRPtr<ir::SparseAxisNode> sparse,
                                 const std::string& var,
                                 ir::IRPtr<ir::WorkloadNode> body) {
            return f.create<ir::SelectNode>(sparse, var, body);
        })
        .def("create_cond", [](ir::IRFactory& f,
                               const std::string& predicate,
                               ir::IRPtr<ir::WorkloadNode> then_branch,
                               ir::IRPtr<ir::WorkloadNode> else_branch) {
            return f.create<ir::CondNode>(predicate, then_branch, else_branch);
        })
        .def("create_combine", [](ir::IRFactory& f,
                                  const std::vector<ir::IRPtr<ir::WorkloadNode>>& workloads) {
            return f.create<ir::CombineNode>(workloads);
        })
        .def("create_sequential", [](ir::IRFactory& f,
                                     const std::vector<ir::IRPtr<ir::WorkloadNode>>& workloads) {
            return f.create<ir::SequentialNode>(workloads);
        })
        .def("create_call", [](ir::IRFactory& f, const std::string& target,
                               const std::vector<std::string>& args) {
            return f.create<ir::CallNode>(target, args);
        })
        .def("create_dispatch", [](ir::IRFactory& f, ir::DispatchPolicy policy,
                                   int num_targets, const std::string& key_expr) {
            return f.create<ir::DispatchNode>(policy, num_targets, key_expr);
        })
        .def("create_stream", [](ir::IRFactory& f, int num_streams, const std::string& key_expr) {
            return f.create<ir::StreamNode>(num_streams, key_expr);
        })
        .def("create_timing", [](ir::IRFactory& f, ir::TimingPolicy policy, int param) {
            return f.create<ir::TimingNode>(policy, param);
        })
        .def("create_spatial_map", [](ir::IRFactory& f, const std::vector<int64_t>& grid) {
            return f.create<ir::SpatialMapNode>(grid);
        })
        .def("create_dispatch_threshold", [](ir::IRFactory& f,
                                             const std::vector<int64_t>& thresholds,
                                             const std::vector<ir::DispatchPolicy>& policies) {
            return f.create<ir::DispatchThresholdNode>(thresholds, policies);
        })
        .def("create_pipeline_depth", [](ir::IRFactory& f, int depth) {
            return f.create<ir::PipelineDepthNode>(depth);
        })
        .def("create_task_window", [](ir::IRFactory& f, int64_t size, const std::string& unit,
                                      ir::TaskWindowOverflowPolicy overflow) {
            return f.create<ir::TaskWindowNode>(size, unit, overflow);
        }, py::arg("size"), py::arg("unit") = "tasks",
           py::arg("overflow") = ir::TaskWindowOverflowPolicy::Stall)
        .def("create_batch_deps", [](ir::IRFactory& f, int64_t threshold, bool range_compression) {
            return f.create<ir::BatchDepsNode>(threshold, range_compression);
        }, py::arg("threshold"), py::arg("range_compression") = false);

    // Parser - use static methods since Parser requires istream constructor
    m.def("parse_string", &ir::Parser::parseString, py::arg("source"),
          "Parse PTO IR from a string");
    m.def("parse_file", &ir::Parser::parseFile, py::arg("path"),
          "Parse PTO IR from a file");

    // Printer
    py::class_<ir::Printer>(m, "Printer")
        .def(py::init<>())
        .def("print_module", [](ir::Printer& p, const ir::Module& module) {
            std::ostringstream oss;
            p.print(module, oss);
            return oss.str();
        });

    // Utility functions
    m.def("node_kind_to_string", &ir::nodeKindToString);
    m.def("dtype_to_string", &ir::dtypeToString);
    m.def("location_to_string", &ir::locationToString);
    m.def("level_to_string", &ir::levelToString);

    // TypeCheckErrorKind enum
    py::enum_<ir::TypeCheckErrorKind>(m, "TypeCheckErrorKind")
        .value("InvalidStructure", ir::TypeCheckErrorKind::InvalidStructure)
        .value("UndefinedReference", ir::TypeCheckErrorKind::UndefinedReference)
        .value("DuplicateDefinition", ir::TypeCheckErrorKind::DuplicateDefinition)
        .value("LayoutIncompatible", ir::TypeCheckErrorKind::LayoutIncompatible)
        .value("TypeMismatch", ir::TypeCheckErrorKind::TypeMismatch)
        .value("AxisBoundsError", ir::TypeCheckErrorKind::AxisBoundsError)
        .value("ChannelTypeMismatch", ir::TypeCheckErrorKind::ChannelTypeMismatch)
        .value("WorkloadLevelError", ir::TypeCheckErrorKind::WorkloadLevelError)
        .value("ScheduleMismatch", ir::TypeCheckErrorKind::ScheduleMismatch);

    // TypeCheckError struct
    py::class_<ir::TypeCheckError>(m, "TypeCheckError")
        .def_readonly("kind", &ir::TypeCheckError::kind)
        .def_readonly("message", &ir::TypeCheckError::message)
        .def_readonly("location", &ir::TypeCheckError::location)
        .def_readonly("hint", &ir::TypeCheckError::hint)
        .def("to_string", &ir::TypeCheckError::to_string);

    // TypeCheckResult struct
    py::class_<ir::TypeCheckResult>(m, "TypeCheckResult")
        .def_readonly("valid", &ir::TypeCheckResult::valid)
        .def_readonly("errors", &ir::TypeCheckResult::errors)
        .def("to_string", &ir::TypeCheckResult::to_string);

    // type_check function
    m.def("type_check", &ir::type_check, py::arg("module"),
          "Run IR-level type checking on a module");
}

// ============================================================
// Graph Bindings
// ============================================================

void bind_graph(py::module_& m) {
    // ExecDomain enum
    py::enum_<graph::ExecDomain>(m, "ExecDomain")
        .value("HostCPU", graph::ExecDomain::HostCPU)
        .value("AscendAICore", graph::ExecDomain::AscendAICore)
        .value("AMDAIETile", graph::ExecDomain::AMDAIETile);

    // ExecPool enum
    py::enum_<graph::ExecPool>(m, "ExecPool")
        .value("Vector", graph::ExecPool::Vector)
        .value("Cube", graph::ExecPool::Cube)
        .value("Any", graph::ExecPool::Any);

    // WindowMode enum
    py::enum_<graph::WindowMode>(m, "WindowMode")
        .value("Stall", graph::WindowMode::Stall)
        .value("Abort", graph::WindowMode::Abort)
        .value("Benchmark", graph::WindowMode::Benchmark);

    // GateScope enum
    py::enum_<graph::GateScope>(m, "GateScope")
        .value("Global", graph::GateScope::Global)
        .value("PerStream", graph::GateScope::PerStream)
        .value("PerPool", graph::GateScope::PerPool);

    // TensorRegion2D
    py::class_<graph::TensorRegion2D>(m, "TensorRegion2D")
        .def(py::init<>())
        .def(py::init([](uint64_t base, int32_t row_off, int32_t col_off, int32_t rows, int32_t cols) {
            return graph::TensorRegion2D{base, row_off, col_off, rows, cols};
        }))
        .def_readwrite("base", &graph::TensorRegion2D::base)
        .def_readwrite("row_off", &graph::TensorRegion2D::row_off)
        .def_readwrite("col_off", &graph::TensorRegion2D::col_off)
        .def_readwrite("rows", &graph::TensorRegion2D::rows)
        .def_readwrite("cols", &graph::TensorRegion2D::cols)
        .def("overlaps", &graph::TensorRegion2D::overlaps)
        .def("contains", &graph::TensorRegion2D::contains);

    // KernelInfo
    py::class_<graph::KernelInfo>(m, "KernelInfo")
        .def(py::init<>())
        .def_readwrite("name", &graph::KernelInfo::name)
        .def_readwrite("symbol", &graph::KernelInfo::symbol)
        .def_readwrite("num_params", &graph::KernelInfo::num_params)
        .def_readwrite("num_io", &graph::KernelInfo::num_io)
        .def_readwrite("default_domain", &graph::KernelInfo::default_domain)
        .def_readwrite("default_pool", &graph::KernelInfo::default_pool);

    // KernelBundle
    py::class_<graph::KernelBundle>(m, "KernelBundle")
        .def(py::init<>())
        .def("register_kernel", &graph::KernelBundle::register_kernel)
        .def("find_kernel", &graph::KernelBundle::find_kernel)
        .def("get_kernel", &graph::KernelBundle::get_kernel, py::return_value_policy::reference)
        .def("size", &graph::KernelBundle::size);

    // TaskGraphStorage
    py::class_<graph::TaskGraphStorage>(m, "TaskGraphStorage")
        .def(py::init<>())
        .def("reserve", &graph::TaskGraphStorage::reserve)
        .def("num_tasks", &graph::TaskGraphStorage::num_tasks)
        .def("num_edges", &graph::TaskGraphStorage::num_edges)
        .def("is_finalized", &graph::TaskGraphStorage::is_finalized)
        .def("finalize", &graph::TaskGraphStorage::finalize)
        .def("kernel_bundle", py::overload_cast<>(&graph::TaskGraphStorage::kernel_bundle),
             py::return_value_policy::reference);

    // TaskGraphBuilder
    py::class_<graph::TaskGraphBuilder>(m, "TaskGraphBuilder")
        .def(py::init<graph::TaskGraphStorage&>())
        .def("begin_task", &graph::TaskGraphBuilder::begin_task, py::return_value_policy::reference)
        .def("set_domain", &graph::TaskGraphBuilder::set_domain, py::return_value_policy::reference)
        .def("set_pool", &graph::TaskGraphBuilder::set_pool, py::return_value_policy::reference)
        .def("set_stream", &graph::TaskGraphBuilder::set_stream, py::return_value_policy::reference)
        .def("set_affinity", &graph::TaskGraphBuilder::set_affinity, py::return_value_policy::reference)
        .def("add_arg", &graph::TaskGraphBuilder::add_arg, py::return_value_policy::reference)
        .def("add_input", &graph::TaskGraphBuilder::add_input, py::return_value_policy::reference)
        .def("add_output", &graph::TaskGraphBuilder::add_output, py::return_value_policy::reference)
        .def("submit", &graph::TaskGraphBuilder::submit)
        .def("add_dependency", &graph::TaskGraphBuilder::add_dependency)
        .def("finalize", &graph::TaskGraphBuilder::finalize);
}

// ============================================================
// Backend Bindings
// ============================================================

void bind_backend(py::module_& m) {
    // ProgramStats
    py::class_<backend::ProgramStats>(m, "ProgramStats")
        .def_readonly("num_tasks", &backend::ProgramStats::num_tasks)
        .def_readonly("num_streams", &backend::ProgramStats::num_streams)
        .def_readonly("num_executors", &backend::ProgramStats::num_executors)
        .def_readonly("compile_time_ms", &backend::ProgramStats::compile_time_ms)
        .def_readonly("execute_time_ms", &backend::ProgramStats::execute_time_ms)
        .def_readonly("peak_memory_bytes", &backend::ProgramStats::peak_memory_bytes)
        .def_readonly("total_edges", &backend::ProgramStats::total_edges);

    // CompileOptions
    py::class_<backend::CompileOptions>(m, "CompileOptions")
        .def(py::init<>())
        .def_readwrite("target", &backend::CompileOptions::target)
        .def_readwrite("optimization_level", &backend::CompileOptions::optimization_level)
        .def_readwrite("enable_profiling", &backend::CompileOptions::enable_profiling)
        .def_readwrite("enable_debug", &backend::CompileOptions::enable_debug)
        .def_readwrite("num_threads", &backend::CompileOptions::num_threads)
        .def_readwrite("num_aicpus", &backend::CompileOptions::num_aicpus)
        .def_readwrite("num_streams", &backend::CompileOptions::num_streams)
        .def_readwrite("task_window_size", &backend::CompileOptions::task_window_size)
        .def_readwrite("task_window_mode", &backend::CompileOptions::task_window_mode)
        .def_readwrite("pipeline_depth", &backend::CompileOptions::pipeline_depth)
        .def_readwrite("gate_scope", &backend::CompileOptions::gate_scope)
        .def_readwrite("batch_deps_threshold", &backend::CompileOptions::batch_deps_threshold);

    // Program base class
    py::class_<backend::Program, std::unique_ptr<backend::Program>>(m, "Program")
        .def("execute", &backend::Program::execute)
        .def("execute_async", &backend::Program::execute_async)
        .def("synchronize", &backend::Program::synchronize)
        .def("is_complete", &backend::Program::is_complete)
        .def("elapsed_ms", &backend::Program::elapsed_ms)
        .def("stats", &backend::Program::stats)
        .def("dump", &backend::Program::dump);

    // Backend base class
    py::class_<backend::Backend, std::unique_ptr<backend::Backend>>(m, "Backend")
        .def("name", &backend::Backend::name)
        .def("supported_targets", &backend::Backend::supported_targets)
        .def("supports_kind", py::overload_cast<ir::NodeKind>(&backend::Backend::supports, py::const_));

    // BackendRegistry - singleton access, non-copyable
    m.def("get_backend", [](const std::string& name) {
        return backend::BackendRegistry::instance().get_backend(name);
    }, py::arg("name"), py::return_value_policy::reference,
       "Get backend by name from the global registry");

    m.def("available_backends", []() {
        return backend::BackendRegistry::instance().available_backends();
    }, "Get list of available backend names");

    // Compile function
    m.def("compile", &backend::compile, py::arg("module"), py::arg("options") = backend::CompileOptions{});

    // NPU IR bindings for codegen testing
    // NOTE: NPUModule/NPUFunction removed from C++ IR (L11).
    // Kernel compilation now happens in Python (python/pto_wsp/kernel.py).
    // The JIT kernel decorator traces Python functions to produce typed IR,
    // then generates backend code. C++ only sees compiled kernel code strings.

    // AscendNPUBackend - accepts pre-compiled kernel code from Python JIT
    py::class_<backend::ascend::AscendNPUBackend, backend::Backend,
               std::unique_ptr<backend::ascend::AscendNPUBackend>>(m, "AscendNPUBackend")
        .def(py::init<>())
        .def("name", &backend::ascend::AscendNPUBackend::name)
        .def("supported_targets", &backend::ascend::AscendNPUBackend::supported_targets);
}

// ============================================================
// Module Definition
// ============================================================

PYBIND11_MODULE(pto_ir_cpp, m) {
    m.doc() = "PTO-RT v9 C++ IR Python Bindings";

    bind_ir(m);
    bind_graph(m);
    bind_backend(m);

    // Version info
    m.attr("__version__") = "0.9.0";
}
