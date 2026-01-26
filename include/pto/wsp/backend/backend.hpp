// Copyright 2026 PTO-RT Authors
// SPDX-License-Identifier: Apache-2.0

#pragma once  // API-2 FIX: Consistent include guard style
// (was: #ifndef PTO_WSP_BACKEND_BACKEND_HPP)

#include "pto/rt/graph/graph.hpp"
#include "pto/rt/ir/ir.hpp"

#include <memory>
#include <string>
#include <vector>
#include <chrono>
#include <unordered_map>
#include <functional>
#include <any>  // SOLID-3 FIX: For extensible backend options

namespace pto::wsp::backend {

// ============================================================
// Program Statistics
// ============================================================

/// Statistics for a compiled program
struct ProgramStats {
    size_t num_tasks = 0;
    size_t num_streams = 0;
    size_t num_executors = 0;  // AICPUs, threads, tiles
    double compile_time_ms = 0;
    double execute_time_ms = 0;
    size_t peak_memory_bytes = 0;
    size_t total_edges = 0;
};

// ============================================================
// Program Interface
// ============================================================

/// Base class for compiled programs
class Program {
public:
    virtual ~Program() = default;

    // API-4 FIX: Capability query for codegen-only programs
    /// Returns true if this program can be executed (vs codegen-only artifacts)
    virtual bool can_execute() const { return true; }

    // Execution
    virtual void execute() = 0;
    virtual void execute_async() = 0;
    virtual void synchronize() = 0;
    virtual bool is_complete() const = 0;

    // Profiling
    virtual double elapsed_ms() const = 0;
    virtual ProgramStats stats() const = 0;

    // Debugging
    virtual std::string dump() const { return ""; }
};

// ============================================================
// Compile Options
// ============================================================

/// Options for compilation
struct CompileOptions {
    std::string target = "cpu_sim";  // Default backend

    // Optimization
    int optimization_level = 2;  // 0-3
    bool enable_profiling = false;
    bool enable_debug = false;

    // CPU-specific
    int num_threads = 0;  // 0 = auto (hardware_concurrency)

    // NPU-specific
    int num_aicpus = 1;
    int num_streams = 2;

    // AIE-specific
    std::vector<int64_t> grid;  // Tile grid dimensions

    // Extended primitive settings
    size_t task_window_size = 8192;
    graph::WindowMode task_window_mode = graph::WindowMode::Stall;
    size_t pipeline_depth = 2;
    graph::GateScope gate_scope = graph::GateScope::Global;
    size_t batch_deps_threshold = 64;

    // SOLID-3 FIX: Extensible backend-specific options
    // Backends can cast to their own options types:
    //   auto* opts = std::any_cast<MyBackendOptions>(&options.backend_options);
    std::any backend_options;
};

// ============================================================
// Schedule Runtime Config
// ============================================================

/// Lowered schedule configuration
struct ScheduleRuntimeConfig {
    // Window settings
    size_t window_size = 8192;
    graph::WindowMode window_mode = graph::WindowMode::Stall;

    // Pipeline depth settings
    size_t pipeline_depth = 2;
    graph::GateScope gate_scope = graph::GateScope::Global;

    // Batch deps settings
    size_t batch_threshold = 64;

    // Stream settings
    size_t num_streams = 2;
    bool dual_queue_enabled = false;
};

// ============================================================
// Lowered Plan (Backend-Neutral)
// ============================================================

/// Lowered form after IR processing (backend-neutral)
struct LoweredPlan {
    graph::TaskGraphStorage graph;
    ScheduleRuntimeConfig sched_config;
    std::string workload_name;
};

// ============================================================
// Backend Interface
// ============================================================

/// Base class for all backends
class Backend {
public:
    virtual ~Backend() = default;

    /// Get backend name
    virtual std::string name() const = 0;

    /// Get supported targets
    virtual std::vector<std::string> supported_targets() const = 0;

    /// Check if backend supports a given IR module
    virtual bool supports(const ir::Module& module) const = 0;

    /// Check if backend supports a given IR node kind
    virtual bool supports(ir::NodeKind kind) const = 0;

    /// Phase 1: Lower IR to backend-neutral LoweredPlan
    virtual LoweredPlan lower(const ir::Module& module,
                              const CompileOptions& options) = 0;

    /// Phase 2: Compile LoweredPlan to executable Program
    virtual std::unique_ptr<Program> compile(const LoweredPlan& plan,
                                              const CompileOptions& options) = 0;

    /// Convenience: Combined lower + compile
    std::unique_ptr<Program> compile_module(const ir::Module& module,
                                             const CompileOptions& options) {
        auto plan = lower(module, options);
        return compile(plan, options);
    }
};

// ============================================================
// Backend Registry
// ============================================================

/// Global registry for backends
class BackendRegistry {
public:
    static BackendRegistry& instance() {
        static BackendRegistry registry;
        return registry;
    }

    /// Register a backend
    void register_backend(std::unique_ptr<Backend> backend) {
        std::string name = backend->name();
        backends_[name] = std::move(backend);
    }

    /// Get backend by name
    Backend* get_backend(const std::string& name) const {
        auto it = backends_.find(name);
        return (it != backends_.end()) ? it->second.get() : nullptr;
    }

    /// Get list of available backends
    std::vector<std::string> available_backends() const {
        std::vector<std::string> names;
        names.reserve(backends_.size());
        for (const auto& [name, _] : backends_) {
            names.push_back(name);
        }
        return names;
    }

    /// Auto-select backend based on IR module and options
    Backend* select_backend(const ir::Module& module,
                            const CompileOptions& options) const {
        // First, try explicit target by name
        if (auto* backend = get_backend(options.target)) {
            if (backend->supports(module)) {
                return backend;
            }
        }

        // API-1 FIX: Check supported_targets() aliases
        for (const auto& [name, backend] : backends_) {
            const auto& targets = backend->supported_targets();
            for (const auto& target : targets) {
                if (target == options.target && backend->supports(module)) {
                    return backend.get();
                }
            }
        }

        // Fall back to cpu_sim
        if (auto* cpu = get_backend("cpu_sim")) {
            return cpu;
        }

        return nullptr;
    }

private:
    BackendRegistry() = default;
    std::unordered_map<std::string, std::unique_ptr<Backend>> backends_;
};

// ============================================================
// Backend Registration Macro
// ============================================================

/// Helper for static backend registration
#define REGISTER_BACKEND(BackendClass) \
    namespace { \
        static bool _registered_##BackendClass = []() { \
            BackendRegistry::instance().register_backend( \
                std::make_unique<BackendClass>()); \
            return true; \
        }(); \
    }

// ============================================================
// Compile Function
// ============================================================

/// Compile IR module with specified options
inline std::unique_ptr<Program> compile(const ir::Module& module,
                                         const CompileOptions& options = {}) {
    auto* backend = BackendRegistry::instance().select_backend(module, options);
    if (!backend) {
        return nullptr;
    }
    return backend->compile_module(module, options);
}

}  // namespace pto::wsp::backend

