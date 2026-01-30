// PTO Workload-Schedule Programming (PTO-WSP) framework v9 - Ascend NPU Backend
// Copyright (c) 2026 PTO Project
// SPDX-License-Identifier: MIT
//
// This backend generates Ascend NPU code (similar to pto-isa-wc approach).
// No Ascend SDK required - pure code generation.

#pragma once

#include "pto/wsp/backend/backend.hpp"
#include "pto/wsp/backend/codegen.hpp"
#include "pto/wsp/ir/npu.hpp"

#include <sstream>

namespace pto::wsp::backend::ascend {

// ============================================================
// Ascend NPU Code Emitter
// ============================================================

/// Code emitter for Ascend NPU (CANN Kernel Language)
class AscendEmitter : public codegen::CodeEmitter {
public:
    [[nodiscard]] std::string name() const override;
    [[nodiscard]] std::string file_extension() const override;

    void emit_header(codegen::CodeGenContext& ctx) override;
    void emit_footer(codegen::CodeGenContext& ctx) override;
    void emit_npu_function(codegen::CodeGenContext& ctx, const ir::NPUFunction& func) override;

    void emit_load(codegen::CodeGenContext& ctx, const ir::LoadOp& op) override;
    void emit_store(codegen::CodeGenContext& ctx, const ir::StoreOp& op) override;
    void emit_binary(codegen::CodeGenContext& ctx, const ir::BinaryOp& op) override;
    void emit_unary(codegen::CodeGenContext& ctx, const ir::UnaryOp& op) override;
    void emit_reduce(codegen::CodeGenContext& ctx, const ir::ReduceOp& op) override;
    void emit_broadcast(codegen::CodeGenContext& ctx, const ir::BroadcastOp& op) override;
    void emit_matmul(codegen::CodeGenContext& ctx, const ir::MatmulOp& op) override;
    void emit_for_begin(codegen::CodeGenContext& ctx, const ir::ForLoopBeginOp& op) override;
    void emit_for_end(codegen::CodeGenContext& ctx) override;
    void emit_wait(codegen::CodeGenContext& ctx, const ir::WaitOp& op) override;

private:
    void emit_tile_decl(codegen::CodeGenContext& ctx, const ir::TileDecl& t);
    void emit_scalar_decl(codegen::CodeGenContext& ctx, const ir::ScalarDecl& s);
    void emit_operation(codegen::CodeGenContext& ctx, const ir::NPUOp& op);
    void emit_kernel_registration(codegen::CodeGenContext& ctx, const ir::NPUFunction& func);

    static std::string dtype_to_cann(ir::DType dtype);
    static std::string location_to_cann(ir::Location loc);
};

// Register Ascend emitter
REGISTER_EMITTER(AscendEmitter, "ascend_npu")

// ============================================================
// Ascend NPU Program
// ============================================================

/// Placeholder program for generated code (codegen-only, not executable)
class AscendNPUProgram : public Program {
public:
    AscendNPUProgram(graph::TaskGraphStorage storage,
                     ScheduleRuntimeConfig config,
                     std::string code);

    // API-4 FIX: This is a codegen-only program
    bool can_execute() const override { return false; }

    void execute() override;
    void execute_async() override;
    void synchronize() override;
    bool is_complete() const override;
    double elapsed_ms() const override;
    ProgramStats stats() const override;
    std::string dump() const override;

    const std::string& generated_code() const;

private:
    graph::TaskGraphStorage storage_;
    ScheduleRuntimeConfig config_;
    std::string generated_code_;
};

// ============================================================
// Ascend NPU Backend
// ============================================================

/// Ascend NPU compilation backend
class AscendNPUBackend : public Backend {
public:
    AscendNPUBackend();

    std::string name() const override;
    std::vector<std::string> supported_targets() const override;
    bool supports(const ir::Module& module) const override;
    bool supports(ir::NodeKind kind) const override;

    LoweredPlan lower(const ir::Module& module, const CompileOptions& options) override;
    std::unique_ptr<Program> compile(const LoweredPlan& plan,
                                      const CompileOptions& options) override;

    /// Generate code for an NPU module
    std::string generate_code(const ir::NPUModule& module);

    /// Get the last generated code
    const std::string& get_generated_code() const;

private:
    std::unique_ptr<AscendEmitter> emitter_;
    std::string generated_code_;
};

// Register Ascend NPU backend
REGISTER_BACKEND(AscendNPUBackend)

}  // namespace pto::wsp::backend::ascend
