// PTO Workload-Schedule Programming (PTO-WSP) framework v9 - NPU Function IR
// Copyright (c) 2024 PTO Project
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "core.hpp"

#include <variant>
#include <unordered_map>

namespace pto::wsp::ir {

// NPU operation kinds
enum class NPUOpKind {
    // Memory operations
    Load,
    Store,
    Wait,

    // Binary operations
    Add, Mul, Sub, Div,

    // Unary operations
    Exp, Rsqrt, Neg, Abs,

    // Reduction operations
    RowSum, RowMax, RowMin, RowMean,

    // Broadcast operations
    RowExpandMul, RowExpandAdd, RowExpandSub,

    // Matrix operations
    Matmul,

    // Control flow
    ForLoopBegin, ForLoopEnd,
    IfThenBegin, ElseBranch, IfThenEnd,

    // Composite (macro) - lowered to primitives
    RMSNorm, Softmax, LayerNorm,
};

// Tile declaration
struct TileDecl {
    std::string name;
    int64_t rows;
    int64_t cols;
    DType dtype;
    Location location;

    void print(std::ostream& os) const {
        os << "tile " << name << "[" << rows << ", " << cols << "] : "
           << dtypeToString(dtype) << "@" << locationToString(location);
    }
};

// Scalar declaration
struct ScalarDecl {
    std::string name;
    DType dtype;
    std::optional<std::variant<int64_t, double>> immediate;

    void print(std::ostream& os) const {
        os << "scalar " << name << " : " << dtypeToString(dtype);
        if (immediate) {
            os << " = ";
            std::visit([&os](auto v) { os << v; }, *immediate);
        }
    }
};

// Memory reference declaration
struct MemrefDecl {
    std::string name;
    DType dtype;
    Location location;
    std::vector<int64_t> shape;  // Empty = dynamic
    bool is_input = true;
    bool is_output = false;

    void print(std::ostream& os) const {
        os << "memref " << name << " : " << dtypeToString(dtype);
        if (!shape.empty()) {
            os << "[";
            for (size_t i = 0; i < shape.size(); ++i) {
                if (i > 0) os << ", ";
                os << shape[i];
            }
            os << "]";
        }
        os << "@" << locationToString(location);
    }
};

// Base class for NPU operations
struct NPUOp {
    NPUOpKind kind;

    NPUOp(NPUOpKind k) : kind(k) {}
    virtual ~NPUOp() = default;
    virtual void print(std::ostream& os) const = 0;
};

// Load operation
struct LoadOp : NPUOp {
    std::string dst_tile;
    std::string src_memref;
    int64_t row_offset;
    int64_t col_offset;
    bool async;
    std::string tag;

    LoadOp(std::string dst, std::string src, int64_t row = 0, int64_t col = 0,
           bool async_ = false, std::string tag_ = "")
        : NPUOp(NPUOpKind::Load), dst_tile(std::move(dst)), src_memref(std::move(src)),
          row_offset(row), col_offset(col), async(async_), tag(std::move(tag_)) {}

    void print(std::ostream& os) const override {
        os << "load " << dst_tile << ", " << src_memref;
        if (row_offset != 0 || col_offset != 0) {
            os << "[" << row_offset << ", " << col_offset << "]";
        }
        if (async) os << " async";
        if (!tag.empty()) os << " tag=" << tag;
    }
};

// Store operation
struct StoreOp : NPUOp {
    std::string src_tile;
    std::string dst_memref;
    int64_t row_offset;
    int64_t col_offset;
    bool async;
    std::string tag;

    StoreOp(std::string src, std::string dst, int64_t row = 0, int64_t col = 0,
            bool async_ = false, std::string tag_ = "")
        : NPUOp(NPUOpKind::Store), src_tile(std::move(src)), dst_memref(std::move(dst)),
          row_offset(row), col_offset(col), async(async_), tag(std::move(tag_)) {}

    void print(std::ostream& os) const override {
        os << "store " << src_tile << ", " << dst_memref;
        if (row_offset != 0 || col_offset != 0) {
            os << "[" << row_offset << ", " << col_offset << "]";
        }
        if (async) os << " async";
        if (!tag.empty()) os << " tag=" << tag;
    }
};

// Wait for async DMA
struct WaitOp : NPUOp {
    std::string tag;

    WaitOp(std::string t) : NPUOp(NPUOpKind::Wait), tag(std::move(t)) {}

    void print(std::ostream& os) const override {
        os << "wait " << tag;
    }
};

// Binary operation
struct BinaryOp : NPUOp {
    std::string dst;
    std::string a;
    std::string b;

    BinaryOp(NPUOpKind k, std::string dst_, std::string a_, std::string b_)
        : NPUOp(k), dst(std::move(dst_)), a(std::move(a_)), b(std::move(b_)) {}

    void print(std::ostream& os) const override {
        const char* op_name;
        switch (kind) {
            case NPUOpKind::Add: op_name = "add"; break;
            case NPUOpKind::Mul: op_name = "mul"; break;
            case NPUOpKind::Sub: op_name = "sub"; break;
            case NPUOpKind::Div: op_name = "div"; break;
            default: op_name = "binary"; break;
        }
        os << op_name << " " << dst << ", " << a << ", " << b;
    }
};

// Unary operation
struct UnaryOp : NPUOp {
    std::string dst;
    std::string src;

    UnaryOp(NPUOpKind k, std::string dst_, std::string src_)
        : NPUOp(k), dst(std::move(dst_)), src(std::move(src_)) {}

    void print(std::ostream& os) const override {
        const char* op_name;
        switch (kind) {
            case NPUOpKind::Exp: op_name = "exp"; break;
            case NPUOpKind::Rsqrt: op_name = "rsqrt"; break;
            case NPUOpKind::Neg: op_name = "neg"; break;
            case NPUOpKind::Abs: op_name = "abs"; break;
            default: op_name = "unary"; break;
        }
        os << op_name << " " << dst << ", " << src;
    }
};

// Reduction operation
struct ReduceOp : NPUOp {
    std::string dst;
    std::string src;

    ReduceOp(NPUOpKind k, std::string dst_, std::string src_)
        : NPUOp(k), dst(std::move(dst_)), src(std::move(src_)) {}

    void print(std::ostream& os) const override {
        const char* op_name;
        switch (kind) {
            case NPUOpKind::RowSum: op_name = "rowsum"; break;
            case NPUOpKind::RowMax: op_name = "rowmax"; break;
            case NPUOpKind::RowMin: op_name = "rowmin"; break;
            case NPUOpKind::RowMean: op_name = "rowmean"; break;
            default: op_name = "reduce"; break;
        }
        os << op_name << " " << dst << ", " << src;
    }
};

// Broadcast operation
struct BroadcastOp : NPUOp {
    std::string dst;
    std::string a;
    std::string b;

    BroadcastOp(NPUOpKind k, std::string dst_, std::string a_, std::string b_)
        : NPUOp(k), dst(std::move(dst_)), a(std::move(a_)), b(std::move(b_)) {}

    void print(std::ostream& os) const override {
        const char* op_name;
        switch (kind) {
            case NPUOpKind::RowExpandMul: op_name = "rowexpandmul"; break;
            case NPUOpKind::RowExpandAdd: op_name = "rowexpandadd"; break;
            case NPUOpKind::RowExpandSub: op_name = "rowexpandsub"; break;
            default: op_name = "broadcast"; break;
        }
        os << op_name << " " << dst << ", " << a << ", " << b;
    }
};

// Matrix multiplication
struct MatmulOp : NPUOp {
    std::string dst;
    std::string a;
    std::string b;
    std::string acc;  // Optional accumulator
    bool use_cube;

    MatmulOp(std::string dst_, std::string a_, std::string b_,
             std::string acc_ = "", bool cube = true)
        : NPUOp(NPUOpKind::Matmul), dst(std::move(dst_)), a(std::move(a_)),
          b(std::move(b_)), acc(std::move(acc_)), use_cube(cube) {}

    void print(std::ostream& os) const override {
        os << "matmul " << dst << ", " << a << ", " << b;
        if (!acc.empty()) os << ", " << acc;
        if (!use_cube) os << " vector_only";
    }
};

// For loop begin
struct ForLoopBeginOp : NPUOp {
    std::string iv;  // Loop variable
    int64_t lb;      // Lower bound
    int64_t ub;      // Upper bound
    int64_t step;
    std::unordered_map<std::string, std::string> attrs;

    ForLoopBeginOp(std::string iv_, int64_t lb_, int64_t ub_, int64_t step_ = 1)
        : NPUOp(NPUOpKind::ForLoopBegin), iv(std::move(iv_)), lb(lb_), ub(ub_), step(step_) {}

    void print(std::ostream& os) const override {
        os << "for " << iv << " = " << lb << " to " << ub;
        if (step != 1) os << " step " << step;
        os << " {";
    }
};

// For loop end
struct ForLoopEndOp : NPUOp {
    ForLoopEndOp() : NPUOp(NPUOpKind::ForLoopEnd) {}

    void print(std::ostream& os) const override {
        os << "}";
    }
};

// If-then begin
struct IfThenBeginOp : NPUOp {
    std::string cond;

    IfThenBeginOp(std::string c) : NPUOp(NPUOpKind::IfThenBegin), cond(std::move(c)) {}

    void print(std::ostream& os) const override {
        os << "if " << cond << " {";
    }
};

// Else branch
struct ElseBranchOp : NPUOp {
    ElseBranchOp() : NPUOp(NPUOpKind::ElseBranch) {}

    void print(std::ostream& os) const override {
        os << "} else {";
    }
};

// If-then end
struct IfThenEndOp : NPUOp {
    IfThenEndOp() : NPUOp(NPUOpKind::IfThenEnd) {}

    void print(std::ostream& os) const override {
        os << "}";
    }
};

// Schedule hints (NPU-specific annotations)
struct ScheduleHints {
    std::unordered_map<std::string, int64_t> tile_policy;  // tm, tn, tk
    std::unordered_map<std::string, int64_t> double_buffer;  // tile -> depth
    struct PipelineHint {
        int stages;
        std::vector<std::string> overlap;
    };
    std::unordered_map<std::string, PipelineHint> pipeline;  // loop -> hint
    int64_t sram_budget_kb = 0;  // 0 = unspecified
    bool use_cube = true;
};

// NPU Function - complete kernel definition
struct NPUFunction {
    std::string name;
    std::vector<TileDecl> tiles;
    std::vector<ScalarDecl> scalars;
    std::vector<MemrefDecl> memrefs;
    std::vector<std::unique_ptr<NPUOp>> ops;
    ScheduleHints schedule;
    int kernel_id = -1;

    NPUFunction() = default;
    NPUFunction(const NPUFunction&) = delete;
    NPUFunction& operator=(const NPUFunction&) = delete;
    NPUFunction(NPUFunction&&) = default;
    NPUFunction& operator=(NPUFunction&&) = default;

    void print(std::ostream& os) const {
        os << "@npu_kernel " << name << " {\n";

        // Declarations
        for (const auto& t : tiles) {
            os << "  ";
            t.print(os);
            os << "\n";
        }
        for (const auto& s : scalars) {
            os << "  ";
            s.print(os);
            os << "\n";
        }
        for (const auto& m : memrefs) {
            os << "  ";
            m.print(os);
            os << "\n";
        }

        if (!tiles.empty() || !scalars.empty() || !memrefs.empty()) {
            os << "\n";
        }

        // Operations
        int indent = 2;
        for (const auto& op : ops) {
            if (op->kind == NPUOpKind::ForLoopEnd ||
                op->kind == NPUOpKind::ElseBranch ||
                op->kind == NPUOpKind::IfThenEnd) {
                indent = std::max(2, indent - 2);
            }

            os << std::string(indent, ' ');
            op->print(os);
            os << "\n";

            if (op->kind == NPUOpKind::ForLoopBegin ||
                op->kind == NPUOpKind::IfThenBegin ||
                op->kind == NPUOpKind::ElseBranch) {
                indent += 2;
            }
        }

        os << "}\n";
    }
};

// NPU Module - container for kernel definitions
struct NPUModule {
    std::string name;
    std::vector<std::unique_ptr<NPUFunction>> functions;
    int kernel_id_counter = 0;

    int registerFunction(std::unique_ptr<NPUFunction> func) {
        int id = kernel_id_counter++;
        func->kernel_id = id;
        functions.push_back(std::move(func));
        return id;
    }

    [[nodiscard]] const NPUFunction* findFunction(const std::string& n) const {
        for (const auto& f : functions) {
            if (f->name == n) return f.get();
        }
        return nullptr;
    }

    void print(std::ostream& os) const {
        os << "// NPU Module: " << name << "\n\n";
        for (const auto& f : functions) {
            f->print(os);
            os << "\n";
        }
    }
};

// Forward declarations for utility functions
const char* dtypeToString(DType dtype);
const char* locationToString(Location loc);

}  // namespace pto::wsp::ir
