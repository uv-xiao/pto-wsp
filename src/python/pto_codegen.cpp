// Copyright 2026 PTO-RT Authors
// SPDX-License-Identifier: MIT

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "pto/rt/codegen/cpp_builder.hpp"
#include "pto/rt/codegen/cmake_compiler.hpp"
#include "pto/rt/codegen/abi/workload_abi.hpp"
#include "pto/rt/backend/backend.hpp"
#include "pto/rt/ir/type_check.hpp"
#include "pto/rt/ir/codegen.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <dlfcn.h>
#include <functional>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace {

namespace fs = std::filesystem;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

std::string dtype_to_cpp_workload(const std::string& dtype) {
    if (dtype == "f32") return "float";
    if (dtype == "f16") return "_Float16";
    if (dtype == "i32") return "int32_t";
    if (dtype == "i64") return "int64_t";
    if (dtype == "u32") return "uint32_t";
    if (dtype == "u64") return "uint64_t";
    if (dtype == "bool") return "bool";
    throw std::runtime_error("Unsupported dtype for workload codegen: " + dtype);
}

std::string dtype_to_cpp_pto_elem(const std::string& dtype) {
    // PTO-ISA CPU sim defines `half` in pto/common/type.hpp.
    if (dtype == "f32") return "float";
    if (dtype == "f16") return "half";
    if (dtype == "i32") return "int32_t";
    if (dtype == "i64") return "int64_t";
    if (dtype == "u32") return "uint32_t";
    if (dtype == "u64") return "uint64_t";
    throw std::runtime_error("Unsupported dtype for PTO-ISA codegen: " + dtype);
}

int dtype_size_bytes(const std::string& dtype) {
    if (dtype == "f16") return 2;
    if (dtype == "f32") return 4;
    if (dtype == "i32") return 4;
    if (dtype == "i64") return 8;
    if (dtype == "u32") return 4;
    if (dtype == "u64") return 8;
    if (dtype == "bool") return 1;
    throw std::runtime_error("Unsupported dtype size: " + dtype);
}

std::string dtype_to_str(const pto::wsp::ir::DType dt) {
    using pto::wsp::ir::DType;
    switch (dt) {
        case DType::F16: return "f16";
        case DType::F32: return "f32";
        case DType::I32: return "i32";
        case DType::I64: return "i64";
        case DType::U32: return "u32";
        case DType::U64: return "u64";
        case DType::Bool: return "bool";
        default: break;
    }
    throw std::runtime_error("Unsupported dtype for codegen: " + std::to_string((int)dt));
}

std::string dtype_to_cpp_workload(const pto::wsp::ir::DType dt) {
    return dtype_to_cpp_workload(dtype_to_str(dt));
}

std::string dtype_to_cpp_pto_elem(const pto::wsp::ir::DType dt) {
    return dtype_to_cpp_pto_elem(dtype_to_str(dt));
}

int dtype_size_bytes(const pto::wsp::ir::DType dt) {
    return dtype_size_bytes(dtype_to_str(dt));
}

int round_up(int n, int multiple) {
    if (multiple <= 0) throw std::runtime_error("round_up: multiple <= 0");
    return ((n + multiple - 1) / multiple) * multiple;
}

int vec_col_multiple(const std::string& dtype) {
    // PTO-ISA Vec tile requires Cols * sizeof(DType) to be 32B aligned.
    return 32 / dtype_size_bytes(dtype);
}

int vec_col_multiple(const pto::wsp::ir::DType dt) {
    return 32 / dtype_size_bytes(dt);
}

uint64_t fnv1a_64(std::string_view s) {
    uint64_t h = 0xCBF29CE484222325ULL;
    for (const unsigned char b : s) {
        h ^= static_cast<uint64_t>(b);
        h *= 0x100000001B3ULL;
    }
    return h;
}

uint64_t fnv1a64_update(uint64_t h, std::string_view data) {
    constexpr uint64_t kPrime = 1099511628211ULL;
    for (unsigned char c : data) {
        h ^= static_cast<uint64_t>(c);
        h *= kPrime;
    }
    return h;
}

std::string hex16(uint64_t v) {
    static constexpr char kHex[] = "0123456789abcdef";
    std::string out(16, '0');
    for (int i = 15; i >= 0; --i) {
        out[i] = kHex[v & 0xF];
        v >>= 4;
    }
    return out;
}

fs::path codegen_cache_dir() {
    if (const char* p = std::getenv("PTO_WSP_CODEGEN_CACHE_DIR"); p && *p) {
        return fs::path(p);
    }
    const char* home = std::getenv("HOME");
    if (!home || !*home) {
        throw std::runtime_error("PTO_WSP_CODEGEN_CACHE_DIR not set and HOME is missing");
    }
    return fs::path(home) / ".cache" / "pto_wsp" / "codegen";
}

void write_text(const fs::path& p, const std::string& s) {
    fs::create_directories(p.parent_path());
    std::ofstream out(p, std::ios::binary);
    if (!out) {
        throw std::runtime_error("Failed to write file: " + p.string());
    }
    out.write(s.data(), static_cast<std::streamsize>(s.size()));
}

struct EmittedSourceTree {
    std::string dir;
    std::string cache_key;
};

EmittedSourceTree emit_sources_to_cache(const std::map<std::string, std::string>& sources,
                                        const std::string& output_name,
                                        const std::string& kind) {
    uint64_t h = 1469598103934665603ULL;
    h = fnv1a64_update(h, std::string("v1-") + kind);
    h = fnv1a64_update(h, output_name);
    for (const auto& [name, content] : sources) {
        h = fnv1a64_update(h, name);
        h = fnv1a64_update(h, content);
    }

    const std::string key = hex16(h);
    const fs::path root = codegen_cache_dir() / ("src_" + kind + "_" + output_name + "_" + key);
    fs::create_directories(root);

    for (const auto& [name, content] : sources) {
        write_text(root / name, content);
    }

    return EmittedSourceTree{root.string(), key};
}

std::string axis_arg_to_u64_literal(const pto::wsp::ir::CodegenAxisArg& a) {
    if (a.is_var) return "(uint64_t)" + a.var;
    if (!a.var.empty()) {
        std::ostringstream oss;
        oss << "ctx->get_symbol_u64(ctx->ctx, (uint64_t)0x" << std::hex << a.u64 << "ULL)";
        return oss.str();
    }
    std::ostringstream oss;
    oss << "(uint64_t)0x" << std::hex << a.u64 << "ULL";
    return oss.str();
}

std::string scalar_type_to_cpp_expr_type(pto::wsp::ir::ScalarType t) {
    switch (t) {
    case pto::wsp::ir::ScalarType::Bool: return "bool";
    case pto::wsp::ir::ScalarType::I64: return "int64_t";
    case pto::wsp::ir::ScalarType::U64: return "uint64_t";
    }
    return "uint64_t";
}

std::string u64_hex_literal(uint64_t v) {
    std::ostringstream oss;
    oss << "(uint64_t)0x" << std::hex << v << "ULL";
    return oss.str();
}

std::string emit_scalar_expr_cpp(
    const pto::wsp::ir::ScalarExpr& e,
    const std::function<std::optional<std::string>(uint64_t)>& axis_lookup,
    const std::function<std::string(uint64_t)>& sym_u64_expr,
    const std::function<std::string(uint32_t, pto::wsp::ir::ScalarType)>& slot_expr = nullptr,
    const std::function<std::string(uint32_t, pto::wsp::ir::ScalarType)>& task_param_expr = nullptr,
    const std::function<std::string(uint64_t)>& task_tag_u64_expr = nullptr) {
    using namespace pto::wsp::ir;
    if (!e) throw std::runtime_error("emit_scalar_expr_cpp: null ScalarExpr");

    switch (e->kind) {
    case ScalarExprKind::LiteralBool: {
        const auto& n = static_cast<const LiteralBoolExpr&>(*e);
        return n.value ? "true" : "false";
    }
    case ScalarExprKind::LiteralI64: {
        const auto& n = static_cast<const LiteralI64Expr&>(*e);
        return "(int64_t)" + std::to_string(n.value);
    }
    case ScalarExprKind::LiteralU64: {
        const auto& n = static_cast<const LiteralU64Expr&>(*e);
        return u64_hex_literal(n.value);
    }
    case ScalarExprKind::TaskParam: {
        const auto& n = static_cast<const TaskParamExpr&>(*e);
        if (!task_param_expr) {
            throw std::runtime_error("emit_scalar_expr_cpp: task_param() requires task_param_expr callback");
        }
        return task_param_expr(n.index, n.type);
    }
    case ScalarExprKind::TaskTagU64: {
        const auto& n = static_cast<const TaskTagU64Expr&>(*e);
        if (!task_tag_u64_expr) {
            throw std::runtime_error("emit_scalar_expr_cpp: task_tag_u64() requires task_tag_u64_expr callback");
        }
        return task_tag_u64_expr(n.tag_id);
    }
    case ScalarExprKind::AxisVar: {
        const auto& n = static_cast<const AxisVarExpr&>(*e);
        const auto var = axis_lookup(n.axis_id);
        if (!var) throw std::runtime_error("emit_scalar_expr_cpp: axis_var not in scope");
        if (n.type == ScalarType::I64) return "(int64_t)" + *var;
        if (n.type == ScalarType::U64) return "(uint64_t)" + *var;
        throw std::runtime_error("emit_scalar_expr_cpp: axis_var invalid type");
    }
    case ScalarExprKind::SymbolU64: {
        const auto& n = static_cast<const SymbolU64Expr&>(*e);
        return sym_u64_expr(n.symbol_id);
    }
    case ScalarExprKind::Slot: {
        const auto& n = static_cast<const SlotExpr&>(*e);
        if (!slot_expr) {
            throw std::runtime_error("emit_scalar_expr_cpp: slot_* requires slot_expr callback");
        }
        return slot_expr(n.index, n.type);
    }
    case ScalarExprKind::Unary: {
        const auto& n = static_cast<const UnaryExpr&>(*e);
        const std::string inner = emit_scalar_expr_cpp(n.expr, axis_lookup, sym_u64_expr, slot_expr, task_param_expr, task_tag_u64_expr);
        switch (n.op) {
        case ScalarUnaryOp::Not: return "(!" + inner + ")";
        case ScalarUnaryOp::Neg: return "(-" + inner + ")";
        case ScalarUnaryOp::BitNot: return "(~" + inner + ")";
        }
        throw std::runtime_error("emit_scalar_expr_cpp: unknown unary op");
    }
    case ScalarExprKind::Binary: {
        const auto& n = static_cast<const BinaryExpr&>(*e);
        const std::string a = emit_scalar_expr_cpp(n.lhs, axis_lookup, sym_u64_expr, slot_expr, task_param_expr, task_tag_u64_expr);
        const std::string b = emit_scalar_expr_cpp(n.rhs, axis_lookup, sym_u64_expr, slot_expr, task_param_expr, task_tag_u64_expr);
        return "(" + a + " " + binaryOpToString(n.op) + " " + b + ")";
    }
    case ScalarExprKind::Ternary: {
        const auto& n = static_cast<const TernaryExpr&>(*e);
        const std::string c = emit_scalar_expr_cpp(n.cond, axis_lookup, sym_u64_expr, slot_expr, task_param_expr, task_tag_u64_expr);
        const std::string t = emit_scalar_expr_cpp(n.then_expr, axis_lookup, sym_u64_expr, slot_expr, task_param_expr, task_tag_u64_expr);
        const std::string f = emit_scalar_expr_cpp(n.else_expr, axis_lookup, sym_u64_expr, slot_expr, task_param_expr, task_tag_u64_expr);
        return "(" + c + " ? " + t + " : " + f + ")";
    }
    case ScalarExprKind::Cast: {
        const auto& n = static_cast<const CastExpr&>(*e);
        const std::string ty = scalar_type_to_cpp_expr_type(n.to);
        const std::string inner = emit_scalar_expr_cpp(n.expr, axis_lookup, sym_u64_expr, slot_expr, task_param_expr, task_tag_u64_expr);
        return "(" + ty + ")(" + inner + ")";
    }
    }

    throw std::runtime_error("emit_scalar_expr_cpp: unknown ScalarExpr kind");
}

std::string index_expr_to_cpp(const pto::wsp::ir::CodegenIndexExpr& e) {
    if (e.is_var) return e.var;
    return std::to_string(e.constant);
}

// ---------------------------------------------------------------------------
// Kernel plan parsing
// ---------------------------------------------------------------------------

struct ValueInfo {
    std::string dtype;
    std::optional<std::pair<int, int>> shape;  // rows, cols
};

struct ParamInfo {
    std::string name;
    int id = 0;
    ValueInfo v;
    int tensor_param_index = -1;  // for task->tensor_ptrs/strides
    int scalar_param_index = -1;  // for task->args[task->num_axis_args + idx]
};

struct OpInfo {
    std::string kind;
    std::optional<int> result;
    std::vector<int> operands;
    py::dict attrs;
};

struct KernelPlan {
    std::string name;
    std::vector<ParamInfo> params;
    std::unordered_map<int, ValueInfo> values;
    std::vector<OpInfo> ops;
};

// ---------------------------------------------------------------------------
// Kernel plan from IR
// ---------------------------------------------------------------------------

KernelPlan kernel_plan_from_ir(const pto::wsp::ir::CodegenKernelDef& kd) {
    KernelPlan kp;
    kp.name = kd.name;

    // Values
    for (const auto& [vid, vi] : kd.values) {
        ValueInfo info;
        info.dtype = dtype_to_str(vi.dtype);
        if (!vi.has_shape) {
            info.shape = std::nullopt;
        } else {
            info.shape = std::make_pair(vi.rows, vi.cols);
        }
        kp.values.emplace(vid, std::move(info));
    }

    // Params
    {
        int tensor_idx = 0;
        int scalar_idx = 0;
        kp.params.reserve(kd.params.size());
        for (const auto& p : kd.params) {
            ParamInfo pi;
            pi.name = p.name;
            pi.id = p.id;
            auto it = kp.values.find(pi.id);
            if (it == kp.values.end()) throw std::runtime_error("Kernel param missing from values map");
            pi.v = it->second;
            if (pi.v.shape.has_value()) {
                pi.tensor_param_index = tensor_idx++;
            } else {
                pi.scalar_param_index = scalar_idx++;
            }
            kp.params.push_back(std::move(pi));
        }
    }

    // Ops
    {
        kp.ops.reserve(kd.ops.size());
        for (const auto& o : kd.ops) {
            OpInfo oi;
            oi.kind = o.kind;
            if (o.has_result) oi.result = o.result;
            else oi.result = std::nullopt;
            oi.operands = o.operands;

            py::dict attrs;
            for (const auto& [k, v] : o.attrs) {
                std::visit(
                    [&](auto&& arg) {
                        using T = std::decay_t<decltype(arg)>;
                        if constexpr (std::is_same_v<T, int64_t> ||
                                      std::is_same_v<T, double> ||
                                      std::is_same_v<T, bool> ||
                                      std::is_same_v<T, std::string> ||
                                      std::is_same_v<T, std::vector<int64_t>> ||
                                      std::is_same_v<T, std::vector<std::string>>) {
                            attrs[py::str(k)] = py::cast(arg);
                        } else {
                            throw std::runtime_error("Unsupported attr type in kernel op");
                        }
                    },
                    v);
            }
            oi.attrs = std::move(attrs);
            kp.ops.push_back(std::move(oi));
        }
    }

    return kp;
}

// ---------------------------------------------------------------------------
// Kernel C++ emission (PTO-ISA CPU sim)
// ---------------------------------------------------------------------------

struct LocalTile {
    int id = 0;
    std::string dtype;
    std::pair<int, int> shape;       // logical
    std::pair<int, int> full_shape;  // padded
    std::string kind;                // "vec" | "acc"
};

std::unordered_map<int, std::string> infer_tile_kinds(
    const KernelPlan& kp,
    const std::unordered_map<int, ParamInfo>& param_by_id) {
    std::unordered_map<int, std::string> kind_by_id;

    (void)param_by_id;
    for (const auto& op : kp.ops) {
        if (!op.result.has_value()) continue;
        auto itv = kp.values.find(*op.result);
        if (itv == kp.values.end() || !itv->second.shape.has_value()) continue;
        // Only Matmul produces accumulator tiles. All other ops run on Vec tiles;
        // if a Matmul result feeds non-matmul ops, we materialize a Vec view via TMOV.
        kind_by_id[*op.result] = (op.kind == "Matmul") ? "acc" : "vec";
    }

    return kind_by_id;
}

std::set<int> collect_param_tiles_to_load(
    const KernelPlan& kp,
    const std::unordered_map<int, ParamInfo>& param_by_id) {
    std::set<int> used;
    for (const auto& op : kp.ops) {
        if (op.kind == "Matmul" || op.kind == "Load") continue;
        for (size_t i = 0; i < op.operands.size(); ++i) {
            const int oid = op.operands[i];
            auto itv = kp.values.find(oid);
            if (itv == kp.values.end() || !itv->second.shape.has_value()) continue;
            if (param_by_id.find(oid) == param_by_id.end()) continue;
            if (op.kind == "Store" && i == 0) continue;  // store dst
            used.insert(oid);
        }
    }
    return used;
}

std::string emit_kernel_cpp(const KernelPlan& kp) {
    namespace cg = pto::wsp::codegen::cpp;
    constexpr int kMatmulAlign = 16;

    cg::TranslationUnit tu;
    tu.includes = {
        "\"pto/rt/codegen/abi/kernel_abi.hpp\"",
        "\"pto/rt/codegen/abi/ptoisa_bridge.hpp\"",
        "<cstdint>",
        "<cstddef>",
        "<memory>",
    };

    cg::FunctionBuilder fn("PTO_WSP_KERNEL_ATTR uint64_t", kp.name);
    fn.extern_c().param("const KernelTaskDesc*", "task").param("CSPTContext*", "cspt");

    struct FnWriter {
        cg::FunctionBuilder* fn = nullptr;
        std::string cur;

        static std::string format_double(double v) {
            char buf[128];
            const int n = std::snprintf(buf, sizeof(buf), "%.17g", v);
            if (n > 0 && static_cast<size_t>(n) < sizeof(buf)) return std::string(buf, buf + n);
            return std::to_string(v);
        }

        void push_line(std::string line) {
            if (line.rfind("  ", 0) == 0) line.erase(0, 2);  // strip function-base indent
            fn->stmt(cg::raw_stmt(std::move(line)));
        }

        void append(std::string_view sv) {
            size_t start = 0;
            for (size_t i = 0; i < sv.size(); ++i) {
                if (sv[i] == '\n') {
                    cur.append(sv.substr(start, i - start));
                    push_line(std::move(cur));
                    cur.clear();
                    start = i + 1;
                }
            }
            cur.append(sv.substr(start));
        }

        void flush() {
            if (!cur.empty()) {
                push_line(std::move(cur));
                cur.clear();
            }
        }

        FnWriter& operator<<(const char* s) {
            append(std::string_view(s ? s : ""));
            return *this;
        }
        FnWriter& operator<<(const std::string& s) {
            append(std::string_view(s));
            return *this;
        }
        FnWriter& operator<<(std::string_view s) {
            append(s);
            return *this;
        }
        FnWriter& operator<<(int v) {
            append(std::to_string(v));
            return *this;
        }
        FnWriter& operator<<(uint32_t v) {
            append(std::to_string(v));
            return *this;
        }
        FnWriter& operator<<(uint64_t v) {
            append(std::to_string(v));
            return *this;
        }
        FnWriter& operator<<(double v) {
            append(format_double(v));
            return *this;
        }
    };

    FnWriter out;
    out.fn = &fn;

    out << "  uint64_t cycles = 0;\n";
    out << "#if defined(__CPU_SIM)\n";
    out << "  pto::cpu_sim::reset_cycles();\n";
    out << "#endif\n";

    // Build param map by id.
    std::unordered_map<int, ParamInfo> param_by_id;
    param_by_id.reserve(kp.params.size());
    for (const auto& p : kp.params) {
        param_by_id.emplace(p.id, p);
    }

    // Scalar values (ValueId -> C++ variable name). Includes kernel scalar params
    // (from task->args) and Constant() ops.
    std::unordered_map<int, std::string> scalar_const;

    // Scalar params (from kernel signature) are passed in task->args after axis args.
    for (const auto& p : kp.params) {
        if (p.v.shape.has_value()) continue;
        if (p.scalar_param_index < 0) continue;
        const int rid = p.id;
        const std::string var = "s" + std::to_string(rid);

        if (p.v.dtype == "f32") {
            scalar_const.emplace(rid, var);
            out << "  union { uint32_t u32; float f32; } _u" << rid << ";\n";
            out << "  _u" << rid << ".u32 = (uint32_t)task->args[task->num_axis_args + "
                << p.scalar_param_index << "];\n";
            out << "  float " << var << " = _u" << rid << ".f32;\n";
        } else if (p.v.dtype == "i32") {
            scalar_const.emplace(rid, var);
            out << "  int32_t " << var << " = (int32_t)task->args[task->num_axis_args + "
                << p.scalar_param_index << "];\n";
        } else if (p.v.dtype == "i64") {
            scalar_const.emplace(rid, var);
            out << "  int64_t " << var << " = (int64_t)task->args[task->num_axis_args + "
                << p.scalar_param_index << "];\n";
        } else if (p.v.dtype == "bool") {
            scalar_const.emplace(rid, var);
            out << "  bool " << var << " = task->args[task->num_axis_args + " << p.scalar_param_index
                << "] != 0;\n";
        } else {
            out << "  // Unsupported scalar param dtype: " << p.v.dtype << "\n";
        }
    }
    if (!scalar_const.empty()) out << "\n";

    for (const auto& op : kp.ops) {
        if (op.kind != "Constant" || !op.result.has_value()) continue;
        const int rid = *op.result;
        auto itv = kp.values.find(rid);
        if (itv == kp.values.end() || itv->second.shape.has_value()) continue;
        if (!op.attrs.contains("value")) continue;

        const std::string var = "s" + std::to_string(rid);
        scalar_const.emplace(rid, var);

        const std::string elem = dtype_to_cpp_pto_elem(itv->second.dtype);
        const double v = py::cast<double>(op.attrs["value"]);
        out << "  " << elem << " " << var << " = (" << elem << ")" << v << ";\n";
    }
    if (!scalar_const.empty()) out << "\n";

    // v9 tail/mask support:
    // Workload emission appends per-tensor (valid_row, valid_col) pairs as axis
    // args, in kernel tensor-param order. Kernels use these to size GlobalTensor
    // shapes and set Tile valid dims (enables partial tiles without whole-artifact
    // rebuild). If the tail args are not present, fall back to static shapes.
    int num_tensor_params = 0;
    for (const auto& p : kp.params) {
        if (p.v.shape.has_value()) ++num_tensor_params;
    }
    if (num_tensor_params > 0) {
        out << "  uint32_t _pto_wsp_tail_base = 0;\n";
        out << "  if (task->num_axis_args >= " << (num_tensor_params * 2) << ") {\n";
        out << "    _pto_wsp_tail_base = task->num_axis_args - " << (num_tensor_params * 2) << ";\n";
        out << "  }\n";
        for (const auto& p : kp.params) {
            if (!p.v.shape.has_value()) continue;
            const auto [r, c] = *p.v.shape;
            out << "  int " << p.name << "_vr = " << r << ";\n";
            out << "  int " << p.name << "_vc = " << c << ";\n";
        }
        out << "  if (task->num_axis_args >= " << (num_tensor_params * 2) << ") {\n";
        for (const auto& p : kp.params) {
            if (!p.v.shape.has_value()) continue;
            out << "    " << p.name << "_vr = (int)task->args[_pto_wsp_tail_base + " << (p.tensor_param_index * 2)
                << "];\n";
            out << "    " << p.name << "_vc = (int)task->args[_pto_wsp_tail_base + " << (p.tensor_param_index * 2 + 1)
                << "];\n";
        }
        out << "  }\n\n";
    }

    // Emit tensor param pointer extraction.
    for (const auto& p : kp.params) {
        if (!p.v.shape.has_value()) continue;
        const auto elem = dtype_to_cpp_pto_elem(p.v.dtype);
        out << "  " << elem << "* " << p.name << " = ("
            << elem << "*)task->tensor_ptrs[" << p.tensor_param_index << "];\n";
        out << "  uint64_t " << p.name << "_s3 = task->tensor_strides[" << (p.tensor_param_index * 2) << "];\n";
        out << "  uint64_t " << p.name << "_s4 = task->tensor_strides[" << (p.tensor_param_index * 2 + 1) << "];\n";
    }
    out << "\n";

    const auto kind_by_id = infer_tile_kinds(kp, param_by_id);
    const auto param_tiles_to_load = collect_param_tiles_to_load(kp, param_by_id);

    // Allocate local tiles (intermediates + param tiles that we load).
    std::unordered_map<int, LocalTile> local_tiles;
    auto add_tile = [&](int id, const ValueInfo& vinfo, const std::string& kind) {
        if (!vinfo.shape.has_value()) return;
        const auto [r, c] = *vinfo.shape;
        LocalTile t;
        t.id = id;
        t.dtype = vinfo.dtype;
        t.shape = {r, c};
        t.kind = kind;
        if (kind == "acc") {
            t.full_shape = {round_up(r, kMatmulAlign), round_up(c, kMatmulAlign)};
        } else {
            t.full_shape = {r, round_up(c, vec_col_multiple(vinfo.dtype))};
        }
        local_tiles.emplace(id, std::move(t));
    };

    for (const auto& op : kp.ops) {
        if (!op.result.has_value()) continue;
        const int rid = *op.result;
        if (param_by_id.find(rid) != param_by_id.end()) continue;
        auto itv = kp.values.find(rid);
        if (itv == kp.values.end()) continue;
        if (!itv->second.shape.has_value()) continue;
        const auto it_kind = kind_by_id.find(rid);
        add_tile(rid, itv->second, (it_kind == kind_by_id.end()) ? "vec" : it_kind->second);
    }

    for (int pid : param_tiles_to_load) {
        auto itp = param_by_id.find(pid);
        if (itp == param_by_id.end()) continue;
        const auto it_kind = kind_by_id.find(pid);
        add_tile(pid, itp->second.v, (it_kind == kind_by_id.end()) ? "vec" : it_kind->second);
    }

    // If a matmul accumulator tile feeds non-matmul ops, materialize a Vec view via TMOV.
    std::unordered_set<int> acc_needs_vec;
    for (const auto& op : kp.ops) {
        if (op.kind == "Matmul" || op.kind == "Store") continue;
        for (int oid : op.operands) {
            auto it = local_tiles.find(oid);
            if (it != local_tiles.end() && it->second.kind == "acc") {
                acc_needs_vec.insert(oid);
            }
        }
    }

    // Runtime valid dims for local tiles that must match a tensor param (e.g. Store dst).
    std::unordered_map<int, std::pair<std::string, std::string>> runtime_dims_by_tile;
    for (const auto& op : kp.ops) {
        if (op.kind != "Store" || op.operands.size() < 2) continue;
        const int dst_id = op.operands[0];
        const int src_id = op.operands[1];
        auto it_dst = param_by_id.find(dst_id);
        if (it_dst == param_by_id.end() || !it_dst->second.v.shape.has_value()) continue;
        runtime_dims_by_tile[src_id] = {it_dst->second.name + "_vr", it_dst->second.name + "_vc"};
    }

    auto tile_expr = [&](int id) -> std::string {
        if (acc_needs_vec.find(id) != acc_needs_vec.end()) {
            return "*v" + std::to_string(id) + "_vec";
        }
        return "*v" + std::to_string(id);
    };

    for (const auto& [id, t] : local_tiles) {
        const auto elem = dtype_to_cpp_pto_elem(t.dtype);
        const auto [r, c] = t.shape;
        const auto [fr, fc] = t.full_shape;
        std::string r_expr = std::to_string(r);
        std::string c_expr = std::to_string(c);
        if (auto it = runtime_dims_by_tile.find(id); it != runtime_dims_by_tile.end()) {
            r_expr = it->second.first;
            c_expr = it->second.second;
        } else if (auto itp = param_by_id.find(id); itp != param_by_id.end() && itp->second.v.shape.has_value()) {
            r_expr = itp->second.name + "_vr";
            c_expr = itp->second.name + "_vc";
        }
        if (t.kind == "acc") {
            out << "  auto v" << id << " = std::make_unique<pto::TileAcc<" << elem
                << ", " << fr << ", " << fc << ", pto::DYNAMIC, pto::DYNAMIC>>((size_t)"
                << r_expr << ", (size_t)" << c_expr << ");\n";
            if (acc_needs_vec.find(id) != acc_needs_vec.end()) {
                const int fr_vec = r;
                const int fc_vec = round_up(c, vec_col_multiple(t.dtype));
                out << "  auto v" << id << "_vec = std::make_unique<pto::Tile<pto::TileType::Vec, "
                    << elem << ", " << fr_vec << ", " << fc_vec
                    << ", pto::BLayout::RowMajor, pto::DYNAMIC, pto::DYNAMIC>>((size_t)"
                    << r_expr << ", (size_t)" << c_expr << ");\n";
            }
        } else {
            out << "  auto v" << id << " = std::make_unique<pto::Tile<pto::TileType::Vec, "
                << elem << ", " << fr << ", " << fc
                << ", pto::BLayout::RowMajor, pto::DYNAMIC, pto::DYNAMIC>>((size_t)"
                << r_expr << ", (size_t)" << c_expr << ");\n";
        }
    }
    out << "\n";

    // Load any param tiles required by ops.
    for (int pid : param_tiles_to_load) {
        const auto itp = param_by_id.find(pid);
        if (itp == param_by_id.end()) continue;
        if (!itp->second.v.shape.has_value()) continue;
        const auto elem = dtype_to_cpp_pto_elem(itp->second.v.dtype);
        out << "  using _GTShape_load_" << pid << " = pto::Shape<1, 1, 1, pto::DYNAMIC, pto::DYNAMIC>;\n";
        out << "  using _GTStride_load_" << pid << " = pto::Stride<1, 1, 1, pto::DYNAMIC, pto::DYNAMIC>;\n";
        out << "  pto::GlobalTensor<" << elem << ", _GTShape_load_" << pid << ", _GTStride_load_" << pid
            << ", pto::Layout::ND> g_load_" << pid << "("
            << itp->second.name << ", _GTShape_load_" << pid << "(" << itp->second.name << "_vr, " << itp->second.name << "_vc"
            << "), _GTStride_load_" << pid << "((int)" << itp->second.name << "_s3, (int)"
            << itp->second.name << "_s4));\n";
        out << "  pto_wsp::ptoisa::TLOAD(*v" << pid << ", g_load_" << pid << ");\n";
    }
    out << "\n";

    // Emit ops (subset matching the Python kernel emitter).
    int load_serial = 0;
    int store_serial = 0;
    for (const auto& op : kp.ops) {
        if (op.kind == "Load") {
            if (!op.result.has_value() || op.operands.empty()) {
                out << "  // Load missing result/operand\n";
                continue;
            }
            const int rid = *op.result;
            const int src_id = op.operands[0];
            auto it_src = param_by_id.find(src_id);
            if (it_src == param_by_id.end() || !it_src->second.v.shape.has_value()) {
                out << "  // Load src must be kernel param tile\n";
                continue;
            }
            auto it_dst = local_tiles.find(rid);
            if (it_dst == local_tiles.end()) {
                out << "  // Load dst tile not allocated\n";
                continue;
            }

            const int suffix = load_serial++;
            const auto elem = dtype_to_cpp_pto_elem(it_src->second.v.dtype);
            out << "  // Load (PTO-ISA)\n";
            out << "  using _GTShape_loadop_" << suffix << " = pto::Shape<1, 1, 1, pto::DYNAMIC, pto::DYNAMIC>;\n";
            out << "  using _GTStride_loadop_" << suffix << " = pto::Stride<1, 1, 1, pto::DYNAMIC, pto::DYNAMIC>;\n";
            out << "  pto::GlobalTensor<" << elem << ", _GTShape_loadop_" << suffix << ", _GTStride_loadop_" << suffix
                << ", pto::Layout::ND> gLoadOp_" << suffix << "(" << it_src->second.name
                << ", _GTShape_loadop_" << suffix << "(" << it_src->second.name << "_vr, " << it_src->second.name << "_vc"
                << "), _GTStride_loadop_" << suffix << "((int)" << it_src->second.name << "_s3, (int)"
                << it_src->second.name << "_s4));\n";
            out << "  pto_wsp::ptoisa::TLOAD(*v" << rid << ", gLoadOp_" << suffix << ");\n";
            continue;
        }

        if (op.kind == "Matmul") {
            if (!op.result.has_value() || op.operands.size() < 2) {
                out << "  // Matmul missing result/operands\n";
                continue;
            }
            const int rid = *op.result;
            auto itd = local_tiles.find(rid);
            if (itd == local_tiles.end()) {
                out << "  // Matmul destination not allocated\n";
                continue;
            }
            const int a_id = op.operands[0];
            const int b_id = op.operands[1];
            auto ita = param_by_id.find(a_id);
            auto itb = param_by_id.find(b_id);

            const bool a_is_param = ita != param_by_id.end();
            const bool b_is_param = itb != param_by_id.end();

            auto get_shape = [&](int oid) -> std::optional<std::pair<int, int>> {
                if (auto itp = param_by_id.find(oid); itp != param_by_id.end()) return itp->second.v.shape;
                if (auto itl = local_tiles.find(oid); itl != local_tiles.end()) return itl->second.shape;
                return std::nullopt;
            };

            auto get_dtype = [&](int oid) -> std::optional<std::string> {
                if (auto itp = param_by_id.find(oid); itp != param_by_id.end()) return itp->second.v.dtype;
                if (auto itl = local_tiles.find(oid); itl != local_tiles.end()) return itl->second.dtype;
                return std::nullopt;
            };

            const auto a_shape = get_shape(a_id);
            const auto b_shape = get_shape(b_id);
            const auto a_dtype = get_dtype(a_id);
            const auto b_dtype = get_dtype(b_id);
            if (!a_shape.has_value() || !b_shape.has_value() || !a_dtype.has_value() || !b_dtype.has_value()) {
                out << "  // Matmul expects tile operands\n";
                continue;
            }
            const auto [m_static, k_static] = *a_shape;
            const auto [k2_static, n_static] = *b_shape;
            if (k_static != k2_static) {
                out << "  // Matmul K mismatch\n";
                continue;
            }

            std::string a_elem, b_elem;
            if (*a_dtype == "f32" && *b_dtype == "f32" && itd->second.dtype == "f32") {
                a_elem = "float";
                b_elem = "float";
            } else if (*a_dtype == "f16" && *b_dtype == "f16" && itd->second.dtype == "f32") {
                a_elem = "half";
                b_elem = "half";
            } else {
                out << "  // Matmul: unsupported dtype combination\n";
                continue;
            }

            const int full_m = round_up(m_static, kMatmulAlign);
            const int full_n = round_up(n_static, kMatmulAlign);
            const int full_k = round_up(k_static, kMatmulAlign);
            const int suffix = rid;

            out << "  // Matmul (PTO-ISA): " << m_static << "x" << k_static << " @ " << k_static << "x" << n_static
                << " (padded: " << full_m << "x" << full_k << " @ " << full_k << "x" << full_n << ")\n";

            std::string m_expr = std::to_string(m_static);
            std::string k_expr_a = std::to_string(k_static);
            std::string k_expr_b = std::to_string(k_static);
            std::string n_expr = std::to_string(n_static);
            if (a_is_param) {
                m_expr = ita->second.name + "_vr";
                k_expr_a = ita->second.name + "_vc";
            }
            if (b_is_param) {
                k_expr_b = itb->second.name + "_vr";
                n_expr = itb->second.name + "_vc";
            }
            out << "  pto::TileLeft<" << a_elem << ", " << full_m << ", " << full_k
                << ", pto::DYNAMIC, pto::DYNAMIC> a_tile_" << suffix << "((size_t)" << m_expr << ", (size_t)" << k_expr_a << ");\n";
            out << "  pto::TileRight<" << b_elem << ", " << full_k << ", " << full_n
                << ", pto::DYNAMIC, pto::DYNAMIC> b_tile_" << suffix << "((size_t)" << k_expr_b << ", (size_t)" << n_expr << ");\n";
            if (a_is_param || b_is_param) {
                out << "  using _GTShape_" << suffix << " = pto::Shape<1, 1, 1, pto::DYNAMIC, pto::DYNAMIC>;\n";
                out << "  using _GTStride_" << suffix << " = pto::Stride<1, 1, 1, pto::DYNAMIC, pto::DYNAMIC>;\n";
            }

            if (a_is_param) {
                out << "  pto::GlobalTensor<" << a_elem << ", _GTShape_" << suffix << ", _GTStride_" << suffix
                    << ", pto::Layout::ND> gA_" << suffix << "(" << ita->second.name
                    << ", _GTShape_" << suffix << "(" << m_expr << ", " << k_expr_a << "), _GTStride_" << suffix
                    << "((int)" << ita->second.name << "_s3, (int)" << ita->second.name << "_s4));\n";
                out << "  pto_wsp::ptoisa::TLOAD(a_tile_" << suffix << ", gA_" << suffix << ");\n";
            } else {
                out << "  pto_wsp::ptoisa::TMOV(a_tile_" << suffix << ", " << tile_expr(a_id) << ");\n";
            }

            if (b_is_param) {
                out << "  pto::GlobalTensor<" << b_elem << ", _GTShape_" << suffix << ", _GTStride_" << suffix
                    << ", pto::Layout::ND> gB_" << suffix << "(" << itb->second.name
                    << ", _GTShape_" << suffix << "(" << k_expr_b << ", " << n_expr << "), _GTStride_" << suffix
                    << "((int)" << itb->second.name << "_s3, (int)" << itb->second.name << "_s4));\n";
                out << "  pto_wsp::ptoisa::TLOAD(b_tile_" << suffix << ", gB_" << suffix << ");\n";
            } else {
                out << "  pto_wsp::ptoisa::TMOV(b_tile_" << suffix << ", " << tile_expr(b_id) << ");\n";
            }

            out << "  pto_wsp::ptoisa::TMATMUL(*v" << rid << ", a_tile_" << suffix << ", b_tile_" << suffix << ");\n";
            if (acc_needs_vec.find(rid) != acc_needs_vec.end()) {
                out << "  pto_wsp::ptoisa::TMOV(*v" << rid << "_vec, *v" << rid << ");\n";
            }
            continue;
        }

        if (op.kind == "Store") {
            const int store_suffix = store_serial++;
            if (op.operands.size() < 2) {
                out << "  // Store expects dst, src\n";
                continue;
            }
            const int dst_id = op.operands[0];
            const int src_id = op.operands[1];
            auto it_dst = param_by_id.find(dst_id);
            if (it_dst == param_by_id.end() || !it_dst->second.v.shape.has_value()) {
                out << "  // Store dst must be kernel param\n";
                continue;
            }
            const std::string store_r = it_dst->second.name + "_vr";
            const std::string store_c = it_dst->second.name + "_vc";

            auto it_src = local_tiles.find(src_id);
            if (it_src == local_tiles.end()) {
                out << "  // Store src must be local tile\n";
                continue;
            }
            const auto dst_dtype = it_dst->second.v.dtype;
            const auto src_dtype = it_src->second.dtype;

            // If needed, insert a cast tile (common: f32 accum -> f16 output).
            std::string src_tile_expr = "*v" + std::to_string(src_id);
            std::string store_elem = dtype_to_cpp_pto_elem(src_dtype);

            if (dst_dtype != src_dtype) {
                if (dst_dtype == "f16" && src_dtype == "f32") {
                    const auto [r, c] = it_src->second.shape;
                    const int full_r = r;
                    const int full_c = round_up(c, vec_col_multiple(dst_dtype));
                    out << "  auto v_store_cast_" << store_suffix
                        << " = std::make_unique<pto::Tile<pto::TileType::Vec, half, " << full_r << ", " << full_c
                        << ", pto::BLayout::RowMajor, pto::DYNAMIC, pto::DYNAMIC>>((size_t)" << store_r << ", (size_t)"
                        << store_c << ");\n";
                    out << "  pto_wsp::ptoisa::TCVT(*v_store_cast_" << store_suffix << ", *v" << src_id
                        << ", pto::RoundMode::CAST_RINT);\n";
                    src_tile_expr = "*v_store_cast_" + std::to_string(store_suffix);
                    store_elem = "half";
                } else {
                    out << "  // Store dtype mismatch (unsupported cast)\n";
                    continue;
                }
            }

            out << "  // Store (PTO-ISA)\n";
            out << "  using _GTShape_store_" << store_suffix << " = pto::Shape<1, 1, 1, pto::DYNAMIC, pto::DYNAMIC>;\n";
            out << "  using _GTStride_store_" << store_suffix << " = pto::Stride<1, 1, 1, pto::DYNAMIC, pto::DYNAMIC>;\n";
            out << "  pto::GlobalTensor<" << store_elem << ", _GTShape_store_" << store_suffix << ", _GTStride_store_"
                << store_suffix << ", pto::Layout::ND> gDst_" << store_suffix << "(" << it_dst->second.name
                << ", _GTShape_store_" << store_suffix << "(" << store_r << ", " << store_c << "), _GTStride_store_"
                << store_suffix << "((int)" << it_dst->second.name << "_s3, (int)" << it_dst->second.name
                << "_s4));\n";
            out << "  pto_wsp::ptoisa::TSTORE(gDst_" << store_suffix << ", " << src_tile_expr << ");\n";
            continue;
        }

        if (op.kind == "Add" || op.kind == "Sub" || op.kind == "Mul" || op.kind == "Div" ||
            op.kind == "Max" || op.kind == "Min") {
            if (!op.result.has_value() || op.operands.size() != 2) {
                out << "  // Binary op missing result/operands\n";
                continue;
            }
            const int rid = *op.result;
            auto it_dst = local_tiles.find(rid);
            if (it_dst == local_tiles.end()) {
                out << "  // Binary op missing dst tile\n";
                continue;
            }

            const int a_id = op.operands[0];
            const int b_id = op.operands[1];
            const bool a_is_scalar = scalar_const.find(a_id) != scalar_const.end();
            const bool b_is_scalar = scalar_const.find(b_id) != scalar_const.end();

            // Scalar-tile fast paths (constant scalar only).
            if (a_is_scalar ^ b_is_scalar) {
                const int tile_id = a_is_scalar ? b_id : a_id;
                const int scalar_id = a_is_scalar ? a_id : b_id;
                auto it_tile = local_tiles.find(tile_id);
                if (it_tile == local_tiles.end()) {
                    out << "  // Binary scalar op expects tile operand\n";
                    continue;
                }
                const std::string s = scalar_const.at(scalar_id);

                if (op.kind == "Add") {
                    out << "  pto_wsp::ptoisa::TADDS(*v" << rid << ", " << tile_expr(tile_id) << ", " << s << ");\n";
                    continue;
                }
                if (op.kind == "Mul") {
                    out << "  pto_wsp::ptoisa::TMULS(*v" << rid << ", " << tile_expr(tile_id) << ", " << s << ");\n";
                    continue;
                }
                if (op.kind == "Div") {
                    if (b_is_scalar) {
                        out << "  pto_wsp::ptoisa::TDIVS(*v" << rid << ", " << tile_expr(tile_id) << ", " << s << ");\n";
                    } else {
                        out << "  pto_wsp::ptoisa::TDIVS(*v" << rid << ", " << s << ", " << tile_expr(tile_id) << ");\n";
                    }
                    continue;
                }
                if (op.kind == "Sub") {
                    if (b_is_scalar) {
                        out << "  pto_wsp::ptoisa::TSUBS(*v" << rid << ", " << tile_expr(tile_id) << ", " << s << ");\n";
                        continue;
                    }
                    out << "  // Sub: scalar - tile not supported\n";
                    continue;
                }
                if (op.kind == "Max") {
                    out << "  pto_wsp::ptoisa::TMAXS(*v" << rid << ", " << tile_expr(tile_id) << ", " << s << ");\n";
                    continue;
                }
                if (op.kind == "Min") {
                    out << "  pto_wsp::ptoisa::TMINS(*v" << rid << ", " << tile_expr(tile_id) << ", " << s << ");\n";
                    continue;
                }
            }

            auto it_a = local_tiles.find(a_id);
            auto it_b = local_tiles.find(b_id);
            if (it_a == local_tiles.end() || it_b == local_tiles.end()) {
                out << "  // Binary op expects tile operands\n";
                continue;
            }

            const auto [r, c] = it_dst->second.shape;
            const auto [ar, ac] = it_a->second.shape;
            const auto [br, bc] = it_b->second.shape;

            std::string base, rowexpand, colexpand;
            if (op.kind == "Add") {
                base = "TADD";
                rowexpand = "TROWEXPANDADD";
            } else if (op.kind == "Sub") {
                base = "TSUB";
                rowexpand = "TROWEXPANDSUB";
                colexpand = "TCOLEXPANDSUB";
            } else if (op.kind == "Mul") {
                base = "TMUL";
                rowexpand = "TROWEXPANDMUL";
                colexpand = "TCOLEXPANDMUL";
            } else if (op.kind == "Div") {
                base = "TDIV";
                rowexpand = "TROWEXPANDDIV";
                colexpand = "TCOLEXPANDDIV";
            } else if (op.kind == "Max") {
                base = "TMAX";
                rowexpand = "TROWEXPANDMAX";
            } else {
                base = "TMIN";
                rowexpand = "TROWEXPANDMIN";
            }

            if (std::make_pair(ar, ac) == std::make_pair(r, c) && std::make_pair(br, bc) == std::make_pair(r, c)) {
                out << "  pto_wsp::ptoisa::" << base << "(*v" << rid << ", " << tile_expr(op.operands[0]) << ", "
                    << tile_expr(op.operands[1])
                    << ");\n";
                continue;
            }
            if (std::make_pair(ar, ac) == std::make_pair(r, c) && (std::make_pair(br, bc) == std::make_pair(r, 1) ||
                                                                   std::make_pair(br, bc) == std::make_pair(1, r))) {
                out << "  pto_wsp::ptoisa::" << rowexpand << "(*v" << rid << ", " << tile_expr(op.operands[0]) << ", "
                    << tile_expr(op.operands[1])
                    << ");\n";
                continue;
            }
            if (!colexpand.empty() && std::make_pair(ar, ac) == std::make_pair(r, c) &&
                (std::make_pair(br, bc) == std::make_pair(1, c) || std::make_pair(br, bc) == std::make_pair(c, 1))) {
                out << "  pto_wsp::ptoisa::" << colexpand << "(*v" << rid << ", " << tile_expr(op.operands[0]) << ", "
                    << tile_expr(op.operands[1])
                    << ");\n";
                continue;
            }
            out << "  // Binary op broadcast pattern unsupported\n";
            continue;
        }

        if (op.kind == "Exp" || op.kind == "Rsqrt") {
            if (!op.result.has_value() || op.operands.size() != 1) {
                out << "  // Unary op missing result/operands\n";
                continue;
            }
            const int rid = *op.result;
            auto it_dst = local_tiles.find(rid);
            auto it_src = local_tiles.find(op.operands[0]);
            if (it_dst == local_tiles.end() || it_src == local_tiles.end()) {
                out << "  // Unary op expects tile operand\n";
                continue;
            }
            if (op.kind == "Exp") {
                out << "  pto_wsp::ptoisa::TEXP(*v" << rid << ", " << tile_expr(op.operands[0]) << ");\n";
            } else {
                out << "  pto_wsp::ptoisa::TRSQRT(*v" << rid << ", " << tile_expr(op.operands[0]) << ");\n";
            }
            continue;
        }

        if (op.kind == "IotaU32") {
            if (!op.result.has_value()) {
                out << "  // IotaU32 missing result\n";
                continue;
            }
            const int rid = *op.result;
            auto it_dst = local_tiles.find(rid);
            if (it_dst == local_tiles.end()) {
                out << "  // IotaU32 missing dst tile\n";
                continue;
            }
            const auto [r, c] = it_dst->second.shape;
            if (r != 1) {
                out << "  // IotaU32 expects shape [1,C]\n";
                continue;
            }
            out << "  for (int _i = 0; _i < " << c << "; ++_i) { v" << rid << "->data()[_i] = (uint32_t)_i; }\n";
            continue;
        }

        if (op.kind == "TSort32") {
            if (!op.result.has_value() || op.operands.size() != 2) {
                out << "  // TSort32 missing result/operands\n";
                continue;
            }
            const int rid = *op.result;
            auto it_dst = local_tiles.find(rid);
            auto it_src = local_tiles.find(op.operands[0]);
            auto it_idx = local_tiles.find(op.operands[1]);
            if (it_dst == local_tiles.end() || it_src == local_tiles.end() || it_idx == local_tiles.end()) {
                out << "  // TSort32 expects local tile operands\n";
                continue;
            }
            out << "  pto_wsp::ptoisa::TSORT32(*v" << rid << ", " << tile_expr(op.operands[0]) << ", "
                << tile_expr(op.operands[1]) << ");\n";
            continue;
        }

        if (op.kind == "ExtractTopkIndices") {
            if (!op.result.has_value() || op.operands.size() != 1) {
                out << "  // ExtractTopkIndices missing result/operands\n";
                continue;
            }
            const int rid = *op.result;
            auto it_dst = local_tiles.find(rid);
            auto it_src = local_tiles.find(op.operands[0]);
            if (it_dst == local_tiles.end() || it_src == local_tiles.end()) {
                out << "  // ExtractTopkIndices expects local tile operand\n";
                continue;
            }
            int k = 0;
            if (op.attrs.contains("k")) {
                k = py::cast<int>(op.attrs["k"]);
            }
            if (k <= 0) k = it_dst->second.shape.second;
            out << "  for (int _i = 0; _i < " << k << "; ++_i) {\n";
            out << "    const float _idx_f = v" << op.operands[0] << "->data()[2 * _i + 1];\n";
            out << "    v" << rid << "->data()[_i] = (int32_t)_idx_f;\n";
            out << "  }\n";
            continue;
        }

        if (op.kind == "RowSum" || op.kind == "RowMax" || op.kind == "RowMean") {
            if (!op.result.has_value() || op.operands.size() != 1) {
                out << "  // Row reduce op missing result/operands\n";
                continue;
            }
            const int rid = *op.result;
            auto it_dst = local_tiles.find(rid);
            auto it_src = local_tiles.find(op.operands[0]);
            if (it_dst == local_tiles.end() || it_src == local_tiles.end()) {
                out << "  // Row reduce expects tile operand\n";
                continue;
            }
            if (op.kind == "RowSum") {
                out << "  pto_wsp::ptoisa::TROWSUM(*v" << rid << ", " << tile_expr(op.operands[0]) << ", "
                    << tile_expr(op.operands[0]) << ");\n";
            } else if (op.kind == "RowMax") {
                out << "  pto_wsp::ptoisa::TROWMAX(*v" << rid << ", " << tile_expr(op.operands[0]) << ", "
                    << tile_expr(op.operands[0]) << ");\n";
            } else {
                const int cols = it_src->second.shape.second;
                const auto elem = dtype_to_cpp_pto_elem(it_dst->second.dtype);
                out << "  pto_wsp::ptoisa::TROWSUM(*v" << rid << ", " << tile_expr(op.operands[0]) << ", "
                    << tile_expr(op.operands[0]) << ");\n";
                out << "  pto_wsp::ptoisa::TDIVS(*v" << rid << ", *v" << rid << ", (" << elem << ")" << (double)cols
                    << ");\n";
            }
            continue;
        }

        // Unhandled op: keep kernels compilable and non-zero timing.
        if (op.result.has_value()) {
            auto itv = kp.values.find(*op.result);
            if (itv != kp.values.end() && itv->second.shape.has_value()) {
                const auto [r, c] = *itv->second.shape;
                out << "  cycles += (uint64_t)(" << r << "ULL * " << c << "ULL);\n";
            }
        }
        out << "  // Unhandled op kind: " << op.kind << "\n";
    }

    out << "\n#if defined(__CPU_SIM)\n";
    out << "  cycles += pto::cpu_sim::read_cycles();\n";
    out << "#endif\n";
    out << "  return cycles;\n";
    out.flush();
    tu.functions.push_back(std::move(fn).build());
    return cg::emit_cpp(tu);
}

// ---------------------------------------------------------------------------
// Workload emission from IR (v9 formal input)
// ---------------------------------------------------------------------------

const pto::wsp::ir::CodegenKernelDef* find_kernel_def(
    const pto::wsp::ir::Module& module, const std::string& name) {
    for (const auto& k : module.kernels) {
        if (k.name == name) return &k;
    }
    return nullptr;
}

std::string emit_workload_cpp_from_ir(const pto::wsp::ir::Module& module) {
    if (module.workloads.empty()) throw std::runtime_error("Module has no workloads");
    const auto& wl = module.workloads[0];
    if (!wl.body) throw std::runtime_error("Workload body is null");

    const std::string module_name = module.name.empty() ? "main" : module.name;
    const std::string entrypoint = module_name + "_main";

    std::ostringstream out;
    out << "#include \"pto/rt/codegen/abi/workload_abi.hpp\"\n";
    out << "#include \"pto/rt/codegen/abi/kernel_abi.hpp\"\n";
    out << "#include <cstdint>\n";
    out << "#include <cstddef>\n\n";

    // Forward declare kernels used by this module.
    for (const auto& k : module.kernels) {
        out << "extern \"C\" uint64_t " << k.name << "(const KernelTaskDesc* task, CSPTContext* cspt);\n";
    }
    if (!module.kernels.empty()) out << "\n";

    out << "extern \"C\" uint64_t " << entrypoint
        << "(RuntimeContext* ctx, CSPTContext* cspt) {\n";
    out << "  uint64_t cycles = 0;\n\n";

    // Cache tensor bases + strides.
    for (size_t tid = 0; tid < module.tensors.size(); ++tid) {
        const auto& tinfo = module.tensors[tid];
        const auto cpp_ty = dtype_to_cpp_workload(tinfo.dtype);
        out << "  " << cpp_ty << "* t" << tid << "_base = static_cast<" << cpp_ty
            << "*>(ctx->get_tensor_ptr(ctx->ctx, " << tid << "));\n";
        out << "  uint64_t t" << tid << "_stride[" << tinfo.shape.size() << "];\n";
        for (size_t d = 0; d < tinfo.shape.size(); ++d) {
            out << "  t" << tid << "_stride[" << d << "] = ctx->get_tensor_stride(ctx->ctx, "
                << tid << ", " << d << ");\n";
        }
    }
    out << "\n";

    std::function<void(const pto::wsp::ir::IRPtr<pto::wsp::ir::WorkloadNode>&)> emit_node;
    emit_node = [&](const auto& n) {
        using pto::wsp::ir::NodeKind;
        if (!n) return;

        switch (n->kind) {
            case NodeKind::ParallelFor: {
                const auto& pf = static_cast<const pto::wsp::ir::ParallelForNode&>(*n);
                const auto var = pf.index_var;
                if (pf.axis->kind == NodeKind::DenseDynAxis) {
                    out << "  int64_t " << var << "_ub = ctx->get_axis_size(ctx->ctx, \"" << var << "\");\n";
                    out << "  for (int64_t " << var << " = 0; " << var << " < " << var << "_ub; ++" << var << ") {\n";
                } else if (pf.axis->kind == NodeKind::DenseAxis) {
                    const auto& ax = static_cast<const pto::wsp::ir::DenseAxisNode&>(*pf.axis);
                    out << "  for (int64_t " << var << " = 0; " << var << " < " << ax.size << "; ++" << var << ") {\n";
                } else {
                    throw std::runtime_error("Unsupported axis kind for parallel_for in codegen");
                }
                out << "    cycles += 1;\n";
                emit_node(pf.body);
                out << "  }\n";
                return;
            }
            case NodeKind::ForEach: {
                const auto& fe = static_cast<const pto::wsp::ir::ForEachNode&>(*n);
                const auto var = fe.index_var;
                if (fe.axis->kind == NodeKind::DenseDynAxis) {
                    out << "  int64_t " << var << "_ub = ctx->get_axis_size(ctx->ctx, \"" << var << "\");\n";
                    out << "  for (int64_t " << var << " = 0; " << var << " < " << var << "_ub; ++" << var << ") {\n";
                } else if (fe.axis->kind == NodeKind::DenseAxis) {
                    const auto& ax = static_cast<const pto::wsp::ir::DenseAxisNode&>(*fe.axis);
                    out << "  for (int64_t " << var << " = 0; " << var << " < " << ax.size << "; ++" << var << ") {\n";
                } else {
                    throw std::runtime_error("Unsupported axis kind for for_each in codegen");
                }
                out << "    cycles += 1;\n";
                emit_node(fe.body);
                out << "  }\n";
                return;
            }
            case NodeKind::Combine: {
                const auto& c = static_cast<const pto::wsp::ir::CombineNode&>(*n);
                for (const auto& ch : c.workloads) emit_node(ch);
                return;
            }
            case NodeKind::Sequential: {
                const auto& s = static_cast<const pto::wsp::ir::SequentialNode&>(*n);
                for (const auto& ch : s.workloads) emit_node(ch);
                return;
            }
            case NodeKind::Task: {
                const auto& t = static_cast<const pto::wsp::ir::TaskNode&>(*n);
                const auto* kd = find_kernel_def(module, t.kernel_name);
                if (!kd) throw std::runtime_error("Missing kernel def for task: " + t.kernel_name);

                // Axis args
                out << "  {\n";
                out << "    uint64_t args[] = {";
                const size_t num_axis_args = t.axis_args.size();
                const size_t num_scalar_args = t.scalar_args.size();
                for (size_t i = 0; i < num_axis_args; ++i) {
                    if (i) out << ", ";
                    out << axis_arg_to_u64_literal(t.axis_args[i]);
                }
                for (size_t i = 0; i < num_scalar_args; ++i) {
                    if (i || num_axis_args) out << ", ";
                    out << axis_arg_to_u64_literal(t.scalar_args[i]);
                }
                out << "};\n";

                // Tensor bindings in kernel param order.
                std::vector<const pto::wsp::ir::CodegenTensorArg*> ordered;
                ordered.reserve(kd->params.size());
                for (const auto& kp : kd->params) {
                    const auto itv = kd->values.find(kp.id);
                    if (itv == kd->values.end()) throw std::runtime_error("Kernel value missing for param");
                    if (!itv->second.has_shape) continue;
                    const pto::wsp::ir::CodegenTensorArg* found = nullptr;
                    for (const auto& ta : t.tensor_args) {
                        if (ta.param == kp.name) {
                            found = &ta;
                            break;
                        }
                    }
                    if (!found) throw std::runtime_error("Missing tensor arg for param: " + kp.name);
                    ordered.push_back(found);
                }

                // Emit per-arg pointer/stride extraction.
                std::vector<std::string> tensor_ptrs;
                std::vector<std::string> tensor_strides;
                tensor_ptrs.reserve(ordered.size());
                tensor_strides.reserve(ordered.size() * 2);

                for (size_t i = 0; i < ordered.size(); ++i) {
                    const auto& a = *ordered[i];
                    const int tid = (int)a.tensor_id;
                    std::string tile_ptr_name = "arg" + std::to_string(i) + "_ptr";
                    out << "    void* " << tile_ptr_name << " = (void*)t" << tid << "_base;\n";

                    // Apply index_exprs to compute pointer.
                    // We treat the view as a slice with element-strides derived from base.
                    // pointer += idx[d] * stride[d] (in elements), then scale by sizeof(T).
                    out << "    {\n";
                    out << "      int64_t off_elems = 0;\n";
                    for (size_t d = 0; d < a.index_exprs.size(); ++d) {
                        out << "      off_elems += (" << index_expr_to_cpp(a.index_exprs[d])
                            << ") * (int64_t)t" << tid << "_stride[" << d << "];\n";
                    }
                    out << "      " << tile_ptr_name << " = (void*)((char*)t" << tid << "_base + off_elems * (int64_t)sizeof(*t"
                        << tid << "_base));\n";
                    out << "    }\n";

                    // 2D view strides (s3=row stride, s4=col stride).
                    int s3_stride_dim = -1;
                    int s4_stride_dim = -1;
                    if (a.view_rank >= 2) {
                        s3_stride_dim = a.base_rank - a.view_rank;
                        s4_stride_dim = s3_stride_dim + 1;
                    } else if (a.view_rank == 1) {
                        s3_stride_dim = -1;
                        s4_stride_dim = a.base_rank - 1;
                    }

                    const std::string s3_name = "arg" + std::to_string(i) + "_s3";
                    const std::string s4_name = "arg" + std::to_string(i) + "_s4";

                    if (s3_stride_dim < 0) {
                        out << "    uint64_t " << s3_name << " = 0;\n";
                    } else {
                        out << "    uint64_t " << s3_name << " = t" << tid << "_stride[" << s3_stride_dim << "];\n";
                    }
                    if (s4_stride_dim < 0) {
                        out << "    uint64_t " << s4_name << " = 1;\n";
                    } else {
                        out << "    uint64_t " << s4_name << " = t" << tid << "_stride[" << s4_stride_dim << "];\n";
                    }
                    tensor_ptrs.push_back(tile_ptr_name);
                    tensor_strides.push_back(s3_name);
                    tensor_strides.push_back(s4_name);
                }

                out << "    void* tensor_ptrs[] = {";
                for (size_t i = 0; i < tensor_ptrs.size(); ++i) {
                    if (i) out << ", ";
                    out << tensor_ptrs[i];
                }
                out << "};\n";

                out << "    uint64_t tensor_strides[] = {";
                for (size_t i = 0; i < tensor_strides.size(); ++i) {
                    if (i) out << ", ";
                    out << tensor_strides[i];
                }
                out << "};\n";

                out << "    KernelTaskDesc task_desc{args, " << (num_axis_args + num_scalar_args) << ", "
                    << num_axis_args << ", tensor_ptrs, tensor_strides, " << tensor_ptrs.size() << ", 0, 0};\n";
                out << "    cycles += 2;\n";
                out << "    cycles += (uint64_t)" << t.kernel_name << "(&task_desc, cspt);\n";
                out << "  }\n";
                return;
            }
            default:
                throw std::runtime_error("Unhandled workload node kind in IR codegen");
        }
    };

    emit_node(wl.body);

    out << "\n  return cycles;\n";
    out << "}\n";
    return out.str();
}

std::string emit_workload_cpp_from_ir_ast(const pto::wsp::ir::Module& module) {
    namespace cg = pto::wsp::codegen::cpp;
    using pto::wsp::ir::NodeKind;
    using pto::wsp::ir::DispatchPolicy;

    if (module.workloads.empty()) throw std::runtime_error("Module has no workloads");
    const auto& wl = module.workloads[0];
    if (!wl.body) throw std::runtime_error("Workload body is null");

    const std::string module_name = module.name.empty() ? "main" : module.name;
    const std::string entrypoint = module_name + "_main";

    cg::TranslationUnit tu;
    tu.includes = {
        "\"pto/rt/codegen/abi/workload_abi.hpp\"",
        "\"pto/rt/codegen/abi/kernel_abi.hpp\"",
        "\"pto/rt/codegen/abi/task_context_abi.hpp\"",
        "<algorithm>",
        "<cstdint>",
        "<cstddef>",
        "<cstdio>",
        "<condition_variable>",
        "<deque>",
        "<functional>",
        "<mutex>",
        "<queue>",
        "<thread>",
        "<vector>",
    };

    // ---------------------------------------------------------------------
    // v9 scheduling (CPU-sim): dispatch + task_window(stall-only).
    // Everything else is diagnostic-only in v9.
    // ---------------------------------------------------------------------
    const pto::wsp::ir::DispatchNode* dispatch = nullptr;
    const pto::wsp::ir::TaskWindowNode* task_window = nullptr;
    const pto::wsp::ir::StreamNode* streams = nullptr;
    std::vector<std::string> unsupported_schedule;

    if (const auto* s = module.findScheduleForWorkload(wl.name)) {
        for (const auto& d : s->directives) {
            if (!d) continue;
            switch (d->kind) {
                case NodeKind::Dispatch:
                    if (!dispatch) dispatch = static_cast<const pto::wsp::ir::DispatchNode*>(d.get());
                    break;
                case NodeKind::TaskWindow:
                    if (!task_window) task_window = static_cast<const pto::wsp::ir::TaskWindowNode*>(d.get());
                    break;
                case NodeKind::Stream:
                    if (!streams) streams = static_cast<const pto::wsp::ir::StreamNode*>(d.get());
                    break;
                default:
                    unsupported_schedule.push_back(pto::wsp::ir::nodeKindToString(d->kind));
                    break;
            }
        }
    }

    int num_workers = 1;
    if (dispatch && dispatch->num_targets > 0) {
        num_workers = dispatch->num_targets;
    } else if (streams && streams->num_streams > 0) {
        num_workers = streams->num_streams;
        if (streams->key) unsupported_schedule.push_back("stream_by");
    }
    if (num_workers <= 0) num_workers = 1;

    uint64_t task_window_tasks = 0;  // 0 = unlimited
    if (task_window) {
        const bool ok_unit = task_window->unit == "tasks";
        const bool ok_overflow = task_window->overflow == pto::wsp::ir::TaskWindowOverflowPolicy::Stall;
        if (ok_unit && ok_overflow) {
            task_window_tasks = task_window->size > 0 ? static_cast<uint64_t>(task_window->size) : 0ULL;
        } else {
            unsupported_schedule.push_back("task_window(non-stall or non-tasks)");
        }
    }

    if (dispatch) {
        const bool key_required =
            dispatch->policy == DispatchPolicy::Affinity || dispatch->policy == DispatchPolicy::Hash ||
            dispatch->policy == DispatchPolicy::Custom;
        if (key_required && !dispatch->key) {
            throw std::runtime_error(
                "dispatch policy requires ScalarExpr key (string-eval is not supported in v9 codegen-first)");
        }
        if (dispatch->policy == DispatchPolicy::WorkSteal) {
            unsupported_schedule.push_back("dispatch(work_steal)");
        }
    }

    tu.raw_toplevel.push_back(R"cpp(
namespace pto_wsp_codegen {
struct Scheduler {
    explicit Scheduler(uint32_t num_workers, uint64_t window_tasks)
        : num_workers_(num_workers ? num_workers : 1),
          window_tasks_(window_tasks),
          worker_available_(num_workers_, 0),
          worker_task_counts_(num_workers_, 0) {}

    void advance_to(uint64_t t) {
        if (t > issue_time_) issue_time_ = t;
        while (!in_flight_.empty() && in_flight_.top() <= issue_time_) {
            in_flight_.pop();
        }
    }

    uint64_t issue(uint64_t duration_cycles, uint32_t worker_id, uint64_t ready_time) {
        if (worker_id >= num_workers_) worker_id = 0;

        advance_to(ready_time);

        if (window_tasks_ > 0) {
            while (in_flight_.size() >= window_tasks_) {
                const uint64_t next_done = in_flight_.top();
                advance_to(next_done);
            }
        }

        const uint64_t start_time = worker_available_[worker_id] > issue_time_ ? worker_available_[worker_id] : issue_time_;
        const uint64_t end_time = start_time + duration_cycles;
        worker_available_[worker_id] = end_time;
        worker_task_counts_[worker_id] += 1;
        in_flight_.push(end_time);
        return end_time;
    }

    uint32_t num_workers() const { return num_workers_; }
    const std::vector<uint64_t>& worker_task_counts() const { return worker_task_counts_; }

private:
    uint32_t num_workers_;
    uint64_t window_tasks_;
    uint64_t issue_time_ = 0;
    std::vector<uint64_t> worker_available_;
    std::vector<uint64_t> worker_task_counts_;
    std::priority_queue<uint64_t, std::vector<uint64_t>, std::greater<uint64_t>> in_flight_;
};
}  // namespace pto_wsp_codegen
)cpp");

    tu.raw_toplevel.push_back(R"cpp(
namespace pto_wsp_cspt {
struct Token {
    uint64_t value;
    uint64_t ts;
};

class Channel {
public:
    explicit Channel(uint64_t capacity, uint64_t latency_cycles)
        : capacity_(capacity), latency_cycles_(latency_cycles) {}

    void close() {
        std::lock_guard<std::mutex> lock(mu_);
        closed_ = true;
        cv_not_empty_.notify_all();
        cv_not_full_.notify_all();
    }

    // Blocking send. Updates proc_time if the send was stalled by backpressure or rendezvous.
    void send(uint64_t& proc_time, uint64_t token) {
        std::unique_lock<std::mutex> lock(mu_);
        if (capacity_ == 0) {
            // Rendezvous: wait for receiver to be ready.
            while (!closed_ && !receiver_waiting_) cv_not_full_.wait(lock);
            if (closed_) return;

            rendezvous_ = Token{token, proc_time};
            has_rendezvous_ = true;
            cv_not_empty_.notify_one();

            // Wait for receiver to consume.
            while (!closed_ && has_rendezvous_) cv_not_full_.wait(lock);
            if (last_pop_time_ > proc_time) proc_time = last_pop_time_;
            return;
        }

        while (!closed_ && buf_.size() >= capacity_) {
            cv_not_full_.wait(lock);
            if (last_pop_time_ > proc_time) proc_time = last_pop_time_;
        }
        if (closed_) return;
        buf_.push_back(Token{token, proc_time});
        cv_not_empty_.notify_one();
    }

    // Blocking recv. Returns false if channel is closed and empty.
    bool recv(uint64_t& proc_time, uint64_t& out_token) {
        std::unique_lock<std::mutex> lock(mu_);
        if (capacity_ == 0) {
            receiver_waiting_ = true;
            cv_not_full_.notify_all();
            while (!closed_ && !has_rendezvous_) cv_not_empty_.wait(lock);
            receiver_waiting_ = false;
            if (!has_rendezvous_) return false;

            const Token t = rendezvous_;
            has_rendezvous_ = false;
            const uint64_t deliver = t.ts + latency_cycles_;
            if (deliver > proc_time) proc_time = deliver;
            out_token = t.value;
            last_pop_time_ = proc_time;
            cv_not_full_.notify_all();
            return true;
        }

        while (!closed_ && buf_.empty()) cv_not_empty_.wait(lock);
        if (buf_.empty()) return false;

        const Token t = buf_.front();
        buf_.pop_front();
        const uint64_t deliver = t.ts + latency_cycles_;
        if (deliver > proc_time) proc_time = deliver;
        out_token = t.value;
        last_pop_time_ = proc_time;
        cv_not_full_.notify_one();
        return true;
    }

private:
    uint64_t capacity_;
    uint64_t latency_cycles_;
    std::deque<Token> buf_;
    bool closed_ = false;
    uint64_t last_pop_time_ = 0;

    // Rendezvous state (capacity == 0)
    bool receiver_waiting_ = false;
    bool has_rendezvous_ = false;
    Token rendezvous_{0, 0};

    std::mutex mu_;
    std::condition_variable cv_not_empty_;
    std::condition_variable cv_not_full_;
};
}  // namespace pto_wsp_cspt
)cpp");

    for (const auto& k : module.kernels) {
        tu.raw_toplevel.push_back(
            "extern \"C\" uint64_t " + k.name + "(const KernelTaskDesc* task, CSPTContext* cspt);");
    }

    cg::FunctionBuilder fn("uint64_t", entrypoint);
    fn.extern_c()
        .param("RuntimeContext*", "ctx")
        .param("CSPTContext*", "cspt");

    auto emit = [&](int indent, std::string code) {
        fn.stmt(cg::raw_stmt(std::string(static_cast<size_t>(indent) * 2, ' ') + std::move(code)));
    };

    emit(0, "pto_wsp_codegen::Scheduler sched(" + std::to_string(num_workers) + "u, " + std::to_string(task_window_tasks) + "ULL);");
    emit(0, "uint64_t rr = 0;");
    if (!unsupported_schedule.empty()) {
        emit(0, "static bool _pto_wsp_schedule_diag_once = false;");
        emit(0, "if (!_pto_wsp_schedule_diag_once) {");
        emit(1, "_pto_wsp_schedule_diag_once = true;");
        emit(1, "std::fprintf(stderr, \"[pto-wsp] v9: unsupported schedule directives ignored:\\n\");");
        for (const auto& u : unsupported_schedule) {
            emit(1, "std::fprintf(stderr, \"  - %s\\n\", \"" + u + "\");");
        }
        emit(0, "}");
    }
    emit(0, "");

    // Cache tensor bases + strides.
    for (size_t tid = 0; tid < module.tensors.size(); ++tid) {
        const auto& tinfo = module.tensors[tid];
        const auto cpp_ty = dtype_to_cpp_workload(tinfo.dtype);
        emit(
            0,
            cpp_ty + "* t" + std::to_string(tid) + "_base = static_cast<" + cpp_ty +
                "*>(ctx->get_tensor_ptr(ctx->ctx, " + std::to_string(tid) + "));");
        emit(0, "uint64_t t" + std::to_string(tid) + "_stride[" + std::to_string(tinfo.shape.size()) + "];");
        for (size_t d = 0; d < tinfo.shape.size(); ++d) {
            emit(
                0,
                "t" + std::to_string(tid) + "_stride[" + std::to_string(d) +
                    "] = ctx->get_tensor_stride(ctx->ctx, " + std::to_string(tid) + ", " + std::to_string(d) + ");");
        }
    }
    emit(0, "");

    // ---------------------------------------------------------------------
    // CSP/CSPT (v9): if the root is a pipeline, emit a codegen-first CSPT
    // runtime specialized for synchronization channels with strict PTO-ISA
    // cycle time.
    // ---------------------------------------------------------------------
    if (wl.body->kind == NodeKind::Pipeline) {
        const auto& pipe = static_cast<const pto::wsp::ir::PipelineNode&>(*wl.body);

        auto sanitize_ident = [](const std::string& s) {
            std::string out;
            out.reserve(s.size() + 8);
            for (char c : s) {
                if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_') {
                    out.push_back(c);
                } else {
                    out.push_back('_');
                }
            }
            if (out.empty() || (out[0] >= '0' && out[0] <= '9')) out = "v_" + out;
            return out;
        };

        // v9 CSPT: constant channel latency (default 0) driven by a stable runtime symbol.
        // This enables latency experiments without recompiling the artifact.
        const uint64_t latency_sym = fnv1a_64("__pto_wsp_channel_latency_cycles");
        emit(0, "const uint64_t pto_wsp_channel_latency_cycles = ctx->get_symbol_u64(ctx->ctx, " + u64_hex_literal(latency_sym) + ");");
        emit(0, "std::mutex pto_wsp_kernel_mu;");
        std::unordered_map<std::string, std::string> ch_var;
        for (const auto& ch : pipe.channels) {
            const std::string v = "ch_" + sanitize_ident(ch->name);
            ch_var[ch->name] = v;
            emit(
                0,
                "pto_wsp_cspt::Channel " + v + "(" + std::to_string((uint64_t)std::max<int64_t>(0, ch->type.capacity)) +
                    "ULL, pto_wsp_channel_latency_cycles);");
        }
        emit(0, "");

        // Process threads + local time accounting.
        std::unordered_map<std::string, std::string> proc_time_out;
        for (const auto& p : pipe.processes) {
            const std::string pv = "p_" + sanitize_ident(p->name);
            const std::string tv = "t_" + pv;
            proc_time_out[p->name] = tv;
            emit(0, "uint64_t " + tv + " = 0;");
        }
        emit(0, "");

        // Codegen helpers for process bodies (sequential, time-accounting).
        int cspt_indent = 0;
        std::unordered_map<uint64_t, std::string> axis_env;
        const auto axis_lookup = [&](uint64_t id) -> std::optional<std::string> {
            const auto it = axis_env.find(id);
            if (it == axis_env.end()) return std::nullopt;
            return it->second;
        };
        const auto sym_u64_expr = [&](uint64_t id) -> std::string {
            return "ctx->get_symbol_u64(ctx->ctx, " + u64_hex_literal(id) + ")";
        };
        const auto slot_u64_expr = [&](uint32_t i, pto::wsp::ir::ScalarType ty) -> std::string {
            const std::string base =
                "(ctx->get_slot_u64 ? ctx->get_slot_u64(ctx->ctx, " + std::to_string(i) + "u) : 0ULL)";
            if (ty == pto::wsp::ir::ScalarType::I64) return "(int64_t)" + base;
            if (ty == pto::wsp::ir::ScalarType::U64) return "(uint64_t)" + base;
            return "((bool)(" + base + " != 0ULL))";
        };

        auto emit_cspt_line = [&](int indent, const std::string& s) {
            fn.stmt(cg::raw_stmt(std::string(static_cast<size_t>(indent) * 2, ' ') + s));
        };

        std::function<void(const pto::wsp::ir::IRPtr<pto::wsp::ir::WorkloadNode>&, const std::string&)> emit_cspt_node;
        emit_cspt_node = [&](const auto& n, const std::string& tvar) {
            if (!n) return;
            switch (n->kind) {
                case NodeKind::SlotSetU64: {
                    const auto& s = static_cast<const pto::wsp::ir::SlotSetU64Node&>(*n);
                    emit_cspt_line(cspt_indent, "if (ctx->set_slot_u64) ctx->set_slot_u64(ctx->ctx, " +
                                                   std::to_string(s.slot) + "u, " +
                                                   u64_hex_literal(s.value) + ");");
                    return;
                }
                case NodeKind::SlotLoadU64: {
                    const auto& l = static_cast<const pto::wsp::ir::SlotLoadU64Node&>(*n);
                    const auto tid = (int)l.tensor.tensor_id;
                    const auto item = dtype_to_cpp_workload(dtype_to_str(module.tensors[tid].dtype));
                    const auto idx_base = std::string("_slot_load_") + std::to_string(reinterpret_cast<uintptr_t>(&l));
                    emit_cspt_line(cspt_indent, "{");
                    cspt_indent++;
                    emit_cspt_line(cspt_indent, "int64_t off_elems = 0;");
                    for (size_t d = 0; d < l.tensor.index_exprs.size(); ++d) {
                        emit_cspt_line(
                            cspt_indent,
                            "off_elems += (" + index_expr_to_cpp(l.tensor.index_exprs[d]) + ") * (int64_t)t" +
                                std::to_string(tid) + "_stride[" + std::to_string(d) + "];");
                    }
                    emit_cspt_line(
                        cspt_indent,
                        "uint8_t* " + idx_base + "_ptr = (uint8_t*)t" + std::to_string(tid) +
                            "_base + off_elems * (int64_t)sizeof(*t" + std::to_string(tid) + "_base);");

                    int s3_stride_dim = -1;
                    int s4_stride_dim = -1;
                    if (l.tensor.view_rank >= 2) {
                        s3_stride_dim = l.tensor.base_rank - l.tensor.view_rank;
                        s4_stride_dim = s3_stride_dim + 1;
                    } else if (l.tensor.view_rank == 1) {
                        s3_stride_dim = -1;
                        s4_stride_dim = l.tensor.base_rank - 1;
                    }

                    if (s3_stride_dim < 0) {
                        emit_cspt_line(cspt_indent, "uint64_t " + idx_base + "_s3 = 0;");
                    } else {
                        emit_cspt_line(
                            cspt_indent,
                            "uint64_t " + idx_base + "_s3 = t" + std::to_string(tid) + "_stride[" +
                                std::to_string(s3_stride_dim) + "];");
                    }
                    if (s4_stride_dim < 0) {
                        emit_cspt_line(cspt_indent, "uint64_t " + idx_base + "_s4 = 1;");
                    } else {
                        emit_cspt_line(
                            cspt_indent,
                            "uint64_t " + idx_base + "_s4 = t" + std::to_string(tid) + "_stride[" +
                                std::to_string(s4_stride_dim) + "];");
                    }

                    emit_cspt_line(
                        cspt_indent,
                        "uint64_t _slot_v = (uint64_t)((" + item + "*)" + idx_base +
                            "_ptr)[(int64_t)" + std::to_string(l.row) + " * (int64_t)" + idx_base +
                            "_s3 + (int64_t)" + std::to_string(l.col) + " * (int64_t)" + idx_base + "_s4];");
                    emit_cspt_line(
                        cspt_indent,
                        "if (ctx->set_slot_u64) ctx->set_slot_u64(ctx->ctx, " + std::to_string(l.slot) + "u, _slot_v);");
                    cspt_indent--;
                    emit_cspt_line(cspt_indent, "}");
                    return;
                }
                case NodeKind::ParallelFor: {
                    const auto& pf = static_cast<const pto::wsp::ir::ParallelForNode&>(*n);
                    const auto var = pf.index_var;
                    const uint64_t var_id = fnv1a_64(var);
                    if (pf.axis->kind == NodeKind::DenseDynAxis) {
                        emit_cspt_line(cspt_indent, "int64_t " + var + "_ub = ctx->get_axis_size(ctx->ctx, \"" + var + "\");");
                        emit_cspt_line(cspt_indent, "for (int64_t " + var + " = 0; " + var + " < " + var + "_ub; ++" + var + ") {");
                    } else if (pf.axis->kind == NodeKind::DenseAxis) {
                        const auto& ax = static_cast<const pto::wsp::ir::DenseAxisNode&>(*pf.axis);
                        emit_cspt_line(cspt_indent, "for (int64_t " + var + " = 0; " + var + " < " + std::to_string(ax.size) + "; ++" + var + ") {");
                    } else {
                        throw std::runtime_error("CSPT codegen: unsupported axis kind in loop");
                    }
                    cspt_indent++;
                    const auto prev = axis_env.find(var_id);
                    std::optional<std::string> prev_val;
                    if (prev != axis_env.end()) prev_val = prev->second;
                    axis_env[var_id] = var;
                    emit_cspt_node(pf.body, tvar);
                    if (prev_val) axis_env[var_id] = *prev_val;
                    else axis_env.erase(var_id);
                    cspt_indent--;
                    emit_cspt_line(cspt_indent, "}");
                    return;
                }
                case NodeKind::ForEach: {
                    const auto& fe = static_cast<const pto::wsp::ir::ForEachNode&>(*n);
                    const auto var = fe.index_var;
                    const uint64_t var_id = fnv1a_64(var);
                    if (fe.axis->kind == NodeKind::DenseDynAxis) {
                        emit_cspt_line(cspt_indent, "int64_t " + var + "_ub = ctx->get_axis_size(ctx->ctx, \"" + var + "\");");
                        emit_cspt_line(cspt_indent, "for (int64_t " + var + " = 0; " + var + " < " + var + "_ub; ++" + var + ") {");
                    } else if (fe.axis->kind == NodeKind::DenseAxis) {
                        const auto& ax = static_cast<const pto::wsp::ir::DenseAxisNode&>(*fe.axis);
                        emit_cspt_line(cspt_indent, "for (int64_t " + var + " = 0; " + var + " < " + std::to_string(ax.size) + "; ++" + var + ") {");
                    } else {
                        throw std::runtime_error("CSPT codegen: unsupported axis kind in loop");
                    }
                    cspt_indent++;
                    const auto prev = axis_env.find(var_id);
                    std::optional<std::string> prev_val;
                    if (prev != axis_env.end()) prev_val = prev->second;
                    axis_env[var_id] = var;
                    emit_cspt_node(fe.body, tvar);
                    if (prev_val) axis_env[var_id] = *prev_val;
                    else axis_env.erase(var_id);
                    cspt_indent--;
                    emit_cspt_line(cspt_indent, "}");
                    return;
                }
                case NodeKind::Cond: {
                    const auto& c = static_cast<const pto::wsp::ir::CondNode&>(*n);
                    if (!c.predicate) throw std::runtime_error("CSPT codegen: cond missing predicate ScalarExpr");
                    const std::string pred = emit_scalar_expr_cpp(c.predicate, axis_lookup, sym_u64_expr, slot_u64_expr);
                    emit_cspt_line(cspt_indent, "if (" + pred + ") {");
                    cspt_indent++;
                    emit_cspt_node(c.then_branch, tvar);
                    cspt_indent--;
                    emit_cspt_line(cspt_indent, "} else {");
                    cspt_indent++;
                    emit_cspt_node(c.else_branch, tvar);
                    cspt_indent--;
                    emit_cspt_line(cspt_indent, "}");
                    return;
                }
                case NodeKind::Combine: {
                    const auto& c = static_cast<const pto::wsp::ir::CombineNode&>(*n);
                    for (const auto& ch : c.workloads) emit_cspt_node(ch, tvar);
                    return;
                }
                case NodeKind::Sequential: {
                    const auto& s = static_cast<const pto::wsp::ir::SequentialNode&>(*n);
                    for (const auto& ch : s.workloads) emit_cspt_node(ch, tvar);
                    return;
                }
                case NodeKind::Task: {
                    const auto& t = static_cast<const pto::wsp::ir::TaskNode&>(*n);
                    const auto* kd = find_kernel_def(module, t.kernel_name);
                    if (!kd) throw std::runtime_error("Missing kernel def for task: " + t.kernel_name);

                    const size_t num_user_axis_args = t.axis_args.size();
                    const size_t num_scalar_args = t.scalar_args.size();

                    std::vector<std::pair<int, int>> tensor_param_rc;
                    tensor_param_rc.reserve(kd->params.size());
                    for (const auto& kp : kd->params) {
                        const auto itv = kd->values.find(kp.id);
                        if (itv == kd->values.end()) throw std::runtime_error("Kernel value missing for param");
                        if (!itv->second.has_shape) continue;
                        tensor_param_rc.emplace_back(itv->second.rows, itv->second.cols);
                    }
                    const size_t num_tail_axis_args = tensor_param_rc.size() * 2;
                    const size_t num_axis_args = num_user_axis_args + num_tail_axis_args;

                    emit_cspt_line(cspt_indent, "{");
                    cspt_indent++;

                    // Axis args
                    {
                        std::ostringstream args;
                        args << "uint64_t args[] = {";
                        for (size_t i = 0; i < num_user_axis_args; ++i) {
                            if (i) args << ", ";
                            args << axis_arg_to_u64_literal(t.axis_args[i]);
                        }
                        for (size_t i = 0; i < tensor_param_rc.size(); ++i) {
                            if (num_user_axis_args || i) args << ", ";
                            args << "(uint64_t)" << tensor_param_rc[i].first;
                            args << ", (uint64_t)" << tensor_param_rc[i].second;
                        }
                        for (size_t i = 0; i < num_scalar_args; ++i) {
                            if (i || num_axis_args) args << ", ";
                            args << axis_arg_to_u64_literal(t.scalar_args[i]);
                        }
                        args << "};";
                        emit_cspt_line(cspt_indent, args.str());
                    }

                    // Tensor bindings in kernel param order.
                    std::vector<const pto::wsp::ir::CodegenTensorArg*> ordered;
                    ordered.reserve(kd->params.size());
                    for (const auto& kp : kd->params) {
                        const auto itv = kd->values.find(kp.id);
                        if (itv == kd->values.end()) throw std::runtime_error("Kernel value missing for param");
                        if (!itv->second.has_shape) continue;
                        const pto::wsp::ir::CodegenTensorArg* found = nullptr;
                        for (const auto& ta : t.tensor_args) {
                            if (ta.param == kp.name) {
                                found = &ta;
                                break;
                            }
                        }
                        if (!found) throw std::runtime_error("Missing tensor arg for param: " + kp.name);
                        ordered.push_back(found);
                    }

                    std::vector<std::string> tensor_ptrs;
                    std::vector<std::string> tensor_strides;
                    tensor_ptrs.reserve(ordered.size());
                    tensor_strides.reserve(ordered.size() * 2);

                    for (size_t i = 0; i < ordered.size(); ++i) {
                        const auto& a = *ordered[i];
                        const int tid = (int)a.tensor_id;
                        const std::string tile_ptr_name = "arg" + std::to_string(i) + "_ptr";
                        emit_cspt_line(cspt_indent, "void* " + tile_ptr_name + " = (void*)t" + std::to_string(tid) + "_base;");

                        emit_cspt_line(cspt_indent, "{");
                        cspt_indent++;
                        emit_cspt_line(cspt_indent, "int64_t off_elems = 0;");
                        for (size_t d = 0; d < a.index_exprs.size(); ++d) {
                            emit_cspt_line(
                                cspt_indent,
                                "off_elems += (" + index_expr_to_cpp(a.index_exprs[d]) + ") * (int64_t)t" +
                                    std::to_string(tid) + "_stride[" + std::to_string(d) + "];");
                        }
                        emit_cspt_line(
                            cspt_indent,
                            tile_ptr_name + " = (void*)((char*)t" + std::to_string(tid) +
                                "_base + off_elems * (int64_t)sizeof(*t" + std::to_string(tid) + "_base));");
                        cspt_indent--;
                        emit_cspt_line(cspt_indent, "}");

                        int s3_stride_dim = -1;
                        int s4_stride_dim = -1;
                        if (a.view_rank >= 2) {
                            s3_stride_dim = a.base_rank - a.view_rank;
                            s4_stride_dim = s3_stride_dim + 1;
                        } else if (a.view_rank == 1) {
                            s3_stride_dim = -1;
                            s4_stride_dim = a.base_rank - 1;
                        }

                        const std::string s3_name = "arg" + std::to_string(i) + "_s3";
                        const std::string s4_name = "arg" + std::to_string(i) + "_s4";

                        if (s3_stride_dim < 0) {
                            emit_cspt_line(cspt_indent, "uint64_t " + s3_name + " = 0;");
                        } else {
                            emit_cspt_line(cspt_indent, "uint64_t " + s3_name + " = t" + std::to_string(tid) + "_stride[" +
                                                          std::to_string(s3_stride_dim) + "];");
                        }
                        if (s4_stride_dim < 0) {
                            emit_cspt_line(cspt_indent, "uint64_t " + s4_name + " = 1;");
                        } else {
                            emit_cspt_line(cspt_indent, "uint64_t " + s4_name + " = t" + std::to_string(tid) + "_stride[" +
                                                          std::to_string(s4_stride_dim) + "];");
                        }

                        tensor_ptrs.push_back(tile_ptr_name);
                        tensor_strides.push_back(s3_name);
                        tensor_strides.push_back(s4_name);
                    }

                    {
                        std::ostringstream ptrs;
                        ptrs << "void* tensor_ptrs[] = {";
                        for (size_t i = 0; i < tensor_ptrs.size(); ++i) {
                            if (i) ptrs << ", ";
                            ptrs << tensor_ptrs[i];
                        }
                        ptrs << "};";
                        emit_cspt_line(cspt_indent, ptrs.str());
                    }
                    {
                        std::ostringstream strides;
                        strides << "uint64_t tensor_strides[] = {";
                        for (size_t i = 0; i < tensor_strides.size(); ++i) {
                            if (i) strides << ", ";
                            strides << tensor_strides[i];
                        }
                        strides << "};";
                        emit_cspt_line(cspt_indent, strides.str());
                    }

                    emit_cspt_line(
                        cspt_indent,
                        "KernelTaskDesc task_desc{args, " + std::to_string(num_axis_args + num_scalar_args) + ", " +
                            std::to_string(num_axis_args) + ", tensor_ptrs, tensor_strides, " +
                            std::to_string(tensor_ptrs.size()) + ", 0, 0};");
                    emit_cspt_line(cspt_indent, "uint64_t _pto_wsp_kcycles = 0;");
                    emit_cspt_line(cspt_indent, "{ std::lock_guard<std::mutex> _lk(pto_wsp_kernel_mu); _pto_wsp_kcycles = (uint64_t)" +
                                          t.kernel_name + "(&task_desc, cspt); }");
                    emit_cspt_line(cspt_indent, tvar + " += _pto_wsp_kcycles;");

                    cspt_indent--;
                    emit_cspt_line(cspt_indent, "}");
                    return;
                }
                case NodeKind::Send: {
                    const auto& s = static_cast<const pto::wsp::ir::SendNode&>(*n);
                    if (!s.value || s.value->kind != NodeKind::Task) {
                        throw std::runtime_error("CSPT codegen: send value must be a Task node");
                    }
                    const auto& task = static_cast<const pto::wsp::ir::TaskNode&>(*s.value);
                    // Execute the task (side-effecting) and send token = first axis arg (if any).
                    emit_cspt_line(cspt_indent, "{");
                    cspt_indent++;
                    // Recurse into the task node (reuses Task case above).
                    emit_cspt_node(std::static_pointer_cast<const pto::wsp::ir::WorkloadNode>(s.value), tvar);
                    emit_cspt_line(cspt_indent, "uint64_t _pto_wsp_token = 0;");
                    if (!task.axis_args.empty()) {
                        emit_cspt_line(cspt_indent, "_pto_wsp_token = (uint64_t)" + axis_arg_to_u64_literal(task.axis_args[0]) + ";");
                    }
                    auto it = ch_var.find(s.channel_name);
                    if (it == ch_var.end()) throw std::runtime_error("CSPT codegen: unknown channel: " + s.channel_name);
                    emit_cspt_line(cspt_indent, it->second + ".send(" + tvar + ", _pto_wsp_token);");
                    cspt_indent--;
                    emit_cspt_line(cspt_indent, "}");
                    return;
                }
                case NodeKind::Consume: {
                    const auto& c = static_cast<const pto::wsp::ir::ConsumeNode&>(*n);
                    auto it = ch_var.find(c.channel_name);
                    if (it == ch_var.end()) throw std::runtime_error("CSPT codegen: unknown channel: " + c.channel_name);
                    const uint64_t var_id = fnv1a_64(c.value_var);
                    emit_cspt_line(cspt_indent, "{");
                    cspt_indent++;
                    emit_cspt_line(cspt_indent, "uint64_t _pto_wsp_tok = 0;");
                    emit_cspt_line(cspt_indent, "while (" + it->second + ".recv(" + tvar + ", _pto_wsp_tok)) {");
                    cspt_indent++;
                    emit_cspt_line(cspt_indent, "int64_t " + c.value_var + " = (int64_t)_pto_wsp_tok;");
                    const auto prev = axis_env.find(var_id);
                    std::optional<std::string> prev_val;
                    if (prev != axis_env.end()) prev_val = prev->second;
                    axis_env[var_id] = c.value_var;
                    emit_cspt_node(c.body, tvar);
                    if (prev_val) axis_env[var_id] = *prev_val;
                    else axis_env.erase(var_id);
                    cspt_indent--;
                    emit_cspt_line(cspt_indent, "}");
                    cspt_indent--;
                    emit_cspt_line(cspt_indent, "}");
                    return;
                }
                default:
                    throw std::runtime_error("CSPT codegen: unsupported node kind in process body");
            }
        };

        // Launch processes.
        for (const auto& p : pipe.processes) {
            const std::string pv = "p_" + sanitize_ident(p->name);
            const std::string tv = proc_time_out[p->name];
            emit(0, "std::thread th_" + pv + "([&]() {");
            emit(1, "uint64_t pto_wsp_time = 0;");
            // Emit process body inside thread.
            // Use the function builder directly for consistent indentation.
            // (We generate body statements at indent=2 because we are inside the lambda.)
            {
                // Temporarily switch cspt_indent to match thread indentation.
                const int saved_indent = cspt_indent;
                cspt_indent = 2;
                axis_env.clear();
                emit_cspt_node(p->body, "pto_wsp_time");
                cspt_indent = saved_indent;
                axis_env.clear();
            }
            // Close produced channels.
            for (const auto& ch_name : p->produces) {
                auto it = ch_var.find(ch_name);
                if (it != ch_var.end()) {
                    emit(1, it->second + ".close();");
                }
            }
            emit(1, tv + " = pto_wsp_time;");
            emit(0, "});");
        }
        emit(0, "");

        // Join all process threads.
        for (const auto& p : pipe.processes) {
            const std::string pv = "p_" + sanitize_ident(p->name);
            emit(0, "th_" + pv + ".join();");
        }

        // Makespan across process-local time (CSPT).
        emit(0, "uint64_t cspt_cycles = 0;");
        for (const auto& p : pipe.processes) {
            emit(0, "cspt_cycles = std::max(cspt_cycles, " + proc_time_out[p->name] + ");");
        }
        fn.stmt(cg::raw_stmt(""));
        fn.stmt(cg::ret(cg::ident("cspt_cycles")));

        tu.functions.push_back(std::move(fn).build());
        return cg::emit_cpp(tu);
    }

    int indent = 0;
    std::unordered_map<uint64_t, std::string> axis_env;
    const auto axis_lookup = [&](uint64_t id) -> std::optional<std::string> {
        const auto it = axis_env.find(id);
        if (it == axis_env.end()) return std::nullopt;
        return it->second;
    };
    const auto sym_u64_expr = [&](uint64_t id) -> std::string {
        return "ctx->get_symbol_u64(ctx->ctx, " + u64_hex_literal(id) + ")";
    };
    const auto slot_u64_expr = [&](uint32_t i, pto::wsp::ir::ScalarType ty) -> std::string {
        const std::string base =
            "(ctx->get_slot_u64 ? ctx->get_slot_u64(ctx->ctx, " + std::to_string(i) + "u) : 0ULL)";
        if (ty == pto::wsp::ir::ScalarType::I64) return "(int64_t)" + base;
        if (ty == pto::wsp::ir::ScalarType::U64) return "(uint64_t)" + base;
        return "((bool)(" + base + " != 0ULL))";
    };

    int tmp_id = 0;
    const auto new_tmp = [&]() { return std::string("_t") + std::to_string(tmp_id++); };

    std::function<std::string(const pto::wsp::ir::IRPtr<pto::wsp::ir::WorkloadNode>&, const std::string&)> emit_node;
    emit_node = [&](const auto& n, const std::string& ready_time) -> std::string {
        if (!n) return ready_time;

        switch (n->kind) {
            case NodeKind::SlotSetU64: {
                const auto& s = static_cast<const pto::wsp::ir::SlotSetU64Node&>(*n);
                emit(indent, "if (ctx->set_slot_u64) ctx->set_slot_u64(ctx->ctx, " + std::to_string(s.slot) +
                                 "u, " + u64_hex_literal(s.value) + ");");
                return ready_time;
            }
            case NodeKind::SlotLoadU64: {
                const auto& l = static_cast<const pto::wsp::ir::SlotLoadU64Node&>(*n);
                const int tid = (int)l.tensor.tensor_id;
                const std::string item = dtype_to_cpp_workload(dtype_to_str(module.tensors[tid].dtype));
                const std::string base = "_slot_load_" + std::to_string(reinterpret_cast<uintptr_t>(&l));
                emit(indent, "{");
                indent++;
                emit(indent, "int64_t off_elems = 0;");
                for (size_t d = 0; d < l.tensor.index_exprs.size(); ++d) {
                    emit(indent,
                         "off_elems += (" + index_expr_to_cpp(l.tensor.index_exprs[d]) + ") * (int64_t)t" +
                             std::to_string(tid) + "_stride[" + std::to_string(d) + "];");
                }
                emit(indent,
                     "uint8_t* " + base + "_ptr = (uint8_t*)t" + std::to_string(tid) +
                         "_base + off_elems * (int64_t)sizeof(*t" + std::to_string(tid) + "_base);");

                int s3_stride_dim = -1;
                int s4_stride_dim = -1;
                if (l.tensor.view_rank >= 2) {
                    s3_stride_dim = l.tensor.base_rank - l.tensor.view_rank;
                    s4_stride_dim = s3_stride_dim + 1;
                } else if (l.tensor.view_rank == 1) {
                    s3_stride_dim = -1;
                    s4_stride_dim = l.tensor.base_rank - 1;
                }
                if (s3_stride_dim < 0) {
                    emit(indent, "uint64_t " + base + "_s3 = 0;");
                } else {
                    emit(indent,
                         "uint64_t " + base + "_s3 = t" + std::to_string(tid) + "_stride[" +
                             std::to_string(s3_stride_dim) + "];");
                }
                if (s4_stride_dim < 0) {
                    emit(indent, "uint64_t " + base + "_s4 = 1;");
                } else {
                    emit(indent,
                         "uint64_t " + base + "_s4 = t" + std::to_string(tid) + "_stride[" +
                             std::to_string(s4_stride_dim) + "];");
                }

                emit(indent,
                     "uint64_t _slot_v = (uint64_t)((" + item + "*)" + base + "_ptr)[(int64_t)" +
                         std::to_string(l.row) + " * (int64_t)" + base + "_s3 + (int64_t)" +
                         std::to_string(l.col) + " * (int64_t)" + base + "_s4];");
                emit(indent, "if (ctx->set_slot_u64) ctx->set_slot_u64(ctx->ctx, " + std::to_string(l.slot) +
                                 "u, _slot_v);");
                indent--;
                emit(indent, "}");
                return ready_time;
            }
            case NodeKind::ParallelFor: {
                const auto& pf = static_cast<const pto::wsp::ir::ParallelForNode&>(*n);
                const auto var = pf.index_var;
                const uint64_t var_id = fnv1a_64(var);
                const std::string done = new_tmp();
                emit(indent, "uint64_t " + done + " = " + ready_time + ";");

                if (pf.axis->kind == NodeKind::DenseDynAxis) {
                    emit(indent, "int64_t " + var + "_ub = ctx->get_axis_size(ctx->ctx, \"" + var + "\");");
                    emit(indent, "for (int64_t " + var + " = 0; " + var + " < " + var + "_ub; ++" + var + ") {");
                } else if (pf.axis->kind == NodeKind::DenseAxis) {
                    const auto& ax = static_cast<const pto::wsp::ir::DenseAxisNode&>(*pf.axis);
                    emit(
                        indent,
                        "for (int64_t " + var + " = 0; " + var + " < " + std::to_string(ax.size) + "; ++" + var +
                            ") {");
                } else if (pf.axis->kind == NodeKind::RaggedAxis) {
                    const auto& ax = static_cast<const pto::wsp::ir::RaggedAxisNode&>(*pf.axis);
                    const uint64_t outer_sym = fnv1a_64(ax.outer_size_var);
                    const uint64_t lengths_sym = fnv1a_64(ax.lengths_var);

                    emit(
                        indent,
                        "int64_t " + var + "_outer = (int64_t)ctx->get_symbol_u64(ctx->ctx, " + u64_hex_literal(outer_sym) + ");");
                    emit(
                        indent,
                        "const int32_t* " + var + "_lengths = (const int32_t*)ctx->get_symbol_ptr(ctx->ctx, " +
                            u64_hex_literal(lengths_sym) + ");");
                    emit(indent, "int64_t " + var + "_ub = 0;");
                    emit(indent, "if (" + var + "_lengths) {");
                    indent++;
                    emit(indent, "for (int64_t _i = 0; _i < " + var + "_outer; ++_i) { " + var + "_ub += (int64_t)" +
                                     var + "_lengths[_i]; }");
                    indent--;
                    emit(indent, "}");
                    emit(indent, "for (int64_t " + var + " = 0; " + var + " < " + var + "_ub; ++" + var + ") {");
                } else {
                    throw std::runtime_error("Unsupported axis kind for parallel_for in codegen");
                }

                indent++;
                const auto prev = axis_env.find(var_id);
                std::optional<std::string> prev_val;
                if (prev != axis_env.end()) prev_val = prev->second;
                axis_env[var_id] = var;
                const std::string iter_done = emit_node(pf.body, ready_time);
                emit(indent, done + " = std::max(" + done + ", " + iter_done + ");");
                if (prev_val) {
                    axis_env[var_id] = *prev_val;
                } else {
                    axis_env.erase(var_id);
                }
                indent--;
                emit(indent, "}");
                return done;
            }
            case NodeKind::ForEach: {
                const auto& fe = static_cast<const pto::wsp::ir::ForEachNode&>(*n);
                const auto var = fe.index_var;
                const uint64_t var_id = fnv1a_64(var);
                const std::string cur = new_tmp();
                emit(indent, "uint64_t " + cur + " = " + ready_time + ";");

                if (fe.axis->kind == NodeKind::DenseDynAxis) {
                    emit(indent, "int64_t " + var + "_ub = ctx->get_axis_size(ctx->ctx, \"" + var + "\");");
                    emit(indent, "for (int64_t " + var + " = 0; " + var + " < " + var + "_ub; ++" + var + ") {");
                } else if (fe.axis->kind == NodeKind::DenseAxis) {
                    const auto& ax = static_cast<const pto::wsp::ir::DenseAxisNode&>(*fe.axis);
                    emit(
                        indent,
                        "for (int64_t " + var + " = 0; " + var + " < " + std::to_string(ax.size) + "; ++" + var +
                            ") {");
                } else if (fe.axis->kind == NodeKind::RaggedAxis) {
                    const auto& ax = static_cast<const pto::wsp::ir::RaggedAxisNode&>(*fe.axis);
                    const uint64_t outer_sym = fnv1a_64(ax.outer_size_var);
                    const uint64_t lengths_sym = fnv1a_64(ax.lengths_var);

                    emit(
                        indent,
                        "int64_t " + var + "_outer = (int64_t)ctx->get_symbol_u64(ctx->ctx, " + u64_hex_literal(outer_sym) + ");");
                    emit(
                        indent,
                        "const int32_t* " + var + "_lengths = (const int32_t*)ctx->get_symbol_ptr(ctx->ctx, " +
                            u64_hex_literal(lengths_sym) + ");");
                    emit(indent, "int64_t " + var + "_ub = 0;");
                    emit(indent, "if (" + var + "_lengths) {");
                    indent++;
                    emit(indent, "for (int64_t _i = 0; _i < " + var + "_outer; ++_i) { " + var + "_ub += (int64_t)" +
                                     var + "_lengths[_i]; }");
                    indent--;
                    emit(indent, "}");
                    emit(indent, "for (int64_t " + var + " = 0; " + var + " < " + var + "_ub; ++" + var + ") {");
                } else {
                    throw std::runtime_error("Unsupported axis kind for for_each in codegen");
                }

                indent++;
                const auto prev = axis_env.find(var_id);
                std::optional<std::string> prev_val;
                if (prev != axis_env.end()) prev_val = prev->second;
                axis_env[var_id] = var;
                const std::string iter_done = emit_node(fe.body, cur);
                emit(indent, cur + " = " + iter_done + ";");
                if (prev_val) {
                    axis_env[var_id] = *prev_val;
                } else {
                    axis_env.erase(var_id);
                }
                indent--;
                emit(indent, "}");
                return cur;
            }
            case NodeKind::Select: {
                const auto& sel = static_cast<const pto::wsp::ir::SelectNode&>(*n);
                const auto& sparse = *sel.sparse;
                const auto idx_var = sel.index_var;

                const uint64_t outer_sym = fnv1a_64(sparse.outer_size_var);
                const uint64_t indptr_sym = fnv1a_64(sparse.indptr_var);
                const uint64_t indices_sym = fnv1a_64(sparse.indices_var);

                const std::string base = "_sel_" + std::to_string(reinterpret_cast<uintptr_t>(&sel));
                const std::string outer = base + "_outer";
                const std::string indptr = base + "_indptr";
                const std::string indices = base + "_indices";
                const std::string nnz = base + "_nnz";
                const std::string done = new_tmp();

                emit(indent, "uint64_t " + done + " = " + ready_time + ";");
                emit(indent, "int64_t " + outer + " = (int64_t)ctx->get_symbol_u64(ctx->ctx, " + u64_hex_literal(outer_sym) + ");");
                emit(indent, "const int32_t* " + indptr + " = (const int32_t*)ctx->get_symbol_ptr(ctx->ctx, " + u64_hex_literal(indptr_sym) + ");");
                emit(indent, "const int32_t* " + indices + " = (const int32_t*)ctx->get_symbol_ptr(ctx->ctx, " + u64_hex_literal(indices_sym) + ");");
                emit(indent, "int64_t " + nnz + " = (" + indptr + " ? (int64_t)" + indptr + "[" + outer + "] : 0);");

                emit(indent, "for (int64_t _p = 0; _p < " + nnz + "; ++_p) {");
                indent++;
                emit(indent, "int64_t " + idx_var + " = " + indices + " ? (int64_t)" + indices + "[_p] : 0;");
                const uint64_t var_id = fnv1a_64(idx_var);
                const auto prev = axis_env.find(var_id);
                std::optional<std::string> prev_val;
                if (prev != axis_env.end()) prev_val = prev->second;
                axis_env[var_id] = idx_var;
                const std::string iter_done = emit_node(sel.body, ready_time);
                emit(indent, done + " = std::max(" + done + ", " + iter_done + ");");
                if (prev_val) {
                    axis_env[var_id] = *prev_val;
                } else {
                    axis_env.erase(var_id);
                }
                indent--;
                emit(indent, "}");
                return done;
            }
            case NodeKind::Cond: {
                const auto& c = static_cast<const pto::wsp::ir::CondNode&>(*n);
                if (!c.predicate) {
                    throw std::runtime_error("Cond predicate missing ScalarExpr (string predicate is not supported in codegen-first path)");
                }
                const std::string pred = emit_scalar_expr_cpp(c.predicate, axis_lookup, sym_u64_expr, slot_u64_expr);
                const std::string done = new_tmp();
                emit(indent, "uint64_t " + done + " = " + ready_time + ";");
                emit(indent, "if (" + pred + ") {");
                indent++;
                const std::string then_done = emit_node(c.then_branch, ready_time);
                emit(indent, done + " = " + then_done + ";");
                indent--;
                emit(indent, "} else {");
                indent++;
                const std::string else_done = emit_node(c.else_branch, ready_time);
                emit(indent, done + " = " + else_done + ";");
                indent--;
                emit(indent, "}");
                return done;
            }
            case NodeKind::Combine: {
                const auto& c = static_cast<const pto::wsp::ir::CombineNode&>(*n);
                const std::string done = new_tmp();
                emit(indent, "uint64_t " + done + " = " + ready_time + ";");
                for (const auto& ch : c.workloads) {
                    const std::string ch_done = emit_node(ch, ready_time);
                    emit(indent, done + " = std::max(" + done + ", " + ch_done + ");");
                }
                return done;
            }
            case NodeKind::Sequential: {
                const auto& s = static_cast<const pto::wsp::ir::SequentialNode&>(*n);
                const std::string cur = new_tmp();
                emit(indent, "uint64_t " + cur + " = " + ready_time + ";");
                for (const auto& ch : s.workloads) {
                    const std::string ch_done = emit_node(ch, cur);
                    emit(indent, cur + " = " + ch_done + ";");
                }
                return cur;
            }
            case NodeKind::Task: {
                const auto& t = static_cast<const pto::wsp::ir::TaskNode&>(*n);
                const auto* kd = find_kernel_def(module, t.kernel_name);
                if (!kd) throw std::runtime_error("Missing kernel def for task: " + t.kernel_name);

                const size_t num_user_axis_args = t.axis_args.size();
                const size_t num_scalar_args = t.scalar_args.size();

                // v9 tail/mask support: append per-tensor (valid_row, valid_col)
                // pairs as axis args, in kernel tensor-param order. Kernels use
                // these to set Tile valid dims and to size GlobalTensor shapes.
                std::vector<std::pair<int, int>> tensor_param_rc;
                tensor_param_rc.reserve(kd->params.size());
                for (const auto& kp : kd->params) {
                    const auto itv = kd->values.find(kp.id);
                    if (itv == kd->values.end()) throw std::runtime_error("Kernel value missing for param");
                    if (!itv->second.has_shape) continue;
                    tensor_param_rc.emplace_back(itv->second.rows, itv->second.cols);
                }
                const size_t num_tail_axis_args = tensor_param_rc.size() * 2;
                const size_t num_axis_args = num_user_axis_args + num_tail_axis_args;

                const std::string done = new_tmp();
                emit(indent, "uint64_t " + done + " = " + ready_time + ";");
                emit(indent, "{");
                indent++;

                // Axis args
                {
                    std::ostringstream args;
                    args << "uint64_t args[] = {";
                    for (size_t i = 0; i < num_user_axis_args; ++i) {
                        if (i) args << ", ";
                        args << axis_arg_to_u64_literal(t.axis_args[i]);
                    }
                    for (size_t i = 0; i < tensor_param_rc.size(); ++i) {
                        if (num_user_axis_args || i) args << ", ";
                        args << "(uint64_t)" << tensor_param_rc[i].first;
                        args << ", (uint64_t)" << tensor_param_rc[i].second;
                    }
                    for (size_t i = 0; i < num_scalar_args; ++i) {
                        if (i || num_axis_args) args << ", ";
                        args << axis_arg_to_u64_literal(t.scalar_args[i]);
                    }
                    args << "};";
                    emit(indent, args.str());
                }

                // Tensor bindings in kernel param order.
                std::vector<const pto::wsp::ir::CodegenTensorArg*> ordered;
                ordered.reserve(kd->params.size());
                for (const auto& kp : kd->params) {
                    const auto itv = kd->values.find(kp.id);
                    if (itv == kd->values.end()) throw std::runtime_error("Kernel value missing for param");
                    if (!itv->second.has_shape) continue;
                    const pto::wsp::ir::CodegenTensorArg* found = nullptr;
                    for (const auto& ta : t.tensor_args) {
                        if (ta.param == kp.name) {
                            found = &ta;
                            break;
                        }
                    }
                    if (!found) throw std::runtime_error("Missing tensor arg for param: " + kp.name);
                    ordered.push_back(found);
                }

                std::vector<std::string> tensor_ptrs;
                std::vector<std::string> tensor_strides;
                tensor_ptrs.reserve(ordered.size());
                tensor_strides.reserve(ordered.size() * 2);

                for (size_t i = 0; i < ordered.size(); ++i) {
                    const auto& a = *ordered[i];
                    const int tid = (int)a.tensor_id;
                    const std::string tile_ptr_name = "arg" + std::to_string(i) + "_ptr";
                    emit(indent, "void* " + tile_ptr_name + " = (void*)t" + std::to_string(tid) + "_base;");

                    // Apply index_exprs.
                    emit(indent, "{");
                    indent++;
                    emit(indent, "int64_t off_elems = 0;");
                    for (size_t d = 0; d < a.index_exprs.size(); ++d) {
                        emit(
                            indent,
                            "off_elems += (" + index_expr_to_cpp(a.index_exprs[d]) + ") * (int64_t)t" +
                                std::to_string(tid) + "_stride[" + std::to_string(d) + "];");
                    }
                    emit(
                        indent,
                        tile_ptr_name + " = (void*)((char*)t" + std::to_string(tid) +
                            "_base + off_elems * (int64_t)sizeof(*t" + std::to_string(tid) + "_base));");
                    indent--;
                    emit(indent, "}");

                    int s3_stride_dim = -1;
                    int s4_stride_dim = -1;
                    if (a.view_rank >= 2) {
                        s3_stride_dim = a.base_rank - a.view_rank;
                        s4_stride_dim = s3_stride_dim + 1;
                    } else if (a.view_rank == 1) {
                        s3_stride_dim = -1;
                        s4_stride_dim = a.base_rank - 1;
                    }

                    const std::string s3_name = "arg" + std::to_string(i) + "_s3";
                    const std::string s4_name = "arg" + std::to_string(i) + "_s4";

                    if (s3_stride_dim < 0) {
                        emit(indent, "uint64_t " + s3_name + " = 0;");
                    } else {
                        emit(indent, "uint64_t " + s3_name + " = t" + std::to_string(tid) + "_stride[" +
                                         std::to_string(s3_stride_dim) + "];");
                    }
                    if (s4_stride_dim < 0) {
                        emit(indent, "uint64_t " + s4_name + " = 1;");
                    } else {
                        emit(indent, "uint64_t " + s4_name + " = t" + std::to_string(tid) + "_stride[" +
                                         std::to_string(s4_stride_dim) + "];");
                    }

                    tensor_ptrs.push_back(tile_ptr_name);
                    tensor_strides.push_back(s3_name);
                    tensor_strides.push_back(s4_name);
                }

                {
                    std::ostringstream ptrs;
                    ptrs << "void* tensor_ptrs[] = {";
                    for (size_t i = 0; i < tensor_ptrs.size(); ++i) {
                        if (i) ptrs << ", ";
                        ptrs << tensor_ptrs[i];
                    }
                    ptrs << "};";
                    emit(indent, ptrs.str());
                }
                {
                    std::ostringstream strides;
                    strides << "uint64_t tensor_strides[] = {";
                    for (size_t i = 0; i < tensor_strides.size(); ++i) {
                        if (i) strides << ", ";
                        strides << tensor_strides[i];
                    }
                    strides << "};";
                    emit(indent, strides.str());
                }

                emit(
                    indent,
                    "KernelTaskDesc task_desc{args, " + std::to_string(num_axis_args + num_scalar_args) + ", " +
                        std::to_string(num_axis_args) + ", tensor_ptrs, tensor_strides, " +
                        std::to_string(tensor_ptrs.size()) + ", 0, 0};");

                const std::string dur = new_tmp();
                emit(indent, "uint64_t " + dur + " = (uint64_t)" + t.kernel_name + "(&task_desc, cspt);");

                const std::string tctx = new_tmp();
                emit(
                    indent,
                    "TaskContext " + tctx + "{args, " + std::to_string(num_axis_args + num_scalar_args) + "u, " +
                        std::to_string(num_axis_args) + "u, ctx};");

                const auto sym_u64_expr_task = [&](uint64_t id) -> std::string {
                    return "pto_wsp_symbol_u64(&" + tctx + ", " + u64_hex_literal(id) + ")";
                };
                const auto slot_u64_expr_task = [&](uint32_t i, pto::wsp::ir::ScalarType ty) -> std::string {
                    const std::string base = "pto_wsp_slot_u64(&" + tctx + ", " + std::to_string(i) + "u)";
                    if (ty == pto::wsp::ir::ScalarType::I64) return "(int64_t)" + base;
                    if (ty == pto::wsp::ir::ScalarType::U64) return "(uint64_t)" + base;
                    return "((bool)(" + base + " != 0ULL))";
                };
                const auto task_param_expr = [&](uint32_t i, pto::wsp::ir::ScalarType ty) -> std::string {
                    const std::string base = "pto_wsp_task_param_u64(&" + tctx + ", " + std::to_string(i) + "u)";
                    if (ty == pto::wsp::ir::ScalarType::I64) return "(int64_t)" + base;
                    if (ty == pto::wsp::ir::ScalarType::U64) return "(uint64_t)" + base;
                    return "((bool)(" + base + " != 0ULL))";
                };
                const auto task_tag_u64_expr = [&](uint64_t id) -> std::string {
                    return "pto_wsp_task_tag_u64(&" + tctx + ", " + u64_hex_literal(id) + ")";
                };

                const std::string wid = new_tmp();
                if (!dispatch) {
                    emit(indent, "uint32_t " + wid + " = 0;");
                } else if (dispatch->policy == DispatchPolicy::RoundRobin || dispatch->policy == DispatchPolicy::WorkSteal) {
                    emit(indent, "uint32_t " + wid + " = (uint32_t)(rr++ % (uint64_t)sched.num_workers());");
                } else {
                    const std::string key =
                        emit_scalar_expr_cpp(
                            dispatch->key, axis_lookup, sym_u64_expr_task, slot_u64_expr_task, task_param_expr, task_tag_u64_expr);
                    emit(indent, "uint64_t " + wid + "_key = (uint64_t)(" + key + ");");
                    emit(indent, "uint32_t " + wid + " = (uint32_t)(" + wid + "_key % (uint64_t)sched.num_workers());");
                }

                emit(indent, done + " = sched.issue(" + dur + ", " + wid + ", " + ready_time + ");");

                indent--;
                emit(indent, "}");
                return done;
            }
            default:
                throw std::runtime_error("Unhandled workload node kind in IR codegen");
        }
    };

    const std::string root_done = emit_node(wl.body, "(uint64_t)0");
    fn.stmt(cg::raw_stmt(""));
    fn.stmt(cg::ret(cg::ident(root_done)));

    tu.functions.push_back(std::move(fn).build());
    return cg::emit_cpp(tu);
}

std::string axis_arg_to_u64_expr_npu(const pto::wsp::ir::CodegenAxisArg& a) {
    if (a.is_var) return "(uint64_t)" + a.var;
    if (!a.var.empty()) {
        return "pto_wsp_sym_u64(plan, " + u64_hex_literal(a.u64) + ")";
    }
    return u64_hex_literal(a.u64);
}

std::string emit_aicpu_expand_cpp_from_ir(const pto::wsp::ir::Module& module) {
    using pto::wsp::ir::NodeKind;
    using pto::wsp::ir::DispatchPolicy;
    namespace cg = pto::wsp::codegen::cpp;

    if (module.workloads.empty()) throw std::runtime_error("Module has no workloads");
    const auto& wl = module.workloads[0];
    if (!wl.body) throw std::runtime_error("Workload body is null");

    // v9 NPU schedule preservation (emit-only here): preserve dispatch + task_window, mark others unsupported.
    const pto::wsp::ir::DispatchNode* dispatch = nullptr;
    const pto::wsp::ir::TaskWindowNode* task_window = nullptr;
    const pto::wsp::ir::StreamNode* streams = nullptr;
    std::vector<std::string> unsupported_schedule;

    if (const auto* s = module.findScheduleForWorkload(wl.name)) {
        for (const auto& d : s->directives) {
            if (!d) continue;
            switch (d->kind) {
                case NodeKind::Dispatch:
                    if (!dispatch) dispatch = static_cast<const pto::wsp::ir::DispatchNode*>(d.get());
                    break;
                case NodeKind::TaskWindow:
                    if (!task_window) task_window = static_cast<const pto::wsp::ir::TaskWindowNode*>(d.get());
                    break;
                case NodeKind::Stream:
                    if (!streams) streams = static_cast<const pto::wsp::ir::StreamNode*>(d.get());
                    break;
                default:
                    unsupported_schedule.push_back(pto::wsp::ir::nodeKindToString(d->kind));
                    break;
            }
        }
    }

    uint32_t dispatch_policy_u32 = 0;
    uint32_t dispatch_num_targets = 1;
    if (dispatch) {
        dispatch_policy_u32 = static_cast<uint32_t>(dispatch->policy);
        if (dispatch->num_targets > 0) {
            dispatch_num_targets = static_cast<uint32_t>(dispatch->num_targets);
        } else if (streams && streams->num_streams > 0) {
            dispatch_num_targets = static_cast<uint32_t>(streams->num_streams);
            if (streams->key) unsupported_schedule.push_back("stream_by");
        } else {
            // v9: target count must be explicit; default to 1 unless streams provided.
            dispatch_num_targets = 1;
        }
        const bool key_required =
            dispatch->policy == DispatchPolicy::Affinity || dispatch->policy == DispatchPolicy::Hash ||
            dispatch->policy == DispatchPolicy::Custom;
        if (key_required && !dispatch->key) {
            throw std::runtime_error(
                "NPU dispatch policy requires ScalarExpr key (string-eval is not supported in v9 codegen-first)");
        }
        if (dispatch->policy == DispatchPolicy::WorkSteal) {
            unsupported_schedule.push_back("dispatch(work_steal)");
        }
    } else if (streams && streams->num_streams > 0) {
        // No dispatch directive; keep defaults but record streams config as unsupported (v9 NPU preserves dispatch + task_window only).
        unsupported_schedule.push_back("streams");
    }
    if (dispatch_num_targets == 0) dispatch_num_targets = 1;

    uint64_t task_window_tasks = 0;  // 0 = unlimited
    if (task_window) {
        const bool ok_unit = task_window->unit == "tasks";
        const bool ok_overflow = task_window->overflow == pto::wsp::ir::TaskWindowOverflowPolicy::Stall;
        if (ok_unit && ok_overflow) {
            task_window_tasks = task_window->size > 0 ? static_cast<uint64_t>(task_window->size) : 0ULL;
        } else {
            unsupported_schedule.push_back("task_window(non-stall or non-tasks)");
        }
    }

    cg::TranslationUnit tu;
    tu.includes = {
        "\"pto/rt/codegen/abi/npu_plan_abi.hpp\"",
        "<cstdint>",
        "<cstddef>",
    };

    if (!unsupported_schedule.empty()) {
        tu.raw_toplevel.push_back("// [pto-wsp] v9: unsupported schedule directives ignored in NPU emission:");
        for (const auto& u : unsupported_schedule) {
            tu.raw_toplevel.push_back("//   - " + u);
        }
        tu.raw_toplevel.push_back("");
    }

    // Helpers (symbol lookup; linear scan is OK for bring-up).
    tu.raw_toplevel.push_back("static inline uint64_t pto_wsp_sym_u64(const NpuPlanDesc* plan, uint64_t id) {");
    tu.raw_toplevel.push_back("  for (uint32_t i = 0; i < plan->num_sym_u64; ++i) {");
    tu.raw_toplevel.push_back("    if (plan->sym_u64[i].id == id) return plan->sym_u64[i].value;");
    tu.raw_toplevel.push_back("  }");
    tu.raw_toplevel.push_back("  return 0;");
    tu.raw_toplevel.push_back("}");
    tu.raw_toplevel.push_back("");
    tu.raw_toplevel.push_back("static inline uint64_t pto_wsp_sym_ptr(const NpuPlanDesc* plan, uint64_t id) {");
    tu.raw_toplevel.push_back("  for (uint32_t i = 0; i < plan->num_sym_ptr; ++i) {");
    tu.raw_toplevel.push_back("    if (plan->sym_ptr[i].id == id) return plan->sym_ptr[i].ptr;");
    tu.raw_toplevel.push_back("  }");
    tu.raw_toplevel.push_back("  return 0;");
    tu.raw_toplevel.push_back("}");
    tu.raw_toplevel.push_back("");

    // Kernel name -> kernel_id mapping (stable within an emitted artifact).
    tu.raw_toplevel.push_back("enum : uint32_t {");
    for (size_t i = 0; i < module.kernels.size(); ++i) {
        tu.raw_toplevel.push_back("  PTO_WSP_KERNEL_ID_" + module.kernels[i].name + " = " + std::to_string(i) + ",");
    }
    tu.raw_toplevel.push_back("};");

    cg::FunctionBuilder fn("void", "pto_wsp_aicpu_expand");
    fn.extern_c().param("NpuPlanDesc*", "plan");
    fn.stmt(cg::raw_stmt("plan->num_tasks = 0;"));
    fn.stmt(cg::raw_stmt("for (uint32_t i = 0; i < PTO_WSP_NPU_MAX_SLOTS; ++i) plan->slots_u64[i] = 0;"));
    fn.stmt(cg::raw_stmt("plan->dispatch_policy = " + std::to_string(dispatch_policy_u32) + "u;"));
    fn.stmt(cg::raw_stmt("plan->dispatch_num_targets = " + std::to_string(dispatch_num_targets) + "u;"));
    fn.stmt(cg::raw_stmt("plan->task_window_tasks = " + std::to_string(task_window_tasks) + "ULL;"));
    fn.stmt(cg::raw_stmt("uint64_t rr = 0;"));

    // Cache tensor stride base pointers for convenience.
    for (size_t tid = 0; tid < module.tensors.size(); ++tid) {
        fn.stmt(cg::raw_stmt(
            "const uint64_t* t" + std::to_string(tid) + "_stride = &plan->tensor_strides[" +
            std::to_string(tid) + " * PTO_WSP_NPU_MAX_TENSOR_RANK];"));
        fn.stmt(cg::raw_stmt(
            "uint64_t t" + std::to_string(tid) + "_base = plan->tensor_base_ptrs[" + std::to_string(tid) + "];"));
    }
    fn.stmt(cg::raw_stmt(""));

    auto emit_line = [&](int indent, const std::string& s) {
        fn.stmt(cg::raw_stmt(std::string(static_cast<size_t>(indent) * 2, ' ') + s));
    };

    int indent = 0;
    std::unordered_map<uint64_t, std::string> axis_env;
    const auto axis_lookup = [&](uint64_t id) -> std::optional<std::string> {
        const auto it = axis_env.find(id);
        if (it == axis_env.end()) return std::nullopt;
        return it->second;
    };
    const auto sym_u64_expr = [&](uint64_t id) -> std::string {
        return "pto_wsp_sym_u64(plan, " + u64_hex_literal(id) + ")";
    };
    const auto slot_u64_expr = [&](uint32_t i, pto::wsp::ir::ScalarType ty) -> std::string {
        const std::string base = "plan->slots_u64[" + std::to_string(i) + "u]";
        if (ty == pto::wsp::ir::ScalarType::I64) return "(int64_t)" + base;
        if (ty == pto::wsp::ir::ScalarType::U64) return "(uint64_t)" + base;
        return "((bool)(" + base + " != 0ULL))";
    };

    std::function<void(const pto::wsp::ir::IRPtr<pto::wsp::ir::WorkloadNode>&)> emit_node;
    emit_node = [&](const auto& n) {
        if (!n) return;

        switch (n->kind) {
            case NodeKind::SlotSetU64: {
                const auto& s = static_cast<const pto::wsp::ir::SlotSetU64Node&>(*n);
                emit_line(indent, "plan->slots_u64[" + std::to_string(s.slot) + "u] = " + u64_hex_literal(s.value) + ";");
                return;
            }
            case NodeKind::SlotLoadU64: {
                const auto& l = static_cast<const pto::wsp::ir::SlotLoadU64Node&>(*n);
                const int tid = (int)l.tensor.tensor_id;
                const int item = dtype_size_bytes(dtype_to_str(module.tensors[tid].dtype));
                emit_line(indent, "{");
                indent++;
                emit_line(indent, "int64_t off_elems = 0;");
                for (size_t d = 0; d < l.tensor.index_exprs.size(); ++d) {
                    emit_line(
                        indent,
                        "off_elems += (" + index_expr_to_cpp(l.tensor.index_exprs[d]) + ") * (int64_t)t" +
                            std::to_string(tid) + "_stride[" + std::to_string(d) + "];");
                }
                emit_line(
                    indent,
                    "uint64_t base = t" + std::to_string(tid) + "_base + (uint64_t)off_elems * " + std::to_string(item) +
                        "ULL;");

                int s3_stride_dim = -1;
                int s4_stride_dim = -1;
                if (l.tensor.view_rank >= 2) {
                    s3_stride_dim = l.tensor.base_rank - l.tensor.view_rank;
                    s4_stride_dim = s3_stride_dim + 1;
                } else if (l.tensor.view_rank == 1) {
                    s3_stride_dim = -1;
                    s4_stride_dim = l.tensor.base_rank - 1;
                }

                emit_line(
                    indent,
                    "uint64_t s3 = " + std::string(s3_stride_dim < 0 ? "0" : ("t" + std::to_string(tid) + "_stride[" + std::to_string(s3_stride_dim) + "]")) + ";");
                emit_line(
                    indent,
                    "uint64_t s4 = " + std::string(s4_stride_dim < 0 ? "1" : ("t" + std::to_string(tid) + "_stride[" + std::to_string(s4_stride_dim) + "]")) + ";");

                emit_line(
                    indent,
                    "uint64_t addr = base + (uint64_t)((int64_t)" + std::to_string(l.row) + " * (int64_t)s3 + (int64_t)" +
                        std::to_string(l.col) + " * (int64_t)s4) * " + std::to_string(item) + "ULL;");

                const auto dt = module.tensors[tid].dtype;
                std::string load_expr = "(uint64_t)0";
                if (dt == pto::wsp::ir::DType::I32) {
                    load_expr = "(uint64_t)(*(const int32_t*)(uintptr_t)addr)";
                } else if (dt == pto::wsp::ir::DType::I64) {
                    load_expr = "(uint64_t)(*(const int64_t*)(uintptr_t)addr)";
                } else if (dt == pto::wsp::ir::DType::U64) {
                    load_expr = "(*(const uint64_t*)(uintptr_t)addr)";
                } else if (dt == pto::wsp::ir::DType::Bool) {
                    load_expr = "(uint64_t)(*(const uint8_t*)(uintptr_t)addr)";
                } else {
                    // v9: keep bring-up simple; treat other types as unsupported for slot loads.
                    load_expr = "(uint64_t)0";
                }

                emit_line(indent, "plan->slots_u64[" + std::to_string(l.slot) + "u] = " + load_expr + ";");
                indent--;
                emit_line(indent, "}");
                return;
            }
            case NodeKind::ParallelFor: {
                const auto& pf = static_cast<const pto::wsp::ir::ParallelForNode&>(*n);
                const auto var = pf.index_var;
                const uint64_t var_id = fnv1a_64(var);
                if (pf.axis->kind == NodeKind::DenseDynAxis) {
                    const auto& ax = static_cast<const pto::wsp::ir::DenseDynAxisNode&>(*pf.axis);
                    const auto sym = fnv1a_64(ax.size_var);
                    emit_line(indent, "int64_t " + var + "_ub = (int64_t)pto_wsp_sym_u64(plan, " + u64_hex_literal(sym) + ");");
                    emit_line(indent, "for (int64_t " + var + " = 0; " + var + " < " + var + "_ub; ++" + var + ") {");
                } else if (pf.axis->kind == NodeKind::DenseAxis) {
                    const auto& ax = static_cast<const pto::wsp::ir::DenseAxisNode&>(*pf.axis);
                    emit_line(indent, "for (int64_t " + var + " = 0; " + var + " < " + std::to_string(ax.size) + "; ++" + var + ") {");
                } else if (pf.axis->kind == NodeKind::RaggedAxis) {
                    const auto& ax = static_cast<const pto::wsp::ir::RaggedAxisNode&>(*pf.axis);
                    const uint64_t outer_sym = fnv1a_64(ax.outer_size_var);
                    const uint64_t lengths_sym = fnv1a_64(ax.lengths_var);
                    emit_line(indent, "int64_t " + var + "_outer = (int64_t)pto_wsp_sym_u64(plan, " + u64_hex_literal(outer_sym) + ");");
                    emit_line(indent, "const int32_t* " + var + "_lengths = (const int32_t*)pto_wsp_sym_ptr(plan, " + u64_hex_literal(lengths_sym) + ");");
                    emit_line(indent, "int64_t " + var + "_ub = 0;");
                    emit_line(indent, "if (" + var + "_lengths) { for (int64_t _i = 0; _i < " + var + "_outer; ++_i) { " + var + "_ub += (int64_t)" + var + "_lengths[_i]; } }");
                    emit_line(indent, "for (int64_t " + var + " = 0; " + var + " < " + var + "_ub; ++" + var + ") {");
                } else {
                    throw std::runtime_error("Unsupported axis kind for parallel_for in AICPU codegen");
                }
                indent++;
                const auto prev = axis_env.find(var_id);
                std::optional<std::string> prev_val;
                if (prev != axis_env.end()) prev_val = prev->second;
                axis_env[var_id] = var;
                emit_node(pf.body);
                if (prev_val) {
                    axis_env[var_id] = *prev_val;
                } else {
                    axis_env.erase(var_id);
                }
                indent--;
                emit_line(indent, "}");
                return;
            }
            case NodeKind::ForEach: {
                const auto& fe = static_cast<const pto::wsp::ir::ForEachNode&>(*n);
                const auto var = fe.index_var;
                const uint64_t var_id = fnv1a_64(var);
                if (fe.axis->kind == NodeKind::DenseDynAxis) {
                    const auto& ax = static_cast<const pto::wsp::ir::DenseDynAxisNode&>(*fe.axis);
                    const auto sym = fnv1a_64(ax.size_var);
                    emit_line(indent, "int64_t " + var + "_ub = (int64_t)pto_wsp_sym_u64(plan, " + u64_hex_literal(sym) + ");");
                    emit_line(indent, "for (int64_t " + var + " = 0; " + var + " < " + var + "_ub; ++" + var + ") {");
                } else if (fe.axis->kind == NodeKind::DenseAxis) {
                    const auto& ax = static_cast<const pto::wsp::ir::DenseAxisNode&>(*fe.axis);
                    emit_line(indent, "for (int64_t " + var + " = 0; " + var + " < " + std::to_string(ax.size) + "; ++" + var + ") {");
                } else if (fe.axis->kind == NodeKind::RaggedAxis) {
                    const auto& ax = static_cast<const pto::wsp::ir::RaggedAxisNode&>(*fe.axis);
                    const uint64_t outer_sym = fnv1a_64(ax.outer_size_var);
                    const uint64_t lengths_sym = fnv1a_64(ax.lengths_var);
                    emit_line(indent, "int64_t " + var + "_outer = (int64_t)pto_wsp_sym_u64(plan, " + u64_hex_literal(outer_sym) + ");");
                    emit_line(indent, "const int32_t* " + var + "_lengths = (const int32_t*)pto_wsp_sym_ptr(plan, " + u64_hex_literal(lengths_sym) + ");");
                    emit_line(indent, "int64_t " + var + "_ub = 0;");
                    emit_line(indent, "if (" + var + "_lengths) { for (int64_t _i = 0; _i < " + var + "_outer; ++_i) { " + var + "_ub += (int64_t)" + var + "_lengths[_i]; } }");
                    emit_line(indent, "for (int64_t " + var + " = 0; " + var + " < " + var + "_ub; ++" + var + ") {");
                } else {
                    throw std::runtime_error("Unsupported axis kind for for_each in AICPU codegen");
                }
                indent++;
                const auto prev = axis_env.find(var_id);
                std::optional<std::string> prev_val;
                if (prev != axis_env.end()) prev_val = prev->second;
                axis_env[var_id] = var;
                emit_node(fe.body);
                if (prev_val) {
                    axis_env[var_id] = *prev_val;
                } else {
                    axis_env.erase(var_id);
                }
                indent--;
                emit_line(indent, "}");
                return;
            }
            case NodeKind::Select: {
                const auto& sel = static_cast<const pto::wsp::ir::SelectNode&>(*n);
                const auto& sparse = *sel.sparse;
                const auto idx_var = sel.index_var;

                const uint64_t outer_sym = fnv1a_64(sparse.outer_size_var);
                const uint64_t indptr_sym = fnv1a_64(sparse.indptr_var);
                const uint64_t indices_sym = fnv1a_64(sparse.indices_var);

                const std::string base = "_sel_" + std::to_string(reinterpret_cast<uintptr_t>(&sel));
                const std::string outer = base + "_outer";
                const std::string indptr = base + "_indptr";
                const std::string indices = base + "_indices";
                const std::string nnz = base + "_nnz";

                emit_line(indent, "int64_t " + outer + " = (int64_t)pto_wsp_sym_u64(plan, " + u64_hex_literal(outer_sym) + ");");
                emit_line(indent, "const int32_t* " + indptr + " = (const int32_t*)pto_wsp_sym_ptr(plan, " + u64_hex_literal(indptr_sym) + ");");
                emit_line(indent, "const int32_t* " + indices + " = (const int32_t*)pto_wsp_sym_ptr(plan, " + u64_hex_literal(indices_sym) + ");");
                emit_line(indent, "int64_t " + nnz + " = (" + indptr + " ? (int64_t)" + indptr + "[" + outer + "] : 0);");
                emit_line(indent, "for (int64_t _p = 0; _p < " + nnz + "; ++_p) {");
                indent++;
                emit_line(indent, "int64_t " + idx_var + " = " + indices + " ? (int64_t)" + indices + "[_p] : 0;");
                const uint64_t var_id = fnv1a_64(idx_var);
                const auto prev = axis_env.find(var_id);
                std::optional<std::string> prev_val;
                if (prev != axis_env.end()) prev_val = prev->second;
                axis_env[var_id] = idx_var;
                emit_node(sel.body);
                if (prev_val) {
                    axis_env[var_id] = *prev_val;
                } else {
                    axis_env.erase(var_id);
                }
                indent--;
                emit_line(indent, "}");
                return;
            }
            case NodeKind::Cond: {
                const auto& c = static_cast<const pto::wsp::ir::CondNode&>(*n);
                if (!c.predicate) {
                    throw std::runtime_error("Cond predicate missing ScalarExpr (string predicate is not supported in codegen-first path)");
                }
                const std::string pred = emit_scalar_expr_cpp(c.predicate, axis_lookup, sym_u64_expr, slot_u64_expr);
                emit_line(indent, "if (" + pred + ") {");
                indent++;
                emit_node(c.then_branch);
                indent--;
                emit_line(indent, "} else {");
                indent++;
                emit_node(c.else_branch);
                indent--;
                emit_line(indent, "}");
                return;
            }
            case NodeKind::Combine: {
                const auto& c = static_cast<const pto::wsp::ir::CombineNode&>(*n);
                for (const auto& ch : c.workloads) emit_node(ch);
                return;
            }
            case NodeKind::Sequential: {
                const auto& s = static_cast<const pto::wsp::ir::SequentialNode&>(*n);
                for (const auto& ch : s.workloads) emit_node(ch);
                return;
            }
            case NodeKind::Task: {
                const auto& t = static_cast<const pto::wsp::ir::TaskNode&>(*n);
                const auto* kd = find_kernel_def(module, t.kernel_name);
                if (!kd) throw std::runtime_error("Missing kernel def for task: " + t.kernel_name);

                const size_t num_user_axis_args = t.axis_args.size();
                const size_t num_scalar_args = t.scalar_args.size();

                std::vector<std::pair<int, int>> tensor_param_rc;
                tensor_param_rc.reserve(kd->params.size());
                for (const auto& kp : kd->params) {
                    const auto itv = kd->values.find(kp.id);
                    if (itv == kd->values.end()) throw std::runtime_error("Kernel value missing for param");
                    if (!itv->second.has_shape) continue;
                    tensor_param_rc.emplace_back(itv->second.rows, itv->second.cols);
                }
                const size_t num_tail_axis_args = tensor_param_rc.size() * 2;
                const size_t num_axis_args = num_user_axis_args + num_tail_axis_args;
                const size_t num_args = num_axis_args + num_scalar_args;

                emit_line(indent, "if (plan->num_tasks >= plan->max_tasks) return;");
                emit_line(indent, "NpuTaskDesc* task = &plan->tasks[plan->num_tasks++];");
                emit_line(indent, "task->kernel_id = PTO_WSP_KERNEL_ID_" + t.kernel_name + ";");
                emit_line(indent, "task->task_id = (uint32_t)(plan->num_tasks - 1);");
                emit_line(indent, "task->num_args = " + std::to_string(num_args) + ";");
                emit_line(indent, "task->num_axis_args = " + std::to_string(num_axis_args) + ";");

                // Args (axis + scalar)
                for (size_t i = 0; i < num_user_axis_args; ++i) {
                    emit_line(indent, "task->args[" + std::to_string(i) + "] = " + axis_arg_to_u64_expr_npu(t.axis_args[i]) + ";");
                }
                for (size_t i = 0; i < tensor_param_rc.size(); ++i) {
                    const size_t base = num_user_axis_args + i * 2;
                    emit_line(indent, "task->args[" + std::to_string(base + 0) + "] = (uint64_t)" + std::to_string(tensor_param_rc[i].first) + ";");
                    emit_line(indent, "task->args[" + std::to_string(base + 1) + "] = (uint64_t)" + std::to_string(tensor_param_rc[i].second) + ";");
                }
                for (size_t i = 0; i < num_scalar_args; ++i) {
                    emit_line(indent, "task->args[" + std::to_string(num_axis_args + i) + "] = " +
                                         axis_arg_to_u64_expr_npu(t.scalar_args[i]) + ";");
                }

                // v9 scheduling: preserve dispatch assignment by writing task->target_id.
                if (!dispatch) {
                    emit_line(indent, "task->target_id = 0;");
                } else if (dispatch->policy == DispatchPolicy::RoundRobin || dispatch->policy == DispatchPolicy::WorkSteal) {
                    emit_line(
                        indent,
                        "task->target_id = (uint32_t)(rr++ % (uint64_t)" + std::to_string(dispatch_num_targets) + "ULL);");
                } else {
                    const auto task_param_expr = [&](uint32_t i, pto::wsp::ir::ScalarType ty) -> std::string {
                        const std::string base = "(uint64_t)task->args[" + std::to_string(i) + "]";
                        if (ty == pto::wsp::ir::ScalarType::I64) return "(int64_t)" + base;
                        if (ty == pto::wsp::ir::ScalarType::U64) return "(uint64_t)" + base;
                        return "((bool)(" + base + " != 0ULL))";
                    };
                    const auto task_tag_u64_expr = [&](uint64_t id) -> std::string {
                        return sym_u64_expr(id);
                    };
                    const std::string key =
                        emit_scalar_expr_cpp(dispatch->key, axis_lookup, sym_u64_expr, slot_u64_expr, task_param_expr, task_tag_u64_expr);
                    emit_line(indent, "uint64_t _pto_wsp_dispatch_key = (uint64_t)(" + key + ");");
                    emit_line(
                        indent,
                        "task->target_id = (uint32_t)(_pto_wsp_dispatch_key % (uint64_t)" +
                            std::to_string(dispatch_num_targets) + "ULL);");
                }

                // Tensor bindings in kernel param order.
                std::vector<const pto::wsp::ir::CodegenTensorArg*> ordered;
                ordered.reserve(kd->params.size());
                for (const auto& kp : kd->params) {
                    const auto itv = kd->values.find(kp.id);
                    if (itv == kd->values.end()) throw std::runtime_error("Kernel value missing for param");
                    if (!itv->second.has_shape) continue;
                    const pto::wsp::ir::CodegenTensorArg* found = nullptr;
                    for (const auto& ta : t.tensor_args) {
                        if (ta.param == kp.name) {
                            found = &ta;
                            break;
                        }
                    }
                    if (!found) throw std::runtime_error("Missing tensor arg for param: " + kp.name);
                    ordered.push_back(found);
                }

                emit_line(indent, "task->num_tensors = " + std::to_string(ordered.size()) + ";");

                for (size_t i = 0; i < ordered.size(); ++i) {
                    const auto& a = *ordered[i];
                    const int tid = (int)a.tensor_id;
                    const int item = dtype_size_bytes(module.tensors[tid].dtype);

                    emit_line(indent, "{");
                    indent++;
                    emit_line(indent, "int64_t off_elems = 0;");
                    for (size_t d = 0; d < a.index_exprs.size(); ++d) {
                        emit_line(
                            indent,
                            "off_elems += (" + index_expr_to_cpp(a.index_exprs[d]) + ") * (int64_t)t" +
                                std::to_string(tid) + "_stride[" + std::to_string(d) + "];");
                    }
                    emit_line(
                        indent,
                        "task->tensor_ptrs[" + std::to_string(i) + "] = (uint64_t)((uint8_t*)t" +
                            std::to_string(tid) + "_base + off_elems * " + std::to_string(item) + ");");

                    int s3_stride_dim = -1;
                    int s4_stride_dim = -1;
                    if (a.view_rank >= 2) {
                        s3_stride_dim = a.base_rank - a.view_rank;
                        s4_stride_dim = s3_stride_dim + 1;
                    } else if (a.view_rank == 1) {
                        s3_stride_dim = -1;
                        s4_stride_dim = a.base_rank - 1;
                    }
                    if (s3_stride_dim < 0) {
                        emit_line(indent, "task->tensor_strides[" + std::to_string(i * 2) + "] = 0;");
                    } else {
                        emit_line(
                            indent,
                            "task->tensor_strides[" + std::to_string(i * 2) + "] = t" +
                                std::to_string(tid) + "_stride[" + std::to_string(s3_stride_dim) + "];");
                    }
                    if (s4_stride_dim < 0) {
                        emit_line(indent, "task->tensor_strides[" + std::to_string(i * 2 + 1) + "] = 0;");
                    } else {
                        emit_line(
                            indent,
                            "task->tensor_strides[" + std::to_string(i * 2 + 1) + "] = t" +
                                std::to_string(tid) + "_stride[" + std::to_string(s4_stride_dim) + "];");
                    }
                    indent--;
                    emit_line(indent, "}");
                }

                // Initialize timing placeholders.
                emit_line(indent, "task->t_begin = 0; task->t_end = 0;");
                return;
            }
            default:
                throw std::runtime_error("Unhandled workload node kind in AICPU codegen");
        }
    };

    emit_node(wl.body);
    tu.functions.push_back(std::move(fn).build());
    return cg::emit_cpp(tu);
}

std::string emit_aicore_dispatch_cpp_from_ir(const pto::wsp::ir::Module& module) {
    namespace cg = pto::wsp::codegen::cpp;

    cg::TranslationUnit tu;
    tu.includes = {
        "\"pto/rt/codegen/abi/npu_plan_abi.hpp\"",
        "\"pto/rt/codegen/abi/kernel_abi.hpp\"",
        "<cstdint>",
        "<cstddef>",
    };

    for (const auto& k : module.kernels) {
        tu.raw_toplevel.push_back(
            "extern \"C\" PTO_WSP_KERNEL_ATTR uint64_t " + k.name + "(const KernelTaskDesc* task, CSPTContext* cspt);");
    }

    cg::FunctionBuilder fn("PTO_WSP_KERNEL_ATTR void", "pto_wsp_aicore_dispatch_one");
    fn.extern_c().param("NpuTaskDesc*", "task");

    // NOTE: In environments without Ascend/CANN, this is emitted to a "looks correct"
    // state and is not executed. We still define the timing contract:
    // - per-kernel cycle reports are returned by the generated PTO-ISA kernel (u64)
    // - the dispatcher records them into task->t_end (task->t_begin reserved for future timestamps)
    fn.stmt(cg::raw_stmt("void* tensor_ptrs[PTO_WSP_NPU_MAX_TENSORS];"));
    fn.stmt(cg::raw_stmt(
        "for (uint32_t i = 0; i < task->num_tensors; ++i) tensor_ptrs[i] = (void*)task->tensor_ptrs[i];"));
    fn.stmt(cg::raw_stmt(
        "KernelTaskDesc kt{task->args, task->num_args, task->num_axis_args, tensor_ptrs, task->tensor_strides, "
        "task->num_tensors, task->kernel_id, task->task_id};"));
    fn.stmt(cg::raw_stmt("uint64_t cycles = 0;"));
    fn.stmt(cg::raw_stmt("switch (task->kernel_id) {"));
    for (size_t i = 0; i < module.kernels.size(); ++i) {
        fn.stmt(cg::raw_stmt(
            "  case " + std::to_string(i) + ": cycles = (uint64_t)" + module.kernels[i].name +
            "(&kt, nullptr); break;"));
    }
    fn.stmt(cg::raw_stmt("  default: break;"));
    fn.stmt(cg::raw_stmt("}"));
    fn.stmt(cg::raw_stmt("task->t_begin = 0;"));
    fn.stmt(cg::raw_stmt("task->t_end = cycles;"));

    tu.functions.push_back(std::move(fn).build());
    return cg::emit_cpp(tu);
}

static std::optional<std::string> get_kernel_attr_string(const pto::wsp::ir::CodegenKernelDef& kd,
                                                         const std::string& key) {
    auto it = kd.attrs.find(key);
    if (it == kd.attrs.end()) return std::nullopt;
    if (auto* s = std::get_if<std::string>(&it->second)) return *s;
    return std::nullopt;
}

static std::vector<std::string> get_kernel_attr_string_list(const pto::wsp::ir::CodegenKernelDef& kd,
                                                            const std::string& key) {
    auto it = kd.attrs.find(key);
    if (it == kd.attrs.end()) return {};
    if (auto* v = std::get_if<std::vector<std::string>>(&it->second)) return *v;
    return {};
}

static std::string emit_custom_kernel_cpp(const pto::wsp::ir::CodegenKernelDef& kd) {
    const auto body = get_kernel_attr_string(kd, "cpp_src").value_or("");
    const auto extra_includes = get_kernel_attr_string_list(kd, "cpp_includes");

    std::ostringstream out;
    out << "// Custom kernel (Path A): user-provided PTO-ISA tile code\n";
    out << "#include \"pto/rt/codegen/abi/kernel_abi.hpp\"\n";
    out << "#include \"pto/rt/codegen/abi/ptoisa_bridge.hpp\"\n";
    out << "#include <cstdint>\n";
    out << "#include <cstddef>\n";
    for (const auto& inc : extra_includes) {
        out << "#include \"" << inc << "\"\n";
    }
    out << "\n";
    out << "extern \"C\" PTO_WSP_KERNEL_ATTR uint64_t " << kd.name
        << "(const KernelTaskDesc* task, CSPTContext* cspt) {\n";
    out << "  (void)task;\n";
    out << "  (void)cspt;\n";
    out << "  uint64_t cycles = 0;\n";
    out << "#if defined(__CPU_SIM)\n";
    out << "  pto::cpu_sim::reset_cycles();\n";
    out << "#endif\n";
    out << body << "\n";
    out << "#if defined(__CPU_SIM)\n";
    out << "  if (cycles == 0) cycles = pto::cpu_sim::read_cycles();\n";
    out << "#endif\n";
    out << "  return cycles;\n";
    out << "}\n";
    return out.str();
}

py::dict codegen_compile_ir(const pto::wsp::ir::Module& module,
                            const std::string& output_name) {
    const auto tc = pto::wsp::ir::type_check(module);
    if (!tc.valid) {
        throw std::runtime_error(tc.to_string());
    }

    std::map<std::string, std::string> sources;
    sources.emplace("workload_main.cpp", emit_workload_cpp_from_ir_ast(module));
    for (const auto& kd : module.kernels) {
        if (get_kernel_attr_string(kd, "cpp_file_src").has_value()) {
            sources.emplace("kernel_" + kd.name + ".cpp", get_kernel_attr_string(kd, "cpp_file_src").value());
        } else if (get_kernel_attr_string(kd, "cpp_src").has_value()) {
            sources.emplace("kernel_" + kd.name + ".cpp", emit_custom_kernel_cpp(kd));
        } else {
            sources.emplace("kernel_" + kd.name + ".cpp", emit_kernel_cpp(kernel_plan_from_ir(kd)));
        }
    }

    auto r = pto::wsp::codegen::compile_sources_via_cmake(sources, output_name);
    py::dict out;
    out["so_path"] = r.so_path;
    out["cache_key"] = r.cache_key;
    out["entrypoint"] = output_name + "_main";
    return out;
}

py::dict compile_codegen(const pto::wsp::ir::Module& module,
                         const pto::wsp::backend::CompileOptions& options) {
    const auto target = options.target.empty() ? std::string("cpu_sim") : options.target;
    if (target == "cpu_sim" || target == "cpu_sim_codegen") {
        const std::string output_name = module.name.empty() ? "main" : module.name;
        auto out = codegen_compile_ir(module, output_name);
        out["target"] = target;
        out["can_execute"] = true;
        return out;
    }

    if (target == "ascend_npu" || target == "ascend") {
        const auto tc = pto::wsp::ir::type_check(module);
        if (!tc.valid) {
            throw std::runtime_error(tc.to_string());
        }

        const std::string output_name = module.name.empty() ? "main" : module.name;

        std::map<std::string, std::string> sources;

        // Host stub always builds without Ascend toolchain.
        {
            std::ostringstream hs;
            hs << "#include \"pto/rt/codegen/abi/npu_plan_abi.hpp\"\n";
            hs << "extern \"C\" void pto_wsp_npu_host_stub(const NpuPlanDesc* plan) { (void)plan; }\n";
            sources.emplace("host/runner_stub.cpp", hs.str());
        }

        sources.emplace("aicpu/expand.cpp", emit_aicpu_expand_cpp_from_ir(module));
        sources.emplace("aicore/dispatch.cpp", emit_aicore_dispatch_cpp_from_ir(module));
        for (const auto& kd : module.kernels) {
            if (get_kernel_attr_string(kd, "cpp_file_src").has_value()) {
                sources.emplace("aicore/kernel_" + kd.name + ".cpp", get_kernel_attr_string(kd, "cpp_file_src").value());
            } else if (get_kernel_attr_string(kd, "cpp_src").has_value()) {
                sources.emplace("aicore/kernel_" + kd.name + ".cpp", emit_custom_kernel_cpp(kd));
            } else {
                sources.emplace("aicore/kernel_" + kd.name + ".cpp", emit_kernel_cpp(kernel_plan_from_ir(kd)));
            }
        }

        // Rewrite CMakeLists now that kernels are known.
        {
            std::ostringstream cm;
            cm << "cmake_minimum_required(VERSION 3.16)\n"
               << "project(pto_wsp_codegen_" << output_name << "_ascend_npu LANGUAGES CXX)\n"
               << "set(CMAKE_CXX_STANDARD 23)\n"
               << "set(CMAKE_CXX_STANDARD_REQUIRED ON)\n"
               << "set(CMAKE_POSITION_INDEPENDENT_CODE ON)\n"
               << "option(PTO_WSP_ENABLE_ASCEND \\\"Build Ascend device artifacts (AICPU/AICore)\\\" OFF)\n"
               << "set(PTO_WSP_SOURCE_DIR \\\"\\\" CACHE PATH \\\"Path to pto-wsp repo (for headers)\\\")\n"
               << "if (PTO_WSP_SOURCE_DIR STREQUAL \\\"\\\")\n"
               << "  message(FATAL_ERROR \\\"Set -DPTO_WSP_SOURCE_DIR=/path/to/pto-wsp\\\")\n"
               << "endif()\n"
               << "set(PTO_ISA_PATH \\\"${PTO_WSP_SOURCE_DIR}/3rdparty/pto-isa\\\" CACHE PATH \\\"Path to pto-isa\\\")\n"
               << "add_library(pto_wsp_npu_host SHARED host/runner_stub.cpp)\n"
               << "target_include_directories(pto_wsp_npu_host PRIVATE \\\"${PTO_WSP_SOURCE_DIR}/include\\\" \\\"${PTO_ISA_PATH}/include\\\")\n"
               << "if (PTO_WSP_ENABLE_ASCEND)\n"
               << "  add_library(pto_wsp_npu_aicpu SHARED aicpu/expand.cpp)\n"
               << "  target_compile_definitions(pto_wsp_npu_aicpu PRIVATE PTO_WSP_TARGET_NPU=1)\n"
               << "  target_include_directories(pto_wsp_npu_aicpu PRIVATE \\\"${PTO_WSP_SOURCE_DIR}/include\\\" \\\"${PTO_ISA_PATH}/include\\\")\n"
               << "  add_library(pto_wsp_npu_aicore OBJECT aicore/dispatch.cpp\n";
            for (const auto& kd : module.kernels) {
                cm << "    aicore/kernel_" << kd.name << ".cpp\n";
            }
            cm << "  )\n"
               << "  target_compile_definitions(pto_wsp_npu_aicore PRIVATE PTO_WSP_TARGET_NPU=1)\n"
               << "  target_include_directories(pto_wsp_npu_aicore PRIVATE \\\"${PTO_WSP_SOURCE_DIR}/include\\\" \\\"${PTO_ISA_PATH}/include\\\")\n"
               << "endif()\n";
            sources["CMakeLists.txt"] = cm.str();
        }

        auto tree = emit_sources_to_cache(sources, output_name, "ascend_npu");
        py::dict out;
        out["target"] = target;
        out["can_execute"] = false;
        out["artifact_dir"] = tree.dir;
        out["cache_key"] = tree.cache_key;
        return out;
    }

    throw std::runtime_error("compile_codegen: unsupported target: " + target);
}

}  // namespace

void bind_codegen(py::module_& m) {
    m.def(
        "codegen_compile_ir",
        &codegen_compile_ir,
        py::arg("module"),
        py::arg("output_name"),
        "v9 formal: typecheck module, then codegen+build artifacts from module-attached codegen IR.");

    m.def(
        "compile_codegen",
        &compile_codegen,
        py::arg("module"),
        py::arg("options") = pto::wsp::backend::CompileOptions{},
        "v9 formal: IR compile + backend codegen + artifact build (CPU-sim today; NPU pending).");

    struct CodegenRuntime {
        std::vector<py::array> arrays;
        std::unordered_map<std::string, int64_t> axis_sizes;
        std::unordered_map<uint64_t, uint64_t> symbols_u64;
        std::unordered_map<uint64_t, py::array> symbols_ptr;
        std::vector<uint64_t> slots_u64;

        static int64_t get_axis_size(void* ctx, const char* name) {
            const auto* self = static_cast<const CodegenRuntime*>(ctx);
            auto it = self->axis_sizes.find(std::string(name ? name : ""));
            return (it == self->axis_sizes.end()) ? 0 : it->second;
        }

        static uint64_t get_symbol_u64(void* ctx, uint64_t symbol_id) {
            const auto* self = static_cast<const CodegenRuntime*>(ctx);
            auto it = self->symbols_u64.find(symbol_id);
            return (it == self->symbols_u64.end()) ? 0ULL : it->second;
        }

        static void* get_symbol_ptr(void* ctx, uint64_t symbol_id) {
            auto* self = static_cast<CodegenRuntime*>(ctx);
            auto it = self->symbols_ptr.find(symbol_id);
            if (it == self->symbols_ptr.end()) return nullptr;
            return it->second.mutable_data();
        }

        static uint64_t get_slot_u64(void* ctx, uint32_t slot) {
            const auto* self = static_cast<const CodegenRuntime*>(ctx);
            if (slot >= self->slots_u64.size()) return 0ULL;
            return self->slots_u64[slot];
        }

        static void set_slot_u64(void* ctx, uint32_t slot, uint64_t value) {
            auto* self = static_cast<CodegenRuntime*>(ctx);
            if (slot >= self->slots_u64.size()) return;
            self->slots_u64[slot] = value;
        }

        static void* get_tensor_ptr(void* ctx, uint32_t tensor_id) {
            auto* self = static_cast<CodegenRuntime*>(ctx);
            if (tensor_id >= self->arrays.size()) return nullptr;
            return self->arrays[tensor_id].mutable_data();
        }

        static uint64_t get_tensor_stride(void* ctx, uint32_t tensor_id, uint32_t dim) {
            const auto* self = static_cast<const CodegenRuntime*>(ctx);
            if (tensor_id >= self->arrays.size()) return 0;
            const auto& a = self->arrays[tensor_id];
            const auto info = a.request();
            if (static_cast<size_t>(dim) >= info.strides.size()) return 0;
            if (info.itemsize <= 0) return 0;
            return static_cast<uint64_t>(info.strides[dim] / info.itemsize);
        }

        static KernelFn get_kernel(void* /*ctx*/, uint32_t /*kernel_id*/) { return nullptr; }
    };

    class CodegenExecutable {
    public:
        CodegenExecutable(std::string so_path, std::string entrypoint)
            : so_path_(std::move(so_path)), entrypoint_(std::move(entrypoint)) {
            handle_ = ::dlopen(so_path_.c_str(), RTLD_NOW | RTLD_LOCAL);
            if (!handle_) {
                const char* err = ::dlerror();
                throw std::runtime_error(std::string("dlopen failed: ") + (err ? err : "unknown"));
            }
            ::dlerror();  // clear
            fn_ = reinterpret_cast<WorkloadFn>(::dlsym(handle_, entrypoint_.c_str()));
            const char* err = ::dlerror();
            if (err || !fn_) {
                throw std::runtime_error(std::string("dlsym failed for entrypoint '") + entrypoint_ + "': " +
                                         (err ? err : "unknown"));
            }
        }

        ~CodegenExecutable() {
            if (handle_) {
                ::dlclose(handle_);
                handle_ = nullptr;
            }
        }

        CodegenExecutable(const CodegenExecutable&) = delete;
        CodegenExecutable& operator=(const CodegenExecutable&) = delete;

        uint64_t run(const std::vector<py::array>& arrays, const std::map<std::string, int64_t>& axis_sizes) {
            return run_with_symbols(arrays, axis_sizes, {}, {});
        }

        uint64_t run_with_symbols(const std::vector<py::array>& arrays,
                                  const std::map<std::string, int64_t>& axis_sizes,
                                  const std::map<uint64_t, uint64_t>& symbols_u64,
                                  const std::map<uint64_t, py::array>& symbols_ptr) {
            CodegenRuntime rt;
            rt.arrays = arrays;
            rt.axis_sizes.insert(axis_sizes.begin(), axis_sizes.end());
            rt.symbols_u64.insert(symbols_u64.begin(), symbols_u64.end());
            rt.symbols_ptr.insert(symbols_ptr.begin(), symbols_ptr.end());
            rt.slots_u64.assign(256, 0ULL);

            RuntimeContext ctx;
            ctx.get_axis_size = &CodegenRuntime::get_axis_size;
            ctx.get_symbol_u64 = &CodegenRuntime::get_symbol_u64;
            ctx.get_symbol_ptr = &CodegenRuntime::get_symbol_ptr;
            ctx.get_slot_u64 = &CodegenRuntime::get_slot_u64;
            ctx.set_slot_u64 = &CodegenRuntime::set_slot_u64;
            ctx.get_tensor_ptr = &CodegenRuntime::get_tensor_ptr;
            ctx.get_tensor_stride = &CodegenRuntime::get_tensor_stride;
            ctx.get_kernel = &CodegenRuntime::get_kernel;
            ctx.ctx = &rt;

            // v9 timing: CPU-sim cycles are returned directly by generated code via PTO-ISA.
            return fn_(&ctx, nullptr);
        }

        const std::string& so_path() const { return so_path_; }
        const std::string& entrypoint() const { return entrypoint_; }

    private:
        std::string so_path_;
        std::string entrypoint_;
        void* handle_ = nullptr;
        WorkloadFn fn_ = nullptr;
    };

    py::class_<CodegenExecutable>(m, "CodegenExecutable")
        .def(py::init<std::string, std::string>(), py::arg("so_path"), py::arg("entrypoint"))
        .def("run", &CodegenExecutable::run, py::arg("arrays"), py::arg("axis_sizes"))
        .def(
            "run_with_symbols",
            &CodegenExecutable::run_with_symbols,
            py::arg("arrays"),
            py::arg("axis_sizes"),
            py::arg("symbols_u64"),
            py::arg("symbols_ptr"))
        .def_property_readonly("so_path", &CodegenExecutable::so_path)
        .def_property_readonly("entrypoint", &CodegenExecutable::entrypoint);
}
