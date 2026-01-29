// PTO Workload-Schedule Programming (PTO-WSP) framework v9 - Codegen data attached to IR
// Copyright (c) 2026 PTO Project
// SPDX-License-Identifier: MIT

#pragma once

#include "pto/rt/ir/core.hpp"
#include "pto/rt/ir/ext.hpp"  // AttrMap/AttrValue for op attrs

#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace pto::wsp::ir {

// =============================================================================
// Workload-side bindings for codegen
// =============================================================================

struct CodegenIndexExpr {
    bool is_var = false;
    std::string var;
    int64_t constant = 0;
};

struct CodegenAxisArg {
    bool is_var = false;
    std::string var;
    uint64_t u64 = 0;
};

struct CodegenTensorArg {
    std::string param;     // kernel param name
    uint32_t tensor_id = 0;
    int base_rank = 0;
    int view_rank = 0;
    std::vector<CodegenIndexExpr> index_exprs;
};

struct CodegenTensorInfo {
    DType dtype = DType::F32;
    std::vector<int64_t> shape;
};

// =============================================================================
// Kernel-side IR for codegen (data-oriented, backend-owned emission)
// =============================================================================

struct CodegenKernelValueInfo {
    DType dtype = DType::F32;
    bool has_shape = false;
    int rows = 0;
    int cols = 0;
};

struct CodegenKernelParamInfo {
    std::string name;
    int id = 0;
};

struct CodegenKernelOpInfo {
    std::string kind;
    bool has_result = false;
    int result = 0;
    std::vector<int> operands;
    AttrMap attrs;
};

struct CodegenKernelDef {
    std::string name;
    std::vector<CodegenKernelParamInfo> params;
    std::map<int, CodegenKernelValueInfo> values;
    std::vector<CodegenKernelOpInfo> ops;
    AttrMap attrs;
};

}  // namespace pto::wsp::ir
