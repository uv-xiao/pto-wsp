// Copyright 2026 PTO-WSP Authors
// SPDX-License-Identifier: MIT

#pragma once

#include "pto/wsp/backend/backend.hpp"
#include "pto/wsp/ir/ir.hpp"

#include <map>
#include <string>

namespace pto::wsp::codegen::pto_runtime {

// Emit a pto-runtime-compatible host_build_graph source tree (Phase 1 scaffold).
//
// Returned map keys are relative file paths inside the artifact directory.
std::map<std::string, std::string> emit_host_build_graph_sources(
    const ir::Module& module,
    const backend::CompileOptions& options);

}  // namespace pto::wsp::codegen::pto_runtime

