// Copyright 2026 PTO-RT Authors
// SPDX-License-Identifier: MIT

#pragma once

#include <map>
#include <string>

namespace pto::wsp::codegen {

struct CompileResult {
    std::string so_path;
    std::string cache_key;
};

// Build a shared library from in-memory C++ sources via CMake.
//
// This is the v9 “formal” integration point: toolchain invocation and caching
// happen in C++ (not Python). Source emission may be migrated to C++ backends
// incrementally; this compiler is the shared artifact builder.
CompileResult compile_sources_via_cmake(
    const std::map<std::string, std::string>& sources,
    const std::string& output_name);

}  // namespace pto::wsp::codegen
