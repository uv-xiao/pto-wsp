// Copyright 2026 PTO-RT Authors
// SPDX-License-Identifier: MIT

#include "pto/rt/codegen/cmake_compiler.hpp"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <vector>
#include <cstdio>

namespace fs = std::filesystem;

namespace pto::wsp::codegen {
namespace {

// Defined by the top-level CMake build for pto_ir_cpp.
#ifndef PTO_WSP_SOURCE_DIR
#define PTO_WSP_SOURCE_DIR ""
#endif

#ifndef PTO_ISA_PATH
#define PTO_ISA_PATH ""
#endif

constexpr std::string_view kCacheVersion = "v5-cpp";

bool executable_in_path(const std::string& exe) {
    const char* path = std::getenv("PATH");
    if (!path || !*path) return false;
    std::string_view paths(path);
    while (!paths.empty()) {
        const auto pos = paths.find(':');
        std::string_view dir = (pos == std::string_view::npos) ? paths : paths.substr(0, pos);
        if (!dir.empty()) {
            fs::path p = fs::path(std::string(dir)) / exe;
            std::error_code ec;
            if (fs::exists(p, ec) && !fs::is_directory(p, ec)) return true;
        }
        if (pos == std::string_view::npos) break;
        paths.remove_prefix(pos + 1);
    }
    return false;
}

std::string pick_cxx() {
    if (const char* p = std::getenv("PTO_WSP_CODEGEN_CXX"); p && *p) {
        return p;
    }
    // PTO-ISA CPU-sim tooling expects clang++>=15 or g++>=14. Many environments
    // still ship older g++, so prefer clang++ when available.
    if (executable_in_path("clang++")) return "clang++";
    if (executable_in_path("g++")) return "g++";
    return "c++";
}

uint64_t fnv1a64_update(uint64_t h, std::string_view data) {
    constexpr uint64_t kPrime = 1099511628211ULL;
    for (unsigned char c : data) {
        h ^= static_cast<uint64_t>(c);
        h *= kPrime;
    }
    return h;
}

uint64_t fnv1a64_update(uint64_t h, const std::string& s) {
    return fnv1a64_update(h, std::string_view{s});
}

uint64_t fnv1a64_update_file(uint64_t h, const fs::path& p) {
    std::ifstream f(p, std::ios::binary);
    if (!f) return fnv1a64_update(h, p.string());
    std::ostringstream oss;
    oss << f.rdbuf();
    h = fnv1a64_update(h, p.string());
    h = fnv1a64_update(h, oss.str());
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

fs::path cache_dir() {
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
    std::ofstream out(p, std::ios::binary);
    if (!out) {
        throw std::runtime_error("Failed to write file: " + p.string());
    }
    out.write(s.data(), static_cast<std::streamsize>(s.size()));
}

int run_cmd(const std::string& cmd) {
    // NOTE: For v9 we prioritize correctness + reproducibility over fancy log
    // capture. Failures report the failing command.
    return std::system(cmd.c_str());
}

std::string shell_quote(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 2);
    out.push_back('"');
    for (char c : s) {
        if (c == '"') out.push_back('\\');
        out.push_back(c);
    }
    out.push_back('"');
    return out;
}

std::string capture_cmd(const std::string& cmd) {
    // Best-effort; used only for cache invalidation.
    std::string out;
    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) return out;
    char buf[256];
    while (true) {
        size_t n = fread(buf, 1, sizeof(buf), pipe);
        if (n == 0) break;
        out.append(buf, buf + n);
    }
    (void)pclose(pipe);
    return out;
}

uint64_t hash_inputs(const std::map<std::string, std::string>& sources,
                     const fs::path& include_dir,
                     const fs::path& pto_isa_include_dir,
                     const std::string& cxx) {
    uint64_t h = 1469598103934665603ULL;
    h = fnv1a64_update(h, std::string(kCacheVersion));
    h = fnv1a64_update(h, cxx);
    h = fnv1a64_update(h, capture_cmd(cxx + " --version 2>/dev/null"));
    h = fnv1a64_update(h, capture_cmd("cmake --version 2>/dev/null"));
    h = fnv1a64_update(h, include_dir.string());
    h = fnv1a64_update(h, pto_isa_include_dir.string());

    // Invalidate when ABI headers change (generated code includes them by name).
    const fs::path abi_dir = include_dir / "pto" / "rt" / "codegen" / "abi";
    h = fnv1a64_update_file(h, abi_dir / "kernel_abi.hpp");
    h = fnv1a64_update_file(h, abi_dir / "workload_abi.hpp");
    h = fnv1a64_update_file(h, abi_dir / "ptoisa_bridge.hpp");

    // Invalidate when PTO-ISA umbrella header changes.
    h = fnv1a64_update_file(h, pto_isa_include_dir / "pto" / "pto-inst.hpp");

    for (const auto& [name, content] : sources) {
        h = fnv1a64_update(h, name);
        h = fnv1a64_update(h, content);
    }
    return h;
}

}  // namespace

CompileResult compile_sources_via_cmake(
    const std::map<std::string, std::string>& sources,
    const std::string& output_name) {
    if (sources.empty()) {
        throw std::runtime_error("compile_sources_via_cmake: no sources provided");
    }

    const fs::path root = fs::path(PTO_WSP_SOURCE_DIR);
    if (root.empty() || !fs::exists(root)) {
        throw std::runtime_error("PTO_WSP_SOURCE_DIR is not set or not found; build pto_ir_cpp via CMake");
    }

    const fs::path include_dir = root / "include";
    fs::path pto_isa_root;
    if (fs::path(PTO_ISA_PATH).empty()) {
        pto_isa_root = root / "3rdparty" / "pto-isa";
    } else {
        fs::path p = fs::path(PTO_ISA_PATH);
        if (p.is_relative()) {
            pto_isa_root = root / p;
        } else {
            pto_isa_root = p;
        }
    }
    const fs::path pto_isa_include_dir = pto_isa_root / "include";

    const fs::path out_cache = cache_dir();
    fs::create_directories(out_cache);

    const std::string cxx = pick_cxx();
    const std::string key = hex16(hash_inputs(sources, include_dir, pto_isa_include_dir, cxx));
    const fs::path so_path = out_cache / (output_name + "_" + key + ".so");
    if (fs::exists(so_path)) {
        return CompileResult{so_path.string(), key};
    }

    const fs::path src_dir = out_cache / ("src_" + output_name + "_" + key);
    const fs::path build_dir = out_cache / ("build_" + output_name + "_" + key);
    fs::create_directories(src_dir);
    fs::create_directories(build_dir);

    std::vector<std::string> cpp_files;
    cpp_files.reserve(sources.size());
    for (const auto& [name, content] : sources) {
        const fs::path p = src_dir / name;
        write_text(p, content);
        cpp_files.push_back(p.filename().string());
    }

    std::ostringstream cmake;
    cmake << "cmake_minimum_required(VERSION 3.16)\n"
          << "project(pto_wsp_codegen_" << output_name << " LANGUAGES CXX)\n"
          << "set(CMAKE_CXX_STANDARD 23)\n"
          << "set(CMAKE_CXX_STANDARD_REQUIRED ON)\n"
          << "set(CMAKE_POSITION_INDEPENDENT_CODE ON)\n"
          << "add_library(pto_wsp_codegen SHARED\n";
    for (const auto& f : cpp_files) {
        cmake << "  " << f << "\n";
    }
    cmake << ")\n"
          << "target_compile_definitions(pto_wsp_codegen PRIVATE __CPU_SIM=1)\n"
          << "target_compile_options(pto_wsp_codegen PRIVATE -O3 -Wno-deprecated-declarations)\n"
          << "if (CMAKE_CXX_COMPILER_ID STREQUAL \"Clang\")\n"
          << "  target_compile_options(pto_wsp_codegen PRIVATE -Wno-pass-failed)\n"
          << "endif()\n"
          << "target_include_directories(pto_wsp_codegen PRIVATE \"" << include_dir.string() << "\")\n"
          << "target_include_directories(pto_wsp_codegen PRIVATE \"" << pto_isa_include_dir.string() << "\")\n"
          << "set_target_properties(pto_wsp_codegen PROPERTIES OUTPUT_NAME \"" << so_path.stem().string() << "\")\n"
          << "set_target_properties(pto_wsp_codegen PROPERTIES PREFIX \"\")\n"
          << "set_target_properties(pto_wsp_codegen PROPERTIES LIBRARY_OUTPUT_DIRECTORY \"" << out_cache.string() << "\")\n";

    write_text(src_dir / "CMakeLists.txt", cmake.str());

    const std::string cfg_cmd =
        "cmake -S " + shell_quote(src_dir.string()) + " -B " + shell_quote(build_dir.string()) +
        " -DCMAKE_CXX_COMPILER=" + shell_quote(cxx) +
        " > " + shell_quote((build_dir / "configure.log").string()) + " 2>&1";
    if (run_cmd(cfg_cmd) != 0) {
        throw std::runtime_error(
            "CMake configure failed. See " + (build_dir / "configure.log").string());
    }

    const std::string build_cmd =
        "cmake --build " + shell_quote(build_dir.string()) + " --target pto_wsp_codegen -j" +
        " > " + shell_quote((build_dir / "build.log").string()) + " 2>&1";
    if (run_cmd(build_cmd) != 0) {
        throw std::runtime_error(
            "CMake build failed. See " + (build_dir / "build.log").string());
    }

    if (!fs::exists(so_path)) {
        throw std::runtime_error("CMake build completed but output .so not found: " + so_path.string());
    }

    return CompileResult{so_path.string(), key};
}

}  // namespace pto::wsp::codegen
