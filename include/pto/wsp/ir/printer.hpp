// PTO Workload-Schedule Programming (PTO-WSP) framework v9 - IR Printer
// Copyright (c) 2024 PTO Project
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ir.hpp"
#include <fstream>
#include <sstream>

namespace pto::wsp::ir {

// Printer for serializing IR to .pto assembly format
class Printer {
public:
    // Print module to stream
    static void print(const Module& module, std::ostream& os) {
        module.print(os);
    }

    // Print to string
    static std::string to_string(const Module& module) {
        std::ostringstream oss;
        print(module, oss);
        return oss.str();
    }

    // Print to file
    static bool to_file(const Module& module, const std::string& path) {
        std::ofstream ofs(path);
        if (!ofs) return false;
        print(module, ofs);
        return ofs.good();
    }

    // Print NPU module to stream
    static void print(const NPUModule& module, std::ostream& os) {
        module.print(os);
    }

    // Print NPU module to string
    static std::string to_string(const NPUModule& module) {
        std::ostringstream oss;
        print(module, oss);
        return oss.str();
    }

    // Print NPU module to file
    static bool to_file(const NPUModule& module, const std::string& path) {
        std::ofstream ofs(path);
        if (!ofs) return false;
        print(module, ofs);
        return ofs.good();
    }

    // Print a single IR node
    static void print(const IRPtr<IRNode>& node, std::ostream& os, int indent = 0) {
        node->print(os, indent);
    }

    static std::string to_string(const IRPtr<IRNode>& node) {
        std::ostringstream oss;
        print(node, oss);
        return oss.str();
    }
};

}  // namespace pto::wsp::ir
