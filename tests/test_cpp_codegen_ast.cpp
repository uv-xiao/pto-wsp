// PTO-RT v9 - C++ codegen AST smoke test

#include "pto/rt/codegen/cpp_ast.hpp"
#include "pto/rt/codegen/cpp_builder.hpp"

#include <cassert>
#include <iostream>

using namespace pto::wsp::codegen::cpp;

int main() {
    TranslationUnit tu;
    tu.includes = {"<cstdint>"};

    FunctionBuilder fb("int", "add1");
    fb.extern_c(true)
      .param("int", "x")
      .stmt(vardecl("int", "y", binary(ident("x"), "+", lit("1"))))
      .stmt(ret(ident("y")));
    tu.functions.push_back(std::move(fb).build());

    const std::string out = emit_cpp(tu);
    std::cout << out << "\n";

    // Minimal assertions (avoid brittle full string match).
    assert(out.find("#include <cstdint>") != std::string::npos);
    assert(out.find("extern \"C\" int add1(int x)") != std::string::npos);
    assert(out.find("int y = (x + 1);") != std::string::npos);
    assert(out.find("return y;") != std::string::npos);

    return 0;
}
