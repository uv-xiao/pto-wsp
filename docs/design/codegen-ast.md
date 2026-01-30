# C++ Codegen AST Design (v9)

> **Goal:** Provide a clean, type-safe C++ API for emitting C++ source in codegen backends.
> **Scope:** C++ implementation only (Python emitters are not part of the v9 “formal” criteria).

---

## Why an AST?

Codegen emitters that concatenate strings tend to accumulate:
- mismatched braces/indentation,
- accidental precedence bugs (missing parentheses),
- hard-to-review diffs when refactoring structure.

The v9 direction is to generate code via a small AST + emitter, with localized
escape hatches for bring-up.

---

## Implementation Targets (in-repo)

- `include/pto/wsp/codegen/cpp_ast.hpp` — AST node types (expressions/statements/functions)
- `include/pto/wsp/codegen/cpp_builder.hpp` — small fluent helpers to construct the AST
- `src/pto/wsp/codegen/cpp_emitter.cpp` — pretty-printer / emitter

These are the primitives intended for backend-owned emitters (CPU-sim and NPU).

---

## Current API Shape

### Handles (value-semantic)

- `Expr` and `Stmt` are *small shared handles* (`std::shared_ptr` internally).
  This avoids recursive value types while keeping call sites concise.

### Escape hatches

- `RawExpr` / `RawStmt` allow incremental bring-up when a construct is not yet modeled.
  v9 target: keep raw strings localized (no whole-function string concatenation).

### Supported nodes (today)

- `Expr`: raw/ident/literal/call/member/index/unary/binary/cast
- `Stmt`: raw/expr/return/vardecl/block/if/for
- `Decl`: `Function`, `TranslationUnit`

---

## Example

```cpp
#include "pto/wsp/codegen/cpp_ast.hpp"
#include "pto/wsp/codegen/cpp_builder.hpp"

using namespace pto::wsp::codegen::cpp;

TranslationUnit tu;
tu.includes = {"<cstdint>"};

FunctionBuilder fb("int", "add1");
fb.extern_c(true)
  .param("int", "x")
  .stmt(vardecl("int", "y", binary(ident("x"), "+", lit("1"))))
  .stmt(ret(ident("y")));
tu.functions.push_back(std::move(fb).build());

std::string code = emit_cpp(tu);
```

---

## Follow-ups (planned)

- Add structured nodes for `switch`, `while`, initializer lists, and `std::span` helpers.
- Add scope utilities so emitters can avoid brace raw statements entirely.
- Migrate backend/codegen emitters to build via this AST (eliminate ad-hoc concatenation).

