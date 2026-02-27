# Impl: IR Model (AST-first + typed metadata)

## Goal

Define the minimal internal representation strategy that preserves Python extensibility.

## IR layers

1. **Source AST**: parsed Python AST of entrypoints.
2. **Canonical AST**: normalized AST for deterministic passes.
3. **Typed AST**: canonical AST + attached metadata:
   - symbol table + type info
   - layout facets
   - schedule directives
   - effect annotations
4. **Backend-ready forms**:
   - for PTO: codegen-ready kernel/task representation
   - for AIE: MLIR-AIE module(s) as an island product

## Metadata requirements

- serializable for artifact dumps
- stable node identity scheme (e.g., symbol path + structural hashes)
- no reliance on Python object identity across processes

