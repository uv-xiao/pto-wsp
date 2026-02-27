# Impl: Capability Solver (pipeline satisfiability)

## Goal

Turn extension compatibility into a checkable constraint problem.

## Inputs

- program capability set (from dialects, intrinsics, AST analysis)
- target backend requirements
- pipeline template:
  - ordered pass list, each with `requires/provides`
  - output artifact contract requirements

## Outputs

- satisfiable pipeline instance (pass parameters bound)
- or a structured failure report:
  - missing capabilities
  - candidate providers (dialects/passes)
  - handler gaps (intrinsic without backend lowering)

## Design note

Start with a simple forward-checking solver; later extensions may add:

- alternative pass choices (OR nodes)
- search-based optimization of pipeline

