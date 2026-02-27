# Feature: Backends and Artifact Contracts

## Goal

Treat backends as first-class and keep integration stable via artifact contracts.

## Backend definition must include

- hardware profile(s)
- supported dialects and intrinsic handlers
- supported layout facets and legality rules
- codegen packaging contract
- binding interface expectations

## Artifact contract principles

- Single compilation → single package directory
- Stable manifest schema
- Separation of:
  - generated code/artifacts
  - intermediate dumps
  - build/run metadata

## “Must-support” initial backends

- Ascend PTO toolchain/runtime (simulation + device)
- AIE via MLIR-AIE

