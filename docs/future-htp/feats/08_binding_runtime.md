# Feature: Binding & Runtime Interface

## Goal

Provide a consistent “compile → bind → run” workflow while allowing backend-specific build/run.

## Binding responsibilities

- validate package against backend contract
- optionally build artifacts (invoke toolchain) into executable form
- load artifacts (dlopen, runtime API, simulator)
- run entrypoints with typed argument marshalling
- provide tracing hooks

## Runtime separation

HTP should not embed device runtimes; it integrates with:

- PTO runtime/toolchain for Ascend
- MLIR-AIE build/run tooling for AIE
- future backends via binding plugins

