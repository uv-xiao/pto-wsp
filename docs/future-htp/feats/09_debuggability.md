# Feature: Debuggability & Introspection

## Goal

Make compiler and runtime behavior observable and explainable.

## Required diagnostics

- capability mismatch diagnostics (missing dialect/intrinsic/pass)
- layout incompatibility diagnostics (where, expected/got, suggested relayout)
- stream protocol/effect diagnostics (mismatched put/get, deadlock cycles)
- backend handler diagnostics (missing lowering/emitter)

## Required artifacts

- pass trace (`pass_trace.jsonl`)
- AST dumps pre/post canonicalization
- backend-specific codegen outputs
- manifest with full pipeline info and versions

