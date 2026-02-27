# Feature: Passes and Pipelines

## Goal

Make transformation pipelines:

- explicit
- inspectable
- extensible
- safe to compose

## Pass types

- Canonicalization passes (AST normalization)
- Typing passes (layout/effect checks and normalization)
- Scheduling passes (apply schedule directives)
- Lowering passes (to backend-ready forms)
- Packaging passes (artifact emission, manifest finalization)
- Island passes (enter/exit MLIR or external toolchains)

## Contract-driven execution

Each pass declares:

- `requires` capabilities
- `provides` capabilities
- invariants and failure diagnostics

## Pipeline definition

A pipeline declares:

- target backend
- ordered pass list + parameters
- output artifact contract + binding requirement

Pipeline selection checks satisfiability before running.

