# Feature: Extensibility Model (capability-typed composition)

## Goal

Enable third parties to extend HTP across *all* dimensions while keeping compositions correct and diagnosable.

## Core mechanism: `requires` / `provides`

Every extension unit declares:

- `requires`: a set of capabilities (types) that must hold before it is used.
- `provides`: a set of capabilities established after it is applied.

Extension units include:

- dialect packages
- intrinsic sets
- passes
- pipelines
- backends
- bindings

Capabilities are **first-class data**, not just documentation: they are checked during pipeline construction and recorded
in the emitted artifact manifest for auditability.

## Capability examples

- Dialects:
  - `Dialect.WSPEnabled`
  - `Dialect.CSPEnabled`
- IR states:
  - `IR.ASTCanonical`
  - `IR.CSPGraphFormed`
  - `IR.ScheduleAttached`
- Typing invariants:
  - `Type.LayoutNormalized`
  - `Type.StreamEffectsChecked`
  - `Type.NoPendingCollectives`
- Backend readiness:
  - `Codegen.ReadyForPTO`
  - `Codegen.ReadyForAIE`

Parameterized capabilities are allowed (recommended) to avoid untyped string soup:

- `Backend.PTO(variant="a2a3sim")`
- `Backend.AIE(family="XDNA2")`
- `Layout.FacetSupported("dist")`, `Layout.FacetSupported("mem")`
- `CSP.Semantics(kind="bounded_fifo")`

## Pipeline selection

Given:

- program capabilities inferred from the AST + enabled dialects/intrinsics
- target backend requirements

Choose a pipeline template and solve:

- is every pass `requires` satisfied by current capabilities?
- after applying pass `provides`, can we reach the pipeline’s declared output contract?

If multiple pipelines satisfy constraints, selection policy can be:

- explicit (user chooses)
- heuristic (prefer fewer passes, or prefer “stable” pipeline)
- search-based (future extension)

## Extension composition is a graph, not a list

The crucial “HTP-ness” is that extensions are related:

- a dialect introduces new AST constructs,
- a canonicalization pass rewrites them,
- a typing pass introduces layout/effect annotations,
- intrinsics require certain annotation invariants,
- the backend requires certain lowered forms and handlers,
- the binding requires a package contract.

The `requires/provides` system turns this dependency graph into:

- a satisfiable pipeline construction problem, and
- a diagnosable failure mode when constraints are unmet.

## Diagnostics

On failure, emit:

- missing capability list,
- where it could be provided (dialect or pass suggestions),
- which intrinsic/backend handler is missing.

---

## Reference alignment (why these choices)

This capability-typed design is chosen to keep HTP’s extensibility coherent across:

- typed stream protocols and boundedness reasoning (Dato-style) — see `docs/reference/16_dato.md`
- explicit hardware abstraction (memory spaces + hierarchy) — see `docs/reference/19_arknife.md`
- unified layout notion across layers — see `docs/reference/20_axe.md`
- artifact-first packaging and manifests — see `docs/reference/18_pypto.md`
