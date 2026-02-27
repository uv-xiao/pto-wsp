# Feature: WSP (Workload-Schedule Programming) Dialect

## Goal

Support an authoring workflow where:

- the **workload** is a pure, declarative description of computation and logical parallelism, and
- the **schedule** provides backend-specific constraints (tiling, mapping, buffering, pipelining, fusion),
  without changing workload semantics.

## Requirements

- Workload must be representable as AST + metadata without executing imperative control flow at compile time.
- Schedule must be composable and overridable.
- Schedule directives must be checkable against backend constraints (memory spaces, vector widths, etc.).

## Recommended surface concepts

- `@workload` for workload definitions.
- `@schedule(workload)` for schedule definitions.
- `Schedule` object with combinators:
  - mapping: `tile`, `split`, `reorder`, `bind(level=...)`
  - locality: `buffer(name, space=...)`, `prefetch`, `async_copy`
  - fusion: `fuse(tasks)`, `inline(kernel)`
  - pipelining: `pipeline(depth=...)`, `stages(...)`
  - resources: `num_warps`, `threads`, `cluster`, backend-specific knobs

## Lowering strategy

- The workload is canonicalized into:
  - explicit task graph (or explicit nested parallel loops with task calls)
- The schedule attaches metadata and triggers:
  - task partitioning decisions
  - fusion decisions
  - memory placement decisions

Backends interpret schedule metadata differently, but the schedule must be validated against backend capability contracts.

