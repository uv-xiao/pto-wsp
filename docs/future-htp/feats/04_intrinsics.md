# Feature: Intrinsics (typed primitives with backend handlers)

## Goal

Make “backend-specific operations” usable from Python while keeping portability and correctness.

## Intrinsic contract

Every intrinsic should declare:

- semantic signature:
  - operand types (tiles/tensors/scalars)
  - result types
- layout constraints:
  - required distribution/memory/hardware facets
  - permitted memory spaces
- scheduling constraints:
  - vector width, alignment, pipeline stage constraints, etc.
- backend handlers:
  - which backends can lower/emit it

## Examples

- `pto.add(tileA, tileB)`:
  - requires: PTO backend handler
  - layout: compatible tile shapes + dtype
  - hardware: specific memory space constraints may apply
- `aie.matmul(tileA, tileB)`:
  - requires: AIE handler and mapping context

## Portability strategy

Support two classes:

1. **Portable intrinsics**: abstract ops that multiple backends can lower (e.g., `add`, `mul`, `dot`).
2. **Backend intrinsics**: explicitly namespaced ops (`pto.*`, `aie.*`) used when authors want control.

The type system prevents accidental use of backend intrinsics in unsupported pipelines.

