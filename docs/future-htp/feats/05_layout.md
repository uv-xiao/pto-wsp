# Feature: Unified Layout (distribution + memory + hardware facets)

## Goal

Make layout first-class so it can drive:

- compiler lowering (tiling, vectorization, placement)
- runtime dispatch (sharding/replication semantics)
- backend codegen constraints (DMA descriptors, FIFO packing, memory banking)

## Facets

### 1) Distribution facet (Dato/Axe-like)

- Per-dimension elements:
  - `R` replicate
  - `S(i)` shard along mesh axis `i`
- Operations:
  - join/compatibility rules
  - explicit relayout
  - collectives as effects (e.g., pending reductions)

### 2) Memory facet (Triton/CuTe-like)

- `strides`, `order`, `swizzle`, `pack`
- composable transforms: `permute`, `tile`, `reshape`

### 3) Hardware facet (Arknife-like)

- hardware hierarchy levels and memory spaces
- constraints like:
  - “tile must reside in UB”
  - “this access must be bank-conflict-free”

## Type checking

HTP should implement:

- layout compatibility checks at kernel-call boundaries
- relayout insertion or explicit user relayout
- backend legality checks (facet subsets supported by backend)

### Distribution join (minimal rule set)

Given two distribution layouts with per-dimension elements, a conservative join is:

- `R ⊔ R = R`
- `S(i) ⊔ S(i) = S(i)`
- `R ⊔ S(i) = S(i)` (shard is “more specific” than replicate)
- `S(i) ⊔ S(j)` for `i != j` is incompatible unless an explicit relayout/transpose is introduced

This “join” is enough to:

- detect obvious incompatibilities early, and
- force explicit relayout points (which is essential for backend correctness).

More advanced rules (normalization/collapse, affine composition) can be added later.

### Collective effects

Operations such as reduce across a sharded dimension should produce a pending effect (e.g., “needs allreduce”) that must be
discharged explicitly by a collective primitive. This keeps cross-device semantics honest and prevents “silent wrongness”.

## References

- Layout types + join motivation: `docs/reference/16_dato.md`
- Cross-layer named-axis layout abstraction: `docs/reference/20_axe.md`
- Hardware hierarchy + memory spaces: `docs/reference/19_arknife.md`
