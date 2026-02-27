# Reference Notes: Axe (arXiv:2601.19092) — unified layout abstraction

Source (gitignored):
- Paper PDF: `references/arxiv/2601.19092.pdf`
- Paper source: `references/arxiv/2601.19092/`

This note extracts the parts of **Axe** that are most relevant to PTO‑WSP v10’s multi-backend story, especially:

- making “layout” a first-class cross-layer concept (distributed ↔ on-device),
- building a compiler/DSL that can target heterogeneous backends without rewriting the whole system.

## 1) What Axe is (high-level)

Axe proposes **Axe Layout**, a hardware-aware abstraction that maps logical tensor coordinates to a **multi-axis physical
space** via **named axes**. It aims to unify:

- on-device tiling (threads/memory hierarchy),
- distributed sharding/replication (device meshes),
- offsets and more complex placement constraints,

under one representation.

Pointers:
- `references/arxiv/2601.19092/sections/abstract.tex`
- `references/arxiv/2601.19092/sections/layout.tex`

## 2) Key representation: named axes + (D, R, O)

Axe layout extends the standard shape–stride model by allowing strides to be bound to **named axes** representing hardware
resources (threads, memory banks, devices).

It decomposes a layout into:

- **D (shard)**: an ordered list of “iters” (extent, stride, axis) that shard the logical index into physical coordinates
- **R (replica)**: a set of iters that replicate/broadcast (set-valued mapping)
- **O (offset)**: constant offsets on axes

Pointer:
- `references/arxiv/2601.19092/sections/layout.tex`

## 3) Why this matters for PTO‑WSP v10

v10 needs to scale beyond “one backend” and beyond “one level of layout”:

- Ascend (heterogeneous NPU) and AIE (spatial/dataflow) have different memory/compute/sync models.
- v10 must keep a stable workload–schedule surface while still enabling backend-specific low-level choices.

Axe’s core lesson for v10:
- treat layout/sharding/tiling as a **unified abstraction across layers** (device mesh ↔ device hierarchy), which can feed
  both compilation (lowering decisions) and runtime (dispatch/cycle accounting constraints).

This aligns with v10’s design goals:
- explicit `ArchModel` layer,
- backend registry + capability matrices,
- optional multi-target layer (one program → multiple backend artifacts).

