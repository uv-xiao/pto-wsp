# Reference Notes: TL / Loom (arXiv:2512.22168) — tile languages → spatial dataflow

Source (gitignored):
- Paper PDF: `references/arxiv/2512.22168.pdf`
- Paper source: `references/arxiv/2512.22168/`

This note captures the parts of **TL / Loom** (“Automatic End-to-End Compiler of Tile-Based Languages for Spatial Dataflow
Architectures”) that are most relevant to PTO‑WSP v10’s **AIE / spatial backend** direction.

## 1) Problem framing (relevant to v10)

Spatial dataflow accelerators expose explicit cores, on-chip networks, and distributed memories. Performance depends on:

- mapping tiles to cores,
- scheduling tile execution in time,
- orchestrating explicit communication/reuse across the on-chip network.

This is a different world from GPU “hardware schedules blocks + caches capture reuse” assumptions.

Pointer:
- `references/arxiv/2512.22168/introduction.tex`

## 2) Key design lessons for PTO‑WSP v10

### 2.1 Explicit hardware representation matters

TL introduces a hardware representation that captures:

- interconnect topology,
- memory hierarchy,
- compute capabilities,

so mapping/scheduling decisions can be reasoned about (and optimized) without hard-coding for one chip.

### 2.2 The compiler must own placement + scheduling across tiles

For spatial backends (AIE-like), “tile program optimization” is not enough; the compiler must also:

- place tile instances across cores,
- schedule them to exploit locality/reuse,
- manage communication patterns explicitly.

This aligns strongly with PTO‑WSP’s workload–schedule philosophy: the schedule is part of the program, not “runtime magic”.

## 3) MLIR as an internal backbone (pragmatic takeaway)

TL is built on MLIR, suggesting a pragmatic design stance for v10:

- keep PTO‑WSP’s Python-driven pipeline (AST passes) as the primary authoring/transform layer,
- keep an internal option to route selected regions through MLIR when stronger IR tooling is beneficial,
- use MLIR as an internal implementation detail rather than as a forced rewrite.

This matches v10’s “AST pass dynamically selects MLIR islands” direction.

