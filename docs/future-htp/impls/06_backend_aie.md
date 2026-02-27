# Impl: AIE Backend (MLIR-AIE island)

## Goal

Compile spatial/dataflow programs into MLIR-AIE artifacts with explicit:

- tile mapping
- FIFO definitions
- host glue for invocation

## Recommended outputs

- `aie.mlir` as primary IR artifact
- structured mapping + FIFO JSON for inspection
- optional build scripts (or CMake) pinned to toolchain versions

## Layout interaction

- distribution facet drives sharding across the virtual grid
- hardware facet constrains which buffers live in which memories
- memory facet influences packing and DMA patterns

