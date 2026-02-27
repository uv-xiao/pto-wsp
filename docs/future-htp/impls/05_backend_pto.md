# Impl: PTO / Ascend Backend Packaging

## Goal

Emit artifacts consumable by PTO runtime/toolchain for:

- simulation (a2a3sim)
- device execution (a2a3)

## Recommended package structure

- kernels vs orchestration separation
- stable kernel registry (IDs, shapes, dtype, required resources)
- build metadata for toolchain invocation

## Manifest extensions (examples)

- required PTO ISA version / toolchain version
- kernel function IDs and entrypoints
- runtime configuration knobs (streams, buffers, async copy support)

