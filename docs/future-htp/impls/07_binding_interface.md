# Impl: Binding Interface

## Goal

Standardize `bind(package)` so users get a consistent experience across backends.

## Binding lifecycle

1. validate package + manifest
2. build (optional) using toolchain integration
3. load (library/runtime/simulator)
4. run entrypoints with typed marshalling
5. trace/report

## Minimal API (illustrative)

- `binding = bind(package_path)`
- `binding.build()` (optional)
- `binding.run(entry="...", args=..., symbols=...)`
- `binding.trace(level=...)`

