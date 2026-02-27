# Reference: pto-runtime — Analysis Notes (for PTO‑WSP v10)

This document summarizes the **current pto-runtime architecture** (as of `references/pto-runtime/`) and highlights the
parts that matter for PTO‑WSP v10.

Primary sources to read alongside this note:
- `references/pto-runtime/README.md`
- `references/pto-runtime/python/runtime_builder.py`
- `references/pto-runtime/python/runtime_compiler.py`
- `references/pto-runtime/python/kernel_compiler.py`

## 0) What pto-runtime is

`pto-runtime` is a modular task runtime for Ascend that deliberately separates execution into **three independently
compiled programs** that cooperate via stable APIs:

1) **Host runtime** (shared library loaded by Python via `ctypes`)
2) **AICPU program** (scheduler / orchestration control plane)
3) **AICore program** (workers executing compute kernels)

This split is important for PTO‑WSP v10 because it matches the “compiled artifact enforces semantics” principle:
Python drives compilation and launches runtime, but scheduling/execution semantics live in the binaries.

## 1) Platforms: `a2a3` vs `a2a3sim`

pto-runtime supports:

- `a2a3`: real Ascend hardware (requires CANN toolchain)
- `a2a3sim`: host-thread simulation of AICPU/AICore (requires only host gcc/g++)

Key property of `a2a3sim` (for v10):
- It exercises the **same host↔(AICPU/AICore) handshake shape** without requiring hardware/toolchains, making it a strong
  correctness backend for PTO‑WSP scheduling/CSP semantics during development and CI.

## 2) Runtime “flavors” shipped today

pto-runtime is not a single runtime; it has multiple runtime implementations under `references/pto-runtime/src/runtime/`:

### 2.1 `host_build_graph` (host-built task graph)

- Orchestration runs on **host**: a generated orchestration `.so` builds a task graph (tasks + deps), then launches.
- Useful for quickly validating kernel registration + task dependency scheduling.

Pointers:
- `references/pto-runtime/src/runtime/host_build_graph/runtime/runtime.h`
- `references/pto-runtime/src/runtime/host_build_graph/aicpu/aicpu_executor.cpp`
- `references/pto-runtime/src/runtime/host_build_graph/aicore/aicore_executor.cpp`
- `references/pto-runtime/examples/host_build_graph/vector_example/kernels/kernel_config.py`

### 2.2 `aicpu_build_graph` (AICPU-built task graph)

- Moves more “build graph” logic onto AICPU.
- Useful stepping stone when transitioning orchestration from host → device.

Pointers:
- `references/pto-runtime/src/runtime/aicpu_build_graph/`

### 2.3 `tensormap_and_ringbuffer` (PTO2 task-buffer runtime)

This is the most relevant runtime for PTO‑WSP v10 “task window / bounded backpressure” goals.

Core idea:
- The orchestration/scheduler use a **bounded shared-memory layout** (task ring + dep list pool) and additional private
  structures (TensorMap + scope stack + ready queues), with explicit backpressure and lifecycle rules.

Pointers:
- `references/pto-runtime/src/runtime/tensormap_and_ringbuffer/runtime/pto_runtime2.h`
- `references/pto-runtime/src/runtime/tensormap_and_ringbuffer/runtime/pto_shared_memory.h`
- `references/pto-runtime/src/runtime/tensormap_and_ringbuffer/runtime/pto_orchestrator.h`
- `references/pto-runtime/src/runtime/tensormap_and_ringbuffer/runtime/pto_scheduler.h`

The `kernel_config.py` for this runtime is also instructive: it configures multiple AICPU threads and a device-side
orchestrator thread.
- `references/pto-runtime/examples/tensormap_and_ringbuffer/vector_example/kernels/kernel_config.py`

## 3) Python-driven build / run toolchain (the part PTO‑WSP should reuse)

pto-runtime provides a Python build toolchain that compiles the three components with the correct platform-specific
toolchains:

- `RuntimeCompiler(platform=...)`: compiles host + aicpu + aicore components
- `KernelCompiler`: compiles kernels and extracts/registers `.text` sections
- `RuntimeBuilder(platform=...)`: higher-level “build runtime + run example” driver

Pointers:
- `references/pto-runtime/python/runtime_builder.py`
- `references/pto-runtime/python/runtime_compiler.py`
- `references/pto-runtime/python/kernel_compiler.py`

Notable practical detail:
- `pto-isa` headers are auto-cloned by the example runner if needed (unless `PTO_ISA_ROOT` is already set). This makes
  “runnable in a clean environment” easier for `a2a3sim` and supports `a2a3` when toolchains exist.

## 4) Observability hooks

pto-runtime includes tooling to export execution traces (e.g., “swimlane” style task timelines) and performance records.
For PTO‑WSP v10, this suggests a concrete direction:

- treat profiling output as part of the backend package contract,
- standardize a minimal “task timeline + stall reasons” schema for schedule/CSP debugging across backends.

Pointers:
- `references/pto-runtime/tools/swimlane_converter.py`
- `references/pto-runtime/tools/perf_to_mermaid.py`
- `references/pto-runtime/src/platform/a2a3sim/host/device_runner.cpp` (profiling export in sim)

## 5) Why this matters for PTO‑WSP v10

PTO‑WSP’s value is the **workload–schedule programming model** (portable semantics). pto-runtime’s value is a credible
execution substrate that can:

- run locally with device-faithful shape (`a2a3sim`),
- run on real devices when toolchains exist (`a2a3`),
- evolve toward bounded, diagnosable execution (`tensormap_and_ringbuffer` / PTO2).

For v10, the right direction is:
- PTO‑WSP keeps its DSL + compilation pipeline general (Python-driven, always-codegen),
- PTO‑WSP backends target pto-runtime package conventions and reuse its build/run toolchain,
- PTO‑WSP gradually maps its semantics (dispatch/task_window/CSP/predicates) into the PTO2-style bounded runtime model.

Related reference note:
- `docs/reference/pto_runtime/task_buffer.md`

