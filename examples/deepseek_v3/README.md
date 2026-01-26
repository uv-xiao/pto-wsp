# DeepSeek-V3.2 Mixture of Experts (MoE) Example

## Overview

This example demonstrates DeepSeek-V3.2 architecture features using PTO-RT v9:
- Multi-head Latent Attention (MLA) with KV compression
- DeepSeekMoE with sparse expert routing
- Shared experts (always activated)
- Top-K expert selection

Based on DeepSeek-V3 Technical Report (2024).

## v9 Features Demonstrated

- `@jit_kernel` decorator with `tl.*` primitives for multiple kernels:
  - `expert_ffn_jit`: Expert FFN with SwiGLU (cube operations)
  - `router_jit`: Router for expert selection
  - `mla_attention_jit`: Multi-head Latent Attention
- `@workload` decorator with `P` namespace
- `P.sel()` for sparse expert iteration
- Task graph scheduling with tracing (`TracePolicy.cycles()`)
- Hybrid dependency inference
- Large task window (16384) for many experts
- Advanced R5 schedule extensions (deprecated but demonstrated):
  - `dispatch_threshold()` for adaptive dispatch
  - `batch_deps()` for batched dependency resolution
  - `pipeline_depth()` for pipelining

## Prerequisites

- Python >= 3.10
- pto-wsp package (from project root's `python/` directory)

## How to Run

```bash
# From project root
python examples/deepseek_v3/deepseek_v3_example.py

# Or using Makefile
cd examples/deepseek_v3
make run
```

## Expected Behavior

### Successful Execution

- Program completes without errors
- Outputs DeepSeek-V3 style configuration (MLA + MoE parameters)
- Shows 3 JIT kernel definitions
- Task estimation shows sparse expert computation (2048 tasks)
- Demonstrates both basic and advanced schedules
- Execution completes with "Example completed successfully!"

**Note**: This example intentionally uses deprecated R5 extensions (`batch_deps()`, `pipeline_depth()`) to demonstrate advanced scheduling features. These emit deprecation warnings which are expected.

### Expected Output Sample

```
============================================================
PTO-RT v9 DeepSeek-V3.2 MoE Example
============================================================

Configuration (DeepSeek-V3 style):
  Batch size: 1
  Sequence length: 256
  Hidden dimension: 4096

MLA Configuration:
  Num heads: 32
  Head dim: 128
  KV LoRA rank: 512

MoE Configuration:
  Total experts: 256
  Active experts (top-K): 8
  Shared experts: 2
  Expert FFN dim: 1536

Task estimation:
  RMSNorm: 16
  MLA: 40
  Router: 16
  Expert FFN (sparse): 2048
  Shared experts: 16
  Combine + residual: 32

JIT Kernels (@jit_kernel + tl.*):
  expert_ffn_jit (cube): tl.matmul, tl.neg, tl.exp, tl.div, tl.mul
  router_jit: tl.matmul
  mla_attention_jit: tl.matmul, tl.rsqrt, tl.rowmax, tl.exp, tl.rowsum, tl.div

Building workload...
  Workload kind: combine

Basic task graph schedule:
  Deps: hybrid
  Window: 16384
  Pools: by_exec_unit
  Trace: cycles

Advanced schedule (with R5 extensions):
  Dispatch threshold: [256, 1024, 4096]
  Batch deps: threshold=128
  Pipeline depth: 3

Compiling program...
  Program type: Program

Executing with CPU simulation...
Execution complete!

Example completed successfully!
============================================================
```

## Checking Rules

### Pass Criteria

- [x] Exit code is 0
- [x] No Python exceptions raised
- [x] "Execution complete!" message printed
- [x] "Example completed successfully!" message printed
- [x] Expert FFN (sparse) = 2048 tasks
- [x] Window size = 16384
- [x] Trace policy shows "cycles"
- [x] No `npu()` deprecation warnings

### Expected Deprecation Warnings

The following deprecation warnings are **expected** (R5 extensions demo):
- `batch_deps() is deprecated (L10)`
- `pipeline_depth() is deprecated (L10)`

### Behavior Checking

Use `/codex` to verify example behavior:
```bash
codex exec "Run examples/deepseek_v3/deepseek_v3_example.py and verify:
1. Output matches expected behavior in README
2. All 3 JIT kernels defined correctly
3. MoE configuration shows 256 experts, top-8 selection
4. Only batch_deps/pipeline_depth warnings appear (not npu() warnings)
5. CPU simulation executes without errors"
```

### Fail Indicators

- ImportError or ModuleNotFoundError
- AttributeError (API mismatch)
- RuntimeError during execution
- Missing MoE configuration output
- DeprecationWarning for npu() usage

## Troubleshooting

**ModuleNotFoundError: No module named 'pto_wsp'**
- Run from project root directory
- Or add `sys.path.insert(0, 'python')` at script start

**Deprecation warnings**
- `batch_deps()` and `pipeline_depth()` warnings are expected
- These R5 extensions are retained for demonstration purposes
- For new code, use `.task_graph(deps=..., window=...)` instead
