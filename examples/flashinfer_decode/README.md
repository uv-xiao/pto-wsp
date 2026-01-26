# FlashInfer-Style Decode Attention Example

## Overview

This example demonstrates FlashInfer-style decode attention using PTO-RT v9:
- Plan-Run execution model (binary search chunk sizing)
- Work descriptors for O(1) work lookup
- Tier-based kernel selection (different chunk sizes)
- Online softmax (FlashAttention style)

Based on FlashInfer CUDA implementation patterns.

## v9 Features Demonstrated

- `@jit_kernel` decorator with `tl.*` primitives:
  - 4 tier-specific kernels: tier0_attention through tier3_attention
  - Online softmax: `tl.rowmax`, `tl.exp`, `tl.rowsum`
  - Typed Value operations (no strings)
- `AttentionPlanner` class for chunk size optimization
- `WorkDescriptor` dataclass for work description
- `@workload` decorator with `P` namespace
- Work-stealing schedule for dynamic load balancing
- Multiple streams for parallelism
- Immediate timing for low latency

## Prerequisites

- Python >= 3.10
- pto-wsp package (from project root's `python/` directory)

## How to Run

```bash
# From project root
python examples/flashinfer_decode/flashinfer_decode_attention.py

# Or using Makefile
cd examples/flashinfer_decode
make run
```

## Expected Behavior

### Successful Execution

- Program shows planning and execution phases
- Planning phase: optimal chunk size, work descriptors
- Tier distribution across 4 tiers
- First 5 work descriptors displayed
- CPU simulation execution
- Comparison table with FlashInfer CUDA

### Expected Output Sample

```
======================================================================
FlashInfer-Style Decode Attention - PTO-RT v9 Example
======================================================================

Configuration:
  Batch size: 16
  Number of heads: 32
  Sequence lengths: [512, 1024, 2048, 4096, 512, 512, 1024, 2048]...
  Total KV tokens: 53248

--- Planning Phase ---
  Optimal chunk size: 4096
  Generated 672 work descriptors

Plan Statistics:
  Chunk size: 4096
  Total work units: 672
  Tier distribution:
    Tier 0: 128 (19.0%)
    Tier 1: 128 (19.0%)
    Tier 2: 96 (14.3%)
    Tier 3: 320 (47.6%)
  Chunks per (request, head):
    min=1, max=4, avg=1.3

  First 5 descriptors:
    [0] req=0 head=0 kv=[0,512) tier=0 flags=0x03
    ...

--- Execution Phase ---
  Program compiled: Program
  Executing with CPU simulation...
  Execution complete!

======================================================================
FlashInfer-Style Decode Attention v9 Summary
======================================================================
...
```

## Checking Rules

### Pass Criteria

- [x] Exit code is 0
- [x] No Python exceptions raised
- [x] Planning phase completes with work descriptors
- [x] Tier distribution shows all 4 tiers
- [x] "Execution complete!" message printed
- [x] Comparison table displayed
- [x] No `npu()` deprecation warnings

### Behavior Checking

Use `/codex` to verify example behavior:
```bash
codex exec "Run examples/flashinfer_decode/flashinfer_decode_attention.py and verify:
1. Planning phase generates work descriptors
2. 4 tier-specific JIT kernels defined
3. Online softmax pattern (tl.rowmax, tl.exp, tl.rowsum)
4. Tier distribution percentages sum to 100%
5. No deprecated API warnings (npu() should not appear)"
```

### Fail Indicators

- ImportError or ModuleNotFoundError
- AttributeError (API mismatch)
- RuntimeError during execution
- Missing planning statistics
- Work descriptors not generated
- DeprecationWarning for npu() usage

## Troubleshooting

**ModuleNotFoundError: No module named 'pto_wsp'**
- Run from project root directory
- Or add `sys.path.insert(0, 'python')` at script start

**Chunk size calculation issues**
- Binary search finds optimal chunk size
- Default target is 512 work units
- Adjust sequence lengths if needed
