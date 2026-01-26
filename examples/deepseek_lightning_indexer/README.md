# DeepSeek Lightning Indexer Example

## Overview

This example demonstrates DeepSeek Lightning Indexer using PTO-RT v9:
- Tier-based TopK computation (2K/8K/64K/128K thresholds)
- Dynamic tier selection based on effective sequence length
- Parallel iteration over batch, sequence position, and index heads

Based on DeepSeek's tier-based indexing approach for efficient TopK with padding.

## v9 Features Demonstrated

- `@jit_kernel` decorator with `tl.*` primitives:
  - 4 tier-specific kernels: `tier_2k_kernel` through `tier_128k_kernel`
  - `tl.matmul`, `tl.rowmax`, `tl.topk` operations
  - Typed Value operations (no strings)
- `@workload` decorator with `P` namespace
- Dynamic tier selection based on `eff_seq = act_seq - causal_offset`
- Task graph scheduling with `.task_graph()`
- Affinity-based dispatch for cache locality
- Multiple streams for pipelined execution

## Prerequisites

- Python >= 3.10
- pto-wsp package (from project root's `python/` directory)

## How to Run

```bash
# From project root
python examples/deepseek_lightning_indexer/deepseek_lightning_indexer.py

# Or using Makefile
cd examples/deepseek_lightning_indexer
make run
```

## Expected Behavior

### Successful Execution

- Program shows configuration and tier distribution analysis
- Tier assignment for each (batch, seq_pos) combination
- Tier task counts with percentages
- CPU simulation execution
- Summary of v9 patterns demonstrated

### Expected Output Sample

```
======================================================================
DeepSeek Lightning Indexer - PTO-RT v9 Example
======================================================================

Configuration:
  Batch size: 4
  Query length (S1): 16
  Index heads (n2): 4
  Actual seq lengths: [512, 2048, 8192, 32768]
  Total tasks: 256

Tier distribution analysis:
  batch=0, seq_pos=0: eff_seq=497 -> tier=2K
  batch=0, seq_pos=15: eff_seq=512 -> tier=2K
  batch=1, seq_pos=0: eff_seq=2033 -> tier=2K
  batch=1, seq_pos=15: eff_seq=2048 -> tier=2K
  batch=2, seq_pos=0: eff_seq=8177 -> tier=8K
  batch=2, seq_pos=15: eff_seq=8192 -> tier=8K
  batch=3, seq_pos=0: eff_seq=32753 -> tier=64K
  batch=3, seq_pos=15: eff_seq=32768 -> tier=64K

Tier task counts:
  2K: 128 tasks (50.0%)
  8K: 64 tasks (25.0%)
  64K: 64 tasks (25.0%)

Creating workload...
Program compiled: Program

Executing with CPU simulation...
Execution complete!

======================================================================
Lightning Indexer v9 Summary
======================================================================
...
```

## Checking Rules

### Pass Criteria

- [x] Exit code is 0
- [x] No Python exceptions raised
- [x] Total tasks = 256 (4 × 16 × 4)
- [x] Tier distribution shows 2K, 8K, 64K tiers
- [x] "Execution complete!" message printed
- [x] No `npu()` deprecation warnings

### Behavior Checking

Use `/codex` to verify example behavior:
```bash
codex exec "Run examples/deepseek_lightning_indexer/deepseek_lightning_indexer.py and verify:
1. Configuration shows batch=4, query_len=16, idx_heads=4
2. Tier distribution analysis shows correct tier assignments
3. Tier task counts sum to 256 total
4. 4 tier-specific JIT kernels defined
5. No deprecated API warnings (npu() should not appear)"
```

### Fail Indicators

- ImportError or ModuleNotFoundError
- AttributeError (API mismatch)
- RuntimeError during execution
- Total tasks != 256
- Missing tier distribution
- DeprecationWarning for npu() usage

## Troubleshooting

**ModuleNotFoundError: No module named 'pto_wsp'**
- Run from project root directory
- Or add `sys.path.insert(0, 'python')` at script start

**Tier selection not matching expected**
- Verify actual sequence lengths configuration
- Check causal_offset calculation: `causal_offset = seq_len - seq_pos - 1`
- Effective sequence: `eff_seq = act_seq - causal_offset`
