# PTO-RT v9 Examples

This directory contains example applications demonstrating PTO-RT v9 features.

## Quick Start

All examples can be run from the project root:

```bash
# Run any example
python examples/<name>/<name>_example.py

# Or use Makefile from example directory
cd examples/<name>
make run
```

## Example Overview

| Example | Description | Key v9 Features |
|---------|-------------|-----------------|
| [attention](attention/) | Multi-head scaled dot-product attention | `@jit_kernel`, streams, TypeChecker |
| [bgemm](bgemm/) | Batched general matrix multiplication | Task graph, `P.seq()`, work-stealing |
| [softmax](softmax/) | Online softmax (numerically stable) | Two-phase algorithm, FIFO scheduling |
| [llama](llama/) | LLaMA-7B transformer layer | Multiple kernels, hybrid deps, pools |
| [deepseek_v3](deepseek_v3/) | DeepSeek-V3 MoE with MLA | Sparse routing, R5 extensions |
| [e2e_rmsnorm](e2e_rmsnorm/) | End-to-end pipeline demonstration | All 7 stages, timing stats |
| [flashinfer_decode](flashinfer_decode/) | FlashInfer-style decode attention | Plan-run model, tier kernels |
| [deepseek_lightning_indexer](deepseek_lightning_indexer/) | Tier-based TopK indexer | Dynamic tier selection, affinity |

## v9 Features by Example

### Core Features (all examples)

- `@jit_kernel` decorator with `tl.*` primitives
- `@workload` decorator with `P` namespace
- `@kernel` stub for task definition
- Combinator-style scheduling

### Scheduling Features

| Feature | Examples |
|---------|----------|
| Stream scheduling | attention, bgemm |
| Task graph | bgemm, softmax, llama, deepseek_v3 |
| Work-stealing | bgemm, llama, flashinfer_decode |
| FIFO ready policy | softmax |
| Hybrid dependencies | llama, deepseek_v3 |
| Pool separation | llama, deepseek_v3 |

### Advanced Features

| Feature | Examples |
|---------|----------|
| `P.seq()` sequential iteration | bgemm, softmax |
| `P.sel()` sparse iteration | deepseek_v3 |
| Tier-based kernels | flashinfer_decode, deepseek_lightning_indexer |
| Plan-run execution | flashinfer_decode |
| TypeChecker / Layout join | attention |

## Directory Structure

Each example directory contains:

```
examples/<name>/
├── <name>_example.py   # Main Python script
├── README.md           # Documentation with expected behavior
└── Makefile            # Build/run scripts
```

## Expected Behavior

All examples should:

1. Run without errors (exit code 0)
2. Complete with "Example completed successfully!" message
3. Show no `npu()` deprecation warnings (all use `@jit_kernel`)
4. Execute CPU simulation successfully

## Running All Examples

```bash
# Run all examples and check for success
for d in examples/*/; do
  echo "=== $d ==="
  python "${d}"*_example.py 2>&1 | tail -3
  echo ""
done
```

## Behavior Verification

Each example's README includes:

- Expected output sample
- Pass/fail criteria
- Codex verification command
- Troubleshooting tips

Use `/codex` to verify behavior:

```bash
codex exec "Run examples/<name>/<name>_example.py and verify behavior matches README"
```

## Prerequisites

- Python >= 3.10
- pto-wsp package (from `python/` directory)

## Troubleshooting

**ModuleNotFoundError: No module named 'pto_wsp'**
- Run from project root directory
- Examples include `sys.path.insert(0, 'python')` for path resolution

**DeprecationWarning for batch_deps/pipeline_depth**
- Expected in `deepseek_v3` example (R5 extensions demonstration)
- Not considered failures

**Test specific examples**

```bash
# Verify single example
python examples/attention/attention_example.py

# Check for any errors
python examples/attention/attention_example.py 2>&1 | grep -i error
```
