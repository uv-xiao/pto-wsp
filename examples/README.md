# PTO-WSP Examples (v9, validated)

These examples are **real-data validations** of PTO-WSP v9’s **codegen-first** execution on the `cpu_sim` target.

Each example:
- computes a result with a **golden** (reference) implementation
- compiles and runs a **generated artifact** (C++ codegen → shared library)
- checks **output correctness**
- checks **`total_cycles`** against a baseline with tolerance

## Run one

```bash
PYTHONPATH=python python examples/<name>/<name>_example.py
```

Most examples also support:

```bash
make -C examples/<name> run
```

## Run all (recommended)

```bash
PYTHONPATH=python python -m pytest -q tests/test_examples_run.py
```

## Directory structure

Every validated example follows:

```
examples/<name>/
  golden.py          # reference implementation
  pto_wsp_impl.py     # PTO-WSP implementation (builds/runs codegen artifacts)
  <name>_example.py  # runner: data prep, run, check, report
  README.md          # explains how PTO-WSP APIs construct the workload
```

## Example index

- `validated_matmul/`: small matmul sanity check
- `tensor_data/`: tensor indexing/views in codegen-first tasks
- `attention/`: tiled attention (matmul + softmax pieces)
- `bgemm/`: batched GEMM
- `softmax/`: numerically-stable softmax
- `llama/`: toy transformer block pieces
- `deepseek_v3/`: toy MoE-style block (routing math)
- `e2e_rmsnorm/`: end-to-end RMSNorm
- `flashinfer_decode/`: toy decode attention (FlashInfer-style structure)
- `deepseek_lightning_indexer/`: TopK-like indexer via custom PTO-ISA kernel (Path A)
- `csp_pipeline/`: CSP/CSPT pipeline (channels as sync tokens)

## Notes

- If you don’t have the Ascend/CANN environment, NPU artifacts can still be emitted, but they can’t be executed here.
