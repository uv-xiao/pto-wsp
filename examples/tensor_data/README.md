# Tensor Data Example

This example validates **tensor data binding** end-to-end:

- NumPy arrays are bound to `pto_wsp.Tensor(data=...)`.
- The workload is compiled to a CPU-sim artifact and executed.
- Results are checked against NumPy references.
- Total cycles is required to be non-zero (tolerance-based policy).

## Overview

The validated runner performs two checks:

1. Tile add: `C = A + B` on `[B,TM,TN,R,C]` tiled data.
2. Tile square: `Y = X * X` on `[B,T,R,C]` tiled data.

## Running the Example

```bash
# From repository root
make -C examples/tensor_data run

# Or directly
PYTHONPATH=python python examples/tensor_data/tensor_data_example.py
```

## Files

- `golden.py`: NumPy references.
- `pto_wsp_impl.py`: PTO‑RT implementation (artifact compilation + execution).
- `tensor_data_example.py`: runner/checker.
