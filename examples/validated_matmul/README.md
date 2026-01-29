# Validated MatMul (Golden Path)

This example is the smallest “golden path” for PTO‑RT v9:

- Defines a single `@kernel` using `pto.matmul` + `pto.store`.
- Defines a `@workload` that invokes that kernel.
- Compiles to the **CPU-sim artifact** (C++ IR → codegen → build `.so`) and executes it.
- Validates numerical correctness vs NumPy.
- Checks that reported cycles are non-zero (tolerance-based policy).

## Files

- `golden.py`: NumPy reference (`matmul_ref`).
- `pto_wsp_impl.py`: PTO‑RT implementation (`run_matmul`).
- `validated_matmul_example.py`: runner that prepares data, runs both, checks outputs + cycles.

## Run

```bash
PYTHONPATH=python python examples/validated_matmul/validated_matmul_example.py
```

