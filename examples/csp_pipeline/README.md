# CSP Pipeline Example (validated)

This example demonstrates **codegen-first CSP/CSPT** execution in **CPU-sim** using:

- `Channel`, `process`, `send`, `consume`, `connect`
- strict timebase in **PTO-ISA cycles**
- constant channel latency (default `0`, controlled by runtime symbol `__pto_wsp_channel_latency_cycles`)

## What this example computes

For each tile `t`, compute:

```
Y[t] = X[t] * X[t]
```

The pipeline is split into three sequential processes:

```
  loader        computer        storer
    |             |              |
    v             v              v
  load_tile --> square_tile --> store_tile
      (T0)        (T1)            (Y)
```

## Important v9 detail: channels carry tokens, not tensors

In v9 codegen-first CSPT artifacts, `send(channel, task(...))`:

- **executes** the task for its side effects (writes tensors)
- sends a **token** equal to the **first axis argument** of the task

`consume(channel, lambda t: ...)` receives that token and binds it as the loop variable `t`.

This matches the v9 design goal: channels are primarily for **synchronization**; large data remains in global tensors.

## Run

```bash
python examples/csp_pipeline/csp_pipeline_example.py
```

## Files

- `examples/csp_pipeline/golden.py`: numpy reference (`square_pipeline_ref`)
- `examples/csp_pipeline/pto_wsp_impl.py`: PTO-WSP implementation (`run_csp_square_pipeline`)
- `examples/csp_pipeline/csp_pipeline_example.py`: runner + checks
