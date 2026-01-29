# DeepSeek Lightning Indexer Example

## Overview

This example demonstrates the **v9 Path A** approach for “Lightning Indexer”-style
tiered TopK routing **without** a `pto.topk` primitive:

- **TopK indices** are computed by a **custom PTO‑ISA kernel** (`TSORT32`), compiled into the CPU-sim artifact.
- **Tier selection** is runtime and **data-driven** via `slot_load_u64(tensor)->slot_u64(i)` and `P.when(...)`.
- Scheduling uses the **v9-enforced knobs**: `dispatch` + `task_window` (stall-only).

This is intentionally a *small* validated workload; it is not a full DeepSeek-V3.2
end-to-end attention system.

## v9 Features Demonstrated

- **Three kernel authoring modes (v9)** used by this example:
  - `pto.*` (`@kernel` IR-traced): high-level kernel IR lowered to PTO-ISA, including TSORT32 helpers.
  - `ptoisa.*` (`@ptoisa_kernel`): Python authoring of PTO-ISA instructions → emitted C++ body.
  - `cpp_body_path` (`@kernel(cpp_body_path=...)`): keep a manual PTO-ISA C++ body snippet in a file.
- **Runtime predicates**:
  - `slot_load_u64(0, EffSeq[b][s])` loads a u64 value from tensor memory into a runtime slot.
  - `slot_u64(0) <= ...` drives `P.when(...)` conditionals inside the artifact.
- **Scheduling (v9 behavior-changing)**:
  - `dispatch(DispatchPolicy.affinity(lambda t: t.get("b")))`
  - `task_graph(window=TaskWindow(..., WindowMode.STALL))` (window overflow stalls)

## Prerequisites

- Python >= 3.10
- pto-wsp package (from project root's `python/` directory)

## How to Run

```bash
# From project root
python examples/deepseek_lightning_indexer/deepseek_lightning_indexer_example.py

# Or using Makefile
cd examples/deepseek_lightning_indexer
make run
```

## Expected Behavior

The example:

- Runs CPU-sim codegen-first execution.
- Prints `Total cycles: <non-zero>`.
- Exits with code 0 and prints `Status: PASS` when:
  - TopK indices match the NumPy reference (`golden.py`) for every tile.
  - Total cycles is non-zero.

## Checking Rules

### Pass Criteria

- [x] Exit code is 0.
- [x] Indices match the stable NumPy TopK reference.
- [x] Total cycles is non-zero.

### Behavior Checking

Use `/codex` to verify example behavior:
```bash
codex exec "PYTHONPATH=python python examples/deepseek_lightning_indexer/deepseek_lightning_indexer_example.py"
```

### Fail Indicators

- ImportError or ModuleNotFoundError
- Wrong indices vs reference
- Total cycles == 0

## Troubleshooting

**ModuleNotFoundError: No module named 'pto_wsp'**
- Run from project root directory
- Requires pip install -e . from project root
