# PTO-RT v9 Tutorial (CPU-sim first, NPU emit-only here)

> **Date:** 2026-01-28  
> This tutorial shows the “golden path” for v9: **codegen-first CPU simulation**.  
> For full examples, see `examples/`.

---

## 0) Build prerequisites

From repo root:

```bash
cmake -B build -DPTO_ISA_PATH=3rdparty/pto-isa
cmake --build build
```

Run Python with the repo package:

```bash
PYTHONPATH=python python -c "import pto_wsp; print(pto_wsp.__version__)"
```

---

## 1) Minimal “hello workload” (CPU-sim)

```python
import numpy as np

from pto_wsp import DType, Dense, In, Out, Tensor, Tile, kernel, pto, workload, P

F32 = DType.F32

@kernel
def add_tile(a: In[Tile[32, 32, F32]], b: In[Tile[32, 32, F32]], out: Out[Tile[32, 32, F32]]):
    x = pto.load(a)
    y = pto.load(b)
    pto.store(out, pto.add(x, y))

@workload
def w():
    one = Dense[1]()
    for i in P(one):
        add_tile[i](a=A, b=B, out=C)

A_np = np.random.randn(32, 32).astype(np.float32)
B_np = np.random.randn(32, 32).astype(np.float32)
C_np = np.zeros((32, 32), dtype=np.float32)

A = Tensor(data=A_np, shape=A_np.shape, dtype=F32)
B = Tensor(data=B_np, shape=B_np.shape, dtype=F32)
C = Tensor(data=C_np, shape=C_np.shape, dtype=F32)

program = w().named("hello_add").compile(target="cpu_sim")
program.execute()
program.synchronize()

stats = program.stats() if callable(program.stats) else program.stats
print("total_cycles =", int(getattr(stats, "total_cycles", 0) or 0))
```

Notes:
- v9 compiles at the **workload** level (`workload.compile(target=...)`), producing a runnable artifact.
- `program.stats.total_cycles` is reported in **PTO‑ISA cycles**.
- `program.enable_tracing(...)` (optional) records **wall-clock** timings for debugging; it is not a substitute for `total_cycles`.

---

## 2) Workloads and loops (`P`)

```python
from pto_wsp import Dense, DenseDyn, workload, P

batch = DenseDyn(4)
heads = Dense[8]()

@workload
def attention(batch, heads):
    for b, h in P(batch, heads):   # independent tasks
        attn[b, h](Q=Q[b, h], K=K[b], V=V[b], O=O[b, h])

@workload
def scan(seq):
    for i in P.seq(seq):           # sequential dependencies
        step[i](x=X[i], y=Y[i])
```

---

## 3) Runtime predicates (`cond`) and tensor-driven slots

`P.when(...)` is a runtime conditional: the predicate is compiled as **ScalarExpr** and evaluated in the artifact.

To drive a predicate from runtime tensor data, use slots:

```python
import numpy as np
from pto_wsp import Tensor, workload, P, sequential, slot_load_u64
from pto_wsp.scalar_expr import slot_u64

@workload
def tiered(scores):
    # Load scores[0,0] into slot 0, then branch on it.
    sequential(
        slot_load_u64(0, scores, row=0, col=0),
        _branch(),
    )

@workload
def _branch():
    with P.when(slot_u64(0) < 10):
        small_path()
    with P.otherwise():
        big_path()
```

This pattern supports data-dependent execution without recompiling the artifact and without a Python “driver loop”.

---

## 4) Scheduling (what v9 enforces)

v9 enforces a behavior-changing subset inside CPU-sim artifacts:
- `dispatch(...)`
- `task_window(..., mode=STALL)` (stall-only)

```python
from pto_wsp import DispatchPolicy, TaskWindow, WindowMode

program = (w()
    .dispatch(DispatchPolicy.round_robin(4))
    .task_graph(window=TaskWindow(8192, "tasks", WindowMode.STALL))
    .compile(target="cpu_sim"))
```

NPU emission preserves these fields, while other schedule knobs are explicitly unsupported in v9 artifacts.

---

## 5) Kernel authoring modes (3 modes)

v9 supports three complementary kernel authoring modes that all compile into the same artifact:

1) **`pto.*` IR-traced kernels** (high-level tile ops lowered by C++ codegen to PTO‑ISA calls)
2) **`ptoisa.*` instruction-traced kernels** (`@ptoisa_kernel`)
3) **File-based custom kernels** (`cpp_body_path` / `cpp_tu_path`)

### 5.1 `@ptoisa_kernel` (Python → emitted C++ kernel body)

```python
from pto_wsp import DType, In, Out, Tile, ptoisa, ptoisa_kernel

F32 = DType.F32

@ptoisa_kernel
def square(x: In[Tile[1, 32, F32]], out: Out[Tile[1, 32, F32]]):
    v = ptoisa.tload(x)
    ptoisa.tstore(out, ptoisa.TMUL(v, v))
```

### 5.2 File-based custom kernel body snippet (`cpp_body_path`)

```python
from pto_wsp import kernel

@kernel(cpp_body_path="path/to/body_snippet.cpp")
def my_kernel(...):
    pass  # never executed in Python
```

### 5.3 Full translation unit kernel (`cpp_tu_path`)

```python
from pto_wsp import kernel

@kernel(cpp_tu_path="path/to/kernel.cpp")
def my_kernel(...):
    pass
```

Use file-based kernels as an escape hatch for handwritten PTO‑ISA code.

---

## 6) NPU backend (emit-only here)

In this environment, `target="ascend_npu"` emits an artifact source tree for inspection:

```python
program = w().named("my_npu_artifact").compile(target="ascend_npu")
print(program.codegen_artifact_dir)
```

Device build/execution requires an Ascend/CANN toolchain.

---

## 7) Run the full example suite

```bash
for f in examples/*/*_example.py; do
  PYTHONPATH=python python "$f"
done
```
