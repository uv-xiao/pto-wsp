# PTO Workload-Schedule Programming (PTO-WSP) framework (pto-wsp)

> **Status: Prototype (v9.3 design, 0.1.0 implementation)**
>
> This is an early-stage implementation. APIs may change. See [docs/spec.md](docs/spec.md) for the current specification.

PTO Workload-Schedule Programming (PTO-WSP) framework enables dynamic LLM workloads on Ascend NPU and other accelerators with typed workload expressions and two parallelism modes (data-parallel + CSP pipeline-parallel).

## Feature status vocabulary (v9)

- **Enforced**: semantics are implemented in the generated **CPU-sim** artifact (codegen-first).
- **Emit-only**: NPU backend emits a source tree for inspection; device execution requires Ascend/CANN.
- **API-only / diagnosed**: API exists but is ignored or only partially applied; v9 artifacts print diagnostics.

## Overview

This project extends PTO-ISA with a runtime layer that provides:

1. **Python frontend** - Declarative workload definition with combinator-style scheduling
2. **C++ IR layer** - Explicit intermediate representation for multi-backend targeting
3. **Multi-backend support** - CPU simulation, Ascend NPU
4. **Layout + distribution types** - layout as a tensor type refinement (`TensorLayout`, `relayout`, collectives)
5. **Codegen-first execution** - compile workloads into runnable artifacts (CPU-sim) or emitted source trees (NPU)

## Quick Start

```python
import numpy as np

from pto_wsp import (
    kernel, workload, P, pto,
    In, Out, Tile,
    Dense, DenseDyn, Tensor, DType,
    DispatchPolicy, TaskWindow, WindowMode,
)

# Example shapes for a minimal run
batch_size = 1
num_tiles = 2

# Define a kernel with pto.* primitives (codegen-first CPU-sim supported ops)
@kernel
def rmsnorm_tile(
    x: In[Tile[16, 128, DType.F32]],
    out: Out[Tile[16, 128, DType.F32]],
):
    sq = pto.mul(x, x)
    mean_sq = pto.rowmean(sq)
    inv = pto.rsqrt(mean_sq)
    pto.store(out, pto.mul(x, inv))

# Define tensors + axes
X_np = np.random.randn(batch_size, num_tiles, 16, 128).astype(np.float32)
Y_np = np.zeros((batch_size, num_tiles, 16, 128), dtype=np.float32)

X = Tensor(data=X_np, shape=X_np.shape, dtype=DType.F32)
Y = Tensor(data=Y_np, shape=Y_np.shape, dtype=DType.F32)

batch = Dense[batch_size]()
tiles = Dense[num_tiles]()

@workload
def attention():
    for b in P(batch):
        for t in P(tiles):
            rmsnorm_tile[b, t](x=X[b][t], out=Y[b][t])

# Schedule (v9 enforced subset in CPU-sim artifacts): dispatch + task_window(stall)
program = (attention()
    .dispatch(DispatchPolicy.round_robin(num_aicpus=4))
    .task_graph(window=TaskWindow(8192, "tasks", WindowMode.STALL))
    .compile(target="cpu_sim"))

# Execute
program.execute()
program.synchronize()
stats = program.stats() if callable(program.stats) else program.stats
print("total_cycles =", int(getattr(stats, "total_cycles", 0) or 0))
```

## Documentation

- [Design Analysis](docs/analysis.md) - v9 design rationale
- [API Specification](docs/spec.md) - Full Python API
- [IR Design](docs/design/ir-design.md) - C++ IR architecture
- [Backend Architecture](docs/design/backend-arch.md) - Backend specifications

## Project Structure

```
pto-wsp/
├── docs/                    # Design documents and research
├── python/pto_wsp/           # Python frontend
│   ├── __init__.py
│   ├── workload.py          # WorkloadBuilder
│   ├── schedule.py          # ScheduleBuilder
│   ├── primitives.py        # Primitive functions
│   └── csp.py               # CSP primitives
├── include/pto/wsp/          # C++ headers
│   ├── ir/                  # IR node types
│   └── backend/             # Backend implementations
├── tests/                   # Unit tests
├── examples/                # Usage examples
└── 3rdparty/                # External dependencies (pto-isa)
```

## Dependencies

- **pto-isa**: PTO Tile Library (3rd party dependency)
- Python >= 3.10
- CMake >= 3.16
- C++23 compiler

## Installation

### Python Package (Development Mode)

```bash
# Install in editable mode
pip install -e .

# Run examples
python examples/attention/attention_example.py
```

### Building C++ Backend

```bash
# Configure with pto-isa path
cmake -B build -DPTO_ISA_PATH=/path/to/pto-isa

# Build
cmake --build build

# Install C++ module to Python package
cp build/*.so python/pto_wsp/

# Run C++ tests
ctest --test-dir build

# Run Python tests
python -m pytest tests/
```

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

Third-party dependencies (notably pto-isa) have separate licenses. See [LICENSE-3RD-PARTY.md](LICENSE-3RD-PARTY.md) for details.

## References

- [PTO-ISA](../pto-isa/) - PTO Tile Library
- [Research Notes](docs/research/) - Background research
