# PTO Workload-Schedule Programming (PTO-WSP) framework (pto-wsp)

PTO Workload-Schedule Programming (PTO-WSP) framework enables dynamic LLM workloads on Ascend NPU and other accelerators with typed workload expressions and two parallelism modes (data-parallel + CSP pipeline-parallel).

## Overview

This project extends PTO-ISA with a runtime layer that provides:

1. **Python frontend** - Declarative workload definition with combinator-style scheduling
2. **C++ IR layer** - Explicit intermediate representation for multi-backend targeting
3. **Multi-backend support** - CPU simulation, Ascend NPU, and AMD AIE (spatial)
4. **Spatial schedule primitives** - `spatial_map` and `layout` for dataflow architectures

## Quick Start

```python
from pto_wsp import (
    kernel, workload, P, tl,
    In, Out, Tile,
    Dense, DenseDyn, Tensor, DType,
    DispatchPolicy, TimingPolicy,
)

# Define kernel with tl.* primitives (Triton-style)
@kernel
def attention_kernel(
    q: In[Tile[64, 128, DType.F16]],
    k: In[Tile[64, 128, DType.F16]],
    v: In[Tile[64, 128, DType.F16]],
    out: Out[Tile[64, 128, DType.F16]],
):
    scores = tl.matmul(q, k)      # Q @ K^T
    weights = tl.softmax(scores)   # softmax(scores)
    result = tl.matmul(weights, v) # weights @ V
    tl.store(out, result)

# Define workload with @workload + P namespace
batch = DenseDyn(batch_size)
heads = Dense[8]()

@workload
def attention():
    for b, h in P(batch, heads):
        attention_kernel[b, h](q=Q[b,h], k=K[b], v=V[b], out=O[b,h])

# Schedule (combinator style)
program = (attention()
    .dispatch(DispatchPolicy.affinity(lambda t: t.get("b")))
    .streams(2)
    .timing(TimingPolicy.immediate)
    .compile())

# Execute
program.execute()
```

## Documentation

- [Design Analysis](docs/analysis.md) - v9 design rationale
- [API Specification](docs/spec.md) - Full Python API
- [IR Design](docs/ir-design.md) - C++ IR architecture
- [Backend Architecture](docs/backend-arch.md) - CPU sim, NPU, AIE backends

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
├── include/pto/rt/          # C++ headers
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

## Building

```bash
# Configure with pto-isa path
cmake -B build -DPTO_ISA_PATH=/path/to/pto-isa

# Build
cmake --build build

# Run tests
ctest --test-dir build
```

## License

[To be determined]

## References

- [PTO-ISA](../pto-isa/) - PTO Tile Library
- [Research Notes](docs/research/) - Background research
