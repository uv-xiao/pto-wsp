# PTO Workload-Schedule Programming (PTO-WSP) framework

This directory contains the design and specification for the PTO Workload-Schedule Programming (PTO-WSP) framework, enabling dynamic LLM workloads on Ascend NPU with **typed workload expressions** and **two parallelism modes** (data-parallel + CSP pipeline-parallel).

## Quick Links

| Document | Description |
|----------|-------------|
| [TASKS.md](TASKS.md) | **Implementation task tracker** |
| [task_plan.md](task_plan.md) | Current task plan and phases |
| [findings.md](findings.md) | Research findings and decisions |
| [progress.md](progress.md) | Session progress log |
| [analysis.md](analysis.md) | **Unified design (v9.3) - Current** |
| [spec.md](spec.md) | **API specification (v9.3) - Current** |
| [requirements.md](requirements.md) | Problem statement and requirements |
| [design-questions.md](design-questions.md) | Design decisions (answered) |

## Directory Structure

```
docs/
├── README.md                      # This file
├── analysis.md                    # Unified design (v9.3) - WHY
├── spec.md                        # API specification (v9.3) - WHAT
├── features.md                    # Feature catalog with code links
├── requirements.md                # Problem statement and LLM dynamic patterns
├── task_plan.md                   # Implementation tasks and progress
├── comments.md                    # User feedback and requirements
│
├── reference/                     # Reference analysis (external research)
│   ├── 01_flashinfer.md           # FlashInfer: Plan-Run model
│   ├── 02_gpu_patterns.md         # GPU kernel patterns (Triton, CuTe)
│   ├── 03_pl_design.md            # PL design (Halide, TVM)
│   ├── 04_pto_isa.md              # PTO-ISA extension points
│   ├── 05_ascend_hw.md            # Ascend hardware model
│   ├── 06_megakernels.md          # Megakernels: instruction-interpreter
│   ├── 07_cuda_streams_workstealing.md  # CUDA streams, work stealing
│   ├── 08_tvm.md                  # TVM: tensor expression, schedules
│   ├── 09_sparsetir.md            # SparseTIR: composable sparse formats
│   ├── 10_relax.md                # Relax: symbolic shapes, cross-level
│   ├── 11_flashinfer.md           # FlashInfer deep dive
│   ├── 12_pypto.md                # PyPTO analysis
│   ├── 13_pto_isa_lh.md           # PTO-ISA-LH patterns
│   ├── 14_pto_isa_wc.md           # PTO-ISA-WC patterns
│   ├── 15_allo.md                 # Allo analysis
│   └── 16_dato.md                 # Dato analysis
│
├── research/                      # Intermediate working analysis
│   ├── context_manager_research.md
│   ├── jit_design_research.md
│   ├── task_graph_design_research.md
│   ├── cpp_restructuring_plan.md  # C++ file organization plan
│   └── ...                        # Other working documents
│
├── design/                        # Detailed design documents
│   ├── backend-arch.md            # Backend architecture
│   ├── ir-design.md               # IR system design
│   ├── npu-design.md              # NPU programming design
│   ├── type-system.md             # Type system design
│   └── on-device-task-gen.md      # On-device task generation design
│
├── archive/                       # Previous versions
│   └── v7/                        # v7 + v7.1 (Workload-Schedule, CSP separate)
│
└── examples/                      # Implementation examples (TBD)
```

### Document Organization

| Directory | Purpose | Contents |
|-----------|---------|----------|
| `docs/` | Core documents | analysis.md (WHY), spec.md (WHAT), features.md (catalog) |
| `docs/reference/` | Reference analysis | External research, numbered notes (01_xxx.md - 16_xxx.md) |
| `docs/research/` | Working analysis | Intermediate research, design explorations |
| `docs/design/` | Detailed design | Backend, IR, NPU, Type system specs |
| `docs/archive/` | Old versions | v1-v8 designs |

## Key Design Principles (v9.3)

### 1. Concise Workload Syntax (TileLang-inspired)

Workloads use `@workload` decorator with `P` namespace for loop constructors:

```python
from pto_wsp import workload, P, kernel, In, Out, Tensor, DenseDyn, Dense

@kernel
def attn_kernel(Q: In[Tensor], K: In[Tensor], V: In[Tensor], O: Out[Tensor]): ...

@workload
def attention(batch, heads):
    for b, h in P(batch, heads):              # Parallel grid
        attn_kernel[b, h](Q=Q[b,h], K=K[b], V=V[b], O=O[b,h])

# Type: Workload[DenseDyn × Dense[8], AttnTask, Independent]
```

### 2. Two Parallelism Modes

| Mode | Loop Constructors | Use Case |
|------|-------------------|----------|
| **Data-Parallel** | `P()`, `P.seq()`, `P.sel()`, `P.when()` | Same op on different data |
| **Pipeline-Parallel (CSP)** | `P.pipe()`, `Channel`, `send`, `consume` | Different ops in pipeline |

```python
# Data-parallel: attention across batches
@workload
def attention(batch, heads):
    for b, h in P(batch, heads):
        attn_kernel[b, h](...)

# Pipeline-parallel: megakernel with Loader→Computer→Storer
@workload
def pipeline(tiles):
    ch = Channel("l2c", depth=2)
    with P.pipe():
        for i in P.seq(tiles):
            send(ch, load[i](...))
        with consume(ch) as t:
            compute[t](...)
```

### 3. Direct Kernel Calls

No string-based task definitions. Call kernels directly:

```python
# RECOMMENDED: Direct kernel call
@kernel
def compute(buf: In[Tensor], out: Out[Tensor]): ...

@workload
def work(tiles):
    for t in P(tiles):
        compute[t](buf=data[t], out=result[t])  # Direct call
```

### 4. Structural Dependencies

Dependencies are **inferred from loop constructors**:

| Loop Constructor | Inferred Dependency |
|------------------|---------------------|
| `for ... in P(axes)` | Independent |
| `for ... in P.seq(axis)` | Sequential (i depends on i-1) |
| `with consume(ch) as t:` | Channel-based (producer-consumer) |

### 5. Combinator-Style Schedule

```python
program = (attention(batch, heads)
    .dispatch(DispatchPolicy.affinity(lambda t: t.get("batch")))
    .streams(4)
    .stream_by(lambda t: t.get("head") % 4)
    .timing(TimingPolicy.immediate)
    .compile())

program.execute()
```

## Core Abstractions (v9.3)

| Abstraction | Purpose |
|-------------|---------|
| `@workload` | Decorated function defining task generation |
| `P` namespace | Loop constructors (`P()`, `P.seq()`, `P.sel()`, `P.pipe()`) |
| `@kernel` | JIT kernel definition with typed signature |
| `Channel` | Bounded communication for CSP (pipeline depth) |
| `Schedule` | Combinator-style execution strategy |
| `Program` | Compiled executable for target backend |

## Axis Types (From SparseTIR)

| Type | Description | Example |
|------|-------------|---------|
| `Dense<N>` | Static size | `Dense<8>` for 8 heads |
| `DenseDyn` | Dynamic size | `DenseDyn(batch_size)` |
| `Ragged` | Variable per-element | `Ragged(n, seq_lens)` |
| `Sparse` | CSR format | `Sparse(n, indptr, indices)` for MoE |

## Problem Statement

LLM inference dynamism (from [requirements.md](requirements.md)):

| Pattern | Example | v8 Support |
|---------|---------|------------|
| Variable KV length | 512-32K per batch | `DenseDyn`, `Ragged` axes |
| TopK Attention | DeepSeek Lightning | `cond()` for tiered kernels |
| MoE Routing | Dynamic expert selection | `Sparse` axis + `select()` |
| Megakernels | Loader→Computer→Storer | CSP `process` + `channel` |
| Human-in-the-loop | Custom scheduling | `dispatch_by()`, `stream_by()` |

## Implementation Status

The CPU simulation prototype is implemented with the following components:

| Component | Header | Description |
|-----------|--------|-------------|
| **Core Types** | `include/pto/rt/types.hpp` | Axis types, Task, Tensor, dependencies |
| **Workload** | `include/pto/rt/workload.hpp` | `Workload<Axes, Task, Deps>` template |
| **Primitives** | `include/pto/rt/primitives.hpp` | `parallel_for`, `for_each`, `select`, `cond`, `task` |
| **Schedule** | `include/pto/rt/schedule.hpp` | `Schedule`, `DispatchPolicy`, `IssuePolicy`, `TimingPolicy` |
| **CPU Simulation** | `include/pto/rt/cpu/simulation.hpp` | `Stream`, `Event`, `Program`, `KernelRegistry` |
| **CSP** | `include/pto/rt/csp.hpp` | `Channel`, `Process`, `consume`, `connect`, `replicate` |

### Examples

| Example | Location | Description |
|---------|----------|-------------|
| **DeepSeek Attention** | `examples/deepseek_v3/attention.cpp` | Variable-length attention with tiered kernels |
| **DeepSeek MoE** | `examples/deepseek_v3/moe.cpp` | Sparse expert routing with `select()` |
| **DeepSeek Transformer** | `examples/deepseek_v3/transformer.cpp` | Full layer composition |
| **DeepSeek Validation** | `examples/deepseek_v3/validation.cpp` | Real-data validation against golden refs |
| **Megakernel Pipeline** | `examples/megakernel/attention.cpp` | CSP pipeline with Loader→Computer→Storer |

### Tests

Run tests with:
```bash
python3 tests/run_cpu.py --testcase rt --verbose
```

Test files:
- `tests/cpu/st/testcase/rt/types_test.cpp` - Type system tests
- `tests/cpu/st/testcase/rt/primitives_test.cpp` - Primitive tests
- `tests/cpu/st/testcase/rt/schedule_test.cpp` - Schedule/execution tests
- `tests/cpu/st/testcase/rt/csp_test.cpp` - CSP primitives tests
- `tests/cpu/st/testcase/rt/deepseek_test.cpp` - DeepSeek verification tests
- `tests/cpu/st/testcase/rt/megakernel_test.cpp` - Pipeline integration tests

### Real-Data Validation

Golden reference validation against NumPy implementations:

```bash
cd examples/deepseek_v3
python3 gen_data.py      # Generate input tensors
python3 gen_golden.py    # Compute golden references
g++ -std=c++23 -O2 -I../../include -o validation validation.cpp
./validation             # Run validation (5/5 tests pass)
```

Validation test results:

| Test | Tasks | Max Error | Status |
|------|-------|-----------|--------|
| Attention | 32 | 4.66e-09 | PASS |
| RMSNorm | 200 | 9.54e-07 | PASS |
| FFN | 200 | 4.00e-11 | PASS |
| MoE | 800 expert calls | 3.27e-11 | PASS |
| RT Integration | 12 | - | PASS |

## Getting Started

1. **Understand the problem**: Read [requirements.md](requirements.md)
2. **See design decisions**: Read [design-questions.md](design-questions.md)
3. **Learn the design**: Read [analysis.md](analysis.md)
4. **See the API**: Read [spec.md](spec.md)
5. **Run examples**: See `examples/deepseek_v3/` and `examples/megakernel/`

## Reference Materials

| Research Note | Key Insights |
|---------------|--------------|
| [01_flashinfer.md](reference/01_flashinfer.md) | Plan-Run model, index-based dispatch |
| [02_gpu_patterns.md](reference/02_gpu_patterns.md) | Triton, CuTe, fusion strategies |
| [03_pl_design.md](reference/03_pl_design.md) | Halide algorithm-schedule separation |
| [04_pto_isa.md](reference/04_pto_isa.md) | DYNAMIC pattern, extension points |
| [05_ascend_hw.md](reference/05_ascend_hw.md) | AICPU latency model (critical!) |
| [06_megakernels.md](reference/06_megakernels.md) | Megakernels: warp specialization, pipelining |
| [07_cuda_streams_workstealing.md](reference/07_cuda_streams_workstealing.md) | Streams/events, Whippletree |
| [08_tvm.md](reference/08_tvm.md) | TVM: tensor expression, schedule primitives |
| [09_sparsetir.md](reference/09_sparsetir.md) | SparseTIR: composable formats, axis types |
| [10_relax.md](reference/10_relax.md) | Relax: symbolic shapes, cross-level |
| [13_pto_isa_lh.md](reference/13_pto_isa_lh.md) | PTO-ISA-LH: task graph runtime |
| [14_pto_isa_wc.md](reference/14_pto_isa_wc.md) | PTO-ISA-WC: work-stealing patterns |

## Version History

| Version | Focus | Status |
|---------|-------|--------|
| v1 | Static task graphs | Archived |
| v2 | Dynamic workload patterns | Archived |
| v3 | Plan-Descriptor-Execute | Archived |
| v4 | AICPU-based planning, tiered dispatch | Archived |
| v5 | Task-Event model, IssuePolicy enums | Archived |
| v6 | Programmable runtime, event-driven handlers | Archived |
| v7 | Workload-Schedule separation, CSP (separate) | Archived |
| v8 | Typed workload, two parallelism modes, declarative CSP | Archived |
| **v9.3** | **Python frontend: `@workload` + `P` namespace, direct kernel calls, backend applicability** | **Current** |

## References

- [PTO-ISA Overview](../PTOISA.md)
- [PTO Programming Model](../coding/ProgrammingModel.md)
- [PyPTO Runtime](https://gitcode.com/cann/pypto/)
- [Megakernels Paper](../../references/megakernels.pdf)
- [DAM Paper](../../references/dam.pdf)
- [Halide Paper](https://halide-lang.org/)
- [TVM Paper (OSDI 2018)](https://www.usenix.org/conference/osdi18/presentation/chen)
- [SparseTIR Paper (ASPLOS 2023)](https://doi.org/10.1145/3582016.3582047)
- [Relax Paper (ASPLOS 2025)](https://doi.org/10.1145/3676641.3716249)
