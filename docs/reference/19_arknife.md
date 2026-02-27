# Reference Notes: Arknife (explicit hardware abstraction)

Local clone (gitignored): `references/arknife/`.

Arknife is a Python-based kernel codegen framework whose most useful lesson for PTO‑WSP v10 is its **explicit hardware
abstraction layer** and how that layer cleanly separates:

- “what the program means” (op graph + axis semantics), from
- “where it runs” (hardware hierarchy + memory spaces), from
- “how it emits code” (instruction/channel bindings + code generator).

## 1) What to read first

- `references/arknife/python/arknife/hardware.py` (hardware hierarchy + memory spaces)
- `references/arknife/python/arknife/instruction.py` (instruction abstraction + binding)
- `references/arknife/python/arknife/channel.py` (channels / synchronization abstraction)
- `references/arknife/python/arknife/op_graph.py` (OpGraph / Tensor / Operator)
- `references/arknife/python/arknife/functors/` (pass infrastructure; visitor/mutator style)
- `references/arknife/python/arknife/functors/code_generator.py` (codegen entrypoints)
- `references/arknife/docs/` (module docs)

## 2) Hardware model: hierarchy + memory spaces

Arknife models hardware using:

- `Hardware`: a full architecture specification
- `ParLevel`: a parallelism level (grid/block/warp/… style hierarchy)
- `MemSpace`: a memory space owned by a `ParLevel` (capacity/alignment, etc.)

Pointer:
- `references/arknife/python/arknife/hardware.py`

Key lesson for PTO‑WSP v10:
- keep backend targeting based on an explicit `ArchModel` rather than “if backend == X” branching,
- make memory spaces and parallelism levels first-class so emitters can reason about placement and constraints.

## 3) Instruction + channel abstractions

Arknife decouples:

- “instruction semantics” (what tensor access pattern and computation is intended) from
- “hardware implementation” (how to emit the instruction on a specific architecture).

Similarly, it introduces channels to model asynchronous data movement and synchronization between memory spaces.

Pointers:
- `references/arknife/python/arknife/instruction.py`
- `references/arknife/python/arknife/channel.py`

Key lesson for PTO‑WSP v10:
- v10 “kernels” should be modeled as **backend-specific intrinsic programs**, and backend emitters should be registered as
  handlers for intrinsic IDs (with typed contracts), not hard-wired decorators.

## 4) Relevance to PTO‑WSP v10

Arknife is directly relevant to the v10 “Emitter/Backend layer” design:

- **Backend registry** should be grounded in an explicit hardware model (memory/exec/sync/dispatch constraints).
- **Intrinsic mechanism** should support backend-specific lowering while keeping the workload–schedule surface stable.
- **Optional multi-target layer** can reuse the idea that a program is interpreted under different `Hardware` instances and
  bound to different instruction/channel implementations, rather than “forking the compiler”.

Related v10 docs:
- `docs/future/v10_analysis.md`
- `docs/future/v10_compiler_pipeline.md`

