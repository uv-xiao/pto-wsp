# HTP (Heterogeneous Tile Programming) — Examples (E2E)

This file provides end-to-end examples across:

- kernel programs
- megakernel / dataflow pipelines
- serving routine orchestration
- multiple backends (PTO simulation/device; AIE/MLIR-AIE)

The code is illustrative pseudo-Python: it prioritizes semantics and contracts over exact API names.

---

## Example 1 — Kernel + WSP schedule to PTO backend

Goal: write a tile kernel and schedule it for the Ascend/PTO backend.

```python
from htp import kernel, workload, schedule, Tile, F16, In, Out, Layout
from htp.intrinsics import pto

M = 1024
N = 1024

@kernel
def add_tile(a: In[Tile[16, 16, F16]], b: In[Tile[16, 16, F16]], c: Out[Tile[16, 16, F16]]):
    c[:] = pto.add(a, b)

@workload
def add(A, B, C):
    for i in P(0, M // 16):            # logical parallel loops
        for j in P(0, N // 16):
            add_tile(A.tile(i, j), B.tile(i, j), C.tile(i, j))

@schedule(add)
def add_sched(s):
    s.tile("i", 4)                     # schedule directives
    s.pipeline(depth=2)
    s.buffer("A", space="UB")          # hardware/memory facet constraint
    s.buffer("B", space="UB")
    s.emit(target="pto-a2a3sim")

pkg = add.compile(out="out/add", target="pto-a2a3sim")
prog = htp.bind(pkg).run()
```

What this demonstrates:

- WSP separation: `@workload` declares logical work; `@schedule` constrains mapping/buffering/pipelining.
- Intrinsic typing: `pto.add` is only legal in PTO-capable backends.
- Artifact-first: `out/add/` contains manifest + IR dumps + PTO artifacts.

---

## Example 2 — CSP pipeline megakernel: load → compute → store

Goal: create a pipeline-parallel program with bounded channels; type checking prevents protocol mismatch.

```python
from htp import process, Channel, Event, connect, consume

l2c = Channel("l2c", depth=2)          # bounded FIFO
c2s = Channel("c2s", depth=2)
done = Event("done", depth=0)          # rendezvous

loader = (
  process("loader")
    .produces(l2c)
    .body(for_each(batch, lambda b:
        send(l2c, load_tiles(b)))))

computer = (
  process("computer")
    .consumes(l2c)
    .produces(c2s)
    .body(consume(l2c, lambda tiles:
        send(c2s, fused_attention(tiles))))))

storer = (
  process("storer")
    .consumes(c2s)
    .body(consume(c2s, lambda out:
        store_tiles(out)))))

pipeline = connect([loader, computer, storer], [l2c, c2s])

pkg = pipeline.compile(out="out/attn_pipe", target="pto-a2a3sim")
htp.bind(pkg).run()
```

What this demonstrates:

- CSP constructs remain first-class, not “lowered arrays”.
- Channels have static types and capacity; effects can enforce put/get consistency.
- A backend can choose how to implement channels (simulation vs device).

---

## Example 3 — AIE backend: spatial mapping + FIFOs

Goal: map a kernel region to an AIE tile grid with streams, emitting MLIR-AIE artifacts.

```python
from htp import df_region, df_kernel, Layout
from htp.targets import aie

P0, P1 = aie.grid(4, 4)
LyA = Layout.dist("S(1)R")   # shard along mesh axis 1, replicate other dim
LyB = Layout.dist("RS(0)")

@df_region
def top(A, B, C):
    @df_kernel(mapping=[P0, P1], args=[A, B, C])
    def gemm(local_A: A @ LyA, local_B: B @ LyB, local_C: C):
        local_C[:] = aie.matmul(local_A, local_B)

pkg = top.compile(out="out/aie_gemm", target="aie-xdna2")
```

Emitted artifacts:

- `codegen/aie/aie.mlir` — AIE compute + FIFO wiring
- `codegen/aie/mapping.json` — tile mapping + layout placement
- `codegen/aie/host.*` — host-side glue for runtime invocation

What this demonstrates:

- AIE is an “external compilation island”: AST passes produce MLIR-AIE.
- Layout annotation influences sharding/placement and FIFO decisions.

---

## Example 4 — Serving routine: multi-backend build + runtime selection

Goal: compile once into multiple packages, then choose backend at deployment time.

```python
pkg_pto = model.compile(out="out/model_pto", target="pto-a2a3")
pkg_aie = model.compile(out="out/model_aie", target="aie-xdna2")

if platform.has_ascend():
    htp.bind(pkg_pto).run(inputs)
else:
    htp.bind(pkg_aie).run(inputs)
```

Key point: the “compile → package” boundary enables deployment-time backend choice without reauthoring the model.

---

## Example 5 — Extending the compiler: custom pass + pipeline

Goal: show how an extension author adds a pass and wires a pipeline without editing the core compiler.

```python
from htp import register_pass, register_pipeline, Capability

@register_pass(
  name="normalize_layout_v1",
  requires={Capability("IR.ASTCanonical")},
  provides={Capability("Type.LayoutNormalized")}
)
def normalize_layout(ast, ctx):
    return ast  # match/apply rewrites + metadata normalization

@register_pipeline(
  name="pto_debug",
  target="pto-a2a3sim",
  passes=[
    "ast_canonicalize",
    "normalize_layout_v1",
    "typecheck_layout_effects",
    "lower_pto",
    "emit_pto_package"
  ]
)
def pto_debug_pipeline():
    ...

pkg = program.compile(out="out/debug", target="pto-a2a3sim", pipeline="pto_debug", dump_ir=True)
print(pkg.manifest_path)  # out/debug/manifest.json
```

What this demonstrates:

- Passes/pipelines are registered extension points.
- Capability typing prevents invalid pipelines (missing `requires`).
- Debug output is standardized (manifest + pass trace + IR dumps).
