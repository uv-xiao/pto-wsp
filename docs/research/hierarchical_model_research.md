## Hierarchical Workload–Schedule (v9) — recommended model

### Core idea (2 levels, with a clean escape hatch)
- **Level 0 (Outer / CPU+AICPU):** schedule a *task graph* (`Workload IR`) onto **executors** (CPU threads, AICPU dispatchers, device streams).
- **Level 1 (Inner / NPU InCore):** schedule a *kernel body* (`NPUFunction IR`) onto **memory hierarchy + pipelines** (GM/L2/UB/L1/L0, async copies, double-buffering, compute/MTE overlap).

This matches `pto-isa-lh`: “orchestration builds tasks; InCore functions define kernels”, and it matches the Ascend A2/A3 model in `references/pto-isa-lh/update.md` (vector vs cube cores + MTE/compute pipe overlap).

---

## 1) How deep should the hierarchy go? (2 vs 3 vs arbitrary)

### Recommendation: **2 levels as the *first-class* IR contract**
- It’s the minimum that captures the real split: **graph scheduling** vs **kernel micro-scheduling**.
- It keeps v9 IR/codegen sane: outer passes don’t need to understand UB/L1; inner passes don’t need to understand global fanin/fanout.

### Escape hatch for “3rd level” needs without redesign
You’ll eventually want concepts like:
- multi-device / multi-chip placement,
- per-core “substreams” / instruction bundles,
- architecture-specific microcode scheduling.

Don’t add a hard 3rd level now. Instead:
- Treat **device placement** as an *outer schedule refinement* (e.g., `spatial_map`, `layout`, `dispatch_by(device_id)`).
- Treat **microcode** as *backend lowering detail* of the inner kernel schedule (e.g., Ascend flags/barriers, AIE ping-pong DMA).

Net: **2-level IR**, but **arbitrary lowering depth** inside a backend.

---

## 2) How does the outer schedule reference inner NPU functions?

### Recommendation: tasks reference kernels via a **stable handle**, not strings
- In outer `TaskNode`, keep `kernel_name` for readability, but make the real linkage a `KernelId/KernelHandle`.
- The handle resolves to a **KernelDef** stored in a module-level kernel registry:
  - signature (inputs/outputs/resources),
  - traits (e.g., `is_cube`, SRAM budget),
  - inner IR (`NPUFunction`) + its schedule hints,
  - per-backend artifacts (compiled binary, cost model).

Concretely:
- **Workload IR:** `TaskNode(kernel_ref=KernelId, resources=[TensorRegion...], params=[...])`
- **Schedule IR (outer):** can dispatch/stream/timing based on *task-visible kernel traits*, e.g.:
  - `dispatch_by(lambda t: t.kernel.is_cube)`
  - `stream_by(lambda t: t.kernel.name_hash)`
  - `timing` policies using `t.kernel.estimated_cycles`

This avoids “outer schedule directly embedding inner schedule”, but still lets outer scheduling *use* inner metadata.

---

## 3) How does the inner NPU schedule specify tile size, double buffering?

### Recommendation: inner kernel IR = “functional ops” + “schedule annotations”
Keep two layers inside a kernel:
1) **Functional IR (portable):** `tile/memref decl`, `load/store`, `compute ops`, `loops/if`
2) **Schedule annotations (backend-meaningful):**
   - `tile_policy`: tm/tn/tk (or named tile shapes)
   - `memory_placement`: where each buffer lives (UB/L1/L0C/etc)
   - `double_buffer(target, depth=2)`
   - `pipeline(loop=..., stages=..., overlap=[load, compute, store])`

This is exactly the shape you already sketched in `docs/npu-design.md` (`ScheduleHints` with `tile_policy`, `double_buffer`, `pipeline`).

### Important detail: double buffering must compile to *explicit* async + sync
To be backend-portable, the inner IR should support either:
- **explicit ops**: `load_async(tag)`, `wait(tag)`, `swap(buf0, buf1)`; OR
- **directive lowering**: `.double_buffer("q")` expands into ping/pong buffers + async copy + wait + barriers.

For Ascend A2/A3 specifically, lowering should target the model in `references/pto-isa-lh/update.md`:
- overlap **MTE** (GM↔UB/L1) with **compute**,
- use flags/barriers (`SET_FLAG`, `WAIT_FLAG`, `PIPE_BARRIER`) to represent pipeline stage boundaries.

Tile size selection:
- prefer **kernel variants** (`attn_32x128`, `attn_64x128`) when hardware constraints are tight (SRAM budgets, alignment),
- allow symbolic tile sizes only if the backend can truly specialize at compile time.

---

## 4) Should we consider AICore cube vs vector distinction (`is_cube`)?

### Recommendation: **yes, but make it a kernel trait with 3 states**
Instead of only `bool is_cube`, use something like:
- `ExecUnit::VectorOnly`
- `ExecUnit::CubeOnly`
- `ExecUnit::Either` (rare, but useful for “vector matmul” fallbacks)

Then:
- infer it automatically (e.g., any `.matmul(use_cube=True)` ⇒ `CubeOnly`),
- allow explicit override per kernel or per op.

Why this matters:
- Outer scheduling must route tasks correctly (your `pto-isa-lh` update shows dual-queue routing depends on `is_cube`).
- Backends differ: CPU sim ignores it; Ascend uses it for AIV/AIC routing; AIE may map it to different kernel implementations.

---

## 5) How does the hierarchical model map to different backends?

### Mapping rule: same IR, different “meaning” of each level
**Outer schedule → executor mapping**
- **CPU sim:** executors = threads; streams = per-thread queues; dispatch = thread selection.
- **Ascend:** executors = AICPU-managed submission to AICores; streams = hardware streams; dispatch/stream_by decide stream + (optionally) core group.
- **AMD AIE:** executors = tiles/columns; `spatial_map` is first-class; streams map to DMA channels / routing.

**Inner schedule → kernel lowering**
- **CPU sim:** interpret functional ops (NumPy/Torch); ignore buffering/pipeline hints (or use them only for a cost model).
- **Ascend:** lower to PTO-ISA-ish ops + AICore code; schedule hints become MTE/compute overlap + flags/barriers; `ExecUnit` selects vector vs cube core path (A2/A3 model).
- **AMD AIE:** lower to AIE kernel code; tile sizes bind to local memory shapes; double buffering becomes ping/pong buffers + DMA; pipeline stages map to software pipeline in the tile.

---

If you want, I can translate this into concrete IR diffs against `docs/ir-design.md` (add `KernelDef`, `KernelSchedule`, `kernel_ref` on `TaskNode`, and a `.pto` text format for kernel defs) in a follow-up.