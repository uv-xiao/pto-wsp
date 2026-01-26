# Research Report 13: pto-isa-lh Analysis

## Overview

**Source**: `references/pto-isa-lh/`
**Purpose**: Python builder API, task graph runtime, CPU simulation, multi-backend code generation
**v9 Relevance**: Primary reference for Python frontend design and task graph data structures

---

## 1. Architecture Overview

```
                    PTO Module (Python DSL)
                            │
            ┌───────────────┼───────────────┐
            ▼               ▼               ▼
      InCore Functions  Orchestration   InCore Functions
      (Platform-Specific) (Platform-Agnostic)  (Platform-Specific)
            │               │               │
       ┌────┴────┐          │          ┌────┴────┐
       ▼         ▼          ▼          ▼         ▼
    ARM64     CUDA      Pure C Code  Ascend    ...
    NEON      Kernels   (Task Graph   A2/A3
    (.c)      (.cu)     Runtime)      (.cpp)
                        (.c)
```

### 1.1 Key Design Principle: InCore vs Orchestration

| Aspect | InCore Function | Orchestration Function |
|--------|-----------------|------------------------|
| **Declaration** | `.in_core()` | `.not_in_core()` |
| **Execution** | Compute cores (GPU/NPU/SIMD) | Host CPU |
| **Physical ISA** | CUDA / Ascend C / ARM64 NEON | Pure C (Task Graph) |
| **Content** | Tile-level tensor operations | Control flow + CALL InCore |
| **Nesting** | Inlined at compile time | Generates task scheduling code |
| **Memory Model** | Core-local tile buffers | Global memory pointers |

**Critical Insight**: The InCore/Orchestration separation is analogous to our v8 Task/Schedule separation:
- **InCore Function** ≈ **v8 Task** (kernel execution unit)
- **Orchestration Function** ≈ **v8 Workload + Schedule** (task generation + scheduling policy)

---

## 2. Python Builder API (PTOFunctionBuilder)

### 2.1 Fluent API Pattern

```python
from pto_compile import PTOFunctionBuilder, PTOModule

module = PTOModule("my_module")

# InCore function (runs on compute cores)
builder = PTOFunctionBuilder("rowmax", module=module)
program = (builder
    .in_core()                                    # Mark as InCore
    .memref("input", MemorySpace.GLOBAL, ElementType.F32)
    .memref("output", MemorySpace.GLOBAL, ElementType.F32)
    .tile("x", 8, 8, ElementType.F32)
    .tile("result", 8, 1, ElementType.F32)
    .load("x", "input")                           # TLOAD
    .rowmax("result", "x")                        # TROWMAX
    .store("output", "result")                    # TSTORE
    .build())
```

### 2.2 Key Methods

| Category | Methods | Description |
|----------|---------|-------------|
| **Declaration** | `.tile()`, `.scalar()`, `.memref()` | Declare variables |
| **Mode** | `.in_core()`, `.not_in_core()`, `.cube()` | Function properties |
| **Memory** | `.load()`, `.store()` | Tile ↔ Memory transfer |
| **Control** | `.for_loop()`, `.end_for()`, `.if_then()`, `.endif()` | Control flow |
| **Compute** | `.add()`, `.mul()`, `.matmul()`, `.rowmax()`, etc. | Tile operations |
| **Call** | `.call()` | Call other functions |

### 2.3 CALL Instruction with Offset Parameters

The CALL instruction supports two argument formats:

```python
# Format 1: Simple (no offset)
.call("func_name", {"param": "tensor_name"})

# Format 2: With offset (for dynamic tiling)
.call("func_name", {"param": ("tensor_name", "row_offset_var", col_offset_int)})
```

**Example**: Flash Attention with cross-tile dependencies:
```python
.for_loop("q_tile", 0, "num_tiles", 1)
    .for_loop("kv_tile", 0, "num_tiles", 1)
        .call("flash_attn_score_block", {
            "input_q": ("all_q_rope", "q_tile", 0),   # Q[q_tile]
            "input_k": ("all_k_rope", "kv_tile", 0),  # K[kv_tile] - cross dependency
            "output_s": ("temp_scores", "q_tile", 0)
        })
    .end_for()
.end_for()
```

**Significance for v9**: Different tiles use different offsets, enabling the runtime to identify true parallelism opportunities.

---

## 3. ISA Definition (pto_isa_definition.py)

### 3.1 Type System

```python
class ElementType(Enum):
    F32 = "f32"
    F16 = "f16"
    F64 = "f64"
    BF16 = "bf16"
    I32 = "i32"
    I8 = "i8"
    # ... etc

class MemorySpace(Enum):
    GM = "gm"    # Global Memory
    L2 = "l2"    # L2 Cache
    UB = "ub"    # Unified Buffer (Ascend)
    L0A = "l0a"  # Cube A buffer
    L0B = "l0b"  # Cube B buffer
    L0C = "l0c"  # Cube C buffer
```

### 3.2 Instruction Categories

| Category | Instructions | Notes |
|----------|--------------|-------|
| **Data Movement** | TLOAD, TSTORE | Tile ↔ Memory |
| **Arithmetic** | TADD, TSUB, TMUL, TDIV | Elementwise |
| **Matrix** | TMATMUL, TMATMUL_ACC | Matrix multiplication |
| **Reduction** | TROWSUM, TROWMAX, TCOLSUM | Row/column reduction |
| **Activation** | TEXP, TLOG, TSQRT, TSILU | Nonlinear |
| **Broadcast** | TROWEXPANDSUB, TROWEXPANDDIV | Row broadcast ops |
| **Scalar** | SADD, SMUL, SLI, SCMP | Scalar operations |
| **Control** | FOR, ENDFOR, IF, ENDIF, CALL, RETURN | Control flow |

### 3.3 Multi-Backend Code Generation

Each instruction has backend-specific code generators:

```python
@dataclass
class PTOInstruction(ABC):
    @abstractmethod
    def codegen_arm64(self, ctx: ARM64CodeGenContext) -> str: ...
    @abstractmethod
    def codegen_cuda(self, ctx: CUDACodeGenContext) -> str: ...
    @abstractmethod
    def codegen_ascend(self, ctx: AscendCodeGenContext) -> str: ...
```

---

## 4. Task Graph Runtime (pto_runtime.h/c)

### 4.1 Core Data Structures

```c
typedef struct {
    void*    raw_tensor;     // Base pointer to tensor data
    int64_t  row_offset;     // Row offset within tensor
    int64_t  col_offset;     // Column offset
    int64_t  rows;           // Region rows
    int64_t  cols;           // Region cols
} TensorRegion;

typedef struct {
    TensorRegion region;
    bool         is_output;
} TaskArg;

typedef struct {
    int32_t      task_id;
    const char*  func_name;
    void*        func_ptr;
    TaskArg      args[16];
    int32_t      num_args;

    // Dependency tracking
    int32_t      fanin;              // Input dependencies remaining
    int32_t      fanout[512];        // Downstream task IDs
    int32_t      fanout_count;

    // Status
    bool         is_active;
    bool         is_complete;
    bool         is_cube;            // Requires cube unit (matmul)

    // Buffer estimation
    int32_t      buffer_size_bytes;
    int32_t      buffer_size_with_reuse;
} PendingTask;

typedef struct {
    PendingTask       pend_task[524288];    // Task table
    TensorMapEntry*   tensor_map[16384];    // Producer lookup
    int32_t           ready_queue[262144];  // Tasks with fanin==0
    int32_t           next_task_id;
    int64_t           total_tasks_scheduled;
    int64_t           total_tasks_completed;

    // Thread synchronization
    pthread_mutex_t   queue_mutex;
    pthread_cond_t    queue_not_empty;
    pthread_cond_t    all_done;
} PTORuntime;
```

### 4.2 Task Graph Construction Flow

```
Orchestration Function Execution
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│  for (tile_idx = 0; tile_idx < num_tiles; tile_idx++) {    │
│      // Each CALL instruction becomes:                      │
│      task_id = pto_task_alloc(rt, "rowmax");               │
│      pto_task_add_input(rt, task_id, input, offset);       │
│      pto_task_add_output(rt, task_id, output, offset);     │
│      pto_task_submit(rt, task_id);  // Auto dependencies   │
│  }                                                          │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
   Task Graph Complete
   (pend_task[], fanin/fanout, ready_queue)
```

### 4.3 Dependency Tracking via TensorMap

- **pto_task_add_output()**: Registers `(tensor, offset, shape) → producer_task_id` in TensorMap
- **pto_task_add_input()**: Looks up producer in TensorMap, increments consumer's fanin
- **pto_task_complete()**: Decrements fanin of downstream tasks, adds newly-ready to queue

### 4.4 Performance: LLaMA 7B Statistics

| SeqLen | Tiles | Tasks | Build Time | Tasks/ms |
|--------|-------|-------|------------|----------|
| 1K | 32 | 1,024 | 0.14 ms | 7,111 |
| 4K | 128 | 13,312 | 1.7 ms | 7,849 |
| 8K | 256 | 51,200 | 8.5 ms | 6,021 |
| 16K | 512 | 200,704 | 34.2 ms | 5,868 |

**Key Insight**: Task graph construction overhead is 2-5% of total execution time, acceptable for dynamic shapes.

---

## 5. Binary Expansion for Dynamic Loops

### 5.1 Concept

For loops with dynamic bounds, binary expansion converts them into fixed-size blocks:

```python
.for_loop("i", 0, "n", 1, max_range=4096, min_range=256)
    .call("process_tile", {...})
.end_for()
```

### 5.2 Generated Code

```c
{
    int _i_base = 0;
    int _i_residual = n & 255;     // < min_range
    int _i_quantized = n - _i_residual;

    // Power-of-2 blocks
    if (_i_quantized & 4096) { for (i = _i_base; i < _i_base + 4096; i++) { BODY } _i_base += 4096; }
    if (_i_quantized & 2048) { ... }
    if (_i_quantized & 1024) { ... }
    // ... down to min_range

    // Residual
    if (_i_residual > 0) { for (i = _i_base; i < _i_base + _i_residual; i++) { BODY } }
}
```

### 5.3 Adaptive Tile Sizing

```python
TILE_ROWS_BY_LEVEL = {
    4096: 64,   # Large blocks: 64-row tiles
    2048: 64,
    1024: 64,
    512:  64,
    256:  64,   # Min quantized block
    0:    32,   # Residual: 32-row tiles (finer granularity)
}

.for_loop("i", 0, "n", 1, max_range=4096, min_range=256, tile_levels=TILE_ROWS_BY_LEVEL)
```

**Optimization Effect**: For LLaMA 7B at 8K sequence:
- Without optimization: 200,704 tasks
- With optimization: 51,200 tasks (**75% reduction**)

---

## 6. Assembly Format (.pto)

### 6.1 Syntax

```
// PTO Module: softmax_module
// Entry: @dynamic_softmax

// Function Type: InCore
func @rowmax(%input: !pto.memref<gm,...,f32>, %output: !pto.memref<gm,...,f32>) {
  %x = alloc_tile : !pto.tile<8x8xf32>
  %result = alloc_tile : !pto.tile<8x1xf32>

  %x = tload %input[0, 0] : (!pto.memref) -> !pto.tile<8x8xf32>
  %result = trowmax %x : !pto.tile<8x8xf32> -> !pto.tile<8x1xf32>
  tstore %result, %output[0, 0]

  return
}

// Function Type: Orchestration
func @dynamic_softmax(%input: !pto.memref, %output: !pto.memref) {
  %num_tiles = alloc_scalar : i32

  FOR %tile_idx:idx, 0:idx, %num_tiles:idx, 1:idx max_range=4096 min_range=256
    CALL @rowmax(%input -> (%input, tile_idx, 0), %output -> (%temp_max, tile_idx, 0))
  ENDFOR

  return
}
```

### 6.2 Bidirectional Conversion

```
PTOFunctionBuilder  ←→  .pto Assembly
   (Python DSL)            (Text File)
        │                       │
        │  compile()            │  parse()
        ▼                       ▼
   PTO Assembly         Python Builder Code
```

---

## 7. Mapping to v8/v9 Concepts

| pto-isa-lh | v8 Runtime Extension | v9 Target |
|------------|---------------------|-----------|
| InCore Function | Task | Task (kernel) |
| Orchestration Function | Workload + Schedule | Workload + Schedule |
| Task Graph (PendingTask) | Compiled Schedule | IR + Compiled Schedule |
| TensorMap | Dependency Inference | Dependency Inference |
| `pto_task_submit()` | `schedule.compile()` output | IR-based compilation |
| PTOFunctionBuilder | - (C++ templates) | Python WorkloadBuilder |
| .pto Assembly | - | Assembly format |

---

## 8. Key Takeaways for v9

### 8.1 What to Learn

1. **Python Builder Pattern**: Fluent API with method chaining is effective
2. **InCore/Orchestration Split**: Clean separation enables platform-agnostic scheduling
3. **Task Graph Data Structure**: `fanin/fanout` edges are simple and effective
4. **TensorMap for Dependencies**: Hash-based region tracking works well
5. **Binary Expansion**: Useful optimization for dynamic loops (optional for v9)
6. **Assembly Format**: Text representation enables tooling and debugging

### 8.2 What to Preserve (Our Strengths)

1. **Typed Workloads**: `Workload[Axes, Task, Deps]` - stronger typing than pto-isa-lh
2. **Declarative Primitives**: `parallel_for`, `for_each`, `select`, `cond` - not imperative loops
3. **CSP Pipeline Parallelism**: `Channel`, `Process`, `consume` - not present in pto-isa-lh
4. **Schedule as Separate Entity**: Our schedule is explicit, not mixed with workload

### 8.3 Gaps to Address in v9

1. **Python Frontend**: Need `WorkloadBuilder` and `ScheduleBuilder` classes
2. **Assembly Format**: Need `.pto` format for workload + schedule
3. **IR Layer**: Need C++ IR between Python and backend code
4. **Multi-Backend Codegen**: Learn from their backend architecture

---

## 9. Open Questions

1. **Workload vs Orchestration**: Should v9 Workload map directly to Orchestration Function?
2. **Schedule Representation**: How to express Schedule in Python/Assembly?
3. **Type Safety**: How to preserve v8's compile-time guarantees in Python DSL?
4. **CSP Integration**: How to express Channels and Processes in the builder API?

---

## 10. Recent Updates (January 2026)

### 10.1 A2A3 Core Model Simulator

A cycle-accurate simulator for Ascend A2/A3 cores was added:

**Location**: `src/runtime/ascend_a2a3_core_model/`

**Architecture**:
```
CUBE CORE                              VECTOR CORE
┌────────────────────────┐            ┌────────────────────────┐
│  Scalar | MTE Pipes    │            │  Scalar | MTE Pipes    │
│         | GM↔L1, L0C   │            │         | GM↔UB        │
│  ───────┴──────────    │            │  ───────┴──────────    │
│       CUBE Unit        │            │      Vector Unit       │
│    (Matrix Multiply)   │            │  (Elem-wise, Reduce)   │
└────────────────────────┘            └────────────────────────┘
```

**Features**:
- Parallel pipe execution model (Scalar, MTE, Compute)
- Synchronization primitives: SET_FLAG, WAIT_FLAG, PIPE_BARRIER
- Instruction parsing and cycle estimation
- InCore function registration and cached simulation
- Heuristic cycle cost API for runtime integration

**Integration**:
```c
// Automatic integration via pto_estimate_cycle_cost():
int64_t cycles = pto_estimate_cycle_cost("rmsnorm_tile");  // Uses core sim if available
```

### 10.2 Dual-Queue Simulation

Tasks are now correctly routed to vector vs cube workers:

```c
if (rt->dual_queue_mode) {
    if (task->is_cube) {
        // Cube task → assign to cube workers (48-71)
        worker_start = rt->num_vector_workers;
        worker_end = NUM_WORKERS;
    } else {
        // Vector task → assign to vector workers (0-47)
        worker_start = 0;
        worker_end = rt->num_vector_workers;
    }
}
```

**Task Routing**:
| Function | Contains Matmul | `is_cube` | Worker Type |
|----------|----------------|-----------|-------------|
| `rmsnorm_tile` | No | false | Vector (0-47) |
| `tile_matmul` | Yes | true | Cube (48-71) |
| `rope_tile` | No | false | Vector (0-47) |
| `attention_score_tile` | Yes | true | Cube (48-71) |

### 10.3 Enhanced Chrome Tracing

Trace export now distinguishes vector vs cube workers:
- `pid=0` for Vector Workers, `pid=1` for Cube Workers
- Thread name metadata (`Vector-0`, `Cube-0`, etc.)
- Process name metadata (`Vector Workers (48)`, `Cube Workers (24)`)

### 10.4 Auto-Detection of Cube Operations

Operations using `.matmul()` automatically set `is_cube = True`:

```python
def _make_matmul_method(instr_class, doc):
    def method(self, dst: str, a: str, b: str) -> "PTOFunctionBuilder":
        self.program.is_cube = True  # Auto-set
        self._add_instr(instr_class(...))
        return self
    return method
```

### 10.5 Sliding Window Task Management

- Window size: `PTO_TASK_WINDOW_SIZE = 8K`
- Task indices wrap around using `PTO_TASK_SLOT(task_id)`
- Overflow handling varies by runtime mode (BENCHMARK_ONLY, DUMP_GRAPH, EXECUTE/SIMULATE)

### 10.6 TensorMap Optimization

- Replaced simple hash with MurmurHash-style function
- Added memory pool for TensorMapEntry allocations
- Improved throughput from 3.5M to 3.77M tasks/ms

### 10.7 SRAM Constraint Analysis

Flash Attention memory layout per block:
```
Q:  tile_rows × head_dim
K:  tile_rows × head_dim
V:  tile_rows × head_dim
S:  tile_rows × tile_rows  ← QUADRATIC!
O:  tile_rows × head_dim
```

| Tile Rows | Q/K/V | S (Score) | Total | Fits 256KB? |
|-----------|-------|-----------|-------|-------------|
| 32 | 16KB each | 4KB | 68KB | ✓ |
| **64** | 32KB each | **16KB** | **144KB** | ✓ |
| 128 | 64KB each | **64KB** | 321KB | ✗ |

**Implication**: With 256KB SRAM (Ascend 910B), maximum tile size is 64 rows.

---

## References

- `references/pto-isa-lh/README.md` - Comprehensive documentation
- `references/pto-isa-lh/pto_compile.py` - Python builder implementation
- `references/pto-isa-lh/pto_isa_definition.py` - ISA instruction definitions
- `references/pto-isa-lh/pto_parser.py` - Assembly parser
- `references/pto-isa-lh/pto_runtime.h/c` - Task graph runtime
- `references/pto-isa-lh/pto_dynamic_tiling.py` - Dynamic tiling utilities
- `references/pto-isa-lh/src/runtime/ascend_a2a3_core_model/` - A2A3 core simulator
- `references/pto-isa-lh/new_updates.md` - January 20, 2026 updates
- `references/pto-isa-lh/update.md` - Update log (A2A3 simulator, dual-queue)
