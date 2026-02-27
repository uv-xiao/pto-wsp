# Research Report 15: Allo Framework Analysis

## Overview

**Source**: `references/allo/`
**Purpose**: MLIR-based composable accelerator design, multi-backend targeting (FPGA, AMD AIE)
**v9 Relevance**: Reference for spatial architecture targeting, dataflow primitives, MLIR-based IR design

---

## 1. Framework Overview

### 1.1 Core Characteristics

From the README:
> Allo is a Python-embedded, MLIR-based language and compiler designed to facilitate the modular and composable development of large-scale, high-performance machine learning accelerators.

Key features:
- **Composable Design**: Behavioral + structural composition
- **End-to-End Deployment**: PyTorch → Accelerator generation
- **Multi-Backend Support**: AMD/Intel FPGAs, AMD Ryzen NPUs (AI Engine)

### 1.2 Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Python DSL Layer                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────────┐  │
│  │  allo.dsl    │  │ allo.dataflow│  │  allo.customize (Schedule)   │  │
│  │  (Primitives)│  │  (Dataflow)  │  │  split/reorder/unroll/fuse   │  │
│  └──────────────┘  └──────────────┘  └──────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         MLIR Dialect Layer                               │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │  allo_d (Custom Dialect)                                          │  │
│  │  • StreamConstructOp, StreamGetOp, StreamPutOp                    │  │
│  │  • CreateOpHandleOp, CreateLoopHandleOp                           │  │
│  │  • SplitOp, ReorderOp, UnrollOp, FuseOp                          │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │  Standard Dialects: func, memref, affine, scf, arith             │  │
│  └──────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         Backend Layer                                    │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────────────┐  │
│  │  llvm   │ │   hls   │ │  vitis  │ │   xls   │ │       aie       │  │
│  │ (CPU)   │ │ (Vivado)│ │ (Xilinx)│ │ (Google)│ │ (AMD AI Engine) │  │
│  └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Python DSL Design

### 2.1 Basic Primitives (`allo/dsl.py`)

```python
# Iteration primitives
def grid(*args, name=None, pipeline=False, unroll=False):
    return np.ndindex(*args)

def reduction(*args, name=None, pipeline=False, unroll=False):
    return np.ndindex(*args)

# Compute primitives
def matmul(lhs, rhs, name=None): ...
def add(lhs, rhs, name=None): ...
def exp(x, name=None): ...
def softmax(x, name=None): ...
def layernorm(x, gamma, beta, eps=1e-5): ...
def gelu(x): ...
```

### 2.2 Dataflow Programming (`allo/dataflow.py`)

```python
import allo.dataflow as df
from allo.ir.types import int32, Stream

@df.region()
def top(A: int32[M, N], B: int32[M, N]):
    pipe: Stream[int32, 4]  # Stream with depth 4

    @df.kernel(mapping=[1], args=[A])
    def producer(local_A: int32[M, N]):
        for i, j in allo.grid(M, N):
            out: int32 = local_A[i, j]
            pipe.put(out)  # Send to stream

    @df.kernel(mapping=[1], args=[B])
    def consumer(local_B: int32[M, N]):
        for i, j in allo.grid(M, N):
            data = pipe.get()  # Receive from stream
            local_B[i, j] = data + 1
```

**Key Constructs**:
- `@df.region()`: Top-level dataflow region
- `@df.kernel(mapping=[...])`: Kernel mapped to compute tiles
- `Stream[T, depth]`: FIFO channel between kernels
- `pipe.put()` / `pipe.get()`: CSP-style communication

### 2.3 Schedule API (`allo/customize.py`)

```python
class Schedule:
    def split(self, axis, factor): ...      # Tile loop
    def reorder(self, *args): ...           # Reorder loops
    def unroll(self, axis, factor=0): ...   # Unroll loop
    def fuse(self, *args): ...              # Fuse loops
    def partition(self, target, type, dim, factor): ...  # Array partition
    def pipeline(self, axis, initiation_interval=1): ...
    def buffer_at(self, buffer, axis): ...
    def compute_at(self, func, parent, axis): ...
```

**Similarity to v8**: This is very similar to our Schedule API:
- `split` ↔ Our tiling
- `reorder` ↔ Our loop reordering
- `pipeline` ↔ Our `timing(Pipelined)`
- `partition` ↔ Could map to dispatch strategy

---

## 3. AMD AIE Backend

### 3.1 Target Architecture

AMD AI Engine (AIE) is a **spatial array** architecture:
- 2D array of compute tiles
- Each tile has local memory + compute
- Communication via streams between tiles
- Supported platforms: `XDNA1`, `XDNA2` (AMD Ryzen AI)

### 3.2 Kernel Mapping

```python
@df.region()
def top(A: int32[M, K], B: int32[K, N], C: int32[M, N]):
    @df.kernel(mapping=[P0, P1], args=[A, B, C])  # Map to P0 x P1 grid
    def gemm(
        local_A: int32[M, K] @ LyA,   # Layout annotation
        local_B: int32[K, N] @ LyB,
        local_C: int32[M, N] @ LyC
    ):
        local_C[:, :] = allo.matmul(local_A, local_B)
```

**Key Concepts**:
- `mapping=[P0, P1]`: Maps kernel to a P0 × P1 grid of tiles
- `@ LyA`: Layout annotation for data distribution
- `Layout.Shard(dim)`: Shard data along dimension
- `Layout.Replicate`: Replicate data to all tiles

### 3.3 Layout Annotations

```python
from allo.memory import Layout

S = Layout.Shard
R = Layout.Replicate

# GEMM data distribution
LyA = [S(1), R]      # Shard rows, replicate cols
LyB = [R, S(0)]      # Replicate rows, shard cols
LyC = [S(1), S(0)]   # Shard both dimensions
```

### 3.4 Building for AIE

```python
# Build for AIE with profiling
mod = df.build(
    top,
    target="aie",
    profile=True,
    warmup=200,
    num_iters=1000,
)

# Execute
A = np.random.randint(0, 64, (M, K)).astype(np.int32)
B = np.random.randint(0, 64, (K, N)).astype(np.int32)
C = np.zeros((M, N)).astype(np.int32)
mod(A, B, C)
```

---

## 4. Dataflow Implementation

### 4.1 Stream Operations

From `allo/dataflow.py`:

```python
# Stream primitives
def gather(pipes: list):
    """Collect from multiple pipes in order"""
    raise NotImplementedError("Called in kernel function")

def scatter(buffer, pipes: list):
    """Distribute to multiple pipes in order"""
    raise NotImplementedError("Called in kernel function")

def get_pid():
    """Get process ID within kernel grid"""
    raise NotImplementedError("Called in kernel function")
```

### 4.2 FIFO Management (`allo/backend/aie/mapping.py`)

```python
class FIFO:
    def __init__(
        self,
        name: str,
        src: str,           # Source kernel name
        dst: list[str],     # Destination kernel names
        data_shape: list[int],
        dtype: str,
        depth: int = 2,
        dimensions_to_stream: tuple[list[int], list[int], list[int]] | None = None,
    ):
        self.name = name
        self.src = src
        self.dst = dst
        self.data_shape = list(data_shape)
        self.dtype = str(dtype)
        self.depth = depth
        self.dimensions_to_stream = ...  # DMA scheduling

class FIFOManager:
    def __init__(self):
        self.fifos: list[FIFO] = []
        self.fifo_map: dict[tuple, FIFO] = {}

    def create_fifo(self, src, dst, ...): ...
```

### 4.3 DTensor Tile Management

```python
@dataclass
class DTensorTile:
    dtensor_id: int
    tensor_tile_label: tuple[int | str, ...]

@dataclass(frozen=True)
class PEInterface:
    pe: str               # Kernel name
    interface_idx: int    # Argument index
    layout: tuple[list[int], list[int], list[int]] | None

class DTensorTileGroup:
    """Maps tensor tiles to PEs using them"""
    def __init__(self, order_tag: str):
        self.dtensor_tile_to_pe_interfaces: dict[DTensorTile, list[PEInterface]]
```

---

## 5. External Kernel Support

### 5.1 User-Defined Kernels

Allo supports external C++ kernels using AIE API:

```cpp
// norm.cc
#include <aie_api/aie.hpp>

template <typename T_in, typename T_out, const int SEQ_LEN, const int HIDDEN>
void rms_norm_single_batch(T_in *input, T_in *weight, T_out *output) {
    constexpr int vec_factor = 16;
    using vec_t = aie::vector<T_in, vec_factor>;
    event0();  // Profiling marker

    for (int iter = 0; iter < SEQ_LEN; iter++) {
        // Vectorized computation
        for (int i = 0; i < HIDDEN / vec_factor; i++) {
            vec_t input_vec = aie::load_v<vec_factor>(input_ptr);
            // ...
        }
    }
    event1();  // Profiling marker
}

extern "C" {
    void layer_norm(float A[4][512], float B[512], float C[4][512]) {
        rms_norm_single_batch<float, float, 4, 512>(&A[0][0], B, &C[0][0]);
    }
}
```

### 5.2 Registration in Python

```python
from allo.backend.aie import ExternalModule

norm = ExternalModule(
    top="layer_norm",           # C function name
    impl_path="norm.cc",        # Source file
    input_idx=[0, 1],           # Input argument indices
    output_idx=[2],             # Output argument indices
)

@df.kernel(mapping=[1])
def core(A: Ty[M, N], B: Ty[N], C: Ty[M, N]):
    norm(A, B, C)  # Call external kernel
```

---

## 6. Mapping to v8/v9 Concepts

### 6.1 Correspondence Table

| Allo Concept | v8 Runtime Extension | v9 Target |
|--------------|---------------------|-----------|
| `@df.region()` | Top-level Workload | Workload definition |
| `@df.kernel(mapping=[...])` | Task | Task with spatial mapping |
| `Stream[T, depth]` | `Channel<T, depth>` | Same (CSP channel) |
| `pipe.put()` / `pipe.get()` | `send()` / `receive()` | Same |
| `allo.grid()` | `parallel_for` | Same |
| `allo.reduction()` | Reduction axis | Same |
| `mapping=[P0, P1]` | **NEW** | `spatial_map(grid=(P0, P1))` |
| `@ LyA` (Layout) | **NEW** | `dispatch_by(layout=...)` |
| `df.build(target="aie")` | Backend selection | `backend="aie"` |

### 6.2 Key Insights

1. **Dataflow Alignment**: Allo's `Stream` + `put/get` maps directly to our CSP `Channel` design
2. **Spatial Mapping**: `mapping=[P0, P1]` is what we need for `spatial_map` in v9
3. **Layout Annotations**: `Shard/Replicate` for data distribution across tiles
4. **External Kernels**: Support for hand-optimized C++ kernels (like our InCore functions)

---

## 7. Schedule Primitives Comparison

| Primitive | Allo | v8 Schedule | v9 Extension |
|-----------|------|-------------|--------------|
| Tiling | `split(axis, factor)` | - | `tile_by(...)` |
| Reordering | `reorder(i, j, k)` | - | `reorder_by(...)` |
| Unrolling | `unroll(axis, factor)` | - | `unroll_by(...)` |
| Fusion | `fuse(i, j)` | - | `fuse_by(...)` |
| Pipelining | `pipeline(axis, II)` | `timing(Pipelined)` | Same |
| Dispatch | - | `dispatch_by(fn)` | Same |
| Stream | - | `stream_by(fn)` | Same |
| **Spatial** | `mapping=[P0, P1]` | - | `spatial_map(grid)` |

---

## 8. Key Takeaways for v9

### 8.1 What to Learn

1. **Spatial Mapping Primitive**: `mapping=[P0, P1]` is elegant and expressive
2. **Layout Annotations**: `Shard`/`Replicate` for data distribution
3. **MLIR-Based IR**: Provides good foundation for multi-backend
4. **External Kernel Support**: Important for hand-optimized code paths

### 8.2 New Schedule Primitives for v9

```python
# Proposed v9 spatial schedule extensions
schedule = Schedule(workload)

# Spatial mapping to AIE tile grid
schedule.spatial_map(workload.attention, grid=(4, 4))

# Data layout for spatial distribution
schedule.layout(workload.Q, Shard(dim=0), Replicate())
schedule.layout(workload.K, Replicate(), Shard(dim=1))

# Route channels through spatial array
schedule.route(channel, path=[(0,0), (0,1), (1,1)])
```

### 8.3 Differences from Our Design

| Aspect | Allo | v8/v9 |
|--------|------|-------|
| Base Language | Python + MLIR | Python + C++ |
| Workload Type | Python functions | `Workload[Axes, Task, Deps]` |
| Schedule | Attribute-based | Explicit Schedule object |
| Backend | MLIR lowering | Direct code generation |
| CSP | Streams only | Full CSP (Channel, Process, Event) |

### 8.4 Architecture Decisions

1. **IR Choice**: Allo uses MLIR; v9 could use lightweight C++ IR (simpler)
2. **Spatial Schedule**: Adopt `mapping` concept, but integrate with our Schedule API
3. **External Kernels**: Support similar pattern for hand-optimized InCore functions

---

## 9. Related Publications

1. **Allo (PLDI'24)**: "A Programming Model for Composable Accelerator Design"
2. **Dato (arXiv:2509.06794)**: "A Task-Based Programming Model for Dataflow Accelerators"
3. **ARIES (FPGA'25)**: "An Agile MLIR-Based Compilation Flow for Reconfigurable Devices with AI Engines"

---

## References

- `references/allo/README.md` - Framework overview
- `references/allo/allo/dsl.py` - Basic primitives
- `references/allo/allo/dataflow.py` - Dataflow programming
- `references/allo/allo/customize.py` - Schedule API
- `references/allo/allo/backend/aie/README.md` - AIE backend guide
- `references/allo/allo/backend/aie/mapping.py` - Spatial mapping implementation
