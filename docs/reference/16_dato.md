# Research Report 16: Dato Paper Analysis

## Overview

**Source**: `references/dato.pdf`
**Title**: "Dato: A Task-Based Programming Model for Dataflow Accelerators"
**Authors**: Shihan Fang, Hongzheng Chen, Niansong Zhang, et al. (Cornell University)
**Publication**: arXiv:2509.06794, September 2025
**v9 Relevance**: Core theoretical foundation for spatial architecture targeting, stream types, layout types

---

## 1. Paper Summary

### 1.1 Problem Statement

> "Recent deep learning workloads increasingly push computational demand beyond what current memory systems can sustain, with many kernels stalling on data movement rather than computation."

**Key Challenges**:
1. **Low-level control vs productivity**: IRON/CUDA require hundreds of lines for simple GEMM
2. **Tile-based languages hide communication**: Triton/ARIES abstract away inter-kernel streaming
3. **Compilers must reconstruct dataflow intent**: Array-based frontends force semantic mismatch

### 1.2 Solution

Dato elevates **data communication** and **data sharding** to **first-class type constructs**:
- Programs are graphs of **tasks** connected via **stream types**
- Sharded inputs specified using **layout types**
- **Virtual-to-physical mapping** automatically compiles to hardware

### 1.3 Results

| Target | Benchmark | Performance |
|--------|-----------|-------------|
| AMD NPU | GEMM | 84% hardware utilization |
| AMD NPU | MHA (attention) | 2.81× speedup vs IRON |
| FPGA (U280) | Systolic array | 98% theoretical peak |

---

## 2. Key Concepts

### 2.1 Stream Type

```python
Z: Stream[Ty[M // P0]][P0]  # Array of P0 streams, each carrying Ty[M // P0] tensor

@task(mapping=[P0])
def producer(A: Ty[M] @ Layout("S")):
    tid = dato.get_tid()
    Z[tid].put(A[:])  # Send to stream

@task(mapping=[P0])
def consumer(B: Ty[M] @ Layout("S")):
    tid = dato.get_tid()
    B[:] = Z[tid].get() + 1  # Receive from stream
```

**Stream Type Definition**: `Stream[T, N, P]`
- `T`: Element type (can be tensor)
- `N`: Logical capacity (FIFO depth)
- `P`: Packing factor (elements per transfer)

**Operations**:
- `.put(x)`: Enqueue element (blocks if full)
- `.get()`: Dequeue element (blocks if empty)

### 2.2 Typing Rules for Streams

From Figure 4 of the paper:

```
Γ ⊢ s : Stream[T, N]    Γ ⊢ x : T    S = id(s)
─────────────────────────────────────────────────────  (T-Put)
Γ ; Δ ⊎ { Free(S) } ⊢ put(s, x) : 1 ; Δ ⊎ { Used(S) }

Γ ⊢ s : Stream[T, N]    S = id(s)
───────────────────────────────────────────────────────  (T-Get)
Γ ; Δ ⊎ { Used(S) } ⊢ get(s) : LFuture T ; Δ ⊎ { Free(S) }
```

**Key Properties**:
- **Linear capability tokens**: `Free(S)` and `Used(S)` track slot availability
- `.put()` converts `Free(S)` → `Used(S)` (tail advances)
- `.get()` converts `Used(S)` → `Free(S)` (head advances)
- Overflow/underflow become **untypeable by construction**

**Type Checking**: Programs rejected if:
- Deadlock (circular dependencies)
- Inconsistent put/get (mismatched counts)

### 2.3 Layout Type

```python
# Layout type definition
ℓ ::= S | R                    # Sharded or Replicated
L ::= ⟨ℓ1, ..., ℓd⟩            # Per-dimension layout
τ ::= D[n̄]@L                   # Tensor with layout

# Example layouts
LyA = Layout("S1R")   # Dim 0 sharded to axis 1, dim 1 replicated
LyB = Layout("RS0")   # Dim 0 replicated, dim 1 sharded to axis 0
LyC = Layout("S1S0")  # Both dims sharded to different axes
```

**Typing Rules for Operations**:

| Operation | Rule | Effect |
|-----------|------|--------|
| Elementwise | `x ⊙ y : D[n̄]@(Lx ⊔ Ly)` | Layout union |
| Reduce (sharded axis) | `reduce⊕,i(e)` | Adds `⊕` to pending effects Π |
| Reduce (replicated axis) | `reduce⊕,i(e)` | Local only, no collective needed |
| AllReduce | `allreduce⊕(x)` | Discharges `⊕` from Π |
| Matmul | `A@(ℓ1a,ℓ2a) × B@(ℓ1b,ℓ2b)` | Requires `ℓ2a = ℓ1b` |

---

## 3. Virtual Mapping

### 3.1 Virtual Mapping Graph (VMG)

```python
@dato.task(mapping=[Pk, Pm, Pn])  # 3D virtual grid
def gemm(A: Ty[M, K] @ LyA,
         B: Ty[K, N] @ LyB,
         C: Ty[M, N] @ LyC):
    part_C = dato.matmul(A, B)
    C[:, :] = dato.allreduce(part_C, op="+")
```

Creates a **3D lattice** of task instances:
- One instance per point in virtual grid `[Pk × Pm × Pn]`
- Edges from `.put()/.get()` operations
- Collective edges from `.allreduce()`

### 3.2 Mapping Primitives

| Primitive | Effect | Use Case |
|-----------|--------|----------|
| `.bundle()` | Merge isomorphic nodes into multi-shot node | Time-multiplex fan-in/fan-out |
| `.chain()` | Fuse producer-consumer pair | Remove intermediate stream |

**Example**: For `C0 = A0×B0 + A1×B1`:
1. Initial VMG: Parallel multiply nodes + adder
2. After `.bundle()`: Single join node
3. After `.chain()`: Single fused node

### 3.3 Automated Mapping Algorithm

```python
def search_optimal_mapping(G: VMG, C: int) -> Module:
    """
    G: Initial VMG
    C: Available physical PE count
    """
    M = set()  # mapping candidates

    def span(state):
        if state.v_node_number <= C:
            apply_optimization_passes(state.vmg)
            if module := build(state.vmg):
                M.add(module)
                if len(M) > threshold:
                    return  # early stop

        for (u, v) in state.v_node_pairs:
            if is_valid_bundle_or_chain(u, v):
                t = state.apply_primitive(u, v)
                span(t)

    root = SearchTree(G)
    span(root)
    return find_optimal_among(M)
```

**Legality Conditions**:
1. **Resource constraints**: I/O ports, per-tile limits
2. **Interface compatibility**: Matching types, shapes
3. **Dependency soundness**: No cycles from transitive paths

---

## 4. Optimizations

### 4.1 Kernel Injection

- Match computation graph to pre-tuned kernel contracts
- Synthesize wrapper with tiling + layout adaptors
- NPU: Enforce VLIW operand layouts
- FPGA: Emit HLS-friendly pipelined loops

### 4.2 Layout Optimization

**Normalization & Collapse**:
- Model layout transforms as affine maps
- Canonicalize compositions
- Collapse to minimal normal form

**DMA-aware Hoisting**:
- Pack → strided gather
- Transpose → 2D stride swap
- Fold into DMA descriptors

---

## 5. Comparison with Existing Frameworks

From Table 1 of the paper:

| Framework | Target | Explicit Comm | Explicit Sharding | Auto Opt | Type Checking |
|-----------|--------|---------------|-------------------|----------|---------------|
| CUTLASS | GPU | ✓ | ✗ | ✗ | ✗ |
| Triton | GPU | ✗ | ✗ | ✓ | ✗ |
| Allo | FPGA | ✓ | ✗ | ✗ | ✗ |
| IRON | NPU | ✓ | ✗ | ✗ | ✗ |
| **Dato** | FPGA/NPU | ✓ | ✓ | ✓ | ✓ |

**Dato's unique combination**: All four features together.

---

## 6. Mapping to v8/v9 Concepts

### 6.1 Direct Correspondences

| Dato Concept | v8 Runtime Extension | v9 Target |
|--------------|---------------------|-----------|
| `@task(mapping=[...])` | Task | Task with spatial mapping |
| `Stream[T, N]` | `Channel<T, N>` | Same |
| `.put()` / `.get()` | `send()` / `receive()` | Same |
| `Layout("S")` / `Layout("R")` | - | New `layout_type` |
| `allreduce(op)` | - | Collective primitive |
| `.bundle()` | - | Schedule primitive |
| `.chain()` | - | Schedule primitive |
| `dato.get_tid()` | Task iteration index | Same |

### 6.2 Key Insights for v9

1. **Stream Types are First-Class**: Not array-to-stream conversion
2. **Layout Types Enable Static Checking**: Catch sharding errors at compile time
3. **Virtual Mapping Separates Concerns**: Logical parallelism vs physical resources
4. **Type System Prevents Deadlocks**: Untypeable programs rejected early

---

## 7. Proposed v9 Extensions Based on Dato

### 7.1 Stream Type in Python DSL

```python
# v9 proposed syntax
from pto_wsp import Stream, Layout

@workload
def attention():
    # Stream array declaration
    Q_stream: Stream[Ty[Tq, d]][P0, P1]
    S_stream: Stream[Ty[Tq, Tkv]][P0, P1]

    @task(mapping=[P0, P1])
    def gemm0(K: Ty[d, N] @ Layout("RS0")):
        ti, tj = get_tid()
        S_stream[ti, tj].put(matmul(Q_stream[ti, tj].get(), K))
```

### 7.2 Layout Type in Schedule API

```python
# v9 proposed syntax
schedule = Schedule(workload)

# Layout annotations
schedule.layout(workload.A, Shard(dim=0), Replicate())
schedule.layout(workload.B, Replicate(), Shard(dim=1))
schedule.layout(workload.C, Shard(dim=0), Shard(dim=1))

# Virtual mapping
schedule.spatial_map(workload.gemm, grid=(4, 4))
```

### 7.3 Mapping Primitives

```python
# v9 proposed syntax
schedule.bundle(workload.task_a, workload.task_b)  # Time-multiplex
schedule.chain(workload.producer, workload.consumer)  # Fuse
```

---

## 8. Performance Analysis

### 8.1 NPU Results (AMD Ryzen AI)

| Precision | Utilization | Comparison |
|-----------|-------------|------------|
| i16 | 75.01% | Matches IRON |
| bf16 | 47.56% | Slightly below IRON |
| i8 | 61.58% | **Exceeds** IRON |
| Mixed (i4×i8) | **84.38%** | Best overall |

### 8.2 Multi-Kernel (FlashAttention)

- **2.81× speedup** over IRON for MHA
- First FlashAttention on AMD NPU using high-level framework
- Streaming eliminates DRAM write-back between kernels

### 8.3 FPGA Systolic Array

- 16×16 i8 output-stationary array at 300 MHz
- **98% of theoretical peak** (150 GOP/s vs 153.6 GOP/s peak)
- **8.2× speedup** over Allo

---

## 9. Implications for v9 Design

### 9.1 What v9 Should Adopt

1. **Stream as First-Class Type**: Not derived from Channel
2. **Layout Type System**: Static sharding validation
3. **Virtual-to-Physical Mapping**: Automated search algorithm
4. **Type Checking on Dataflow**: Prevent deadlocks at compile time

### 9.2 What v9 Should Preserve

1. **CSP Foundation**: Channels, Processes, Events
2. **Typed Workloads**: `Workload[Axes, Task, Deps]`
3. **Declarative Primitives**: `parallel_for`, `for_each`, `select`
4. **Schedule Separation**: Explicit Schedule object

### 9.3 Open Questions

1. **IR Representation**: How to represent Layout/Stream types in C++ IR?
2. **Type Inference**: Can we infer layouts from computation patterns?
3. **Collective Support**: How to express `allreduce` in our CSP model?
4. **Backward Compatibility**: How to support existing v8 code?

---

## 10. Code Size Comparison

From Table 2 of the paper:

| Kernel | IRON (lines) | Dato (lines) | Reduction |
|--------|--------------|--------------|-----------|
| GEMM | 101 | 8 | 12× |
| MHA | 239 | 44 | 5× |
| FFN | 101 | 14 | 7× |

**Key Insight**: First-class stream/layout types dramatically reduce boilerplate.

---

## References

- Primary: `references/dato.pdf` - arXiv:2509.06794
- Related: `references/allo/` - Allo framework (Dato builds on Allo)
- Background: Cornell Zhang Lab publications on accelerator design
