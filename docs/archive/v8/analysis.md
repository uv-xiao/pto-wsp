# PTO Workload-Schedule Programming (PTO-WSP): Design Analysis

## 1. Design Rationale

### 1.1 The Core Problems

LLM inference on Ascend NPU faces three fundamental challenges:

| Problem | Root Cause | Impact |
|---------|------------|--------|
| **Serial task generation** | Single-threaded control flow | Tasks enumerated one-by-one, blocking parallelism |
| **Serial execution queue** | Tasks wait in single queue | Cannot overlap independent operations |
| **Fixed scheduling** | No programmer control over dispatch/issue | Cannot optimize for specific workloads |

**Key insight**: These problems concern **when** tasks are generated, **where** they execute, and **in what order**. A unified solution must address all three.

### 1.2 Why Workload-Schedule Separation

The design separates **what to compute** (Workload) from **how to execute** (Schedule):

| Concern | Workload | Schedule |
|---------|----------|----------|
| Responsibility | Declare task structure and dependencies | Control dispatch, issue, timing |
| When defined | Once per model | Tuned per deployment |
| Change frequency | Rarely | Often (based on profiling) |

**Rationale**: This separation enables:
1. **Reusability** - Same workload, different schedules for different hardware
2. **Optimization iteration** - Tune schedule without touching compute logic
3. **Static analysis** - Type system encodes parallelism guarantees

### 1.3 Why Two Parallelism Modes

Different patterns require different abstractions:

| Pattern | Best Fit | Example |
|---------|----------|---------|
| Same operation on different data | Data-parallel (`parallel_for`) | Attention heads, MoE experts |
| Different operations concurrently | Pipeline-parallel (CSP) | Megakernel load/compute/store |

**Rationale**: A single abstraction cannot elegantly express both. Data-parallel is simpler and covers most cases; CSP handles complex pipelines.

---

## 2. Type System

### 2.1 Workload Type

A **Workload** is a typed expression describing task generation:

```
Workload[Axes, Task, Deps] where:
  - Axes : Iteration space (product of axis types)
  - Task : Kernel invocation type
  - Deps : Dependency structure
```

**Rationale**: Encoding structure in types enables compile-time reasoning about parallelism and dependencies.

### 2.2 Axis Types

| Axis Type | Description | Use Case |
|-----------|-------------|----------|
| `Dense<N>` | Static size N | Fixed dimensions (8 heads) |
| `DenseDyn` | Runtime size | Dynamic batch size |
| `Ragged` | Per-element lengths | Variable seq_len per batch |
| `Sparse` | CSR format | MoE routing |

**Rationale**: Axes are types, not values. This enables static optimization and prevents invalid compositions.

### 2.3 Dependency Types

| Type | Meaning | Inferred From |
|------|---------|---------------|
| `Independent` | All tasks can run in parallel | `parallel_for` |
| `Sequential` | Tasks run in order | `for_each`, `sequential()` |
| `Combined` | Schedule determines order | `combine()` |
| `ChannelDep` | Producer-consumer | CSP channels |
| `None` | Single task | `task()` |

**Rationale**: Dependencies are structural, inferred from composition. This eliminates manual dependency declaration and prevents errors.

---

## 3. Data-Parallel Primitives

### 3.1 Core Primitives

```cpp
// Independent iteration
parallel_for(axis, body) → Workload[..., Independent]

// Sequential iteration
for_each(axis, body) → Workload[..., Sequential]

// Composition (schedule determines timing)
combine(w1, w2, ...) → Workload[..., Combined]

// Explicit ordering (w2 after w1 completes)
sequential(w1, w2, ...) → Workload[..., Sequential]

// Sparse iteration (MoE routing)
select(sparse, body) → Workload[..., Independent]

// Conditional
cond(pred, then, else) → Workload[...]

// Leaf task
task(kernel, params, resources) → Workload[Unit, Task, None]
```

**Rationale**: Declarative primitives (vs imperative loops) enable:
- Parallel task enumeration
- JIT analysis and optimization
- Dependency inference from structure

### 3.2 Dependency Inference

Dependencies come from composition, not manual declaration:

```cpp
// Independent: all B×H tasks can run in parallel
auto attn = parallel_for(batch, [](b) {
    return parallel_for(head, [](h) {
        return task(attn_kernel, {b, h}, ...);
    });
});
// Type: Workload[Dense[B] × Dense[H], AttnTask, Independent]

// Sequential: task[i] depends on task[i-1]
auto scan = for_each(seq, [](i) {
    return task(scan_kernel, {i}, ...);
});
// Type: Workload[Dense[S], ScanTask, Sequential]
```

**Rationale**: Structure determines dependencies. `parallel_for` means independent; `for_each` means sequential. No room for error.

---

## 4. CSP Primitives (Pipeline-Parallel)

### 4.1 Why CSP

Data-parallel expresses "same operation on different data." Some patterns need "different operations running concurrently":

- **Megakernels**: Loader → Computer → Storer pipeline
- **Streaming**: Producer-consumer with backpressure
- **Work stealing**: Multiple processes share a queue

### 4.2 Core Primitives

```cpp
// Process: concurrent execution unit
process(name)
    .consumes(channel)
    .produces(channel)
    .body(computation)  // Must use declarative primitives

// Channel: bounded buffer (capacity controls pipeline depth)
Channel<T, N>  // N = buffer size; N=0 for rendezvous

// Event: unbuffered signal (unified with Channel)
Event = Channel<Signal, 0>
```

**Key design decision**: Channels carry **Workloads** (including Tasks), not raw values. This unifies CSP with data-parallel primitives.

### 4.3 Example: Megakernel Pipeline

```cpp
Channel<Workload, 2> load_to_compute;  // Depth 2 = double buffering
Channel<Workload, 2> compute_to_store;

auto loader = process("loader")
    .produces(load_to_compute)
    .body(for_each(DenseDyn(num_tiles), [](i) {
        return send(load_to_compute, task(load_kernel, {i}, ...));
    }));

auto computer = process("computer")
    .consumes(load_to_compute)
    .produces(compute_to_store)
    .body(consume(load_to_compute, [](load_task) {
        return send(compute_to_store, task(compute_kernel, ...));
    }));

auto storer = process("storer")
    .consumes(compute_to_store)
    .body(consume(compute_to_store, [](compute_task) {
        return task(store_kernel, ...);
    }));

Pipeline pipeline = connect({loader, computer, storer}, {...});
```

**Rationale**:
- `consume()` replaces `while(recv())` - declarative, JIT-analyzable
- Channel capacity = pipeline depth - explicit control
- `Event = Channel<Signal, 0>` - unified synchronization model

---

## 5. Schedule API

### 5.1 Two-Phase Model

```
Workload → [Enumerate] → Tasks → [Dispatch] → AICPUs → [Issue] → Streams
```

**Dispatch**: Which AICPU handles each task
**Issue**: Order and timing of tasks within each AICPU

### 5.2 Dispatch Policies

```cpp
dispatch_by(fn)      // Custom: fn(task) → AICPU index
round_robin(n)       // Distribute evenly across n AICPUs
affinity(axis)       // Same axis value → same AICPU
work_steal           // Dynamic load balancing
```

**Rationale**: `dispatch_by(fn)` is the fundamental primitive; others are sugar. User has full control over task placement.

### 5.3 Issue Policies

```cpp
stream_by(fn)        // Assign tasks to streams
streams(n)           // Number of concurrent streams
timing(policy)       // eager, batched, rate_limited
```

**Rationale**: Streams control concurrency within each AICPU. Multiple streams = overlapped execution on different AICores.

### 5.4 Complete Example

```cpp
auto workload = parallel_for(batch, [](b) {
    return parallel_for(head, [](h) {
        return task(attn_kernel, {b, h}, ...);
    });
});

auto schedule = workload.schedule()
    .dispatch(affinity([](t) { return t.batch; }))  // Same batch → same AICPU
    .streams(2)                                      // 2 concurrent streams
    .stream_by([](t) { return t.head % 2; })        // Alternate heads
    .timing(TimingPolicy::immediate);                // Issue when ready

auto program = schedule.compile();
program.execute();
```

**Rationale**: Schedule is separate from workload. Change dispatch strategy without touching compute logic.

---

## 6. Stream Synchronization

### 6.1 Stream Model

- Tasks in **same stream**: Execute in order
- Tasks in **different streams**: Execute concurrently
- **Events**: Cross-stream synchronization points

### 6.2 Event = Unbuffered Channel

```cpp
using Event = Channel<Signal, 0>;  // Capacity 0 = rendezvous

record(e);  // ≡ send(e, Signal{})
wait(e);    // ≡ recv(e)
```

**Rationale**: Unifying Event with Channel simplifies the model. All synchronization uses the same primitive.

---

## 7. Type Soundness

The type system ensures correctness:

```
parallel_for(axis, body) : Workload[axis × A, T, Independent]
for_each(axis, body)     : Workload[axis × A, T, Sequential]
combine(w1, w2)          : Workload[A1 ∪ A2, T1 ∪ T2, Combined]
sequential(w1, w2)       : Workload[A1 ∪ A2, T1 ∪ T2, Sequential]
task(k, p, r)            : Workload[Unit, Task[K,P,R], None]
```

**Rationale**: Type rules guarantee that:
- `parallel_for` always produces `Independent` dependencies
- `for_each` always produces `Sequential` dependencies
- Composition rules are enforced at compile time

---

## 8. Summary

| Design Choice | Rationale |
|---------------|-----------|
| **Workload-Schedule separation** | Optimize schedule without touching compute logic |
| **Typed workloads** | Static reasoning about parallelism and dependencies |
| **Declarative primitives** | JIT-friendly, parallel enumeration possible |
| **Dependency inference** | Structure determines dependencies, prevents errors |
| **Two parallelism modes** | Data-parallel for most cases, CSP for pipelines |
| **Event = Channel<Signal, 0>** | Unified synchronization model |
| **dispatch_by, stream_by** | Programmer control over execution |

**What we control**: Task generation, dispatch to AICPUs, issue to AICores, timing.
**What we don't control**: Kernel internals (that's PTO-ISA's job).

---

## References

- [FlashInfer: Index-based dispatch](research/01_flashinfer.md)
- [Megakernels: Inter-instruction pipelining](research/06_megakernels.md)
- [CUDA Streams](research/07_cuda_streams_workstealing.md)
- [SparseTIR: Axis types](research/09_sparsetir.md)
