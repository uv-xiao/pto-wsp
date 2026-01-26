# PTO Workload-Schedule Programming (PTO-WSP): Conceptual Analysis (v7)

## Executive Summary

v7 introduces a **Workload-Schedule** separation paradigm (inspired by Halide) combined with a **CSP-based execution model** (inspired by DAM). The extension compiles to AICPU binary via JIT, enabling programmable runtime scheduling for dynamic LLM workloads.

**Key Principles:**
1. **Workload** = declarative specification of *what* computation happens (algorithm in Halide terms)
2. **Schedule** = specification of *how* computation is executed (scheduling strategy)
3. **CSP execution** = processes communicate via channels with local time (no global sync)
4. **JIT compilation** = Workload + Schedule → AICPU binary

## 1. Problem Recap

From [requirements.md](requirements.md):

| Challenge | Current Limitation | v7 Solution |
|-----------|-------------------|-------------|
| Dynamic KV lengths | Static graphs can't handle | Workload with dynamic bounds |
| MoE routing | Can't express data-dependent | `select()` primitive in workload |
| Task scheduling | No human-in-the-loop | Programmable schedule |
| Stitching | Head overhead, cold start | Pipeline/stitch schedule primitives |
| Dependencies | Hard to manage across graphs | CSP channels |

## 2. Workload-Schedule Separation

### 2.1 The Halide Insight (Adapted)

Halide separates **what to compute** from **how to compute**:

```cpp
// Halide: Image processing
Func blur_x(x,y) = (input(x-1,y) + input(x,y) + input(x+1,y)) / 3;  // Algorithm
blur_x.vectorize(x, 8).parallel(y);  // Schedule
```

**v7 adapts this for task scheduling:**

```cpp
// v7: Task scheduling
Workload attn = for_each(batch, [](b) {           // Workload (what)
    for_each(head, [](h) {
        task(attention_kernel, {b, h, seq_len[b]});
    });
});

Schedule sched = attn                              // Schedule (how)
    .dispatch(round_robin(num_aicpu))
    .issue(batch_affinity())
    .pipeline(depth=2);
```

### 2.2 Why Separation Matters

| Concern | Workload | Schedule |
|---------|----------|----------|
| **What** | Task generation logic | Task execution order |
| **When decided** | Mostly compile-time | Compile-time + runtime |
| **Changed by** | Algorithm designer | Performance engineer |
| **Correctness** | Defines semantics | Must preserve semantics |

The same workload can have **different schedules** for different scenarios:
- Small batch: static round-robin
- Large batch with variable seq_len: work-stealing
- Multi-batch: pipeline + stitch

### 2.3 Granularity

Unlike Halide (loop-level), v7 operates at **task-level**:

| Level | Halide | v7 |
|-------|--------|-----|
| Atomic unit | Loop iteration | Task (one kernel call) |
| Composition | Loop nests | Task graphs |
| Data | Arrays | Tensors |
| Execution | CPU/GPU threads | AICore invocations |

## 3. Workload Specification

### 3.1 Workload as Iteration Space + Task Generation

A **Workload** defines:
1. **Iteration space**: The set of points to iterate over
2. **Task generator**: How to create a task for each point
3. **Dependencies**: Data flow between tasks

```
Workload := IterationSpace × TaskGenerator × Dependencies
```

### 3.2 Workload Primitives

#### Iteration Primitives

| Primitive | Description | Example |
|-----------|-------------|---------|
| `for_each(range, f)` | Sequential iteration | `for_each(0..batch, ...)` |
| `parallel_for(range, f)` | Independent iterations | `parallel_for(heads, ...)` |
| `reduce(range, init, f)` | Reduction pattern | `reduce(blocks, 0, sum)` |

#### Control Flow Primitives

| Primitive | Description | Example |
|-----------|-------------|---------|
| `cond(pred, then, else)` | Conditional | `cond(seq_len < 2K, kernel_2k, kernel_8k)` |
| `select(indices, options)` | Dynamic selection (MoE) | `select(routing, experts)` |
| `spawn(workload)` | Dynamic workload generation | `spawn(expert_workload)` |

#### Task Primitives

| Primitive | Description | Example |
|-----------|-------------|---------|
| `task(kernel, params)` | Single kernel invocation | `task(gemm, {M, N, K})` |
| `fused(tasks...)` | Logically fused tasks | `fused(norm, proj)` |

### 3.3 Workload Examples

**Example 1: Attention**
```
attn_workload = parallel_for(batch, b =>
    parallel_for(head, h =>
        task(attention_kernel, {
            batch_idx: b,
            head_idx: h,
            seq_len: seq_lens[b]
        })
    )
)
```

**Example 2: MoE with Dynamic Routing**
```
moe_workload = parallel_for(batch, b =>
    parallel_for(token, t =>
        // Router determines which experts
        let selected = router(input[b][t])
        select(selected, expert_idx =>
            task(expert_kernels[expert_idx], {b, t})
        )
    )
)
```

**Example 3: Tiered Kernels**
```
tiered_attn = parallel_for(batch, b =>
    let len = seq_lens[b]
    cond(len <= 2048,
        task(attn_2k, {b, len}),
        cond(len <= 8192,
            task(attn_8k, {b, len}),
            task(attn_32k, {b, len})
        )
    )
)
```

### 3.4 Workload Properties

A well-formed workload must satisfy:
1. **Determinism**: Same inputs → same tasks (modulo scheduling)
2. **Finite generation**: Task generation terminates
3. **Type safety**: Task parameters match kernel signatures

## 4. Schedule Specification

### 4.1 Schedule as Transformation

A **Schedule** transforms a workload for efficient execution without changing semantics.

```
Schedule := Workload → ExecutionPlan
```

### 4.2 Schedule Primitives

#### Dispatch Primitives (Host → AICPU)

| Primitive | Description | Effect |
|-----------|-------------|--------|
| `dispatch(policy)` | Assign tasks to AICPUs | Partitions task space |
| `colocate(tasks)` | Keep tasks on same AICPU | Improves locality |
| `replicate(n)` | Duplicate workload | For redundancy |

Dispatch policies:
- `round_robin(n)`: Distribute evenly across n AICPUs
- `hash(key)`: Hash-based assignment
- `affinity(axis)`: Keep same axis value together
- `dynamic()`: Runtime load balancing

#### Issue Primitives (AICPU → AICore)

| Primitive | Description | Effect |
|-----------|-------------|--------|
| `issue(order)` | Order tasks on AICores | Controls execution sequence |
| `bind(axis, cores)` | Bind axis to specific cores | Static assignment |
| `steal()` | Enable work stealing | Dynamic load balancing |

Issue orders:
- `fifo()`: First-in-first-out
- `lifo()`: Last-in-first-out (better locality)
- `priority(f)`: Custom priority function
- `batch_affinity()`: Same batch on same core

#### Pipeline Primitives (Overlap)

| Primitive | Description | Effect |
|-----------|-------------|--------|
| `pipeline(depth)` | Overlap execution phases | Hides latency |
| `double_buffer()` | Overlap load/compute/store | Memory hiding |
| `prefetch(n)` | Prefetch n tasks ahead | Reduces stalls |

#### Stitch Primitives (Composition)

| Primitive | Description | Effect |
|-----------|-------------|--------|
| `stitch(w1, w2)` | Merge independent workloads | Batch combination |
| `fuse(w1, w2)` | Fuse dependent workloads | Reduce dispatch |
| `interleave(w1, w2)` | Interleave execution | Wave overlap |

### 4.3 Schedule Examples

**Example 1: Simple Static Scheduling**
```
schedule_static = attn_workload
    .dispatch(round_robin(4))        // 4 AICPUs
    .issue(fifo())                   // FIFO order
```

**Example 2: Work-Stealing for Variable Lengths**
```
schedule_dynamic = attn_workload
    .dispatch(affinity(batch))       // Same batch → same AICPU
    .issue(steal())                  // Work stealing enabled
    .pipeline(depth=2)               // 2-deep pipeline
```

**Example 3: Batch Stitching with Pipeline**
```
schedule_stitched = stitch(batch0_workload, batch1_workload)
    .dispatch(hash(batch_id))
    .issue(batch_affinity())
    .interleave(compute_pool, memory_pool)  // Wave interleaving
    .pipeline(depth=3)
```

**Example 4: MoE with Dynamic Routing**
```
schedule_moe = moe_workload
    .dispatch(dynamic())             // Dynamic load balance
    .issue(priority(expert_load))    // Prioritize underloaded experts
    .colocate(same_expert_tasks)     // Keep same expert together
```

### 4.4 Schedule Composition

Schedules compose:
```
s1 = workload.dispatch(policy1)
s2 = s1.issue(order1)
s3 = s2.pipeline(depth=2)
// Equivalent to:
s = workload.dispatch(policy1).issue(order1).pipeline(depth=2)
```

## 5. CSP Execution Model

### 5.1 Why CSP (Not Events)

v6 used event-driven handlers (`on_dispatch`, `on_ready`, etc.). This is general but has issues:

| Event-Driven (v6) | CSP (v7) |
|-------------------|----------|
| Callbacks invoked by runtime | Processes run continuously |
| Complex state management | Simple sequential code |
| Hard to reason about | Clear control flow |
| Global synchronization | Local time, no global sync |

**CSP with Time (CSPT)** from DAM paper provides:
- **Local time**: Each process has its own clock
- **Channels**: Communication and synchronization
- **No global barriers**: Processes can diverge in time

### 5.2 CSP Primitives

#### Process Primitives

| Primitive | Description |
|-----------|-------------|
| `process(body)` | Create a process |
| `spawn(process)` | Start a process |
| `time.tick()` | Get current local time |
| `time.advance(t)` | Advance local time by t |
| `time.wait_until(t)` | Wait until time t |

#### Channel Primitives

| Primitive | Description |
|-----------|-------------|
| `chan<T>(cap)` | Create channel with capacity |
| `send(ch, val)` | Send value (may block if full) |
| `recv(ch)` | Receive value (blocks if empty) |
| `try_send(ch, val)` | Non-blocking send |
| `try_recv(ch)` | Non-blocking receive |
| `select(cases)` | Wait on multiple channels |

### 5.3 CSP Execution Model

The runtime consists of **processes** communicating via **channels**:

```
┌─────────────────────────────────────────────────────────────┐
│ AICPU Runtime                                               │
│                                                             │
│  ┌──────────┐    tasks    ┌───────────┐    ready    ┌─────┐ │
│  │ Generator │──────────→│ Scheduler  │───────────→│Issuer│ │
│  └──────────┘    chan     └───────────┘    chan     └─────┘ │
│       │                        │                      │     │
│       │ spawn                  │ deps                 │issue│
│       ▼                        ▼                      ▼     │
│  ┌──────────┐           ┌───────────┐          ┌──────────┐ │
│  │ Spawner  │           │ DepTracker │          │ AICores  │ │
│  └──────────┘           └───────────┘          └──────────┘ │
└─────────────────────────────────────────────────────────────┘
```

**Processes:**
1. **Generator**: Produces tasks from workload specification
2. **Scheduler**: Applies scheduling decisions
3. **Issuer**: Issues tasks to AICores
4. **DepTracker**: Tracks task dependencies
5. **Spawner**: Handles dynamic task generation (MoE)

**Channels:**
- `tasks: chan<Task>` - Generated tasks
- `ready: chan<Task>` - Tasks ready to execute
- `complete: chan<TaskId>` - Completed task notifications
- `spawn: chan<Workload>` - Dynamically generated workloads

### 5.4 CSP Example: Work-Stealing Scheduler

```
process Scheduler(tasks: recv<Task>, ready: send<Task>, config) {
    queues = [Queue() for _ in 0..num_cores]

    loop {
        select {
            // New task arrived
            task <- tasks:
                core = select_core(task, config)
                queues[core].push(task)

            // Core requests work
            (core, req) <- work_request:
                if !queues[core].empty() {
                    send(ready, queues[core].pop())
                } else {
                    // Work stealing
                    victim = find_victim(queues)
                    if victim >= 0 {
                        stolen = queues[victim].steal()
                        send(ready, stolen)
                    }
                }

            // Timeout - advance time
            after 100ns:
                time.advance(100ns)
        }
    }
}
```

### 5.5 CSP for Dependencies

Dependencies are expressed via channels:

```
// Producer-consumer dependency
process Producer(out: send<Tensor>) {
    result = compute()
    send(out, result)
}

process Consumer(in: recv<Tensor>) {
    data = recv(in)  // Blocks until producer sends
    use(data)
}

// Connect with channel
ch = chan<Tensor>(1)
spawn(Producer(ch))
spawn(Consumer(ch))
```

**Barrier synchronization:**
```
process BarrierWait(barrier: chan<()>, n: int) {
    for _ in 0..n {
        recv(barrier)  // Wait for n signals
    }
}
```

### 5.6 CSP vs Events Comparison

**Event-driven (v6):**
```cpp
RuntimeProgram work_stealing = {
    .on_dispatch = [](WorkRequest& req, RuntimeContext& ctx) {
        // Complex callback logic
        for (auto& task : req.tasks) {
            ctx.enqueue(shared_queue, task);
        }
    },
    .on_ready = [](Task& t, RuntimeContext& ctx) {
        // Another callback
        int core = ctx.get_free_core();
        ctx.issue(t, core);
    },
};
```

**CSP (v7):**
```
process WorkStealer(tasks: recv<Task>, cores: [send<Task>]) {
    shared_queue = Queue()

    loop {
        select {
            task <- tasks:
                shared_queue.push(task)

            (core_id, _) <- core_requests:
                if !shared_queue.empty() {
                    send(cores[core_id], shared_queue.pop())
                }
        }
    }
}
```

## 6. JIT Compilation Model

### 6.1 Why JIT

The runtime extension must run on AICPU. Normal C++ won't work for dynamic strategies because:
1. **Dynamic code generation**: MoE routing creates tasks at runtime
2. **Schedule specialization**: Different schedules need different code
3. **Hardware constraints**: AICPU has limited resources

**Solution**: JIT compile Workload + Schedule → AICPU binary

### 6.2 Compilation Pipeline

```
┌──────────────┐     ┌───────────────┐     ┌──────────────┐
│   Workload   │     │    Schedule   │     │   Runtime    │
│ Specification│     │ Specification │     │   Library    │
└──────┬───────┘     └───────┬───────┘     └──────┬───────┘
       │                     │                    │
       └─────────┬───────────┘                    │
                 │                                │
                 ▼                                │
       ┌─────────────────┐                       │
       │  Workload-Sched │                       │
       │    Compiler     │                       │
       └────────┬────────┘                       │
                │                                │
                ▼                                │
       ┌─────────────────┐                       │
       │   AICPU IR      │◄──────────────────────┘
       │  (Intermediate) │
       └────────┬────────┘
                │
                ▼
       ┌─────────────────┐
       │  AICPU Binary   │
       │  (Executable)   │
       └─────────────────┘
```

### 6.3 AICPU IR

The intermediate representation supports:

**Control flow:**
- Loops, conditionals, function calls
- CSP process/channel primitives

**Task operations:**
- Task creation, parameter binding
- Kernel invocation

**Scheduling:**
- Queue operations
- Core assignment
- Synchronization

### 6.4 Compilation Strategies

| Component | Strategy |
|-----------|----------|
| Static workload | Fully compile |
| Dynamic workload (MoE) | Generate interpreter |
| Schedule | Specialize based on policy |
| CSP processes | Compile to coroutines |
| Channels | Compile to lock-free queues |

### 6.5 Example Compilation

**Input (Workload + Schedule):**
```
workload = parallel_for(batch, b =>
    task(attn, {b, seq_lens[b]})
)

schedule = workload
    .dispatch(round_robin(4))
    .issue(fifo())
```

**Compiled AICPU IR:**
```
define @runtime_main() {
entry:
    %aicpu_id = call @get_aicpu_id()

generate:
    %batch_start = mul %aicpu_id, %batch_per_aicpu
    %batch_end = add %batch_start, %batch_per_aicpu

    for %b = %batch_start to %batch_end {
        %seq_len = load @seq_lens[%b]
        %task = call @make_task(@attn_kernel, %b, %seq_len)
        call @enqueue_task(%task)
    }

issue_loop:
    %task = call @dequeue_task()
    cmp %task, null
    br.eq done

    call @issue_to_aicore(%task)
    br issue_loop

done:
    ret
}
```

## 7. Complete Architecture

### 7.1 System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                           HOST                                   │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                    User Code                                │ │
│  │  workload = parallel_for(...)                              │ │
│  │  schedule = workload.dispatch(...).issue(...)              │ │
│  └────────────────────────────────────────────────────────────┘ │
│                              │                                   │
│                              ▼                                   │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                 JIT Compiler                                │ │
│  │  Workload + Schedule → AICPU Binary                        │ │
│  └────────────────────────────────────────────────────────────┘ │
│                              │                                   │
│                              ▼ dispatch (~3μs)                   │
└──────────────────────────────┼───────────────────────────────────┘
                               │
┌──────────────────────────────┼───────────────────────────────────┐
│                           AICPU                                   │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │              Compiled Runtime Program                       │ │
│  │  ┌─────────┐   ┌───────────┐   ┌────────┐                 │ │
│  │  │Generator│──→│ Scheduler │──→│ Issuer │                 │ │
│  │  └─────────┘   └───────────┘   └────────┘                 │ │
│  │       │              │              │                      │ │
│  │       └──────────────┴──────────────┘                      │ │
│  │                CSP Channels                                 │ │
│  └────────────────────────────────────────────────────────────┘ │
│                              │                                   │
│                              ▼ issue (~0μs)                      │
└──────────────────────────────┼───────────────────────────────────┘
                               │
┌──────────────────────────────┼───────────────────────────────────┐
│                          AICores                                  │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐               │
│  │ AICore0 │ │ AICore1 │ │ AICore2 │ │   ...   │               │
│  │ (Task)  │ │ (Task)  │ │ (Task)  │ │         │               │
│  └─────────┘ └─────────┘ └─────────┘ └─────────┘               │
└─────────────────────────────────────────────────────────────────┘
```

### 7.2 Execution Flow

1. **User specifies** Workload and Schedule
2. **JIT compiles** to AICPU binary
3. **Host dispatches** compiled program to AICPU (~3μs)
4. **AICPU runs** CSP processes:
   - Generator produces tasks
   - Scheduler orders tasks
   - Issuer sends to AICores (~0μs per issue)
5. **AICores execute** tasks
6. **Completion notifications** flow back via channels

### 7.3 Dynamic Workloads (MoE)

For MoE, the workload includes `select()` which generates tasks at runtime:

```
moe = parallel_for(token, t =>
    select(router(t), expert_idx =>
        task(experts[expert_idx], t)
    )
)
```

The JIT compiler generates:
1. **Router invocation**: Calls router kernel
2. **Task generation**: Creates expert tasks based on routing
3. **Scheduling**: Applies schedule to generated tasks

## 8. Comparison with Previous Versions

| Aspect | v5 | v6 | v7 |
|--------|-----|-----|-----|
| **Task model** | Task + TaskSet | Task + WorkRequest | Task (from Workload) |
| **Scheduling** | IssuePolicy enums | Handler callbacks | Schedule primitives |
| **Execution** | Fixed patterns | Event-driven | CSP processes |
| **Dynamic** | Spawn API | on_spawn handler | select() + spawn() |
| **Dependencies** | Events | Events + Barriers | CSP channels |
| **Compilation** | N/A | Normal C++ | JIT to AICPU |
| **Stitching** | Explicit stitch() | Pattern in handler | Schedule primitive |

## 9. Key Innovations

### 9.1 Workload-Schedule Separation
- Clear separation of concerns
- Same workload, different schedules
- Human-in-the-loop via schedule tuning

### 9.2 CSP Execution Model
- No global synchronization
- Local time for each process
- Channels for communication
- Simple, composable code

### 9.3 JIT Compilation
- Workload + Schedule → AICPU binary
- Supports dynamic code generation
- Specializes for specific scenarios

### 9.4 Unified Primitives
- Workload primitives for task generation
- Schedule primitives for execution strategy
- CSP primitives for communication

## 10. References

- [Halide: Decoupling Algorithms from Schedules](research/03_pl_design.md)
- [Megakernels: Inter-instruction Pipelining](research/06_megakernels.md)
- [DAM: CSP with Time](references/dam.pdf)
- [CUDA Streams and Work Stealing](research/07_cuda_streams_workstealing.md)
- [Requirements](requirements.md)
- [Design Questions](design-questions.md)

---
*Version: 7.0*
*Last Updated: 2025-01-17*
