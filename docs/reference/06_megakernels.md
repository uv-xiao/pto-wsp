# Research Note 6: Megakernels Architecture Deep Dive

## Overview

Megakernels is a research framework that implements **entire AI workloads** (e.g., LLM forward pass) as a single fused GPU kernel, using an **instruction-interpreter** execution model. This note extracts key architectural insights relevant to PTO-ISA runtime extension design.

**Source**: Analysis of Megakernels paper (throughput-optimized megakernels) and source code (megakernel.cuh, controller.cuh, scheduling.cpp, instructions.hpp)

## 1. Core Design Principle: Instruction-Interpreter Model

### 1.1 The Persistent Kernel Approach

Unlike traditional kernel execution (many small kernels launched sequentially), Megakernels launches **one persistent kernel** that runs an interpreter:

```
┌─────────────────────────────────────────────────────────────────────┐
│ TRADITIONAL APPROACH                                                │
│                                                                     │
│ Launch Kernel A → Wait → Launch Kernel B → Wait → Launch Kernel C  │
│      └─────────────── High launch overhead ─────────────────┘       │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│ MEGAKERNELS APPROACH                                                │
│                                                                     │
│ Launch Interpreter ─┬─ Execute Instruction A                       │
│                     ├─ Execute Instruction B                        │
│                     ├─ Execute Instruction C                        │
│                     └─ ... (no kernel launch overhead)              │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.2 Instruction Format

Each instruction is a fixed 32-integer array (128 bytes) where:
- `instruction[0]` = **opcode** (operation type)
- `instruction[1..31]` = **arguments** (layer index, batch indices, etc.)

```cpp
// From instructions.hpp
static constexpr int INTS_PER_INSTRUCTION = 32;

class Instruction {
    virtual int opcode() const noexcept = 0;
    virtual int prev_opcode() const noexcept = 0;  // For dependency tracking
    virtual Pool pool() const noexcept;            // Compute vs Memory
    virtual void write32(int* dst) const noexcept = 0;
};
```

### 1.3 Instruction Set (Llama-70B Example)

| Opcode | Instruction | Type | Description |
|--------|-------------|------|-------------|
| 0 | NoOp | - | No operation (padding) |
| 1 | AttnNorm | Memory | RMS normalization before attention |
| 2 | QKV_RopeAppend | Compute | QKV projection + RoPE + KV cache append |
| 3 | AttentionPrefill | Compute | Prefill attention computation |
| 4 | AttentionDecode | Compute | Decode attention computation |
| 5 | O_ProjResidual | Compute | Output projection + residual |
| 6 | MLP_Norm | Memory | RMS normalization before MLP |
| 7 | GateSilu | Compute | Gate MLP + SiLU activation |
| 8 | UpMatMul | Compute | Up projection + elementwise multiply |
| 9 | DownProjResidual | Compute | Down projection + residual |
| 10 | LM_Head_Norm | Memory | Final layer normalization |
| 11 | LM_Head | Compute | LM head projection |
| 12 | IncBarrier | Memory | Increment barrier counter |
| 13 | AllDeviceBarrier | Memory | Cross-device synchronization |
| -1 | Die | Compute | Terminate interpreter |

**Key Insight**: Instructions are classified as **Compute** or **Memory** pool, enabling wave interleaving (overlapping compute with memory operations).

## 2. Warp Specialization

### 2.1 The Five Warp Roles

The megakernel uses **heavy warp specialization** where different warps have different roles:

```cpp
// From megakernel.cuh:121-143
if (kittens::warpid() < config::NUM_CONSUMER_WARPS) {
    ::megakernel::consumer::main_loop<...>(g, mks);  // 16 warps: compute
} else {
    switch (kittens::warpgroup::warpid()) {
    case 0: ::megakernel::loader::main_loop<...>(g, mks);    // 1 warp: load data
    case 1: ::megakernel::storer::main_loop<...>(g, mks);    // 1 warp: store results
    case 2: ::megakernel::launcher::main_loop<...>(g, mks);  // 1 warp: (unused)
    case 3: ::megakernel::controller::main_loop<...>(g, mks); // 1 warp: orchestrate
    }
}
```

```
┌─────────────────────────────────────────────────────────────────────┐
│ WARP SPECIALIZATION (20 warps total = 640 threads)                 │
│                                                                     │
│ ┌─────────────────────────────────────────────────────────────────┐ │
│ │ CONTROLLER WARP (1 warp)                                       │ │
│ │ - Fetch next instruction from global memory                     │ │
│ │ - Setup semaphores for inter-warp signaling                    │ │
│ │ - Manage virtual page allocation                                │ │
│ │ - Store timing data                                             │ │
│ └─────────────────────────────────────────────────────────────────┘ │
│                                                                     │
│ ┌────────────────┐ ┌────────────────┐ ┌────────────────────────────┐│
│ │ LOADER (1)     │ │ STORER (1)     │ │ CONSUMER (16 warps)       ││
│ │ Load input     │ │ Store output   │ │ Matrix multiply, GEMM,    ││
│ │ operands from  │ │ to global      │ │ Attention, etc.           ││
│ │ global memory  │ │ memory         │ │                           ││
│ └────────────────┘ └────────────────┘ └────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 Independent Progress

**Critical Insight**: Each warp type runs **independently** and can pipeline across instruction boundaries:

```
Timeline:
         ┌─Instruction N─┐┌─Instruction N+1─┐┌─Instruction N+2─┐
Loader:  │ L─────────────│L─────────────────│L────────────────│
Consumer:│    C──────────│───C──────────────│───C─────────────│
Storer:  │        S──────│─────S────────────│─────S───────────│
         └───────────────┘└─────────────────┘└────────────────┘
                         ^overlap^           ^overlap^
```

When the loader finishes loading for instruction N, it immediately starts loading for instruction N+1, even while consumers are still computing N.

### 2.3 Worker Implementation Pattern

```cpp
// From util.cuh:328-365 (MAKE_WORKER macro)
template <typename config, typename globals, typename... ops>
__device__ void main_loop(const globals &g, state<config> &mks) {
    for (mks.instruction_index = 0, mks.instruction_ring = 0;
         mks.instruction_index < num_iters;
         mks.next_instruction()) {

        mks.await_instruction();  // Wait for controller signal
        if (mks.instruction()[0] == -1) break;  // Die instruction

        // Dispatch to correct operation based on opcode
        dispatch_op<...>::run(mks.instruction()[0], g, mks);
    }
}
```

## 3. Virtual Memory Page System

### 3.1 The Problem

Shared memory is limited (~200KB). Different instructions need different amounts. Without coordination, a loader for instruction N+1 might overwrite data still needed by consumers for instruction N.

### 3.2 Page-Based Solution

Shared memory is divided into **pages** (typically 16KB each):

```cpp
// From config.cuh
static constexpr int PAGE_SIZE = 16384;  // 16KB per page
static constexpr int NUM_PAGES = DYNAMIC_SHARED_MEMORY / PAGE_SIZE;  // 13 pages
```

### 3.3 Logical vs Physical Page IDs

**Key Insight**: Instructions use **logical page IDs** (0, 1, 2...) that are mapped to **physical page IDs** at runtime:

```
Instruction N:     Uses physical pages [0, 5, 3, 7]
                   Releases in order: 3 → 0 → 7 → 5

Instruction N+1:   Logical page 0 → physical page 3 (first released by N)
                   Logical page 1 → physical page 0 (second released by N)
                   ...
```

```cpp
// From page_allocator.cuh
// The controller tracks which physical pages are free and assigns them
kvms.pid_order()[laneid] =
    kvms.all_instructions[last_instruction_ring].pid_order[lid];
```

### 3.4 Page Semaphores

Each page has a semaphore to signal when it's available:

```cpp
// From util.cuh:202-215
__device__ inline void wait_page_ready(int pid) {
    for (int i = 0; i < config::INSTRUCTION_PIPELINE_STAGES_BITS; i++) {
        auto bit = (instruction_index >> i) & 1;
        kittens::wait(page_finished[pid][i], bit);
    }
}

__device__ inline void finish_page(int pid, int count) {
    for (int i = 0; i < config::INSTRUCTION_PIPELINE_STAGES_BITS; i++) {
        arrive(page_finished[pid][i], count);
    }
}
```

## 4. Inter-Instruction Pipelining

### 4.1 Pipeline Stages

The configuration defines how many instructions can be "in flight":

```cpp
// From config.cuh
static constexpr int INSTRUCTION_PIPELINE_STAGES = 2;
```

This means:
- While instruction N is being executed (compute phase)
- Instruction N+1 can be set up (controller) and start loading (loader)

### 4.2 Ring Buffer Management

Instructions are managed in a circular buffer:

```cpp
// From controller.cuh
for (mks.instruction_index = 0, mks.instruction_ring = 0;
     mks.instruction_index < num_iters;
     mks.instruction_index++,
     mks.instruction_ring = ring_advance<INSTRUCTION_PIPELINE_STAGES>(mks.instruction_ring))
{
    // Wait for previous instruction in this slot to finish
    if (mks.instruction_index >= INSTRUCTION_PIPELINE_STAGES) {
        kittens::wait(mks.instruction_finished[mks.instruction_ring], phasebit);
        // Invalidate semaphores from previous instruction
        // Store timing data
    }

    // Fetch new instruction
    load_instructions<...>(&mks.instruction()[0], global_instruction_index, g);

    // Setup page mapping
    // Construct semaphores

    // Signal that instruction is ready
    arrive(mks.instruction_arrived[mks.instruction_ring], 1);
}
```

### 4.3 Performance Impact

From the paper:
> "Inter-instruction pipelining alone provides a **6-14% improvement** by eliminating memory pipeline bubbles between operations."

## 5. Global Work Queue (Work Stealing)

### 5.1 Static vs Dynamic Scheduling

**Static Scheduling** (default):
- Pre-assign instructions to SMs in round-robin
- Each SM has its own queue: `instruction_tensor[sm_idx][instr_idx]`
- Predictable, low overhead

**Dynamic Scheduling** (Global Work Queue):
- All SMs share a single queue: `instruction_tensor[global_idx]`
- Each SM atomically fetches the next instruction
- Better load balancing when work is uneven

### 5.2 Implementation

```cpp
// From controller.cuh:71-80
if constexpr (config::ENABLE_GLOBAL_WORK_QUEUE) {
    wait(kvms.instruction_fetch_ready, (kvms.instruction_index % 2) ^ 1);
    // Atomically get next instruction index
    if (laneid == 0)
        global_instruction_index = atomicAdd(&g.global_instruction_index[{}], 1);
    global_instruction_index = __shfl_sync(0xffffffff, global_instruction_index, 0);
} else {
    global_instruction_index = kvms.instruction_index;
}
```

### 5.3 When to Use

From the paper:
> "For small batch sizes (64-512), static scheduling is preferred (1-2% faster).
> For large batch sizes (4096-8192), global work queue provides **14.2% improvement** due to better load balancing across variable-length sequences."

```cpp
// From config.cuh
static constexpr bool ENABLE_GLOBAL_WORK_QUEUE = false;
static constexpr int GLOBAL_WORK_QUEUE_PARTITIONS = 1;  // Can partition for locality
```

## 6. Wave Interleaving

### 6.1 Concept

Different operation types (Compute vs Memory) can run **simultaneously** on different SMs:

```
Without Wave Interleaving:
SM0: [Norm][Norm][Norm][QKV][QKV][QKV][Attn][Attn][Attn]
SM1: [Norm][Norm][Norm][QKV][QKV][QKV][Attn][Attn][Attn]
     ^^^^Memory^^^^    ^^^^Compute^^^^    ^^^^Compute^^^^

With Wave Interleaving:
SM0: [Norm][QKV][Norm][QKV][Attn][Norm][QKV][Attn][Attn]
SM1: [QKV][Norm][QKV][Norm][QKV][Attn][Norm][Attn][Attn]
     ^^^^ Memory & Compute interleaved ^^^^
```

### 6.2 Pool Classification

```cpp
// From instructions.hpp
enum class Pool : uint8_t { None = 0, Compute = 1, Memory = 2 };

class ComputeInstruction : public Instruction {
    Pool pool() const noexcept override { return Pool::Compute; }
};

class MemoryInstruction : public Instruction {
    Pool pool() const noexcept override { return Pool::Memory; }
};
```

### 6.3 Interleaving Algorithm

```cpp
// From scheduling.cpp:407-545
std::vector<std::unique_ptr<Instruction>> interleave_instruction_waves(
    std::vector<std::vector<std::unique_ptr<Instruction>>>& instruction_waves,
    const std::vector<int>& wave_buffer_sizes,
    int overlap_buffer_size)
{
    // 1. Calculate how much each wave can overlap with neighbors
    // 2. Partition each wave into (pre, middle, post) sections
    // 3. Interleave the pre/post sections with adjacent waves
    // 4. Keep middle sections intact
}
```

### 6.4 Performance Impact

From the paper:
> "Wave interleaving provides **6.4% improvement** at batch size 8192 by better utilizing compute and memory units simultaneously."

## 7. Synchronization Mechanisms

### 7.1 Barrier Counters (Cross-SM Dependencies)

For data dependencies between SMs, Megakernels uses spin-loop counting:

```cpp
// Producer SM (after completing work)
atomicAdd(&barrier_counter[layer][row], 1);

// Consumer SM (before starting dependent work)
while (barrier_counter[layer][row] < expected_count) {
    // Spin with backoff
    __nanosleep(config::GMEM_SPIN_LOOP_SLEEP_NANOS);
}
```

### 7.2 Semaphores (Intra-SM Coordination)

For coordination within an SM (between warps), use GPU semaphores:

```cpp
// From util.cuh
kittens::semaphore instruction_arrived[INSTRUCTION_PIPELINE_STAGES];
kittens::semaphore instruction_finished[INSTRUCTION_PIPELINE_STAGES];
kittens::semaphore page_finished[NUM_PAGES][INSTRUCTION_PIPELINE_STAGES_BITS];
```

### 7.3 Barrier Instructions

For explicit synchronization points:

```cpp
// From instructions.hpp
class IncBarrier final : public MemoryInstruction {
    int opcode() const noexcept override { return 12; }
    // Increments a barrier counter when executed
};

class AllDeviceBarrier final : public MemoryInstruction {
    int opcode() const noexcept override { return 13; }
    // Waits for all GPUs to reach this point
};
```

## 8. Scheduling System

### 8.1 Three-Level Hierarchy

```
┌─────────────────────────────────────────────────────────────────────┐
│ Level 1: MODEL SCHEDULING                                           │
│ schedule_model() → [IncBarrier, Layer0, Layer1, ..., LM_Head, Die] │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Level 2: LAYER SCHEDULING                                           │
│ schedule_layer() → [[AttnNorm], [QKV], [Prefill], [Decode], ...]   │
│                     (each [] is a "wave")                           │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Level 3: OPERATION SCHEDULING                                        │
│ schedule_matmul() → [Inst0, Inst1, Inst2, ...]                      │
│ schedule_norm() → [Inst0, Inst1, ...]                               │
│ (handles tiling, batching, etc.)                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 8.2 SM Assignment Strategies

**Round-Robin** (Static):
```cpp
// From scheduling.cpp:907-922
std::vector<std::vector<std::unique_ptr<Instruction>>>
round_robin_assign_to_sms(
    std::vector<std::unique_ptr<Instruction>>& instructions, int sm_count)
{
    std::vector<std::vector<std::unique_ptr<Instruction>>> buckets(sm_count);
    for (size_t i = 0; i < instructions.size(); ++i) {
        buckets[i % sm_count].push_back(std::move(instructions[i]));
    }
    return buckets;
}
```

**Supergroups** (for locality):
```cpp
// From scheduling.cpp:28-49
// Group batch tiles together before output tiles for better cache locality
auto pairs = make_supergroup(local_batch_idx_range, local_output_idx_range,
                             supergroup_size);  // typically 8
```

## 9. Multi-GPU Support

### 9.1 Tensor Parallelism

Llama-70B is sharded across 8 GPUs with:
- Data parallel: Different batch slices per GPU
- Tensor parallel: Different attention heads per GPU

```cpp
// From scheduling.cpp
int num_local_batch_blocks;
if (is_split_batch_dim) {
    num_local_batch_blocks = globs.local_batch_blocks(device_idx);
} else {
    num_local_batch_blocks = globs.num_batch_blocks();
}
```

### 9.2 Cross-GPU Synchronization

```cpp
// From instructions.hpp
class AllDeviceBarrier final : public MemoryInstruction {
    int layer_idx;
    int bar_idx;
    // All GPUs wait at this barrier before proceeding
};
```

## 10. Key Takeaways for PTO-ISA

### 10.1 What to Adopt

| Megakernels Pattern | PTO-ISA Adaptation |
|---------------------|-------------------|
| Instruction-interpreter model | AICPU as interpreter, AICore as executor |
| Fixed-size instruction format | WorkDescriptor (24 bytes) - similar concept |
| Warp specialization | AICPU thread specialization (scheduler, dispatcher, etc.) |
| Virtual page system | UB buffer pool management |
| Inter-instruction pipelining | Pipeline AICPU planning with AICore execution |
| Global work queue | Work stealing for dynamic batches |
| Wave interleaving | Overlap memory-bound and compute-bound operations |
| Barrier-based synchronization | Task dependencies via counters/events |

### 10.2 Key Differences

| Megakernels (CUDA) | PTO-ISA (Ascend) |
|-------------------|-----------------|
| GPU interpreter (in-kernel) | AICPU interpreter (separate processor) |
| Warp-level specialization | Core-level specialization |
| Shared memory pages | UB (Unified Buffer) pages |
| ~0 instruction overhead | ~0μs AICPU→AICore dispatch (same!) |
| Spin-loop sync | Hardware event/semaphore |
| All SMs see same shared memory | Each AICore has private UB |

### 10.3 Critical Insights for Design

1. **Cold Start Elimination**: The persistent kernel approach eliminates kernel launch overhead entirely. On Ascend, keeping a persistent AICPU task with pre-registered AICores achieves similar effect.

2. **Pipelining is Key**: Inter-instruction pipelining (6-14% gain) comes from overlapping phases. PTO-ISA can pipeline AICPU planning with AICore execution.

3. **Work Stealing Matters**: At high batch sizes, dynamic scheduling (14% gain) beats static. This supports the need for flexible dispatch strategies.

4. **Wave Interleaving**: Mixing compute and memory operations (6.4% gain) requires classifying operations and intelligent scheduling.

5. **Dependency Tracking**: The `prev_opcode()` method hints at dependency chains. PTO-ISA should support explicit task dependencies.

### 10.4 Questions for Further Investigation

1. **How to express instruction dependencies?** Megakernels uses `prev_opcode()` for implicit chaining. Should PTO-ISA use explicit DAG or implicit sequencing?

2. **AICPU threading model?** Megakernels uses warp specialization. Can AICPU run multiple threads for parallel planning?

3. **Memory management?** Megakernels' page system requires coordinated release. How to manage UB across concurrent AICore tasks?

4. **Work stealing granularity?** Should tasks be stolen individually or in batches?

## 11. Code References

| Component | File | Key Functions/Classes |
|-----------|------|----------------------|
| Main kernel | megakernel.cuh:16-186 | `mk_internal`, `mk` |
| Controller | controller/controller.cuh:14-217 | `main_loop` |
| Page allocator | controller/page_allocator.cuh:10-75 | `page_allocator_loop` |
| Configuration | config.cuh:7-95 | `default_config` |
| State management | util.cuh:120-314 | `state`, `instruction_state_t` |
| Worker pattern | util.cuh:328-366 | `MAKE_WORKER` macro |
| Instructions | instructions.hpp:15-318 | `Instruction` hierarchy |
| Scheduling | scheduling.cpp:720-904 | `schedule_model`, `schedule_layer` |
| Wave interleaving | scheduling.cpp:407-545 | `interleave_instruction_waves` |
| SM assignment | scheduling.cpp:906-922 | `round_robin_assign_to_sms` |

---
*Note Version: 1.0*
*Last Updated: 2025-01-16*
