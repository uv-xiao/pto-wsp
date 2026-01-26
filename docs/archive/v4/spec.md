# PTO Workload-Schedule Programming (PTO-WSP): Formal Specification v4

## 1. Introduction

### 1.1 Purpose

This specification defines the PTO Workload-Schedule Programming (PTO-WSP), a set of abstractions and APIs that enable efficient execution of dynamic LLM workloads on Ascend NPU. The extension provides FlashInfer-equivalent capabilities while preserving PTO-ISA's type safety and compile-time guarantees.

### 1.2 Scope

This specification covers:
- Core data structures and their memory layouts
- Type system extensions for dynamic dimensions
- Planning algorithms and APIs
- Kernel interface and dispatch mechanisms
- Integration with existing PTO-ISA infrastructure

### 1.3 Conformance

Implementations MUST support all REQUIRED features. Implementations SHOULD support RECOMMENDED features. OPTIONAL features MAY be implemented.

### 1.4 Notation

- `REQUIRED`: Feature must be implemented
- `RECOMMENDED`: Feature should be implemented unless there is a specific reason not to
- `OPTIONAL`: Feature may be implemented
- `SHALL/MUST`: Absolute requirement
- `SHOULD`: Recommendation
- `MAY`: Permitted but not required

---

## 2. Constants and Definitions

### 2.1 Namespace

All symbols defined in this specification SHALL be placed in the `pto::runtime` namespace.

```cpp
namespace pto {
namespace runtime {
    // All specification symbols
}  // namespace runtime
}  // namespace pto
```

### 2.2 DYNAMIC Constant

**REQUIRED**

```cpp
constexpr int DYNAMIC = -1;
```

The `DYNAMIC` constant indicates that a dimension size is determined at runtime rather than compile time. This value MUST be consistent with any existing `DYNAMIC` constant in PTO-ISA.

### 2.3 Platform Macros

**REQUIRED for NPU implementations**

| Macro | Description |
|-------|-------------|
| `AICORE` | Function attribute for AICore execution |
| `AICPU` | Function attribute for AICPU execution |
| `__gm__` | Global memory address space qualifier |
| `__CPU_SIM` | Defined when building for CPU simulation |

### 2.4 Hardware Model

**REQUIRED** understanding for correct architecture decisions.

```
1A (Host CPU)
  └── 16B (AICPU) ── ~3μs latency from Host
        └── All AICPU Threads ── ~0μs latency to AICores (register-based)
              ├── 24C (Cube Cores)
              └── 48D (Vector Cores)
```

**Critical Latency Model**:

| Dispatch Path | Latency | Notes |
|---------------|---------|-------|
| Host → AICPU | ~3μs | Cross-domain communication |
| **AICPU → AICore** | **~0μs** | Register-based (ALL AICPU threads) |

**Key Insight**: ALL AICPU→AICore dispatch has ~0μs latency. The bottleneck is not dispatch latency but **task generation throughput** (single thread generates tasks serially).

**Design Implication**: Descriptor generation SHOULD occur on AICPU (not host) to achieve zero-latency dispatch. Use multiple AICPU threads for parallel task generation.

### 2.5 Programming Models

**REQUIRED** support for both SPMD and MPMD:

| Model | Description | Use Case |
|-------|-------------|----------|
| **SPMD** | All cores execute same kernel, different data | Uniform workloads |
| **MPMD** | Different cores execute different kernels | Multi-tier, MoE, pipelines |

---

## 3. Type System

### 3.1 Dim Template

**REQUIRED**

#### 3.1.1 Synopsis

```cpp
template <int StaticSize = DYNAMIC>
struct Dim {
    static constexpr int static_size = StaticSize;
    static constexpr bool is_dynamic = (StaticSize == DYNAMIC);

    int size;

    constexpr Dim();
    explicit Dim(int runtime_size);
    constexpr int get_size() const;
};
```

#### 3.1.2 Template Parameters

| Parameter | Constraint | Description |
|-----------|------------|-------------|
| `StaticSize` | Integer or `DYNAMIC` | Compile-time size, or `DYNAMIC` for runtime |

#### 3.1.3 Member Specification

**`static constexpr int static_size`**
- Value: The template parameter `StaticSize`
- Constraint: NONE

**`static constexpr bool is_dynamic`**
- Value: `true` if `StaticSize == DYNAMIC`, otherwise `false`

**`int size`**
- Storage: 4 bytes
- Purpose: Runtime size storage (used only when `is_dynamic == true`)

**`constexpr Dim()`**
- Effect: Default constructs a dimension
- Postcondition: `size == StaticSize`

**`explicit Dim(int runtime_size)`**
- Precondition: `is_dynamic == true`
- Effect: Constructs a dynamic dimension with given size
- Constraint: SHALL produce a compile-time error if `is_dynamic == false`

**`constexpr int get_size() const`**
- Returns: `size` if `is_dynamic == true`, otherwise `StaticSize`

#### 3.1.4 Type Aliases

**REQUIRED**

```cpp
template <int N> using StaticDim = Dim<N>;
using DynamicDim = Dim<DYNAMIC>;
```

### 3.2 IterationSpace Template

**REQUIRED**

#### 3.2.1 Synopsis

```cpp
template <typename... Dims>
class IterationSpace {
public:
    static constexpr size_t num_dims = sizeof...(Dims);
    using DimsTuple = std::tuple<Dims...>;

    constexpr IterationSpace();

    template <size_t I> void set_dim(int size);
    template <size_t I> constexpr int get_dim() const;

    int total_work() const;
    void index_to_coords(int work_idx, int* coords) const;
};
```

#### 3.2.2 Template Parameters

| Parameter | Constraint | Description |
|-----------|------------|-------------|
| `Dims...` | Each type SHALL be a `Dim<>` specialization | Dimension types |

#### 3.2.3 Member Specification

**`static constexpr size_t num_dims`**
- Value: Number of dimensions

**`template <size_t I> void set_dim(int size)`**
- Precondition: `0 <= I < num_dims`
- Precondition: `std::tuple_element_t<I, DimsTuple>::is_dynamic == true`
- Effect: Sets the runtime size of dimension `I`
- Constraint: SHALL produce a compile-time error if dimension `I` is static

**`template <size_t I> constexpr int get_dim() const`**
- Precondition: `0 <= I < num_dims`
- Returns: Size of dimension `I`

**`int total_work() const`**
- Returns: Product of all dimension sizes
- Complexity: O(num_dims)

**`void index_to_coords(int work_idx, int* coords) const`**
- Precondition: `0 <= work_idx < total_work()`
- Precondition: `coords` points to array of at least `num_dims` elements
- Effect: Converts linear index to multi-dimensional coordinates
- Postcondition: `coords[i]` contains coordinate for dimension `i`
- Ordering: Row-major (dimension 0 varies fastest)

#### 3.2.4 Common Space Aliases

**RECOMMENDED**

```cpp
template <int Batch = DYNAMIC, int Head = DYNAMIC, int Chunk = DYNAMIC>
using AttentionSpace = IterationSpace<Dim<Batch>, Dim<Head>, Dim<Chunk>>;

template <int Batch = DYNAMIC, int Expert = DYNAMIC>
using MoESpace = IterationSpace<Dim<Batch>, Dim<Expert>>;
```

---

## 4. Work Descriptor

### 4.1 WorkDescriptor Structure

**REQUIRED**

#### 4.1.1 Definition

```cpp
struct alignas(8) WorkDescriptor {
    uint32_t work_id;       // Offset: 0, Size: 4
    uint8_t  tier;          // Offset: 4, Size: 1
    uint8_t  flags;         // Offset: 5, Size: 1
    uint16_t reserved;      // Offset: 6, Size: 2
    uint32_t params[4];     // Offset: 8, Size: 16
    // Total: 24 bytes

    static constexpr uint8_t FLAG_FIRST = 0x01;
    static constexpr uint8_t FLAG_LAST  = 0x02;
    static constexpr uint8_t FLAG_INIT  = 0x04;
};
```

#### 4.1.2 Memory Layout

```
Byte Offset:  0    1    2    3    4    5    6    7
            +----+----+----+----+----+----+----+----+
            |       work_id     |tier|flag| reserved|
            +----+----+----+----+----+----+----+----+
Byte Offset:  8    9   10   11   12   13   14   15
            +----+----+----+----+----+----+----+----+
            |     params[0]     |     params[1]     |
            +----+----+----+----+----+----+----+----+
Byte Offset: 16   17   18   19   20   21   22   23
            +----+----+----+----+----+----+----+----+
            |     params[2]     |     params[3]     |
            +----+----+----+----+----+----+----+----+
```

#### 4.1.3 Field Specification

| Field | Type | Size | Description |
|-------|------|------|-------------|
| `work_id` | `uint32_t` | 4 bytes | Unique identifier for this work unit |
| `tier` | `uint8_t` | 1 byte | Kernel tier index (0-255) |
| `flags` | `uint8_t` | 1 byte | Execution flags (bitfield) |
| `reserved` | `uint16_t` | 2 bytes | Reserved for future use (SHALL be 0) |
| `params[4]` | `uint32_t[4]` | 16 bytes | Pattern-specific parameters |

#### 4.1.4 Flag Values

| Flag | Value | Description |
|------|-------|-------------|
| `FLAG_FIRST` | `0x01` | First chunk of a sequence |
| `FLAG_LAST` | `0x02` | Last chunk of a sequence |
| `FLAG_INIT` | `0x04` | Initialize accumulators |

#### 4.1.5 Alignment Requirements

- Structure alignment: 8 bytes (REQUIRED)
- Total size: 24 bytes (REQUIRED)
- Implementations SHALL ensure `sizeof(WorkDescriptor) == 24`
- Implementations SHALL ensure `alignof(WorkDescriptor) == 8`

### 4.2 Parameter Accessors

**REQUIRED**

Pattern-specific accessors SHALL be provided in the `pto::runtime::params` namespace.

#### 4.2.1 Attention Parameters

```cpp
namespace params {
struct Attention {
    static void set(WorkDescriptor& d, uint32_t request_idx, uint32_t head_idx,
                   uint32_t kv_start, uint32_t kv_len);

    static uint32_t request_idx(const WorkDescriptor& d);  // params[0]
    static uint32_t head_idx(const WorkDescriptor& d);     // params[1]
    static uint32_t kv_start(const WorkDescriptor& d);     // params[2]
    static uint32_t kv_len(const WorkDescriptor& d);       // params[3]
    static uint32_t kv_end(const WorkDescriptor& d);       // params[2] + params[3]
};
}  // namespace params
```

#### 4.2.2 MoE Parameters

**RECOMMENDED**

```cpp
namespace params {
struct MoE {
    static void set(WorkDescriptor& d, uint32_t batch_idx, uint32_t expert_idx,
                   uint32_t token_start, uint32_t token_count);

    static uint32_t batch_idx(const WorkDescriptor& d);    // params[0]
    static uint32_t expert_idx(const WorkDescriptor& d);   // params[1]
    static uint32_t token_start(const WorkDescriptor& d);  // params[2]
    static uint32_t token_count(const WorkDescriptor& d);  // params[3]
};
}  // namespace params
```

---

## 5. Tier Configuration

### 5.1 Tier Template

**REQUIRED**

#### 5.1.1 Synopsis

```cpp
template <int TierID, int MinSize, int MaxSize>
struct Tier {
    static constexpr int id = TierID;
    static constexpr int min_size = MinSize;
    static constexpr int max_size = MaxSize;

    static constexpr bool matches(int size);
};
```

#### 5.1.2 Template Parameters

| Parameter | Constraint | Description |
|-----------|------------|-------------|
| `TierID` | 0-255 | Unique tier identifier |
| `MinSize` | > 0 | Minimum size (inclusive) |
| `MaxSize` | >= MinSize | Maximum size (inclusive) |

#### 5.1.3 Member Specification

**`static constexpr bool matches(int size)`**
- Returns: `true` if `MinSize <= size <= MaxSize`, otherwise `false`

### 5.2 TierConfig Template

**REQUIRED**

#### 5.2.1 Synopsis

```cpp
template <typename... Tiers>
struct TierConfig {
    static constexpr size_t num_tiers = sizeof...(Tiers);

    static constexpr int select_tier(int size);
};
```

#### 5.2.2 Template Parameters

| Parameter | Constraint | Description |
|-----------|------------|-------------|
| `Tiers...` | Each type SHALL be a `Tier<>` specialization | Tier definitions |

#### 5.2.3 Member Specification

**`static constexpr int select_tier(int size)`**
- Returns: `TierID` of first matching tier, or `-1` if no tier matches
- Complexity: O(num_tiers)
- Order: Tiers are checked in template argument order

#### 5.2.4 Standard Tier Configurations

**REQUIRED**

```cpp
using DecodeAttentionTiers = TierConfig<
    Tier<0, 1, 1024>,       // Short: 1-1K
    Tier<1, 1025, 4096>,    // Medium: 1K-4K
    Tier<2, 4097, 16384>,   // Long: 4K-16K
    Tier<3, 16385, 131072>  // Very Long: 16K-128K
>;
```

---

## 6. Work Planner

### 6.1 PlanResult Enumeration

**REQUIRED**

```cpp
enum class PlanResult {
    OK = 0,              // Success
    BUFFER_OVERFLOW,     // Output buffer too small
    UNSUPPORTED_SIZE,    // Size not covered by any tier
    INVALID_PARAMS,      // Invalid input parameters
};
```

### 6.2 PlanConfig Structure

**REQUIRED**

```cpp
struct PlanConfig {
    int chunk_min = 256;           // Minimum chunk size
    int chunk_max = 4096;          // Maximum chunk size
    int max_work_units = 65536;    // Maximum work units
    bool balance_chunks = true;    // Balance chunk sizes
};
```

#### 6.2.1 Field Constraints

| Field | Constraint | Default |
|-------|------------|---------|
| `chunk_min` | > 0 | 256 |
| `chunk_max` | >= chunk_min | 4096 |
| `max_work_units` | > 0 | 65536 |
| `balance_chunks` | NONE | true |

### 6.3 WorkPlanner Template

**REQUIRED**

#### 6.3.1 Synopsis

```cpp
template <typename Space, typename TierCfg, typename PatternParams>
class WorkPlanner {
public:
    explicit WorkPlanner(const PlanConfig& config = {});

    int plan_chunk_size(const int* seq_lens, int batch_size, int num_heads);

    PlanResult generate(
        const int* seq_lens,
        int batch_size,
        int num_heads,
        int chunk_size,
        WorkDescriptor* out_buffer,
        int buffer_capacity,
        int* out_work_count
    );

    int get_total_work(const int* seq_lens, int batch_size, int num_heads, int chunk_size);
};
```

#### 6.3.2 Template Parameters

| Parameter | Constraint | Description |
|-----------|------------|-------------|
| `Space` | IterationSpace specialization | Work space type |
| `TierCfg` | TierConfig specialization | Tier configuration |
| `PatternParams` | Has `set()` method | Parameter accessors |

#### 6.3.3 Constructor

**`explicit WorkPlanner(const PlanConfig& config = {})`**
- Effect: Constructs planner with given configuration
- Postcondition: Planner is ready for use

#### 6.3.4 plan_chunk_size Method

**`int plan_chunk_size(const int* seq_lens, int batch_size, int num_heads)`**

- Precondition: `seq_lens` points to array of `batch_size` elements
- Precondition: `batch_size > 0`, `num_heads > 0`
- Returns: Optimal chunk size within `[chunk_min, chunk_max]`
- Algorithm: Binary search for largest chunk size that yields `<= max_work_units`
- Complexity: O(batch_size * log(chunk_max - chunk_min))

**Algorithm Pseudocode:**

```
function plan_chunk_size(seq_lens, batch_size, num_heads):
    low = config.chunk_min
    high = config.chunk_max

    while low < high:
        mid = (low + high) / 2
        total = count_work_units(seq_lens, batch_size, num_heads, mid)

        if total > config.max_work_units:
            low = mid + 1
        else:
            high = mid

    return low
```

#### 6.3.5 generate Method

**`PlanResult generate(...)`**

- Precondition: `seq_lens` points to array of `batch_size` elements
- Precondition: `out_buffer` points to array of `buffer_capacity` descriptors
- Precondition: `out_work_count` is not null
- Effect: Generates work descriptors for all work units
- Postcondition (on OK): `*out_work_count` contains number of generated descriptors
- Returns: `PlanResult` indicating success or failure mode

**Generation Order:**
1. Iterate over batches (outer)
2. For each batch, iterate over heads
3. For each head, iterate over chunks (inner)

**Descriptor Fields:**
| Field | Value |
|-------|-------|
| `work_id` | Sequential from 0 |
| `tier` | From `TierCfg::select_tier(seq_len)` |
| `flags` | `FLAG_FIRST` for chunk 0, `FLAG_LAST` for last chunk |
| `params` | Set via `PatternParams::set()` |

### 6.4 Convenience Aliases

**REQUIRED**

```cpp
using AttentionPlanner = WorkPlanner<
    AttentionSpace<DYNAMIC, DYNAMIC, DYNAMIC>,
    DecodeAttentionTiers,
    params::Attention
>;
```

### 6.5 AICPU-Based Planning

**REQUIRED for optimal performance**

#### 6.5.1 Execution Location

The `WorkPlanner` class SHOULD be executable on AICPU for zero-latency dispatch:

```cpp
// AICPU manager thread function
AICPU void aicpu_plan_and_dispatch(
    const int* seq_lens,        // In global memory
    int batch_size,
    int num_heads,
    WorkDescriptor* desc_buffer, // In global memory
    /* kernel parameters */
) {
    AttentionPlanner planner;

    // Planning on AICPU (not host)
    int chunk_size = planner.plan_chunk_size(seq_lens, batch_size, num_heads);

    int work_count;
    planner.generate(seq_lens, batch_size, num_heads, chunk_size,
                     desc_buffer, MAX_WORK, &work_count);

    // Zero-latency dispatch from AICPU manager
    for (int i = 0; i < work_count; i++) {
        dispatch_to_aicore(desc_buffer[i], /* kernel params */);
    }
}
```

#### 6.5.2 Parallel Descriptor Generation

For large workloads, descriptor generation SHOULD be parallelized across multiple AICPU threads:

```cpp
// Parallel generation across P AICPU threads
AICPU void parallel_generate(int thread_id, int num_threads, ...) {
    // Each thread generates descriptors for a subset of batches
    int batch_start = (batch_size * thread_id) / num_threads;
    int batch_end = (batch_size * (thread_id + 1)) / num_threads;

    // Generate descriptors for assigned batches
    generate_subset(batch_start, batch_end, ...);
}
```

#### 6.5.3 Why AICPU, Not Host

| Aspect | Host-Based | AICPU-Based |
|--------|------------|-------------|
| Dispatch latency | 3μs per kernel | **0μs** (register) |
| Descriptor copy | Required (H2D) | Not needed |
| Parallelization | Host threads | AICPU threads |
| Data locality | Poor | Excellent |

---

## 7. Kernel Interface

### 7.1 Kernel Function Signature

**REQUIRED**

Kernel functions SHALL have the following general form:

```cpp
template <int Tier>
AICORE void kernel_impl(const WorkDescriptor& desc, /* tensor parameters */);
```

### 7.2 Dispatch Table

**REQUIRED**

Implementations SHALL support static dispatch tables:

```cpp
using KernelFnPtr = void (*)(const WorkDescriptor&, /* parameters */);

constexpr KernelFnPtr dispatch_table[NUM_TIERS] = {
    &kernel_impl<0>,
    &kernel_impl<1>,
    // ...
};
```

### 7.3 Dispatch Mechanism

**REQUIRED**

Kernel dispatch SHALL use table lookup:

```cpp
// O(1) dispatch, no branch
dispatch_table[desc.tier](desc, /* parameters */);
```

### 7.4 Kernel Macros

**RECOMMENDED**

```cpp
// Define tiered kernel
#define PTO_TIERED_KERNEL(name, num_tiers, ...)                               \
    template <int Tier> AICORE void name##_impl(__VA_ARGS__);                 \
    static constexpr KernelFnPtr name##_dispatch[num_tiers] = {               \
        &name##_impl<0>, &name##_impl<1>, &name##_impl<2>, &name##_impl<3>    \
    }

// Dispatch to correct tier
#define PTO_DISPATCH_TIER(name, desc, ...)                                    \
    name##_dispatch[desc.tier](desc, __VA_ARGS__)
```

---

## 8. Memory Management

### 8.1 Descriptor Buffer Allocation

**IMPLEMENTATION-DEFINED**

Implementations SHALL provide a mechanism to allocate descriptor buffers in device-accessible memory.

**RECOMMENDED Interface:**

```cpp
WorkDescriptor* allocate_descriptors(int count);
void free_descriptors(WorkDescriptor* buffer);
```

### 8.2 Buffer Sizing

The required buffer size can be computed as:

```cpp
int required_size = planner.get_total_work(seq_lens, batch_size, num_heads, chunk_size);
```

Conservative estimate (without planning):

```cpp
int max_chunks_per_seq = (max_seq_len + chunk_min - 1) / chunk_min;
int max_work = batch_size * num_heads * max_chunks_per_seq;
```

---

## 9. Integration

### 9.1 Header Organization

**REQUIRED**

```
include/pto/runtime/
├── runtime.hpp              # Main include (aggregates all)
├── dim.hpp                  # Dim template
├── iteration_space.hpp      # IterationSpace template
├── work_descriptor.hpp      # WorkDescriptor and params
├── tier_config.hpp          # TierConfig template
├── work_planner.hpp         # WorkPlanner class
└── kernel_dispatch.hpp      # Dispatch macros
```

### 9.2 Include Model

Users SHALL include the main header:

```cpp
#include <pto/runtime/runtime.hpp>
```

### 9.3 Compatibility

#### 9.3.1 C++ Standard

- Minimum: C++17
- Recommended: C++20 or later
- Required features: `constexpr if`, fold expressions, structured bindings

#### 9.3.2 PTO-ISA Compatibility

- The extension SHALL NOT modify existing PTO-ISA symbols
- The extension SHALL be usable with or without other PTO-ISA features
- Existing kernels SHALL continue to work without modification

---

## 10. CPU Simulation

### 10.1 Simulation Requirements

**REQUIRED**

When `__CPU_SIM` is defined, implementations SHALL:
1. Compile all templates without device-specific attributes (`AICORE`, `AICPU`)
2. Execute descriptor generation on host (simulating AICPU)
3. Simulate kernel dispatch via function calls

### 10.2 Simulation Behavior

The following SHALL be equivalent between CPU and device:
- `WorkDescriptor` memory layout
- `plan_chunk_size()` results
- `generate()` results
- Tier selection

### 10.3 AICPU Simulation

For CPU simulation, AICPU functions are executed on host:
- `AICPU` attribute is ignored
- Global memory pointers become regular host pointers
- Dispatch functions become direct function calls

```cpp
#ifdef __CPU_SIM
#define AICPU  // No-op in simulation
#define aicpu_dispatch_kernel(fn, ...) fn(__VA_ARGS__)  // Direct call
#endif
```

---

## 11. Error Handling

### 11.1 Planning Errors

| Error | Condition | Recovery |
|-------|-----------|----------|
| `BUFFER_OVERFLOW` | `work_count > buffer_capacity` | Allocate larger buffer |
| `UNSUPPORTED_SIZE` | No tier matches sequence length | Extend tier config |
| `INVALID_PARAMS` | Null pointers or invalid sizes | Fix input parameters |

### 11.2 Runtime Errors

Kernel implementations SHOULD handle:
- Out-of-bounds memory access (via bounds checking)
- Uninitialized accumulator state (via FLAG_INIT)

---

## 12. Performance Requirements

### 12.1 Planning Overhead (AICPU-Based)

| Operation | Target (AICPU) | Target (Host) | Notes |
|-----------|----------------|---------------|-------|
| `plan_chunk_size()` | < 50μs for 10K | < 100μs for 10K | AICPU is faster due to data locality |
| `generate()` | < 5μs per 1K | < 10μs per 1K | No H2D copy overhead |
| Parallel generate | O(N/P) | O(N) | P = number of AICPU threads |

### 12.2 Dispatch Overhead

| Dispatch Path | Latency | Notes |
|---------------|---------|-------|
| Host → AICPU | ~3μs | Cross-domain communication |
| **AICPU → AICore** | **~0μs** | Register-based (ALL threads) |
| Descriptor read (AICore) | < 100 cycles | From global memory |
| Tier dispatch (AICore) | < 10 cycles | Table lookup |

**Note**: ALL AICPU→AICore dispatch has ~0μs latency. The bottleneck is task generation throughput, not dispatch latency.

### 12.3 Memory Overhead

| Component | Size |
|-----------|------|
| Per-descriptor | 24 bytes |
| Dispatch table | 8 bytes × num_tiers |

### 12.4 End-to-End Comparison

| Approach | Planning | Dispatch | Task Gen Bottleneck |
|----------|----------|----------|---------------------|
| Host-based | Parallel on host | ~3μs Host→AICPU | None |
| AICPU (single thread) | Serial | ~0μs | **Serial generation** |
| **AICPU (multi-thread)** | **Parallel** | **~0μs** | **None** |

**Recommendation**: Use multiple AICPU threads for parallel task generation to eliminate the serial bottleneck while maintaining ~0μs dispatch latency.

---

## 13. Example Usage

### 13.1 AICPU-Based Planning (Recommended for NPU)

```cpp
#include <pto/runtime/runtime.hpp>

// AICPU manager thread: plan and dispatch with 0μs latency
AICPU void decode_attention_aicpu(
    __gm__ int* kv_lengths,       // In global memory
    int batch_size, int num_heads, int head_dim,
    __gm__ WorkDescriptor* desc_buffer,  // Pre-allocated in GM
    __gm__ half* Q, __gm__ half* K_cache, __gm__ half* V_cache,
    __gm__ half* Output
) {
    using namespace pto::runtime;

    // 1. Create planner (on AICPU)
    AttentionPlanner planner(PlanConfig{
        .chunk_min = 256,
        .chunk_max = 4096,
        .max_work_units = 65536
    });

    // 2. Plan chunk size (AICPU has direct access to GM)
    int chunk_size = planner.plan_chunk_size(kv_lengths, batch_size, num_heads);

    // 3. Generate descriptors (writes directly to GM)
    int work_count;
    planner.generate(
        kv_lengths, batch_size, num_heads, chunk_size,
        desc_buffer, MAX_WORK_BUFFER, &work_count
    );

    // 4. Dispatch to AICores (0μs latency from AICPU manager)
    for (int i = 0; i < work_count; i++) {
        // Register-based dispatch - zero latency
        aicpu_dispatch_kernel(
            decode_attention_entry,
            desc_buffer + i,  // Pointer to descriptor in GM
            Q, K_cache, V_cache, Output, head_dim
        );
    }
}
```

### 13.2 Host-Side Planning (For CPU Simulation)

```cpp
#include <pto/runtime/runtime.hpp>

// Host-side example (for __CPU_SIM or when AICPU unavailable)
void run_decode_attention_host(
    int batch_size, int num_heads, int head_dim,
    const int* kv_lengths,
    half* d_Q, half* d_K_cache, half* d_V_cache, half* d_Output
) {
    using namespace pto::runtime;

    // 1. Create planner
    AttentionPlanner planner(PlanConfig{
        .chunk_min = 256,
        .chunk_max = 4096,
        .max_work_units = 65536
    });

    // 2. Plan chunk size
    int chunk_size = planner.plan_chunk_size(kv_lengths, batch_size, num_heads);

    // 3. Compute buffer size
    int max_work = planner.get_total_work(kv_lengths, batch_size, num_heads, chunk_size);

    // 4. Allocate descriptor buffer
    auto desc_buffer = allocate_descriptors(max_work);

    // 5. Generate descriptors
    int work_count;
    auto result = planner.generate(
        kv_lengths, batch_size, num_heads, chunk_size,
        desc_buffer, max_work, &work_count
    );

    if (result != PlanResult::OK) {
        handle_error(result);
        return;
    }

    // 6. Copy descriptors to device (3μs overhead)
    copy_to_device(desc_buffer, work_count);

    // 7. Launch kernel (3μs overhead per launch)
    launch_kernel(
        decode_attention_entry,
        /*grid=*/dim3(work_count),
        /*block=*/dim3(1),
        desc_buffer, d_Q, d_K_cache, d_V_cache, d_Output, head_dim
    );

    // 8. Cleanup
    synchronize();
    free_descriptors(desc_buffer);
}
```

### 13.2 Device Kernel Example

```cpp
template <int Tier>
AICORE void decode_attention_impl(
    const WorkDescriptor& desc,
    __gm__ half* Q, __gm__ half* K_cache, __gm__ half* V_cache,
    __gm__ half* Output, int head_dim
) {
    using namespace pto::runtime;

    // Extract parameters
    auto req = params::Attention::request_idx(desc);
    auto head = params::Attention::head_idx(desc);
    auto kv_start = params::Attention::kv_start(desc);
    auto kv_len = params::Attention::kv_len(desc);

    // Tier-specific tile size
    constexpr int TILE_K = (Tier == 0) ? 256 :
                           (Tier == 1) ? 512 :
                           (Tier == 2) ? 1024 : 2048;

    // Kernel implementation using PTO-ISA operations...
}

// Entry point
__global__ __aicore__ void decode_attention_entry(
    __gm__ WorkDescriptor* descs,
    __gm__ half* Q, __gm__ half* K_cache, __gm__ half* V_cache,
    __gm__ half* Output, int head_dim
) {
    auto desc = descs[get_block_idx()];
    PTO_DISPATCH_TIER(decode_attention, desc, Q, K_cache, V_cache, Output, head_dim);
}
```

---

## Appendix A: FlashInfer Correspondence

| FlashInfer (CUDA) | PTO Workload-Schedule Programming (PTO-WSP) framework |
|-------------------|----------------------|
| `request_indices[]` | `WorkDescriptor.params[0]` via `params::Attention::request_idx()` |
| `kv_tile_indices[]` | `WorkDescriptor.params[2,3]` via `params::Attention::kv_start()`, `kv_len()` |
| `PrefillSplitQOKVIndptr` | `WorkPlanner::generate()` |
| `BinarySearchKVChunkSize` | `WorkPlanner::plan_chunk_size()` |
| `state_t::merge()` | Online softmax in kernel (FLAG_FIRST/FLAG_LAST) |
| Grid dispatch | `dispatch_table[tier]` lookup |

---

## Appendix B: Version History

| Version | Date | Changes |
|---------|------|---------|
| v4.0 | 2024-01-13 | Initial formal specification |
| v4.1 | 2024-01-13 | Architecture correction: AICPU-based planning |
| v4.2 | 2024-01-14 | **Key fixes**: (1) ALL AICPU→AICore is ~0μs, not just manager threads; (2) DYNAMIC pattern demoted to impl choice; (3) Added SPMD/MPMD programming model support |

---

*Specification Version: 4.2*
*Status: DRAFT*
*Last Updated: 2024-01-14*
