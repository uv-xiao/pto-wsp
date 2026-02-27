# Research Note 4: PTO-ISA Architecture and Extension Points

## Overview

This note analyzes the PTO-ISA (Parallel Tile Operation ISA) architecture to identify extension points for runtime dynamic workload support. PTO-ISA is a virtual instruction set that provides tile-level abstractions across Ascend platforms (A2/910B, A3/910C, A5/950) and CPU simulation.

**Source Analysis**: `include/pto/common/*.hpp`, `include/pto/npu/{a2a3,a5}/*.hpp`, `kernels/manual/`

## 1. Core Architecture

### 1.1 Instruction Dispatch Mechanism

PTO-ISA uses a macro-based platform dispatch system:

```cpp
// pto_instr.hpp
#define MAP_INSTR_IMPL(API, ...) API##_IMPL(__VA_ARGS__)

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1>
PTO_INST RecordEvent TADD(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1) {
    TSYNC(events...);
    MAP_INSTR_IMPL(TADD, dst, src0, src1);  // → TADD_IMPL(dst, src0, src1)
    return {};
}
```

Platform selection via preprocessor:

| Define | Platform | Implementation Path |
|--------|----------|---------------------|
| `MEMORY_BASE` | A2/A3 (910B/910C) | `include/pto/npu/a2a3/*.hpp` |
| `REGISTER_BASE` | A5 (950) | `include/pto/npu/a5/*.hpp` |
| `__CPU_SIM` | CPU Simulation | `include/pto/cpu/*.hpp` |

**Extension Point**: Add new `*_IMPL` functions in platform-specific directories.

### 1.2 Type System

#### Tile Types

```cpp
enum class TileType {
    Vec,    // Vector tile (UB buffer)
    Mat,    // Matrix tile (L1 buffer)
    Left,   // Left operand for Cube (L0A)
    Right,  // Right operand for Cube (L0B)
    Acc,    // Accumulator (L0C)
    Bias,   // Bias data
};
```

#### Tile Template

```cpp
template <TileType Type_, typename Element_,
          int Rows_, int Cols_,
          BLayout BufferLayout_,
          int ValidRows_, int ValidCols_,
          SLayout StorageLayout_ = SLayout::RowMajor,
          PadValue PadVal_ = PadValue::Null,
          int SFractalSize_ = 0>
class Tile {
    static constexpr int Rows = Rows_;          // Compile-time shape
    static constexpr int Cols = Cols_;
    int validRow_;                               // Runtime valid size
    int validCol_;
    // ...
};
```

**Key Insight**: Tile shape is fixed at compile time (`Rows`, `Cols`), but **valid region** is runtime (`validRow_`, `validCol_`).

#### Convenient Type Aliases

```cpp
// Tile aliases for common use cases
template <typename T, int R, int C, int VR = R, int VC = C>
using VecTile = Tile<TileType::Vec, T, R, C, BLayout::RowMajor, VR, VC>;

template <typename T, int R, int C, int VR = R, int VC = C>
using MatTile = Tile<TileType::Mat, T, R, C, BLayout::ColMajor, VR, VC>;

template <typename T, int R, int C, int VR = R, int VC = C>
using TileLeft = Tile<TileType::Left, T, R, C, BLayout::ColMajor, VR, VC, SLayout::RowMajor>;

template <typename T, int R, int C, int VR = R, int VC = C>
using TileRight = Tile<TileType::Right, T, R, C, BLayout::RowMajor, VR, VC, SLayout::ColMajor>;

template <typename T, int R, int C, int VR = R, int VC = C>
using TileAcc = Tile<TileType::Acc, T, R, C, BLayout::RowMajor, VR, VC>;
```

### 1.3 GlobalTensor with Dynamic Shapes

**Critical Finding**: PTO-ISA already supports `DYNAMIC` dimensions:

```cpp
constexpr int DYNAMIC = -1;

template <int N1 = DYNAMIC, int N2 = DYNAMIC, int N3 = DYNAMIC, int N4 = DYNAMIC, int N5 = DYNAMIC>
struct Shape {
    static constexpr int staticShape[5] = {N1, N2, N3, N4, N5};
    int shape[5] = {1};  // Runtime storage for DYNAMIC dimensions

    // Constructors handle mixed static/dynamic
    Shape(int n1, int n2, int n3, int n4, int n5) {
        if constexpr (N1 == DYNAMIC) shape[0] = n1;
        if constexpr (N2 == DYNAMIC) shape[1] = n2;
        // ...
    }
};

template <typename Element_, typename Shape_, typename Stride_, Layout Layout_ = Layout::ND>
struct GlobalTensor {
    // Mixed static/dynamic shape access
    int GetShape(const int dim) {
        if constexpr (staticShape[dim] == DYNAMIC) {
            return shape_.shape[dim];  // Runtime
        } else {
            return staticShape[dim];   // Compile-time
        }
    }
};
```

**Extension Opportunity**: The `DYNAMIC` mechanism exists but is primarily used for GlobalTensor, not for iteration space or work distribution.

## 2. Memory Hierarchy Mapping

### 2.1 Buffer Qualifiers

```cpp
// Memory address space qualifiers (CCE intrinsics)
__gm__   // Global Memory (HBM/DDR)
__cbuf__ // L1 Buffer (for MatTile)
__ubuf__ // Unified Buffer (for VecTile)
__ca__   // L0A (Left matrix input)
__cb__   // L0B (Right matrix input)
__cc__   // L0C (Accumulator output)
```

### 2.2 Memory Movement Instructions

| Instruction | Source → Dest | Description |
|-------------|---------------|-------------|
| `TLOAD` | GM → L1/UB | Load from global memory |
| `TSTORE` | L1/UB → GM | Store to global memory |
| `TEXTRACT` | L1 → L0A/L0B | Extract tile for Cube input |
| `TMOV` | Within UB | Move within unified buffer |

### 2.3 TLOAD Implementation Patterns

```cpp
// TLOAD supports multiple layout transformations
template <typename TileData, typename GlobalData>
AICORE void TLOAD_IMPL(TileData &dst, GlobalData &src) {
    // Layout dispatch at compile time
    if constexpr (isSameLayout) {
        TLoadGm2ub<TileData, GlobalData>(...);  // ND2ND, DN2DN, NZ2NZ
    } else if constexpr (GlobalData::layout == Layout::ND &&
                         GetTileLayoutCustom<TileData>() == TileLayoutCustom::NZ) {
        TLoadGm2L1Nd2nz<TileData, GlobalData>(...);  // ND → NZ (fractal)
    }
    // ...
}
```

**Key Observation**: TLOAD handles shape/stride at runtime through `GetShape()` and `GetStride()` calls, enabling strided access patterns.

## 3. Kernel Programming Patterns

### 3.1 SPMD Work Distribution

```cpp
// From gemm_performance_kernel.cpp
template <typename T, int m, int k, int n, uint32_t singleCoreM, uint32_t singleCoreK, uint32_t singleCoreN>
AICORE inline void InitGMOffsets(__gm__ T *&currentDst, ...) {
    // Work partition (SPMD-style):
    // Each core owns a contiguous C tile of shape [singleCoreM, singleCoreN]
    constexpr uint32_t mIter = m / singleCoreM;
    uint32_t mIterIdx = get_block_idx() % mIter;  // Current core's M index
    uint32_t nIterIdx = get_block_idx() / mIter;  // Current core's N index

    uint64_t gmOffsetA = mIterIdx * singleCoreM * k;
    uint64_t gmOffsetB = nIterIdx * k * singleCoreN;
    uint64_t gmOffsetC = mIterIdx * singleCoreM * n + nIterIdx * singleCoreN;
    // ...
}
```

**Current Limitation**: `get_block_idx()` returns the hardware core index, but **work → core mapping is implicit** (one-to-one). There's no runtime work assignment mechanism.

### 3.2 Pipeline Synchronization

```cpp
// Event-based synchronization between pipeline stages
template <pipe_t srcPipe, pipe_t dstPipe>
AICORE inline void SetFlag(uint32_t id) {
    set_flag(srcPipe, dstPipe, static_cast<event_t>(id));
}

template <pipe_t srcPipe, pipe_t dstPipe>
AICORE inline void WaitFlag(uint32_t id) {
    wait_flag(srcPipe, dstPipe, static_cast<event_t>(id));
}

// Typical pipeline:
// TLOAD (MTE2) → TEXTRACT (MTE1) → TMATMUL (M) → TSTORE (FIX)
```

### 3.3 Double Buffering Pattern

```cpp
constexpr uint32_t BUFFER_NUM = 2;

// Double-buffered tiles
Tile<...> aMatTile[BUFFER_NUM];
Tile<...> bMatTile[BUFFER_NUM];

// Ping-pong between buffers
uint8_t mte2DBFlag = 0;
for (uint32_t kIter = 0; kIter < totalK; ++kIter) {
    WaitFlag<PIPE_MTE1, PIPE_MTE2>(mte2DBFlag);
    TLOAD(aMatTile[mte2DBFlag], gmA);
    mte2DBFlag = (mte2DBFlag == 0) ? 1 : 0;  // Toggle
}
```

### 3.4 FlashAttention Streaming Softmax

```cpp
// From pto_macro_fa_softmax.hpp
template <bool init, int HEAD_SIZE, typename ReduceTile, typename DataTile>
AICORE inline void pto_macro_fa_softmax(
    DataTile x_exp, DataTile input_x,
    ReduceTile local_max, ReduceTile local_sum,
    ReduceTile new_global_max, ReduceTile new_global_sum, ...) {

    if constexpr (init) {
        // First tile: initialize running state
        TROWMAX(new_global_max, input_x, tmp);
        TROWEXPANDSUB(tmp, input_x, new_global_max);
        TMULS(tmp, tmp, scale);
        TEXP(p_tile_f32, tmp);
        TROWSUM(new_global_sum, p_tile_f32, tmp);
    } else {
        // Subsequent tiles: update and merge state
        TROWMAX(local_max, input_x, tmp);
        TMAX(local_max, local_max, new_global_max);  // Update max
        // ... rescale and accumulate
    }
}
```

**Key Pattern**: Online softmax state (`max`, `sum`) is maintained in ReduceTile between loop iterations, exactly like FlashInfer's `state_t`.

## 4. Extension Point Analysis

### 4.1 What Already Exists

| Feature | Status | Location |
|---------|--------|----------|
| Dynamic GlobalTensor shapes | ✅ Exists | `Shape<DYNAMIC, ...>` |
| Runtime valid region | ✅ Exists | `Tile::validRow_`, `validCol_` |
| Multi-platform dispatch | ✅ Exists | `MAP_INSTR_IMPL` macro |
| Pipeline synchronization | ✅ Exists | `set_flag`/`wait_flag` |
| Online softmax state | ✅ Exists | FlashAttention macros |

### 4.2 What's Missing for Dynamic Workloads

| Feature | Status | Needed For |
|---------|--------|------------|
| Runtime work assignment | ❌ Missing | Variable batch/seq handling |
| Descriptor-based dispatch | ❌ Missing | FlashInfer-style execution |
| Dynamic loop bounds | ❌ Missing | Variable KV length |
| Work index arrays | ❌ Missing | Block → work mapping |
| Multi-tier kernel selection | ❌ Missing | Optimized kernel variants |

### 4.3 Potential Extension Points

#### 4.3.1 Iteration Space Extension

```cpp
// Proposed: IterationSpace template with dynamic bounds
template <int StaticBatch = DYNAMIC, int StaticSeq = DYNAMIC, int StaticHead = DYNAMIC>
struct IterationSpace {
    static constexpr int staticBatch = StaticBatch;
    int batch;   // Runtime if DYNAMIC
    int seq;     // Runtime if DYNAMIC
    int head;    // Compile-time or runtime

    // Work enumeration
    int GetTotalWork() const;
    WorkDescriptor GetWork(int workIdx) const;
};
```

#### 4.3.2 Work Descriptor Extension

```cpp
// Proposed: Descriptor for runtime work specification
struct WorkDescriptor {
    uint32_t request_idx;    // Which request in batch
    uint32_t chunk_idx;      // Which chunk of sequence
    uint32_t kv_start;       // Start of KV range
    uint32_t kv_end;         // End of KV range (exclusive)
    uint32_t tier;           // Which kernel tier to use
};

// Global memory buffer for descriptors
using DescriptorBuffer = GlobalTensor<WorkDescriptor, Shape<DYNAMIC>, Stride<1>, Layout::ND>;
```

#### 4.3.3 Planning Phase Extension

```cpp
// Proposed: Host-side planning API
template <typename IterSpace>
class WorkPlanner {
public:
    // Binary search for optimal chunking (like FlashInfer)
    void PlanChunking(const int* kv_lengths, int batch_size);

    // Generate work descriptors
    void GenerateDescriptors(DescriptorBuffer& out);

    // Get total work count
    int GetTotalWork() const;
};
```

#### 4.3.4 Kernel Tier Extension

```cpp
// Proposed: Multi-tier kernel definition
#define PTO_TIER_DEF(name, tier_id, seq_range, kernel_fn) \
    template <> struct KernelTier<name, tier_id> { \
        static constexpr int min_seq = seq_range.first; \
        static constexpr int max_seq = seq_range.second; \
        static constexpr auto kernel = kernel_fn; \
    };

// Example usage
PTO_TIER_DEF(decode_attention, 0, {1, 1024}, decode_kernel_1k);
PTO_TIER_DEF(decode_attention, 1, {1025, 4096}, decode_kernel_4k);
PTO_TIER_DEF(decode_attention, 2, {4097, 16384}, decode_kernel_16k);
```

## 5. Comparison with FlashInfer

| Concept | FlashInfer | PTO-ISA Current | PTO-ISA Extension |
|---------|------------|-----------------|-------------------|
| Work assignment | `request_indices[]` | `get_block_idx()` | `WorkDescriptor` buffer |
| Chunk bounds | `kv_tile_indices[]` | Compile-time loops | `kv_start`, `kv_end` in descriptor |
| Tier selection | N/A (single kernel) | N/A | `tier` field in descriptor |
| State merging | `state_t::merge()` | FlashAttention macros | Already present |
| Planning | CPU `PrefillSplit...` | N/A | `WorkPlanner` |

## 6. Implementation Strategy

### 6.1 Phase 1: Descriptor Infrastructure

1. Add `WorkDescriptor` struct to `pto_tile.hpp`
2. Add descriptor buffer type alias
3. Implement descriptor read in kernel preamble

### 6.2 Phase 2: Iteration Space

1. Define `IterationSpace` template with `DYNAMIC` support
2. Implement work enumeration (`GetTotalWork`, `GetWork`)
3. Add compile-time/runtime mixed iteration

### 6.3 Phase 3: Planning API

1. Implement `WorkPlanner` for host-side planning
2. Add binary search chunk sizing (FlashInfer pattern)
3. Integrate with CANN compilation flow

### 6.4 Phase 4: Multi-Tier Support

1. Define tier registration macros
2. Implement tier selection based on descriptor
3. Add cost model for tier selection

## 7. Key Takeaways

### 7.1 Strengths of Current PTO-ISA

- **Clean abstraction**: Tile-level operations hide hardware details
- **Dynamic shapes exist**: `DYNAMIC` constant and runtime `Shape`
- **Online algorithms**: FlashAttention streaming softmax pattern
- **Multi-platform**: Same code for CPU/A2A3/A5

### 7.2 Gaps for Dynamic LLM Workloads

- **Static work assignment**: Core index = work index (no indirection)
- **Compile-time loops**: Loop bounds must be known at compile time
- **No descriptor infrastructure**: No mechanism for runtime work specification
- **No tier selection**: Single kernel version per problem

### 7.3 Implementation Choices

The extension can reuse existing PTO-ISA patterns:

1. **DYNAMIC constant**: Reuse `DYNAMIC = -1` for mixed static/runtime dimensions (implementation choice, not design principle)
2. **Descriptor-based bounds**: `WorkDescriptor` for runtime work specification
3. **Tier dispatch**: Compile-time kernel variants with runtime selection

### 7.4 Design Principles for Extension

The core design principles are:

1. **Separate work specification from execution**: Define what work exists at planning time, instantiate at execution time
2. **Support both SPMD and MPMD**: Single program (all cores same kernel) and multiple programs (different kernels on different cores)
3. **Descriptor-based dispatch**: Work assignment via index arrays, not control flow
4. **Leverage AICPU**: 0μs dispatch from AICPU to AICore enables on-device planning

---
*Note Version: 1.1*
*Last Updated: 2024-01-14*
