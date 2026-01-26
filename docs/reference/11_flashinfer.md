# FlashInfer: Handling Variable `seq_len` and `batch_size` During KV-Cache Attention

This analysis traces how FlashInfer handles variant parameters like sequence length (`kv_len`) and batch size during attention with KV-Cache, with detailed code explanations and visual diagrams.

---

## Table of Contents

1. [Overview: The Plan-Run Pattern](#1-overview-the-plan-run-pattern)
2. [Phase 1: CPU Planning - Tiling and Work Distribution](#2-phase-1-cpu-planning---tiling-and-work-distribution)
3. [Phase 2: GPU Execution - Decode Kernel](#3-phase-2-gpu-execution---decode-kernel)
4. [Data Flow: Global Memory → Shared Memory → Registers](#4-data-flow-global-memory--shared-memory--registers)
5. [Paged KV-Cache: Handling Variable Sequence Lengths](#5-paged-kv-cache-handling-variable-sequence-lengths)
6. [Visual Summary](#6-visual-summary)

---

## 1. Overview: The Plan-Run Pattern

FlashInfer separates attention into two phases:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           PLAN PHASE (CPU)                                  │
│                                                                             │
│   Input: Variable-length sequences                                          │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │  Batch 0: kv_len = 1024                                             │   │
│   │  Batch 1: kv_len = 512                                              │   │
│   │  Batch 2: kv_len = 2048                                             │   │
│   │  Batch 3: kv_len = 256                                              │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                              ↓                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │  1. Compute tile counts per request                                 │   │
│   │  2. Binary search optimal kv_chunk_size                             │   │
│   │  3. Generate work indices: request_indices[], kv_tile_indices[]     │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                              ↓                                              │
│   Output: Flattened work assignments for GPU                               │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │  Block 0 → (request=0, kv_tile=0)                                   │   │
│   │  Block 1 → (request=0, kv_tile=1)                                   │   │
│   │  Block 2 → (request=1, kv_tile=0)                                   │   │
│   │  Block 3 → (request=2, kv_tile=0)                                   │   │
│   │  ...                                                                │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                           RUN PHASE (GPU)                                   │
│                                                                             │
│   Each CUDA block reads its assignment and processes independently          │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Phase 1: CPU Planning - Tiling and Work Distribution

### 2.1 Extracting Sequence Lengths from CSR Format

The scheduler first extracts per-request KV lengths from the CSR-format page table.

**Source: `scheduler.cuh:494-524`**

```cpp
template <typename IdType>
inline auto PrefillSplitQOKVIndptr(
    IdType* qo_indptr_h,           // [batch_size+1] CSR indptr for query lengths
    IdType* kv_indptr_h,           // [batch_size+1] CSR indptr for KV page counts
    uint32_t total_num_rows,       // Total query tokens across all requests
    uint32_t batch_size,           // Number of requests in batch
    ...) {

  const uint32_t gqa_group_size = num_qo_heads / num_kv_heads;
  //                              ↑ GQA ratio (e.g., 8 for 32 qo heads / 4 kv heads)

  // Step 1: Extract per-request lengths from CSR format
  std::vector<int64_t> packed_qo_len_arr(batch_size), kv_len_arr(batch_size);
  for (uint32_t i = 0; i < batch_size; ++i) {
    // qo_indptr_h is cumulative sum: [0, 10, 25, 50, ...]
    // qo_indptr_h[i+1] - qo_indptr_h[i] = number of query tokens for request i
    packed_qo_len_arr[i] = int64_t(qo_indptr_h[i + 1] - qo_indptr_h[i]) * int64_t(gqa_group_size);
    //                     ↑ "Packed" means: qo_len × gqa_group_size
    //                       This treats each GQA head as a separate "row" for tiling

    // kv_indptr_h[i+1] - kv_indptr_h[i] = number of pages for request i
    kv_len_arr[i] = int64_t(kv_indptr_h[i + 1] - kv_indptr_h[i]);
    //              ↑ This is page count, will be multiplied by page_size later
  }
```

**Visual: CSR Format for Variable Lengths**

```
Example: 3 requests with kv_len = [1024, 512, 2048], page_size = 16

kv_indptr_h = [0, 64, 96, 224]
               │   │   │   │
               │   │   │   └── Total pages: 224
               │   │   └── Request 2 starts at page 96 (has 2048/16 = 128 pages)
               │   └── Request 1 starts at page 64 (has 512/16 = 32 pages)
               └── Request 0 starts at page 0 (has 1024/16 = 64 pages)

kv_len_arr after extraction: [64, 32, 128]  (in pages)
```

### 2.2 Computing CTA Tile Size Based on Workload

**Source: `scheduler.cuh:543-555`**

```cpp
  // Step 2: Determine CTA tile size based on average sequence length
  if (enable_cuda_graph) {
    // CUDA graph mode: use max possible length
    const uint64_t max_seq_len = total_num_rows - batch_size + 1;
    cta_tile_q = FA2DetermineCtaTileQ(max_seq_len * gqa_group_size, head_dim);
  } else {
    // Normal mode: use average length for better efficiency
    int64_t sum_packed_qo_len = 0;
    for (uint32_t i = 0; i < batch_size; ++i) {
      sum_packed_qo_len += packed_qo_len_arr[i];
    }
    const int64_t avg_packed_qo_len = sum_packed_qo_len / batch_size;
    //                                ↑ Average across batch determines tile size

    cta_tile_q = FA2DetermineCtaTileQ(avg_packed_qo_len, head_dim);
    //           ↑ Returns 64, 128, or 192 based on workload characteristics

    // Count total Q tiles needed
    total_num_tiles_q = 0;
    for (uint32_t i = 0; i < batch_size; ++i) {
      total_num_tiles_q += ceil_div(packed_qo_len_arr[i], cta_tile_q);
      //                   ↑ Each request needs ceil(packed_qo_len / cta_tile_q) tiles
    }
  }
```

**Visual: CTA Tile Size Selection**

```
cta_tile_q selection based on workload:

  avg_qo_len      │ cta_tile_q │ Rationale
  ────────────────┼────────────┼─────────────────────────────────
  < 64            │     64     │ Small sequences: minimize waste
  64 - 128        │    128     │ Medium: balance compute/memory
  > 128           │    192     │ Large: maximize throughput

  head_dim also affects choice (smaller head_dim → larger tile possible)
```

### 2.3 Binary Search for Optimal KV Chunk Size

When sequences are too long, FlashInfer partitions them across multiple CTAs.

**Source: `scheduler.cuh:101-130`**

```cpp
inline auto PrefillBinarySearchKVChunkSize(
    const bool enable_cuda_graph,
    const uint32_t max_batch_size_if_split,  // Max CTAs available (num_SM × blocks_per_SM)
    const std::vector<int64_t>& packed_qo_len_arr,
    const std::vector<int64_t>& kv_len_arr,
    const uint32_t qo_chunk_size,            // CTA tile size for Q dimension
    const uint32_t min_kv_chunk_size = 1) {

  const int64_t batch_size = packed_qo_len_arr.size();
  int64_t max_kv_len = 1;
  for (const int64_t& kv_len : kv_len_arr) {
    max_kv_len = std::max(max_kv_len, kv_len);
  }
  //         ↑ Find longest sequence in batch

  // Binary search: find smallest kv_chunk_size that fits in available CTAs
  int64_t low = min_kv_chunk_size;
  int64_t high = max_kv_len;
  while (low < high) {
    const int64_t mid = (low + high) / 2;

    // Count total CTAs needed with this chunk size
    int64_t new_batch_size = 0;
    for (uint32_t i = 0; i < batch_size; ++i) {
      new_batch_size += ceil_div(packed_qo_len_arr[i], qo_chunk_size)  // Q tiles
                      * ceil_div(std::max(kv_len_arr[i], 1L), mid);    // KV chunks
      //               ↑ Total work = (Q tiles) × (KV chunks) per request
    }

    if (new_batch_size > max_batch_size_if_split) {
      low = mid + 1;  // Need larger chunks (fewer CTAs)
    } else {
      high = mid;     // Can use smaller chunks (more parallelism)
    }
  }
  return std::make_tuple(enable_cuda_graph || low < max_kv_len, low);
  //                     ↑ split_kv = true if we're actually splitting
}
```

**Visual: KV Partitioning**

```
Example: kv_len = 4096, kv_chunk_size = 1024

Original sequence:
┌────────────────────────────────────────────────────────────────┐
│                        kv_len = 4096                           │
└────────────────────────────────────────────────────────────────┘

After partitioning (4 chunks):
┌───────────────┬───────────────┬───────────────┬───────────────┐
│  Chunk 0      │  Chunk 1      │  Chunk 2      │  Chunk 3      │
│  [0:1024)     │  [1024:2048)  │  [2048:3072)  │  [3072:4096)  │
│  → CTA 0      │  → CTA 1      │  → CTA 2      │  → CTA 3      │
└───────────────┴───────────────┴───────────────┴───────────────┘
                              ↓
        Each CTA computes partial attention, then merge with MergeStates()
```

### 2.4 Generating Work Assignments

**Source: `scheduler.cuh:576-613`**

```cpp
  // Step 3: Generate flattened work indices
  uint32_t new_batch_size = 0;
  for (uint32_t request_idx = 0; request_idx < batch_size; ++request_idx) {
    const int64_t packed_qo_len = packed_qo_len_arr[request_idx];
    const int64_t num_tiles_q = ceil_div(packed_qo_len, cta_tile_q);
    //                          ↑ How many Q tiles for this request

    const int64_t kv_len = std::max(int(effective_kv_len_arr[request_idx]), 1);
    const int64_t num_chunks_kv = disable_split_kv ? 1 : ceil_div(kv_len, kv_chunk_size);
    //                            ↑ How many KV chunks for this request

    // Generate 2D grid of work items: (q_tile, kv_chunk)
    for (uint32_t q_tile_idx = 0; q_tile_idx < num_tiles_q; ++q_tile_idx) {
      for (uint32_t kv_tile_idx = 0; kv_tile_idx < num_chunks_kv; ++kv_tile_idx) {
        new_batch_size += 1;
        request_indices.push_back(request_idx);     // Which request
        qo_tile_indices.push_back(q_tile_idx);      // Which Q tile
        kv_tile_indices.push_back(kv_tile_idx);     // Which KV chunk
        //              ↑ These arrays are copied to GPU for kernel dispatch
      }
    }
  }
```

**Visual: Work Assignment Generation**

```
Example: 2 requests
  Request 0: qo_len=128, kv_len=2048 → 1 Q tile, 2 KV chunks
  Request 1: qo_len=256, kv_len=1024 → 2 Q tiles, 1 KV chunk

Work assignment arrays (copied to GPU):

Index │ request_indices │ qo_tile_indices │ kv_tile_indices │ Work Description
──────┼─────────────────┼─────────────────┼─────────────────┼───────────────────
  0   │        0        │        0        │        0        │ Req0, Q[0:128], KV[0:1024]
  1   │        0        │        0        │        1        │ Req0, Q[0:128], KV[1024:2048]
  2   │        1        │        0        │        0        │ Req1, Q[0:128], KV[0:1024]
  3   │        1        │        1        │        0        │ Req1, Q[128:256], KV[0:1024]

Grid dimension: dim3(4, num_kv_heads)
                     ↑ padded_batch_size = 4 CTAs needed
```

---

## 3. Phase 2: GPU Execution - Decode Kernel

### 3.1 Block-Level Work Assignment

Each CUDA block reads its assigned work from the pre-computed indices.

**Source: `decode.cuh:393-430`**

```cpp
template <PosEncodingMode POS_ENCODING_MODE, uint32_t num_stages_smem, uint32_t tile_size_per_bdx,
          uint32_t vec_size, uint32_t bdx, uint32_t bdy, uint32_t bdz, ...>
__device__ __inline__ void BatchDecodeWithPagedKVCacheDevice(
    const Params& params,
    uint8_t smem[],
    const uint32_t bx = blockIdx.x,    // Block X index (which work item)
    const uint32_t by = blockIdx.y,    // Block Y index (which KV head)
    const uint32_t tx = threadIdx.x,   // Thread X (within-warp index, 0-31)
    const uint32_t ty = threadIdx.y,   // Thread Y (GQA head index)
    const uint32_t tz = threadIdx.z) { // Thread Z (KV parallelism)

  // ═══════════════════════════════════════════════════════════════════════
  // STEP 1: Read work assignment from pre-computed indices
  // ═══════════════════════════════════════════════════════════════════════

  constexpr uint32_t head_dim = bdx * vec_size;
  //                           ↑ e.g., 32 threads × 4 elements = 128-dim head

  const uint32_t batch_idx = params.request_indices[bx];
  //                         ↑ Which request this block processes
  //                           (from CPU-computed work assignment)

  const uint32_t kv_tile_idx = params.kv_tile_indices[bx];
  //                           ↑ Which KV chunk this block processes

  const uint32_t kv_head_idx = by;
  //                           ↑ Grid Y dimension = num_kv_heads

  const uint32_t qo_head_idx = kv_head_idx * bdy + ty;
  //                           ↑ Map thread Y to QO head (GQA expansion)
  //                             e.g., kv_head=0, ty=3 → qo_head=3

  // Skip invalid blocks (for CUDA graph padding)
  if (block_valid_mask && !block_valid_mask[bx]) return;

  // ═══════════════════════════════════════════════════════════════════════
  // STEP 2: Compute this block's KV range from kv_len
  // ═══════════════════════════════════════════════════════════════════════

  const uint32_t kv_chunk_size = *(params.kv_chunk_size_ptr);
  //                             ↑ Chunk size computed during planning

  const uint32_t kv_len = paged_kv.get_length(batch_idx);
  //                      ↑ ACTUAL sequence length for this request
  //                        Retrieved from paged KV cache metadata

  const uint32_t max_chunk_size = partition_kv ? kv_chunk_size : kv_len;
  const uint32_t chunk_start = partition_kv ? kv_tile_idx * max_chunk_size : 0;
  //                           ↑ Start position in KV sequence
  //                             kv_tile_idx=1, chunk_size=1024 → start=1024

  const uint32_t chunk_end = partition_kv
      ? min((kv_tile_idx + 1) * max_chunk_size, kv_len)
      : kv_len;
  //    ↑ End position, clamped to actual sequence length
  //      Handles last chunk being smaller than chunk_size

  const uint32_t chunk_size = chunk_end - chunk_start;
  //                          ↑ ACTUAL work size for this block
  //                            May be < kv_chunk_size for last chunk
```

**Visual: Block Work Assignment**

```
Grid Configuration: dim3(padded_batch_size, num_kv_heads)

         │ by=0 (kv_head_0) │ by=1 (kv_head_1) │ ...
─────────┼──────────────────┼──────────────────┼─────
  bx=0   │ Req0, Chunk0     │ Req0, Chunk0     │
  bx=1   │ Req0, Chunk1     │ Req0, Chunk1     │
  bx=2   │ Req1, Chunk0     │ Req1, Chunk0     │
  bx=3   │ (padding)        │ (padding)        │

Thread Block Configuration: dim3(bdx, bdy, bdz)
  bdx = head_dim / vec_size = 128/4 = 32  (covers head dimension)
  bdy = GQA group size = 8                 (covers QO heads per KV head)
  bdz = KV parallelism = 4                 (parallel KV tile processing)

  Total threads = 32 × 8 × 4 = 1024 threads per block
```

### 3.2 Thread-Level Warp Assignment

**Source: `decode.cuh:62-116`**

```cpp
template <PosEncodingMode pos_encoding_mode, uint32_t vec_size, uint32_t bdx, uint32_t tile_size,
          typename AttentionVariant, typename Params, typename T>
__device__ __forceinline__ void compute_qk(
    const Params& params, AttentionVariant variant,
    const uint32_t batch_idx,
    const T* smem,                          // K tile in shared memory
    const vec_t<float, vec_size>& q_vec,    // Q vector in registers
    const vec_t<float, vec_size>& freq,     // RoPE frequencies
    uint32_t kv_idx_base,                   // Start position in KV sequence
    uint32_t iter_base,                     // Iteration counter
    uint32_t iter_bound,                    // chunk_size
    uint32_t qo_head_idx, uint32_t kv_head_idx,
    float* s,                               // Output: attention scores
    state_t<vec_size>& st,                  // Attention state (m, d, o)
    const uint32_t tx, const uint32_t ty, const uint32_t tz) {

  float m_prev = st.m;  // Previous max for online softmax

  // ═══════════════════════════════════════════════════════════════════════
  // Process tile_size KV positions (e.g., tile_size = 8)
  // Each thread handles vec_size elements of head_dim
  // ═══════════════════════════════════════════════════════════════════════

#pragma unroll
  for (uint32_t j = 0; j < tile_size; ++j) {
    vec_t<float, vec_size> k_vec;

    if constexpr (pos_encoding_mode == PosEncodingMode::kRoPELlama) {
      // Apply RoPE on-the-fly during load
      k_vec = vec_apply_llama_rope<vec_size, bdx>(
          smem + j * bdx * vec_size,          // K position j in smem
          freq,                                // RoPE frequencies
          kv_idx_base + tz * tile_size + j);  // Absolute position in sequence
      //               ↑ tz indexes parallel KV tiles
    } else {
      // Direct load without RoPE
      k_vec.cast_load(smem + (j * bdx + tx) * vec_size);
      //                      ↑ Each thread loads its vec_size elements
      //                        tx=0 loads [0:4], tx=1 loads [4:8], etc.
    }

    // ═══════════════════════════════════════════════════════════════════
    // Compute dot product: q · k
    // ═══════════════════════════════════════════════════════════════════

    s[j] = 0.f;
#pragma unroll
    for (uint32_t i = 0; i < vec_size; ++i) {
      s[j] += q_vec[i] * k_vec[i];  // Partial dot product (vec_size elements)
    }

    // ═══════════════════════════════════════════════════════════════════
    // Warp-level reduction to sum across all threads
    // ═══════════════════════════════════════════════════════════════════

#pragma unroll
    for (uint32_t offset = bdx / 2; offset > 0; offset /= 2) {
      s[j] += math::shfl_xor_sync(s[j], offset);
      //      ↑ Butterfly reduction pattern
      //        offset=16: threads 0-15 exchange with 16-31
      //        offset=8:  threads 0-7 exchange with 8-15, etc.
      //        After all iterations, all threads have the full dot product
    }

    // Bounds check: mask invalid positions
    const uint32_t pos = kv_idx_base + tz * tile_size + j;
    s[j] = (iter_base + tz * tile_size + j < iter_bound) ? s[j] : -math::inf;
    //     ↑ Set to -inf if position >= chunk_size (for softmax)

    st.m = max(st.m, s[j]);  // Track running max for online softmax
  }
```

**Visual: Warp Shuffle Reduction**

```
Example: bdx=32 threads computing dot product for head_dim=128

Initial state (each thread has partial sum of 4 elements):
  Thread:   0    1    2    3   ...   31
  Value:   a0   a1   a2   a3  ...  a31

After shfl_xor(offset=16):
  Thread:   0    1   ...   15  │  16   17  ...   31
  Value:  a0+a16 a1+a17 ... │  a16+a0 a17+a1 ...

After shfl_xor(offset=8):
  Pairs 0-8, 1-9, etc. add together

After shfl_xor(offset=4), (offset=2), (offset=1):
  All 32 threads have the same final sum = Σ(a0..a31)
```

---

## 4. Data Flow: Global Memory → Shared Memory → Registers

### 4.1 Asynchronous Copy with `cp.async`

**Source: `decode.cuh:474-517`**

```cpp
  // ═══════════════════════════════════════════════════════════════════════
  // STEP: Preload K/V tiles using asynchronous copy
  // ═══════════════════════════════════════════════════════════════════════

  uint32_t stage_idx = 0;
  constexpr uint32_t vec_bits = sizeof(DTypeKV) * vec_size * 8;
  //                            ↑ e.g., fp16 × 4 elements × 8 bits = 64 bits

  const IdType last_indptr = paged_kv.indptr[paged_kv.batch_size];
  //                         ↑ Total page count (for bounds checking)

  // ═══════════════════════════════════════════════════════════════════════
  // Compute paged KV offsets: translate (token_idx) → (page, offset_in_page)
  // ═══════════════════════════════════════════════════════════════════════

  uint32_t packed_page_iter_base = paged_kv.indptr[batch_idx] * paged_kv.page_size + chunk_start;
  //                               ↑ Starting position in flattened page space
  //                                 indptr[batch_idx] = first page of this request
  //                                 × page_size = first token position
  //                                 + chunk_start = offset for this chunk

#pragma unroll
  for (uint32_t j = 0; j < tile_size_per_bdx; ++j) {
    uint32_t q, r;  // q = page index, r = offset within page

    paged_kv.page_size.divmod(
        packed_page_iter_base + ((j * bdz + tz) * bdy + ty) * bdx + tx,
        //                      ↑ Linearized thread index within tile
        q, r);
    //  ↑ Fast integer division: q = position / page_size, r = position % page_size

    kv_offset_smem[((j * bdz + tz) * bdy + ty) * bdx + tx] =
        paged_kv.protective_get_kv_offset(q, kv_head_idx, r, 0, last_indptr);
    //  ↑ Compute global memory offset for this thread's KV element
    //    If page_idx >= last_indptr, returns 0 (safe default)
  }
  block.sync();  // All threads must finish offset computation

  // ═══════════════════════════════════════════════════════════════════════
  // Multi-stage pipelining: preload num_stages_smem tiles ahead
  // ═══════════════════════════════════════════════════════════════════════

  size_t kv_offset[tile_size_per_bdx];
#pragma unroll
  for (uint32_t iter = 0; iter < num_stages_smem; ++iter) {
#pragma unroll
    for (uint32_t j = 0; j < tile_size_per_bdx; ++j) {
      kv_offset[j] = kv_offset_smem[...] + tx * vec_size;
      //             ↑ Read precomputed offset, add thread's element offset
    }

#pragma unroll
    for (uint32_t j = 0; j < tile_size_per_bdx; ++j) {
      cp_async::pred_load<vec_bits, PrefetchMode::kPrefetch, SharedMemFillMode::kNoFill>(
          k_smem + (((stage_idx * bdz + tz) * bdy + ty) * tile_size_per_bdx + j) * head_dim
                 + tx * vec_size,
          //       ↑ Destination in shared memory (stage-aware addressing)

          paged_kv.k_data + kv_offset[j],
          //                ↑ Source in global memory (paged KV cache)

          ((iter * bdz + tz) * bdy + ty) * tile_size_per_bdx + j < chunk_size);
          //                                                      ↑ Predicate: only load valid positions
          //                                                        Avoids out-of-bounds access
    }
    cp_async::commit_group();  // Submit this batch of async copies
```

**Visual: cp.async Pipeline**

```
Timeline showing 2-stage pipeline (num_stages_smem = 2):

              Compute              Memory
Iteration │ (previous tile)   │  (current tile)
──────────┼───────────────────┼─────────────────
    0     │     idle          │  cp.async K[0]
          │                   │  cp.async V[0]
──────────┼───────────────────┼─────────────────
    1     │     idle          │  cp.async K[1]
          │                   │  cp.async V[1]
──────────┼───────────────────┼─────────────────
    2     │  compute QK[0]    │  cp.async K[2]
          │  wait_group<1>    │
──────────┼───────────────────┼─────────────────
    3     │  compute S*V[0]   │  cp.async V[2]
          │  compute QK[1]    │  cp.async K[3]
──────────┼───────────────────┼─────────────────
   ...    │    (overlap)      │    (overlap)

Key insight: While computing on tile N, we're loading tile N+2
```

### 4.2 Shared Memory Layout with Swizzling

**Source: `prefill.cuh:75-93`**

```cpp
template <uint32_t NUM_WARPS_KV, uint32_t CTA_TILE_Q, uint32_t CTA_TILE_KV, uint32_t HEAD_DIM_QK,
          uint32_t HEAD_DIM_VO, typename DTypeQ, typename DTypeKV, typename DTypeO>
struct SharedStorageQKVO {
  union {  // Union allows memory reuse between compute and sync phases
    struct {
      alignas(16) DTypeQ q_smem[CTA_TILE_Q * HEAD_DIM_QK];      // Q tile storage
      //         ↑ 16-byte alignment for vectorized access
      //                    ↑ e.g., 128 × 128 × 2 bytes = 32 KB for Q

      alignas(16) DTypeKV k_smem[CTA_TILE_KV * HEAD_DIM_QK];    // K tile storage
      alignas(16) DTypeKV v_smem[CTA_TILE_KV * HEAD_DIM_VO];    // V tile storage
    };
    struct {  // Alternate layout for warp synchronization
      alignas(16) float cta_sync_o_smem[NUM_WARPS_KV * CTA_TILE_Q * HEAD_DIM_VO];
      //                ↑ Temporary storage for merging warp-local outputs
      alignas(16) float2 cta_sync_md_smem[NUM_WARPS_KV * CTA_TILE_Q];
      //                 ↑ (max, denom) pairs for online softmax merge
    };
    alignas(16) DTypeO smem_o[CTA_TILE_Q * HEAD_DIM_VO];  // Output staging
  };
};
```

**Visual: Shared Memory Layout**

```
Shared Memory (e.g., 96 KB on Ampere):

Phase 1: Compute
┌──────────────────────────────────────────────────────────────┐
│  Q tile (32KB)         │  K tile (32KB)    │  V tile (32KB) │
│  [128 × 128 × fp16]    │  [128 × 128]      │  [128 × 128]   │
└──────────────────────────────────────────────────────────────┘

Phase 2: Warp Sync (union reuses same memory)
┌──────────────────────────────────────────────────────────────┐
│  cta_sync_o_smem        │  cta_sync_md_smem │  (unused)     │
│  [4 warps × 128 × 128]  │  [4 × 128 × 2]    │               │
└──────────────────────────────────────────────────────────────┘
```

### 4.3 Register-Resident State for Online Softmax

**Source: `state.cuh:28-79`**

```cpp
template <size_t vec_size>
struct state_t {
  vec_t<float, vec_size> o;  // Output accumulator [vec_size floats]
  float m;                    // Running max (for numerical stability)
  float d;                    // Running denominator (sum of exp)

  __device__ __forceinline__ void init() {
    o.fill(0.f);                      // Zero output accumulator
    m = -math::inf;                   // Initialize max to -infinity
    d = 0.f;                          // Initialize sum to 0
  }

  // ═══════════════════════════════════════════════════════════════════════
  // Merge another partial state using online softmax formula
  // ═══════════════════════════════════════════════════════════════════════

  __device__ __forceinline__ void merge(
      const vec_t<float, vec_size>& other_o,
      float other_m,
      float other_d) {

    float m_prev = m, d_prev = d;
    m = max(m_prev, other_m);          // New max = max(old_max, incoming_max)

    // Rescale denominators to new max
    d = d_prev * math::ptx_exp2(m_prev - m)      // Old sum rescaled
      + other_d * math::ptx_exp2(other_m - m);   // Incoming sum rescaled
    //           ↑ exp2(x) = 2^x, faster than exp(x) on GPU

    // Rescale and merge output accumulators
#pragma unroll
    for (size_t i = 0; i < vec_size; ++i) {
      o[i] = o[i] * math::ptx_exp2(m_prev - m)        // Rescale old output
           + other_o[i] * math::ptx_exp2(other_m - m); // Add rescaled incoming
    }
  }

  __device__ __forceinline__ float get_lse() const {
    return m + math::ptx_log2(d) * math::loge2;  // log(sum(exp)) for debugging
  }
};
```

**Visual: Online Softmax State Evolution**

```
Processing KV chunks sequentially with state merging:

Initial: state = {o=[0,0,0,0], m=-inf, d=0}

After Chunk 0 (kv positions 0-1023):
  ┌─────────────────────────────────────────────────────────────┐
  │  Computed: exp(QK[0:1024] - m0) / d0                        │
  │  state = {o=o0, m=m0, d=d0}                                 │
  └─────────────────────────────────────────────────────────────┘

After Chunk 1 (kv positions 1024-2047):
  ┌─────────────────────────────────────────────────────────────┐
  │  Incoming: {o1, m1, d1}                                     │
  │                                                             │
  │  Merge:                                                     │
  │    m_new = max(m0, m1)                                      │
  │    d_new = d0 * exp(m0 - m_new) + d1 * exp(m1 - m_new)      │
  │    o_new = o0 * exp(m0 - m_new) + o1 * exp(m1 - m_new)      │
  │                                                             │
  │  state = {o=o_new, m=m_new, d=d_new}                        │
  └─────────────────────────────────────────────────────────────┘

Final output: o / d (divide accumulator by sum)
```

---

## 5. Paged KV-Cache: Handling Variable Sequence Lengths

### 5.1 Paged KV-Cache Structure

**Source: `page.cuh:37-152`**

```cpp
template <typename DType, typename IdType>
struct paged_kv_t {
  // ═══════════════════════════════════════════════════════════════════════
  // Core parameters
  // ═══════════════════════════════════════════════════════════════════════

  uint_fastdiv page_size;      // Page size with fast division support
  //                           ↑ Stores both value and magic number for fast modulo
  uint32_t num_heads;          // Number of KV heads
  uint32_t head_dim;           // Dimension per head
  uint32_t batch_size;         // Number of requests

  // ═══════════════════════════════════════════════════════════════════════
  // Stride parameters for memory layout
  // ═══════════════════════════════════════════════════════════════════════

  uint32_t stride_page;        // Stride between pages: num_heads * page_size * head_dim
  uint32_t stride_n;           // Stride between tokens within page
  uint32_t stride_h;           // Stride between heads

  // ═══════════════════════════════════════════════════════════════════════
  // Data pointers
  // ═══════════════════════════════════════════════════════════════════════

  DType* k_data;               // [max_num_pages, num_heads, page_size, head_dim]
  DType* v_data;               // Same layout as k_data

  // ═══════════════════════════════════════════════════════════════════════
  // Per-request metadata (CSR format)
  // ═══════════════════════════════════════════════════════════════════════

  IdType* indices;             // [total_num_pages] Physical page indices
  //                           ↑ Indirection table: logical page → physical page

  IdType* indptr;              // [batch_size + 1] CSR format pointers
  //                           ↑ indptr[i] = start of request i's pages in indices[]

  IdType* last_page_len;       // [batch_size] Tokens in last page
  //                           ↑ Last page may be partially filled

  // ═══════════════════════════════════════════════════════════════════════
  // Compute actual sequence length for a request
  // ═══════════════════════════════════════════════════════════════════════

  __host__ __device__ __forceinline__ uint32_t get_length(uint32_t batch_idx) const {
    if (indptr[batch_idx + 1] == indptr[batch_idx]) {
      return 0;  // No pages → empty sequence
    }
    return (indptr[batch_idx + 1] - indptr[batch_idx] - 1) * page_size
           + last_page_len[batch_idx];
    //     ↑ (full_pages × page_size) + tokens_in_last_page
  }

  // ═══════════════════════════════════════════════════════════════════════
  // Compute global memory offset for element access
  // ═══════════════════════════════════════════════════════════════════════

  __host__ __device__ __forceinline__ size_t get_elem_offset(
      size_t page_idx, size_t head_idx, size_t entry_idx, size_t feat_idx) const {
    return page_idx * stride_page    // Which page
         + head_idx * stride_h       // Which head within page
         + entry_idx * stride_n      // Which token within page
         + feat_idx;                 // Which element of head_dim
  }
};
```

**Visual: Paged KV-Cache Layout**

```
Physical Memory Layout (HND format):
┌─────────────────────────────────────────────────────────────────────────────┐
│                             k_data / v_data                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│ Page 0                          │ Page 1                          │ ...    │
│ ┌───────────────────────────┐   │ ┌───────────────────────────┐   │        │
│ │ Head 0 [page_size×head_dim]│  │ │ Head 0 [page_size×head_dim]│  │        │
│ ├───────────────────────────┤   │ ├───────────────────────────┤   │        │
│ │ Head 1                    │   │ │ Head 1                    │   │        │
│ ├───────────────────────────┤   │ ├───────────────────────────┤   │        │
│ │ ...                       │   │ │ ...                       │   │        │
│ └───────────────────────────┘   │ └───────────────────────────┘   │        │
└─────────────────────────────────────────────────────────────────────────────┘

Logical → Physical Mapping (via indices[]):

Request 0 (kv_len=48, page_size=16 → 3 pages):
  indptr = [0, 3, ...]
  indices = [5, 12, 7, ...]   ← Physical pages 5, 12, 7
  last_page_len = [16, ...]   ← Last page is full (16 tokens)

  Logical token 0-15  → Physical page 5,  entry 0-15
  Logical token 16-31 → Physical page 12, entry 0-15
  Logical token 32-47 → Physical page 7,  entry 0-15

Request 1 (kv_len=25, page_size=16 → 2 pages):
  indptr = [0, 3, 5, ...]
  indices = [..., 3, 8, ...]  ← Physical pages 3, 8
  last_page_len = [..., 9, ...] ← Last page has 9 tokens

  Logical token 0-15 → Physical page 3, entry 0-15
  Logical token 16-24 → Physical page 8, entry 0-8
```

### 5.2 Fast Division for Page Index Computation

**Source: `decode.cuh:480-487`**

```cpp
  // ═══════════════════════════════════════════════════════════════════════
  // Translate token position to (page_index, offset_in_page) using fast division
  // ═══════════════════════════════════════════════════════════════════════

  static_assert(num_stages_smem <= bdx);

  uint32_t packed_page_iter_base = paged_kv.indptr[batch_idx] * paged_kv.page_size + chunk_start;
  //                               ↑ Virtual start position in flattened page array
  //                                 For request with indptr[batch_idx]=10, page_size=16:
  //                                 Base = 10 × 16 = 160 (virtual token index)

#pragma unroll
  for (uint32_t j = 0; j < tile_size_per_bdx; ++j) {
    uint32_t q, r;

    paged_kv.page_size.divmod(
        packed_page_iter_base + ((j * bdz + tz) * bdy + ty) * bdx + tx,
        //                      ↑ This thread's token position in virtual space
        q,   // Output: page index (quotient)
        r);  // Output: offset within page (remainder)
    //  ↑ Fast division: avoids expensive integer divide instruction
    //    Uses precomputed magic multiplier for division by constant

    kv_offset_smem[((j * bdz + tz) * bdy + ty) * bdx + tx] =
        paged_kv.protective_get_kv_offset(q, kv_head_idx, r, 0, last_indptr);
    //  ↑ Convert (page_iter, head, entry, feat=0) to global memory offset
    //    Uses indices[] for page indirection
    //    Returns 0 if page_iter >= last_indptr (out of bounds protection)
  }
```

**Visual: Token Position to Page Offset Translation**

```
Example: page_size=16, accessing token 42 of request with indptr[batch_idx]=3

Step 1: Compute virtual position
  packed_page_iter_base = 3 × 16 + chunk_start = 48 + 0 = 48
  thread_position = 48 + 42 = 90  (in virtual page space)

Step 2: Fast divmod
  q = 90 / 16 = 5     (page iterator index)
  r = 90 % 16 = 10    (entry within page)

Step 3: Page indirection
  physical_page = indices[5]  (look up physical page)

Step 4: Compute offset
  offset = physical_page × stride_page + kv_head_idx × stride_h + r × stride_n + 0
```

---

## 6. Visual Summary

### Complete Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              FlashInfer KV-Cache Attention                          │
└─────────────────────────────────────────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════════════════════════
                                  CPU PLANNING PHASE
═══════════════════════════════════════════════════════════════════════════════════════

Input: Variable-length batch
┌────────────────────────────────────────────────────────────────────────────────────┐
│  Request 0: kv_len=1024  │  Request 1: kv_len=512  │  Request 2: kv_len=2048  │ ...│
└────────────────────────────────────────────────────────────────────────────────────┘
                                         │
                                         ▼
┌────────────────────────────────────────────────────────────────────────────────────┐
│  1. Extract lengths from CSR: kv_len_arr = [1024, 512, 2048, ...]                  │
│  2. Determine CTA tile size: cta_tile_q = FA2DetermineCtaTileQ(avg_len)            │
│  3. Binary search kv_chunk_size to fit max_grid_size                               │
│  4. Generate work arrays: request_indices[], qo_tile_indices[], kv_tile_indices[]  │
└────────────────────────────────────────────────────────────────────────────────────┘
                                         │
                                         ▼
Output: Work assignment copied to GPU
┌────────────────────────────────────────────────────────────────────────────────────┐
│  Block 0 → (req=0, kv_chunk=0)    Block 1 → (req=0, kv_chunk=1)                    │
│  Block 2 → (req=1, kv_chunk=0)    Block 3 → (req=2, kv_chunk=0)    ...             │
└────────────────────────────────────────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════════════════════════
                                  GPU EXECUTION PHASE
═══════════════════════════════════════════════════════════════════════════════════════

Each CUDA Block:
┌────────────────────────────────────────────────────────────────────────────────────┐
│ batch_idx = request_indices[blockIdx.x]   ← Which request                         │
│ kv_tile_idx = kv_tile_indices[blockIdx.x] ← Which KV chunk                         │
│ kv_len = paged_kv.get_length(batch_idx)   ← ACTUAL sequence length                 │
│ chunk_start = kv_tile_idx × kv_chunk_size                                          │
│ chunk_end = min(chunk_start + kv_chunk_size, kv_len)  ← Handle last chunk          │
└────────────────────────────────────────────────────────────────────────────────────┘
                                         │
                                         ▼
┌────────────────────────────────────────────────────────────────────────────────────┐
│                           GLOBAL MEMORY (Paged KV Cache)                           │
│                                                                                    │
│  ┌───────────────────────────────────────────────────────────────────────────────┐ │
│  │ k_data/v_data: [max_pages, num_heads, page_size, head_dim]                    │ │
│  │                                                                               │ │
│  │ Page indirection: token_pos → indices[page_iter] → physical_page             │ │
│  └───────────────────────────────────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────────────────────────────────┘
                                         │
                          cp.async.cg.shared.global
                          (predicated, L2 prefetch)
                                         │
                                         ▼
┌────────────────────────────────────────────────────────────────────────────────────┐
│                        SHARED MEMORY (Multi-Stage Pipeline)                        │
│                                                                                    │
│  ┌─────────────────────────────────────────────────────────────────────────────┐   │
│  │ Stage 0:  K_smem[tile_size × head_dim]  │  V_smem[tile_size × head_dim]     │   │
│  ├─────────────────────────────────────────────────────────────────────────────┤   │
│  │ Stage 1:  K_smem[tile_size × head_dim]  │  V_smem[tile_size × head_dim]     │   │
│  │           (prefetching while Stage 0 computes)                              │   │
│  └─────────────────────────────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────────────────────────────┘
                                         │
                             ldmatrix / vectorized load
                                         │
                                         ▼
┌────────────────────────────────────────────────────────────────────────────────────┐
│                             REGISTERS (Per Thread)                                 │
│                                                                                    │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐                     │
│  │ q_vec[vec_size] │  │ k_vec[vec_size] │  │ v_vec[vec_size] │                     │
│  │ (loaded once)   │  │ (from smem)     │  │ (from smem)     │                     │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘                     │
│           │                    │                    │                              │
│           └─────── dot product ┴────────────────────┘                              │
│                         │                                                          │
│                         ▼                                                          │
│  ┌──────────────────────────────────────────────────────────────────────────────┐  │
│  │                     s[tile_size] (attention scores)                          │  │
│  │                                                                              │  │
│  │                     shfl_xor_sync for warp reduction                         │  │
│  │                                 ↓                                            │  │
│  │  ┌────────────────────────────────────────────────────────────────────────┐  │  │
│  │  │           state_t { o[vec_size], m, d }                                │  │  │
│  │  │           (online softmax accumulation)                                │  │  │
│  │  │                                                                        │  │  │
│  │  │   o_new = o_old × exp(m_old - m_new) + v × softmax(s) × exp(s - m_new) │  │  │
│  │  │   m_new = max(m_old, max(s))                                           │  │  │
│  │  │   d_new = d_old × exp(m_old - m_new) + sum(exp(s - m_new))             │  │  │
│  │  └────────────────────────────────────────────────────────────────────────┘  │  │
│  └──────────────────────────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────────────────────┘
                                         │
                    (if bdz > 1: sync via shared memory)
                                         │
                                         ▼
┌────────────────────────────────────────────────────────────────────────────────────┐
│                    WARP SYNCHRONIZATION (if bdz > 1)                               │
│                                                                                    │
│  Each warp writes to smem:  o_smem[(tz,ty) × head_dim], md_smem[(tz,ty) × 2]      │
│  Block sync                                                                        │
│  Each warp reads all others and merges: state.merge(o_other, m_other, d_other)     │
└────────────────────────────────────────────────────────────────────────────────────┘
                                         │
                        cast_store (fp32 → fp16/bf16)
                                         │
                                         ▼
┌────────────────────────────────────────────────────────────────────────────────────┐
│                           GLOBAL MEMORY (Output)                                   │
│                                                                                    │
│  o[(blockIdx.x, qo_head_idx) × head_dim + tx × vec_size]                          │
│  lse[blockIdx.x × num_qo_heads + qo_head_idx]  (optional)                          │
└────────────────────────────────────────────────────────────────────────────────────┘
                                         │
                  (if split_kv: MergeStates across chunks)
                                         │
                                         ▼
                                   Final Output
```

### Key Variable Tracking Summary

| Variable | Origin | Transformation | Usage |
|----------|--------|----------------|-------|
| `kv_len` | `paged_kv.get_length(batch_idx)` | `(num_pages - 1) × page_size + last_page_len` | Bounds checking, iteration count |
| `batch_size` | Input parameter | Split into `new_batch_size` work items | Grid dimension X |
| `kv_chunk_size` | Binary search result | Determines `num_chunks = ceil(kv_len / kv_chunk_size)` | Per-block work range |
| `chunk_start` | `kv_tile_idx × kv_chunk_size` | Starting KV position for this block | Memory offset base |
| `chunk_size` | `min(chunk_end, kv_len) - chunk_start` | Actual work for this block (handles last chunk) | Loop bound, predicate |

---

## Source Code References

| Component | File | Lines | Key Functions |
|-----------|------|-------|---------------|
| Work distribution | `scheduler.cuh` | 494-613 | `PrefillSplitQOKVIndptr` |
| Binary search | `scheduler.cuh` | 101-130 | `PrefillBinarySearchKVChunkSize` |
| Decode kernel | `decode.cuh` | 393-608 | `BatchDecodeWithPagedKVCacheDevice` |
| QK compute | `decode.cuh` | 62-116 | `compute_qk` |
| State update | `decode.cuh` | 131-144 | `update_local_state` |
| Warp sync | `decode.cuh` | 155-189 | `sync_state` |
| Online softmax | `state.cuh` | 28-79 | `state_t::merge` |
| Paged KV cache | `page.cuh` | 37-200 | `paged_kv_t` |
| Async copy | `cp_async.cuh` | 72-140 | `pred_load`, `commit_group` |
