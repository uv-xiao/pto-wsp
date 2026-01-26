# FlashInfer-Style Decode Attention Example

This example demonstrates how to implement FlashInfer-style decode attention using the proposed PTO-ISA runtime extension.

## Overview

FlashInfer's key innovation is the **Plan-Run** execution model that efficiently handles variable-length sequences through index-based work assignment. This example shows how to achieve the same pattern on Ascend NPU using PTO-ISA.

## Directory Structure

```
flashinfer_decode/
├── README.md           # This file
├── include/
│   ├── runtime.hpp     # Runtime extension headers
│   └── decode_attention.hpp  # Kernel interface
├── src/
│   ├── host_planning.cpp     # Host-side planning
│   └── device_kernel.cpp     # Device kernel implementation
├── tests/
│   └── test_decode.cpp       # Unit tests
└── CMakeLists.txt      # Build configuration
```

## Key Concepts

### 1. Work Descriptor

Each AICore receives a compact descriptor specifying its work:

```cpp
struct WorkDescriptor {
    uint32_t work_id;     // Unique work identifier
    uint8_t  tier;        // Kernel tier (size-optimized variant)
    uint8_t  flags;       // First/last chunk flags
    uint16_t reserved;
    uint32_t params[4];   // [request_idx, head_idx, kv_start, kv_len]
};
```

### 2. Planning Phase (Host)

The host generates descriptors for all work units:

```cpp
// Binary search for optimal chunk size
int chunk_size = planner.plan_chunk_size(kv_lengths, batch_size, num_heads);

// Generate descriptors
planner.generate(kv_lengths, batch_size, num_heads, chunk_size,
                 desc_buffer, max_work, &work_count);
```

### 3. Execution Phase (Device)

Each AICore reads its descriptor and executes:

```cpp
__global__ void decode_attention_entry(__gm__ WorkDescriptor* descs, ...) {
    auto desc = descs[get_block_idx()];  // O(1) lookup
    dispatch_by_tier(desc, ...);          // Tier-specific kernel
}
```

## Usage

### Building

```bash
# From pto-isa root
mkdir build && cd build
cmake -DCPU_SIM=ON ..
make flashinfer_decode_example
```

### Running Tests

```bash
./tests/flashinfer_decode_test
```

### Example Output

```
FlashInfer Decode Attention Example
===================================
Batch size: 32
Sequence lengths: [512, 1024, 2048, 512, ...]
Number of heads: 128
Head dimension: 128

Planning phase:
  Optimal chunk size: 512
  Total work units: 8192

Execution phase:
  Kernel launches: 1
  Total time: 2.5ms
  Throughput: 3.28 TFLOPS
```

## Comparison with FlashInfer

| Aspect | FlashInfer (CUDA) | This Example (PTO-ISA) |
|--------|-------------------|------------------------|
| Planning | Python/C++ | C++ WorkPlanner |
| Descriptors | `request_indices[]`, `kv_tile_indices[]` | `WorkDescriptor[]` |
| Kernel dispatch | CUDA blocks | AICore via descriptors |
| Tile operations | Manual register management | PTO Tile abstractions |
| Online softmax | `state_t::merge()` | `pto_macro_fa_softmax` |

## References

- [FlashInfer Paper](https://arxiv.org/abs/2310.08000)
- [PTO-ISA Documentation](../../../docs/PTOISA.md)
- [Runtime Extension Design](../analysis_round3.md)
