// Copyright 2026 PTO-WSP Authors
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <cstddef>

extern "C" {

#if defined(PTO_WSP_TARGET_NPU) && defined(__aicore__)
#define PTO_WSP_KERNEL_ATTR __aicore__
#else
#define PTO_WSP_KERNEL_ATTR
#endif

//==============================================================================
// Task Descriptor - passed to every generated kernel
//==============================================================================

struct KernelTaskDesc {
    // Loop indices and scalar args (u64 for simplicity/stability)
    const uint64_t* args;
    uint32_t num_args;
    // Number of axis args stored at the beginning of `args`. The remaining
    // entries (if any) are kernel scalar params encoded as u64.
    uint32_t num_axis_args;

    // Tensor I/O (base pointers + strides in elements)
    void* const* tensor_ptrs;
    // Flattened 2D strides for each tensor: [s3(row), s4(col)] pairs, in elements.
    // For a 2D tile view, s3/s4 correspond to base tensor strides for the last
    // two dimensions of the view. This supports transposed / non-contiguous
    // tiles without recompilation.
    const uint64_t* tensor_strides;
    uint32_t num_tensors;

    // Metadata
    uint32_t kernel_id;
    uint32_t task_id;
};

//==============================================================================
// CSPT Context - for cycle-accurate timing (optional; may be null)
//==============================================================================

struct TimingConfig {
    uint64_t loop_overhead;
    uint64_t dispatch_overhead;
    uint64_t channel_latency;
};

struct CSPTContext {
    void* ctx;  // Opaque pointer to implementation-defined timing context

    // Timing callbacks
    void (*advance_cycles)(void* ctx, uint64_t cycles);
    uint64_t (*get_time)(void* ctx);

    // Timing config (read-only, may be null)
    const TimingConfig* timing;
};

//==============================================================================
// Kernel Function Signature
//==============================================================================

// Returns total cycles consumed (for timing verification). If cspt is null,
// implementations should return 0.
using KernelFn = uint64_t (*)(const KernelTaskDesc* task, CSPTContext* cspt);

}  // extern "C"
