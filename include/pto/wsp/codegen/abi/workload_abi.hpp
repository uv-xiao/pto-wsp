// Copyright 2026 PTO-RT Authors
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <cstdint>

#include "pto/rt/codegen/abi/kernel_abi.hpp"

extern "C" {

//==============================================================================
// Runtime Context - dynamic sizes, tensor pointers, kernel lookup
//==============================================================================

struct RuntimeContext {
    // Dynamic axis sizes (by name)
    int64_t (*get_axis_size)(void* ctx, const char* name);

    // Runtime-bound symbols (stable IDs, e.g. FNV-1a hash of a name).
    // These enable dynamic axes/scalars/CSR buffers to be fed at runtime without
    // recompiling the whole program artifact.
    uint64_t (*get_symbol_u64)(void* ctx, uint64_t symbol_id);
    void* (*get_symbol_ptr)(void* ctx, uint64_t symbol_id);

    // Runtime slot storage (v9): mutable u64 slots for data-dependent control.
    // Slots are internal to the artifact runtime; default-initialized to 0.
    uint64_t (*get_slot_u64)(void* ctx, uint32_t slot);
    void (*set_slot_u64)(void* ctx, uint32_t slot, uint64_t value);

    // Tensor data access
    void* (*get_tensor_ptr)(void* ctx, uint32_t tensor_id);
    uint64_t (*get_tensor_stride)(void* ctx, uint32_t tensor_id, uint32_t dim);

    // Kernel registry
    KernelFn (*get_kernel)(void* ctx, uint32_t kernel_id);

    // Opaque context pointer (implementation-defined)
    void* ctx;
};

//==============================================================================
// Channel Operations (for CSP workloads; optional)
//==============================================================================

struct ChannelHandle {
    void* ch;
    uint32_t id;
};

// Send with timestamp (for CSPT). Implementations may ignore timestamp if CSPT
// is not enabled.
void channel_send(ChannelHandle ch, const void* data, size_t size, uint64_t timestamp);

// Blocking recv; returns timestamp for time acceleration (0 if unused).
uint64_t channel_recv(ChannelHandle ch, void* data, size_t size);

// Try operations (non-blocking). Return 1 on success, 0 on failure.
int channel_try_send(ChannelHandle ch, const void* data, size_t size, uint64_t timestamp);
int channel_try_recv(ChannelHandle ch, void* data, size_t size, uint64_t* timestamp);

// Close channel (signals completion)
void channel_close(ChannelHandle ch);

//==============================================================================
// Workload Function Signature
//==============================================================================

// Returns max cycles across all processes (or 0 if CSPT disabled).
using WorkloadFn = uint64_t (*)(RuntimeContext* ctx, CSPTContext* cspt);

}  // extern "C"
