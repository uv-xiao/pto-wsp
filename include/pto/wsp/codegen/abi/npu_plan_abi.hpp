// Copyright 2026 PTO-WSP Authors
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>

// This header defines the minimal stable ABI for v9 Ascend NPU artifacts:
// host ↔ AICPU ↔ AICore task execution with on-device task expansion.
//
// It intentionally mirrors the "u64 args" convention used by PTO-ISA and the
// `KernelTaskDesc` ABI used by CPU-sim codegen, so the same kernel IR can be
// codegenned for both CPU sim and NPU.
//
// NOTE: This environment may not have Ascend/CANN. The codegen pipeline should
// still emit these ABIs + sources to a "looks correct" state; execution is
// validated on real devices.

extern "C" {

// -----------------------------------------------------------------------------
// Symbols (runtime binding, shared between host/AICPU)
// -----------------------------------------------------------------------------

// Stable symbol ID: v9 uses a 64-bit name hash (FNV-1a).
struct NpuSymbolU64KV {
    uint64_t id;
    uint64_t value;
};

struct NpuSymbolPtrKV {
    uint64_t id;
    uint64_t ptr;
};

// -----------------------------------------------------------------------------
// Task descriptor (AICPU → AICore)
// -----------------------------------------------------------------------------

// Conservative fixed maxima for device-side queues (aligned with pto-isa-wc style).
// If workloads exceed these, codegen should lower differently (e.g., multi-stage).
constexpr uint32_t PTO_WSP_NPU_MAX_ARGS = 32;
constexpr uint32_t PTO_WSP_NPU_MAX_TENSORS = 16;
constexpr uint32_t PTO_WSP_NPU_MAX_TENSOR_RANK = 8;
constexpr uint32_t PTO_WSP_NPU_MAX_SLOTS = 256;

struct NpuTaskDesc {
    uint32_t kernel_id;
    uint32_t task_id;

    // v9 scheduling programmability (preserved in emitted artifacts):
    // - dispatch assigns each task a target_id (e.g., AICPU/AICore queue index)
    // - task_window is enforced by the runtime/driver (if available)
    //
    // NOTE: This environment may not run these artifacts. We still preserve
    // these fields so the emitted source tree is "looks correct" and runnable
    // in real Ascend/CANN environments.
    uint32_t target_id;

    uint32_t num_args;
    uint32_t num_axis_args;
    uint32_t num_tensors;
    uint32_t reserved0;

    uint64_t args[PTO_WSP_NPU_MAX_ARGS];
    uint64_t tensor_ptrs[PTO_WSP_NPU_MAX_TENSORS];
    uint64_t tensor_strides[PTO_WSP_NPU_MAX_TENSORS * 2];

    // Optional timing fields.
    // v9 contract (no CANN required for codegen bring-up):
    // - device kernels return a u64 "cycle report" (PTO-ISA convention)
    // - dispatcher records it into `t_end` (with `t_begin` reserved for future
    //   real device timestamps)
    uint64_t t_begin;
    uint64_t t_end;
};

// -----------------------------------------------------------------------------
// Compact plan (host → AICPU)
// -----------------------------------------------------------------------------

struct NpuPlanDesc {
    // Tensor table (host uploads device pointers + element strides).
    uint32_t num_tensors;
    uint32_t tensor_ranks[PTO_WSP_NPU_MAX_TENSORS];
    uint64_t tensor_base_ptrs[PTO_WSP_NPU_MAX_TENSORS];
    uint64_t tensor_strides[PTO_WSP_NPU_MAX_TENSORS * PTO_WSP_NPU_MAX_TENSOR_RANK];

    // Task buffer allocated by host in device memory.
    NpuTaskDesc* tasks;
    uint32_t max_tasks;
    uint32_t num_tasks;  // set by AICPU after expansion

    // Runtime-bound symbols (host uploads per-run).
    const NpuSymbolU64KV* sym_u64;
    uint32_t num_sym_u64;
    const NpuSymbolPtrKV* sym_ptr;
    uint32_t num_sym_ptr;

    // v9 runtime slots (AICPU working storage, default-initialized to 0).
    // Used for tensor-driven predicates and other data-dependent control logic.
    uint64_t slots_u64[PTO_WSP_NPU_MAX_SLOTS];

    // v9 schedule preservation (emit-only in this environment):
    // - dispatch: policy + target count (AICPU-side may also encode per-task target_id)
    // - task_window: stall-only constraint on in-flight tasks (tasks unit only)
    uint32_t dispatch_policy;
    uint32_t dispatch_num_targets;
    uint64_t task_window_tasks;
};

}  // extern "C"
