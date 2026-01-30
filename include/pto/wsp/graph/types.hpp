// Copyright 2026 PTO-WSP Authors
// SPDX-License-Identifier: MIT

#pragma once  // API-2 FIX: Consistent include guard style

#include <cstdint>
#include <atomic>

namespace pto::wsp::graph {

// ============================================================
// Type Aliases
// ============================================================

using TaskId = uint32_t;
using KernelId = uint16_t;
using StreamId = uint16_t;
using TargetId = uint16_t;  // Per-core/per-tile affinity

constexpr TaskId INVALID_TASK_ID = UINT32_MAX;
constexpr KernelId INVALID_KERNEL_ID = UINT16_MAX;
constexpr StreamId INVALID_STREAM_ID = UINT16_MAX;

// ============================================================
// Execution Domain and Pool
// ============================================================

/// Where does the task run?
enum class ExecDomain : uint8_t {
    HostCPU = 0,
    AscendAICore = 1,
    AMDAIETile = 2,
};

/// For dual-queue scheduling (vector/cube workers)
enum class ExecPool : uint8_t {
    Vector = 0,
    Cube = 1,
    Any = 255,
};

// ============================================================
// Tensor Region for Dependency Tracking
// ============================================================

/// 2D tile-based tensor region (device-copyable)
struct TensorRegion2D {
    uint64_t base;           // Pointer/handle (device-visible)
    int32_t row_off;
    int32_t col_off;
    int32_t rows;
    int32_t cols;

    /// Check if this region overlaps with another
    bool overlaps(const TensorRegion2D& other) const {
        if (base != other.base) return false;
        // Check if rectangles overlap
        bool no_overlap_x = (col_off + cols <= other.col_off) ||
                            (other.col_off + other.cols <= col_off);
        bool no_overlap_y = (row_off + rows <= other.row_off) ||
                            (other.row_off + other.rows <= row_off);
        return !(no_overlap_x || no_overlap_y);
    }

    /// Check if this region contains another
    bool contains(const TensorRegion2D& other) const {
        if (base != other.base) return false;
        return (row_off <= other.row_off) &&
               (col_off <= other.col_off) &&
               (row_off + rows >= other.row_off + other.rows) &&
               (col_off + cols >= other.col_off + other.cols);
    }

    /// Check if regions are exactly equal
    bool operator==(const TensorRegion2D& other) const {
        return base == other.base &&
               row_off == other.row_off &&
               col_off == other.col_off &&
               rows == other.rows &&
               cols == other.cols;
    }
};

// ============================================================
// Task I/O Descriptor
// ============================================================

/// Task input/output descriptor (device-copyable)
struct TaskIO {
    TensorRegion2D region;
    bool is_output;     // API-6 FIX: Use bool for clarity
    uint8_t reserved[3];
};

// ============================================================
// Schedule Tags (Bitfield)
// ============================================================

/// Schedule metadata flags
enum class SchedTag : uint16_t {
    None = 0,
    Barrier = 1 << 0,      // Sync point
    FlushBatch = 1 << 1,   // Flush dependency batch
    WindowStart = 1 << 2,  // Start of task window
    WindowEnd = 1 << 3,    // End of task window
    ProfileMark = 1 << 4,  // Profiling marker
};

inline SchedTag operator|(SchedTag a, SchedTag b) {
    return static_cast<SchedTag>(static_cast<uint16_t>(a) | static_cast<uint16_t>(b));
}

inline SchedTag operator&(SchedTag a, SchedTag b) {
    return static_cast<SchedTag>(static_cast<uint16_t>(a) & static_cast<uint16_t>(b));
}

inline bool has_tag(SchedTag tags, SchedTag flag) {
    return (static_cast<uint16_t>(tags) & static_cast<uint16_t>(flag)) != 0;
}

// ============================================================
// Task Node (Device-Copyable POD)
// ============================================================

/// Maximum limits for task arguments and I/O
constexpr size_t MAX_TASK_ARGS = 16;
constexpr size_t MAX_TASK_IO = 8;

/// Plain-old-data task node (device-copyable)
/// Note: Variable-length data (args, io) stored separately
struct TaskNodePod {
    TaskId id;
    KernelId kernel;
    ExecDomain domain;
    ExecPool pool;
    StreamId stream;
    TargetId affinity;

    // Dependencies (fanin = remaining input deps to decrement)
    int32_t fanin;
    uint32_t fanout_begin;   // Index into flat fanout_edges[]
    uint16_t fanout_count;
    uint16_t reserved;

    // Arguments and I/O counts
    uint16_t num_u64_args;
    uint16_t num_io;

    // Schedule metadata
    SchedTag sched_tags;
    uint16_t sched_reserved;

    // Inline storage for small tasks
    uint64_t args[MAX_TASK_ARGS];
    TaskIO io[MAX_TASK_IO];
};

// ============================================================
// Producer Reference (for TensorMap)
// ============================================================

/// Reference to a task that produces a tensor region
struct ProducerRef {
    TaskId producer;
    uint32_t generation;  // For GC/window support

    bool operator==(const ProducerRef& other) const {
        return producer == other.producer && generation == other.generation;
    }
};

// ============================================================
// Atomic Task Node (for runtime execution)
// ============================================================

/// Runtime task node with atomic fanin counter
/// UNUSED-1 NOTE: Currently not used - TaskGraphRuntime stores atomic
/// fanin counters in a separate vector for better cache locality.
/// Retained for potential future use (e.g., per-task completion tracking).
struct TaskNodeRuntime {
    TaskId id;
    KernelId kernel;
    ExecDomain domain;
    ExecPool pool;
    StreamId stream;

    std::atomic<int32_t> fanin_remaining;
    std::atomic<bool> is_complete;

    TaskNodeRuntime() : fanin_remaining(0), is_complete(false) {}

    TaskNodeRuntime(const TaskNodePod& pod)
        : id(pod.id), kernel(pod.kernel), domain(pod.domain),
          pool(pod.pool), stream(pod.stream),
          fanin_remaining(pod.fanin), is_complete(false) {}
};

}  // namespace pto::wsp::graph
