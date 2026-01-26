// Copyright 2026 PTO-RT Authors
// SPDX-License-Identifier: Apache-2.0

#pragma once  // API-2 FIX: Consistent include guard style
// (was: #ifndef PTO_WSP_GRAPH_GRAPH_HPP)

/// @file graph.hpp
/// @brief Unified header for PTO-RT task graph infrastructure.
///
/// This header provides the shared infrastructure for all backends:
/// - TaskNodePod: Device-copyable task descriptor
/// - TaskGraphStorage: Immutable built graph (device-copyable)
/// - TensorMap: Dependency inference and tracking
/// - ReadyQueueSet: Multi-queue task scheduling
/// - TaskGraphRuntime: Runtime execution state

#include "pto/rt/graph/types.hpp"
#include "pto/rt/graph/storage.hpp"
#include "pto/rt/graph/tensor_map.hpp"
#include "pto/rt/graph/ready_queue.hpp"
#include "pto/rt/graph/runtime.hpp"

#include <memory>  // INC-3 FIX: for std::unique_ptr, std::make_unique

namespace pto::wsp::graph {

/// Version of the graph infrastructure
constexpr const char* GRAPH_VERSION = "1.0.0";

/// Create a fixed tensor map (fast, bounded capacity)
inline std::unique_ptr<TensorMap> make_fixed_tensor_map(size_t capacity = FixedTensorMap::DEFAULT_CAPACITY) {
    return std::make_unique<FixedTensorMap>(capacity);
}

/// Create a dynamic tensor map (flexible, unbounded)
inline std::unique_ptr<TensorMap> make_dynamic_tensor_map() {
    return std::make_unique<DynamicTensorMap>();
}

/// Create a single ready queue set (no dual-queue)
inline std::unique_ptr<ReadyQueueSet> make_single_queue_set() {
    return std::make_unique<SingleQueueSet>();
}

/// Create a dual queue set (vector/cube)
inline std::unique_ptr<ReadyQueueSet> make_dual_queue_set() {
    return std::make_unique<DualQueueSet>();
}

}  // namespace pto::wsp::graph

