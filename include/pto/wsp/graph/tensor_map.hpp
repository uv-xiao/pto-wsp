// Copyright 2026 PTO-WSP Authors
// SPDX-License-Identifier: MIT

#pragma once  // API-2 FIX: Consistent include guard style
// (was: #ifndef PTO_WSP_GRAPH_TENSOR_MAP_HPP)

#include "pto/wsp/graph/types.hpp"

#include <vector>
#include <unordered_map>
#include <unordered_set>  // PERF-3 FIX: for O(1) duplicate check
#include <optional>
#include <cstdint>
#include <algorithm>

namespace pto::wsp::graph {

// ============================================================
// TensorMap Interface
// ============================================================

/// Abstract interface for tensor region → producer mapping
class TensorMap {
public:
    virtual ~TensorMap() = default;

    /// Register output region produced by task
    virtual void insert_output(const TensorRegion2D& region, ProducerRef producer) = 0;

    /// Find latest producer overlapping region (exact match first)
    virtual std::optional<ProducerRef> find_producer(const TensorRegion2D& region) const = 0;

    /// task_window support: garbage collect entries before oldest_live
    virtual void gc_before(TaskId oldest_live_task) = 0;

    /// Clear all entries
    virtual void clear() = 0;

    /// Get number of entries
    virtual size_t size() const = 0;
};

// ============================================================
// Fixed Tensor Map (Hash Table + Entry Pool)
// ============================================================

/// Entry in the fixed tensor map
struct FixedMapEntry {
    TensorRegion2D region;
    ProducerRef producer;
    bool valid;
};

/// Hash function for tensor base addresses
struct TensorBaseHash {
    size_t operator()(uint64_t base) const {
        // Simple hash mixing
        return static_cast<size_t>(base ^ (base >> 32));
    }
};

/// Fixed-capacity tensor map suitable for AICPU and fast CPU simulation
/// Uses open addressing with linear probing
class FixedTensorMap : public TensorMap {
public:
    static constexpr size_t DEFAULT_CAPACITY = 8192;
    static constexpr size_t MAX_ENTRIES_PER_BASE = 64;

    explicit FixedTensorMap(size_t capacity = DEFAULT_CAPACITY)
        : capacity_(capacity) {
        entries_.resize(capacity);
        for (auto& e : entries_) {
            e.valid = false;
        }
    }

    void insert_output(const TensorRegion2D& region, ProducerRef producer) override {
        // Find bucket for this base address
        size_t bucket = hash_(region.base) % capacity_;

        // Linear probing to find empty slot or existing entry for same region
        for (size_t i = 0; i < MAX_ENTRIES_PER_BASE; ++i) {
            size_t idx = (bucket + i) % capacity_;
            auto& entry = entries_[idx];

            if (!entry.valid) {
                // Empty slot - insert here
                entry.region = region;
                entry.producer = producer;
                entry.valid = true;
                size_++;
                return;
            }

            if (entry.region == region) {
                // Same region - update producer (latest write wins)
                entry.producer = producer;
                return;
            }
        }

        // Table too full - this shouldn't happen with proper sizing
        // In production, would need to handle overflow
    }

    std::optional<ProducerRef> find_producer(const TensorRegion2D& region) const override {
        size_t bucket = hash_(region.base) % capacity_;

        std::optional<ProducerRef> best_match;
        TaskId best_task = 0;

        // Search for exact match first, then overlapping
        for (size_t i = 0; i < MAX_ENTRIES_PER_BASE; ++i) {
            size_t idx = (bucket + i) % capacity_;
            const auto& entry = entries_[idx];

            if (!entry.valid) {
                // End of chain for this bucket
                break;
            }

            if (entry.region.base != region.base) {
                continue;  // Different tensor
            }

            // Check for exact match
            if (entry.region == region) {
                return entry.producer;
            }

            // Check for overlap (and prefer latest producer)
            if (entry.region.overlaps(region) && entry.producer.producer > best_task) {
                best_match = entry.producer;
                best_task = entry.producer.producer;
            }
        }

        return best_match;
    }

    void gc_before(TaskId oldest_live_task) override {
        for (auto& entry : entries_) {
            if (entry.valid && entry.producer.producer < oldest_live_task) {
                entry.valid = false;
                size_--;
            }
        }
    }

    void clear() override {
        for (auto& e : entries_) {
            e.valid = false;
        }
        size_ = 0;
    }

    size_t size() const override { return size_; }
    size_t capacity() const { return capacity_; }

private:
    std::vector<FixedMapEntry> entries_;
    size_t capacity_;
    size_t size_ = 0;
    TensorBaseHash hash_;
};

// ============================================================
// Dynamic Tensor Map (std::unordered_map based)
// ============================================================

/// Dynamic tensor map for debugging and large workloads
/// More flexible but slower than FixedTensorMap
class DynamicTensorMap : public TensorMap {
public:
    DynamicTensorMap() = default;

    void insert_output(const TensorRegion2D& region, ProducerRef producer) override {
        auto& entries = map_[region.base];

        // Check for existing entry with same region
        for (auto& entry : entries) {
            if (entry.region == region) {
                entry.producer = producer;
                return;
            }
        }

        // Add new entry
        entries.push_back({region, producer, true});
        size_++;
    }

    std::optional<ProducerRef> find_producer(const TensorRegion2D& region) const override {
        auto it = map_.find(region.base);
        if (it == map_.end()) {
            return std::nullopt;
        }

        std::optional<ProducerRef> best_match;
        TaskId best_task = 0;

        for (const auto& entry : it->second) {
            if (!entry.valid) continue;

            // Exact match
            if (entry.region == region) {
                return entry.producer;
            }

            // Overlap (prefer latest)
            if (entry.region.overlaps(region) && entry.producer.producer > best_task) {
                best_match = entry.producer;
                best_task = entry.producer.producer;
            }
        }

        return best_match;
    }

    void gc_before(TaskId oldest_live_task) override {
        for (auto& [base, entries] : map_) {
            entries.erase(
                std::remove_if(entries.begin(), entries.end(),
                    [&](const FixedMapEntry& e) {
                        if (e.valid && e.producer.producer < oldest_live_task) {
                            size_--;
                            return true;
                        }
                        return false;
                    }),
                entries.end());
        }
    }

    void clear() override {
        map_.clear();
        size_ = 0;
    }

    size_t size() const override { return size_; }

private:
    std::unordered_map<uint64_t, std::vector<FixedMapEntry>> map_;
    size_t size_ = 0;
};

// ============================================================
// Dependency Analyzer
// ============================================================

/// Analyzes task I/O to infer dependencies via TensorMap
class DependencyAnalyzer {
public:
    explicit DependencyAnalyzer(TensorMap& tensor_map)
        : tensor_map_(tensor_map), generation_(0) {}

    /// Analyze a task's I/O and return dependencies
    std::vector<TaskId> analyze_dependencies(const TaskNodePod& task) {
        std::vector<TaskId> deps;
        // PERF-3 FIX: Use unordered_set for O(1) duplicate check instead of O(n) linear search
        std::unordered_set<TaskId> seen;

        // For each input, find the producer
        for (uint16_t i = 0; i < task.num_io; ++i) {
            const auto& io = task.io[i];
            if (io.is_output) continue;  // Skip outputs

            auto producer = tensor_map_.find_producer(io.region);
            if (producer) {
                // O(1) duplicate check
                if (seen.insert(producer->producer).second) {
                    deps.push_back(producer->producer);
                }
            }
        }

        return deps;
    }

    /// Register a task's outputs in the tensor map
    void register_outputs(const TaskNodePod& task) {
        for (uint16_t i = 0; i < task.num_io; ++i) {
            const auto& io = task.io[i];
            if (!io.is_output) continue;  // Skip inputs

            tensor_map_.insert_output(io.region, ProducerRef{task.id, generation_});
        }
    }

    /// Increment generation counter (for task window GC)
    void next_generation() {
        generation_++;
    }

    uint32_t generation() const { return generation_; }

private:
    TensorMap& tensor_map_;
    uint32_t generation_;
};

}  // namespace pto::wsp::graph
