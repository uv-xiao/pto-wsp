/**
 * PTO Workload-Schedule Programming (PTO-WSP)
 *
 * This header provides the core abstractions for dynamic workload support:
 * - Dim: Static or dynamic dimension
 * - IterationSpace: Multi-dimensional work space
 * - WorkDescriptor: Runtime work specification
 * - TierConfig: Compile-time kernel tier definitions
 * - WorkPlanner: Host-side planning and descriptor generation
 */

#ifndef PTO_RUNTIME_HPP
#define PTO_RUNTIME_HPP

#include <cstdint>
#include <array>
#include <algorithm>
#include <tuple>
#include <cassert>

namespace pto {
namespace runtime {

// =============================================================================
// Constants
// =============================================================================

constexpr int DYNAMIC = -1;

// =============================================================================
// Dim: Static or Dynamic Dimension
// =============================================================================

template <int StaticSize = DYNAMIC>
struct Dim {
    static constexpr int static_size = StaticSize;
    static constexpr bool is_dynamic = (StaticSize == DYNAMIC);

    int size;

    constexpr Dim() : size(StaticSize) {}

    explicit Dim(int runtime_size) : size(runtime_size) {
        static_assert(is_dynamic, "Cannot set runtime size for static dimension");
    }

    constexpr int get_size() const {
        if constexpr (is_dynamic) {
            return size;
        } else {
            return StaticSize;
        }
    }
};

template <int N> using StaticDim = Dim<N>;
using DynamicDim = Dim<DYNAMIC>;

// =============================================================================
// IterationSpace: Multi-dimensional Work Space
// =============================================================================

template <typename... Dims>
class IterationSpace {
public:
    static constexpr size_t num_dims = sizeof...(Dims);
    using DimsTuple = std::tuple<Dims...>;

private:
    DimsTuple dims_;

public:
    constexpr IterationSpace() = default;

    template <size_t I>
    void set_dim(int size) {
        static_assert(std::tuple_element_t<I, DimsTuple>::is_dynamic,
                     "Cannot set size of static dimension");
        std::get<I>(dims_).size = size;
    }

    template <size_t I>
    constexpr int get_dim() const {
        return std::get<I>(dims_).get_size();
    }

    int total_work() const {
        return total_work_impl(std::make_index_sequence<num_dims>{});
    }

    void index_to_coords(int work_idx, int* coords) const {
        index_to_coords_impl(work_idx, coords, std::make_index_sequence<num_dims>{});
    }

private:
    template <size_t... Is>
    int total_work_impl(std::index_sequence<Is...>) const {
        return (std::get<Is>(dims_).get_size() * ...);
    }

    template <size_t... Is>
    void index_to_coords_impl(int idx, int* coords, std::index_sequence<Is...>) const {
        ((coords[Is] = idx % std::get<Is>(dims_).get_size(),
          idx /= std::get<Is>(dims_).get_size()), ...);
    }
};

// Common patterns
template <int Batch = DYNAMIC, int Head = DYNAMIC, int Chunk = DYNAMIC>
using AttentionSpace = IterationSpace<Dim<Batch>, Dim<Head>, Dim<Chunk>>;

// =============================================================================
// WorkDescriptor: Runtime Work Specification
// =============================================================================

struct alignas(8) WorkDescriptor {
    uint32_t work_id;
    uint8_t  tier;
    uint8_t  flags;
    uint16_t reserved;
    uint32_t params[4];

    static constexpr uint8_t FLAG_FIRST = 0x01;
    static constexpr uint8_t FLAG_LAST  = 0x02;
    static constexpr uint8_t FLAG_INIT  = 0x04;
};

// Pattern-specific parameter accessors
namespace params {

struct Attention {
    static void set(WorkDescriptor& d, uint32_t req, uint32_t head,
                   uint32_t kv_start, uint32_t kv_len) {
        d.params[0] = req;
        d.params[1] = head;
        d.params[2] = kv_start;
        d.params[3] = kv_len;
    }
    static uint32_t request_idx(const WorkDescriptor& d) { return d.params[0]; }
    static uint32_t head_idx(const WorkDescriptor& d) { return d.params[1]; }
    static uint32_t kv_start(const WorkDescriptor& d) { return d.params[2]; }
    static uint32_t kv_len(const WorkDescriptor& d) { return d.params[3]; }
    static uint32_t kv_end(const WorkDescriptor& d) { return d.params[2] + d.params[3]; }
};

}  // namespace params

// =============================================================================
// TierConfig: Compile-time Kernel Tier Definitions
// =============================================================================

template <int TierID, int MinSize, int MaxSize>
struct Tier {
    static constexpr int id = TierID;
    static constexpr int min_size = MinSize;
    static constexpr int max_size = MaxSize;

    static constexpr bool matches(int size) {
        return size >= MinSize && size <= MaxSize;
    }
};

template <typename... Tiers>
struct TierConfig {
    static constexpr size_t num_tiers = sizeof...(Tiers);

    static constexpr int select_tier(int size) {
        return select_tier_impl<Tiers...>(size);
    }

private:
    template <typename T, typename... Rest>
    static constexpr int select_tier_impl(int size) {
        if (T::matches(size)) return T::id;
        if constexpr (sizeof...(Rest) > 0) {
            return select_tier_impl<Rest...>(size);
        }
        return -1;
    }
};

// Common tier configurations
using DecodeAttentionTiers = TierConfig<
    Tier<0, 1, 1024>,
    Tier<1, 1025, 4096>,
    Tier<2, 4097, 16384>,
    Tier<3, 16385, 131072>
>;

// =============================================================================
// WorkPlanner: Host-side Planning
// =============================================================================

enum class PlanResult {
    OK,
    BUFFER_OVERFLOW,
    UNSUPPORTED_SIZE,
    INVALID_PARAMS,
};

struct PlanConfig {
    int chunk_min = 256;
    int chunk_max = 4096;
    int max_work_units = 65536;
    bool balance_chunks = true;
};

template <typename Space, typename TierCfg, typename PatternParams>
class WorkPlanner {
public:
    explicit WorkPlanner(const PlanConfig& config = {}) : config_(config) {}

    int plan_chunk_size(const int* seq_lens, int batch_size, int num_heads) {
        int low = config_.chunk_min;
        int high = config_.chunk_max;

        while (low < high) {
            int mid = (low + high) / 2;
            int total = count_work_units(seq_lens, batch_size, num_heads, mid);

            if (total > config_.max_work_units) {
                low = mid + 1;
            } else {
                high = mid;
            }
        }
        return low;
    }

    PlanResult generate(
        const int* seq_lens,
        int batch_size,
        int num_heads,
        int chunk_size,
        WorkDescriptor* out_buffer,
        int buffer_capacity,
        int* out_work_count
    ) {
        int work_idx = 0;

        for (int batch = 0; batch < batch_size; batch++) {
            int seq_len = seq_lens[batch];
            int num_chunks = (seq_len + chunk_size - 1) / chunk_size;
            int tier = TierCfg::select_tier(seq_len);

            if (tier < 0) return PlanResult::UNSUPPORTED_SIZE;

            for (int head = 0; head < num_heads; head++) {
                for (int chunk = 0; chunk < num_chunks; chunk++) {
                    if (work_idx >= buffer_capacity) {
                        return PlanResult::BUFFER_OVERFLOW;
                    }

                    int kv_start = chunk * chunk_size;
                    int kv_len = std::min(chunk_size, seq_len - kv_start);

                    out_buffer[work_idx] = WorkDescriptor{
                        .work_id = static_cast<uint32_t>(work_idx),
                        .tier = static_cast<uint8_t>(tier),
                        .flags = static_cast<uint8_t>(
                            (chunk == 0 ? WorkDescriptor::FLAG_FIRST : 0) |
                            (chunk == num_chunks - 1 ? WorkDescriptor::FLAG_LAST : 0)
                        ),
                        .reserved = 0,
                        .params = {}
                    };
                    PatternParams::set(out_buffer[work_idx], batch, head, kv_start, kv_len);
                    work_idx++;
                }
            }
        }

        *out_work_count = work_idx;
        return PlanResult::OK;
    }

    int get_total_work(const int* seq_lens, int batch_size, int num_heads, int chunk_size) {
        return count_work_units(seq_lens, batch_size, num_heads, chunk_size);
    }

private:
    PlanConfig config_;

    int count_work_units(const int* seq_lens, int batch, int heads, int chunk) {
        int total = 0;
        for (int i = 0; i < batch; i++) {
            total += heads * ((seq_lens[i] + chunk - 1) / chunk);
        }
        return total;
    }
};

// Convenience alias
using AttentionPlanner = WorkPlanner<
    AttentionSpace<DYNAMIC, DYNAMIC, DYNAMIC>,
    DecodeAttentionTiers,
    params::Attention
>;

}  // namespace runtime
}  // namespace pto

#endif  // PTO_RUNTIME_HPP
