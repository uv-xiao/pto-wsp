// Body snippet for @kernel(cpp_body_path=...).
//
// This file is inserted into:
//   extern "C" uint64_t <kernel>(const KernelTaskDesc* task, CSPTContext* cspt)
// by PTO-RT's custom-kernel wrapper. The wrapper already includes:
//   - pto/rt/codegen/abi/kernel_abi.hpp
//   - pto/rt/codegen/abi/ptoisa_bridge.hpp
//
// Contract:
// - tensor_ptrs[0] = scores: f32 [1,32]
// - tensor_ptrs[1] = out_idx: i32 [1,8]
// - scalar args: args[num_axis_args + 0] = pad_to (i32) (tier-dependent)

float* scores = (float*)task->tensor_ptrs[0];
int32_t* out = (int32_t*)task->tensor_ptrs[1];
const uint64_t scores_s3 = task->tensor_strides[0];
const uint64_t scores_s4 = task->tensor_strides[1];
const uint64_t out_s3 = task->tensor_strides[2];
const uint64_t out_s4 = task->tensor_strides[3];

(void)out_s3;

const int32_t pad_to = (int32_t)task->args[task->num_axis_args + 0];

// Load scores to a local tile.
using GTShapeS = pto::Shape<1, 1, 1, pto::DYNAMIC, pto::DYNAMIC>;
using GTStrideS = pto::Stride<1, 1, 1, pto::DYNAMIC, pto::DYNAMIC>;
pto::GlobalTensor<float, GTShapeS, GTStrideS, pto::Layout::ND> g_scores(
    scores, GTShapeS(1, 32), GTStrideS((int)scores_s3, (int)scores_s4));

auto v_scores = std::make_unique<pto::Tile<pto::TileType::Vec, float, 1, 32, pto::BLayout::RowMajor, pto::DYNAMIC, pto::DYNAMIC>>(1, 32);
pto_wsp::ptoisa::TLOAD(*v_scores, g_scores);

// Build an index tile [0..31].
auto v_idx = std::make_unique<pto::Tile<pto::TileType::Vec, uint32_t, 1, 32, pto::BLayout::RowMajor, pto::DYNAMIC, pto::DYNAMIC>>(1, 32);
for (int i = 0; i < 32; ++i) {
  v_idx->data()[i] = (uint32_t)i;
}

// TSORT32 writes (score,index) pairs to dst; for float, index is stored as a float value.
auto v_pairs = std::make_unique<pto::Tile<pto::TileType::Vec, float, 1, 64, pto::BLayout::RowMajor, pto::DYNAMIC, pto::DYNAMIC>>(1, 64);
pto_wsp::ptoisa::TSORT32(*v_pairs, *v_scores, *v_idx);

// Emit top-8 indices.
for (int i = 0; i < 8; ++i) {
  const float idx_f = v_pairs->data()[2 * i + 1];
  const int32_t idx_i = (int32_t)idx_f;
  out[i * (int64_t)out_s4] = idx_i;
}

// Tier padding "cost": add deterministic extra cycles so tiers have distinct timings.
#if defined(__CPU_SIM)
if (pad_to > 32) {
  const uint64_t extra = (uint64_t)(pad_to / 32) * 128ull;
  pto::cpu_sim::add_cycles(extra);
}
#endif

