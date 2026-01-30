// Copyright 2026 PTO-WSP Authors
// SPDX-License-Identifier: MIT

#pragma once

#include "pto/wsp/codegen/abi/kernel_abi.hpp"

// PTO-ISA entrypoint.
//
// PTO-ISA entrypoint.
//
// - CPU simulation builds define `__CPU_SIM` to enable portable stubs.
// - NPU builds should provide the required Ascend toolchain headers/macros.
//
// PTO-ISA's common instruction wrappers reference these enums (for UF-aware
// overloads) but they are normally provided by NPU headers. For CPU simulation
// builds we provide minimal definitions to keep compilation self-contained.
#if defined(__CPU_SIM)
namespace pto {
enum class STPhase : uint8_t { Unspecified = 0 };
enum class AccPhase : uint8_t { Unspecified = 0 };
}  // namespace pto
#endif

// We intentionally include the public PTO-ISA umbrella header so generated code
// can stay minimal.
#include <pto/pto-inst.hpp>

//------------------------------------------------------------------------------
// Minimal timing wrappers
//------------------------------------------------------------------------------

#define PTO_WSP_ADVANCE(cspt_ptr, cycles_expr) \
    do { \
        if ((cspt_ptr) && (cspt_ptr)->advance_cycles && (cycles_expr) != 0) { \
            (cspt_ptr)->advance_cycles((cspt_ptr)->ctx, static_cast<uint64_t>(cycles_expr)); \
        } \
    } while (0)

//------------------------------------------------------------------------------
// NOTE
//------------------------------------------------------------------------------
// This header intentionally does not prescribe a particular lowering strategy.
// Kernel emitters may:
// - Call PTO-ISA tile primitives directly, OR
// - Use helper functions/macros provided by PTO-WSP.
//
// As codegen matures, add wrappers here to model CSPT timing in a consistent way.

//------------------------------------------------------------------------------
// CPU-sim cycle counter + instruction wrappers (v9)
//------------------------------------------------------------------------------

#if defined(__CPU_SIM)
#include <cstdint>
#include <utility>

namespace pto::cpu_sim {
inline thread_local std::uint64_t cycle_counter = 0;

inline void reset_cycles() { cycle_counter = 0; }
inline void add_cycles(std::uint64_t cycles) { cycle_counter += cycles; }
inline std::uint64_t read_cycles() { return cycle_counter; }
}  // namespace pto::cpu_sim
#endif

namespace pto_wsp::ptoisa {

template <typename Tile>
inline void _cpu_sim_add_tile_cycles(const Tile& t) {
#if defined(__CPU_SIM)
    pto::cpu_sim::add_cycles(
        static_cast<std::uint64_t>(t.GetValidRow()) * static_cast<std::uint64_t>(t.GetValidCol()));
#else
    (void)t;
#endif
}

template <typename TileData, typename GlobalData, typename... WaitEvents>
inline auto TLOAD(TileData& dst, GlobalData& src, WaitEvents&... events) {
    auto ret = pto::TLOAD(dst, src, events...);
    _cpu_sim_add_tile_cycles(dst);
    return ret;
}

template <typename GlobalData, typename TileData, typename... WaitEvents>
inline auto TSTORE(GlobalData& dst, TileData& src, WaitEvents&... events) {
    auto ret = pto::TSTORE(dst, src, events...);
    _cpu_sim_add_tile_cycles(src);
    return ret;
}

template <typename TileRes, typename TileLeft, typename TileRight, typename... WaitEvents>
inline auto TMATMUL(TileRes& c, TileLeft& a, TileRight& b, WaitEvents&... events) {
    auto ret = pto::TMATMUL(c, a, b, events...);
#if defined(__CPU_SIM)
    pto::cpu_sim::add_cycles(
        static_cast<std::uint64_t>(a.GetValidRow()) * static_cast<std::uint64_t>(a.GetValidCol()) *
        static_cast<std::uint64_t>(b.GetValidCol()));
#endif
    return ret;
}

template <typename TileDst, typename TileSrc>
inline auto TMOV(TileDst& dst, TileSrc& src) {
    auto ret = pto::TMOV(dst, src);
    _cpu_sim_add_tile_cycles(dst);
    return ret;
}

template <typename TileDst, typename TileSrc, typename... Rest>
inline auto TCVT(TileDst& dst, TileSrc& src, Rest&&... rest) {
    auto ret = pto::TCVT(dst, src, std::forward<Rest>(rest)...);
    _cpu_sim_add_tile_cycles(dst);
    return ret;
}

template <typename TileDst, typename TileA, typename TileB, typename... WaitEvents>
inline auto TADD(TileDst& dst, TileA& a, TileB& b, WaitEvents&... events) {
    auto ret = pto::TADD(dst, a, b, events...);
    _cpu_sim_add_tile_cycles(dst);
    return ret;
}
template <typename TileDst, typename TileA, typename TileB, typename... WaitEvents>
inline auto TSUB(TileDst& dst, TileA& a, TileB& b, WaitEvents&... events) {
    auto ret = pto::TSUB(dst, a, b, events...);
    _cpu_sim_add_tile_cycles(dst);
    return ret;
}
template <typename TileDst, typename TileA, typename TileB, typename... WaitEvents>
inline auto TMUL(TileDst& dst, TileA& a, TileB& b, WaitEvents&... events) {
    auto ret = pto::TMUL(dst, a, b, events...);
    _cpu_sim_add_tile_cycles(dst);
    return ret;
}
template <typename TileDst, typename TileA, typename TileB, typename... WaitEvents>
inline auto TDIV(TileDst& dst, TileA& a, TileB& b, WaitEvents&... events) {
    auto ret = pto::TDIV(dst, a, b, events...);
    _cpu_sim_add_tile_cycles(dst);
    return ret;
}
template <typename TileDst, typename TileA, typename TileB, typename... WaitEvents>
inline auto TMAX(TileDst& dst, TileA& a, TileB& b, WaitEvents&... events) {
    auto ret = pto::TMAX(dst, a, b, events...);
    _cpu_sim_add_tile_cycles(dst);
    return ret;
}
template <typename TileDst, typename TileA, typename TileB, typename... WaitEvents>
inline auto TMIN(TileDst& dst, TileA& a, TileB& b, WaitEvents&... events) {
    auto ret = pto::TMIN(dst, a, b, events...);
    _cpu_sim_add_tile_cycles(dst);
    return ret;
}

template <typename TileDst, typename TileA, typename TileB, typename... WaitEvents>
inline auto TROWEXPANDADD(TileDst& dst, TileA& a, TileB& b, WaitEvents&... events) {
    auto ret = pto::TROWEXPANDADD(dst, a, b, events...);
    _cpu_sim_add_tile_cycles(dst);
    return ret;
}
template <typename TileDst, typename TileA, typename TileB, typename... WaitEvents>
inline auto TROWEXPANDSUB(TileDst& dst, TileA& a, TileB& b, WaitEvents&... events) {
    auto ret = pto::TROWEXPANDSUB(dst, a, b, events...);
    _cpu_sim_add_tile_cycles(dst);
    return ret;
}
template <typename TileDst, typename TileA, typename TileB, typename... WaitEvents>
inline auto TROWEXPANDMUL(TileDst& dst, TileA& a, TileB& b, WaitEvents&... events) {
    auto ret = pto::TROWEXPANDMUL(dst, a, b, events...);
    _cpu_sim_add_tile_cycles(dst);
    return ret;
}
template <typename TileDst, typename TileA, typename TileB, typename... WaitEvents>
inline auto TROWEXPANDDIV(TileDst& dst, TileA& a, TileB& b, WaitEvents&... events) {
    auto ret = pto::TROWEXPANDDIV(dst, a, b, events...);
    _cpu_sim_add_tile_cycles(dst);
    return ret;
}
template <typename TileDst, typename TileA, typename TileB, typename... WaitEvents>
inline auto TROWEXPANDMAX(TileDst& dst, TileA& a, TileB& b, WaitEvents&... events) {
    auto ret = pto::TROWEXPANDMAX(dst, a, b, events...);
    _cpu_sim_add_tile_cycles(dst);
    return ret;
}
template <typename TileDst, typename TileA, typename TileB, typename... WaitEvents>
inline auto TROWEXPANDMIN(TileDst& dst, TileA& a, TileB& b, WaitEvents&... events) {
    auto ret = pto::TROWEXPANDMIN(dst, a, b, events...);
    _cpu_sim_add_tile_cycles(dst);
    return ret;
}

template <typename TileDst, typename TileA, typename TileB, typename... WaitEvents>
inline auto TCOLEXPANDSUB(TileDst& dst, TileA& a, TileB& b, WaitEvents&... events) {
    auto ret = pto::TCOLEXPANDSUB(dst, a, b, events...);
    _cpu_sim_add_tile_cycles(dst);
    return ret;
}
template <typename TileDst, typename TileA, typename TileB, typename... WaitEvents>
inline auto TCOLEXPANDMUL(TileDst& dst, TileA& a, TileB& b, WaitEvents&... events) {
    auto ret = pto::TCOLEXPANDMUL(dst, a, b, events...);
    _cpu_sim_add_tile_cycles(dst);
    return ret;
}
template <typename TileDst, typename TileA, typename TileB, typename... WaitEvents>
inline auto TCOLEXPANDDIV(TileDst& dst, TileA& a, TileB& b, WaitEvents&... events) {
    auto ret = pto::TCOLEXPANDDIV(dst, a, b, events...);
    _cpu_sim_add_tile_cycles(dst);
    return ret;
}

template <typename TileDst, typename TileSrc, typename Scalar>
inline auto TADDS(TileDst& dst, TileSrc& src, Scalar s) {
    auto ret = pto::TADDS(dst, src, s);
    _cpu_sim_add_tile_cycles(dst);
    return ret;
}
template <typename TileDst, typename TileSrc, typename Scalar>
inline auto TMULS(TileDst& dst, TileSrc& src, Scalar s) {
    auto ret = pto::TMULS(dst, src, s);
    _cpu_sim_add_tile_cycles(dst);
    return ret;
}
template <typename TileDst, typename TileSrc, typename Scalar>
inline auto TDIVS(TileDst& dst, TileSrc& src, Scalar s) {
    auto ret = pto::TDIVS(dst, src, s);
    _cpu_sim_add_tile_cycles(dst);
    return ret;
}
template <typename TileDst, typename Scalar, typename TileSrc>
inline auto TDIVS(TileDst& dst, Scalar s, TileSrc& src) {
    auto ret = pto::TDIVS(dst, s, src);
    _cpu_sim_add_tile_cycles(dst);
    return ret;
}
template <typename TileDst, typename TileSrc, typename Scalar>
inline auto TSUBS(TileDst& dst, TileSrc& src, Scalar s) {
    auto ret = pto::TSUBS(dst, src, s);
    _cpu_sim_add_tile_cycles(dst);
    return ret;
}
template <typename TileDst, typename TileSrc, typename Scalar>
inline auto TMAXS(TileDst& dst, TileSrc& src, Scalar s) {
    auto ret = pto::TMAXS(dst, src, s);
    _cpu_sim_add_tile_cycles(dst);
    return ret;
}
template <typename TileDst, typename TileSrc, typename Scalar>
inline auto TMINS(TileDst& dst, TileSrc& src, Scalar s) {
    auto ret = pto::TMINS(dst, src, s);
    _cpu_sim_add_tile_cycles(dst);
    return ret;
}

template <typename TileDst, typename TileSrc>
inline auto TEXP(TileDst& dst, TileSrc& src) {
    auto ret = pto::TEXP(dst, src);
    _cpu_sim_add_tile_cycles(dst);
    return ret;
}
template <typename TileDst, typename TileSrc>
inline auto TRSQRT(TileDst& dst, TileSrc& src) {
    auto ret = pto::TRSQRT(dst, src);
    _cpu_sim_add_tile_cycles(dst);
    return ret;
}

template <typename TileDst, typename TileSrc, typename TileTmp>
inline auto TROWSUM(TileDst& dst, TileSrc& src, TileTmp& tmp) {
    auto ret = pto::TROWSUM(dst, src, tmp);
    _cpu_sim_add_tile_cycles(dst);
    return ret;
}
template <typename TileDst, typename TileSrc, typename TileTmp>
inline auto TROWMAX(TileDst& dst, TileSrc& src, TileTmp& tmp) {
    auto ret = pto::TROWMAX(dst, src, tmp);
    _cpu_sim_add_tile_cycles(dst);
    return ret;
}

template <typename TileDst, typename TileSrc, typename TileIdx>
inline auto TSORT32(TileDst& dst, TileSrc& src, TileIdx& idx) {
    auto ret = pto::TSORT32(dst, src, idx);
#if defined(__CPU_SIM)
    // CPU-sim accounting: treat TSORT32 as a vector op with a constant-factor
    // overhead per element. This keeps cycle reports non-zero and scaling with
    // tile size. (Strict vendor-accurate sort timing is out of scope for v9.)
    const std::uint64_t n =
        static_cast<std::uint64_t>(src.GetValidRow()) * static_cast<std::uint64_t>(src.GetValidCol());
    pto::cpu_sim::add_cycles(n * 5u);
#endif
    return ret;
}

template <typename TileDst, typename TileSrc, typename TileIdx, typename TileTmp>
inline auto TSORT32(TileDst& dst, TileSrc& src, TileIdx& idx, TileTmp& tmp) {
    auto ret = pto::TSORT32(dst, src, idx, tmp);
#if defined(__CPU_SIM)
    const std::uint64_t n =
        static_cast<std::uint64_t>(src.GetValidRow()) * static_cast<std::uint64_t>(src.GetValidCol());
    pto::cpu_sim::add_cycles(n * 5u);
#endif
    return ret;
}

}  // namespace pto_wsp::ptoisa
