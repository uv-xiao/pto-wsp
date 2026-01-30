// Copyright 2026 PTO-WSP Authors
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>

#include "pto/wsp/codegen/abi/workload_abi.hpp"

// Minimal task-context ABI for codegen-first v9 artifacts.
//
// This struct is intentionally small: emitted code can pass a `TaskContext`
// to helper functions (or inlined ScalarExpr code) without depending on any
// Python-side runtime semantics.
//
// NOTE: In v9, ScalarExpr is typically compiled into C++ expressions by the
// codegen stage (instead of interpreted from a serialized tree).
extern "C" {

struct TaskContext {
    const uint64_t* args;
    uint32_t num_args;
    uint32_t num_axis_args;

    RuntimeContext* rt;
};

static inline uint64_t pto_wsp_task_param_u64(const TaskContext* t, uint32_t i) {
    if (!t || !t->args) return 0;
    return (i < t->num_args) ? t->args[i] : 0;
}

static inline int64_t pto_wsp_task_param_i64(const TaskContext* t, uint32_t i) {
    return (int64_t)pto_wsp_task_param_u64(t, i);
}

static inline uint64_t pto_wsp_symbol_u64(const TaskContext* t, uint64_t sym_id) {
    if (!t || !t->rt || !t->rt->get_symbol_u64) return 0;
    return t->rt->get_symbol_u64(t->rt->ctx, sym_id);
}

static inline uint64_t pto_wsp_slot_u64(const TaskContext* t, uint32_t slot) {
    if (!t || !t->rt || !t->rt->get_slot_u64) return 0;
    return t->rt->get_slot_u64(t->rt->ctx, slot);
}

// v9: tags are modeled as u64 entries addressed by stable IDs.
// Today we treat them as symbols keyed by `tag_id` (same stable 64-bit ID).
static inline uint64_t pto_wsp_task_tag_u64(const TaskContext* t, uint64_t tag_id) {
    return pto_wsp_symbol_u64(t, tag_id);
}

}  // extern "C"
