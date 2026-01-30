from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

from pto_wsp.builder import kernel as kernel_decorator
from pto_wsp.builder import extract_kernel_params
from pto_wsp.kernel import ScalarType, TileType
from pto_wsp.types import DType


# -----------------------------------------------------------------------------
# PTO-ISA kernel JIT (Python authoring -> C++ source emission)
#
# Goal: let users author kernels in Python using PTO-ISA instruction names,
# without adding new "primitive ops" to the PTO-WSP kernel IR. The output is
# emitted as a custom C++ kernel body (`cpp_src`) compiled into codegen-first
# artifacts via the existing custom-kernel facility.
# -----------------------------------------------------------------------------


def _dtype_to_str(dt: Any) -> str:
    v = getattr(dt, "value", str(dt))
    if v in ("f16", "f32", "i32", "i64", "u32", "u64", "bool"):
        return v
    raise ValueError(f"Unsupported dtype: {dt!r}")


def _dtype_to_cpp_pto_elem(dtype: str) -> str:
    if dtype == "f32":
        return "float"
    if dtype == "f16":
        return "half"
    if dtype == "i32":
        return "int32_t"
    if dtype == "i64":
        return "int64_t"
    if dtype == "u32":
        return "uint32_t"
    if dtype == "u64":
        return "uint64_t"
    if dtype == "bool":
        return "bool"
    raise ValueError(f"Unsupported dtype for PTO-ISA emission: {dtype}")


def _dtype_size_bytes(dtype: str) -> int:
    if dtype == "f16":
        return 2
    if dtype in ("f32", "i32", "u32"):
        return 4
    if dtype in ("i64", "u64"):
        return 8
    if dtype == "bool":
        return 1
    raise ValueError(f"Unsupported dtype size: {dtype}")


def _round_up(n: int, multiple: int) -> int:
    return ((n + multiple - 1) // multiple) * multiple


def _vec_col_multiple(dtype: str) -> int:
    # Match the C++ codegen rule: Vec tile requires Cols * sizeof(DType) to be 32B aligned.
    return 32 // _dtype_size_bytes(dtype)


@dataclass(frozen=True)
class _TileSpec:
    rows: int
    cols: int
    dtype: str
    kind: str = "vec"  # "vec" | "acc"


class _Val:
    __slots__ = ("id", "spec")

    def __init__(self, vid: int, spec: _TileSpec):
        self.id = int(vid)
        self.spec = spec


class _ParamTile:
    __slots__ = ("name", "spec", "tensor_param_index")

    def __init__(self, name: str, spec: _TileSpec, tensor_param_index: int):
        self.name = str(name)
        self.spec = spec
        self.tensor_param_index = int(tensor_param_index)


class _ParamScalar:
    __slots__ = ("name", "dtype", "scalar_param_index")

    def __init__(self, name: str, dtype: str, scalar_param_index: int):
        self.name = str(name)
        self.dtype = str(dtype)
        self.scalar_param_index = int(scalar_param_index)


@dataclass(frozen=True)
class _Op:
    kind: str
    out: Optional[int]
    args: tuple[Any, ...]
    attrs: dict[str, Any]


class _Trace:
    def __init__(self) -> None:
        self.ops: list[_Op] = []
        self._next_id = 1_000_000  # avoid collisions with param ids

    def new_val(self, spec: _TileSpec) -> _Val:
        vid = self._next_id
        self._next_id += 1
        return _Val(vid, spec)

    def add(self, op: _Op) -> None:
        self.ops.append(op)


class _PtoisaNamespace:
    """Python authoring surface for PTO-ISA kernels (v9).

    This API is tracing-only: it is meant to run inside `@ptoisa_kernel` to
    produce a C++ kernel body.
    """

    def __init__(self) -> None:
        self._trace: Optional[_Trace] = None
        self._unary_ops = {
            "TEXP",
            "TRSQRT",
        }
        self._binary_ops = {
            "TADD",
            "TSUB",
            "TMUL",
            "TDIV",
            "TMAX",
            "TMIN",
            "TMOV",  # treated as unary in the tracer, but keep for emission guard
            "TROWEXPANDADD",
            "TROWEXPANDSUB",
            "TROWEXPANDMUL",
            "TROWEXPANDDIV",
            "TROWEXPANDMAX",
            "TROWEXPANDMIN",
            "TCOLEXPANDSUB",
            "TCOLEXPANDMUL",
            "TCOLEXPANDDIV",
        }
        self._scalar_ops = {
            "TADDS",
            "TMULS",
            "TDIVS",
            "TSUBS",
            "TMAXS",
            "TMINS",
        }

    def _require_trace(self) -> _Trace:
        if self._trace is None:
            raise RuntimeError("ptoisa ops can only be used inside @ptoisa_kernel during tracing")
        return self._trace

    # --- memory ops ---
    def tload(self, src: _ParamTile) -> _Val:
        tr = self._require_trace()
        if not isinstance(src, _ParamTile):
            raise TypeError("ptoisa.tload expects a kernel param tile (In[Tile[...]])")
        out = tr.new_val(src.spec)
        tr.add(_Op("TLOAD", out.id, (src,), {}))
        return out

    def tstore(self, dst: _ParamTile, src: _Val) -> None:
        tr = self._require_trace()
        if not isinstance(dst, _ParamTile):
            raise TypeError("ptoisa.tstore expects a kernel param tile (Out[Tile[...]])")
        if not isinstance(src, _Val):
            raise TypeError("ptoisa.tstore expects a tile value")
        tr.add(_Op("TSTORE", None, (dst, src), {}))

    # --- helpers (still emitted as C++ in custom kernel bodies) ---
    def iota_u32(self, cols: int) -> _Val:
        tr = self._require_trace()
        c = int(cols)
        if c <= 0:
            raise ValueError("iota_u32: cols must be > 0")
        spec = _TileSpec(rows=1, cols=c, dtype="u32", kind="vec")
        out = tr.new_val(spec)
        tr.add(_Op("IOTA_U32", out.id, (), {"cols": c}))
        return out

    def store_topk_indices_i32(self, dst: _ParamTile, pairs: _Val, k: int) -> None:
        tr = self._require_trace()
        if not isinstance(dst, _ParamTile):
            raise TypeError("store_topk_indices_i32 expects an output param tile")
        if not isinstance(pairs, _Val):
            raise TypeError("store_topk_indices_i32 expects a tile value")
        kk = int(k)
        tr.add(_Op("STORE_TOPK_INDICES_I32", None, (dst, pairs), {"k": kk}))

    def cpu_sim_add_cycles(self, cycles: Any) -> None:
        tr = self._require_trace()
        tr.add(_Op("CPU_SIM_ADD_CYCLES", None, (cycles,), {}))

    # --- instruction-like ops (wrapper calls to pto_wsp::ptoisa::<NAME>) ---
    def _bin(self, name: str, a: _Val, b: _Val) -> _Val:
        tr = self._require_trace()
        if not isinstance(a, _Val) or not isinstance(b, _Val):
            raise TypeError(f"{name}: expects tile values")
        if a.spec.rows != b.spec.rows or a.spec.cols != b.spec.cols or a.spec.dtype != b.spec.dtype:
            raise ValueError(f"{name}: shape/dtype mismatch: {a.spec} vs {b.spec}")
        out = tr.new_val(a.spec)
        tr.add(_Op(name, out.id, (a, b), {}))
        return out

    def TADD(self, a: _Val, b: _Val) -> _Val:  # noqa: N802
        return self._bin("TADD", a, b)

    def TSUB(self, a: _Val, b: _Val) -> _Val:  # noqa: N802
        return self._bin("TSUB", a, b)

    def TMUL(self, a: _Val, b: _Val) -> _Val:  # noqa: N802
        return self._bin("TMUL", a, b)

    def TDIV(self, a: _Val, b: _Val) -> _Val:  # noqa: N802
        return self._bin("TDIV", a, b)

    def TMAX(self, a: _Val, b: _Val) -> _Val:  # noqa: N802
        return self._bin("TMAX", a, b)

    def TMIN(self, a: _Val, b: _Val) -> _Val:  # noqa: N802
        return self._bin("TMIN", a, b)

    def TMOV(self, a: _Val) -> _Val:  # noqa: N802
        tr = self._require_trace()
        if not isinstance(a, _Val):
            raise TypeError("TMOV expects a tile value")
        out = tr.new_val(a.spec)
        tr.add(_Op("TMOV", out.id, (a,), {}))
        return out

    def TSORT32(self, src: _Val, idx: _Val) -> _Val:  # noqa: N802
        tr = self._require_trace()
        if not isinstance(src, _Val) or not isinstance(idx, _Val):
            raise TypeError("TSORT32 expects tile values")
        if src.spec.rows != 1 or src.spec.cols != 32:
            raise ValueError("TSORT32 v9: requires src tile shape [1,32]")
        if idx.spec.rows != 1 or idx.spec.cols != 32 or idx.spec.dtype != "u32":
            raise ValueError("TSORT32 v9: requires idx tile shape [1,32] dtype u32")
        if src.spec.dtype != "f32":
            raise ValueError("TSORT32 v9: requires src dtype f32")
        out = tr.new_val(_TileSpec(rows=1, cols=64, dtype="f32", kind="vec"))
        tr.add(_Op("TSORT32", out.id, (src, idx), {}))
        return out

    def _unary(self, name: str, a: _Val) -> _Val:
        tr = self._require_trace()
        if not isinstance(a, _Val):
            raise TypeError(f"{name}: expects a tile value")
        out = tr.new_val(a.spec)
        tr.add(_Op(name, out.id, (a,), {}))
        return out

    def _scalar(self, name: str, a: _Val, s: Any) -> _Val:
        tr = self._require_trace()
        if not isinstance(a, _Val):
            raise TypeError(f"{name}: expects a tile value")
        out = tr.new_val(a.spec)
        tr.add(_Op(name, out.id, (a, s), {}))
        return out

    def __getattr__(self, name: str) -> Any:
        # Allow `ptoisa.TEXP(x)` style for a growing instruction surface without
        # hand-authoring every wrapper.
        if name in self._unary_ops:
            return lambda a: self._unary(name, a)
        if name in self._scalar_ops:
            return lambda a, s: self._scalar(name, a, s)
        if name in self._binary_ops and name not in ("TMOV",):
            return lambda a, b: self._bin(name, a, b)
        raise AttributeError(name)


ptoisa = _PtoisaNamespace()


def _emit_cpp_body(trace: _Trace, *, params_tiles: list[_ParamTile], params_scalars: list[_ParamScalar]) -> str:
    # Preamble: extract tensor ptrs/strides and tail valid dims (same contract as normal kernel emission).
    lines: list[str] = []

    # Tail args: per tensor param (valid_row, valid_col) pairs, appended to axis args.
    num_tensor_params = len(params_tiles)
    if num_tensor_params:
        lines.append(f"  uint32_t _pto_wsp_tail_base = 0;")
        lines.append(f"  if (task->num_axis_args >= {num_tensor_params * 2}) {{")
        lines.append(f"    _pto_wsp_tail_base = task->num_axis_args - {num_tensor_params * 2};")
        lines.append(f"  }}")
        for p in params_tiles:
            lines.append(f"  int {p.name}_vr = {p.spec.rows};")
            lines.append(f"  int {p.name}_vc = {p.spec.cols};")
        lines.append(f"  if (task->num_axis_args >= {num_tensor_params * 2}) {{")
        for p in params_tiles:
            bi = p.tensor_param_index * 2
            lines.append(f"    {p.name}_vr = (int)task->args[_pto_wsp_tail_base + {bi}];")
            lines.append(f"    {p.name}_vc = (int)task->args[_pto_wsp_tail_base + {bi + 1}];")
        lines.append(f"  }}")
        lines.append("")

    # Scalar params decode.
    for s in params_scalars:
        if s.dtype == "i32":
            lines.append(f"  int32_t {s.name} = (int32_t)task->args[task->num_axis_args + {s.scalar_param_index}];")
        elif s.dtype == "u32":
            lines.append(f"  uint32_t {s.name} = (uint32_t)task->args[task->num_axis_args + {s.scalar_param_index}];")
        elif s.dtype == "i64":
            lines.append(f"  int64_t {s.name} = (int64_t)task->args[task->num_axis_args + {s.scalar_param_index}];")
        elif s.dtype == "u64":
            lines.append(f"  uint64_t {s.name} = (uint64_t)task->args[task->num_axis_args + {s.scalar_param_index}];")
        elif s.dtype == "bool":
            lines.append(f"  bool {s.name} = task->args[task->num_axis_args + {s.scalar_param_index}] != 0;")
        elif s.dtype == "f32":
            lines.append(f"  union {{ uint32_t u32; float f32; }} _u_{s.name};")
            lines.append(f"  _u_{s.name}.u32 = (uint32_t)task->args[task->num_axis_args + {s.scalar_param_index}];")
            lines.append(f"  float {s.name} = _u_{s.name}.f32;")
        else:
            raise ValueError(f"Unsupported scalar dtype in ptoisa_kernel: {s.dtype}")
    if params_scalars:
        lines.append("")

    # Tensor param pointer extraction.
    for p in params_tiles:
        elem = _dtype_to_cpp_pto_elem(p.spec.dtype)
        lines.append(f"  {elem}* {p.name} = ({elem}*)task->tensor_ptrs[{p.tensor_param_index}];")
        lines.append(f"  uint64_t {p.name}_s3 = task->tensor_strides[{p.tensor_param_index * 2}];")
        lines.append(f"  uint64_t {p.name}_s4 = task->tensor_strides[{p.tensor_param_index * 2 + 1}];")
    if params_tiles:
        lines.append("")

    # Build a stable out->spec map and runtime valid-dim expressions.
    out_spec: dict[int, _TileSpec] = {}
    out_dims: dict[int, tuple[str, str]] = {}
    for op in trace.ops:
        if op.out is None:
            continue
        if op.kind == "TLOAD":
            src = op.args[0]
            if isinstance(src, _ParamTile):
                out_spec[op.out] = src.spec
                out_dims[op.out] = (f"{src.name}_vr", f"{src.name}_vc")
        elif op.kind == "IOTA_U32":
            out_spec[op.out] = _TileSpec(rows=1, cols=int(op.attrs["cols"]), dtype="u32", kind="vec")
            out_dims[op.out] = ("1", str(int(op.attrs["cols"])))
        elif op.kind == "TSORT32":
            out_spec[op.out] = _TileSpec(rows=1, cols=64, dtype="f32", kind="vec")
            out_dims[op.out] = ("1", "64")
        elif op.kind in ("TADD", "TSUB", "TMUL", "TDIV", "TMAX", "TMIN", "TMOV"):
            a0 = op.args[0]
            if isinstance(a0, _Val):
                out_spec[op.out] = a0.spec
                out_dims[op.out] = (f"v{a0.id}->GetValidRow()", f"v{a0.id}->GetValidCol()")
        else:
            # Default: treat as vec same as first arg (common elementwise ops).
            a0 = op.args[0] if op.args else None
            if isinstance(a0, _Val):
                out_spec[op.out] = a0.spec
                out_dims[op.out] = (f"v{a0.id}->GetValidRow()", f"v{a0.id}->GetValidCol()")

    # Emit local tile allocations.
    for vid, spec in out_spec.items():
        elem = _dtype_to_cpp_pto_elem(spec.dtype)
        fr = spec.rows
        fc = spec.cols
        r_expr, c_expr = out_dims.get(vid, (str(fr), str(fc)))
        if spec.kind == "acc":
            fr_full = _round_up(fr, 16)
            fc_full = _round_up(fc, 16)
            lines.append(
                f"  auto v{vid} = std::make_unique<pto::TileAcc<{elem}, {fr_full}, {fc_full}, pto::DYNAMIC, pto::DYNAMIC>>((size_t)({r_expr}), (size_t)({c_expr}));"
            )
        else:
            fc_full = _round_up(fc, _vec_col_multiple(spec.dtype))
            lines.append(
                f"  auto v{vid} = std::make_unique<pto::Tile<pto::TileType::Vec, {elem}, {fr}, {fc_full}, pto::BLayout::RowMajor, pto::DYNAMIC, pto::DYNAMIC>>((size_t)({r_expr}), (size_t)({c_expr}));"
            )
    if out_spec:
        lines.append("")

    # Emit ops.
    def tile_expr(a: Any) -> str:
        if isinstance(a, _Val):
            return f"*v{a.id}"
        raise TypeError(f"Expected tile value, got: {a!r}")

    def scalar_expr(a: Any) -> str:
        if isinstance(a, _ParamScalar):
            return a.name
        if isinstance(a, int):
            return str(int(a))
        if isinstance(a, str):
            return a
        return str(a)

    for op in trace.ops:
        if op.kind == "TLOAD":
            src = op.args[0]
            if not isinstance(src, _ParamTile):
                raise TypeError("TLOAD expects a param tile")
            elem = _dtype_to_cpp_pto_elem(src.spec.dtype)
            pid = src.tensor_param_index
            lines.append(f"  using _GTShape_load_{pid} = pto::Shape<1, 1, 1, pto::DYNAMIC, pto::DYNAMIC>;")
            lines.append(f"  using _GTStride_load_{pid} = pto::Stride<1, 1, 1, pto::DYNAMIC, pto::DYNAMIC>;")
            lines.append(
                f"  pto::GlobalTensor<{elem}, _GTShape_load_{pid}, _GTStride_load_{pid}, pto::Layout::ND> g_load_{pid}("
                f"{src.name}, _GTShape_load_{pid}({src.name}_vr, {src.name}_vc), _GTStride_load_{pid}((int){src.name}_s3, (int){src.name}_s4));"
            )
            lines.append(f"  pto_wsp::ptoisa::TLOAD(*v{op.out}, g_load_{pid});")
            lines.append("")
            continue

        if op.kind == "TSTORE":
            dst, srcv = op.args
            if not isinstance(dst, _ParamTile) or not isinstance(srcv, _Val):
                raise TypeError("TSTORE expects (param_tile, tile_val)")
            elem = _dtype_to_cpp_pto_elem(dst.spec.dtype)
            pid = dst.tensor_param_index
            lines.append(f"  using _GTShape_store_{pid} = pto::Shape<1, 1, 1, pto::DYNAMIC, pto::DYNAMIC>;")
            lines.append(f"  using _GTStride_store_{pid} = pto::Stride<1, 1, 1, pto::DYNAMIC, pto::DYNAMIC>;")
            lines.append(
                f"  pto::GlobalTensor<{elem}, _GTShape_store_{pid}, _GTStride_store_{pid}, pto::Layout::ND> g_store_{pid}("
                f"{dst.name}, _GTShape_store_{pid}({dst.name}_vr, {dst.name}_vc), _GTStride_store_{pid}((int){dst.name}_s3, (int){dst.name}_s4));"
            )
            lines.append(f"  pto_wsp::ptoisa::TSTORE(g_store_{pid}, {tile_expr(srcv)});")
            lines.append("")
            continue

        if op.kind == "IOTA_U32":
            cols = int(op.attrs["cols"])
            lines.append(f"  for (int _i = 0; _i < {cols}; ++_i) {{ v{op.out}->data()[_i] = (uint32_t)_i; }}")
            continue

        if op.kind == "STORE_TOPK_INDICES_I32":
            dst, pairs = op.args
            kk = int(op.attrs["k"])
            if not isinstance(dst, _ParamTile) or not isinstance(pairs, _Val):
                raise TypeError("STORE_TOPK_INDICES_I32 expects (param_tile, pairs_tile)")
            lines.append(f"  // store_topk_indices_i32(k={kk})")
            lines.append(f"  for (int _i = 0; _i < {kk}; ++_i) {{")
            lines.append(f"    const float _idx_f = v{pairs.id}->data()[2 * _i + 1];")
            lines.append(f"    const int32_t _idx_i = (int32_t)_idx_f;")
            lines.append(f"    {dst.name}[_i * (int64_t){dst.name}_s4] = _idx_i;")
            lines.append(f"  }}")
            continue

        if op.kind == "CPU_SIM_ADD_CYCLES":
            (c,) = op.args
            lines.append("#if defined(__CPU_SIM)")
            lines.append(f"  pto::cpu_sim::add_cycles((uint64_t)({scalar_expr(c)}));")
            lines.append("#endif")
            continue

        # Default: wrapper call, using functional SSA style.
        if op.kind in ("TADD", "TSUB", "TMUL", "TDIV", "TMAX", "TMIN"):
            a, b = op.args
            lines.append(f"  pto_wsp::ptoisa::{op.kind}(*v{op.out}, {tile_expr(a)}, {tile_expr(b)});")
            continue
        if op.kind == "TMOV":
            (a,) = op.args
            lines.append(f"  pto_wsp::ptoisa::TMOV(*v{op.out}, {tile_expr(a)});")
            continue
        if op.kind == "TSORT32":
            src, idx = op.args
            lines.append(f"  pto_wsp::ptoisa::TSORT32(*v{op.out}, {tile_expr(src)}, {tile_expr(idx)});")
            continue

        # Generic unary ops: pto_wsp::ptoisa::<OP>(dst, src)
        if op.kind in ("TEXP", "TRSQRT"):
            (a,) = op.args
            if not isinstance(a, _Val):
                raise TypeError(f"{op.kind}: expects tile arg")
            lines.append(f"  pto_wsp::ptoisa::{op.kind}(*v{op.out}, {tile_expr(a)});")
            continue

        # Generic scalar ops: pto_wsp::ptoisa::<OP>(dst, src, scalar)
        if op.kind in ("TADDS", "TMULS", "TDIVS", "TSUBS", "TMAXS", "TMINS"):
            a, s = op.args
            if not isinstance(a, _Val):
                raise TypeError(f"{op.kind}: expects tile arg")
            elem = _dtype_to_cpp_pto_elem(a.spec.dtype)
            lines.append(f"  pto_wsp::ptoisa::{op.kind}(*v{op.out}, {tile_expr(a)}, ({elem})({scalar_expr(s)}));")
            continue

        # Generic binary tile ops: pto_wsp::ptoisa::<OP>(dst, a, b)
        if op.kind.startswith("TROWEXPAND") or op.kind.startswith("TCOLEXPAND"):
            a, b = op.args
            if not isinstance(a, _Val) or not isinstance(b, _Val):
                raise TypeError(f"{op.kind}: expects tile args")
            lines.append(f"  pto_wsp::ptoisa::{op.kind}(*v{op.out}, {tile_expr(a)}, {tile_expr(b)});")
            continue

        raise ValueError(f"Unsupported ptoisa op: {op.kind}")

    return "\n".join(lines).rstrip() + "\n"


def ptoisa_kernel(func: Callable | None = None, *, cpp_includes: Optional[list[str]] = None) -> Any:
    """Define a kernel authored with PTO-ISA instruction calls in Python.

    The decorated function is executed at decoration time in tracing mode to
    produce a custom C++ kernel body (`cpp_src`) that calls PTO-ISA wrappers.

    This stays within v9 Path A: it does not extend PTO-WSP primitive ops; it
    emits user kernel code compiled into the artifact.
    """

    def decorator(fn: Callable) -> Any:
        params = extract_kernel_params(fn)

        # Determine tensor/scalar param indices in signature order.
        tensor_params: list[_ParamTile] = []
        scalar_params: list[_ParamScalar] = []
        ti = 0
        si = 0

        call_kwargs: dict[str, Any] = {}
        for p in params:
            if isinstance(p.inner_type, TileType) and p.shape is not None:
                r, c = p.shape
                spec = _TileSpec(rows=int(r), cols=int(c), dtype=_dtype_to_str(p.dtype or DType.F16))
                call_kwargs[p.name] = _ParamTile(p.name, spec, ti)
                tensor_params.append(call_kwargs[p.name])
                ti += 1
            elif isinstance(p.inner_type, ScalarType):
                dtype = _dtype_to_str(p.dtype or DType.F32)
                call_kwargs[p.name] = _ParamScalar(p.name, dtype, si)
                scalar_params.append(call_kwargs[p.name])
                si += 1
            else:
                raise TypeError(
                    f"ptoisa_kernel only supports Tile[...] and Scalar[...] params (got {p.name}: {p.inner_type!r})"
                )

        tr = _Trace()
        old = ptoisa._trace
        try:
            ptoisa._trace = tr
            fn(**call_kwargs)
        finally:
            ptoisa._trace = old

        body = _emit_cpp_body(tr, params_tiles=tensor_params, params_scalars=scalar_params)
        incs = list(cpp_includes) if cpp_includes else None
        return kernel_decorator(fn, cpp_src=body, cpp_includes=incs)

    return decorator(func) if func is not None else decorator
