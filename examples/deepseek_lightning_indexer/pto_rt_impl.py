from pathlib import Path
import numpy as np

from pto_wsp import (
    kernel,
    pto,
    In,
    Out,
    Tile,
    Scalar,
    workload,
    P,
    Dense,
    Tensor,
    DType,
    DispatchPolicy,
    TaskWindow,
    WindowMode,
    slot_load_u64,
    ptoisa,
    ptoisa_kernel,
)
from pto_wsp.scalar_expr import slot_u64

F32 = DType.F32
I32 = DType.I32
U64 = DType.U64


def run(scores: np.ndarray, eff_seq: np.ndarray, *, k: int = 8, impl: str = "ptoisa") -> tuple[np.ndarray, int]:
    """Run a tiered TopK indexer using Path A custom PTO-ISA kernels.

    - Tier selection predicate is *data-driven* via slot_load_u64(tensor)->slot_u64(i).
    - TopK is implemented as a custom kernel using PTO-ISA TSORT32 (no pto.topk primitive).
    """
    if scores.dtype != np.float32:
        raise ValueError("scores must be float32")
    if eff_seq.dtype != np.uint64:
        raise ValueError("eff_seq must be uint64")
    if eff_seq.shape[-2:] != (1, 1):
        raise ValueError("eff_seq must be shaped [...,1,1]")

    bsz, tiles, cand = scores.shape
    if cand != 32:
        raise ValueError("This example fixes candidates=32 to use PTO-ISA TSORT32.")
    if k != 8:
        raise ValueError("This example fixes k=8 for simplicity.")

    out_idx = np.zeros((bsz, tiles, 1, k), dtype=np.int32)

    Scores = Tensor(data=scores.reshape(bsz, tiles, 1, cand), shape=(bsz, tiles, 1, cand), dtype=F32)
    EffSeq = Tensor(data=eff_seq, shape=eff_seq.shape, dtype=U64)
    OutIdx = Tensor(data=out_idx, shape=out_idx.shape, dtype=I32)

    batch = Dense[bsz]()
    stiles = Dense[tiles]()

    # ------------------------------------------------------------------
    # TopK kernel implementations (select via `impl`)
    #   - "pto": pto.* traced IR lowered to PTO-ISA (TSORT32 + extract helper ops)
    #   - "ptoisa": Python-authored PTO-ISA instruction trace (ptoisa.*)
    #   - "cppfile": manual PTO-ISA C++ body stored in a file (cpp_body_path)
    # ------------------------------------------------------------------

    @kernel
    def topk8_sort32_pto(
        scores_tile: In[Tile[1, 32, F32]],
        out_idx_tile: Out[Tile[1, 8, I32]],
        pad_to: Scalar[I32],
    ):
        _ = pad_to  # v9: tier overhead is modeled only in ptoisa/cppfile implementations
        idx = pto.iota_u32(32)
        pairs = pto.tsort32(scores_tile, idx)
        out = pto.extract_topk_indices(pairs, 8)
        pto.store(out_idx_tile, out)

    @ptoisa_kernel
    def topk8_sort32(
        scores_tile: In[Tile[1, 32, F32]],
        out_idx_tile: Out[Tile[1, 8, I32]],
        pad_to: Scalar[I32],
    ):
        scores = ptoisa.tload(scores_tile)
        idx = ptoisa.iota_u32(32)
        pairs = ptoisa.TSORT32(scores, idx)
        ptoisa.store_topk_indices_i32(out_idx_tile, pairs, 8)

        # Model tier-dependent overhead as in-kernel cycles (still part of the
        # kernel cycle report; not a separate estimator).
        extra_expr = f"(({pad_to.name} > 32) ? ((uint64_t)({pad_to.name} / 32) * 128ull) : 0ull)"
        ptoisa.cpu_sim_add_cycles(extra_expr)

    _HERE = Path(__file__).resolve().parent

    @kernel(cpp_body_path=str(_HERE / "topk8_sort32_body.cpp"))
    def topk8_sort32_cppfile(
        scores_tile: In[Tile[1, 32, F32]],
        out_idx_tile: Out[Tile[1, 8, I32]],
        pad_to: Scalar[I32],
    ):
        _ = scores_tile
        _ = out_idx_tile
        _ = pad_to
        raise RuntimeError("cpp_body_path kernel should not execute in Python")

    impl = str(impl).lower().strip()
    if impl not in ("pto", "ptoisa", "cppfile"):
        raise ValueError("impl must be one of: pto | ptoisa | cppfile")
    if impl == "pto":
        topk_kernel = topk8_sort32_pto
    elif impl == "ptoisa":
        topk_kernel = topk8_sort32
    else:
        topk_kernel = topk8_sort32_cppfile

    @workload
    def lightning_indexer():
        for b in P(batch):
            for s in P(stiles):
                slot_load_u64(0, EffSeq[b][s], row=0, col=0)

                with P.when(slot_u64(0) <= 2048):
                    topk_kernel[b, s](scores_tile=Scores[b][s], out_idx_tile=OutIdx[b][s], pad_to=2048)
                with P.otherwise():
                    with P.when(slot_u64(0) <= 8192):
                        topk_kernel[b, s](scores_tile=Scores[b][s], out_idx_tile=OutIdx[b][s], pad_to=8192)
                    with P.otherwise():
                        with P.when(slot_u64(0) <= 65536):
                            topk_kernel[b, s](scores_tile=Scores[b][s], out_idx_tile=OutIdx[b][s], pad_to=65536)
                        with P.otherwise():
                            topk_kernel[b, s](scores_tile=Scores[b][s], out_idx_tile=OutIdx[b][s], pad_to=131072)

    program = (
        lightning_indexer()
        .dispatch(DispatchPolicy.affinity(lambda t: t.get("b"), num_aicpus=4))
        .task_graph(window=TaskWindow(1024, "tasks", WindowMode.STALL))
        .compile(target="cpu_sim")
    )

    program.execute()
    program.synchronize()

    stats = program.stats() if callable(program.stats) else program.stats
    total_cycles = int(getattr(stats, "total_cycles", 0) or 0)
    return out_idx.reshape(bsz, tiles, k).copy(), total_cycles
