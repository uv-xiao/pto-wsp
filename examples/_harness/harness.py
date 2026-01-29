from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np


def seed_all(seed: int) -> None:
    np.random.seed(int(seed))


def max_abs_err(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b)))


def mean_abs_err(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))


def assert_allclose(
    got: np.ndarray,
    ref: np.ndarray,
    *,
    rtol: float,
    atol: float,
    name: str = "value",
) -> None:
    if got.shape != ref.shape:
        raise AssertionError(f"{name}: shape mismatch got={got.shape} ref={ref.shape}")
    if not np.allclose(got, ref, rtol=rtol, atol=atol):
        mx = max_abs_err(got, ref)
        mn = mean_abs_err(got, ref)
        raise AssertionError(f"{name}: not close (rtol={rtol} atol={atol}) max_abs={mx:.3e} mean_abs={mn:.3e}")


@dataclass(frozen=True)
class CycleCheck:
    expected: Optional[int] = None
    rel_tol: float = 0.10
    abs_tol: int = 0
    min_cycles: int = 1


def assert_cycles(total_cycles: int, check: CycleCheck, *, name: str = "total_cycles") -> None:
    c = int(total_cycles)
    if c < int(check.min_cycles):
        raise AssertionError(f"{name}: expected >= {check.min_cycles}, got {c}")
    if check.expected is None:
        return
    exp = int(check.expected)
    lo = exp - int(check.abs_tol)
    hi = exp + int(check.abs_tol)
    lo = max(lo, int(exp * (1.0 - float(check.rel_tol))))
    hi = max(hi, int(exp * (1.0 + float(check.rel_tol))))
    if not (lo <= c <= hi):
        raise AssertionError(f"{name}: expected ~{exp} (range [{lo},{hi}]), got {c}")


def program_total_cycles(program: Any) -> int:
    stats = program.stats() if callable(program.stats) else program.stats
    return int(getattr(stats, "total_cycles", 0) or 0)


def run_example(
    name: str,
    *,
    run_pto: Callable[[], tuple[np.ndarray, int]],
    run_golden: Callable[[], np.ndarray],
    rtol: float,
    atol: float,
    cycles: CycleCheck,
) -> None:
    ref = run_golden()
    got, total_cycles = run_pto()

    assert_allclose(got, ref, rtol=rtol, atol=atol, name=f"{name}:output")
    assert_cycles(total_cycles, cycles, name=f"{name}:cycles")

    print(f"{name}: PASS")
    print(f"{name}: total_cycles={total_cycles}")

