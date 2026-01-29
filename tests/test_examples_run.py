from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


_ROOT = Path(__file__).resolve().parents[1]
_EXAMPLES = _ROOT / "examples"
_PYTHON = _ROOT / "python"


def _example_scripts() -> list[Path]:
    out: list[Path] = []
    for p in sorted(_EXAMPLES.glob("*/*_example.py")):
        if not p.is_file():
            continue
        out.append(p)
    return out


@pytest.mark.parametrize(
    "example",
    _example_scripts(),
    ids=lambda p: str(Path(p).relative_to(_ROOT)),
)
def test_examples_run(example: Path) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(_PYTHON) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )

    res = subprocess.run(
        [sys.executable, str(example)],
        cwd=str(_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    if res.returncode != 0:
        out = (res.stdout or "").splitlines()[-80:]
        err = (res.stderr or "").splitlines()[-80:]
        raise AssertionError(
            f"example failed: {example}\n"
            f"exit={res.returncode}\n"
            f"--- stdout (tail) ---\n" + "\n".join(out) + "\n"
            f"--- stderr (tail) ---\n" + "\n".join(err) + "\n"
        )

