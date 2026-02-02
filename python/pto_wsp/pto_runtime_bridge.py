from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from types import ModuleType


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def pto_runtime_root() -> Path:
    """Locate the pto-runtime repo root directory.

    Resolution order:
    1) `PTO_RUNTIME_PATH` (explicit override)
    2) repo submodule at `3rdparty/pto-runtime`
    """
    env = os.environ.get("PTO_RUNTIME_PATH")
    if env:
        return Path(env).expanduser().resolve()
    return (_repo_root() / "3rdparty" / "pto-runtime").resolve()


def pto_runtime_python_dir() -> Path:
    return pto_runtime_root() / "python"


def _import_from_dir(module_name: str, directory: Path) -> ModuleType:
    directory = directory.resolve()
    if not directory.is_dir():
        raise ImportError(
            f"pto-runtime python dir not found: {directory}. "
            "Set PTO_RUNTIME_PATH=/path/to/pto-runtime or init submodules via "
            "`git submodule update --init --recursive`."
        )

    directory_str = str(directory)
    added = False
    if directory_str not in sys.path:
        sys.path.insert(0, directory_str)
        added = True
    try:
        return importlib.import_module(module_name)
    finally:
        if added:
            try:
                sys.path.remove(directory_str)
            except ValueError:
                pass


def import_runtime_builder() -> ModuleType:
    """Import pto-runtime's `runtime_builder` module."""
    return _import_from_dir("runtime_builder", pto_runtime_python_dir())

