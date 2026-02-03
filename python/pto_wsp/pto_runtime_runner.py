from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from types import ModuleType
from typing import Iterable

import numpy as np

from pto_wsp.pto_runtime_abi import build_orch_func_args
from pto_wsp import pto_runtime_bridge


_RUNTIME_BIN_CACHE: dict[str, tuple[bytes, bytes, bytes]] = {}


def _load_kernel_config(artifact_dir: Path) -> ModuleType:
    cfg = artifact_dir / "kernels" / "kernel_config.py"
    if not cfg.is_file():
        raise FileNotFoundError(f"pto-runtime kernel_config.py not found: {cfg}")

    mod_name = f"pto_wsp_codegen_kernel_config_{abs(hash(str(cfg)))}"
    spec = importlib.util.spec_from_file_location(mod_name, cfg)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load kernel config: {cfg}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def run_host_build_graph(
    *,
    artifact_dir: str,
    platform: str,
    arrays: Iterable[np.ndarray],
    device_id: int = 0,
    aicpu_thread_num: int = 1,
    block_dim: int = 1,
) -> None:
    """Build + run a pto-runtime host_build_graph artifact emitted by PTO-WSP."""
    if platform not in ("a2a3sim", "a2a3"):
        raise ValueError(f"Unsupported pto-runtime platform: {platform}")

    artifact_path = Path(artifact_dir).resolve()
    runtime_root = pto_runtime_bridge.pto_runtime_root()

    # Ensure pto-runtime kernel_config selects the right kernel sources.
    prev_platform = os.environ.get("PTO_RUNTIME_PLATFORM")
    os.environ["PTO_RUNTIME_PLATFORM"] = platform
    try:
        kernel_cfg = _load_kernel_config(artifact_path)
    finally:
        if prev_platform is None:
            os.environ.pop("PTO_RUNTIME_PLATFORM", None)
        else:
            os.environ["PTO_RUNTIME_PLATFORM"] = prev_platform

    runtime_builder = pto_runtime_bridge.import_runtime_builder()
    bindings = pto_runtime_bridge.import_bindings()
    elf_parser = pto_runtime_bridge.import_elf_parser()

    # Build runtime binaries (cached in-process).
    if platform not in _RUNTIME_BIN_CACHE:
        builder = runtime_builder.RuntimeBuilder(platform=platform, runtime_root=runtime_root)
        _RUNTIME_BIN_CACHE[platform] = builder.build("host_build_graph")

    host_binary, aicpu_binary, aicore_binary = _RUNTIME_BIN_CACHE[platform]

    # Bind runtime and compile orchestration/kernels.
    Runtime = bindings.bind_host_binary(host_binary)
    bindings.set_device(int(device_id))

    # Compile orchestration.
    builder = runtime_builder.RuntimeBuilder(platform=platform, runtime_root=runtime_root)
    pto_compiler = builder.get_pto_compiler()
    orch = kernel_cfg.ORCHESTRATION
    runtime_inc = runtime_root / "src" / "runtime" / "host_build_graph" / "runtime"
    orch_so_binary = pto_compiler.compile_orchestration(
        orch["source"],
        extra_include_dirs=[str(runtime_inc)] + list(pto_compiler.get_platform_include_dirs()),
    )

    # Compile and register kernels.
    kernels = list(kernel_cfg.KERNELS)
    pto_isa_root = None
    if platform == "a2a3":
        pto_isa_root = str((_repo_root() / "3rdparty" / "pto-isa").resolve())

    for k in kernels:
        core_type = k.get("core_type", "aiv")
        incore_o = pto_compiler.compile_incore(
            k["source"],
            core_type=core_type,
            pto_isa_root=pto_isa_root,
        )
        kernel_bin = elf_parser.extract_text_section(incore_o)
        bindings.register_kernel(int(k["func_id"]), kernel_bin)

    # Prepare orchestration args (ptr/nbytes pairs).
    func_args = build_orch_func_args(arrays)

    runtime = Runtime()
    runtime.initialize(orch_so_binary, orch["function_name"], func_args)
    bindings.launch_runtime(
        runtime,
        aicpu_thread_num=int(aicpu_thread_num),
        block_dim=int(block_dim),
        device_id=int(device_id),
        aicpu_binary=aicpu_binary,
        aicore_binary=aicore_binary,
    )
    runtime.finalize()
