#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AscendServerConfig:
    ssh_host: str
    workspace: str
    ascend_root: str
    ascend_env_script: str | None
    repo_dirname: str

    @property
    def remote_repo_dir(self) -> str:
        return f"{self.workspace.rstrip('/')}/{self.repo_dirname}"


def _parse_toml_value(raw: str) -> Any:
    v = raw.strip()
    if not v:
        return ""
    if v.startswith('"') and v.endswith('"') and len(v) >= 2:
        return v[1:-1]
    if v.startswith("'") and v.endswith("'") and len(v) >= 2:
        return v[1:-1]
    if v.lower() in ("true", "false"):
        return v.lower() == "true"
    try:
        return int(v, 10)
    except Exception:
        return v


def _load_simple_toml(path: Path) -> dict[str, dict[str, Any]]:
    """
    Minimal TOML loader for the repo-local `.ASCEND_SERVER.toml`.

    Supported syntax:
    - sections: [section]
    - key = "string" | 'string' | int | true/false
    - comments: # ...
    """
    current: str | None = None
    out: dict[str, dict[str, Any]] = {}
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        body = line.split("#", 1)[0].strip()
        if not body:
            continue
        if body.startswith("[") and body.endswith("]"):
            current = body[1:-1].strip()
            if not current:
                raise ValueError(f"{path}:{lineno}: empty section name")
            out.setdefault(current, {})
            continue
        if current is None:
            raise ValueError(f"{path}:{lineno}: key/value outside any [section]")
        key, sep, raw_val = body.partition("=")
        if sep != "=":
            raise ValueError(f"{path}:{lineno}: expected key = value")
        k = key.strip()
        if not k:
            raise ValueError(f"{path}:{lineno}: empty key")
        out[current][k] = _parse_toml_value(raw_val)
    return out


def load_config(config_path: Path) -> AscendServerConfig:
    if not config_path.exists():
        example = Path(".ASCEND_SERVER.toml.example")
        hint = f"Create `{config_path}` (start from `{example}`)." if example.exists() else f"Create `{config_path}`."
        raise FileNotFoundError(hint)

    cfg = _load_simple_toml(config_path)

    ssh_host = str(cfg.get("ssh", {}).get("host", "")).strip()
    workspace = str(cfg.get("ssh", {}).get("workspace", "")).strip()
    ascend_root = str(cfg.get("ascend", {}).get("root", "")).strip()
    ascend_env_script_raw = cfg.get("ascend", {}).get("env_script", None)
    ascend_env_script = str(ascend_env_script_raw).strip() if ascend_env_script_raw else None
    repo_dirname = str(cfg.get("repo", {}).get("dir", "pto-wsp")).strip() or "pto-wsp"

    missing = []
    if not ssh_host:
        missing.append("[ssh].host")
    if not workspace:
        missing.append("[ssh].workspace")
    if not ascend_root:
        missing.append("[ascend].root")
    if missing:
        raise ValueError(f"{config_path}: missing required keys: {', '.join(missing)}")

    return AscendServerConfig(
        ssh_host=ssh_host,
        workspace=workspace,
        ascend_root=ascend_root,
        ascend_env_script=ascend_env_script,
        repo_dirname=repo_dirname,
    )


def run_local(argv: list[str]) -> None:
    subprocess.run(argv, check=True)


def remote_bash(cfg: AscendServerConfig, cmd: str, *, cwd: str | None = None, tty: bool = False) -> None:
    parts = ["set -euo pipefail"]
    if cfg.ascend_env_script:
        parts.append(f"source {shlex.quote(cfg.ascend_env_script)} >/dev/null 2>&1 || true")
    if cwd:
        parts.append(f"cd {shlex.quote(cwd)}")
    parts.append(cmd)
    full = " ; ".join(parts)
    ssh = ["ssh"]
    if tty:
        ssh.append("-t")
    ssh.extend([cfg.ssh_host, "bash", "-lc", full])
    run_local(ssh)


def rsync_repo(cfg: AscendServerConfig, *, delete: bool = True) -> None:
    remote_dir = cfg.remote_repo_dir
    remote_bash(cfg, f"mkdir -p {shlex.quote(remote_dir)}")

    excludes = [
        ".git/",
        "build/",
        "dist/",
        "__pycache__/",
        "*.pyc",
        ".venv/",
        "venv/",
        ".pytest_cache/",
        ".ralph-loop/",
        ".codex/",
        ".claude/",
        "references/",
        ".ASCEND_SERVER.toml",
    ]
    cmd = ["rsync", "-az"]
    if delete:
        cmd.append("--delete")
    for e in excludes:
        cmd.extend(["--exclude", e])
    cmd.extend(["./", f"{cfg.ssh_host}:{remote_dir}/"])
    run_local(cmd)


def cmd_status(cfg: AscendServerConfig, args: argparse.Namespace) -> None:
    print(f"ssh_host={cfg.ssh_host}")
    print(f"remote_repo_dir={cfg.remote_repo_dir}")
    print(f"ascend_root={cfg.ascend_root}")
    print(f"ascend_env_script={cfg.ascend_env_script!r}")
    remote_bash(cfg, "hostname && whoami && pwd", cwd=cfg.remote_repo_dir)
    remote_bash(cfg, "ls -la", cwd=cfg.remote_repo_dir)


def cmd_push(cfg: AscendServerConfig, args: argparse.Namespace) -> None:
    rsync_repo(cfg, delete=not args.no_delete)
    print(f"Synced repo to {cfg.ssh_host}:{cfg.remote_repo_dir}")


def cmd_run(cfg: AscendServerConfig, args: argparse.Namespace) -> None:
    if not args.cmd:
        raise SystemExit("run: missing command after --")
    remote_bash(cfg, " ".join(shlex.quote(x) for x in args.cmd), cwd=cfg.remote_repo_dir)


def cmd_shell(cfg: AscendServerConfig, args: argparse.Namespace) -> None:
    # Open an interactive shell in the remote repo directory.
    parts = []
    if cfg.ascend_env_script:
        parts.append(f"source {shlex.quote(cfg.ascend_env_script)} >/dev/null 2>&1 || true")
    parts.append(f"cd {shlex.quote(cfg.remote_repo_dir)}")
    parts.append("exec bash -l")
    ssh_cmd = ["ssh", "-t", cfg.ssh_host, "bash", "-lc", " ; ".join(parts)]
    run_local(ssh_cmd)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Remote Ascend/CANN helper (sync repo, run commands) driven by .ASCEND_SERVER.toml.",
    )
    p.add_argument(
        "--config",
        default=".ASCEND_SERVER.toml",
        help="Path to config TOML (default: .ASCEND_SERVER.toml).",
    )
    sub = p.add_subparsers(dest="subcmd", required=True)

    s = sub.add_parser("status", help="Print config + basic remote connectivity checks.")
    s.set_defaults(fn=cmd_status)

    s = sub.add_parser("push", help="Rsync this repo to the remote workspace.")
    s.add_argument("--no-delete", action="store_true", help="Do not delete remote files not present locally.")
    s.set_defaults(fn=cmd_push)

    s = sub.add_parser("run", help="Run a command in the remote repo dir (use `--` to separate).")
    s.add_argument("cmd", nargs=argparse.REMAINDER, help="Command to run after `--`.")
    s.set_defaults(fn=cmd_run)

    s = sub.add_parser("shell", help="Open an interactive shell in the remote repo dir.")
    s.set_defaults(fn=cmd_shell)

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    cfg_path = Path(args.config)
    try:
        cfg = load_config(cfg_path)
    except Exception as e:
        raise SystemExit(str(e))

    os.chdir(Path(__file__).resolve().parent.parent)
    args.fn(cfg, args)


if __name__ == "__main__":
    main()

