from __future__ import annotations

import platform
import subprocess
import sys
from pathlib import Path
from typing import Any


def _command(args: list[str], cwd: Path) -> str:
    try:
        return subprocess.run(args, cwd=cwd, check=True, capture_output=True, text=True, timeout=15).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unavailable"


def environment_snapshot(root: Path) -> dict[str, Any]:
    packages: dict[str, str] = {}
    for name in ("numpy", "torch", "mujoco"):
        try:
            module = __import__(name)
            packages[name] = str(getattr(module, "__version__", "unknown"))
        except ImportError:
            packages[name] = "missing"
    status = _command(["git", "status", "--porcelain"], root)
    return {
        "git_commit": _command(["git", "rev-parse", "HEAD"], root),
        "git_dirty": bool(status and status != "unavailable"),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "gpu": _command(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"], root),
    }
