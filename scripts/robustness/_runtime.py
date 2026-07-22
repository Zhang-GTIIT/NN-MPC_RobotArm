"""Runtime helpers for standalone robustness tools."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


ROOT = Path(__file__).resolve().parents[2]
DYNAMICS_ROOT = ROOT / "dynamics_modeling"


def ensure_import_paths() -> None:
    for path in (ROOT, DYNAMICS_ROOT):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


def load_runner(module_name: str) -> ModuleType:
    ensure_import_paths()
    path = ROOT / "scripts" / "run_cem_mpc.py"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load MPC runner: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
