"""CLI defaults for the primary threaded ASAP experiment paths."""
from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


RUNNER = load_module("cli_defaults_runner", ROOT / "scripts" / "run_cem_mpc.py")
SWEEP = load_module("cli_defaults_sweep", ROOT / "scripts" / "run_cem_budget_sweep.py")


class ThreadedASAPDefaultTests(unittest.TestCase):
    def test_runner_defaults_to_threaded_asap(self) -> None:
        args = RUNNER.parse_args([])
        self.assertEqual(args.multirate_mode, "threaded_asap")
        self.assertEqual(args.delay_protocol, "full")
        self.assertEqual(args.ik_preview_steps, 0)

    def test_budget_sweep_defaults_to_and_accepts_threaded_asap(self) -> None:
        with mock.patch.object(sys, "argv", ["run_cem_budget_sweep.py"]):
            self.assertEqual(SWEEP.parse_args().multirate_mode, "threaded_asap")
        with mock.patch.object(sys, "argv", ["run_cem_budget_sweep.py", "--multirate_mode", "threaded_asap"]):
            self.assertEqual(SWEEP.parse_args().multirate_mode, "threaded_asap")


if __name__ == "__main__":
    unittest.main()
