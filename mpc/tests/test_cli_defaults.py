"""CLI defaults for the primary threaded ASAP experiment paths."""
from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[2]


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
        self.assertEqual(args.planner_projection, "on")
        self.assertEqual(args.planner_projection_backend, "compiled")
        self.assertEqual(args.planner_projection_strategy, "two_stage")
        self.assertEqual(args.horizon, 20)
        self.assertEqual(args.residual_cost_semantics, "requested")
        self.assertEqual(args.packet_residual_semantics, "requested")
        self.assertEqual(args.residual_feasibility_semantics, "finite")
        self.assertEqual(args.nominal_command_semantics, "raw_ik")
        self.assertEqual(args.ik_command_projection, "raw")

    def test_budget_sweep_defaults_to_and_accepts_threaded_asap(self) -> None:
        with mock.patch.object(sys, "argv", ["run_cem_budget_sweep.py"]):
            self.assertEqual(SWEEP.parse_args().multirate_mode, "threaded_asap")

    def test_task_reference_validation_honors_execution_cap(self) -> None:
        bundle = SimpleNamespace(
            q_des=np.zeros((10, 6), dtype=np.float32),
            dq_des=np.zeros((10, 6), dtype=np.float32),
            execution_steps=8,
            task_positions_des=np.zeros((10, 3), dtype=np.float32),
            task_rotations_des=np.zeros((10, 3, 3), dtype=np.float32),
            segment_ids=np.zeros(10, dtype=np.int64),
            lap_ids=np.zeros(10, dtype=np.int64),
        )
        with self.assertRaisesRegex(ValueError, "too short"):
            RUNNER._validate_task_reference(bundle, 6, 3)
        RUNNER._validate_task_reference(bundle, 6, 3, execution_steps=6)
        with mock.patch.object(sys, "argv", ["run_cem_budget_sweep.py", "--multirate_mode", "threaded_asap"]):
            self.assertEqual(SWEEP.parse_args().multirate_mode, "threaded_asap")


if __name__ == "__main__":
    unittest.main()
