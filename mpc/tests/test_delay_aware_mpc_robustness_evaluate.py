from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "test_delay_aware_mpc_robustness_evaluate",
    ROOT / "scripts" / "robustness" / "evaluate_delay_aware_mpc.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class DelayAwareMPCRobustnessEvaluatorTests(unittest.TestCase):
    def test_medium_matrix_has_432_unique_runs_and_exact_protocols(self) -> None:
        cases = [
            {"id": f"{shape}_{index:02d}", "reference_type": shape, "run_args": {"reference_mode": "task"}}
            for shape in ("circle", "figure8", "ellipse", "square")
            for index in range(3)
        ]
        conditions = MODULE.direct.build_conditions((0, 3, 6), MODULE.direct.PERTURBATIONS)
        specs = MODULE.build_specs(cases, conditions, MODULE.METHODS)

        self.assertEqual(len(specs), 432)
        self.assertEqual(
            len({(spec.method.name, spec.condition.name, spec.case_id) for spec in specs}),
            432,
        )
        protocols = {
            method.name: (method.multirate_mode, method.delay_protocol, method.uses_delay)
            for method in MODULE.METHODS
        }
        self.assertEqual(protocols["IdealZeroDelay"], ("virtual_asap", "full", False))
        self.assertEqual(protocols["NaiveDelayed"], ("virtual_asap", "naive_delayed", True))
        self.assertEqual(protocols["VirtualDelayAware"], ("virtual_asap", "full", True))
        self.assertEqual(protocols["ThreadedAsync"], ("threaded_asap", "full", True))

    def test_run_args_freeze_model_protocol_delay_and_single_perturbation(self) -> None:
        base = MODULE.parse_args([])
        condition = MODULE.direct.build_conditions((0, 6), ("payload",))[1]
        case = {
            "id": "circle_00",
            "reference_type": "circle",
            "run_args": {
                "reference_mode": "task",
                "reference_file": "reference.npz",
                "checkpoint": "wrong.pt",
                "normalizer": "wrong-normalizer.pt",
                "multirate_mode": "threaded_asap",
                "anticipation_delay_steps": 99,
                "max_execution_steps": None,
            },
        }
        ideal = MODULE.ExperimentSpec(MODULE.METHOD_BY_NAME["IdealZeroDelay"], condition, "circle_00", "circle", case)
        threaded = MODULE.ExperimentSpec(MODULE.METHOD_BY_NAME["ThreadedAsync"], condition, "circle_00", "circle", case)
        base.max_execution_steps = 20
        base.planner_projection = "on"
        base.planner_projection_backend = "compiled"
        base.planner_projection_strategy = "two_stage"

        ideal_args = MODULE._run_args(base, ideal, Path("/tmp/ideal"), 5)
        threaded_args = MODULE._run_args(base, threaded, Path("/tmp/threaded"), 5)

        self.assertEqual(ideal_args.anticipation_delay_steps, 0)
        self.assertEqual(ideal_args.multirate_mode, "virtual_asap")
        self.assertEqual(threaded_args.anticipation_delay_steps, 5)
        self.assertEqual(threaded_args.multirate_mode, "threaded_asap")
        self.assertEqual(threaded_args.checkpoint, MODULE.DEFAULT_CHECKPOINT)
        self.assertEqual(threaded_args.normalizer, MODULE.DEFAULT_NORMALIZER)
        self.assertEqual(threaded_args.max_execution_steps, 20)
        self.assertEqual(threaded_args.payload_level, 6)
        self.assertEqual(threaded_args.actuator_gain_level, 0)
        self.assertEqual(threaded_args.planner_projection, "on")
        self.assertEqual(threaded_args.planner_projection_backend, "compiled")
        self.assertEqual(threaded_args.planner_projection_strategy, "two_stage")

    def test_summary_reports_fallback_safety_and_robustness(self) -> None:
        condition = MODULE.direct.build_conditions((0, 3), ("force_pulse",))[1]
        case = {"id": "circle_00", "reference_type": "circle", "run_args": {"reference_mode": "task"}}
        spec = MODULE.ExperimentSpec(MODULE.METHOD_BY_NAME["ThreadedAsync"], condition, "circle_00", "circle", case)
        steps = 20
        arrays = {
            "actual_states": np.zeros((steps, 12), dtype=np.float32),
            "observed_states": np.zeros((steps, 12), dtype=np.float32),
            "q_des": np.zeros((steps, 6), dtype=np.float32),
            "actuator_q_ref": np.linspace(0, 0.1, steps, dtype=np.float32)[:, None] * np.ones((1, 6)),
            "ee_position_errors": np.linspace(0.01, 0.02, steps, dtype=np.float32),
            "ee_orientation_errors": np.linspace(0.02, 0.03, steps, dtype=np.float32),
            "lap_ids": np.zeros(steps, dtype=np.int64),
            "planning_time": np.zeros(steps, dtype=np.float32),
            "mpc_replanned": np.zeros(steps, dtype=np.int64),
            "failure_flags": np.zeros(steps, dtype=np.int64),
            "planner_failure_event": np.zeros(steps, dtype=np.int64),
            "fallback_active": np.asarray([1, 1] + [0] * (steps - 2)),
            "command_acceleration": np.ones((steps, 6), dtype=np.float32),
            "tau_actuator": np.ones((steps, 6), dtype=np.float32),
            "tau_total": np.ones((steps, 6), dtype=np.float32),
            "joint_limit_violation_flags": np.zeros(steps, dtype=np.int64),
            "command_velocity_violation_flags": np.asarray([1] + [0] * (steps - 1)),
            "command_acceleration_violation_flags": np.asarray([1, 1] + [0] * (steps - 2)),
            "force_pulse_level": np.asarray(3),
            "force_pulse_start_step": np.asarray(5),
            "force_pulse_stop_step": np.asarray(7),
            "force_pulse_n": np.asarray(200.0),
            "payload_level": np.asarray(0),
            "actuator_gain_level": np.asarray(0),
            "observation_noise_level": np.asarray(0),
        }

        row = MODULE.summarize_mpc(spec, arrays)

        self.assertEqual(row["method"], "ThreadedAsync")
        self.assertAlmostEqual(row["fallback_step_rate"], 0.1)
        self.assertAlmostEqual(row["velocity_violation_rate"], 0.05)
        self.assertAlmostEqual(row["acceleration_violation_rate"], 0.1)
        self.assertTrue(np.isfinite(row["force_peak_tracking_error"]))

    def test_pairing_uses_identical_method_condition_case_keys(self) -> None:
        rows = []
        for method, method_offset in (
            ("IdealZeroDelay", 0.0),
            ("NaiveDelayed", 0.2),
            ("VirtualDelayAware", 0.1),
            ("ThreadedAsync", 0.15),
        ):
            for condition, condition_offset in (("nominal", 0.0), ("payload_l3", 0.5)):
                for index in range(2):
                    row = {
                        "method": method,
                        "condition": condition,
                        "perturbation": "nominal" if condition == "nominal" else "payload",
                        "level": 0 if condition == "nominal" else 3,
                        "reference_type": "circle",
                        "case_id": f"circle_{index:02d}",
                    }
                    for metric in MODULE.METRICS:
                        row[metric] = 1.0 + index + method_offset + condition_offset
                    rows.append(row)

        report = MODULE.paired_report(rows, 100, 4)

        degradation = report["paired_degradation"]["NaiveDelayed:payload_l3_minus_nominal"]["tcp_rmse_m"]
        comparison = report["paired_method_comparison"]["payload_l3:VirtualDelayAware_minus_NaiveDelayed"]["tcp_rmse_m"]
        self.assertAlmostEqual(degradation["mean_delta"], 0.5)
        self.assertAlmostEqual(comparison["mean_delta"], -0.1)


if __name__ == "__main__":
    unittest.main()
