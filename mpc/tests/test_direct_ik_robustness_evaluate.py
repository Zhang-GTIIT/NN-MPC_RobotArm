from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "test_direct_ik_robustness_evaluate",
    ROOT / "scripts" / "robustness" / "evaluate_direct_ik.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class DirectIKRobustnessEvaluatorTests(unittest.TestCase):
    def test_medium_matrix_has_216_unique_single_factor_runs(self) -> None:
        cases = [
            {"id": f"{shape}_{index:02d}", "reference_type": shape, "run_args": {"reference_mode": "task"}}
            for shape in ("circle", "figure8", "ellipse", "square")
            for index in range(3)
        ]
        conditions = MODULE.build_conditions((0, 3, 6), MODULE.PERTURBATIONS)
        specs = MODULE.build_specs(cases, conditions, ("raw", "physical"))

        self.assertEqual(len(conditions), 9)
        self.assertEqual(len(specs), 216)
        identities = {
            (spec.projection, spec.condition.name, spec.case_id)
            for spec in specs
        }
        self.assertEqual(len(identities), len(specs))
        self.assertEqual(sum(condition.name == "nominal" for condition in conditions), 1)
        for condition in conditions:
            nonzero = [value for value in condition.levels.values() if value]
            self.assertEqual(len(nonzero), 0 if condition.name == "nominal" else 1)
            if nonzero:
                self.assertIn(nonzero[0], (3, 6))

    def test_task_case_loading_checks_mode_and_reference_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task = root / "task.npz"
            joint = root / "joint.npz"
            task.write_bytes(b"task")
            joint.write_bytes(b"joint")
            manifest = root / "benchmark.json"
            manifest.write_text(
                json.dumps(
                    {
                        "kind": "model_a_robustness",
                        "cases": [
                            {
                                "id": "circle_00",
                                "reference_type": "circle",
                                "reference_sha256": digest(task),
                                "run_args": {"reference_mode": "task", "reference_file": str(task)},
                            },
                            {
                                "id": "joint_00",
                                "reference_type": "waypoint",
                                "reference_sha256": digest(joint),
                                "run_args": {"reference_mode": "joint_file", "reference_file": str(joint)},
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )

            _, selected = MODULE.load_task_cases(manifest, ["circle_00"])
            self.assertEqual([case["id"] for case in selected], ["circle_00"])
            with self.assertRaisesRegex(ValueError, "task-space"):
                MODULE.load_task_cases(manifest, ["joint_00"])
            task.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                MODULE.load_task_cases(manifest, ["circle_00"])

    def test_summary_reports_safety_noise_force_and_command_variation(self) -> None:
        condition = MODULE.build_conditions((0, 3), ("force_pulse",))[1]
        case = {"id": "circle_00", "reference_type": "circle", "run_args": {"reference_mode": "task"}}
        spec = MODULE.ExperimentSpec("physical", condition, "circle_00", "circle", case)
        steps = 20
        actual = np.zeros((steps, 12), dtype=np.float32)
        observed = actual.copy()
        observed[:, :6] = 0.001
        arrays = {
            "actual_states": actual,
            "observed_states": observed,
            "q_des": np.zeros((steps, 6), dtype=np.float32),
            "actuator_q_ref": np.linspace(0.0, 0.1, steps, dtype=np.float32)[:, None] * np.ones((1, 6)),
            "ee_position_errors": np.linspace(0.01, 0.02, steps, dtype=np.float32),
            "ee_orientation_errors": np.linspace(0.02, 0.03, steps, dtype=np.float32),
            "lap_ids": np.zeros(steps, dtype=np.int64),
            "planning_time": np.zeros(steps, dtype=np.float32),
            "mpc_replanned": np.zeros(steps, dtype=np.int64),
            "failure_flags": np.zeros(steps, dtype=np.int64),
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

        row = MODULE.summarize_direct_ik(spec, arrays)

        self.assertEqual(row["condition"], "force_pulse_l3")
        self.assertAlmostEqual(row["velocity_violation_rate"], 1 / steps)
        self.assertAlmostEqual(row["acceleration_violation_rate"], 2 / steps)
        self.assertGreater(row["command_total_variation_rad"], 0.0)
        self.assertAlmostEqual(row["observation_q_rmse_rad"], 0.001, places=6)
        self.assertTrue(np.isfinite(row["force_peak_tracking_error"]))

    def test_aggregation_and_pairing_preserve_case_matching(self) -> None:
        rows = []
        for projection, projection_offset in (("raw", 0.0), ("physical", -0.1)):
            for condition, condition_offset in (("nominal", 0.0), ("payload_l3", 0.5)):
                for index in range(2):
                    row = {
                        "projection": projection,
                        "condition": condition,
                        "perturbation": "nominal" if condition == "nominal" else "payload",
                        "level": 0 if condition == "nominal" else 3,
                        "reference_type": "circle",
                        "case_id": f"circle_{index:02d}",
                    }
                    for metric in MODULE.SUMMARY_METRICS:
                        row[metric] = 1.0 + index + projection_offset + condition_offset
                    rows.append(row)

        aggregates = MODULE.aggregate_rows(rows)
        pooled = [
            row
            for row in aggregates
            if row["projection"] == "raw"
            and row["condition"] == "payload_l3"
            and row["reference_type"] == "all"
        ]
        self.assertEqual(len(pooled), 1)
        self.assertEqual(pooled[0]["n_cases"], 2)
        report = MODULE.paired_report(rows, samples=100, seed=4)
        degradation = report["paired_degradation"]["raw:payload_l3_minus_nominal"]["tcp_rmse_m"]
        projection = report["paired_projection_delta"]["payload_l3:physical_minus_raw"]["tcp_rmse_m"]
        self.assertAlmostEqual(degradation["mean_delta"], 0.5)
        self.assertAlmostEqual(projection["mean_delta"], -0.1)


if __name__ == "__main__":
    unittest.main()
