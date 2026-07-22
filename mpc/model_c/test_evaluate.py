from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("test_model_c_evaluate", ROOT / "scripts" / "model_c" / "evaluate.py")
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ModelABCSummaryTests(unittest.TestCase):
    def test_no_feedback_ablation_only_duplicates_learned_models(self) -> None:
        specs = [
            {"label": "DirectIK", "kind": "direct_ik"},
            {"label": "Oracle", "kind": "oracle"},
            {"label": "A", "kind": "learned"},
        ]
        expanded = MODULE.expand_model_specs(specs, True)
        self.assertEqual([item["label"] for item in expanded], ["DirectIK", "Oracle", "A", "A_NoFeedback"])
        self.assertTrue(expanded[-1]["no_feedback"])
        self.assertNotIn("no_feedback", specs[-1])

    def test_robustness_summary_reports_noise_force_and_activity(self) -> None:
        actual = np.zeros((30, 4), dtype=np.float32)
        observed = actual.copy()
        observed[:, :2] = 0.001
        observed[:, 2:] = 0.02
        tracking = np.full(30, 0.001, dtype=np.float32)
        tracking[10:15] = np.asarray((0.02, 0.015, 0.01, 0.006, 0.004), dtype=np.float32)
        arrays = {
            "actual_states": actual,
            "observed_states": observed,
            "ee_position_errors": tracking,
            "force_pulse_start_step": np.asarray(10),
            "force_pulse_stop_step": np.asarray(12),
            "force_pulse_level": np.asarray(2),
            "force_pulse_n": np.asarray(100.0),
            "payload_level": np.asarray(1),
            "actuator_gain_level": np.asarray(3),
            "observation_noise_level": np.asarray(4),
            "payload_mass_kg": np.asarray(1.0),
            "executed_residual": np.zeros((30, 2)),
            "residual_max": np.ones(2),
            "feedback_correction": np.zeros((30, 2)),
            "feedback_max": np.ones(2),
            "command_acceleration": np.ones((30, 2)),
            "actuator_q_ref": np.arange(60, dtype=np.float32).reshape(30, 2),
            "packet_age": np.asarray([-1] * 3 + [0] * 27),
        }
        summary = MODULE.robustness_summary(arrays)
        self.assertAlmostEqual(summary["observation_q_rmse_rad"], 0.001)
        self.assertAlmostEqual(summary["observation_dq_rmse_rad_s"], 0.02)
        self.assertAlmostEqual(summary["force_peak_tracking_error"], 0.02)
        self.assertEqual(summary["force_recovery_steps"], 2.0)
        self.assertAlmostEqual(summary["direct_ik_fallback_rate"], 0.1)

    def test_threaded_sparse_solve_fields_are_aggregated_finitely(self) -> None:
        arrays = {
            "realized_tracking_error": np.array([1.0, 2.0, 3.0], dtype=np.float32),
            "failure_flags": np.zeros(3, dtype=np.int64),
            "planning_time": np.array([np.nan, 0.02, np.nan], dtype=np.float32),
            "best_cost": np.array([np.nan, 4.0, np.nan], dtype=np.float32),
            "mpc_replanned": np.array([0, 1, 0], dtype=np.int64),
            "actual_states": np.zeros((3, 2), dtype=np.float32),
            "q_des": np.zeros((3, 1), dtype=np.float32),
            "ee_position_errors": np.array([0.01, 0.02, 0.03], dtype=np.float32),
            "ee_orientation_errors": np.array([0.1, 0.2, 0.3], dtype=np.float32),
            "actual_control_period_s": np.array([np.nan, 0.01, 0.012], dtype=np.float32),
            "control_step_wall_time": np.array([0.001, 0.002, 0.003], dtype=np.float32),
            "control_wakeup_lateness_s": np.array([0.0, 0.001, 0.002], dtype=np.float32),
            "control_deadline_miss": np.zeros(3, dtype=np.int64),
            "packet_age": np.array([-1, 0, 1], dtype=np.int64),
            "planner_solve_count": np.asarray(5),
            "planner_actual_update_rate_hz": np.asarray(25.0),
            "planner_late_drop_count": np.asarray(1),
        }
        summary = MODULE.summarize("A", arrays, "")
        self.assertAlmostEqual(summary["planning_time_mean"], 0.02)
        self.assertAlmostEqual(summary["best_cost_mean"], 4.0)
        self.assertAlmostEqual(summary["active_packet_ratio"], 2.0 / 3.0)
        self.assertAlmostEqual(summary["planner_late_drop_rate"], 0.2)
        self.assertTrue(np.isfinite(summary["tcp_position_rmse_m"]))


if __name__ == "__main__":
    unittest.main()
