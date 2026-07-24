"""Tests for threaded MPC summary metrics."""
from __future__ import annotations

import unittest

import numpy as np

from mpc.logging import build_run_summary


class ThreadedLoggingTests(unittest.TestCase):
    def test_threaded_summary_uses_worker_rate_not_anticipation_deadline(self) -> None:
        arrays = {
            "controller_mode": np.asarray("mpc"),
            "mpc_policy": np.asarray("residual"),
            "cost_profile": np.asarray("default"),
            "multirate_mode": np.asarray("threaded_asap"),
            "actual_states": np.zeros((3, 2), dtype=np.float32),
            "q_des": np.zeros((3, 1), dtype=np.float32),
            "replan_interval_steps": np.asarray(-1),
            "replan_deadline_s": np.asarray(np.nan),
            "mpc_replanned": np.asarray([0, 1, 1]),
            "replan_deadline_miss": np.asarray([0, 0, 1]),
            "planner_solve_count": np.asarray(7),
            "planner_actual_update_rate_hz": np.asarray(27.5),
            "planner_late_drop_count": np.asarray(1),
            "packet_publish_deadline_s": np.asarray(0.055),
            "actual_control_period_s": np.asarray([np.nan, 0.010, 0.012]),
            "control_wakeup_lateness_s": np.asarray([0.0, 0.0, 0.002]),
        }
        summary = build_run_summary(arrays)
        self.assertIsNone(summary["replanning"]["interval_steps"])
        self.assertTrue(np.isnan(summary["replanning"]["nominal_frequency_hz"]))
        self.assertEqual(summary["planner"]["solve_count"], 7)
        self.assertAlmostEqual(summary["planner"]["actual_update_rate_hz"], 27.5)
        self.assertAlmostEqual(summary["planner"]["late_drop_rate"], 1.0 / 7.0)
        self.assertAlmostEqual(summary["timing"]["actual_control_period_s"]["p95"], 0.0119)


if __name__ == "__main__":
    unittest.main()
