from __future__ import annotations

import unittest
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT / "dynamics_modeling") not in sys.path:
    sys.path.insert(0, str(ROOT / "dynamics_modeling"))

from mpc.model_c.branches import first_action_is_executable, project_activation_q_ref_sequence


class ActivationProjectedBranchTests(unittest.TestCase):
    def test_projection_makes_each_command_rate_feasible(self) -> None:
        previous = np.zeros(2, dtype=np.float32)
        previous_velocity = np.zeros(2, dtype=np.float32)
        velocity_limit = np.asarray([1.0, 2.0], dtype=np.float32)
        acceleration_limit = np.asarray([5.0, 10.0], dtype=np.float32)
        projected = project_activation_q_ref_sequence(
            np.asarray([[0.8, -0.8], [-0.8, 0.8], [0.7, -0.7]], dtype=np.float32),
            previous, previous_velocity,
            np.asarray([-1.0, -1.0], dtype=np.float32), np.asarray([1.0, 1.0], dtype=np.float32),
            0.02, velocity_limit, acceleration_limit, 0.01,
        )
        command, velocity = previous, previous_velocity
        for action in projected:
            feasible, reason = first_action_is_executable(
                action, command, velocity, np.asarray([-1.0, -1.0], dtype=np.float32),
                np.asarray([1.0, 1.0], dtype=np.float32), 0.02, velocity_limit, acceleration_limit, 0.01,
            )
            self.assertTrue(feasible, reason)
            velocity = (action - command) / 0.01
            command = action

    def test_projection_changes_an_infeasible_first_action(self) -> None:
        projected = project_activation_q_ref_sequence(
            np.asarray([[0.2]], dtype=np.float32), np.asarray([0.0], dtype=np.float32),
            np.asarray([0.0], dtype=np.float32), np.asarray([-1.0], dtype=np.float32),
            np.asarray([1.0], dtype=np.float32), 0.0, np.asarray([1.0], dtype=np.float32),
            np.asarray([5.0], dtype=np.float32), 0.01,
        )
        self.assertLess(float(projected[0, 0]), 0.2)
        self.assertAlmostEqual(float(projected[0, 0]), 0.0005, places=7)


if __name__ == "__main__":
    unittest.main()
