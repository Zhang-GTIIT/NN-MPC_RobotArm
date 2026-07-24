from __future__ import annotations

import unittest
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
DYNAMICS_ROOT = ROOT / "dynamics_modeling"
if str(DYNAMICS_ROOT) not in sys.path:
    sys.path.insert(0, str(DYNAMICS_ROOT))

from mpc.cem_controller import CEMMPCConfig, CEMMPCController
from mpc.asap_planner_worker import mean_anchor_after_plan, warm_start_shift_for_anchor


class _FailingPlanner:
    def evaluate(self, action: torch.Tensor) -> dict[str, torch.Tensor]:
        raise RuntimeError("injected failure")


class CEMWarmStartFailureTests(unittest.TestCase):
    def test_failed_async_plan_does_not_shift_or_replace_mean(self) -> None:
        controller = CEMMPCController(
            CEMMPCConfig(horizon=3, action_dim=1, num_samples=4, cem_iters=1, device="cpu"),
            _FailingPlanner(), np.array([-1.0], dtype=np.float32), np.array([1.0], dtype=np.float32),
        )
        controller.mean = torch.tensor([[0.1], [0.2], [0.3]])
        before = controller.mean.clone()
        result = controller.plan(np.zeros(2, dtype=np.float32), np.zeros(1, dtype=np.float32), warm_start_shift_steps=2)
        self.assertTrue(result.failure)
        torch.testing.assert_close(controller.mean, before)

    def test_failure_keeps_last_successful_mean_anchor(self) -> None:
        mean_anchor_step = mean_anchor_after_plan(None, 10, planner_failure=False)
        mean_anchor_step = mean_anchor_after_plan(mean_anchor_step, 15, planner_failure=True)
        shift, reset = warm_start_shift_for_anchor(mean_anchor_step, 20)
        self.assertEqual(mean_anchor_step, 10)
        self.assertEqual(shift, 10)
        self.assertFalse(reset)

    def test_late_success_still_advances_mean_anchor(self) -> None:
        # Packet publication is independent: a successful CEM call owns a new
        # mean even if the worker subsequently drops its packet as late.
        mean_anchor_step = mean_anchor_after_plan(10, 15, planner_failure=False)
        shift, reset = warm_start_shift_for_anchor(mean_anchor_step, 20)
        self.assertEqual(mean_anchor_step, 15)
        self.assertEqual(shift, 5)
        self.assertFalse(reset)


if __name__ == "__main__":
    unittest.main()
