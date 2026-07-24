"""Tests for the threaded ASAP planner worker."""
from __future__ import annotations

import unittest

import numpy as np
import torch

from mpc.constraints import project_position_command_sequence
from mpc.delay_aware import project_executable_command_np


class ExecutableAnchorProjectionTests(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "compiled projection requires CUDA")
    def test_compiled_projection_matches_eager_at_h20(self) -> None:
        generator = torch.Generator(device="cuda").manual_seed(11)
        requested = torch.randn((128, 20, 6), generator=generator, device="cuda")
        kwargs = dict(
            previous_q_ref=torch.zeros(6, device="cuda"),
            previous_q_ref_velocity=torch.zeros(6, device="cuda"),
            control_dt=0.01,
            velocity_limit=torch.tensor([1, 1, 1, 2, 2, 2], device="cuda"),
            acceleration_limit=torch.tensor([5, 5, 5, 10, 10, 12.5], device="cuda"),
            joint_low=torch.full((6,), -2.0, device="cuda"),
            joint_high=torch.full((6,), 2.0, device="cuda"),
            joint_limit_margin=0.02,
        )
        eager = project_position_command_sequence(requested, backend="eager", **kwargs)
        compiled = project_position_command_sequence(requested, backend="compiled", **kwargs)
        torch.testing.assert_close(compiled, eager, atol=2e-6, rtol=2e-6)
    def test_numpy_and_torch_sequence_projection_match_with_braking(self) -> None:
        rng = np.random.default_rng(7)
        requested = rng.uniform(-0.9, 0.9, size=(12, 3)).astype(np.float32)
        previous = np.array([0.8, -0.7, 0.1], dtype=np.float32)
        previous_velocity = np.array([0.2, -0.1, 0.0], dtype=np.float32)
        low = np.full(3, -1.0, dtype=np.float32)
        high = np.full(3, 1.0, dtype=np.float32)
        velocity_limit = np.array([0.8, 0.9, 1.0], dtype=np.float32)
        acceleration_limit = np.array([2.0, 3.0, 4.0], dtype=np.float32)
        expected = project_position_command_sequence(
            torch.from_numpy(requested).unsqueeze(0),
            previous_q_ref=torch.from_numpy(previous),
            previous_q_ref_velocity=torch.from_numpy(previous_velocity),
            control_dt=0.01,
            velocity_limit=torch.from_numpy(velocity_limit),
            acceleration_limit=torch.from_numpy(acceleration_limit),
            joint_low=torch.from_numpy(low),
            joint_high=torch.from_numpy(high),
            joint_limit_margin=0.02,
        )[0].numpy()
        actual = []
        command, velocity = previous.copy(), previous_velocity.copy()
        for target in requested:
            command, _, velocity = project_executable_command_np(
                target, np.zeros(3, dtype=np.float32), command, velocity,
                low, high, 0.02, velocity_limit, acceleration_limit, 0.01,
            )
            actual.append(command.copy())
        np.testing.assert_allclose(np.asarray(actual), expected, atol=2e-6, rtol=2e-6)

    def test_projection_is_idempotent_for_an_already_feasible_step(self) -> None:
        previous = np.array([0.1, -0.2], dtype=np.float32)
        velocity = np.array([0.02, -0.03], dtype=np.float32)
        requested = previous + velocity * 0.01
        command, _, projected_velocity = project_executable_command_np(
            requested, np.zeros(2, dtype=np.float32), previous, velocity,
            np.full(2, -1.0, dtype=np.float32), np.full(2, 1.0, dtype=np.float32), 0.01,
            np.ones(2, dtype=np.float32), np.full(2, 5.0, dtype=np.float32), 0.01,
        )
        np.testing.assert_allclose(command, requested, atol=1e-7, rtol=1e-7)
        np.testing.assert_allclose(projected_velocity, velocity, atol=2e-6, rtol=2e-6)

    def test_projected_command_and_residual_replace_raw_packet_value(self) -> None:
        command, executed, velocity = project_executable_command_np(
            nominal_q_ref=np.array([0.50], dtype=np.float32),
            requested_correction=np.array([0.08], dtype=np.float32),
            previous_command=np.array([0.50], dtype=np.float32),
            previous_velocity=np.array([0.0], dtype=np.float32),
            joint_low=np.array([-2.0], dtype=np.float32), joint_high=np.array([2.0], dtype=np.float32),
            joint_limit_margin=0.0, velocity_limit=np.array([10.0], dtype=np.float32),
            acceleration_limit=np.array([300.0], dtype=np.float32), control_dt=0.01,
        )
        np.testing.assert_allclose(command, [0.53], atol=1e-6)
        np.testing.assert_allclose(executed, [0.03], atol=1e-6)
        np.testing.assert_allclose(velocity, [3.0], atol=5e-6)
        self.assertFalse(np.allclose(executed, [0.08]))

    def test_two_step_executed_residual_velocity_uses_projected_values(self) -> None:
        dt = 0.01
        prior_command = np.array([0.50], dtype=np.float32)
        prior_velocity = np.array([0.0], dtype=np.float32)
        commands = []
        residuals = []
        for correction in (np.array([0.02], dtype=np.float32), np.array([0.03], dtype=np.float32)):
            command, residual, velocity = project_executable_command_np(
                np.array([0.50], dtype=np.float32), correction, prior_command, prior_velocity,
                np.array([-2.0], dtype=np.float32), np.array([2.0], dtype=np.float32), 0.0,
                np.array([10.0], dtype=np.float32), np.array([100.0], dtype=np.float32), dt,
            )
            commands.append(command); residuals.append(residual)
            prior_command, prior_velocity = command, velocity
        np.testing.assert_allclose(residuals[0], [0.01], atol=1e-6)
        np.testing.assert_allclose(residuals[1], [0.03], atol=1e-6)
        np.testing.assert_allclose((residuals[1] - residuals[0]) / dt, [2.0], atol=1e-5)


if __name__ == "__main__":
    unittest.main()
