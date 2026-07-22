from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

DYNAMICS_ROOT = Path(__file__).resolve().parents[1]
if str(DYNAMICS_ROOT) not in sys.path:
    sys.path.insert(0, str(DYNAMICS_ROOT))

from neural_dynamics.mujoco_env import MuJoCoArmEnv


MODEL_XML = DYNAMICS_ROOT / "ABB_IRB2400.xml"
N_JOINTS = 6
ZERO_Q = np.zeros(N_JOINTS, dtype=np.float32)


class MuJoCoArmObservationTest(unittest.TestCase):
    def make_env(self, **kwargs: object) -> MuJoCoArmEnv:
        env = MuJoCoArmEnv(
            str(MODEL_XML),
            n_joints=N_JOINTS,
            gravity_compensation=False,
            **kwargs,
        )
        self.addCleanup(env.close)
        env.reset_to_configuration(ZERO_Q)
        return env

    def test_default_state_and_observation_are_deterministic(self) -> None:
        env = self.make_env(seed=11)

        state_1 = env.get_state()
        state_2 = env.get_state()
        observation_1 = env.get_observation()
        observation_2 = env.get_observation()

        np.testing.assert_array_equal(state_1, state_2)
        np.testing.assert_array_equal(observation_1, state_1)
        np.testing.assert_array_equal(observation_2, state_1)

    def test_noisy_observations_are_reproducible_with_the_same_seed(self) -> None:
        env_1 = self.make_env(seed=3, observation_noise_std=0.01, observation_seed=17)
        env_2 = self.make_env(seed=3, observation_noise_std=0.01, observation_seed=17)

        for _ in range(10):
            np.testing.assert_array_equal(env_1.get_observation(), env_2.get_observation())

    def test_observation_reads_do_not_change_environment_rng(self) -> None:
        env_with_reads = self.make_env(seed=23, observation_noise_std=0.01, observation_seed=29)
        env_without_reads = self.make_env(seed=23, observation_noise_std=0.01, observation_seed=29)

        for _ in range(25):
            env_with_reads.get_observation()

        reset_state_with_reads = env_with_reads.reset_random()
        reset_state_without_reads = env_without_reads.reset_random()
        np.testing.assert_array_equal(reset_state_with_reads, reset_state_without_reads)

    def test_get_state_remains_truth_when_noise_is_enabled(self) -> None:
        env = self.make_env(seed=31, observation_noise_std=0.01, observation_seed=37)

        truth_before = env.get_state()
        observation = env.get_observation()
        truth_after = env.get_state()

        np.testing.assert_array_equal(truth_before, truth_after)
        self.assertFalse(np.array_equal(observation, truth_before))

    def test_observation_noise_has_configured_standard_deviation(self) -> None:
        noise_std = 0.01
        env = self.make_env(seed=41, observation_noise_std=noise_std, observation_seed=43)
        truth = env.get_state()

        samples = np.stack([env.get_observation() - truth for _ in range(4000)], axis=0)
        empirical_mean = samples.mean(axis=0)
        empirical_std = samples.std(axis=0)

        self.assertTrue(np.all(np.abs(empirical_mean) < 5.0e-4))
        self.assertTrue(np.all(np.abs(empirical_std - noise_std) < 7.5e-4))

    def test_split_q_and_dq_noise_have_independent_scales(self) -> None:
        q_std, dq_std = 0.001, 0.02
        env = self.make_env(
            seed=47,
            observation_q_noise_std=q_std,
            observation_dq_noise_std=dq_std,
            observation_seed=53,
        )
        truth = env.get_state()
        samples = np.stack([env.get_observation() - truth for _ in range(5000)], axis=0)
        empirical_std = samples.std(axis=0)
        self.assertTrue(np.all(np.abs(empirical_std[:N_JOINTS] - q_std) < 8.0e-5))
        self.assertTrue(np.all(np.abs(empirical_std[N_JOINTS:] - dq_std) < 1.5e-3))

    def test_legacy_and_split_noise_cannot_be_combined(self) -> None:
        with self.assertRaises(ValueError):
            MuJoCoArmEnv(
                str(MODEL_XML), observation_noise_std=0.01, observation_q_noise_std=0.001
            )

    def test_invalid_observation_noise_configuration_is_rejected(self) -> None:
        for invalid_std in (-0.1, float("nan"), float("inf")):
            with self.subTest(observation_noise_std=invalid_std):
                with self.assertRaises(ValueError):
                    MuJoCoArmEnv(str(MODEL_XML), observation_noise_std=invalid_std)

        with self.assertRaises(ValueError):
            MuJoCoArmEnv(str(MODEL_XML), observation_seed=-1)


if __name__ == "__main__":
    unittest.main()
