from __future__ import annotations

import argparse
import sys
import unittest
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DYNAMICS_ROOT = ROOT / "dynamics_modeling"
if str(DYNAMICS_ROOT) not in sys.path:
    sys.path.insert(0, str(DYNAMICS_ROOT))

from neural_dynamics.mujoco_env import MuJoCoArmEnv
from mpc.robustness import (
    ACTUATOR_KP_SCALE,
    FORCE_PULSE_DURATION_STEPS,
    NOMINAL_MODEL_XML,
    OBSERVATION_DQ_STD_RAD_S,
    OBSERVATION_Q_STD_RAD,
    PAYLOAD_COM_TOOL_M,
    PAYLOAD_DIR,
    PAYLOAD_MASS_KG,
    payload_diaginertia,
    resolve_robustness_config,
)


def resolve(value: str) -> Path:
    return Path(value).resolve()


def args(**overrides: object) -> argparse.Namespace:
    values: dict[str, object] = {
        "model_xml": str(NOMINAL_MODEL_XML),
        "n_joints": 6,
        "payload_level": 0,
        "actuator_gain_level": 0,
        "force_pulse_level": 0,
        "observation_noise_level": 0,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


class RobustnessPresetTests(unittest.TestCase):
    def test_default_is_exactly_nominal(self) -> None:
        config = resolve_robustness_config(args(), resolve)
        self.assertFalse(config.enabled)
        self.assertEqual(config.plant_model_xml, NOMINAL_MODEL_XML.resolve())
        self.assertEqual(config.payload_mass_kg, 0.0)
        self.assertEqual(config.actuator_kp_scale, 1.0)
        self.assertEqual(config.force_pulse_n, 0.0)
        self.assertEqual(config.observation_q_std_rad, 0.0)
        self.assertEqual(config.observation_dq_std_rad_s, 0.0)

    def test_all_levels_resolve_independently_and_combine(self) -> None:
        for level in range(1, 7):
            with self.subTest(level=level):
                config = resolve_robustness_config(
                    args(
                        payload_level=level,
                        actuator_gain_level=level,
                        force_pulse_level=level,
                        observation_noise_level=level,
                    ),
                    resolve,
                )
                self.assertTrue(config.enabled)
                self.assertEqual(config.payload_mass_kg, PAYLOAD_MASS_KG[level])
                self.assertEqual(config.actuator_kp_scale, ACTUATOR_KP_SCALE[level])
                self.assertEqual(config.observation_q_std_rad, OBSERVATION_Q_STD_RAD[level])
                self.assertEqual(config.observation_dq_std_rad_s, OBSERVATION_DQ_STD_RAD_S[level])
                self.assertTrue(config.plant_model_xml.is_file())
                start, stop = config.pulse_window(500)
                self.assertEqual((start, stop), (250, 250 + FORCE_PULSE_DURATION_STEPS))

    def test_payload_xmls_have_expected_mass_com_and_inertia(self) -> None:
        for level in range(1, 7):
            with self.subTest(level=level):
                model = mujoco.MjModel.from_xml_path(
                    str(PAYLOAD_DIR / f"ABB_IRB2400_payload_level_{level}.xml")
                )
                body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "payload")
                self.assertGreaterEqual(body_id, 0)
                self.assertAlmostEqual(float(model.body_mass[body_id]), PAYLOAD_MASS_KG[level], places=6)
                np.testing.assert_allclose(model.body_pos[body_id], PAYLOAD_COM_TOOL_M, atol=1e-7)
                np.testing.assert_allclose(
                    model.body_inertia[body_id], payload_diaginertia(PAYLOAD_MASS_KG[level]),
                    rtol=1e-6, atol=1e-7,
                )

    def test_payload_plant_uses_nominal_gravity_compensation(self) -> None:
        q = np.asarray((0.1, -0.3, 0.25, 0.2, -0.15, 0.1), dtype=np.float32)
        nominal = MuJoCoArmEnv(str(NOMINAL_MODEL_XML), n_joints=6)
        payload = MuJoCoArmEnv(
            str(PAYLOAD_DIR / "ABB_IRB2400_payload_level_4.xml"),
            n_joints=6,
            gravity_compensation_model_xml=str(NOMINAL_MODEL_XML),
        )
        self.addCleanup(nominal.close)
        self.addCleanup(payload.close)
        nominal.reset_to_configuration(q)
        payload.reset_to_configuration(q)
        nominal_terms = nominal.compute_torque_components(q)
        payload_terms = payload.compute_torque_components(q)
        np.testing.assert_allclose(payload_terms["gravity_tau"], nominal_terms["gravity_tau"], atol=1e-8)
        self.assertGreater(np.linalg.norm(payload_terms["gravity_mismatch_tau"]), 1e-3)

    def test_gain_scaling_preserves_nominal_metadata(self) -> None:
        nominal = MuJoCoArmEnv(str(NOMINAL_MODEL_XML), n_joints=6)
        scaled = MuJoCoArmEnv(
            str(NOMINAL_MODEL_XML), n_joints=6, actuator_kp_scale=0.6, actuator_kd_scale=np.sqrt(0.6)
        )
        self.addCleanup(nominal.close)
        self.addCleanup(scaled.close)
        nominal_kp, nominal_kd = nominal.position_actuator_gains
        effective_kp, effective_kd = scaled.position_actuator_gains
        metadata_kp, metadata_kd = scaled.nominal_position_actuator_gains
        np.testing.assert_allclose(effective_kp, nominal_kp * 0.6, rtol=1e-6)
        np.testing.assert_allclose(effective_kd, nominal_kd * np.sqrt(0.6), rtol=1e-6)
        np.testing.assert_array_equal(metadata_kp, nominal_kp)
        np.testing.assert_array_equal(metadata_kd, nominal_kd)

    def test_external_force_is_explicit_and_cleared_next_step(self) -> None:
        env = MuJoCoArmEnv(str(NOMINAL_MODEL_XML), n_joints=6, seed=4)
        self.addCleanup(env.close)
        q = np.asarray((0.0, -0.2, 0.2, 0.0, 0.0, 0.0), dtype=np.float32)
        env.reset_to_configuration(q)
        env.step(q, external_force_world=np.asarray((0.0, 100.0, 0.0)))
        self.assertGreater(np.linalg.norm(env.last_external_generalized_force), 0.0)
        env.step(q)
        np.testing.assert_array_equal(env.last_external_force_world, np.zeros(3))
        np.testing.assert_array_equal(env.last_external_generalized_force, np.zeros(6))


if __name__ == "__main__":
    unittest.main()
