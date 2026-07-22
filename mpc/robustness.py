"""Fixed, reproducible robustness presets shared by MPC runners."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
NOMINAL_MODEL_XML = ROOT / "dynamics_modeling" / "ABB_IRB2400.xml"
PAYLOAD_DIR = ROOT / "dynamics_modeling" / "payloads"

PAYLOAD_MASS_KG = (0.0, 1.0, 2.0, 4.0, 6.0, 9.0, 12.0)
ACTUATOR_KP_SCALE = (1.0, 0.90, 0.80, 0.70, 0.60, 0.45, 0.30)
FORCE_PULSE_N = (0.0, 50.0, 100.0, 200.0, 300.0, 400.0, 500.0)
OBSERVATION_Q_STD_RAD = (0.0, 0.0002, 0.0005, 0.0010, 0.0020, 0.0035, 0.0050)
OBSERVATION_DQ_STD_RAD_S = (0.0, 0.002, 0.005, 0.010, 0.020, 0.035, 0.050)
PAYLOAD_COM_TOOL_M = np.asarray((0.10, 0.0, 0.0), dtype=np.float32)
PAYLOAD_RADIUS_M = 0.04
PAYLOAD_LENGTH_M = 0.20
FORCE_DIRECTION_WORLD = np.asarray((0.0, 1.0, 0.0), dtype=np.float32)
FORCE_PULSE_DURATION_STEPS = 10


def _level(value: Any, name: str) -> int:
    level = int(value)
    if level < 0 or level > 6:
        raise ValueError(f"{name} must be in [0, 6], got {level}")
    return level


def payload_diaginertia(mass_kg: float) -> np.ndarray:
    """Principal inertia of the fixed tool-x cylinder about its COM."""
    axial = 0.5 * mass_kg * PAYLOAD_RADIUS_M**2
    transverse = mass_kg * (3.0 * PAYLOAD_RADIUS_M**2 + PAYLOAD_LENGTH_M**2) / 12.0
    return np.asarray((axial, transverse, transverse), dtype=np.float32)


@dataclass(frozen=True)
class RobustnessConfig:
    payload_level: int
    actuator_gain_level: int
    force_pulse_level: int
    observation_noise_level: int
    nominal_model_xml: Path
    plant_model_xml: Path
    payload_mass_kg: float
    payload_diaginertia_kg_m2: np.ndarray
    actuator_kp_scale: float
    actuator_kd_scale: float
    force_pulse_n: float
    observation_q_std_rad: float
    observation_dq_std_rad_s: float

    @property
    def enabled(self) -> bool:
        return any((self.payload_level, self.actuator_gain_level, self.force_pulse_level, self.observation_noise_level))

    def pulse_window(self, execution_steps: int) -> tuple[int, int]:
        start = max(0, int(execution_steps) // 2)
        return start, min(int(execution_steps), start + FORCE_PULSE_DURATION_STEPS)

    def force_world(self) -> np.ndarray:
        return FORCE_DIRECTION_WORLD * self.force_pulse_n


def resolve_robustness_config(args: Any, resolve_path: Any) -> RobustnessConfig:
    payload_level = _level(getattr(args, "payload_level", 0), "payload_level")
    gain_level = _level(getattr(args, "actuator_gain_level", 0), "actuator_gain_level")
    force_level = _level(getattr(args, "force_pulse_level", 0), "force_pulse_level")
    noise_level = _level(getattr(args, "observation_noise_level", 0), "observation_noise_level")
    nominal = Path(resolve_path(args.model_xml)).resolve()
    if payload_level:
        if int(args.n_joints) != 6 or nominal != NOMINAL_MODEL_XML.resolve():
            raise ValueError("payload_level>0 is only supported for the default 6-DoF ABB_IRB2400.xml")
        plant = PAYLOAD_DIR / f"ABB_IRB2400_payload_level_{payload_level}.xml"
    else:
        plant = nominal
    if not plant.is_file():
        raise FileNotFoundError(f"Robustness plant XML does not exist: {plant}")
    kp_scale = ACTUATOR_KP_SCALE[gain_level]
    mass = PAYLOAD_MASS_KG[payload_level]
    return RobustnessConfig(
        payload_level=payload_level,
        actuator_gain_level=gain_level,
        force_pulse_level=force_level,
        observation_noise_level=noise_level,
        nominal_model_xml=nominal,
        plant_model_xml=plant.resolve(),
        payload_mass_kg=mass,
        payload_diaginertia_kg_m2=payload_diaginertia(mass),
        actuator_kp_scale=kp_scale,
        actuator_kd_scale=float(np.sqrt(kp_scale)),
        force_pulse_n=FORCE_PULSE_N[force_level],
        observation_q_std_rad=OBSERVATION_Q_STD_RAD[noise_level],
        observation_dq_std_rad_s=OBSERVATION_DQ_STD_RAD_S[noise_level],
    )


def config_arrays(config: RobustnessConfig, env: Any) -> dict[str, np.ndarray]:
    kp, kd = env.position_actuator_gains
    return {
        "payload_level": np.asarray(config.payload_level, dtype=np.int64),
        "actuator_gain_level": np.asarray(config.actuator_gain_level, dtype=np.int64),
        "force_pulse_level": np.asarray(config.force_pulse_level, dtype=np.int64),
        "observation_noise_level": np.asarray(config.observation_noise_level, dtype=np.int64),
        "payload_mass_kg": np.asarray(config.payload_mass_kg, dtype=np.float32),
        "payload_com_tool_m": PAYLOAD_COM_TOOL_M.copy(),
        "payload_radius_m": np.asarray(PAYLOAD_RADIUS_M, dtype=np.float32),
        "payload_length_m": np.asarray(PAYLOAD_LENGTH_M, dtype=np.float32),
        "payload_diaginertia_kg_m2": config.payload_diaginertia_kg_m2.copy(),
        "actuator_kp_scale": np.asarray(config.actuator_kp_scale, dtype=np.float32),
        "actuator_kd_scale": np.asarray(config.actuator_kd_scale, dtype=np.float32),
        "effective_actuator_kp": kp,
        "effective_actuator_kd": kd,
        "force_pulse_n": np.asarray(config.force_pulse_n, dtype=np.float32),
        "force_pulse_duration_steps": np.asarray(FORCE_PULSE_DURATION_STEPS, dtype=np.int64),
        "force_direction_world": FORCE_DIRECTION_WORLD.copy(),
        "observation_q_std_rad": np.asarray(config.observation_q_std_rad, dtype=np.float32),
        "observation_dq_std_rad_s": np.asarray(config.observation_dq_std_rad_s, dtype=np.float32),
        "plant_model_xml": np.asarray(str(config.plant_model_xml)),
        "gravity_compensation_model_xml": np.asarray(str(config.nominal_model_xml)),
    }
