from __future__ import annotations

from pathlib import Path
from typing import Optional

import mujoco
import numpy as np


class MuJoCoArmEnv:
    """Small MuJoCo wrapper for joint-space arm dynamics collection."""

    _EE_NAMES = ("ee_site", "tool0", "flange", "end_effector")
    _ZERO_GRAVITY_COMPENSATION_JOINT_INDICES = (5,)
    _JOINT_LIMIT_TOLERANCE = 1e-6

    def __init__(
        self,
        model_xml: str,
        n_joints: int = 6,
        control_mode: str = "position",
        gravity_compensation: bool = True,
        frame_skip: int = 5,
        dt: Optional[float] = None,
        seed: Optional[int] = None,
        observation_noise_std: float = 0.0,
        observation_seed: Optional[int] = None,
    ) -> None:
        self.model_xml = Path(model_xml).expanduser()
        if not self.model_xml.exists():
            raise FileNotFoundError(f"MuJoCo XML file does not exist: {self.model_xml}")
        if n_joints <= 0:
            raise ValueError(f"n_joints must be positive, got {n_joints}")
        if frame_skip <= 0:
            raise ValueError(f"frame_skip must be positive, got {frame_skip}")
        if control_mode != "position":
            raise ValueError(
                "control_mode must be 'position'. Velocity control is no longer supported by this "
                f"position-actuator environment, got {control_mode!r}."
            )
        if not np.isfinite(observation_noise_std) or observation_noise_std < 0.0:
            raise ValueError(
                "observation_noise_std must be finite and non-negative, "
                f"got {observation_noise_std}"
            )
        if observation_seed is not None and observation_seed < 0:
            raise ValueError(f"observation_seed must be non-negative when provided, got {observation_seed}")

        self.n_joints = int(n_joints)
        self.control_mode = control_mode
        self.gravity_compensation = bool(gravity_compensation)
        self.frame_skip = int(frame_skip)

        # Keep environment randomization and observation noise on independent
        # random streams. Extra observation reads therefore cannot change a
        # later reset_random() result.
        self.rng = np.random.default_rng(seed)
        self.observation_noise_std = float(observation_noise_std)
        if observation_seed is None:
            observation_seed_source: int | np.random.SeedSequence = np.random.SeedSequence(seed).spawn(1)[0]
        else:
            observation_seed_source = observation_seed
        self._observation_rng = np.random.default_rng(observation_seed_source)

        self.model = mujoco.MjModel.from_xml_path(str(self.model_xml))
        self.data = mujoco.MjData(self.model)
        self._gravity_data = mujoco.MjData(self.model) if self.gravity_compensation else None
        if self.model.nu < self.n_joints:
            raise ValueError(
                f"Actuator count mismatch: model has {self.model.nu} actuators, "
                f"but n_joints={self.n_joints}. Provide a model with at least one actuator per controlled joint."
            )
        if self.model.nq < self.n_joints or self.model.nv < self.n_joints:
            raise ValueError(
                f"Joint state dimension mismatch: model has nq={self.model.nq}, nv={self.model.nv}, "
                f"but n_joints={self.n_joints}."
            )
        if dt is not None:
            if dt <= 0:
                raise ValueError(f"dt must be positive when provided, got {dt}")
            self.model.opt.timestep = float(dt)
        self.action_low = np.asarray(self.model.actuator_ctrlrange[: self.n_joints, 0], dtype=np.float32)
        self.action_high = np.asarray(self.model.actuator_ctrlrange[: self.n_joints, 1], dtype=np.float32)
        self.joint_low = self.action_low.copy()
        self.joint_high = self.action_high.copy()
        if self.model.njnt >= self.n_joints:
            joint_limited = np.asarray(self.model.jnt_limited[: self.n_joints], dtype=bool)
            joint_range = np.asarray(self.model.jnt_range[: self.n_joints], dtype=np.float32)
            self.joint_low = np.where(joint_limited, joint_range[:, 0], self.joint_low).astype(np.float32)
            self.joint_high = np.where(joint_limited, joint_range[:, 1], self.joint_high).astype(np.float32)

    @property
    def state_dim(self) -> int:
        return 2 * self.n_joints

    @property
    def action_dim(self) -> int:
        return self.n_joints

    @property
    def control_dt(self) -> float:
        return float(self.model.opt.timestep * self.frame_skip)

    @property
    def position_actuator_gains(self) -> tuple[np.ndarray, np.ndarray]:
        """Return Kp/Kd for MuJoCo position actuators in joint order.

        This metadata is deliberately optional: only the actuator-aware MPC
        profile consumes it, while the default black-box controller does not.
        """
        kp = np.asarray(self.model.actuator_gainprm[: self.n_joints, 0], dtype=np.float32)
        kd = -np.asarray(self.model.actuator_biasprm[: self.n_joints, 2], dtype=np.float32)
        if np.any(~np.isfinite(kp)) or np.any(~np.isfinite(kd)) or np.any(kp <= 0.0) or np.any(kd < 0.0):
            raise ValueError("Position actuator metadata does not provide finite non-negative Kp/Kd gains")
        return kp.copy(), kd.copy()

    def get_state(self) -> np.ndarray:
        """Return the deterministic MuJoCo truth state ``[q, dq]``."""
        qpos = np.asarray(self.data.qpos[: self.n_joints], dtype=np.float64)
        qvel = np.asarray(self.data.qvel[: self.n_joints], dtype=np.float64)
        return np.concatenate([qpos, qvel]).astype(np.float32)

<<<<<<< Updated upstream
=======
    def get_observation(self) -> np.ndarray:
        """Return ``[q, dq]`` with optional zero-mean Gaussian sensor noise.

        Noise is disabled by default. The truth-state API remains
        :meth:`get_state`, which is deterministic and never advances the
        observation random-number stream.
        """
        state = self.get_state()
        if self.observation_noise_std == 0.0:
            return state
        noise = self._observation_rng.normal(
            loc=0.0,
            scale=self.observation_noise_std,
            size=state.shape,
        )
        return (state.astype(np.float64) + noise).astype(np.float32)

>>>>>>> Stashed changes
    def validate_joint_positions(self, context: str = "step") -> None:
        qpos = np.asarray(self.data.qpos[: self.n_joints], dtype=np.float64)
        low = np.asarray(self.joint_low, dtype=np.float64) - self._JOINT_LIMIT_TOLERANCE
        high = np.asarray(self.joint_high, dtype=np.float64) + self._JOINT_LIMIT_TOLERANCE
        violations = np.where((qpos < low) | (qpos > high))[0]
        if violations.size == 0:
            return
        joint_idx = int(violations[0])
        raise RuntimeError(
            f"Joint position limit violation during {context}: joint {joint_idx} "
            f"qpos={qpos[joint_idx]:.8f}, limit=[{self.joint_low[joint_idx]:.8f}, "
            f"{self.joint_high[joint_idx]:.8f}]"
        )

    def _gravity_compensation_force(self) -> np.ndarray:
        if self._gravity_data is None:
            return np.zeros(self.n_joints, dtype=np.float64)
        mujoco.mj_resetData(self.model, self._gravity_data)
        self._gravity_data.qpos[: self.model.nq] = self.data.qpos[: self.model.nq]
        self._gravity_data.qvel[: self.model.nv] = 0.0
        mujoco.mj_forward(self.model, self._gravity_data)
        gravity_tau = np.asarray(self._gravity_data.qfrc_bias[: self.n_joints], dtype=np.float64).copy()
        for joint_idx in self._ZERO_GRAVITY_COMPENSATION_JOINT_INDICES:
            if joint_idx < self.n_joints:
                gravity_tau[joint_idx] = 0.0
        return gravity_tau

    def compute_torque_components(self, action: np.ndarray | None = None) -> dict[str, np.ndarray]:
        if action is not None:
            action_array = np.asarray(action, dtype=np.float64)
            if action_array.shape != (self.n_joints,):
                raise ValueError(f"Action must have shape ({self.n_joints},), got {action_array.shape}")
            self.data.ctrl[: self.n_joints] = np.clip(action_array, self.action_low, self.action_high)

        if self.gravity_compensation:
            gravity_tau = self._gravity_compensation_force()
        else:
            gravity_tau = np.zeros(self.n_joints, dtype=np.float64)
        self.data.qfrc_applied[: self.n_joints] = gravity_tau
        mujoco.mj_forward(self.model, self.data)
        actuator_tau = np.asarray(self.data.qfrc_actuator[: self.n_joints], dtype=np.float64).copy()
        return {
            "actuator_tau": actuator_tau,
            "gravity_tau": gravity_tau.copy(),
            "total_tau": actuator_tau + gravity_tau,
        }

    def step(self, action: np.ndarray) -> np.ndarray:
        action_array = np.asarray(action, dtype=np.float64)
        if action_array.shape != (self.n_joints,):
            raise ValueError(f"Action must have shape ({self.n_joints},), got {action_array.shape}")

        action_array = np.clip(action_array, self.action_low, self.action_high)
        self.data.ctrl[: self.n_joints] = action_array
        for _ in range(self.frame_skip):
            if self.gravity_compensation:
                self.data.qfrc_applied[: self.n_joints] = self._gravity_compensation_force()
            mujoco.mj_step(self.model, self.data)
        self.validate_joint_positions("step")
        return self.get_state()

    def reset_random(self) -> np.ndarray:
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[: self.n_joints] = self.rng.uniform(-0.25, 0.25, size=self.n_joints)
        self.data.qvel[: self.n_joints] = self.rng.uniform(-0.05, 0.05, size=self.n_joints)
        self.data.ctrl[: self.n_joints] = 0.0
        self.data.qfrc_applied[: self.n_joints] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self.validate_joint_positions("reset_random")
        return self.get_state()

    def reset_to_configuration(self, qpos: np.ndarray) -> np.ndarray:
        qpos_array = np.asarray(qpos, dtype=np.float64)
        if qpos_array.shape != (self.n_joints,):
            raise ValueError(f"qpos must have shape ({self.n_joints},), got {qpos_array.shape}")

        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[: self.n_joints] = qpos_array
        self.data.qvel[: self.n_joints] = 0.0
        self.data.ctrl[: self.n_joints] = np.clip(qpos_array, self.action_low, self.action_high)
        self.data.qfrc_applied[: self.n_joints] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self.validate_joint_positions("reset_to_configuration")
        return self.get_state()

    def get_ee_position(self) -> Optional[np.ndarray]:
        for name in self._EE_NAMES:
            site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
            if site_id >= 0:
                return np.asarray(self.data.site_xpos[site_id], dtype=np.float32).copy()

        for name in self._EE_NAMES:
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            if body_id >= 0:
                return np.asarray(self.data.xpos[body_id], dtype=np.float32).copy()
        return None

    def close(self) -> None:
        return None