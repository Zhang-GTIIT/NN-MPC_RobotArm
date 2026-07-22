"""Offline task-space reference construction and validation.

The online controller still consumes joint-space ``q_des`` and ``dq_des``.
This module is the explicit boundary between a desired TCP pose trajectory and
the existing joint-space CEM-MPC implementation.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import mujoco
import numpy as np

from mpc.ik_solver import IKConfig, MujocoDLSIKSolver, TrajectoryIKResult
from mpc.kinematics_utils import MujocoKinematics, orientation_error, wrap_to_pi
from mpc.task_space_reference import (
    SEGMENT_FINAL_HOLD,
    SEGMENT_HORIZON_PADDING,
    SEGMENT_INITIAL_HOLD,
    SEGMENT_JOINT_DEPARTURE,
    SEGMENT_JOINT_RETURN,
    SEGMENT_NAMES,
    SEGMENT_SHAPE_LOOP,
    SEGMENT_TASK_APPROACH,
    SEGMENT_TASK_RETURN,
    TaskSpaceTrajectory,
    generate_task_space_trajectory,
    orthonormal_plane_axes,
    quintic_time_scaling,
)


TASK_SEGMENT_IDS = frozenset({SEGMENT_TASK_APPROACH, SEGMENT_SHAPE_LOOP, SEGMENT_TASK_RETURN})
REFERENCE_FILE_NAME = "reference.npz"
REFERENCE_FORMAT_VERSION = 1


@dataclass(frozen=True)
class ReferenceConfig:
    """Configuration for one complete zero-pose -> shape -> zero-pose episode."""

    shape_name: str = "circle"
    repeat_count: int = 3

    start_hold_duration: float = 0.5
    joint_departure_duration: float = 2.0
    approach_duration: float = 2.0
    lap_duration: float = 4.0
    return_duration: float = 2.0
    joint_return_duration: float = 2.0
    final_hold_duration: float = 0.5
    collection_only: bool = False
    shape_end_hold_duration: float = 0.2

    center_mode: str = "relative"
    center_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    plane_axis_u: tuple[float, float, float] = (0.0, 1.0, 0.0)
    plane_axis_v: tuple[float, float, float] = (0.0, 0.0, 1.0)
    fixed_orientation: str = "safe"
    ee_site_name: str = "ee_site"

    circle_radius: float = 0.03
    ellipse_axis_a: float = 0.04
    ellipse_axis_b: float = 0.025
    figure8_axis_a: float = 0.035
    figure8_axis_b: float = 0.02
    square_half_side: float = 0.025
    rounded_square_corner_radius: float = 0.006

    safe_departure_mode: str = "auto"
    safe_sigma_threshold: float = 0.10
    safe_search_samples: int = 2048
    safe_search_seed: int = 0
    safe_joint_limit_margin: float = 0.05
    safe_q: tuple[float, ...] | None = None

    ik_config: IKConfig = field(default_factory=IKConfig)
    singularity_warning: float = 0.02
    singularity_reject: float = 0.005
    max_joint_jump: float = 0.15
    final_return_tolerance: float = 1e-3
    small_return_tolerance: float = 0.05
    lap_position_closure_tolerance: float = 1e-3
    lap_joint_closure_tolerance: float = 0.05
    max_joint_velocity: tuple[float, ...] = (1.0, 1.0, 1.0, 2.0, 2.0, 2.5)
    max_joint_acceleration: tuple[float, ...] = (5.0, 5.0, 5.0, 10.0, 10.0, 12.5)

    def __post_init__(self) -> None:
        if self.shape_name.lower() not in {"circle", "ellipse", "figure8", "square", "rounded_square"}:
            raise ValueError(f"Unsupported shape_name {self.shape_name!r}")
        if not isinstance(self.repeat_count, int) or self.repeat_count <= 0:
            raise ValueError("repeat_count must be a positive integer")
        for name in (
            "start_hold_duration",
            "joint_departure_duration",
            "approach_duration",
            "lap_duration",
            "return_duration",
            "joint_return_duration",
            "final_hold_duration",
        ):
            if float(getattr(self, name)) <= 0.0:
                raise ValueError(f"{name} must be positive")
        if self.center_mode not in {"relative", "absolute"}:
            raise ValueError("center_mode must be 'relative' or 'absolute'")
        if self.fixed_orientation not in {"initial", "safe"}:
            raise ValueError("fixed_orientation must be 'initial' or 'safe'")
        if self.safe_departure_mode not in {"auto", "always", "never"}:
            raise ValueError("safe_departure_mode must be 'auto', 'always', or 'never'")
        if self.safe_sigma_threshold <= 0.0:
            raise ValueError("safe_sigma_threshold must be positive")
        if self.safe_search_samples <= 0:
            raise ValueError("safe_search_samples must be positive")
        if self.safe_joint_limit_margin < 0.0:
            raise ValueError("safe_joint_limit_margin must be non-negative")
        if self.shape_end_hold_duration <= 0.0:
            raise ValueError("shape_end_hold_duration must be positive")
        if self.singularity_reject < 0.0 or self.singularity_warning < self.singularity_reject:
            raise ValueError("singularity thresholds must satisfy 0 <= reject <= warning")
        if self.max_joint_jump <= 0.0:
            raise ValueError("max_joint_jump must be positive")
        if self.final_return_tolerance <= 0.0 or self.small_return_tolerance < self.final_return_tolerance:
            raise ValueError("return tolerances must satisfy 0 < final <= small")
        if self.lap_position_closure_tolerance <= 0.0 or self.lap_joint_closure_tolerance <= 0.0:
            raise ValueError("lap closure tolerances must be positive")
        if any(float(value) <= 0.0 for value in self.max_joint_velocity):
            raise ValueError("max_joint_velocity values must all be positive")
        if any(float(value) <= 0.0 for value in self.max_joint_acceleration):
            raise ValueError("max_joint_acceleration values must all be positive")


@dataclass(frozen=True)
class SafePoseSearchResult:
    """Result of the deterministic, constrained offline safe-pose search."""

    q: np.ndarray
    sigma_min: float
    joint_limit_margin: float
    candidates_evaluated: int
    valid_candidates: int


@dataclass
class ReferenceBundle:
    """A complete MPC reference plus the diagnostics used to validate it."""

    time: np.ndarray
    q_des: np.ndarray
    dq_des: np.ndarray
    ddq_des: np.ndarray
    task_positions_des: np.ndarray | None = None
    task_rotations_des: np.ndarray | None = None
    segment_ids: np.ndarray | None = None
    lap_ids: np.ndarray | None = None
    ik_position_errors: np.ndarray | None = None
    ik_orientation_errors: np.ndarray | None = None
    ik_iterations: np.ndarray | None = None
    ik_sigma_min: np.ndarray | None = None
    ik_joint_limit_margins: np.ndarray | None = None
    execution_steps: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.time = np.asarray(self.time, dtype=np.float64)
        self.q_des = np.asarray(self.q_des, dtype=np.float64)
        self.dq_des = np.asarray(self.dq_des, dtype=np.float64)
        self.ddq_des = np.asarray(self.ddq_des, dtype=np.float64)
        if self.q_des.ndim != 2 or self.q_des.shape[0] == 0:
            raise ValueError(f"q_des must have shape [N, n_joints] with N > 0, got {self.q_des.shape}")
        sample_count, n_joints = self.q_des.shape
        if self.time.shape != (sample_count,):
            raise ValueError(f"time must have shape ({sample_count},), got {self.time.shape}")
        for name in ("dq_des", "ddq_des"):
            value = getattr(self, name)
            if value.shape != (sample_count, n_joints):
                raise ValueError(f"{name} must have shape ({sample_count}, {n_joints}), got {value.shape}")
        if not np.all(np.isfinite(self.time)) or not np.all(np.isfinite(self.q_des)):
            raise ValueError("time and q_des must contain only finite values")
        if not np.all(np.isfinite(self.dq_des)) or not np.all(np.isfinite(self.ddq_des)):
            raise ValueError("dq_des and ddq_des must contain only finite values")

        self.task_positions_des = _optional_array(self.task_positions_des, (sample_count, 3), "task_positions_des")
        self.task_rotations_des = _optional_array(self.task_rotations_des, (sample_count, 3, 3), "task_rotations_des")
        if (self.task_positions_des is None) != (self.task_rotations_des is None):
            raise ValueError("task_positions_des and task_rotations_des must either both be set or both be None")
        self.segment_ids = _optional_array(self.segment_ids, (sample_count,), "segment_ids", np.int64)
        if self.segment_ids is None:
            self.segment_ids = np.full(sample_count, -1, dtype=np.int64)
        self.lap_ids = _optional_array(self.lap_ids, (sample_count,), "lap_ids", np.int64)
        if self.lap_ids is None:
            self.lap_ids = np.full(sample_count, -1, dtype=np.int64)
        self.ik_position_errors = _optional_array(self.ik_position_errors, (sample_count,), "ik_position_errors")
        self.ik_orientation_errors = _optional_array(self.ik_orientation_errors, (sample_count,), "ik_orientation_errors")
        self.ik_iterations = _optional_array(self.ik_iterations, (sample_count,), "ik_iterations", np.int64)
        self.ik_sigma_min = _optional_array(self.ik_sigma_min, (sample_count,), "ik_sigma_min")
        self.ik_joint_limit_margins = _optional_array(
            self.ik_joint_limit_margins,
            (sample_count,),
            "ik_joint_limit_margins",
        )
        self.execution_steps = int(self.execution_steps)
        if self.execution_steps <= 0 or self.execution_steps > sample_count:
            raise ValueError(
                f"execution_steps must be in [1, {sample_count}], got {self.execution_steps}"
            )
        self.metadata = dict(self.metadata)

    @property
    def n_joints(self) -> int:
        return int(self.q_des.shape[1])

    @property
    def reference_length(self) -> int:
        return int(self.q_des.shape[0])


def _optional_array(
    value: np.ndarray | None,
    shape: tuple[int, ...],
    name: str,
    dtype: np.dtype[Any] | type[np.generic] | None = None,
) -> np.ndarray | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=dtype)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if dtype is None and not np.all(np.isfinite(array)):
        # IK diagnostics intentionally use NaN outside task-space segments.
        # They are therefore exempted below by their explicit names.
        if name not in {
            "ik_position_errors",
            "ik_orientation_errors",
            "ik_sigma_min",
            "ik_joint_limit_margins",
        }:
            raise ValueError(f"{name} must contain only finite values")
    return array.copy()


def _duration_steps(duration: float, control_dt: float) -> int:
    if control_dt <= 0.0:
        raise ValueError(f"control_dt must be positive, got {control_dt}")
    return max(1, int(round(float(duration) / float(control_dt))))


def _joint_quintic_segment(start_q: np.ndarray, end_q: np.ndarray, duration: float, control_dt: float) -> np.ndarray:
    steps = _duration_steps(duration, control_dt)
    tau = np.linspace(0.0, 1.0, steps + 1, dtype=np.float64)
    scale = quintic_time_scaling(tau)[:, None]
    return start_q[None, :] + scale * (end_q - start_q)[None, :]


def _joint_hold(q: np.ndarray, duration: float, control_dt: float) -> np.ndarray:
    return np.repeat(q[None, :], _duration_steps(duration, control_dt), axis=0)


def _normalize_q(q: np.ndarray, n_joints: int, name: str) -> np.ndarray:
    values = np.asarray(q, dtype=np.float64)
    if values.shape != (n_joints,):
        raise ValueError(f"{name} must have shape ({n_joints},), got {values.shape}")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain only finite values")
    return values.copy()


def find_safe_pose(
    kinematics: MujocoKinematics,
    initial_q: np.ndarray,
    *,
    sigma_threshold: float = 0.10,
    joint_limit_margin: float = 0.05,
    sample_count: int = 2048,
    seed: int = 0,
) -> SafePoseSearchResult:
    """Find a non-singular, limit-safe departure pose without hard-coding one.

    The deterministic candidate set contains mostly local samples around the
    zero pose, plus a smaller global coverage set.  Candidates are ranked by
    normalized joint displacement from ``initial_q`` and then by larger minimum
    Jacobian singular value.  This preserves the approved "closest validated
    safe pose" policy while still escaping a singular zero wrist configuration.
    """

    if sigma_threshold <= 0.0:
        raise ValueError("sigma_threshold must be positive")
    if joint_limit_margin < 0.0:
        raise ValueError("joint_limit_margin must be non-negative")
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    q0 = _normalize_q(initial_q, kinematics.n_joints, "initial_q")
    low = np.asarray(kinematics.joint_low, dtype=np.float64) + float(joint_limit_margin)
    high = np.asarray(kinematics.joint_high, dtype=np.float64) - float(joint_limit_margin)
    if np.any(low >= high):
        raise ValueError("joint_limit_margin leaves no safe pose search interval")
    if np.any(q0 < low) or np.any(q0 > high):
        raise ValueError("initial_q violates the requested safe joint-limit margin")

    initial_sigma = kinematics.sigma_min(q0)
    if initial_sigma >= sigma_threshold:
        return SafePoseSearchResult(q=q0, sigma_min=initial_sigma, joint_limit_margin=kinematics.joint_limit_margin(q0), candidates_evaluated=1, valid_candidates=1)

    rng = np.random.default_rng(seed)
    span = high - low
    local_count = max(1, int(round(sample_count * 0.75)))
    global_count = max(0, sample_count - local_count)
    # A local normal cloud keeps the selected pose close to the current
    # configuration.  The global samples keep the search from being trapped by
    # a broad near-singular neighborhood.
    local = q0[None, :] + rng.normal(loc=0.0, scale=0.18, size=(local_count, kinematics.n_joints)) * span[None, :]
    local = np.clip(local, low[None, :], high[None, :])
    global_samples = rng.uniform(low, high, size=(global_count, kinematics.n_joints))
    candidates = np.concatenate([q0[None, :], local, global_samples], axis=0)

    best_q: np.ndarray | None = None
    best_sigma = float("-inf")
    best_distance = float("inf")
    valid_count = 0
    for candidate in candidates:
        sigma = kinematics.sigma_min(candidate)
        if sigma < sigma_threshold:
            continue
        valid_count += 1
        distance = float(np.linalg.norm((candidate - q0) / span))
        if distance < best_distance - 1e-12 or (np.isclose(distance, best_distance) and sigma > best_sigma):
            best_q = candidate.copy()
            best_sigma = float(sigma)
            best_distance = distance
    if best_q is None:
        raise RuntimeError(
            "Could not find a safe departure pose: no candidate satisfied "
            f"sigma_min >= {sigma_threshold:.6g} after {candidates.shape[0]} evaluations"
        )
    return SafePoseSearchResult(
        q=best_q,
        sigma_min=best_sigma,
        joint_limit_margin=kinematics.joint_limit_margin(best_q),
        candidates_evaluated=int(candidates.shape[0]),
        valid_candidates=valid_count,
    )


def _select_departure_pose(
    config: ReferenceConfig,
    kinematics: MujocoKinematics,
    initial_q: np.ndarray,
) -> tuple[np.ndarray, bool, SafePoseSearchResult | None]:
    initial_sigma = kinematics.sigma_min(initial_q)
    if config.safe_q is not None:
        safe_q = _normalize_q(np.asarray(config.safe_q, dtype=np.float64), kinematics.n_joints, "safe_q")
        margin = kinematics.joint_limit_margin(safe_q)
        sigma = kinematics.sigma_min(safe_q)
        if margin < config.safe_joint_limit_margin:
            raise ValueError("configured safe_q violates safe_joint_limit_margin")
        if sigma < config.safe_sigma_threshold:
            raise ValueError("configured safe_q violates safe_sigma_threshold")
        return safe_q, True, SafePoseSearchResult(safe_q, sigma, margin, 0, 1)

    needs_safe_pose = (
        config.safe_departure_mode == "always"
        or (config.safe_departure_mode == "auto" and initial_sigma < config.safe_sigma_threshold)
    )
    if config.safe_departure_mode == "never":
        needs_safe_pose = False
    if not needs_safe_pose:
        return initial_q.copy(), False, None
    result = find_safe_pose(
        kinematics,
        initial_q,
        sigma_threshold=config.safe_sigma_threshold,
        joint_limit_margin=config.safe_joint_limit_margin,
        sample_count=config.safe_search_samples,
        seed=config.safe_search_seed,
    )
    return result.q.copy(), True, result


def _resolve_center(config: ReferenceConfig, departure_position: np.ndarray) -> np.ndarray:
    value = np.asarray(config.center_offset, dtype=np.float64)
    if value.shape != (3,):
        raise ValueError(f"center_offset must have shape (3,), got {value.shape}")
    if config.center_mode == "relative":
        return departure_position + value
    return value.copy()


def _combine_parts(parts: Iterable[np.ndarray]) -> np.ndarray:
    arrays = [np.asarray(part, dtype=np.float64) for part in parts if np.asarray(part).size]
    if not arrays:
        raise RuntimeError("reference assembly produced no samples")
    return np.concatenate(arrays, axis=0)


def _compute_fk_trajectory(kinematics: MujocoKinematics, q_des: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    count = q_des.shape[0]
    positions = np.empty((count, 3), dtype=np.float64)
    rotations = np.empty((count, 3, 3), dtype=np.float64)
    sigma_min = np.empty(count, dtype=np.float64)
    margins = np.empty(count, dtype=np.float64)
    for index, q in enumerate(q_des):
        position, rotation = kinematics.forward(q)
        positions[index] = position
        rotations[index] = rotation
        sigma_min[index] = kinematics.sigma_min(q)
        margins[index] = kinematics.joint_limit_margin(q)
    return positions, rotations, sigma_min, margins


def _finite_difference(values: np.ndarray, control_dt: float) -> np.ndarray:
    if values.shape[0] <= 1:
        return np.zeros_like(values, dtype=np.float64)
    return np.gradient(values, float(control_dt), axis=0, edge_order=1)


def _task_mask(segment_ids: np.ndarray) -> np.ndarray:
    return np.isin(segment_ids, tuple(TASK_SEGMENT_IDS))


def build_reference(
    config: ReferenceConfig,
    model: mujoco.MjModel,
    initial_q: np.ndarray,
    control_dt: float,
    horizon: int,
    lookahead_steps: int = 0,
) -> ReferenceBundle:
    """Generate, solve, validate, and pad an offline task-space reference.

    ``initial_q`` is intentionally checked to be zero in this project version:
    each MPC episode starts from the verified ABB zero configuration.  A caller
    that needs arbitrary starts should make that a deliberate future API change,
    rather than accidentally mixing task references with a different reset pose.
    """

    if horizon < 0 or lookahead_steps < 0:
        raise ValueError(f"horizon must be non-negative, got {horizon}")
    if control_dt <= 0.0:
        raise ValueError(f"control_dt must be positive, got {control_dt}")
    kinematics = MujocoKinematics(model=model, ee_site_name=config.ee_site_name, n_joints=model.nu)
    q0 = _normalize_q(initial_q, kinematics.n_joints, "initial_q")
    if not np.allclose(q0, 0.0, atol=1e-12):
        raise ValueError("Task-space reference generation requires the fixed zero initial_q in this project version")
    if np.any(q0 < kinematics.joint_low) or np.any(q0 > kinematics.joint_high):
        raise ValueError("zero initial_q is outside the model joint limits")

    departure_q, uses_safe_departure, safe_result = _select_departure_pose(config, kinematics, q0)
    initial_position, initial_rotation = kinematics.forward(q0)
    departure_position, departure_rotation = kinematics.forward(departure_q)
    fixed_rotation = initial_rotation if config.fixed_orientation == "initial" else departure_rotation
    center = _resolve_center(config, departure_position)
    axis_u, axis_v, plane_normal = orthonormal_plane_axes(config.plane_axis_u, config.plane_axis_v)
    task_trajectory: TaskSpaceTrajectory = generate_task_space_trajectory(
        shape_name=config.shape_name,
        start_position=departure_position,
        center=center,
        plane_axis_u=axis_u,
        plane_axis_v=axis_v,
        fixed_rotation=fixed_rotation,
        control_dt=control_dt,
        approach_duration=config.approach_duration,
        lap_duration=config.lap_duration,
        return_duration=config.return_duration,
        repeat_count=config.repeat_count,
        circle_radius=config.circle_radius,
        ellipse_axis_a=config.ellipse_axis_a,
        ellipse_axis_b=config.ellipse_axis_b,
        figure8_axis_a=config.figure8_axis_a,
        figure8_axis_b=config.figure8_axis_b,
        square_half_side=config.square_half_side,
        rounded_square_corner_radius=config.rounded_square_corner_radius,
        include_return=not config.collection_only,
    )

    solver = MujocoDLSIKSolver(
        model=model,
        ee_site_name=config.ee_site_name,
        config=config.ik_config,
        n_joints=kinematics.n_joints,
    )
    ik_result: TrajectoryIKResult = solver.solve_trajectory(
        task_trajectory.positions,
        task_trajectory.rotations,
        departure_q,
    )
    if not ik_result.success:
        index = int(ik_result.failure_index) if ik_result.failure_index is not None else -1
        segment = int(task_trajectory.segment_ids[index]) if index >= 0 else -1
        lap = int(task_trajectory.lap_ids[index]) if index >= 0 else -1
        raise RuntimeError(
            "Continuous DLS IK failed while building task reference: "
            f"index={index}, segment={SEGMENT_NAMES.get(segment, segment)}, lap={lap}, "
            f"message={ik_result.failure_message}"
        )

    task_q = np.asarray(ik_result.q_des, dtype=np.float64)
    return_difference = wrap_to_pi(task_q[-1] - departure_q)
    return_error = float(np.max(np.abs(return_difference)))
    if not config.collection_only and return_error > config.small_return_tolerance:
        raise RuntimeError(
            "Task-space return converged to a different IK branch: "
            f"max wrapped joint difference={return_error:.6g} rad exceeds "
            f"small_return_tolerance={config.small_return_tolerance:.6g}"
        )

    initial_hold = _joint_hold(q0, config.start_hold_duration, control_dt)
    q_parts: list[np.ndarray] = [initial_hold]
    segment_parts: list[np.ndarray] = [np.full(initial_hold.shape[0], SEGMENT_INITIAL_HOLD, dtype=np.int64)]
    lap_parts: list[np.ndarray] = [np.full(initial_hold.shape[0], -1, dtype=np.int64)]
    if uses_safe_departure:
        departure = _joint_quintic_segment(q0, departure_q, config.joint_departure_duration, control_dt)
        q_parts.append(departure)
        segment_parts.append(np.full(departure.shape[0], SEGMENT_JOINT_DEPARTURE, dtype=np.int64))
        lap_parts.append(np.full(departure.shape[0], -1, dtype=np.int64))

    task_start = sum(part.shape[0] for part in q_parts)
    q_parts.append(task_q)
    segment_parts.append(task_trajectory.segment_ids)
    lap_parts.append(task_trajectory.lap_ids)

    # The fixed task return should arrive very close to departure_q.  The joint
    # return starts from the actual final IK solution so there is never a hidden
    # discontinuity before returning exactly to the zero configuration.
    if not config.collection_only and (uses_safe_departure or return_error > config.final_return_tolerance):
        joint_return = _joint_quintic_segment(task_q[-1], q0, config.joint_return_duration, control_dt)
        q_parts.append(joint_return)
        segment_parts.append(np.full(joint_return.shape[0], SEGMENT_JOINT_RETURN, dtype=np.int64))
        lap_parts.append(np.full(joint_return.shape[0], -1, dtype=np.int64))

    final_hold = _joint_hold(task_q[-1] if config.collection_only else q0, config.shape_end_hold_duration if config.collection_only else config.final_hold_duration, control_dt)
    q_parts.append(final_hold)
    segment_parts.append(np.full(final_hold.shape[0], SEGMENT_FINAL_HOLD, dtype=np.int64))
    lap_parts.append(np.full(final_hold.shape[0], -1, dtype=np.int64))

    q_execution = _combine_parts(q_parts)
    segment_execution = _combine_parts(segment_parts).astype(np.int64)
    lap_execution = _combine_parts(lap_parts).astype(np.int64)
    execution_steps = int(q_execution.shape[0])
    padding_anchor = task_q[-1] if config.collection_only else q0
    padding = np.repeat(padding_anchor[None, :], int(horizon) + int(lookahead_steps) + 1, axis=0)
    q_des = np.concatenate([q_execution, padding], axis=0)
    segment_ids = np.concatenate(
        [segment_execution, np.full(padding.shape[0], SEGMENT_HORIZON_PADDING, dtype=np.int64)]
    )
    lap_ids = np.concatenate([lap_execution, np.full(padding.shape[0], -1, dtype=np.int64)])
    time = np.arange(q_des.shape[0], dtype=np.float64) * float(control_dt)

    dq_des = _finite_difference(q_des, control_dt)
    ddq_des = _finite_difference(dq_des, control_dt)
    zero_derivative_mask = np.isin(
        segment_ids,
        (SEGMENT_INITIAL_HOLD, SEGMENT_FINAL_HOLD, SEGMENT_HORIZON_PADDING),
    )
    dq_des[zero_derivative_mask] = 0.0
    ddq_des[zero_derivative_mask] = 0.0

    desired_positions, desired_rotations, sigma_min, joint_margins = _compute_fk_trajectory(kinematics, q_des)
    task_slice = slice(task_start, task_start + task_q.shape[0])
    desired_positions[task_slice] = task_trajectory.positions
    desired_rotations[task_slice] = task_trajectory.rotations
    ik_position_errors = np.full(q_des.shape[0], np.nan, dtype=np.float64)
    ik_orientation_errors = np.full(q_des.shape[0], np.nan, dtype=np.float64)
    ik_iterations = np.full(q_des.shape[0], -1, dtype=np.int64)
    ik_sigma_min = sigma_min.copy()
    ik_joint_limit_margins = joint_margins.copy()
    ik_position_errors[task_slice] = ik_result.position_errors
    ik_orientation_errors[task_slice] = ik_result.orientation_errors
    ik_iterations[task_slice] = ik_result.iteration_counts

    metadata: dict[str, Any] = {
        "format_version": REFERENCE_FORMAT_VERSION,
        "shape_name": config.shape_name.lower(),
        "control_dt": float(control_dt),
        "horizon_padding_steps": int(padding.shape[0]),
        "lookahead_steps": int(lookahead_steps),
        "collection_only": bool(config.collection_only),
        "uses_safe_departure": bool(uses_safe_departure),
        "initial_q": q0.tolist(),
        "departure_q": departure_q.tolist(),
        "initial_tcp_position": initial_position.tolist(),
        "departure_tcp_position": departure_position.tolist(),
        "center": center.tolist(),
        "plane_axis_u": axis_u.tolist(),
        "plane_axis_v": axis_v.tolist(),
        "plane_normal": plane_normal.tolist(),
        "fixed_orientation": config.fixed_orientation,
        "ee_site_name": config.ee_site_name,
        "safe_search": None
        if safe_result is None
        else {
            "q": safe_result.q.tolist(),
            "sigma_min": safe_result.sigma_min,
            "joint_limit_margin": safe_result.joint_limit_margin,
            "candidates_evaluated": safe_result.candidates_evaluated,
            "valid_candidates": safe_result.valid_candidates,
        },
        "task_trajectory": task_trajectory.metadata,
        "config": _json_safe(asdict(config)),
        # The XML deliberately disables all contact geoms, so reporting a pass
        # would be misleading.  This explicit flag is persisted with the ref.
        "self_collision_check": "not_available",
    }
    bundle = ReferenceBundle(
        time=time,
        q_des=q_des,
        dq_des=dq_des,
        ddq_des=ddq_des,
        task_positions_des=desired_positions,
        task_rotations_des=desired_rotations,
        segment_ids=segment_ids,
        lap_ids=lap_ids,
        ik_position_errors=ik_position_errors,
        ik_orientation_errors=ik_orientation_errors,
        ik_iterations=ik_iterations,
        ik_sigma_min=ik_sigma_min,
        ik_joint_limit_margins=ik_joint_limit_margins,
        execution_steps=execution_steps,
        metadata=metadata,
    )
    diagnostics = validate_reference_bundle(bundle, kinematics, config)
    bundle.metadata["validation"] = diagnostics
    return bundle


def validate_reference_bundle(
    bundle: ReferenceBundle,
    kinematics: MujocoKinematics,
    config: ReferenceConfig | None = None,
) -> dict[str, Any]:
    """Recompute all offline validation metrics and raise on a hard failure.

    The initial zero pose is intentionally a wrist singularity in this model.
    Singularity rejection is therefore applied to IK-controlled task segments,
    while all segments are still recorded and warned about in the diagnostics.
    """

    config = ReferenceConfig() if config is None else config
    if bundle.n_joints != kinematics.n_joints:
        raise ValueError(f"Reference has {bundle.n_joints} joints, kinematics has {kinematics.n_joints}")
    q = bundle.q_des
    q_low = np.asarray(kinematics.joint_low, dtype=np.float64)
    q_high = np.asarray(kinematics.joint_high, dtype=np.float64)
    hard_low_violation = np.where(q < q_low[None, :] - 1e-9)
    hard_high_violation = np.where(q > q_high[None, :] + 1e-9)
    if hard_low_violation[0].size or hard_high_violation[0].size:
        index = int(hard_low_violation[0][0] if hard_low_violation[0].size else hard_high_violation[0][0])
        raise RuntimeError(f"Reference violates a hard joint limit at sample {index}")

    actual_positions, actual_rotations, sigma_min, joint_margins = _compute_fk_trajectory(kinematics, q)
    task_mask = _task_mask(bundle.segment_ids)
    position_errors = np.full(q.shape[0], np.nan, dtype=np.float64)
    orientation_errors = np.full(q.shape[0], np.nan, dtype=np.float64)
    if bundle.task_positions_des is not None:
        position_errors = np.linalg.norm(actual_positions - bundle.task_positions_des, axis=1)
        orientation_errors = np.asarray(
            [
                np.linalg.norm(orientation_error(target, current))
                for target, current in zip(bundle.task_rotations_des, actual_rotations, strict=True)
            ],
            dtype=np.float64,
        )
        if np.any(task_mask):
            max_position_error = float(np.max(position_errors[task_mask]))
            max_orientation_error = float(np.max(orientation_errors[task_mask]))
            if max_position_error > config.ik_config.position_tolerance * 10.0:
                raise RuntimeError(
                    "FK position validation failed: "
                    f"max task error={max_position_error:.6g} m"
                )
            if max_orientation_error > config.ik_config.orientation_tolerance * 10.0:
                raise RuntimeError(
                    "FK orientation validation failed: "
                    f"max task error={max_orientation_error:.6g} rad"
                )
        else:
            max_position_error = 0.0
            max_orientation_error = 0.0
    else:
        max_position_error = float("nan")
        max_orientation_error = float("nan")

    joint_velocity_limits = _broadcast_limits(config.max_joint_velocity, bundle.n_joints, "max_joint_velocity")
    joint_acceleration_limits = _broadcast_limits(config.max_joint_acceleration, bundle.n_joints, "max_joint_acceleration")
    max_velocity = np.max(np.abs(bundle.dq_des), axis=0)
    max_acceleration = np.max(np.abs(bundle.ddq_des), axis=0)
    velocity_violations = np.flatnonzero(max_velocity > joint_velocity_limits + 1e-9)
    acceleration_violations = np.flatnonzero(max_acceleration > joint_acceleration_limits + 1e-9)
    if velocity_violations.size:
        raise RuntimeError(
            "Reference joint velocity limit exceeded for joints "
            f"{velocity_violations.tolist()}: observed={max_velocity.tolist()}"
        )
    if acceleration_violations.size:
        raise RuntimeError(
            "Reference joint acceleration limit exceeded for joints "
            f"{acceleration_violations.tolist()}: observed={max_acceleration.tolist()}"
        )

    jumps = np.abs(np.diff(q, axis=0))
    max_joint_jump = float(np.max(jumps)) if jumps.size else 0.0
    if max_joint_jump > config.max_joint_jump + 1e-12:
        raise RuntimeError(
            "Reference has a discontinuous joint-space step: "
            f"max={max_joint_jump:.6g} rad, allowed={config.max_joint_jump:.6g}"
        )
    task_singular = np.flatnonzero(task_mask & (sigma_min < config.singularity_reject))
    if task_singular.size:
        index = int(task_singular[0])
        raise RuntimeError(
            "Task-space IK segment enters a rejected singularity at "
            f"sample {index}: sigma_min={sigma_min[index]:.6g}"
        )
    warning_indices = np.flatnonzero(sigma_min < config.singularity_warning)

    final_q_error = float(np.max(np.abs(wrap_to_pi(q[bundle.execution_steps - 1] - np.zeros(bundle.n_joints)))))
    final_dq_error = float(np.max(np.abs(bundle.dq_des[bundle.execution_steps - 1])))
    if not config.collection_only and final_q_error > config.final_return_tolerance:
        raise RuntimeError(
            "Reference does not return to zero joint pose: "
            f"max wrapped error={final_q_error:.6g} rad"
        )
    if not config.collection_only and final_dq_error > config.final_return_tolerance:
        raise RuntimeError(
            "Reference does not settle at zero joint velocity: "
            f"max error={final_dq_error:.6g} rad/s"
        )
    if not np.allclose(q[bundle.execution_steps :], q[bundle.execution_steps - 1], atol=config.final_return_tolerance):
        raise RuntimeError("Horizon padding must be an exact terminal joint reference")

    lap_closure: list[dict[str, float | int]] = []
    if bundle.task_positions_des is not None:
        for lap_id in sorted(int(item) for item in np.unique(bundle.lap_ids) if item >= 0):
            indices = np.flatnonzero(bundle.lap_ids == lap_id)
            if indices.size < 2:
                continue
            position_closure = float(
                np.linalg.norm(bundle.task_positions_des[indices[-1]] - bundle.task_positions_des[indices[0]])
            )
            joint_delta = q[indices[-1]] - q[indices[0]]
            joint_closure = float(np.max(np.abs(wrap_to_pi(joint_delta))))
            if position_closure > config.lap_position_closure_tolerance:
                raise RuntimeError(
                    f"Lap {lap_id} does not close in task space: {position_closure:.6g} m exceeds "
                    f"{config.lap_position_closure_tolerance:.6g} m"
                )
            if joint_closure > config.lap_joint_closure_tolerance:
                raise RuntimeError(
                    f"Lap {lap_id} has an IK branch discontinuity: {joint_closure:.6g} rad exceeds "
                    f"{config.lap_joint_closure_tolerance:.6g} rad"
                )
            lap_closure.append(
                {
                    "lap_id": lap_id,
                    "position_closure": position_closure,
                    "joint_closure": joint_closure,
                }
            )

    return {
        "ik_success": bool(bundle.ik_position_errors is None or np.all(np.isfinite(bundle.ik_position_errors[task_mask]))),
        "max_fk_position_error": max_position_error,
        "max_fk_orientation_error": max_orientation_error,
        "max_joint_velocity": max_velocity.tolist(),
        "max_joint_acceleration": max_acceleration.tolist(),
        "max_joint_jump": max_joint_jump,
        "min_sigma_min": float(np.min(sigma_min)),
        "min_task_sigma_min": float(np.min(sigma_min[task_mask])) if np.any(task_mask) else float("nan"),
        "singularity_warning_indices": warning_indices.astype(int).tolist(),
        "min_joint_limit_margin": float(np.min(joint_margins)),
        "final_q_error": final_q_error,
        "final_dq_error": final_dq_error,
        "lap_closure": lap_closure,
        "self_collision_check": "not_available",
    }


def _broadcast_limits(values: Iterable[float], n_joints: int, name: str) -> np.ndarray:
    array = np.asarray(tuple(values), dtype=np.float64)
    if array.shape == (1,):
        array = np.repeat(array, n_joints)
    if array.shape != (n_joints,):
        raise ValueError(f"{name} must have one value or {n_joints} values, got {array.shape}")
    return array


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    return value


def save_reference_bundle(bundle: ReferenceBundle, save_dir: str | Path) -> Path:
    """Persist a reference bundle as ``<save_dir>/reference.npz``."""

    directory = Path(save_dir).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / REFERENCE_FILE_NAME
    arrays: dict[str, np.ndarray] = {
        "time": bundle.time,
        "q_des": bundle.q_des,
        "dq_des": bundle.dq_des,
        "ddq_des": bundle.ddq_des,
        "segment_ids": bundle.segment_ids,
        "lap_ids": bundle.lap_ids,
        "execution_steps": np.asarray(bundle.execution_steps, dtype=np.int64),
        "metadata_json": np.asarray(json.dumps(_json_safe(bundle.metadata), sort_keys=True), dtype=np.str_),
    }
    for name in (
        "task_positions_des",
        "task_rotations_des",
        "ik_position_errors",
        "ik_orientation_errors",
        "ik_iterations",
        "ik_sigma_min",
        "ik_joint_limit_margins",
    ):
        value = getattr(bundle, name)
        if value is not None:
            arrays[name] = value
    np.savez_compressed(path, **arrays)
    return path


def load_reference_bundle(
    path: str | Path,
    *,
    expected_n_joints: int | None = None,
    min_horizon: int | None = None,
) -> ReferenceBundle:
    """Load a saved bundle and verify it can supply every MPC future window."""

    input_path = Path(path).expanduser()
    if input_path.is_dir():
        input_path = input_path / REFERENCE_FILE_NAME
    if not input_path.exists():
        raise FileNotFoundError(f"Reference file does not exist: {input_path}")
    with np.load(input_path, allow_pickle=False) as archive:
        required = {"time", "q_des", "dq_des", "ddq_des", "execution_steps"}
        missing = sorted(required.difference(archive.files))
        if missing:
            raise ValueError(f"Reference file is missing required arrays: {missing}")
        metadata: dict[str, Any] = {}
        if "metadata_json" in archive.files:
            raw_metadata = archive["metadata_json"].item()
            if isinstance(raw_metadata, bytes):
                raw_metadata = raw_metadata.decode("utf-8")
            try:
                metadata = json.loads(str(raw_metadata))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Reference metadata_json is invalid JSON in {input_path}") from exc
        kwargs: dict[str, Any] = {
            "time": archive["time"],
            "q_des": archive["q_des"],
            "dq_des": archive["dq_des"],
            "ddq_des": archive["ddq_des"],
            "execution_steps": int(np.asarray(archive["execution_steps"]).item()),
            "metadata": metadata,
        }
        for name in (
            "task_positions_des",
            "task_rotations_des",
            "segment_ids",
            "lap_ids",
            "ik_position_errors",
            "ik_orientation_errors",
            "ik_iterations",
            "ik_sigma_min",
            "ik_joint_limit_margins",
        ):
            if name in archive.files:
                kwargs[name] = archive[name]
    bundle = ReferenceBundle(**kwargs)
    if expected_n_joints is not None and bundle.n_joints != int(expected_n_joints):
        raise ValueError(
            f"Reference has {bundle.n_joints} joints but expected_n_joints={expected_n_joints}"
        )
    if min_horizon is not None:
        if min_horizon < 0:
            raise ValueError("min_horizon must be non-negative")
        required_length = bundle.execution_steps + int(min_horizon) + 1
        if bundle.reference_length < required_length:
            raise ValueError(
                "Reference is too short for its requested MPC horizon: "
                f"length={bundle.reference_length}, need at least {required_length}"
            )
    return bundle


def reference_summary(bundle: ReferenceBundle) -> dict[str, Any]:
    """Return JSON-ready high-level information for ``summary.json``."""

    summary: dict[str, Any] = {
        "format_version": REFERENCE_FORMAT_VERSION,
        "reference_length": bundle.reference_length,
        "execution_steps": bundle.execution_steps,
        "n_joints": bundle.n_joints,
        "control_dt": float(bundle.time[1] - bundle.time[0]) if bundle.reference_length > 1 else None,
        "shape_name": bundle.metadata.get("shape_name"),
        "segment_counts": {
            SEGMENT_NAMES.get(int(segment), str(int(segment))): int(np.sum(bundle.segment_ids == segment))
            for segment in np.unique(bundle.segment_ids)
        },
        "lap_counts": {str(int(lap)): int(np.sum(bundle.lap_ids == lap)) for lap in np.unique(bundle.lap_ids)},
    }
    validation = bundle.metadata.get("validation")
    if isinstance(validation, dict):
        summary["validation"] = validation
    return _json_safe(summary)
