"""Task-space pose trajectory generation for the ABB IRB2400 references.

This module deliberately has no MuJoCo dependency.  It describes desired TCP
poses only; :mod:`mpc.reference_pipeline` is responsible for turning the poses
into a feasible joint-space reference through continuous inverse kinematics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


# Keep these integer labels stable.  They are persisted in reference.npz and
# are also used by rollout logging to compute per-segment metrics.
SEGMENT_INITIAL_HOLD = 0
SEGMENT_JOINT_DEPARTURE = 1
SEGMENT_TASK_APPROACH = 2
SEGMENT_SHAPE_LOOP = 3
SEGMENT_TASK_RETURN = 4
SEGMENT_JOINT_RETURN = 5
SEGMENT_FINAL_HOLD = 6
SEGMENT_HORIZON_PADDING = 7

SEGMENT_NAMES = {
    SEGMENT_INITIAL_HOLD: "initial_hold",
    SEGMENT_JOINT_DEPARTURE: "joint_departure",
    SEGMENT_TASK_APPROACH: "task_approach",
    SEGMENT_SHAPE_LOOP: "shape_loop",
    SEGMENT_TASK_RETURN: "task_return",
    SEGMENT_JOINT_RETURN: "joint_return",
    SEGMENT_FINAL_HOLD: "final_hold",
    SEGMENT_HORIZON_PADDING: "horizon_padding",
}

SHAPE_NAMES = frozenset({"circle", "ellipse", "figure8", "square", "rounded_square"})


@dataclass
class TaskSpaceTrajectory:
    """A sampled, fixed-orientation TCP trajectory.

    ``lap_ids`` is ``-1`` outside the drawing segment and is zero based for
    samples belonging to a shape lap.  The arrays can contain repeated endpoint
    samples.  Those repetitions intentionally make zero-velocity boundaries
    explicit when a segment is joined to the next one.
    """

    time: np.ndarray
    positions: np.ndarray
    rotations: np.ndarray
    segment_ids: np.ndarray
    lap_ids: np.ndarray
    shape_name: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.time = np.asarray(self.time, dtype=np.float64)
        self.positions = np.asarray(self.positions, dtype=np.float64)
        self.rotations = np.asarray(self.rotations, dtype=np.float64)
        self.segment_ids = np.asarray(self.segment_ids, dtype=np.int64)
        self.lap_ids = np.asarray(self.lap_ids, dtype=np.int64)
        self.shape_name = str(self.shape_name).lower()

        sample_count = self.time.shape[0]
        if sample_count == 0:
            raise ValueError("TaskSpaceTrajectory must contain at least one sample")
        if self.time.ndim != 1:
            raise ValueError(f"time must have shape [N], got {self.time.shape}")
        if self.positions.shape != (sample_count, 3):
            raise ValueError(f"positions must have shape ({sample_count}, 3), got {self.positions.shape}")
        if self.rotations.shape != (sample_count, 3, 3):
            raise ValueError(f"rotations must have shape ({sample_count}, 3, 3), got {self.rotations.shape}")
        if self.segment_ids.shape != (sample_count,):
            raise ValueError(f"segment_ids must have shape ({sample_count},), got {self.segment_ids.shape}")
        if self.lap_ids.shape != (sample_count,):
            raise ValueError(f"lap_ids must have shape ({sample_count},), got {self.lap_ids.shape}")
        if self.shape_name not in SHAPE_NAMES:
            raise ValueError(f"Unknown shape_name {self.shape_name!r}; expected one of {sorted(SHAPE_NAMES)}")
        if not np.all(np.isfinite(self.time)) or not np.all(np.isfinite(self.positions)) or not np.all(np.isfinite(self.rotations)):
            raise ValueError("TaskSpaceTrajectory arrays must be finite")


def quintic_time_scaling(tau: np.ndarray | float) -> np.ndarray:
    """Return the standard zero-velocity, zero-acceleration quintic profile."""

    tau_array = np.asarray(tau, dtype=np.float64)
    tau_clamped = np.clip(tau_array, 0.0, 1.0)
    return 10.0 * tau_clamped**3 - 15.0 * tau_clamped**4 + 6.0 * tau_clamped**5


def _as_vector(name: str, value: np.ndarray | list[float] | tuple[float, ...]) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return array


def orthonormal_plane_axes(
    axis_u: np.ndarray | list[float] | tuple[float, ...],
    axis_v: np.ndarray | list[float] | tuple[float, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate and orthonormalize a task-plane basis.

    A small numerical non-orthogonality is harmless and is removed with a
    Gram-Schmidt step.  Parallel or nearly parallel inputs are rejected because
    they do not define a plane.
    """

    u = _as_vector("plane_axis_u", axis_u)
    v = _as_vector("plane_axis_v", axis_v)
    u_norm = float(np.linalg.norm(u))
    if u_norm <= 1e-12:
        raise ValueError("plane_axis_u must have nonzero norm")
    u = u / u_norm
    v = v - float(np.dot(u, v)) * u
    v_norm = float(np.linalg.norm(v))
    if v_norm <= 1e-8:
        raise ValueError("plane_axis_u and plane_axis_v must not be parallel")
    v = v / v_norm
    normal = np.cross(u, v)
    normal /= np.linalg.norm(normal)
    return u, v, normal


def shape_start_position(
    shape_name: str,
    center: np.ndarray,
    axis_u: np.ndarray,
    axis_v: np.ndarray,
    *,
    circle_radius: float = 0.03,
    ellipse_axis_a: float = 0.04,
    ellipse_axis_b: float = 0.025,
    figure8_axis_a: float = 0.035,
    figure8_axis_b: float = 0.02,
    square_half_side: float = 0.025,
    rounded_square_corner_radius: float = 0.006,
) -> np.ndarray:
    """Return the fixed start point used for one closed shape traversal."""

    name = str(shape_name).lower()
    center = _as_vector("center", center)
    u, v, _ = orthonormal_plane_axes(axis_u, axis_v)
    _validate_shape_dimensions(
        name,
        circle_radius=circle_radius,
        ellipse_axis_a=ellipse_axis_a,
        ellipse_axis_b=ellipse_axis_b,
        figure8_axis_a=figure8_axis_a,
        figure8_axis_b=figure8_axis_b,
        square_half_side=square_half_side,
        rounded_square_corner_radius=rounded_square_corner_radius,
    )
    if name == "circle":
        return center + float(circle_radius) * u
    if name == "ellipse":
        return center + float(ellipse_axis_a) * u
    if name == "figure8":
        # Gerono lemniscate with theta_0 = pi / 2 starts at the outer right end.
        return center + float(figure8_axis_a) * u
    if name == "square":
        return center + float(square_half_side) * (u + v)
    if name == "rounded_square":
        return center + (float(square_half_side) - float(rounded_square_corner_radius)) * u + float(square_half_side) * v
    raise ValueError(f"Unknown shape_name {shape_name!r}; expected one of {sorted(SHAPE_NAMES)}")


def _validate_shape_dimensions(
    shape_name: str,
    *,
    circle_radius: float,
    ellipse_axis_a: float,
    ellipse_axis_b: float,
    figure8_axis_a: float,
    figure8_axis_b: float,
    square_half_side: float,
    rounded_square_corner_radius: float,
) -> None:
    if shape_name not in SHAPE_NAMES:
        raise ValueError(f"Unknown shape_name {shape_name!r}; expected one of {sorted(SHAPE_NAMES)}")
    values = {
        "circle_radius": circle_radius,
        "ellipse_axis_a": ellipse_axis_a,
        "ellipse_axis_b": ellipse_axis_b,
        "figure8_axis_a": figure8_axis_a,
        "figure8_axis_b": figure8_axis_b,
        "square_half_side": square_half_side,
        "rounded_square_corner_radius": rounded_square_corner_radius,
    }
    for key, value in values.items():
        if not np.isfinite(value) or float(value) <= 0.0:
            raise ValueError(f"{key} must be finite and positive, got {value}")
    if shape_name == "rounded_square" and rounded_square_corner_radius >= square_half_side:
        raise ValueError("rounded_square_corner_radius must be smaller than square_half_side")


def _motion_samples(duration: float, control_dt: float) -> int:
    if not np.isfinite(duration) or duration <= 0.0:
        raise ValueError(f"duration must be positive, got {duration}")
    if not np.isfinite(control_dt) or control_dt <= 0.0:
        raise ValueError(f"control_dt must be positive, got {control_dt}")
    return max(1, int(round(float(duration) / float(control_dt))))


def _quintic_line(start: np.ndarray, end: np.ndarray, sample_count: int) -> np.ndarray:
    tau = np.linspace(0.0, 1.0, sample_count + 1, dtype=np.float64)
    scale = quintic_time_scaling(tau)[:, None]
    return start[None, :] + scale * (end - start)[None, :]


def _smooth_shape_positions(
    shape_name: str,
    center: np.ndarray,
    axis_u: np.ndarray,
    axis_v: np.ndarray,
    *,
    repeat_count: int,
    lap_samples: int,
    circle_radius: float,
    ellipse_axis_a: float,
    ellipse_axis_b: float,
    figure8_axis_a: float,
    figure8_axis_b: float,
    square_half_side: float,
    rounded_square_corner_radius: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate the drawing segment and its zero-based lap identifiers."""

    if shape_name == "square":
        return _square_positions(
            center,
            axis_u,
            axis_v,
            repeat_count=repeat_count,
            lap_samples=lap_samples,
            half_side=square_half_side,
        )
    if shape_name == "rounded_square":
        return _rounded_square_positions(
            center,
            axis_u,
            axis_v,
            repeat_count=repeat_count,
            lap_samples=lap_samples,
            half_side=square_half_side,
            corner_radius=rounded_square_corner_radius,
        )

    # Each lap has an explicit initial and final sample.  That gives offline
    # validation a real, discrete closure point for every lap and makes the
    # zero-speed corner between repeated drawings deliberate rather than an
    # unobservable phase-boundary interpolation artifact.
    positions_by_lap: list[np.ndarray] = []
    lap_ids_by_lap: list[np.ndarray] = []
    tau = np.linspace(0.0, 1.0, lap_samples + 1, dtype=np.float64)
    phase = 2.0 * np.pi * quintic_time_scaling(tau)
    for lap_index in range(repeat_count):
        if shape_name == "circle":
            positions = center[None, :] + float(circle_radius) * (
                np.cos(phase)[:, None] * axis_u[None, :] + np.sin(phase)[:, None] * axis_v[None, :]
            )
        elif shape_name == "ellipse":
            positions = center[None, :] + float(ellipse_axis_a) * np.cos(phase)[:, None] * axis_u[None, :]
            positions += float(ellipse_axis_b) * np.sin(phase)[:, None] * axis_v[None, :]
        elif shape_name == "figure8":
            theta = np.pi / 2.0 + phase
            positions = center[None, :] + float(figure8_axis_a) * np.sin(theta)[:, None] * axis_u[None, :]
            positions += float(figure8_axis_b) * np.sin(theta)[:, None] * np.cos(theta)[:, None] * axis_v[None, :]
        else:  # pragma: no cover - _validate_shape_dimensions protects this branch.
            raise ValueError(f"Unsupported shape_name {shape_name!r}")
        positions_by_lap.append(positions)
        lap_ids_by_lap.append(np.full(positions.shape[0], lap_index, dtype=np.int64))
    return np.concatenate(positions_by_lap, axis=0), np.concatenate(lap_ids_by_lap, axis=0)


def _square_positions(
    center: np.ndarray,
    axis_u: np.ndarray,
    axis_v: np.ndarray,
    *,
    repeat_count: int,
    lap_samples: int,
    half_side: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a strict square, stopping smoothly at every corner."""

    edge_samples = max(1, int(round(lap_samples / 4.0)))
    vertices = np.asarray(
        [
            center + half_side * (axis_u + axis_v),
            center + half_side * (-axis_u + axis_v),
            center + half_side * (-axis_u - axis_v),
            center + half_side * (axis_u - axis_v),
            center + half_side * (axis_u + axis_v),
        ],
        dtype=np.float64,
    )
    positions: list[np.ndarray] = []
    lap_ids: list[np.ndarray] = []
    for lap_index in range(repeat_count):
        for edge_index in range(4):
            edge = _quintic_line(vertices[edge_index], vertices[edge_index + 1], edge_samples)
            # Keep both endpoints of every edge, including repeated lap starts.
            # The repeated samples make zero-speed corners explicit in the
            # discrete signal and provide exact per-lap closure samples.
            positions.append(edge)
            lap_ids.append(np.full(edge.shape[0], lap_index, dtype=np.int64))
    return np.concatenate(positions, axis=0), np.concatenate(lap_ids, axis=0)


def _rounded_square_positions(
    center: np.ndarray,
    axis_u: np.ndarray,
    axis_v: np.ndarray,
    *,
    repeat_count: int,
    lap_samples: int,
    half_side: float,
    corner_radius: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a tangent-continuous rounded square with arc-length sampling."""

    h, r = float(half_side), float(corner_radius)
    points_2d: list[np.ndarray] = []

    def line(start: tuple[float, float], stop: tuple[float, float]) -> None:
        points_2d.append(np.linspace(start, stop, 128, endpoint=False, dtype=np.float64))

    def arc(cx: float, cy: float, start: float, stop: float) -> None:
        angle = np.linspace(start, stop, 128, endpoint=False, dtype=np.float64)
        points_2d.append(np.stack([cx + r * np.cos(angle), cy + r * np.sin(angle)], axis=1))

    line((h - r, h), (-h + r, h))
    arc(-h + r, h - r, np.pi / 2.0, np.pi)
    line((-h, h - r), (-h, -h + r))
    arc(-h + r, -h + r, np.pi, 3.0 * np.pi / 2.0)
    line((-h + r, -h), (h - r, -h))
    arc(h - r, -h + r, 3.0 * np.pi / 2.0, 2.0 * np.pi)
    line((h, -h + r), (h, h - r))
    arc(h - r, h - r, 0.0, np.pi / 2.0)
    dense = np.concatenate([*points_2d, points_2d[0][:1]], axis=0)
    distance = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(dense, axis=0), axis=1))])
    tau = np.linspace(0.0, 1.0, lap_samples + 1, dtype=np.float64)
    targets = quintic_time_scaling(tau) * distance[-1]
    sampled = np.stack(
        [np.interp(targets, distance, dense[:, dimension]) for dimension in range(2)], axis=1
    )
    one_lap = center[None, :] + sampled[:, :1] * axis_u[None, :] + sampled[:, 1:] * axis_v[None, :]
    positions = [one_lap.copy() for _ in range(repeat_count)]
    lap_ids = [np.full(one_lap.shape[0], index, dtype=np.int64) for index in range(repeat_count)]
    return np.concatenate(positions, axis=0), np.concatenate(lap_ids, axis=0)


def generate_task_space_trajectory(
    *,
    shape_name: str,
    start_position: np.ndarray | list[float] | tuple[float, ...],
    center: np.ndarray | list[float] | tuple[float, ...],
    plane_axis_u: np.ndarray | list[float] | tuple[float, ...],
    plane_axis_v: np.ndarray | list[float] | tuple[float, ...],
    fixed_rotation: np.ndarray,
    control_dt: float,
    approach_duration: float,
    lap_duration: float,
    return_duration: float,
    repeat_count: int = 3,
    circle_radius: float = 0.03,
    ellipse_axis_a: float = 0.04,
    ellipse_axis_b: float = 0.025,
    figure8_axis_a: float = 0.035,
    figure8_axis_b: float = 0.02,
    square_half_side: float = 0.025,
    rounded_square_corner_radius: float = 0.006,
    include_return: bool = True,
) -> TaskSpaceTrajectory:
    """Build ``approach -> shape repeat_count times -> return`` TCP poses.

    The approach begins at ``start_position`` and ends at the shape's canonical
    start point.  The return starts at the same closed-loop endpoint and returns
    to ``start_position``.  Both use quintic interpolation.  The orientation is
    intentionally fixed for this first task-space reference implementation.
    """

    name = str(shape_name).lower()
    if not isinstance(repeat_count, (int, np.integer)) or int(repeat_count) <= 0:
        raise ValueError(f"repeat_count must be a positive integer, got {repeat_count!r}")
    repeat_count = int(repeat_count)
    _validate_shape_dimensions(
        name,
        circle_radius=circle_radius,
        ellipse_axis_a=ellipse_axis_a,
        ellipse_axis_b=ellipse_axis_b,
        figure8_axis_a=figure8_axis_a,
        figure8_axis_b=figure8_axis_b,
        square_half_side=square_half_side,
        rounded_square_corner_radius=rounded_square_corner_radius,
    )

    start = _as_vector("start_position", start_position)
    center_array = _as_vector("center", center)
    u, v, normal = orthonormal_plane_axes(plane_axis_u, plane_axis_v)
    rotation = np.asarray(fixed_rotation, dtype=np.float64)
    if rotation.shape != (3, 3):
        raise ValueError(f"fixed_rotation must have shape (3, 3), got {rotation.shape}")
    if not np.all(np.isfinite(rotation)):
        raise ValueError("fixed_rotation must be finite")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6) or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6):
        raise ValueError("fixed_rotation must be a proper rotation matrix")

    approach_samples = _motion_samples(approach_duration, control_dt)
    lap_samples = _motion_samples(lap_duration, control_dt)
    return_samples = _motion_samples(return_duration, control_dt)
    shape_start = shape_start_position(
        name,
        center_array,
        u,
        v,
        circle_radius=circle_radius,
        ellipse_axis_a=ellipse_axis_a,
        ellipse_axis_b=ellipse_axis_b,
        figure8_axis_a=figure8_axis_a,
        figure8_axis_b=figure8_axis_b,
        square_half_side=square_half_side,
        rounded_square_corner_radius=rounded_square_corner_radius,
    )

    approach = _quintic_line(start, shape_start, approach_samples)
    loop, loop_lap_ids = _smooth_shape_positions(
        name,
        center_array,
        u,
        v,
        repeat_count=repeat_count,
        lap_samples=lap_samples,
        circle_radius=circle_radius,
        ellipse_axis_a=ellipse_axis_a,
        ellipse_axis_b=ellipse_axis_b,
        figure8_axis_a=figure8_axis_a,
        figure8_axis_b=figure8_axis_b,
        square_half_side=square_half_side,
        rounded_square_corner_radius=rounded_square_corner_radius,
    )
    task_return = _quintic_line(shape_start, start, return_samples) if include_return else np.empty((0, 3), dtype=np.float64)

    positions = np.concatenate([approach, loop, task_return], axis=0)
    segment_ids = np.concatenate(
        [
            np.full(approach.shape[0], SEGMENT_TASK_APPROACH, dtype=np.int64),
            np.full(loop.shape[0], SEGMENT_SHAPE_LOOP, dtype=np.int64),
            np.full(task_return.shape[0], SEGMENT_TASK_RETURN, dtype=np.int64),
        ]
    )
    lap_ids = np.concatenate(
        [
            np.full(approach.shape[0], -1, dtype=np.int64),
            loop_lap_ids,
            np.full(task_return.shape[0], -1, dtype=np.int64),
        ]
    )
    rotations = np.repeat(rotation[None, :, :], positions.shape[0], axis=0)
    metadata: dict[str, Any] = {
        "control_dt": float(control_dt),
        "repeat_count": repeat_count,
        "center": center_array.tolist(),
        "plane_axis_u": u.tolist(),
        "plane_axis_v": v.tolist(),
        "plane_normal": normal.tolist(),
        "fixed_rotation": rotation.tolist(),
        "shape_start_position": shape_start.tolist(),
        "approach_duration": float(approach_duration),
        "lap_duration": float(lap_duration),
        "return_duration": float(return_duration),
        "include_return": bool(include_return),
        "circle_radius": float(circle_radius),
        "ellipse_axis_a": float(ellipse_axis_a),
        "ellipse_axis_b": float(ellipse_axis_b),
        "figure8_axis_a": float(figure8_axis_a),
        "figure8_axis_b": float(figure8_axis_b),
        "square_half_side": float(square_half_side),
        "rounded_square_corner_radius": float(rounded_square_corner_radius),
    }
    return TaskSpaceTrajectory(
        time=np.arange(positions.shape[0], dtype=np.float64) * float(control_dt),
        positions=positions,
        rotations=rotations,
        segment_ids=segment_ids,
        lap_ids=lap_ids,
        shape_name=name,
        metadata=metadata,
    )


# A short alias makes downstream call sites read naturally while keeping the
# longer public name explicit in documentation.
build_task_space_trajectory = generate_task_space_trajectory
