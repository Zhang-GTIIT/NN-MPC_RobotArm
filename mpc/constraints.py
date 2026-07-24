from __future__ import annotations

import torch


def joint_bounds_with_margin(joint_low: torch.Tensor, joint_high: torch.Tensor, joint_limit_margin: float) -> tuple[torch.Tensor, torch.Tensor]:
    low = joint_low + float(joint_limit_margin)
    high = joint_high - float(joint_limit_margin)
    if torch.any(low >= high):
        raise ValueError("joint_limit_margin leaves no valid joint range")
    return low, high


def clip_to_joint_limits(q_ref: torch.Tensor, joint_low: torch.Tensor, joint_high: torch.Tensor, joint_limit_margin: float = 0.0) -> torch.Tensor:
    low, high = joint_bounds_with_margin(joint_low.to(q_ref.device), joint_high.to(q_ref.device), joint_limit_margin)
    return torch.minimum(torch.maximum(q_ref, low), high)


def _batch_joint_vector(
    value: torch.Tensor | float | None,
    reference: torch.Tensor,
    name: str,
    *,
    validate_values: bool = True,
) -> torch.Tensor | None:
    if value is None:
        return None
    vector = torch.as_tensor(value, device=reference.device, dtype=reference.dtype)
    if vector.ndim == 0:
        vector = vector.expand(reference.shape[-1])
    if vector.ndim != 1 or vector.shape[0] != reference.shape[-1]:
        raise ValueError(f"{name} must be a scalar or shape ({reference.shape[-1]},), got {tuple(vector.shape)}")
    if validate_values and (
        not bool(torch.all(torch.isfinite(vector))) or bool(torch.any(vector <= 0))
    ):
        raise ValueError(f"{name} must contain finite positive values")
    return vector.view(1, -1)


def _batch_previous(value: torch.Tensor, reference: torch.Tensor, name: str) -> torch.Tensor:
    previous = value.to(device=reference.device, dtype=reference.dtype)
    if previous.ndim == 1:
        previous = previous.unsqueeze(0).expand(reference.shape[0], -1)
    elif previous.ndim == 2 and previous.shape[0] == 1 and reference.shape[0] > 1:
        previous = previous.expand(reference.shape[0], -1)
    if previous.shape != (reference.shape[0], reference.shape[-1]):
        raise ValueError(f"{name} has incompatible shape {tuple(previous.shape)}")
    return previous


def _limited_position_step(
    requested_target: torch.Tensor,
    previous: torch.Tensor,
    previous_velocity: torch.Tensor,
    control_dt: float,
    velocity_limit: torch.Tensor,
    acceleration_limit: torch.Tensor,
    low: torch.Tensor,
    high: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project one absolute position command onto the command kinematic limits.

    The projection follows the nearest requested target while respecting the
    velocity, acceleration and discrete braking constraints used by the legacy
    acceleration controller.  It is intentionally independent of the action
    parameterisation so nominal references and residual candidates share the
    exact same executable-command definition.
    """
    requested_target = torch.minimum(torch.maximum(requested_target, low), high)
    requested_velocity = (requested_target - previous) / control_dt
    velocity = torch.minimum(
        torch.maximum(requested_velocity, previous_velocity - acceleration_limit * control_dt),
        previous_velocity + acceleration_limit * control_dt,
    )
    velocity = torch.clamp(velocity, min=-velocity_limit, max=velocity_limit)

    # Limit velocity by the distance needed to stop after this position update.
    distance_to_high = torch.clamp(high - previous, min=0.0)
    distance_to_low = torch.clamp(previous - low, min=0.0)
    positive_braking_velocity = torch.sqrt(
        torch.square(acceleration_limit * control_dt) + 2.0 * acceleration_limit * distance_to_high
    ) - acceleration_limit * control_dt
    negative_braking_velocity = torch.sqrt(
        torch.square(acceleration_limit * control_dt) + 2.0 * acceleration_limit * distance_to_low
    ) - acceleration_limit * control_dt
    velocity = torch.minimum(velocity, torch.clamp(positive_braking_velocity, min=0.0))
    velocity = torch.maximum(velocity, -torch.clamp(negative_braking_velocity, min=0.0))

    target = torch.minimum(torch.maximum(previous + velocity * control_dt, low), high)
    return target, (target - previous) / control_dt


def _project_position_command_sequence_core(
    requested_q_ref_sequence: torch.Tensor,
    previous: torch.Tensor,
    previous_velocity: torch.Tensor,
    control_dt: float,
    velocity_limit: torch.Tensor,
    acceleration_limit: torch.Tensor,
    low: torch.Tensor,
    high: torch.Tensor,
) -> torch.Tensor:
    """Pure fixed-shape projection core suitable for full-graph compilation."""
    limited = []
    for step_idx in range(requested_q_ref_sequence.shape[1]):
        target, previous_velocity = _limited_position_step(
            requested_q_ref_sequence[:, step_idx],
            previous,
            previous_velocity,
            control_dt,
            velocity_limit,
            acceleration_limit,
            low,
            high,
        )
        limited.append(target)
        previous = target
    return torch.stack(limited, dim=1)


_COMPILED_POSITION_PROJECTOR = None


def _compiled_position_projector():
    global _COMPILED_POSITION_PROJECTOR
    if _COMPILED_POSITION_PROJECTOR is None:
        _COMPILED_POSITION_PROJECTOR = torch.compile(
            _project_position_command_sequence_core,
            fullgraph=True,
            mode="default",
        )
    return _COMPILED_POSITION_PROJECTOR


def project_position_command_sequence(
    requested_q_ref_sequence: torch.Tensor,
    previous_q_ref: torch.Tensor,
    previous_q_ref_velocity: torch.Tensor,
    control_dt: float,
    velocity_limit: torch.Tensor | float,
    acceleration_limit: torch.Tensor | float,
    joint_low: torch.Tensor,
    joint_high: torch.Tensor,
    joint_limit_margin: float = 0.0,
    backend: str = "eager",
) -> torch.Tensor:
    """Return an executable sequence nearest to absolute position requests.

    ``requested_q_ref_sequence`` has shape ``[batch, horizon, joints]``.  The
    preceding command and velocity are part of the projection, so the first
    element is continuous with the command actually sent by the previous MPC
    cycle.
    """
    if control_dt <= 0:
        raise ValueError("control_dt must be positive")
    if requested_q_ref_sequence.ndim != 3:
        raise ValueError("requested_q_ref_sequence must have shape [batch, horizon, action_dim]")
    if backend not in {"eager", "compiled"}:
        raise ValueError("projection backend must be 'eager' or 'compiled'")
    velocity_limit = _batch_joint_vector(
        velocity_limit, requested_q_ref_sequence, "velocity_limit", validate_values=backend == "eager"
    )
    acceleration_limit = _batch_joint_vector(
        acceleration_limit, requested_q_ref_sequence, "acceleration_limit", validate_values=backend == "eager"
    )
    if velocity_limit is None or acceleration_limit is None:
        raise ValueError("position-command projection requires velocity_limit and acceleration_limit")
    previous = _batch_previous(previous_q_ref, requested_q_ref_sequence, "previous_q_ref")
    previous_velocity = _batch_previous(
        previous_q_ref_velocity, requested_q_ref_sequence, "previous_q_ref_velocity"
    )
    low = joint_low.to(requested_q_ref_sequence.device, requested_q_ref_sequence.dtype) + float(joint_limit_margin)
    high = joint_high.to(requested_q_ref_sequence.device, requested_q_ref_sequence.dtype) - float(joint_limit_margin)
    if backend == "eager" and bool(torch.any(low >= high)):
        raise ValueError("joint_limit_margin leaves no valid joint range")
    projector = (
        _project_position_command_sequence_core
        if backend == "eager"
        else _compiled_position_projector()
    )
    return projector(
        requested_q_ref_sequence,
        previous,
        previous_velocity,
        control_dt,
        velocity_limit,
        acceleration_limit,
        low.view(1, -1),
        high.view(1, -1),
    )


def project_nominal_q_ref_sequence(
    q_des_sequence: torch.Tensor,
    previous_q_ref: torch.Tensor,
    previous_q_ref_velocity: torch.Tensor,
    control_dt: float,
    velocity_limit: torch.Tensor | float,
    acceleration_limit: torch.Tensor | float,
    joint_low: torch.Tensor,
    joint_high: torch.Tensor,
    joint_limit_margin: float = 0.0,
) -> torch.Tensor:
    """Project one ``[horizon, joints]`` desired trajectory into an executable nominal."""
    if q_des_sequence.ndim != 2:
        raise ValueError("q_des_sequence must have shape [horizon, action_dim]")
    return project_position_command_sequence(
        q_des_sequence.unsqueeze(0),
        previous_q_ref,
        previous_q_ref_velocity,
        control_dt,
        velocity_limit,
        acceleration_limit,
        joint_low,
        joint_high,
        joint_limit_margin,
    )[0]


def apply_command_kinematic_limits(
    normalized_acceleration_sequence: torch.Tensor,
    previous_q_ref: torch.Tensor,
    previous_q_ref_velocity: torch.Tensor,
    control_dt: float,
    velocity_limit: torch.Tensor | float | None,
    acceleration_limit: torch.Tensor | float | None,
    joint_low: torch.Tensor,
    joint_high: torch.Tensor,
    joint_limit_margin: float = 0.0,
) -> torch.Tensor:
    """Integrate normalized command accelerations into safe absolute ``q_ref`` commands.

    Each candidate is expressed in the per-joint, dimensionless range ``[-1, 1]``;
    ``+/-1`` means ``+/- acceleration_limit``.  Velocity and acceleration are
    connected to the previously executed command.  Before integrating the next
    command, a braking envelope limits its velocity so that the remaining joint
    distance is sufficient to stop with ``acceleration_limit``.
    """
    if control_dt <= 0:
        raise ValueError("control_dt must be positive")
    if normalized_acceleration_sequence.ndim != 3:
        raise ValueError("normalized_acceleration_sequence must have shape [batch, horizon, action_dim]")
    velocity_limit = _batch_joint_vector(velocity_limit, normalized_acceleration_sequence, "velocity_limit")
    acceleration_limit = _batch_joint_vector(acceleration_limit, normalized_acceleration_sequence, "acceleration_limit")
    if velocity_limit is None or acceleration_limit is None:
        raise ValueError("normalized command acceleration requires velocity_limit and acceleration_limit")
    previous = _batch_previous(previous_q_ref, normalized_acceleration_sequence, "previous_q_ref")
    previous_velocity = _batch_previous(
        previous_q_ref_velocity, normalized_acceleration_sequence, "previous_q_ref_velocity"
    )
    low, high = joint_bounds_with_margin(
        joint_low.to(normalized_acceleration_sequence.device, normalized_acceleration_sequence.dtype),
        joint_high.to(normalized_acceleration_sequence.device, normalized_acceleration_sequence.dtype),
        joint_limit_margin,
    )
    limited = []
    for step_idx in range(normalized_acceleration_sequence.shape[1]):
        normalized_acceleration = torch.clamp(normalized_acceleration_sequence[:, step_idx], min=-1.0, max=1.0)
        requested_acceleration = normalized_acceleration * acceleration_limit
        velocity = previous_velocity + requested_acceleration * control_dt
        velocity = torch.clamp(velocity, min=-velocity_limit, max=velocity_limit)

        # If the next position uses velocity v and then brakes at ``a_max``, its
        # total remaining travel is v*dt + v^2/(2*a_max).  Solving that against
        # the available distance gives the following discrete-time-safe envelope.
        distance_to_high = torch.clamp(high - previous, min=0.0)
        distance_to_low = torch.clamp(previous - low, min=0.0)
        positive_braking_velocity = torch.sqrt(
            torch.square(acceleration_limit * control_dt) + 2.0 * acceleration_limit * distance_to_high
        ) - acceleration_limit * control_dt
        negative_braking_velocity = torch.sqrt(
            torch.square(acceleration_limit * control_dt) + 2.0 * acceleration_limit * distance_to_low
        ) - acceleration_limit * control_dt
        velocity = torch.minimum(velocity, torch.clamp(positive_braking_velocity, min=0.0))
        velocity = torch.maximum(velocity, -torch.clamp(negative_braking_velocity, min=0.0))

        target = torch.minimum(torch.maximum(previous + velocity * control_dt, low), high)
        velocity = (target - previous) / control_dt
        limited.append(target)
        previous = target
        previous_velocity = velocity
    return torch.stack(limited, dim=1)
