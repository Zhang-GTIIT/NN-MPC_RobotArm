from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class JointSpaceCostConfig:
    """Dimensionless joint-space MPC costs.

    ``cost_mode='residual'`` is the default policy's anchored black-box cost.
    ``legacy`` retains the former absolute-command velocity/acceleration cost
    for reproducible ablations.
    """

    cost_mode: str = "residual"
    w_q: float = 1.0
    w_dq: float = 0.10
    w_residual: float = 0.20
    w_servo: float = 0.05
    w_residual_velocity: float = 0.05
    w_residual_acceleration: float = 0.02
    w_first: float = 0.20
    w_terminal: float = 0.0
    w_joint_limit: float = 10.0
    w_dq_limit: float = 5.0
    q_tracking_scale: torch.Tensor | float | None = None
    dq_tracking_scale: torch.Tensor | float = 0.5
    residual_scale: torch.Tensor | float | None = None
    servo_scale: torch.Tensor | float | None = None
    residual_velocity_scale: torch.Tensor | float | None = None
    residual_acceleration_scale: torch.Tensor | float | None = None
    temporal_discount: float = 0.95
    barrier_max_weight: float = 2.0
    state_velocity_limit: torch.Tensor | float | None = None
    q_amp_fraction: float = 0.2
    q_tol: float = 0.04
    joint_limit_safe_margin: float = 0.08
    joint_limit_temp: float = 0.02
    dq_limit_temp: float = 0.1
    control_dt: float = 0.01
    velocity_cost_mode: str = "track"
    # Legacy absolute-command terms.  They are ignored by residual mode.
    w_qref_velocity: float = 0.05
    w_qref_acceleration: float = 0.02
    qref_velocity_scale: torch.Tensor | float = 1.0
    qref_acceleration_scale: torch.Tensor | float = 5.0
    # The fields below are enabled only by the actuator-aware profile.
    w_tau: float = 0.0
    w_delta_tau: float = 0.0
    actuator_kp: torch.Tensor | float | None = None
    actuator_kd: torch.Tensor | float | None = None
    torque_scale: torch.Tensor | float | None = None
    delta_torque_scale: torch.Tensor | float | None = None


def _match_horizon(target: torch.Tensor, horizon: int) -> torch.Tensor:
    if target.ndim == 2:
        target = target.unsqueeze(0)
    if target.shape[1] < horizon:
        raise ValueError(f"target horizon={target.shape[1]} is shorter than required horizon={horizon}")
    return target[:, :horizon]


def _expand_batch(target: torch.Tensor, batch_size: int) -> torch.Tensor:
    if target.shape[0] == 1 and batch_size > 1:
        return target.expand(batch_size, -1, -1)
    if target.shape[0] != batch_size:
        raise ValueError(f"target batch={target.shape[0]} is incompatible with batch={batch_size}")
    return target


def _joint_vector(value: torch.Tensor | float, reference: torch.Tensor, name: str, *, allow_zero: bool = False) -> torch.Tensor:
    vector = torch.as_tensor(value, device=reference.device, dtype=reference.dtype)
    if vector.ndim == 0:
        vector = vector.expand(reference.shape[-1])
    if vector.ndim != 1 or vector.shape[0] != reference.shape[-1]:
        raise ValueError(f"{name} must be a scalar or have shape ({reference.shape[-1]},), got {tuple(vector.shape)}")
    if not bool(torch.all(torch.isfinite(vector))) or bool(torch.any(vector < 0 if allow_zero else vector <= 0)):
        raise ValueError(f"{name} must contain finite {'non-negative' if allow_zero else 'positive'} values")
    return vector.view(1, 1, -1)


def _previous_vector(value: torch.Tensor | None, batch_size: int, reference: torch.Tensor, name: str) -> torch.Tensor:
    if value is None:
        raise ValueError(f"{name} is required")
    previous = value.to(device=reference.device, dtype=reference.dtype)
    if previous.ndim == 1:
        previous = previous.unsqueeze(0).expand(batch_size, -1)
    elif previous.ndim == 2 and previous.shape[0] == 1 and batch_size > 1:
        previous = previous.expand(batch_size, -1)
    if previous.shape != (batch_size, reference.shape[-1]):
        raise ValueError(f"{name} must have shape ({reference.shape[-1]},) or ({batch_size}, {reference.shape[-1]}), got {tuple(previous.shape)}")
    return previous


def _temporal_weights(reference: torch.Tensor, discount: float) -> torch.Tensor:
    if not 0.0 < discount <= 1.0:
        raise ValueError("temporal_discount must be in (0, 1]")
    horizon = reference.shape[1]
    weights = torch.pow(torch.as_tensor(discount, device=reference.device, dtype=reference.dtype), torch.arange(horizon, device=reference.device, dtype=reference.dtype))
    return weights / weights.sum()


def _weighted_square(value: torch.Tensor, discount: float) -> torch.Tensor:
    weights = _temporal_weights(value, discount).view(1, -1)
    return torch.sum(torch.mean(torch.square(value), dim=2) * weights, dim=1)


def _weighted_mean_plus_max(value: torch.Tensor, discount: float, max_weight: float) -> torch.Tensor:
    weights = _temporal_weights(value, discount).view(1, -1)
    mean = torch.sum(torch.mean(value, dim=2) * weights, dim=1)
    return mean + float(max_weight) * torch.amax(value, dim=(1, 2))


def _zeros(batch_size: int, reference: torch.Tensor) -> torch.Tensor:
    return torch.zeros(batch_size, device=reference.device, dtype=reference.dtype)


def joint_space_tracking_cost(
    pred_states: torch.Tensor,
    q_des: torch.Tensor,
    dq_des: torch.Tensor | None,
    actuator_q_ref: torch.Tensor,
    previous_q_ref: torch.Tensor,
    previous_q_ref_velocity: torch.Tensor,
    joint_low: torch.Tensor,
    joint_high: torch.Tensor,
    config: JointSpaceCostConfig,
    *,
    nominal_q_ref: torch.Tensor | None = None,
    requested_residual: torch.Tensor | None = None,
    residual_cost_sequence: torch.Tensor | None = None,
    previous_residual: torch.Tensor | None = None,
    previous_residual_velocity: torch.Tensor | None = None,
    return_terms: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Score executable command sequences against learned state rollouts.

    ``pred_states[:, k + 1]`` is generated by ``actuator_q_ref[:, k]``.
    Residual servo effort intentionally compares action ``k`` with the state
    *before* that action, ``pred_states[:, k]``.
    """
    if pred_states.ndim != 3:
        raise ValueError(f"pred_states must have shape [batch, horizon + 1, state_dim], got {tuple(pred_states.shape)}")
    if pred_states.shape[-1] % 2 != 0:
        raise ValueError("pred_states state dimension must contain equally sized q and dq vectors")
    if config.control_dt <= 0:
        raise ValueError("control_dt must be positive")
    if config.cost_mode not in {"residual", "legacy"}:
        raise ValueError("cost_mode must be 'residual' or 'legacy'")

    batch_size = pred_states.shape[0]
    n_joints = pred_states.shape[-1] // 2
    horizon = pred_states.shape[1] - 1
    if actuator_q_ref.shape != (batch_size, horizon, n_joints):
        raise ValueError(
            "actuator_q_ref must have shape "
            f"({batch_size}, {horizon}, {n_joints}), got {tuple(actuator_q_ref.shape)}"
        )

    q_pred = pred_states[:, 1:, :n_joints]
    dq_pred = pred_states[:, 1:, n_joints : 2 * n_joints]
    q_pre = pred_states[:, :-1, :n_joints]
    dq_pre = pred_states[:, :-1, n_joints : 2 * n_joints]
    q_target = _expand_batch(_match_horizon(q_des, horizon).to(device=pred_states.device, dtype=pred_states.dtype), batch_size)
    actuator_q_ref = actuator_q_ref.to(device=pred_states.device, dtype=pred_states.dtype)

    if config.q_tracking_scale is None:
        q_amp = q_target.max(dim=1).values - q_target.min(dim=1).values
        q_scale = torch.clamp(float(config.q_amp_fraction) * q_amp, min=float(config.q_tol)).unsqueeze(1)
    else:
        q_scale = _joint_vector(config.q_tracking_scale, q_pred, "q_tracking_scale")
    e_q = (q_pred - q_target) / q_scale
    terms: dict[str, torch.Tensor] = {"q_tracking": _weighted_square(e_q, config.temporal_discount)}

    if config.velocity_cost_mode not in {"track", "damping"}:
        raise ValueError(f"velocity_cost_mode must be 'track' or 'damping', got {config.velocity_cost_mode!r}")
    dq_scale = _joint_vector(config.dq_tracking_scale, dq_pred, "dq_tracking_scale")
    if dq_des is not None and config.velocity_cost_mode == "track":
        dq_target = _expand_batch(_match_horizon(dq_des, horizon).to(device=pred_states.device, dtype=pred_states.dtype), batch_size)
        e_dq = (dq_pred - dq_target) / dq_scale
    else:
        e_dq = dq_pred / dq_scale
    terms["dq_tracking"] = _weighted_square(e_dq, config.temporal_discount)
    terms["terminal"] = torch.mean(torch.square(e_q[:, -1]), dim=1)

    previous_q_ref = _previous_vector(previous_q_ref, batch_size, actuator_q_ref, "previous_q_ref")
    previous_command_velocity = _previous_vector(
        previous_q_ref_velocity, batch_size, actuator_q_ref, "previous_q_ref_velocity"
    )
    zero = _zeros(batch_size, actuator_q_ref)
    terms.update(
        residual=zero,
        servo=zero,
        residual_velocity=zero,
        residual_acceleration=zero,
        first=zero,
        qref_velocity=zero,
        qref_acceleration=zero,
    )

    if config.cost_mode == "residual":
        if nominal_q_ref is None:
            raise ValueError("nominal_q_ref is required for residual cost")
        if requested_residual is None:
            raise ValueError("requested_residual is required for residual cost")
        nominal = _expand_batch(
            _match_horizon(nominal_q_ref, horizon).to(device=pred_states.device, dtype=pred_states.dtype), batch_size
        )
        residual_source = requested_residual if residual_cost_sequence is None else residual_cost_sequence
        residual = residual_source.to(device=pred_states.device, dtype=pred_states.dtype)
        if residual.shape != actuator_q_ref.shape:
            raise ValueError(
                f"requested_residual must have shape {tuple(actuator_q_ref.shape)}, got {tuple(residual.shape)}"
            )
        residual_scale = _joint_vector(config.residual_scale, actuator_q_ref, "residual_scale")
        servo_scale = _joint_vector(config.servo_scale, actuator_q_ref, "servo_scale")
        residual_velocity_scale = _joint_vector(
            config.residual_velocity_scale, actuator_q_ref, "residual_velocity_scale"
        )
        residual_acceleration_scale = _joint_vector(
            config.residual_acceleration_scale, actuator_q_ref, "residual_acceleration_scale"
        )
        previous_residual = _previous_vector(previous_residual, batch_size, actuator_q_ref, "previous_residual")
        previous_residual_velocity = _previous_vector(
            previous_residual_velocity, batch_size, actuator_q_ref, "previous_residual_velocity"
        )
        residual_full = torch.cat([previous_residual.unsqueeze(1), residual], dim=1)
        residual_velocity = (residual_full[:, 1:] - residual_full[:, :-1]) / float(config.control_dt)
        residual_velocity_full = torch.cat([previous_residual_velocity.unsqueeze(1), residual_velocity], dim=1)
        residual_acceleration = (residual_velocity_full[:, 1:] - residual_velocity_full[:, :-1]) / float(config.control_dt)
        terms["residual"] = _weighted_square(residual / residual_scale, config.temporal_discount)
        terms["servo"] = _weighted_square((actuator_q_ref - q_pre) / servo_scale, config.temporal_discount)
        terms["residual_velocity"] = _weighted_square(residual_velocity / residual_velocity_scale, config.temporal_discount)
        terms["residual_acceleration"] = _weighted_square(
            residual_acceleration / residual_acceleration_scale, config.temporal_discount
        )
        terms["first"] = torch.mean(torch.square(residual_velocity[:, 0] / residual_velocity_scale[:, 0]), dim=1)
    else:
        qref_full = torch.cat([previous_q_ref.unsqueeze(1), actuator_q_ref], dim=1)
        command_velocity = (qref_full[:, 1:] - qref_full[:, :-1]) / float(config.control_dt)
        velocity_scale = _joint_vector(config.qref_velocity_scale, actuator_q_ref, "qref_velocity_scale")
        velocity_full = torch.cat([previous_command_velocity.unsqueeze(1), command_velocity], dim=1)
        command_acceleration = (velocity_full[:, 1:] - velocity_full[:, :-1]) / float(config.control_dt)
        acceleration_scale = _joint_vector(config.qref_acceleration_scale, actuator_q_ref, "qref_acceleration_scale")
        terms["qref_velocity"] = _weighted_square(command_velocity / velocity_scale, config.temporal_discount)
        terms["qref_acceleration"] = _weighted_square(command_acceleration / acceleration_scale, config.temporal_discount)

    joint_low = joint_low.to(q_pred.device, q_pred.dtype)
    joint_high = joint_high.to(q_pred.device, q_pred.dtype)
    margin = torch.minimum(q_pred - joint_low, joint_high - q_pred)
    position_barrier = F.softplus((float(config.joint_limit_safe_margin) - margin) / float(config.joint_limit_temp))
    terms["joint_limit"] = _weighted_mean_plus_max(position_barrier, config.temporal_discount, config.barrier_max_weight)
    hard_joint_violation = torch.any((q_pred < joint_low) | (q_pred > joint_high), dim=(1, 2))
    terms["hard_state_constraint_violation"] = hard_joint_violation.to(dtype=pred_states.dtype)
    if config.state_velocity_limit is None:
        terms["dq_limit"] = zero
    else:
        state_velocity_limit = _joint_vector(config.state_velocity_limit, dq_pred, "state_velocity_limit")
        velocity_barrier = F.softplus((torch.abs(dq_pred) - state_velocity_limit) / float(config.dq_limit_temp))
        terms["dq_limit"] = _weighted_mean_plus_max(velocity_barrier, config.temporal_discount, config.barrier_max_weight)

    if config.w_tau != 0.0 or config.w_delta_tau != 0.0:
        if any(value is None for value in (config.actuator_kp, config.actuator_kd, config.torque_scale, config.delta_torque_scale)):
            raise ValueError("actuator-aware cost requires kp, kd, torque_scale and delta_torque_scale")
        kp = _joint_vector(config.actuator_kp, actuator_q_ref, "actuator_kp")
        kd = _joint_vector(config.actuator_kd, actuator_q_ref, "actuator_kd", allow_zero=True)
        torque = kp * (actuator_q_ref - q_pre) - kd * dq_pre
        torque_scale = _joint_vector(config.torque_scale, actuator_q_ref, "torque_scale")
        terms["torque"] = _weighted_square(torque / torque_scale, config.temporal_discount)
        previous_torque = kp[:, 0] * (previous_q_ref - q_pre[:, 0]) - kd[:, 0] * dq_pre[:, 0]
        torque_full = torch.cat([previous_torque.unsqueeze(1), torque], dim=1)
        delta_torque_scale = _joint_vector(config.delta_torque_scale, actuator_q_ref, "delta_torque_scale")
        terms["delta_torque"] = _weighted_square(
            (torque_full[:, 1:] - torque_full[:, :-1]) / delta_torque_scale, config.temporal_discount
        )
    else:
        terms["torque"] = zero
        terms["delta_torque"] = zero

    if config.cost_mode == "residual":
        total = (
            float(config.w_q) * terms["q_tracking"]
            + float(config.w_dq) * terms["dq_tracking"]
            + float(config.w_residual) * terms["residual"]
            + float(config.w_servo) * terms["servo"]
            + float(config.w_residual_velocity) * terms["residual_velocity"]
            + float(config.w_residual_acceleration) * terms["residual_acceleration"]
            + float(config.w_first) * terms["first"]
            + float(config.w_terminal) * terms["terminal"]
            + float(config.w_joint_limit) * terms["joint_limit"]
            + float(config.w_dq_limit) * terms["dq_limit"]
            + float(config.w_tau) * terms["torque"]
            + float(config.w_delta_tau) * terms["delta_torque"]
        )
    else:
        total = (
            float(config.w_q) * terms["q_tracking"]
            + float(config.w_dq) * terms["dq_tracking"]
            + float(config.w_qref_velocity) * terms["qref_velocity"]
            + float(config.w_qref_acceleration) * terms["qref_acceleration"]
            + float(config.w_terminal) * terms["terminal"]
            + float(config.w_joint_limit) * terms["joint_limit"]
            + float(config.w_dq_limit) * terms["dq_limit"]
            + float(config.w_tau) * terms["torque"]
            + float(config.w_delta_tau) * terms["delta_torque"]
        )
    total = torch.where(hard_joint_violation, torch.full_like(total, float("inf")), total)
    terms["total"] = total
    return (total, terms) if return_terms else total
