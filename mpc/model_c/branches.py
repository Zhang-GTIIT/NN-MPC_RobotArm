"""Exact-action counterfactual branch execution for MPC data collection."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from mpc.cost_functions import JointSpaceCostConfig, joint_space_tracking_cost
from neural_dynamics.mujoco_env import MuJoCoArmEnv


@dataclass(frozen=True)
class CounterfactualBranch:
    role_mask: tuple[str, ...]
    planned_q_ref_sequence: np.ndarray
    executed_q_ref_sequence: np.ndarray
    predicted_state_sequence: np.ndarray
    realized_state_sequence: np.ndarray
    predicted_cost: float
    realized_cost: float
    activation_infeasible: bool
    infeasible_reason: str
    actual_activation_state: np.ndarray
    actual_previous_command: np.ndarray
    actual_previous_velocity: np.ndarray


def project_activation_q_ref_sequence(
    requested_q_ref_sequence: np.ndarray,
    previous_command: np.ndarray,
    previous_velocity: np.ndarray,
    joint_low: np.ndarray,
    joint_high: np.ndarray,
    joint_limit_margin: float,
    velocity_limit: np.ndarray,
    acceleration_limit: np.ndarray,
    control_dt: float,
) -> np.ndarray:
    """Project absolute command requests from the *actual* activation state.

    This is deliberately separate from strict exact-action execution.  It
    mirrors the position-command rate projection used by deployment, but does
    not add feedback or replan during a counterfactual branch.
    """
    requested = np.asarray(requested_q_ref_sequence, dtype=np.float32)
    if requested.ndim != 2:
        raise ValueError("requested_q_ref_sequence must have shape [horizon, joints]")
    previous = np.asarray(previous_command, dtype=np.float32).copy()
    velocity = np.asarray(previous_velocity, dtype=np.float32).copy()
    low = np.asarray(joint_low, dtype=np.float32) + float(joint_limit_margin)
    high = np.asarray(joint_high, dtype=np.float32) - float(joint_limit_margin)
    velocity_limit = np.asarray(velocity_limit, dtype=np.float32)
    acceleration_limit = np.asarray(acceleration_limit, dtype=np.float32)
    if any(value.shape != previous.shape for value in (velocity, low, high, velocity_limit, acceleration_limit)):
        raise ValueError("all command vectors must have the same joint dimension")
    result = np.empty_like(requested)
    for index, target in enumerate(requested):
        target = np.clip(target, low, high)
        requested_velocity = (target - previous) / float(control_dt)
        velocity = np.clip(
            requested_velocity,
            velocity - acceleration_limit * float(control_dt),
            velocity + acceleration_limit * float(control_dt),
        )
        velocity = np.clip(velocity, -velocity_limit, velocity_limit)
        command = np.clip(previous + velocity * float(control_dt), low, high)
        velocity = (command - previous) / float(control_dt)
        result[index] = command
        previous = command
    return result.astype(np.float32)


def evaluate_branch_cost(
    *,
    states: np.ndarray,
    actions: np.ndarray,
    q_des: np.ndarray | None,
    dq_des: np.ndarray | None,
    previous_command: np.ndarray,
    previous_velocity: np.ndarray,
    parent_env: MuJoCoArmEnv,
    cost_config: JointSpaceCostConfig | None,
) -> float:
    if q_des is None or cost_config is None:
        return float("nan")
    device = cost_config.q_tracking_scale.device if isinstance(cost_config.q_tracking_scale, torch.Tensor) else torch.device("cpu")
    states_tensor = torch.as_tensor(np.asarray(states, dtype=np.float32), device=device).unsqueeze(0)
    actions_tensor = torch.as_tensor(np.asarray(actions, dtype=np.float32), device=device).unsqueeze(0)
    q_tensor = torch.as_tensor(q_des, dtype=torch.float32, device=device).unsqueeze(0)
    dq_tensor = None if dq_des is None else torch.as_tensor(dq_des, dtype=torch.float32, device=device).unsqueeze(0)
    previous = torch.as_tensor(previous_command, dtype=torch.float32, device=device)
    velocity = torch.as_tensor(previous_velocity, dtype=torch.float32, device=device)
    low, high = torch.as_tensor(parent_env.joint_low, device=device), torch.as_tensor(parent_env.joint_high, device=device)
    return float(joint_space_tracking_cost(
        states_tensor, q_tensor, dq_tensor, actions_tensor, previous, velocity, low, high, cost_config,
        nominal_q_ref=q_tensor[0], requested_residual=actions_tensor - q_tensor,
        previous_residual=torch.zeros_like(previous),
        previous_residual_velocity=torch.zeros_like(previous),
    )[0].detach().cpu())


def execute_open_loop_action_branch(
    *,
    parent_env: MuJoCoArmEnv,
    full_state: np.ndarray,
    planned_q_ref_sequence: np.ndarray,
    executed_q_ref_sequence: np.ndarray,
    predicted_state_sequence: np.ndarray,
    predicted_cost: float,
    role_mask: tuple[str, ...],
    actual_state: np.ndarray,
    previous_command: np.ndarray,
    previous_velocity: np.ndarray,
    q_des: np.ndarray | None = None,
    dq_des: np.ndarray | None = None,
    cost_config: JointSpaceCostConfig | None = None,
) -> CounterfactualBranch:
    """Run a finite open-loop action sequence from a captured MuJoCo state."""
    planned = np.asarray(planned_q_ref_sequence, dtype=np.float32)
    executed = np.asarray(executed_q_ref_sequence, dtype=np.float32)
    if planned.shape != executed.shape:
        raise ValueError("planned and executed q_ref sequences must have equal shape")
    branch_env = MuJoCoArmEnv(
        str(parent_env.model_xml), n_joints=parent_env.n_joints,
        gravity_compensation=parent_env.gravity_compensation, frame_skip=parent_env.frame_skip,
        gravity_compensation_model_xml=str(parent_env.gravity_compensation_model_xml),
        actuator_kp_scale=parent_env.actuator_kp_scale,
        actuator_kd_scale=parent_env.actuator_kd_scale,
    )
    try:
        branch_env.restore_full_state(full_state)
        states = [branch_env.get_state()]
        for action in executed:
            states.append(branch_env.step(action))
        realized = np.asarray(states, dtype=np.float32)
        return CounterfactualBranch(
            role_mask=tuple(role_mask), planned_q_ref_sequence=planned,
            executed_q_ref_sequence=executed.copy(),
            predicted_state_sequence=np.asarray(predicted_state_sequence, dtype=np.float32),
            realized_state_sequence=realized,
            predicted_cost=float(predicted_cost),
            realized_cost=evaluate_branch_cost(
                states=realized, actions=executed, q_des=q_des, dq_des=dq_des,
                previous_command=previous_command, previous_velocity=previous_velocity,
                parent_env=parent_env, cost_config=cost_config,
            ),
            activation_infeasible=False, infeasible_reason="",
            actual_activation_state=np.asarray(actual_state, dtype=np.float32),
            actual_previous_command=np.asarray(previous_command, dtype=np.float32),
            actual_previous_velocity=np.asarray(previous_velocity, dtype=np.float32),
        )
    finally:
        branch_env.close()


def first_action_is_executable(
    action: np.ndarray,
    previous_command: np.ndarray,
    previous_velocity: np.ndarray,
    joint_low: np.ndarray,
    joint_high: np.ndarray,
    joint_limit_margin: float,
    velocity_limit: np.ndarray,
    acceleration_limit: np.ndarray,
    control_dt: float,
) -> tuple[bool, str]:
    action = np.asarray(action, dtype=np.float32)
    lower = np.asarray(joint_low, dtype=np.float32) + float(joint_limit_margin)
    upper = np.asarray(joint_high, dtype=np.float32) - float(joint_limit_margin)
    if np.any(action < lower - 1e-6) or np.any(action > upper + 1e-6):
        return False, "joint_limit"
    velocity = (action - previous_command) / float(control_dt)
    if np.any(np.abs(velocity) > np.asarray(velocity_limit, dtype=np.float32) + 1e-6):
        return False, "velocity"
    acceleration = (velocity - previous_velocity) / float(control_dt)
    if np.any(np.abs(acceleration) > np.asarray(acceleration_limit, dtype=np.float32) + 1e-6):
        return False, "acceleration"
    return True, ""


def execute_exact_action_branch(
    *,
    parent_env: MuJoCoArmEnv,
    full_state: np.ndarray,
    candidate: object,
    actual_state: np.ndarray,
    previous_command: np.ndarray,
    previous_velocity: np.ndarray,
    joint_limit_margin: float,
    velocity_limit: np.ndarray,
    acceleration_limit: np.ndarray,
    q_des: np.ndarray | None = None,
    dq_des: np.ndarray | None = None,
    cost_config: JointSpaceCostConfig | None = None,
) -> CounterfactualBranch:
    """Execute a candidate unchanged from a saved activation state.

    The caller owns ``parent_env``.  This function always uses a new MuJoCo
    instance so neither the online environment nor its controller state can be
    modified by a branch.
    """
    planned = np.asarray(candidate.q_ref_sequence, dtype=np.float32)
    feasible, reason = first_action_is_executable(
        planned[0], np.asarray(previous_command, dtype=np.float32), np.asarray(previous_velocity, dtype=np.float32),
        parent_env.joint_low, parent_env.joint_high, joint_limit_margin,
        velocity_limit, acceleration_limit, parent_env.control_dt,
    )
    if not feasible:
        return CounterfactualBranch(
            role_mask=tuple(candidate.role_mask), planned_q_ref_sequence=planned,
            executed_q_ref_sequence=np.empty((0, parent_env.n_joints), dtype=np.float32),
            predicted_state_sequence=np.asarray(candidate.predicted_state_sequence, dtype=np.float32),
            realized_state_sequence=np.empty((0, 2 * parent_env.n_joints), dtype=np.float32),
            predicted_cost=float(candidate.predicted_cost), activation_infeasible=True, infeasible_reason=reason,
            realized_cost=float("nan"),
            actual_activation_state=np.asarray(actual_state, dtype=np.float32),
            actual_previous_command=np.asarray(previous_command, dtype=np.float32),
            actual_previous_velocity=np.asarray(previous_velocity, dtype=np.float32),
        )
    return execute_open_loop_action_branch(
        parent_env=parent_env, full_state=full_state,
        planned_q_ref_sequence=planned, executed_q_ref_sequence=planned,
        predicted_state_sequence=np.asarray(candidate.predicted_state_sequence, dtype=np.float32),
        predicted_cost=float(candidate.predicted_cost), role_mask=tuple(candidate.role_mask),
        actual_state=actual_state, previous_command=previous_command, previous_velocity=previous_velocity,
        q_des=q_des, dq_des=dq_des, cost_config=cost_config,
    )
