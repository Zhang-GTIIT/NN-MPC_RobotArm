from __future__ import annotations

from dataclasses import dataclass

import torch

from neural_dynamics.rollout import rollout_dynamics_batch
from mpc.constraints import (
    clip_to_joint_limits,
    apply_command_kinematic_limits,
    project_nominal_q_ref_sequence,
    project_position_command_sequence,
)
from mpc.cost_functions import JointSpaceCostConfig, joint_space_tracking_cost


def construct_actuator_q_ref_sequence(
    candidate_normalized_acceleration: torch.Tensor,
    previous_q_ref: torch.Tensor,
    previous_q_ref_velocity: torch.Tensor,
    joint_low: torch.Tensor,
    joint_high: torch.Tensor,
    joint_limit_margin: float,
    q_ref_velocity_limit: torch.Tensor | float,
    q_ref_acceleration_limit: torch.Tensor | float,
    control_dt: float = 0.01,
) -> torch.Tensor:
    """Map normalized command accelerations to executable absolute position commands."""
    if candidate_normalized_acceleration.ndim != 3:
        raise ValueError(
            "candidate_normalized_acceleration must have shape [batch, horizon, action_dim], "
            f"got {tuple(candidate_normalized_acceleration.shape)}"
        )
    device = candidate_normalized_acceleration.device
    dtype = candidate_normalized_acceleration.dtype
    joint_low = joint_low.to(device=device, dtype=dtype)
    joint_high = joint_high.to(device=device, dtype=dtype)
    previous_q_ref = previous_q_ref.to(device=device, dtype=dtype)

    return apply_command_kinematic_limits(
        normalized_acceleration_sequence=candidate_normalized_acceleration,
        previous_q_ref=previous_q_ref,
        previous_q_ref_velocity=previous_q_ref_velocity,
        control_dt=control_dt,
        velocity_limit=q_ref_velocity_limit,
        acceleration_limit=q_ref_acceleration_limit,
        joint_low=joint_low,
        joint_high=joint_high,
        joint_limit_margin=joint_limit_margin,
    )


def construct_residual_q_ref_sequence(
    candidate_normalized_residual: torch.Tensor,
    nominal_q_ref: torch.Tensor,
    residual_max: torch.Tensor | float,
    previous_q_ref: torch.Tensor,
    previous_q_ref_velocity: torch.Tensor,
    joint_low: torch.Tensor,
    joint_high: torch.Tensor,
    joint_limit_margin: float,
    q_ref_velocity_limit: torch.Tensor | float,
    q_ref_acceleration_limit: torch.Tensor | float,
    control_dt: float = 0.01,
    project_kinematics: bool = True,
    projection_backend: str = "eager",
    enforce_projected_offset_bound: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Map bounded residual candidates to executable q_ref sequences.

    Returns projected commands, bounded requested residuals, projected nominal
    offsets, and a feasibility mask.  Projection lag is not an MPC
    residual-bound violation: a raw IK nominal may itself be unreachable in
    one command step.
    """
    if candidate_normalized_residual.ndim != 3:
        raise ValueError(
            "candidate_normalized_residual must have shape [batch, horizon, action_dim], "
            f"got {tuple(candidate_normalized_residual.shape)}"
        )
    batch_size, horizon, action_dim = candidate_normalized_residual.shape
    nominal = nominal_q_ref.to(
        device=candidate_normalized_residual.device, dtype=candidate_normalized_residual.dtype
    )
    if nominal.ndim == 2:
        nominal = nominal.unsqueeze(0).expand(batch_size, -1, -1)
    elif nominal.ndim == 3 and nominal.shape[0] == 1 and batch_size > 1:
        nominal = nominal.expand(batch_size, -1, -1)
    if nominal.shape != (batch_size, horizon, action_dim):
        raise ValueError(
            "nominal_q_ref must have shape "
            f"({horizon}, {action_dim}) or ({batch_size}, {horizon}, {action_dim}), got {tuple(nominal.shape)}"
        )
    residual_limit = torch.as_tensor(
        residual_max, device=candidate_normalized_residual.device, dtype=candidate_normalized_residual.dtype
    )
    if residual_limit.ndim == 0:
        residual_limit = residual_limit.expand(action_dim)
    if residual_limit.shape != (action_dim,) or not bool(torch.all(torch.isfinite(residual_limit))) or bool(torch.any(residual_limit <= 0)):
        raise ValueError(f"residual_max must contain {action_dim} finite positive values")
    proposed_residual = torch.clamp(candidate_normalized_residual, min=-1.0, max=1.0) * residual_limit.view(1, 1, -1)
    requested = nominal + proposed_residual
    if project_kinematics:
        q_ref_sequences = project_position_command_sequence(
            requested,
            previous_q_ref=previous_q_ref,
            previous_q_ref_velocity=previous_q_ref_velocity,
            control_dt=control_dt,
            velocity_limit=q_ref_velocity_limit,
            acceleration_limit=q_ref_acceleration_limit,
            joint_low=joint_low,
            joint_high=joint_high,
            joint_limit_margin=joint_limit_margin,
            backend=projection_backend,
        )
    else:
        # The delay-aware controller keeps Direct IK as its exact zero-correction
        # baseline.  It constrains only the MPC/feedback correction at execution
        # time, rather than slowing the nominal trajectory towards an old command.
        q_ref_sequences = clip_to_joint_limits(requested, joint_low, joint_high, joint_limit_margin)
    projected_nominal_offset = q_ref_sequences - nominal
    feasible = torch.all(torch.isfinite(q_ref_sequences), dim=(1, 2))
    if enforce_projected_offset_bound:
        feasible = feasible & torch.all(
            torch.abs(projected_nominal_offset) <= residual_limit.view(1, 1, -1) + 1e-5,
            dim=(1, 2),
        )
    return q_ref_sequences, proposed_residual, projected_nominal_offset, feasible


def reanchor_residual_command(
    buffered_residual: torch.Tensor,
    nominal_q_ref: torch.Tensor,
    residual_max: torch.Tensor | float,
    previous_q_ref: torch.Tensor,
    previous_q_ref_velocity: torch.Tensor,
    joint_low: torch.Tensor,
    joint_high: torch.Tensor,
    joint_limit_margin: float,
    q_ref_velocity_limit: torch.Tensor | float,
    q_ref_acceleration_limit: torch.Tensor | float,
    control_dt: float = 0.01,
) -> tuple[torch.Tensor, torch.Tensor, bool]:
    """Re-anchor one cached *executed* residual to the current nominal command.

    Multi-rate MPC caches residual corrections, not old absolute commands.  At
    every 100 Hz actuator update this helper reconstructs and projects the
    command using the latest previous command and kinematic state.  A false
    feasibility result means callers must use the zero-residual nominal path.
    """
    if buffered_residual.ndim != 1 or nominal_q_ref.ndim != 1:
        raise ValueError("buffered_residual and nominal_q_ref must have shape [action_dim]")
    if buffered_residual.shape != nominal_q_ref.shape:
        raise ValueError("buffered_residual and nominal_q_ref must have matching shapes")
    residual_limit = torch.as_tensor(
        residual_max, device=buffered_residual.device, dtype=buffered_residual.dtype
    )
    if residual_limit.ndim == 0:
        residual_limit = residual_limit.expand_as(buffered_residual)
    if residual_limit.shape != buffered_residual.shape or bool(torch.any(residual_limit <= 0)):
        raise ValueError("residual_max must contain one positive limit per action")
    normalized = torch.clamp(buffered_residual / residual_limit, min=-1.0, max=1.0)
    q_ref, _, projected_nominal_offset, feasible = construct_residual_q_ref_sequence(
        normalized.view(1, 1, -1),
        nominal_q_ref=nominal_q_ref.view(1, -1),
        residual_max=residual_limit,
        previous_q_ref=previous_q_ref,
        previous_q_ref_velocity=previous_q_ref_velocity,
        joint_low=joint_low,
        joint_high=joint_high,
        joint_limit_margin=joint_limit_margin,
        q_ref_velocity_limit=q_ref_velocity_limit,
        q_ref_acceleration_limit=q_ref_acceleration_limit,
        control_dt=control_dt,
    )
    return q_ref[0, 0], projected_nominal_offset[0, 0], bool(feasible[0])


@dataclass(frozen=True)
class PlannerRolloutConfig:
    mpc_policy: str = "residual"
    q_ref_velocity_limit: torch.Tensor | float = 1.0
    q_ref_acceleration_limit: torch.Tensor | float = 1.0
    residual_max: torch.Tensor | float | None = None
    joint_limit_margin: float = 0.0
    rollout_batch_size: int | None = None
    project_residual_kinematics: bool = True
    projection_backend: str = "eager"
    projection_strategy: str = "full"
    residual_cost_semantics: str = "requested"
    residual_feasibility_semantics: str = "finite"


@dataclass
class LearnedDynamicsPlanner:
    model: torch.nn.Module
    normalizer: object
    model_type: str
    state_dim: int
    target_mode: str
    control_dt: float
    initial_history: torch.Tensor
    q_des: torch.Tensor
    dq_des: torch.Tensor | None
    nominal_q_ref: torch.Tensor | None
    previous_q_ref: torch.Tensor
    previous_q_ref_velocity: torch.Tensor
    previous_residual: torch.Tensor | None
    previous_residual_velocity: torch.Tensor | None
    joint_low: torch.Tensor
    joint_high: torch.Tensor
    cost_config: JointSpaceCostConfig
    rollout_config: PlannerRolloutConfig

    def nominal_sequence(self) -> torch.Tensor:
        if self.nominal_q_ref is not None:
            return self.nominal_q_ref
        return project_nominal_q_ref_sequence(
            self.q_des,
            previous_q_ref=self.previous_q_ref,
            previous_q_ref_velocity=self.previous_q_ref_velocity,
            control_dt=self.control_dt,
            velocity_limit=self.rollout_config.q_ref_velocity_limit,
            acceleration_limit=self.rollout_config.q_ref_acceleration_limit,
            joint_low=self.joint_low,
            joint_high=self.joint_high,
            joint_limit_margin=self.rollout_config.joint_limit_margin,
        )

    def evaluate(
        self,
        candidate_action: torch.Tensor,
        *,
        project_kinematics_override: bool | None = None,
    ) -> dict[str, torch.Tensor]:
        if self.rollout_config.mpc_policy == "residual":
            if self.rollout_config.residual_max is None:
                raise ValueError("residual MPC requires residual_max")
            project_kinematics = (
                self.rollout_config.project_residual_kinematics
                if project_kinematics_override is None
                else project_kinematics_override
            )
            q_ref_sequences, requested_residual_sequences, projected_nominal_offsets, feasible = construct_residual_q_ref_sequence(
                candidate_action,
                nominal_q_ref=self.nominal_sequence(),
                residual_max=self.rollout_config.residual_max,
                previous_q_ref=self.previous_q_ref,
                previous_q_ref_velocity=self.previous_q_ref_velocity,
                joint_low=self.joint_low,
                joint_high=self.joint_high,
                joint_limit_margin=self.rollout_config.joint_limit_margin,
                q_ref_velocity_limit=self.rollout_config.q_ref_velocity_limit,
                q_ref_acceleration_limit=self.rollout_config.q_ref_acceleration_limit,
                control_dt=self.control_dt,
                project_kinematics=project_kinematics,
                projection_backend=self.rollout_config.projection_backend,
                enforce_projected_offset_bound=self.rollout_config.residual_feasibility_semantics == "projected_bound",
            )
            if self.rollout_config.residual_feasibility_semantics not in {"finite", "projected_bound"}:
                raise ValueError("residual_feasibility_semantics must be 'finite' or 'projected_bound'")
            if self.rollout_config.residual_cost_semantics not in {"requested", "projected_offset"}:
                raise ValueError(
                    "residual_cost_semantics must be 'requested' or 'projected_offset'"
                )
            residual_cost_sequence = (
                requested_residual_sequences
                if self.rollout_config.residual_cost_semantics == "requested"
                else projected_nominal_offsets
            )
        elif self.rollout_config.mpc_policy == "legacy_acceleration":
            q_ref_sequences = construct_actuator_q_ref_sequence(
                candidate_action,
                previous_q_ref=self.previous_q_ref,
                previous_q_ref_velocity=self.previous_q_ref_velocity,
                joint_low=self.joint_low,
                joint_high=self.joint_high,
                joint_limit_margin=self.rollout_config.joint_limit_margin,
                q_ref_velocity_limit=self.rollout_config.q_ref_velocity_limit,
                q_ref_acceleration_limit=self.rollout_config.q_ref_acceleration_limit,
                control_dt=self.control_dt,
            )
            requested_residual_sequences = torch.empty_like(q_ref_sequences)
            projected_nominal_offsets = torch.empty_like(q_ref_sequences)
            residual_cost_sequence = torch.empty_like(q_ref_sequences)
            feasible = torch.ones(q_ref_sequences.shape[0], dtype=torch.bool, device=q_ref_sequences.device)
        else:
            raise ValueError("mpc_policy must be 'residual' or 'legacy_acceleration'")
        pred_states = rollout_dynamics_batch(
            model=self.model,
            normalizer=self.normalizer,
            model_type=self.model_type,
            initial_history=self.initial_history,
            future_q_ref=q_ref_sequences,
            state_dim=self.state_dim,
            target_mode=self.target_mode,
            control_dt=self.control_dt,
            rollout_batch_size=self.rollout_config.rollout_batch_size,
        )
        costs, cost_terms = joint_space_tracking_cost(
            pred_states=pred_states,
            q_des=self.q_des.to(device=pred_states.device, dtype=pred_states.dtype),
            dq_des=None if self.dq_des is None else self.dq_des.to(device=pred_states.device, dtype=pred_states.dtype),
            actuator_q_ref=q_ref_sequences,
            previous_q_ref=self.previous_q_ref.to(device=pred_states.device, dtype=pred_states.dtype),
            previous_q_ref_velocity=self.previous_q_ref_velocity.to(device=pred_states.device, dtype=pred_states.dtype),
            joint_low=self.joint_low.to(device=pred_states.device, dtype=pred_states.dtype),
            joint_high=self.joint_high.to(device=pred_states.device, dtype=pred_states.dtype),
            config=self.cost_config,
            nominal_q_ref=None if self.rollout_config.mpc_policy == "legacy_acceleration" else self.nominal_sequence().to(device=pred_states.device, dtype=pred_states.dtype),
            requested_residual=None if self.rollout_config.mpc_policy == "legacy_acceleration" else requested_residual_sequences,
            residual_cost_sequence=None if self.rollout_config.mpc_policy == "legacy_acceleration" else residual_cost_sequence,
            previous_residual=None if self.previous_residual is None else self.previous_residual.to(device=pred_states.device, dtype=pred_states.dtype),
            previous_residual_velocity=None
            if self.previous_residual_velocity is None
            else self.previous_residual_velocity.to(device=pred_states.device, dtype=pred_states.dtype),
            return_terms=True,
        )
        costs = torch.where(feasible.to(device=costs.device), costs, torch.full_like(costs, float("inf")))
        cost_terms["total"] = costs
        return {
            "costs": costs,
            "cost_terms": cost_terms,
            "q_ref_sequences": q_ref_sequences,
            "residual_sequences": requested_residual_sequences,
            "requested_residual_sequences": requested_residual_sequences,
            "projected_nominal_offsets": projected_nominal_offsets,
            "residual_cost_sequences": residual_cost_sequence,
            "candidate_feasible": feasible,
            "requested_residual_valid": torch.all(torch.isfinite(requested_residual_sequences), dim=(1, 2)),
            "projected_command_valid": torch.all(torch.isfinite(q_ref_sequences), dim=(1, 2)),
            "rollout_valid": torch.all(torch.isfinite(pred_states), dim=(1, 2)),
            "hard_state_constraint_valid": cost_terms["hard_state_constraint_violation"] == 0,
            "cost_valid": torch.isfinite(costs),
            "pred_states": pred_states,
        }

    def evaluate_exact(self, candidate_action: torch.Tensor) -> dict[str, torch.Tensor]:
        """Evaluate candidates with exact physical projection regardless of stage-one mode."""
        return self.evaluate(candidate_action, project_kinematics_override=True)
