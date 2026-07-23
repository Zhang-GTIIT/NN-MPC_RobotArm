"""MuJoCo-backed dynamics oracle for offline CEM-MPC upper-bound studies."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from mpc.cost_functions import JointSpaceCostConfig, joint_space_tracking_cost
from mpc.planner_rollout import PlannerRolloutConfig, construct_residual_q_ref_sequence


@dataclass
class MuJoCoOraclePlanner:
    """Evaluate CEM candidates by restoring and stepping one MuJoCo clone.

    Every candidate starts from the exact same full-physics snapshot.  The
    command construction and cost are intentionally shared with the learned
    planner, so the rollout dynamics are the only experimental variable.
    """

    env: Any
    anchor_snapshot: np.ndarray
    q_des: torch.Tensor
    dq_des: torch.Tensor | None
    nominal_q_ref: torch.Tensor
    previous_q_ref: torch.Tensor
    previous_q_ref_velocity: torch.Tensor
    previous_residual: torch.Tensor | None
    previous_residual_velocity: torch.Tensor | None
    joint_low: torch.Tensor
    joint_high: torch.Tensor
    cost_config: JointSpaceCostConfig
    rollout_config: PlannerRolloutConfig

    def __post_init__(self) -> None:
        if self.rollout_config.mpc_policy != "residual":
            raise ValueError("MuJoCo oracle supports residual MPC only")
        if self.rollout_config.residual_max is None:
            raise ValueError("MuJoCo oracle residual MPC requires residual_max")
        self.anchor_snapshot = np.asarray(self.anchor_snapshot, dtype=np.float64).copy()

    def nominal_sequence(self) -> torch.Tensor:
        return self.nominal_q_ref

    def evaluate(self, candidate_action: torch.Tensor) -> dict[str, torch.Tensor]:
        q_ref_sequences, residual_sequences, projected_nominal_offsets, feasible = construct_residual_q_ref_sequence(
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
            control_dt=self.cost_config.control_dt,
            project_kinematics=self.rollout_config.project_residual_kinematics,
            enforce_projected_offset_bound=self.rollout_config.residual_feasibility_semantics == "projected_bound",
        )
        if self.rollout_config.residual_cost_semantics not in {"requested", "projected_offset"}:
            raise ValueError("residual_cost_semantics must be 'requested' or 'projected_offset'")
        residual_cost_sequence = (
            residual_sequences
            if self.rollout_config.residual_cost_semantics == "requested"
            else projected_nominal_offsets
        )
        batch_size, horizon, _ = q_ref_sequences.shape
        device, dtype = candidate_action.device, candidate_action.dtype
        predicted = np.zeros((batch_size, horizon + 1, self.env.state_dim), dtype=np.float32)
        simulated = np.ones(batch_size, dtype=bool)
        q_ref_cpu = q_ref_sequences.detach().cpu().numpy()

        for candidate_index in range(batch_size):
            try:
                self.env.restore_full_state(self.anchor_snapshot)
                predicted[candidate_index, 0] = self.env.get_state()
                for horizon_index in range(horizon):
                    predicted[candidate_index, horizon_index + 1] = self.env.step(
                        q_ref_cpu[candidate_index, horizon_index]
                    )
            except (RuntimeError, ValueError):
                # Keep the batch contract intact.  Invalid candidates receive
                # infinite cost below and cannot enter the elite set.
                simulated[candidate_index] = False
                predicted[candidate_index] = 0.0

        pred_states = torch.as_tensor(predicted, device=device, dtype=dtype)
        simulated_t = torch.as_tensor(simulated, device=device, dtype=torch.bool)
        feasible = feasible & simulated_t
        costs, cost_terms = joint_space_tracking_cost(
            pred_states=pred_states,
            q_des=self.q_des.to(device=device, dtype=dtype),
            dq_des=None if self.dq_des is None else self.dq_des.to(device=device, dtype=dtype),
            actuator_q_ref=q_ref_sequences,
            previous_q_ref=self.previous_q_ref.to(device=device, dtype=dtype),
            previous_q_ref_velocity=self.previous_q_ref_velocity.to(device=device, dtype=dtype),
            joint_low=self.joint_low.to(device=device, dtype=dtype),
            joint_high=self.joint_high.to(device=device, dtype=dtype),
            config=self.cost_config,
            nominal_q_ref=self.nominal_sequence().to(device=device, dtype=dtype),
            requested_residual=residual_sequences,
            residual_cost_sequence=residual_cost_sequence,
            previous_residual=None
            if self.previous_residual is None
            else self.previous_residual.to(device=device, dtype=dtype),
            previous_residual_velocity=None
            if self.previous_residual_velocity is None
            else self.previous_residual_velocity.to(device=device, dtype=dtype),
            return_terms=True,
        )
        costs = torch.where(feasible, costs, torch.full_like(costs, float("inf")))
        cost_terms["total"] = costs
        return {
            "costs": costs,
            "cost_terms": cost_terms,
            "q_ref_sequences": q_ref_sequences,
            "residual_sequences": residual_sequences,
            "requested_residual_sequences": residual_sequences,
            "projected_nominal_offsets": projected_nominal_offsets,
            "residual_cost_sequences": residual_cost_sequence,
            "candidate_feasible": feasible,
            "requested_residual_valid": torch.all(torch.isfinite(residual_sequences), dim=(1, 2)),
            "projected_command_valid": torch.all(torch.isfinite(q_ref_sequences), dim=(1, 2)),
            "rollout_valid": simulated_t & torch.all(torch.isfinite(pred_states), dim=(1, 2)),
            "hard_state_constraint_valid": cost_terms["hard_state_constraint_violation"] == 0,
            "cost_valid": torch.isfinite(costs),
            "pred_states": pred_states,
        }
