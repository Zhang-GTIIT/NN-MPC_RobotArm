"""Immutable values exchanged by the real-time ASAP-MPC threads."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class ASAPPlanPacket:
    plan_id: int
    launch_step: int
    launch_time_ns: int
    activation_step: int
    activation_time_ns: int
    publish_time_ns: int
    residual_sequence: np.ndarray
    predicted_state_sequence: np.ndarray
    planning_time_s: float
    anchor_state: np.ndarray
    selection_mode: str
    selected_cost: float
    q_ref_sequence: np.ndarray = field(default_factory=lambda: np.empty((0, 0), dtype=np.float32))
    requested_residual_sequence: np.ndarray = field(default_factory=lambda: np.empty((0, 0), dtype=np.float32))
    planned_projection_offset_sequence: np.ndarray = field(default_factory=lambda: np.empty((0, 0), dtype=np.float32))

    @property
    def horizon(self) -> int:
        return int(self.residual_sequence.shape[0])

    def index_at(self, step: int) -> int | None:
        index = int(step) - self.activation_step
        return index if 0 <= index < self.horizon else None


@dataclass(frozen=True)
class PlanningSnapshot:
    request_id: int
    launch_step: int
    launch_time_ns: int
    states_history: np.ndarray
    command_history: np.ndarray
    previous_q_ref: np.ndarray
    previous_q_ref_velocity: np.ndarray
    previous_requested_mpc_residual: np.ndarray
    previous_requested_mpc_residual_velocity: np.ndarray
    previous_command_nominal_offset: np.ndarray
    previous_command_nominal_offset_velocity: np.ndarray
    packet_schedule: tuple[ASAPPlanPacket, ...]


@dataclass(frozen=True)
class PlannerResultEvent:
    """One immutable worker result; unlike status fields it is never repeated."""

    result_id: int
    request_id: int
    result_type: str
    reason_code: str
    reason_detail: str
    plan_id: int
    planning_time_s: float
    end_to_end_latency_s: float
    candidate_count: int
    valid_candidate_count: int
    candidate_diagnostics: dict[str, int] = field(default_factory=dict)
