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
    previous_executed_residual: np.ndarray
    previous_executed_residual_velocity: np.ndarray
    packet_schedule: tuple[ASAPPlanPacket, ...]
