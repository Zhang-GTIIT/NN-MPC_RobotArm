"""Small lock-protected stores used by the ASAP controller and worker."""
from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from mpc.asap_types import ASAPPlanPacket, PlannerResultEvent, PlanningSnapshot


@dataclass(frozen=True)
class PacketFallbackTransition:
    state: str
    reason: str
    active: int
    first_packet_activated_event: int
    packet_expired_event: int
    fallback_started_event: int
    fallback_ended_event: int


class PacketFallbackStateMachine:
    """Edge-triggered packet-gap attribution for the 100 Hz control loop."""

    _BLOCKING_RESULTS = frozenset({"planner_failure", "success_late_dropped", "worker_fatal"})

    def __init__(self) -> None:
        self.ever_activated = False
        self.previous_active_plan_id = -1
        self.active_last_tick = False
        self.fallback_last_tick = True
        self.expiration_count = 0
        self.last_blocking_result_type = ""

    def observe_result(self, result_type: str) -> None:
        if result_type in self._BLOCKING_RESULTS:
            self.last_blocking_result_type = result_type

    def update(self, active_plan_id: int | None) -> PacketFallbackTransition:
        active_now = active_plan_id is not None
        normalized_plan_id = -1 if active_plan_id is None else int(active_plan_id)
        new_packet_activated = active_now and normalized_plan_id != self.previous_active_plan_id
        first_packet_activated = active_now and not self.ever_activated
        packet_expired = self.active_last_tick and not active_now
        if first_packet_activated:
            self.ever_activated = True
        if new_packet_activated:
            self.last_blocking_result_type = ""
        if packet_expired:
            self.expiration_count += 1

        if active_now:
            state, reason = "PACKET_ACTIVE", ""
        elif not self.ever_activated:
            state, reason = "STARTUP_NO_PACKET", "startup_waiting_for_first_packet"
        elif self.last_blocking_result_type == "planner_failure":
            state, reason = "GAP_AFTER_PLANNER_FAILURE", "packet_expired_after_planner_failure"
        elif self.last_blocking_result_type == "success_late_dropped":
            state, reason = "GAP_AFTER_LATE_DROP", "packet_expired_after_late_drop"
        elif self.last_blocking_result_type == "worker_fatal":
            state, reason = "WORKER_FATAL_FALLBACK", "worker_fatal_error"
        else:
            state, reason = "PACKET_EXPIRED_WAITING", "packet_expired_waiting_for_replacement"

        fallback_active = not active_now
        transition = PacketFallbackTransition(
            state=state,
            reason=reason,
            active=int(fallback_active),
            first_packet_activated_event=int(first_packet_activated),
            packet_expired_event=int(packet_expired),
            fallback_started_event=int(fallback_active and not self.fallback_last_tick and self.ever_activated),
            fallback_ended_event=int(not fallback_active and self.fallback_last_tick),
        )
        self.active_last_tick = active_now
        self.previous_active_plan_id = normalized_plan_id
        self.fallback_last_tick = fallback_active
        return transition


def _copy_packet(packet: ASAPPlanPacket) -> ASAPPlanPacket:
    return ASAPPlanPacket(
        plan_id=packet.plan_id, launch_step=packet.launch_step, launch_time_ns=packet.launch_time_ns,
        activation_step=packet.activation_step, activation_time_ns=packet.activation_time_ns,
        publish_time_ns=packet.publish_time_ns, residual_sequence=packet.residual_sequence.copy(),
        predicted_state_sequence=packet.predicted_state_sequence.copy(), planning_time_s=packet.planning_time_s,
        anchor_state=packet.anchor_state.copy(), selection_mode=packet.selection_mode, selected_cost=packet.selected_cost,
        q_ref_sequence=packet.q_ref_sequence.copy(),
        requested_residual_sequence=packet.requested_residual_sequence.copy(),
        planned_projection_offset_sequence=packet.planned_projection_offset_sequence.copy(),
    )


def copy_snapshot(snapshot: PlanningSnapshot) -> PlanningSnapshot:
    return PlanningSnapshot(
        request_id=snapshot.request_id, launch_step=snapshot.launch_step, launch_time_ns=snapshot.launch_time_ns,
        states_history=snapshot.states_history.copy(), command_history=snapshot.command_history.copy(),
        previous_q_ref=snapshot.previous_q_ref.copy(), previous_q_ref_velocity=snapshot.previous_q_ref_velocity.copy(),
        previous_requested_mpc_residual=snapshot.previous_requested_mpc_residual.copy(),
        previous_requested_mpc_residual_velocity=snapshot.previous_requested_mpc_residual_velocity.copy(),
        previous_command_nominal_offset=snapshot.previous_command_nominal_offset.copy(),
        previous_command_nominal_offset_velocity=snapshot.previous_command_nominal_offset_velocity.copy(),
        packet_schedule=tuple(_copy_packet(packet) for packet in snapshot.packet_schedule),
    )


class PlannerResultStore:
    """Lossless result-event queue drained by the faster control thread."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._events: list[PlannerResultEvent] = []

    def publish(self, event: PlannerResultEvent) -> None:
        with self._lock:
            self._events.append(event)

    def drain(self) -> list[PlannerResultEvent]:
        with self._lock:
            events = self._events
            self._events = []
        return events


class LatestSnapshotStore:
    """A coalescing mailbox: planning is always based on the newest state."""
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._event = threading.Event()
        self._latest: PlanningSnapshot | None = None

    def publish(self, snapshot: PlanningSnapshot) -> None:
        with self._lock:
            self._latest = copy_snapshot(snapshot)
            self._event.set()

    def wait_for_newer(self, request_id: int, timeout: float | None) -> PlanningSnapshot | None:
        with self._lock:
            latest = self._latest
            self._event.clear()
            if latest is not None and latest.request_id > request_id:
                return copy_snapshot(latest)
        self._event.wait(timeout)
        with self._lock:
            latest = self._latest
            self._event.clear()
            return None if latest is None or latest.request_id <= request_id else copy_snapshot(latest)

    def wake(self) -> None:
        self._event.set()


class PlanPacketStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._active: ASAPPlanPacket | None = None
        self._pending: list[ASAPPlanPacket] = []

    def publish(self, packet: ASAPPlanPacket) -> None:
        with self._lock:
            self._pending.append(_copy_packet(packet))
            self._pending.sort(key=lambda item: (item.activation_step, item.plan_id))

    def activate_due(self, current_step: int, current_time_ns: int) -> ASAPPlanPacket | None:
        with self._lock:
            due = [item for item in self._pending if item.activation_step <= current_step and item.activation_time_ns <= current_time_ns]
            if due:
                newest = max(due, key=lambda item: (item.activation_step, item.plan_id))
                self._active = newest
                self._pending = [item for item in self._pending if item.activation_step > newest.activation_step]
            if self._active is not None and self._active.index_at(current_step) is None:
                self._active = None
            return None if self._active is None else _copy_packet(self._active)

    def schedule(self) -> tuple[ASAPPlanPacket, ...]:
        with self._lock:
            packets: Iterable[ASAPPlanPacket] = (() if self._active is None else (self._active,))
            return tuple(_copy_packet(packet) for packet in (*packets, *self._pending))
