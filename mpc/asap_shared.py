"""Small lock-protected stores used by the ASAP controller and worker."""
from __future__ import annotations

import threading
from typing import Iterable

import numpy as np

from mpc.asap_types import ASAPPlanPacket, PlanningSnapshot


def _copy_packet(packet: ASAPPlanPacket) -> ASAPPlanPacket:
    return ASAPPlanPacket(
        plan_id=packet.plan_id, launch_step=packet.launch_step, launch_time_ns=packet.launch_time_ns,
        activation_step=packet.activation_step, activation_time_ns=packet.activation_time_ns,
        publish_time_ns=packet.publish_time_ns, residual_sequence=packet.residual_sequence.copy(),
        predicted_state_sequence=packet.predicted_state_sequence.copy(), planning_time_s=packet.planning_time_s,
        anchor_state=packet.anchor_state.copy(), selection_mode=packet.selection_mode, selected_cost=packet.selected_cost,
        q_ref_sequence=packet.q_ref_sequence.copy(),
        requested_residual_sequence=packet.requested_residual_sequence.copy(),
    )


def copy_snapshot(snapshot: PlanningSnapshot) -> PlanningSnapshot:
    return PlanningSnapshot(
        request_id=snapshot.request_id, launch_step=snapshot.launch_step, launch_time_ns=snapshot.launch_time_ns,
        states_history=snapshot.states_history.copy(), command_history=snapshot.command_history.copy(),
        previous_q_ref=snapshot.previous_q_ref.copy(), previous_q_ref_velocity=snapshot.previous_q_ref_velocity.copy(),
        previous_executed_residual=snapshot.previous_executed_residual.copy(),
        previous_executed_residual_velocity=snapshot.previous_executed_residual_velocity.copy(),
        packet_schedule=tuple(_copy_packet(packet) for packet in snapshot.packet_schedule),
    )


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
