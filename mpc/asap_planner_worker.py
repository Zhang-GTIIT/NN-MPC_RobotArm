"""CUDA-owning background worker for real wall-clock ASAP-MPC."""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from neural_dynamics.rollout import rollout_dynamics_batch
from mpc.constraints import project_nominal_q_ref_sequence
from mpc.asap_shared import LatestSnapshotStore, PlanPacketStore, PlannerResultStore
from mpc.asap_types import ASAPPlanPacket, PlannerResultEvent, PlanningSnapshot
from mpc.cem_controller import CEMMPCConfig, CEMMPCController
from mpc.cost_functions import JointSpaceCostConfig
from mpc.delay_aware import project_executable_command_np
from mpc.history import future_history_tokens, history_tokens
from mpc.planner_rollout import LearnedDynamicsPlanner, PlannerRolloutConfig


def warm_start_shift_for_anchor(mean_anchor_step: int | None, new_anchor_step: int) -> tuple[int, bool]:
    """Return the one explicit CEM shift and whether an index reset is needed."""
    if mean_anchor_step is None:
        return 0, False
    if new_anchor_step <= mean_anchor_step:
        return 0, True
    return new_anchor_step - mean_anchor_step, False


def mean_anchor_after_plan(mean_anchor_step: int | None, new_anchor_step: int, planner_failure: bool) -> int | None:
    """Only a successful CEM call changes the anchor represented by ``mean``."""
    return mean_anchor_step if planner_failure else new_anchor_step


def split_failure_reason(reason: str) -> tuple[str, str]:
    """Split a stable paper-facing code from an optional runtime detail."""
    if not reason:
        return "", ""
    code, separator, detail = reason.partition(":")
    return code, detail if separator else ""


@dataclass(frozen=True)
class PlannerWorkerStatus:
    failure_reason: str
    solve_count: int
    late_drop_count: int
    last_planning_time_s: float
    last_best_cost: float
    last_selected_cost: float
    first_solve_complete_ns: int
    last_solve_complete_ns: int
    last_end_to_end_latency_s: float
    anchor_raw_residual: np.ndarray
    anchor_executed_residual: np.ndarray
    anchor_residual_projection_error: np.ndarray
    anchor_previous_residual_velocity: np.ndarray
    warm_start_shift_steps: int
    mean_anchor_step_before: int
    mean_anchor_step_after: int
    planner_mean_updated: bool
    planner_failure: bool
    packet_late_dropped: bool
    planner_result_id: int
    planner_failure_reason: str
    planner_failure_count: int
    consecutive_planner_failure_count: int
    last_successful_plan_id: int


class ASAPPlannerWorker(threading.Thread):
    """Runs all model and CUDA work; the control thread never touches torch."""
    def __init__(self, args: Any, api: dict[str, Any], snapshots: LatestSnapshotStore, packets: PlanPacketStore, results: PlannerResultStore, stop_event: threading.Event, joint_low: np.ndarray, joint_high: np.ndarray, reference: np.ndarray, dq_reference: np.ndarray, ddq_reference: np.ndarray) -> None:
        super().__init__(name="asap-mpc-planner", daemon=True)
        self.args, self.api, self.snapshots, self.packets, self.results, self.stop_event = args, api, snapshots, packets, results, stop_event
        self.joint_low, self.joint_high = joint_low.astype(np.float32).copy(), joint_high.astype(np.float32).copy()
        self.reference = reference.astype(np.float32).copy()
        self.dq_reference = dq_reference.astype(np.float32).copy()
        self.ddq_reference = ddq_reference.astype(np.float32).copy()
        self.ready = threading.Event()
        self._status_lock = threading.Lock()
        self.failure_reason = ""
        self.control_dt: float | None = None
        self.history_len: int | None = None
        self.solve_count = self.late_drop_count = 0
        self.last_planning_time_s = float("nan")
        self.last_best_cost = float("nan")
        self.last_selected_cost = float("nan")
        self.first_solve_complete_ns = 0
        self.last_solve_complete_ns = 0
        self.last_end_to_end_latency_s = float("nan")
        zeros = np.zeros(args.n_joints, dtype=np.float32)
        self._anchor_raw_residual = zeros.copy()
        self._anchor_executed_residual = zeros.copy()
        self._anchor_residual_projection_error = zeros.copy()
        self._anchor_previous_residual_velocity = zeros.copy()
        self._warm_start_shift_steps = 0
        self._mean_anchor_step_before = -1
        self._mean_anchor_step_after = -1
        self._planner_mean_updated = False
        self._planner_failure = False
        self._packet_late_dropped = False
        self._planner_result_id = -1
        self._planner_failure_reason = ""
        self._planner_failure_count = 0
        self._consecutive_planner_failure_count = 0
        self._last_successful_plan_id = -1

    def status(self) -> PlannerWorkerStatus:
        with self._status_lock:
            return PlannerWorkerStatus(
                self.failure_reason, self.solve_count, self.late_drop_count, self.last_planning_time_s,
                self.last_best_cost, self.last_selected_cost, self.first_solve_complete_ns,
                self.last_solve_complete_ns, self.last_end_to_end_latency_s,
                self._anchor_raw_residual.copy(), self._anchor_executed_residual.copy(),
                self._anchor_residual_projection_error.copy(), self._anchor_previous_residual_velocity.copy(),
                self._warm_start_shift_steps, self._mean_anchor_step_before, self._mean_anchor_step_after,
                self._planner_mean_updated, self._planner_failure, self._packet_late_dropped,
                self._planner_result_id, self._planner_failure_reason,
                self._planner_failure_count, self._consecutive_planner_failure_count,
                self._last_successful_plan_id,
            )

    def _fail(self, reason: str) -> None:
        with self._status_lock:
            self.failure_reason = reason
            self._planner_result_id += 1
            result_id = self._planner_result_id
        self.results.publish(PlannerResultEvent(
            result_id=result_id,
            request_id=-1,
            result_type="worker_fatal",
            reason_code="worker_fatal",
            reason_detail=reason,
            plan_id=-1,
            planning_time_s=float("nan"),
            end_to_end_latency_s=float("nan"),
            candidate_count=0,
            valid_candidate_count=0,
            candidate_diagnostics={},
        ))
        self.ready.set()

    def _packet_residual(self, schedule: tuple[ASAPPlanPacket, ...], step: int, *, requested: bool) -> np.ndarray:
        candidates = [packet for packet in schedule if packet.activation_step <= step and packet.index_at(step) is not None]
        if not candidates:
            return np.zeros(self.args.n_joints, dtype=np.float32)
        selected = max(candidates, key=lambda packet: (packet.activation_step, packet.plan_id))
        index = selected.index_at(step)
        assert index is not None
        sequence = selected.residual_sequence
        if requested and selected.requested_residual_sequence.shape == selected.residual_sequence.shape:
            sequence = selected.requested_residual_sequence
        return sequence[index].astype(np.float32, copy=True)

    def _forecast_anchor(self, snapshot: PlanningSnapshot, bundle: Any, device: torch.device, velocity_limit: np.ndarray, acceleration_limit: np.ndarray) -> tuple[torch.Tensor, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        delay = self.args.anticipation_delay_steps
        previous_command, previous_velocity = snapshot.previous_q_ref.copy(), snapshot.previous_q_ref_velocity.copy()
        actions: list[np.ndarray] = []
        command_nominal_offsets: list[np.ndarray] = []
        payload_residuals: list[np.ndarray] = []
        requested_residuals: list[np.ndarray] = []
        for offset in range(delay):
            step = snapshot.launch_step + offset
            nominal = np.asarray(self.reference[step + 1], dtype=np.float32)
            payload_residual = self._packet_residual(snapshot.packet_schedule, step, requested=False)
            requested_residual = self._packet_residual(snapshot.packet_schedule, step, requested=True)
            execution_nominal = nominal
            if self.args.nominal_command_semantics == "executable_ik":
                execution_nominal, _, _ = project_executable_command_np(
                    nominal, np.zeros(self.args.n_joints, dtype=np.float32), previous_command, previous_velocity,
                    self.joint_low, self.joint_high, self.args.joint_limit_margin,
                    velocity_limit, acceleration_limit, bundle.control_dt,
                )
            command, _, velocity = project_executable_command_np(execution_nominal, payload_residual, previous_command, previous_velocity, self.joint_low, self.joint_high, self.args.joint_limit_margin, velocity_limit, acceleration_limit, bundle.control_dt)
            actions.append(command)
            payload_residuals.append(payload_residual)
            requested_residuals.append(requested_residual)
            command_nominal_offsets.append((command - execution_nominal).astype(np.float32))
            previous_command, previous_velocity = command, velocity.astype(np.float32)
        action_array = np.stack(actions).astype(np.float32)
        history = torch.as_tensor(
            history_tokens(snapshot.states_history, snapshot.command_history, bundle.history_len),
            dtype=torch.float32,
            device=device,
        )
        predicted = rollout_dynamics_batch(model=bundle.model, normalizer=bundle.normalizer, model_type=bundle.model_type, initial_history=history, future_q_ref=torch.as_tensor(action_array, dtype=torch.float32, device=device).unsqueeze(0), state_dim=bundle.state_dim, target_mode=bundle.target_mode, control_dt=bundle.control_dt)[0].detach().cpu().numpy().astype(np.float32)
        future_history = future_history_tokens(
            snapshot.states_history, snapshot.command_history, predicted, action_array, bundle.history_len
        )
        anchor = snapshot.launch_step + delay
        previous_requested_residual = requested_residuals[-1]
        previous_requested_residual_velocity = (
            (requested_residuals[-1] - requested_residuals[-2]) / bundle.control_dt
            if delay > 1 else (requested_residuals[-1] - snapshot.previous_requested_mpc_residual) / bundle.control_dt
        )
        previous_command_offset = command_nominal_offsets[-1]
        previous_command_offset_velocity = (
            (command_nominal_offsets[-1] - command_nominal_offsets[-2]) / bundle.control_dt
            if delay > 1 else (command_nominal_offsets[-1] - snapshot.previous_command_nominal_offset) / bundle.control_dt
        )
        return (
            torch.as_tensor(future_history, dtype=torch.float32, device=device), predicted[-1], previous_command,
            previous_velocity, previous_requested_residual, previous_requested_residual_velocity.astype(np.float32),
            previous_command_offset, previous_command_offset_velocity.astype(np.float32), payload_residuals[-1],
        )

    def run(self) -> None:
        try:
            device = self.api["resolve_device"](self.args.device)
            if device.type != "cuda":
                raise ValueError("threaded_asap requires a CUDA device so the worker exclusively owns GPU operations")
            self.api["set_seed"](self.args.seed)
            bundle = self.api["load_dynamics_bundle"](checkpoint_path=self.api["resolve_runtime_path"](self.args.checkpoint), normalizer_path=self.api["resolve_runtime_path"](self.args.normalizer), model_type=self.args.model_type, n_joints=self.args.n_joints, device=device, history_len=self.args.history_len)
            self.control_dt = float(bundle.control_dt)
            self.history_len = int(bundle.history_len)
            parse = self.api["_parse_joint_vector"]
            velocity_limit = parse(self.args.command_velocity_physical_limit, self.args.n_joints, "command_velocity_physical_limit")
            acceleration_limit = parse(self.args.command_acceleration_physical_limit, self.args.n_joints, "command_acceleration_physical_limit")
            residual_max = parse(self.args.residual_max, self.args.n_joints, "residual_max")
            calibration = self.api["_reference_calibration"](self.reference, self.dq_reference, self.ddq_reference, velocity_limit, acceleration_limit)
            t = lambda value: torch.as_tensor(value, dtype=torch.float32, device=device)
            cost = JointSpaceCostConfig(cost_mode="residual", w_q=self.args.w_q, w_dq=self.args.w_dq, w_residual=self.args.w_residual, w_servo=self.args.w_servo, w_residual_velocity=self.args.w_residual_velocity, w_residual_acceleration=self.args.w_residual_acceleration, w_first=self.args.w_first, w_qref_velocity=self.args.w_qref_velocity, w_qref_acceleration=self.args.w_qref_acceleration, w_terminal=self.args.w_terminal, w_joint_limit=self.args.w_joint_limit, w_dq_limit=self.args.w_dq_limit, q_tracking_scale=t(calibration["q_tracking_scale"]), dq_tracking_scale=t(calibration["dq_tracking_scale"]), residual_scale=t(0.5 * residual_max), servo_scale=t(parse(self.args.servo_scale, self.args.n_joints, "servo_scale")), residual_velocity_scale=t(residual_max / bundle.control_dt), residual_acceleration_scale=t(residual_max / bundle.control_dt**2), qref_velocity_scale=t(velocity_limit), qref_acceleration_scale=t(acceleration_limit), temporal_discount=self.args.temporal_discount, barrier_max_weight=self.args.barrier_max_weight, state_velocity_limit=t(parse(self.args.state_velocity_limit, self.args.n_joints, "state_velocity_limit")), joint_limit_safe_margin=self.args.joint_limit_safe_margin, joint_limit_temp=self.args.joint_limit_temp, dq_limit_temp=self.args.dq_limit_temp, control_dt=bundle.control_dt, velocity_cost_mode=self.args.velocity_cost_mode)
            rollout = PlannerRolloutConfig(mpc_policy="residual", q_ref_velocity_limit=t(velocity_limit), q_ref_acceleration_limit=t(acceleration_limit), residual_max=t(residual_max), joint_limit_margin=self.args.joint_limit_margin, rollout_batch_size=self.args.rollout_batch_size, project_residual_kinematics=self.args.planner_projection == "on" and self.args.planner_projection_strategy == "full", projection_backend=self.args.planner_projection_backend, projection_strategy=self.args.planner_projection_strategy, residual_cost_semantics=self.args.residual_cost_semantics, residual_feasibility_semantics=self.args.residual_feasibility_semantics)
            controller: CEMMPCController | None = None
            last_request = -1
            mean_anchor_step: int | None = None
            last_launch_ns: int | None = None
            plan_id = 0
            while not self.stop_event.is_set():
                snapshot = self.snapshots.wait_for_newer(last_request, timeout=0.01)
                if snapshot is None:
                    continue
                if last_launch_ns is not None and self.args.planner_min_interval_ms > 0:
                    min_interval_ns = int(self.args.planner_min_interval_ms * 1e6)
                    remaining_ns = min_interval_ns - (time.perf_counter_ns() - last_launch_ns)
                    if remaining_ns > 0:
                        time.sleep(remaining_ns / 1e9)
                refreshed = self.snapshots.wait_for_newer(snapshot.request_id, timeout=0.0)
                if refreshed is not None:
                    snapshot = refreshed
                last_request = snapshot.request_id
                if snapshot.launch_step + self.args.anticipation_delay_steps + self.args.horizon >= self.reference.shape[0]:
                    continue
                last_launch_ns = time.perf_counter_ns()
                future_history, anchor_state, anchor_command, anchor_velocity, anchor_requested_residual, anchor_requested_residual_velocity, anchor_command_nominal_offset, anchor_command_nominal_offset_velocity, anchor_payload_residual = self._forecast_anchor(snapshot, bundle, device, velocity_limit, acceleration_limit)
                anchor = snapshot.launch_step + self.args.anticipation_delay_steps
                future_q = t(self.reference[anchor + 1:anchor + 1 + self.args.horizon])
                planner_nominal = future_q
                if self.args.nominal_command_semantics == "executable_ik":
                    planner_nominal = project_nominal_q_ref_sequence(
                        future_q, previous_q_ref=t(anchor_command), previous_q_ref_velocity=t(anchor_velocity),
                        control_dt=bundle.control_dt, velocity_limit=t(velocity_limit), acceleration_limit=t(acceleration_limit),
                        joint_low=t(self.joint_low), joint_high=t(self.joint_high), joint_limit_margin=self.args.joint_limit_margin,
                    )
                previous_cost_residual = anchor_requested_residual if self.args.residual_cost_semantics == "requested" else anchor_command_nominal_offset
                previous_cost_residual_velocity = anchor_requested_residual_velocity if self.args.residual_cost_semantics == "requested" else anchor_command_nominal_offset_velocity
                planner = LearnedDynamicsPlanner(model=bundle.model, normalizer=bundle.normalizer, model_type=bundle.model_type, state_dim=bundle.state_dim, target_mode=bundle.target_mode, control_dt=bundle.control_dt, initial_history=future_history, q_des=future_q, dq_des=t(self.dq_reference[anchor + 1:anchor + 1 + self.args.horizon]), nominal_q_ref=planner_nominal, previous_q_ref=t(anchor_command), previous_q_ref_velocity=t(anchor_velocity), previous_residual=t(previous_cost_residual), previous_residual_velocity=t(previous_cost_residual_velocity), joint_low=t(self.joint_low), joint_high=t(self.joint_high), cost_config=cost, rollout_config=rollout)
                if controller is None:
                    controller = CEMMPCController(CEMMPCConfig(horizon=self.args.horizon, action_dim=self.args.n_joints, num_samples=self.args.num_samples, num_elites=self.args.num_elites, elite_ratio=self.args.elite_ratio, cem_iters=self.args.cem_iters, init_std=self.args.init_std, min_std=self.args.min_std, smoothing_alpha=self.args.smoothing_alpha, temporal_noise_alpha=self.args.temporal_noise_alpha, reset_std_each_step=self.args.reset_std_each_step, uniform_sample_ratio=self.args.uniform_sample_ratio, force_baseline_candidate=True, execute=self.args.cem_execute, seed=self.args.seed, device=str(device), selection_validation="exact_final_pool" if self.args.planner_projection_strategy == "two_stage" else "none"), planner, self.joint_low, self.joint_high)
                    generator_state = controller.generator.get_state()
                    for _ in range(self.args.mpc_warmup_plans):
                        controller.plan(anchor_state, anchor_command)
                    controller.generator.set_state(generator_state)
                    controller.reset()
                    self.ready.set()
                    # This snapshot exists solely to initialise CUDA.  A fresh
                    # timestamp is published by the runner after ``ready``.
                    continue
                else:
                    controller.planner = planner
                mean_anchor_before = mean_anchor_step
                shift, reset_mean_anchor = warm_start_shift_for_anchor(mean_anchor_step, anchor)
                if reset_mean_anchor:
                    controller.reset()
                    mean_anchor_step = None
                result = controller.plan(anchor_state, anchor_command, warm_start_shift_steps=shift)
                mean_anchor_step = mean_anchor_after_plan(mean_anchor_step, anchor, result.failure)
                publish_ns = time.perf_counter_ns()
                activation_ns = snapshot.launch_time_ns + int(self.args.anticipation_delay_steps * bundle.control_dt * 1e9)
                late_dropped = bool(not result.failure and publish_ns >= activation_ns - int(self.args.planner_guard_ms * 1e6))
                reason_code, reason_detail = split_failure_reason(result.failure_reason)
                result_type = "planner_failure" if result.failure else "success_late_dropped" if late_dropped else "success_published"
                result_id = self._planner_result_id + 1
                with self._status_lock:
                    self.solve_count += 1
                    self.last_planning_time_s = float(result.planning_time)
                    self.last_best_cost = float(result.best_cost)
                    self.last_selected_cost = float(result.selected_cost)
                    if self.first_solve_complete_ns == 0:
                        self.first_solve_complete_ns = publish_ns
                    self.last_solve_complete_ns = publish_ns
                    self.last_end_to_end_latency_s = (publish_ns - snapshot.launch_time_ns) / 1e9
                    self._anchor_raw_residual = anchor_payload_residual.copy()
                    self._anchor_executed_residual = anchor_command_nominal_offset.copy()
                    self._anchor_residual_projection_error = (anchor_payload_residual - anchor_command_nominal_offset).copy()
                    self._anchor_previous_residual_velocity = previous_cost_residual_velocity.copy()
                    self._warm_start_shift_steps = shift
                    self._mean_anchor_step_before = -1 if mean_anchor_before is None else mean_anchor_before
                    self._mean_anchor_step_after = -1 if mean_anchor_step is None else mean_anchor_step
                    self._planner_mean_updated = not result.failure
                    self._planner_failure = bool(result.failure)
                    self._packet_late_dropped = late_dropped
                    self.late_drop_count += int(late_dropped)
                    self._planner_result_id = result_id
                    self._planner_failure_reason = result.failure_reason
                    self._planner_failure_count += int(result.failure)
                    self._consecutive_planner_failure_count = (
                        self._consecutive_planner_failure_count + 1 if result.failure else 0
                    )
                self.results.publish(PlannerResultEvent(
                    result_id=result_id,
                    request_id=snapshot.request_id,
                    result_type=result_type,
                    reason_code=reason_code,
                    reason_detail=reason_detail,
                    plan_id=-1 if result.failure or late_dropped else plan_id,
                    planning_time_s=float(result.planning_time),
                    end_to_end_latency_s=(publish_ns - snapshot.launch_time_ns) / 1e9,
                    candidate_count=int(result.candidate_count),
                    valid_candidate_count=int(result.valid_candidate_count),
                    candidate_diagnostics=dict(result.candidate_diagnostics),
                ))
                if result.failure or late_dropped:
                    continue
                requested_residual_sequence = result.selected_residual_sequence.copy()
                planner_nominal_np = planner_nominal.detach().cpu().numpy()
                projected_offset_sequence = (result.selected_q_ref_sequence - planner_nominal_np).astype(np.float32)
                packet_residual_sequence = (
                    requested_residual_sequence
                    if self.args.packet_residual_semantics == "requested"
                    else projected_offset_sequence
                )
                planned_projection_offset = (
                    result.selected_q_ref_sequence - (planner_nominal_np + requested_residual_sequence)
                ).astype(np.float32)
                packet = ASAPPlanPacket(
                    plan_id=plan_id, launch_step=snapshot.launch_step, launch_time_ns=snapshot.launch_time_ns,
                    activation_step=anchor, activation_time_ns=activation_ns, publish_time_ns=publish_ns,
                    residual_sequence=packet_residual_sequence,
                    predicted_state_sequence=result.selected_predicted_state_sequence.copy(),
                    planning_time_s=float(result.planning_time), anchor_state=anchor_state.copy(),
                    selection_mode=result.selection_mode, selected_cost=float(result.selected_cost),
                    q_ref_sequence=result.selected_q_ref_sequence.copy(),
                    requested_residual_sequence=requested_residual_sequence,
                    planned_projection_offset_sequence=planned_projection_offset,
                )
                with self._status_lock:
                    self._last_successful_plan_id = plan_id
                plan_id += 1
                self.packets.publish(packet)
        except Exception as exc:  # never strand the control thread during shutdown
            self._fail(f"planner_worker_error:{type(exc).__name__}:{exc}")
