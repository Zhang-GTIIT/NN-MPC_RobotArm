"""Real wall-clock, dual-thread ASAP-MPC rollout runner."""
from __future__ import annotations

import threading
import time
from typing import Any

import numpy as np

from mpc.asap_planner_worker import ASAPPlannerWorker
from mpc.asap_shared import LatestSnapshotStore, PacketFallbackStateMachine, PlanPacketStore, PlannerResultStore
from mpc.asap_types import PlanningSnapshot
from mpc.delay_aware import feedback_correction, project_executable_command_np
from mpc.history import commit_command_and_append_placeholder
from mpc.asap_timing import control_timing_sample


def run(args: Any, api: dict[str, Any]) -> dict[str, Any]:
    if args.controller_mode != "mpc" or args.mpc_policy != "residual":
        raise ValueError("threaded_asap requires --controller_mode mpc --mpc_policy residual")
    if args.visualize:
        raise ValueError("--visualize is not supported by threaded_asap")
    if not args.checkpoint or not args.normalizer:
        raise ValueError("--checkpoint and --normalizer are required by threaded_asap")
    if args.anticipation_delay_steps <= 0:
        raise ValueError("anticipation_delay_steps must be positive")
    if args.asap_history_mode not in {"aligned", "legacy_shifted"}:
        raise ValueError("asap_history_mode must be aligned or legacy_shifted")
    if args.asap_snapshot_mode not in {"tick_start", "post_step_legacy"}:
        raise ValueError("asap_snapshot_mode must be tick_start or post_step_legacy")
    robustness = api["_robustness_config"](args)
    env = api["_build_control_env"](args)
    stop = threading.Event()
    snapshots, packets, planner_results = LatestSnapshotStore(), PlanPacketStore(), PlannerResultStore()
    worker: ASAPPlannerWorker | None = None
    try:
        true_state = env.reset_to_configuration(api["MPC_HOME_Q"][: args.n_joints])
        previous_command = np.asarray(true_state[: args.n_joints], dtype=np.float32).copy()
        previous_velocity = np.zeros(args.n_joints, dtype=np.float32)
        for _ in range(args.settle_steps):
            true_state = env.step(previous_command)
        state = env.get_observation()
        reference, dq_reference, ddq_reference, execution_steps, task_reference = api["_reference_for_run"](args=args, state=true_state, env=env, control_dt=env.control_dt)
        if args.max_execution_steps is not None:
            execution_steps = min(execution_steps, args.max_execution_steps)
        execution_steps = min(execution_steps, reference.shape[0] - args.horizon - args.anticipation_delay_steps - 1)
        if execution_steps <= 0:
            raise ValueError("reference is too short for threaded ASAP horizon plus anticipation delay")
        parse = api["_parse_joint_vector"]
        physical_v = parse(args.command_velocity_physical_limit, args.n_joints, "command_velocity_physical_limit")
        physical_a = parse(args.command_acceleration_physical_limit, args.n_joints, "command_acceleration_physical_limit")
        residual_max = parse(args.residual_max, args.n_joints, "residual_max")
        feedback_max = parse(args.feedback_max, args.n_joints, "feedback_max")
        states_history, command_history = [state.copy()], [previous_command.copy()]
        previous_requested_mpc_residual = np.zeros(args.n_joints, dtype=np.float32)
        previous_requested_mpc_residual_velocity = np.zeros(args.n_joints, dtype=np.float32)
        previous_command_nominal_offset = np.zeros(args.n_joints, dtype=np.float32)
        previous_command_nominal_offset_velocity = np.zeros(args.n_joints, dtype=np.float32)
        request_id = 0
        snapshot_history_len = 1

        def publish_snapshot(launch_step: int, launch_time_ns: int) -> None:
            nonlocal request_id
            entries_s = np.stack(states_history[-snapshot_history_len:]).astype(np.float32, copy=True)
            entries_u = np.stack(command_history[-snapshot_history_len:]).astype(np.float32, copy=True)
            snapshots.publish(PlanningSnapshot(request_id=request_id, launch_step=launch_step, launch_time_ns=launch_time_ns, states_history=entries_s, command_history=entries_u, previous_q_ref=previous_command.copy(), previous_q_ref_velocity=previous_velocity.copy(), previous_requested_mpc_residual=previous_requested_mpc_residual.copy(), previous_requested_mpc_residual_velocity=previous_requested_mpc_residual_velocity.copy(), previous_command_nominal_offset=previous_command_nominal_offset.copy(), previous_command_nominal_offset_velocity=previous_command_nominal_offset_velocity.copy(), packet_schedule=packets.schedule()))
            request_id += 1

        publish_snapshot(0, time.perf_counter_ns())
        worker = ASAPPlannerWorker(args, api, snapshots, packets, planner_results, stop, env.joint_low, env.joint_high, reference, dq_reference, ddq_reference)
        worker.start()
        if not worker.ready.wait(timeout=60.0):
            raise RuntimeError("ASAP planner worker did not become ready within 60 seconds")
        worker_status = worker.status()
        if worker_status.failure_reason:
            raise RuntimeError(worker_status.failure_reason)
        if worker.control_dt is None or not np.isclose(worker.control_dt, env.control_dt, atol=1e-9, rtol=0.0):
            raise ValueError(f"dynamics control_dt {worker.control_dt} must equal MuJoCo control_dt {env.control_dt}")
        if worker.history_len is None:
            raise RuntimeError("ASAP planner worker did not report a history length")
        snapshot_history_len = worker.history_len
        if args.asap_snapshot_mode == "post_step_legacy":
            # Reproduce the former runner's initial live request. This mode is
            # retained only to isolate snapshot phase in the 2x2 ablation.
            publish_snapshot(0, time.perf_counter_ns())
        keys = "actual_states observed_states observation_noise next_states q_des dq_des actuator_q_ref delta_q_ref command_velocity command_acceleration planning_time replan_time mpc_replanned replan_deadline_miss control_step_wall_time control_deadline_miss control_lateness_s actual_control_period_s control_wakeup_lateness_s control_start_jitter_s planner_solve_completion_elapsed_s planner_end_to_end_latency_s buffer_index buffer_length best_cost mean_cost baseline_cost selected_cost elite_mean_cost selection_mode failure_flags joint_limit_violation_flags command_velocity_violation_flags command_acceleration_violation_flags realized_tracking_error nominal_q_ref execution_nominal_q_ref planner_requested_residual buffered_residual requested_mpc_residual feedback_raw feedback_correction requested_feedback_correction requested_correction requested_total_correction requested_absolute_command executed_residual command_nominal_offset safety_projection_offset projection_discrepancy residual_saturated feedback_saturated projection_active predicted_feedback_state planned_q_ref planner_execution_qref_error packet_age packet_event fallback_state fallback_active fallback_reason first_packet_activated_event packet_expired_event fallback_started_event fallback_ended_event planner_result_event_count planner_result_id planner_result_type planner_failure_event planner_failure_reason tau_actuator tau_gravity tau_total tau_gravity_true tau_gravity_mismatch external_force_world external_generalized_force worker_failure worker_solve_count worker_late_drop_count anchor_raw_residual anchor_executed_residual anchor_residual_projection_error anchor_previous_residual_velocity warm_start_shift_steps mean_anchor_step_before mean_anchor_step_after planner_mean_updated planner_failure packet_late_dropped".split()
        if task_reference is not None:
            keys.extend("desired_ee_positions desired_ee_rotations actual_ee_positions actual_ee_rotations ee_position_errors ee_orientation_errors segment_ids lap_ids".split())
        rec: dict[str, list[Any]] = {key: [] for key in keys}
        rows: list[dict[str, Any]] = []
        planner_event_rows: list[dict[str, Any]] = []
        last_seen_solve_count = 0
        fallback_machine = PacketFallbackStateMachine()
        control_epoch_ns = time.perf_counter_ns()
        next_deadline = time.perf_counter()
        previous_tick_start: float | None = None
        for step in range(execution_steps):
            started = time.perf_counter()
            started_ns = time.perf_counter_ns()
            actual_period, wakeup_lateness, start_jitter = control_timing_sample(
                started, previous_tick_start, next_deadline, env.control_dt
            )
            previous_tick_start = started
            if args.asap_snapshot_mode == "tick_start":
                # x_k only becomes available at the start of its control tick.
                publish_snapshot(step, started_ns)
            result_events = planner_results.drain()
            for result_event in result_events:
                planner_event_rows.append({
                    **result_event.__dict__,
                    "observed_control_step": step,
                })
                fallback_machine.observe_result(result_event.result_type)
            active = packets.activate_due(step, started_ns)
            event = ""
            age = -1
            plan_residual = np.zeros(args.n_joints, dtype=np.float32)
            planner_requested = np.zeros(args.n_joints, dtype=np.float32)
            predicted = np.full(2 * args.n_joints, np.nan, dtype=np.float32)
            planned_q_ref = np.full(args.n_joints, np.nan, dtype=np.float32)
            if active is not None:
                index = active.index_at(step)
                if index is not None:
                    age = index
                    plan_residual = active.residual_sequence[index].copy()
                    requested_sequence = active.requested_residual_sequence if active.requested_residual_sequence.shape == active.residual_sequence.shape else active.residual_sequence
                    planner_requested = requested_sequence[index].copy()
                    predicted = active.predicted_state_sequence[index].copy()
                    if active.q_ref_sequence.shape == active.residual_sequence.shape:
                        planned_q_ref = active.q_ref_sequence[index].copy()
                    event = "packet_active"
            active_now = age >= 0
            active_plan_id = -1 if not active_now or active is None else int(active.plan_id)
            fallback = fallback_machine.update(None if active_plan_id < 0 else active_plan_id)
            fallback_state = fallback.state
            fallback_reason = fallback.reason
            fallback_active = fallback.active
            first_packet_activated_event = fallback.first_packet_activated_event
            packet_expired_event = fallback.packet_expired_event
            fallback_started_event = fallback.fallback_started_event
            fallback_ended_event = fallback.fallback_ended_event
            feedback_raw = np.zeros(args.n_joints, dtype=np.float32)
            if age >= 0:
                feedback_raw = (
                    float(args.feedback_kq) * (predicted[: args.n_joints] - state[: args.n_joints])
                    + float(args.feedback_kdq) * (predicted[args.n_joints :] - state[args.n_joints :])
                ).astype(np.float32)
            feedback = np.clip(feedback_raw, -feedback_max, feedback_max).astype(np.float32)
            nominal = np.asarray(reference[step + 1], dtype=np.float32)
            execution_nominal = nominal.copy()
            if args.nominal_command_semantics == "executable_ik":
                execution_nominal, _, _ = project_executable_command_np(
                    nominal, np.zeros(args.n_joints, dtype=np.float32), previous_command, previous_velocity,
                    env.joint_low, env.joint_high, args.joint_limit_margin, physical_v, physical_a, env.control_dt,
                )
            requested_correction = np.clip(plan_residual + feedback, -residual_max - feedback_max, residual_max + feedback_max).astype(np.float32)
            requested_absolute_command = (execution_nominal + requested_correction).astype(np.float32)
            command, _, velocity = project_executable_command_np(execution_nominal, requested_correction, previous_command, previous_velocity, env.joint_low, env.joint_high, args.joint_limit_margin, physical_v, physical_a, env.control_dt)
            command_nominal_offset = (command - nominal).astype(np.float32)
            safety_projection_offset = (command - requested_absolute_command).astype(np.float32)
            projection_discrepancy = (-safety_projection_offset).astype(np.float32)
            planner_execution_qref_error = (
                command - planned_q_ref if np.all(np.isfinite(planned_q_ref)) else np.full(args.n_joints, np.nan, dtype=np.float32)
            ).astype(np.float32)
            acceleration = (velocity - previous_velocity) / env.control_dt
            torque = env.compute_torque_components(command)
            current_state = true_state.copy()
            current_observation = state.copy()
            if task_reference is not None:
                desired_position = np.asarray(task_reference.task_positions_des[step], dtype=np.float32)
                desired_rotation = np.asarray(task_reference.task_rotations_des[step], dtype=np.float32)
                actual_position, actual_rotation = api["site_pose"](env.model, env.data, args.ee_site_name)
                actual_position = np.asarray(actual_position, dtype=np.float32)
                actual_rotation = np.asarray(actual_rotation, dtype=np.float32)
                rec["desired_ee_positions"].append(desired_position)
                rec["desired_ee_rotations"].append(desired_rotation)
                rec["actual_ee_positions"].append(actual_position)
                rec["actual_ee_rotations"].append(actual_rotation)
                rec["ee_position_errors"].append(float(np.linalg.norm(actual_position - desired_position)))
                rec["ee_orientation_errors"].append(api["_orientation_error"](desired_rotation, actual_rotation))
                rec["segment_ids"].append(int(task_reference.segment_ids[step]))
                rec["lap_ids"].append(int(task_reference.lap_ids[step]))
            external_force = api["_force_for_step"](robustness, step, execution_steps)
            true_state = env.step(command, external_force_world=external_force)
            state = env.get_observation()
            if args.asap_history_mode == "aligned":
                commit_command_and_append_placeholder(states_history, command_history, command, state)
            else:
                # Former behaviour: pair x_{k+1} with u_k. This is intentionally
                # available only as an ablation baseline, not a deployment mode.
                states_history.append(state.copy())
                command_history.append(command.copy())
            previous_requested_mpc_residual_velocity = (planner_requested - previous_requested_mpc_residual) / env.control_dt
            previous_requested_mpc_residual = planner_requested.copy()
            previous_command_nominal_offset_velocity = (
                (command - execution_nominal) - previous_command_nominal_offset
            ) / env.control_dt
            previous_command_nominal_offset = (command - execution_nominal).astype(np.float32)
            previous_command, previous_velocity = command.copy(), velocity.astype(np.float32)
            if args.asap_snapshot_mode == "post_step_legacy":
                publish_snapshot(step + 1, time.perf_counter_ns())
            worker_status = worker.status()
            replanned = int(worker_status.solve_count > last_seen_solve_count)
            if replanned:
                last_seen_solve_count = worker_status.solve_count
            planning_time = worker_status.last_planning_time_s if replanned else float("nan")
            best_cost = worker_status.last_best_cost if replanned else float("nan")
            selected_cost = worker_status.last_selected_cost if replanned else float("nan")
            planner_complete_elapsed = (
                (worker_status.last_solve_complete_ns - control_epoch_ns) / 1e9 if replanned else float("nan")
            )
            planner_end_to_end_latency = worker_status.last_end_to_end_latency_s if replanned else float("nan")
            replan_deadline_miss = int(replanned and worker_status.packet_late_dropped)
            control_compute_time = time.perf_counter() - started
            next_deadline += env.control_dt
            remaining = next_deadline - time.perf_counter()
            lateness = max(0.0, -remaining)
            deadline_miss = int(remaining <= 0.0)
            if remaining > 0.0:
                time.sleep(remaining)
            tracking = float(np.linalg.norm(true_state[:args.n_joints] - reference[step + 1]))
            latest_result_event = result_events[-1] if result_events else None
            planner_failure_event = int(any(item.result_type == "planner_failure" for item in result_events))
            acceleration_tolerance = 1e-3
            rec["actual_states"].append(current_state); rec["observed_states"].append(current_observation); rec["observation_noise"].append((current_observation - current_state).astype(np.float32)); rec["next_states"].append(true_state.copy()); rec["external_force_world"].append(env.last_external_force_world.astype(np.float32).copy()); rec["external_generalized_force"].append(env.last_external_generalized_force.astype(np.float32).copy()); rec["q_des"].append(reference[step].copy()); rec["dq_des"].append(dq_reference[step].copy())
            rec["actuator_q_ref"].append(command); rec["delta_q_ref"].append(command - (command - velocity * env.control_dt)); rec["command_velocity"].append(velocity); rec["command_acceleration"].append(acceleration)
            rec["planning_time"].append(planning_time); rec["replan_time"].append(planning_time); rec["mpc_replanned"].append(replanned); rec["replan_deadline_miss"].append(replan_deadline_miss); rec["control_step_wall_time"].append(control_compute_time); rec["control_deadline_miss"].append(deadline_miss); rec["control_lateness_s"].append(lateness); rec["actual_control_period_s"].append(actual_period); rec["control_wakeup_lateness_s"].append(wakeup_lateness); rec["control_start_jitter_s"].append(start_jitter); rec["planner_solve_completion_elapsed_s"].append(planner_complete_elapsed); rec["planner_end_to_end_latency_s"].append(planner_end_to_end_latency)
            rec["buffer_index"].append(age); rec["buffer_length"].append(0 if active is None else active.horizon); rec["best_cost"].append(best_cost); rec["mean_cost"].append(float("nan")); rec["baseline_cost"].append(float("nan")); rec["selected_cost"].append(selected_cost); rec["elite_mean_cost"].append(float("nan")); rec["selection_mode"].append("constraint_projected_nominal_fallback" if active is None else active.selection_mode); rec["failure_flags"].append(int(bool(worker_status.failure_reason))); rec["joint_limit_violation_flags"].append(0); rec["command_velocity_violation_flags"].append(int(np.any(np.abs(velocity) > physical_v + acceleration_tolerance))); rec["command_acceleration_violation_flags"].append(int(np.any(np.abs(acceleration) > physical_a + acceleration_tolerance))); rec["realized_tracking_error"].append(tracking); rec["nominal_q_ref"].append(nominal); rec["execution_nominal_q_ref"].append(execution_nominal); rec["planner_requested_residual"].append(planner_requested); rec["buffered_residual"].append(plan_residual); rec["requested_mpc_residual"].append(planner_requested); rec["feedback_raw"].append(feedback_raw); rec["feedback_correction"].append(feedback); rec["requested_feedback_correction"].append(feedback); rec["requested_correction"].append(requested_correction); rec["requested_total_correction"].append(requested_correction); rec["requested_absolute_command"].append(requested_absolute_command); rec["executed_residual"].append(command_nominal_offset); rec["command_nominal_offset"].append(command_nominal_offset); rec["safety_projection_offset"].append(safety_projection_offset); rec["projection_discrepancy"].append(projection_discrepancy); rec["residual_saturated"].append(int(np.any(np.abs(planner_requested) >= 0.95 * residual_max))); rec["feedback_saturated"].append(int(age >= 0 and np.any(np.abs(feedback_raw) >= feedback_max - 1e-8))); rec["projection_active"].append(int(np.any(np.abs(safety_projection_offset) > 1e-6))); rec["predicted_feedback_state"].append(predicted); rec["planned_q_ref"].append(planned_q_ref); rec["planner_execution_qref_error"].append(planner_execution_qref_error); rec["packet_age"].append(age); rec["packet_event"].append(event); rec["fallback_state"].append(fallback_state); rec["fallback_active"].append(fallback_active); rec["fallback_reason"].append(fallback_reason); rec["first_packet_activated_event"].append(first_packet_activated_event); rec["packet_expired_event"].append(packet_expired_event); rec["fallback_started_event"].append(fallback_started_event); rec["fallback_ended_event"].append(fallback_ended_event); rec["planner_result_event_count"].append(len(result_events)); rec["planner_result_id"].append(-1 if latest_result_event is None else latest_result_event.result_id); rec["planner_result_type"].append("" if latest_result_event is None else latest_result_event.result_type); rec["planner_failure_event"].append(planner_failure_event); rec["planner_failure_reason"].append("" if latest_result_event is None or latest_result_event.result_type != "planner_failure" else latest_result_event.reason_code); rec["worker_failure"].append(worker_status.failure_reason); rec["worker_solve_count"].append(worker_status.solve_count); rec["worker_late_drop_count"].append(worker_status.late_drop_count); rec["anchor_raw_residual"].append(worker_status.anchor_raw_residual); rec["anchor_executed_residual"].append(worker_status.anchor_executed_residual); rec["anchor_residual_projection_error"].append(worker_status.anchor_residual_projection_error); rec["anchor_previous_residual_velocity"].append(worker_status.anchor_previous_residual_velocity); rec["warm_start_shift_steps"].append(worker_status.warm_start_shift_steps); rec["mean_anchor_step_before"].append(worker_status.mean_anchor_step_before); rec["mean_anchor_step_after"].append(worker_status.mean_anchor_step_after); rec["planner_mean_updated"].append(int(worker_status.planner_mean_updated)); rec["planner_failure"].append(int(worker_status.planner_failure)); rec["packet_late_dropped"].append(int(worker_status.packet_late_dropped))
            for source, target in (("actuator_tau", "tau_actuator"), ("gravity_tau", "tau_gravity"), ("total_tau", "tau_total"), ("true_gravity_tau", "tau_gravity_true"), ("gravity_mismatch_tau", "tau_gravity_mismatch")):
                rec[target].append(torque[source].astype(np.float32))
            row = {"step": step, "controller_mode": "mpc", "multirate_mode": "threaded_asap", "tracking_error": tracking, "planning_time": planning_time, "mpc_replanned": replanned, "replan_deadline_miss": replan_deadline_miss, "packet_event": event, "packet_age": age, "control_step_wall_time": rec["control_step_wall_time"][-1], "control_deadline_miss": deadline_miss, "control_lateness_s": lateness, "actual_control_period_s": actual_period, "control_wakeup_lateness_s": wakeup_lateness, "control_start_jitter_s": start_jitter, "planner_end_to_end_latency_s": planner_end_to_end_latency, "worker_failure": worker_status.failure_reason, "worker_solve_count": worker_status.solve_count, "worker_late_drop_count": worker_status.late_drop_count, "warm_start_shift_steps": worker_status.warm_start_shift_steps, "mean_anchor_step_before": worker_status.mean_anchor_step_before, "mean_anchor_step_after": worker_status.mean_anchor_step_after, "planner_mean_updated": int(worker_status.planner_mean_updated), "planner_failure": int(worker_status.planner_failure), "packet_late_dropped": int(worker_status.packet_late_dropped)}
            if task_reference is not None:
                row.update({"ee_position_error": rec["ee_position_errors"][-1], "ee_orientation_error": rec["ee_orientation_errors"][-1], "segment_id": rec["segment_ids"][-1], "lap_id": rec["lap_ids"][-1]})
            rows.append(row)
    finally:
        stop.set()
        snapshots.wake()
        if worker is not None:
            worker.join(timeout=5.0)
        # A solve may finish between the last control-tick drain and worker
        # shutdown. Preserve that result in the event log without pretending
        # it was observed by a control tick.
        if "planner_event_rows" in locals():
            for result_event in planner_results.drain():
                planner_event_rows.append({
                    **result_event.__dict__,
                    "observed_control_step": execution_steps,
                })
        env.close()
    int_keys = {"mpc_replanned", "replan_deadline_miss", "control_deadline_miss", "buffer_index", "buffer_length", "failure_flags", "joint_limit_violation_flags", "command_velocity_violation_flags", "command_acceleration_violation_flags", "packet_age", "residual_saturated", "feedback_saturated", "projection_active", "worker_solve_count", "worker_late_drop_count", "warm_start_shift_steps", "mean_anchor_step_before", "mean_anchor_step_after", "planner_mean_updated", "planner_failure", "packet_late_dropped", "fallback_active", "first_packet_activated_event", "packet_expired_event", "fallback_started_event", "fallback_ended_event", "planner_result_event_count", "planner_result_id", "planner_failure_event", "segment_ids", "lap_ids"}
    string_keys = {"selection_mode", "packet_event", "fallback_state", "fallback_reason", "planner_result_type", "planner_failure_reason", "worker_failure"}
    arrays = {key: api["_stack_records"](value, dtype=(str if key in string_keys else np.int64 if key in int_keys else np.float32)) for key, value in rec.items()}
    final_status = worker.status() if worker is not None else None
    planner_rate = float("nan")
    if final_status is not None and final_status.solve_count > 1 and final_status.last_solve_complete_ns > final_status.first_solve_complete_ns:
        planner_rate = (final_status.solve_count - 1) / ((final_status.last_solve_complete_ns - final_status.first_solve_complete_ns) / 1e9)
    arrays.update({"controller_mode": np.asarray("mpc"), "mpc_policy": np.asarray("residual"), "cost_profile": np.asarray(args.cost_profile), "multirate_mode": np.asarray("threaded_asap"), "asap_history_mode": np.asarray(args.asap_history_mode), "asap_snapshot_mode": np.asarray(args.asap_snapshot_mode), "control_semantics_version": np.asarray(2, dtype=np.int64), "projection_semantics_version": np.asarray(2, dtype=np.int64), "projection_backend": np.asarray("shared_physical_v2"), "planner_projection": np.asarray(args.planner_projection), "projection_tolerance": np.asarray(1e-6, dtype=np.float32), "replan_interval_steps": np.asarray(-1, dtype=np.int64), "replan_deadline_s": np.asarray(np.nan, dtype=np.float32), "packet_publish_deadline_s": np.asarray(env.control_dt * args.anticipation_delay_steps - args.planner_guard_ms / 1000.0, dtype=np.float32), "planner_solve_count": np.asarray(0 if final_status is None else final_status.solve_count, dtype=np.int64), "planner_failure_count": np.asarray(0 if final_status is None else final_status.planner_failure_count, dtype=np.int64), "planner_consecutive_failure_count": np.asarray(0 if final_status is None else final_status.consecutive_planner_failure_count, dtype=np.int64), "planner_last_successful_plan_id": np.asarray(-1 if final_status is None else final_status.last_successful_plan_id, dtype=np.int64), "planner_actual_update_rate_hz": np.asarray(planner_rate, dtype=np.float32), "planner_late_drop_count": np.asarray(0 if final_status is None else final_status.late_drop_count, dtype=np.int64), "packet_expiration_count": np.asarray(fallback_machine.expiration_count, dtype=np.int64), "anticipation_delay_steps": np.asarray(args.anticipation_delay_steps, dtype=np.int64), "planner_guard_ms": np.asarray(args.planner_guard_ms, dtype=np.float32), "planner_min_interval_ms": np.asarray(args.planner_min_interval_ms, dtype=np.float32), "feedback_kq": np.asarray(args.feedback_kq, dtype=np.float32), "feedback_kdq": np.asarray(args.feedback_kdq, dtype=np.float32), "feedback_max": feedback_max, "residual_max": residual_max, "q_ref_velocity_limit": physical_v, "q_ref_acceleration_limit": physical_a, "cem_reset_std_each_step": np.asarray(args.reset_std_each_step), "cem_uniform_sample_ratio": np.asarray(args.uniform_sample_ratio, dtype=np.float32), "cem_uniform_sample_count": np.asarray(int(round((args.num_samples - 2) * args.uniform_sample_ratio)), dtype=np.int64), "recovery_active_flags": np.zeros(len(rec["actuator_q_ref"]), dtype=np.int64), "recovery_trigger_reasons": np.asarray([""] * len(rec["actuator_q_ref"])), "ddq_des": api["_stack_records"]([ddq_reference[index] for index in range(len(rec["q_des"]))]), **api["config_arrays"](robustness, env)})
    arrays["delay_protocol"] = np.asarray("full")
    arrays["residual_cost_semantics"] = np.asarray(args.residual_cost_semantics)
    arrays["packet_residual_semantics"] = np.asarray(args.packet_residual_semantics)
    arrays["residual_feasibility_semantics"] = np.asarray(args.residual_feasibility_semantics)
    arrays["nominal_command_semantics"] = np.asarray(args.nominal_command_semantics)
    pulse_start, pulse_stop = robustness.pulse_window(execution_steps)
    arrays["force_pulse_start_step"] = np.asarray(pulse_start, dtype=np.int64)
    arrays["force_pulse_stop_step"] = np.asarray(pulse_stop, dtype=np.int64)
    if task_reference is not None:
        arrays["execution_steps"] = np.asarray(execution_steps, dtype=np.int64)
    return {"arrays": arrays, "rows": rows, "planner_events": planner_event_rows, "failure_reasons": []}
