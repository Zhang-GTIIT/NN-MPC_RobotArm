"""Real wall-clock, dual-thread ASAP-MPC rollout runner."""
from __future__ import annotations

import threading
import time
from typing import Any

import numpy as np

from mpc.asap_planner_worker import ASAPPlannerWorker
from mpc.asap_shared import LatestSnapshotStore, PlanPacketStore
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
    snapshots, packets = LatestSnapshotStore(), PlanPacketStore()
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
        previous_executed_residual = np.zeros(args.n_joints, dtype=np.float32)
        previous_executed_residual_velocity = np.zeros(args.n_joints, dtype=np.float32)
        request_id = 0
        snapshot_history_len = 1

        def publish_snapshot(launch_step: int, launch_time_ns: int) -> None:
            nonlocal request_id
            entries_s = np.stack(states_history[-snapshot_history_len:]).astype(np.float32, copy=True)
            entries_u = np.stack(command_history[-snapshot_history_len:]).astype(np.float32, copy=True)
            snapshots.publish(PlanningSnapshot(request_id=request_id, launch_step=launch_step, launch_time_ns=launch_time_ns, states_history=entries_s, command_history=entries_u, previous_q_ref=previous_command.copy(), previous_q_ref_velocity=previous_velocity.copy(), previous_executed_residual=previous_executed_residual.copy(), previous_executed_residual_velocity=previous_executed_residual_velocity.copy(), packet_schedule=packets.schedule()))
            request_id += 1

        publish_snapshot(0, time.perf_counter_ns())
        worker = ASAPPlannerWorker(args, api, snapshots, packets, stop, env.joint_low, env.joint_high, reference, dq_reference, ddq_reference)
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
        keys = "actual_states observed_states observation_noise next_states q_des dq_des actuator_q_ref delta_q_ref command_velocity command_acceleration planning_time replan_time mpc_replanned replan_deadline_miss control_step_wall_time control_deadline_miss control_lateness_s actual_control_period_s control_wakeup_lateness_s control_start_jitter_s planner_solve_completion_elapsed_s planner_end_to_end_latency_s buffer_index buffer_length best_cost mean_cost baseline_cost selected_cost elite_mean_cost selection_mode failure_flags joint_limit_violation_flags command_velocity_violation_flags command_acceleration_violation_flags realized_tracking_error nominal_q_ref buffered_residual executed_residual feedback_correction predicted_feedback_state packet_age packet_event tau_actuator tau_gravity tau_total tau_gravity_true tau_gravity_mismatch external_force_world external_generalized_force worker_failure worker_solve_count worker_late_drop_count anchor_raw_residual anchor_executed_residual anchor_residual_projection_error anchor_previous_residual_velocity warm_start_shift_steps mean_anchor_step_before mean_anchor_step_after planner_mean_updated planner_failure packet_late_dropped".split()
        if task_reference is not None:
            keys.extend("desired_ee_positions desired_ee_rotations actual_ee_positions actual_ee_rotations ee_position_errors ee_orientation_errors segment_ids lap_ids".split())
        rec: dict[str, list[Any]] = {key: [] for key in keys}
        rows: list[dict[str, Any]] = []
        last_seen_solve_count = 0
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
            active = packets.activate_due(step, started_ns)
            event = ""
            age = -1
            plan_residual = np.zeros(args.n_joints, dtype=np.float32)
            predicted = np.full(2 * args.n_joints, np.nan, dtype=np.float32)
            if active is not None:
                index = active.index_at(step)
                if index is not None:
                    age = index
                    plan_residual = active.residual_sequence[index].copy()
                    predicted = active.predicted_state_sequence[index].copy()
                    event = "packet_active"
            feedback = np.zeros(args.n_joints, dtype=np.float32) if age < 0 else feedback_correction(predicted, state, args.feedback_kq, args.feedback_kdq, feedback_max)
            nominal = np.asarray(reference[step + 1], dtype=np.float32)
            proposed = np.clip(plan_residual + feedback, -residual_max - feedback_max, residual_max + feedback_max)
            command, executed, velocity = project_executable_command_np(nominal, proposed, previous_command, previous_velocity, env.joint_low, env.joint_high, args.joint_limit_margin, physical_v, physical_a, env.control_dt)
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
            previous_executed_residual_velocity = (executed - previous_executed_residual) / env.control_dt
            previous_executed_residual = executed.copy()
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
            rec["actual_states"].append(current_state); rec["observed_states"].append(current_observation); rec["observation_noise"].append((current_observation - current_state).astype(np.float32)); rec["next_states"].append(true_state.copy()); rec["external_force_world"].append(env.last_external_force_world.astype(np.float32).copy()); rec["external_generalized_force"].append(env.last_external_generalized_force.astype(np.float32).copy()); rec["q_des"].append(reference[step].copy()); rec["dq_des"].append(dq_reference[step].copy())
            rec["actuator_q_ref"].append(command); rec["delta_q_ref"].append(command - (command - velocity * env.control_dt)); rec["command_velocity"].append(velocity); rec["command_acceleration"].append(acceleration)
            rec["planning_time"].append(planning_time); rec["replan_time"].append(planning_time); rec["mpc_replanned"].append(replanned); rec["replan_deadline_miss"].append(replan_deadline_miss); rec["control_step_wall_time"].append(control_compute_time); rec["control_deadline_miss"].append(deadline_miss); rec["control_lateness_s"].append(lateness); rec["actual_control_period_s"].append(actual_period); rec["control_wakeup_lateness_s"].append(wakeup_lateness); rec["control_start_jitter_s"].append(start_jitter); rec["planner_solve_completion_elapsed_s"].append(planner_complete_elapsed); rec["planner_end_to_end_latency_s"].append(planner_end_to_end_latency)
            rec["buffer_index"].append(age); rec["buffer_length"].append(0 if active is None else active.horizon); rec["best_cost"].append(best_cost); rec["mean_cost"].append(float("nan")); rec["baseline_cost"].append(float("nan")); rec["selected_cost"].append(selected_cost); rec["elite_mean_cost"].append(float("nan")); rec["selection_mode"].append("direct_ik_nominal" if active is None else active.selection_mode); rec["failure_flags"].append(int(bool(worker_status.failure_reason))); rec["joint_limit_violation_flags"].append(0); rec["command_velocity_violation_flags"].append(int(np.any(np.abs(velocity) > physical_v + 1e-6))); rec["command_acceleration_violation_flags"].append(int(np.any(np.abs(acceleration) > physical_a + 1e-6))); rec["realized_tracking_error"].append(tracking); rec["nominal_q_ref"].append(nominal); rec["buffered_residual"].append(plan_residual); rec["executed_residual"].append(executed); rec["feedback_correction"].append(feedback); rec["predicted_feedback_state"].append(predicted); rec["packet_age"].append(age); rec["packet_event"].append(event); rec["worker_failure"].append(worker_status.failure_reason); rec["worker_solve_count"].append(worker_status.solve_count); rec["worker_late_drop_count"].append(worker_status.late_drop_count); rec["anchor_raw_residual"].append(worker_status.anchor_raw_residual); rec["anchor_executed_residual"].append(worker_status.anchor_executed_residual); rec["anchor_residual_projection_error"].append(worker_status.anchor_residual_projection_error); rec["anchor_previous_residual_velocity"].append(worker_status.anchor_previous_residual_velocity); rec["warm_start_shift_steps"].append(worker_status.warm_start_shift_steps); rec["mean_anchor_step_before"].append(worker_status.mean_anchor_step_before); rec["mean_anchor_step_after"].append(worker_status.mean_anchor_step_after); rec["planner_mean_updated"].append(int(worker_status.planner_mean_updated)); rec["planner_failure"].append(int(worker_status.planner_failure)); rec["packet_late_dropped"].append(int(worker_status.packet_late_dropped))
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
        env.close()
    int_keys = {"mpc_replanned", "replan_deadline_miss", "control_deadline_miss", "buffer_index", "buffer_length", "failure_flags", "joint_limit_violation_flags", "command_velocity_violation_flags", "command_acceleration_violation_flags", "packet_age", "worker_solve_count", "worker_late_drop_count", "warm_start_shift_steps", "mean_anchor_step_before", "mean_anchor_step_after", "planner_mean_updated", "planner_failure", "packet_late_dropped", "segment_ids", "lap_ids"}
    string_keys = {"selection_mode", "packet_event", "worker_failure"}
    arrays = {key: api["_stack_records"](value, dtype=(str if key in string_keys else np.int64 if key in int_keys else np.float32)) for key, value in rec.items()}
    final_status = worker.status() if worker is not None else None
    planner_rate = float("nan")
    if final_status is not None and final_status.solve_count > 1 and final_status.last_solve_complete_ns > final_status.first_solve_complete_ns:
        planner_rate = (final_status.solve_count - 1) / ((final_status.last_solve_complete_ns - final_status.first_solve_complete_ns) / 1e9)
    arrays.update({"controller_mode": np.asarray("mpc"), "mpc_policy": np.asarray("residual"), "cost_profile": np.asarray(args.cost_profile), "multirate_mode": np.asarray("threaded_asap"), "asap_history_mode": np.asarray(args.asap_history_mode), "asap_snapshot_mode": np.asarray(args.asap_snapshot_mode), "replan_interval_steps": np.asarray(-1, dtype=np.int64), "replan_deadline_s": np.asarray(np.nan, dtype=np.float32), "packet_publish_deadline_s": np.asarray(env.control_dt * args.anticipation_delay_steps - args.planner_guard_ms / 1000.0, dtype=np.float32), "planner_solve_count": np.asarray(0 if final_status is None else final_status.solve_count, dtype=np.int64), "planner_actual_update_rate_hz": np.asarray(planner_rate, dtype=np.float32), "planner_late_drop_count": np.asarray(0 if final_status is None else final_status.late_drop_count, dtype=np.int64), "anticipation_delay_steps": np.asarray(args.anticipation_delay_steps, dtype=np.int64), "planner_guard_ms": np.asarray(args.planner_guard_ms, dtype=np.float32), "planner_min_interval_ms": np.asarray(args.planner_min_interval_ms, dtype=np.float32), "feedback_kq": np.asarray(args.feedback_kq, dtype=np.float32), "feedback_kdq": np.asarray(args.feedback_kdq, dtype=np.float32), "feedback_max": feedback_max, "residual_max": residual_max, "q_ref_velocity_limit": physical_v, "q_ref_acceleration_limit": physical_a, "cem_reset_std_each_step": np.asarray(args.reset_std_each_step), "cem_uniform_sample_ratio": np.asarray(args.uniform_sample_ratio, dtype=np.float32), "cem_uniform_sample_count": np.asarray(int(round((args.num_samples - 2) * args.uniform_sample_ratio)), dtype=np.int64), "recovery_active_flags": np.zeros(len(rec["actuator_q_ref"]), dtype=np.int64), "recovery_trigger_reasons": np.asarray([""] * len(rec["actuator_q_ref"])), "ddq_des": api["_stack_records"]([ddq_reference[index] for index in range(len(rec["q_des"]))]), **api["config_arrays"](robustness, env)})
    pulse_start, pulse_stop = robustness.pulse_window(execution_steps)
    arrays["force_pulse_start_step"] = np.asarray(pulse_start, dtype=np.int64)
    arrays["force_pulse_stop_step"] = np.asarray(pulse_stop, dtype=np.int64)
    if task_reference is not None:
        arrays["execution_steps"] = np.asarray(execution_steps, dtype=np.int64)
    return {"arrays": arrays, "rows": rows, "failure_reasons": []}
