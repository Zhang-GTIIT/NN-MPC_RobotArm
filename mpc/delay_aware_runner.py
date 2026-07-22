"""Deterministic virtual-time runner for delayed multi-rate CEM-MPC.

It receives the host script namespace to avoid a circular import with the CLI
entry point.  The virtual schedule is deliberate: CEM is measured normally but
its result only becomes eligible after ``anticipation_delay_steps`` control
ticks, so controller behaviour is reproducible without Python-thread jitter.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch

from neural_dynamics.rollout import rollout_dynamics_batch
from mpc.cem_controller import CEMMPCConfig, CEMMPCController
from mpc.cost_functions import JointSpaceCostConfig
from mpc.delay_aware import DelayedPlanPacket, corrected_direct_ik_command, feedback_correction
from mpc.delay_protocol import resolve_delay_protocol
from mpc.history import commit_command_and_append_placeholder, future_history_tokens
from mpc.model_c.oracle import MuJoCoOraclePlanner
from mpc.planner_rollout import LearnedDynamicsPlanner, PlannerRolloutConfig


def run(args: Any, api: dict[str, Any]) -> dict[str, Any]:
    if args.controller_mode != "mpc" or args.mpc_policy != "residual":
        raise ValueError("virtual delay-aware modes require --controller_mode mpc --mpc_policy residual")
    if args.visualize:
        raise ValueError("--visualize is not supported by virtual delay-aware modes")
    dynamics_backend = getattr(args, "dynamics_backend", "learned")
    if dynamics_backend not in {"learned", "mujoco_oracle"}:
        raise ValueError(f"Unsupported dynamics backend: {dynamics_backend!r}")
    if dynamics_backend == "mujoco_oracle" and args.multirate_mode != "virtual_asap":
        raise ValueError("mujoco_oracle only supports deterministic virtual_asap")
    delay = int(args.anticipation_delay_steps)
    if delay < 0:
        raise ValueError("anticipation_delay_steps must be non-negative")
    protocol = resolve_delay_protocol(getattr(args, "delay_protocol", "full"))
    device = api["resolve_device"](args.device)
    api["set_seed"](args.seed)
    bundle = None
    if dynamics_backend == "learned":
        if not args.checkpoint or not args.normalizer:
            raise ValueError("--checkpoint and --normalizer are required for learned dynamics")
        bundle = api["load_dynamics_bundle"](
            checkpoint_path=api["resolve_runtime_path"](args.checkpoint), normalizer_path=api["resolve_runtime_path"](args.normalizer),
            model_type=args.model_type, n_joints=args.n_joints, device=device, history_len=args.history_len,
        )
    robustness = api["_robustness_config"](args)
    env = api["_build_control_env"](args)
    oracle_env = (
        api["MuJoCoArmEnv"](str(api["resolve_runtime_path"](args.model_xml)), n_joints=args.n_joints, seed=args.seed)
        if dynamics_backend == "mujoco_oracle"
        else None
    )
    control_dt = env.control_dt if bundle is None else bundle.control_dt
    activation_observer = getattr(args, "activation_observer", None)
    if activation_observer is not None and bundle is None:
        raise ValueError("activation_observer data collection requires learned dynamics, not mujoco_oracle")
    stack = api["_stack_records"]
    try:
        true_state = env.reset_to_configuration(api["MPC_HOME_Q"][: args.n_joints])
        previous_command = np.asarray(true_state[: args.n_joints], dtype=np.float32).copy()
        previous_velocity = np.zeros(args.n_joints, dtype=np.float32)
        for _ in range(args.settle_steps):
            true_state = env.step(previous_command)
        state = env.get_observation()
        states_history, command_history = [state.copy()], [previous_command.copy()]
        reference, dq_reference, ddq_reference, execution_steps, task_reference = api["_reference_for_run"](
            args=args, state=true_state, env=env, control_dt=control_dt
        )
        if args.max_execution_steps is not None:
            execution_steps = min(execution_steps, args.max_execution_steps)
        execution_steps = min(execution_steps, reference.shape[0] - args.horizon - delay - 1)
        if execution_steps <= 0:
            raise ValueError("reference is too short for horizon plus anticipation delay")
        parse = api["_parse_joint_vector"]
        physical_v = parse(args.command_velocity_physical_limit, args.n_joints, "command_velocity_physical_limit")
        physical_a = parse(args.command_acceleration_physical_limit, args.n_joints, "command_acceleration_physical_limit")
        residual_max = parse(args.residual_max, args.n_joints, "residual_max")
        feedback_max = parse(args.feedback_max, args.n_joints, "feedback_max")
        calibration = api["_reference_calibration"](reference, dq_reference, ddq_reference, physical_v, physical_a)
        t = lambda value: torch.as_tensor(value, dtype=torch.float32, device=device)
        cost = JointSpaceCostConfig(
            cost_mode="residual", w_q=args.w_q, w_dq=args.w_dq, w_residual=args.w_residual, w_servo=args.w_servo,
            w_residual_velocity=args.w_residual_velocity, w_residual_acceleration=args.w_residual_acceleration,
            w_first=args.w_first, w_qref_velocity=args.w_qref_velocity, w_qref_acceleration=args.w_qref_acceleration,
            w_terminal=args.w_terminal, w_joint_limit=args.w_joint_limit, w_dq_limit=args.w_dq_limit,
            q_tracking_scale=t(calibration["q_tracking_scale"]), dq_tracking_scale=t(calibration["dq_tracking_scale"]),
            residual_scale=t(0.5 * residual_max), servo_scale=t(parse(args.servo_scale, args.n_joints, "servo_scale")),
            residual_velocity_scale=t(residual_max / control_dt), residual_acceleration_scale=t(residual_max / control_dt**2),
            qref_velocity_scale=t(physical_v), qref_acceleration_scale=t(physical_a), temporal_discount=args.temporal_discount,
            barrier_max_weight=args.barrier_max_weight, state_velocity_limit=t(parse(args.state_velocity_limit, args.n_joints, "state_velocity_limit")),
            joint_limit_safe_margin=args.joint_limit_safe_margin, joint_limit_temp=args.joint_limit_temp,
            dq_limit_temp=args.dq_limit_temp, control_dt=control_dt, velocity_cost_mode=args.velocity_cost_mode,
        )
        rollout = PlannerRolloutConfig(
            mpc_policy="residual", q_ref_velocity_limit=t(physical_v), q_ref_acceleration_limit=t(physical_a),
            residual_max=t(residual_max), joint_limit_margin=args.joint_limit_margin,
            rollout_batch_size=args.rollout_batch_size, project_residual_kinematics=True,
        )
        joint_low, joint_high = t(env.joint_low), t(env.joint_high)
        controller: CEMMPCController | None = None
        active: DelayedPlanPacket | None = None
        pending: dict[int, DelayedPlanPacket] = {}
        rec: dict[str, list[Any]] = {key: [] for key in (
            "actual_states observed_states observation_noise next_states q_des dq_des actuator_q_ref delta_q_ref command_velocity command_acceleration planning_time replan_time mpc_replanned replan_deadline_miss control_step_wall_time buffer_index buffer_length best_cost mean_cost baseline_cost selected_cost elite_mean_cost selection_mode failure_flags joint_limit_violation_flags command_velocity_violation_flags command_acceleration_violation_flags realized_tracking_error nominal_q_ref planner_requested_residual buffered_residual feedback_raw feedback_correction requested_correction executed_residual projection_discrepancy residual_saturated feedback_saturated projection_active predicted_feedback_state packet_age packet_event tau_actuator tau_gravity tau_total tau_gravity_true tau_gravity_mismatch external_force_world external_generalized_force desired_ee_positions desired_ee_rotations actual_ee_positions actual_ee_rotations ee_position_errors ee_orientation_errors segment_ids lap_ids".split()
        )}
        rows: list[dict[str, Any]] = []

        def active_action(absolute_step: int) -> np.ndarray:
            nominal = reference[absolute_step + 1]
            if active is not None:
                index = active.index_at(absolute_step)
                if index is not None:
                    if protocol.replay_absolute and active.q_ref_sequence.shape == active.residual_sequence.shape:
                        return active.q_ref_sequence[index].copy()
                    return nominal + active.residual_sequence[index]
            return nominal

        def prediction_context(
            step: int,
        ) -> tuple[torch.Tensor | None, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
            if not protocol.future_state or delay == 0:
                if bundle is not None:
                    current_history = api["build_history_tensor"](
                        states_history, command_history, bundle.history_len, device
                    )
                    current_snapshot = None
                else:
                    current_history = None
                    current_snapshot = env.capture_full_state()
                return (
                    current_history,
                    np.asarray(state, dtype=np.float32).copy(),
                    previous_command.copy(),
                    previous_velocity.copy(),
                    current_snapshot,
                )
            actions = np.stack([active_action(step + i) for i in range(delay)]).astype(np.float32)
            if bundle is not None:
                history = api["build_history_tensor"](states_history, command_history, bundle.history_len, device)
                predicted = rollout_dynamics_batch(
                    model=bundle.model, normalizer=bundle.normalizer, model_type=bundle.model_type, initial_history=history,
                    future_q_ref=t(actions).unsqueeze(0), state_dim=bundle.state_dim, target_mode=bundle.target_mode,
                    control_dt=control_dt,
                )[0].detach().cpu().numpy().astype(np.float32)
                future_history = future_history_tokens(
                    states_history, command_history, predicted, actions, bundle.history_len
                )
                anchor_snapshot = None
            else:
                if oracle_env is None:
                    raise RuntimeError("MuJoCo oracle environment was not initialized")
                oracle_env.restore_full_state(env.capture_full_state())
                predicted = np.stack([oracle_env.step(action) for action in actions]).astype(np.float32)
                future_history = None
                anchor_snapshot = oracle_env.capture_full_state()
            velocity = previous_velocity if delay == 1 else (actions[-1] - actions[-2]) / control_dt
            return (
                None if future_history is None else t(future_history),
                predicted[-1],
                actions[-1],
                velocity.astype(np.float32),
                anchor_snapshot,
            )

        def execution_for_step(step: int) -> tuple[Any, ...]:
            nominal = np.asarray(reference[step + 1], dtype=np.float32)
            age = -1
            plan_residual = np.zeros(args.n_joints, dtype=np.float32)
            planner_requested = np.zeros(args.n_joints, dtype=np.float32)
            predicted_feedback = np.full(2 * args.n_joints, np.nan, dtype=np.float32)
            cached_absolute = nominal.copy()
            if active is not None:
                packet_index = active.index_at(step)
                if packet_index is not None:
                    age = packet_index
                    plan_residual = active.residual_sequence[age].copy()
                    predicted_feedback = active.predicted_state_sequence[age].copy()
                    if active.requested_residual_sequence.shape == active.residual_sequence.shape:
                        planner_requested = active.requested_residual_sequence[age].copy()
                    else:
                        planner_requested = plan_residual.copy()
                    if active.q_ref_sequence.shape == active.residual_sequence.shape:
                        cached_absolute = active.q_ref_sequence[age].copy()
            feedback_raw = np.zeros(args.n_joints, dtype=np.float32)
            feedback = np.zeros(args.n_joints, dtype=np.float32)
            if age >= 0 and protocol.feedback:
                feedback_raw = (
                    float(args.feedback_kq) * (predicted_feedback[: args.n_joints] - state[: args.n_joints])
                    + float(args.feedback_kdq) * (predicted_feedback[args.n_joints :] - state[args.n_joints :])
                ).astype(np.float32)
                feedback = np.clip(feedback_raw, -feedback_max, feedback_max).astype(np.float32)
            if age < 0:
                requested_correction = np.zeros(args.n_joints, dtype=np.float32)
                command_t, correction_t = corrected_direct_ik_command(
                    t(nominal), t(requested_correction), t(previous_command), t(previous_velocity),
                    joint_low, joint_high, args.joint_limit_margin, t(physical_v), t(physical_a), control_dt,
                )
                command = command_t.detach().cpu().numpy().astype(np.float32)
                executed = correction_t.detach().cpu().numpy().astype(np.float32)
            elif protocol.reanchor_residual:
                requested_correction = np.clip(
                    plan_residual + feedback,
                    -residual_max - feedback_max,
                    residual_max + feedback_max,
                )
                command_t, correction_t = corrected_direct_ik_command(
                    t(nominal), t(requested_correction), t(previous_command), t(previous_velocity),
                    joint_low, joint_high, args.joint_limit_margin, t(physical_v), t(physical_a), control_dt,
                )
                command = command_t.detach().cpu().numpy().astype(np.float32)
                executed = correction_t.detach().cpu().numpy().astype(np.float32)
            else:
                # Cached-command protocols replay the absolute command that
                # CEM actually evaluated at the predicted anchor.  Applying
                # the 100 Hz velocity/acceleration projection again here
                # would itself be execution-time reconciliation and make the
                # ablation collapse back onto the full controller.
                requested_correction = cached_absolute + feedback - nominal
                command = np.clip(
                    cached_absolute + feedback,
                    env.joint_low + args.joint_limit_margin,
                    env.joint_high - args.joint_limit_margin,
                ).astype(np.float32)
                executed = (command - nominal).astype(np.float32)
            requested_correction = requested_correction.astype(np.float32)
            discrepancy = (requested_correction - executed).astype(np.float32)
            return (
                nominal, age, planner_requested, plan_residual, predicted_feedback, feedback_raw, feedback,
                requested_correction, command, executed, discrepancy,
            )

        for step in range(execution_steps):
            started = api["time"].perf_counter()
            event = ""
            if step in pending:
                active = pending.pop(step); event = "packet_activated"
                if activation_observer is not None:
                    activation_observer(
                        step=step,
                        env=env,
                        state=true_state.copy(),
                        previous_command=previous_command.copy(),
                        previous_velocity=previous_velocity.copy(),
                        states_history=np.asarray(states_history, dtype=np.float32),
                        command_history=np.asarray(command_history, dtype=np.float32),
                        dynamics_bundle=bundle,
                        packet=active,
                        q_des_sequence=reference[step + 1 : step + 1 + args.horizon].copy(),
                        dq_des_sequence=dq_reference[step + 1 : step + 1 + args.horizon].copy(),
                        cost_config=cost,
                    )
            (
                nominal, age, planner_requested, plan_residual, predicted_feedback, feedback_raw, feedback,
                requested_correction, command, executed, projection_discrepancy,
            ) = execution_for_step(step)
            planning_time = float("nan"); replanned = 0; failure = 0
            best = mean = baseline = selected = elite = float("nan")
            selection = "direct_ik_nominal" if age < 0 else "delayed_packet_feedback"
            if step % args.replan_interval_steps == 0 and step + delay + args.horizon < reference.shape[0]:
                future_history, anchor_state, anchor_command, anchor_velocity, anchor_snapshot = prediction_context(step)
                anchor = step + delay
                reference_anchor = anchor if protocol.future_reference else step
                future_q = t(reference[reference_anchor + 1 : reference_anchor + 1 + args.horizon])
                common_planner_args = dict(
                    q_des=future_q,
                    dq_des=t(dq_reference[reference_anchor + 1 : reference_anchor + 1 + args.horizon]),
                    nominal_q_ref=future_q, previous_q_ref=t(anchor_command), previous_q_ref_velocity=t(anchor_velocity),
                    previous_residual=torch.zeros(args.n_joints, dtype=torch.float32, device=device),
                    previous_residual_velocity=torch.zeros(args.n_joints, dtype=torch.float32, device=device),
                    joint_low=joint_low, joint_high=joint_high, cost_config=cost, rollout_config=rollout,
                )
                if bundle is not None:
                    if future_history is None:
                        raise RuntimeError("Learned dynamics prediction context is missing history")
                    planner = LearnedDynamicsPlanner(
                        model=bundle.model, normalizer=bundle.normalizer, model_type=bundle.model_type,
                        state_dim=bundle.state_dim, target_mode=bundle.target_mode, control_dt=control_dt,
                        initial_history=future_history, **common_planner_args,
                    )
                else:
                    if oracle_env is None or anchor_snapshot is None:
                        raise RuntimeError("MuJoCo oracle prediction context is missing its anchor snapshot")
                    planner = MuJoCoOraclePlanner(
                        env=oracle_env, anchor_snapshot=anchor_snapshot, **common_planner_args
                    )
                if controller is None:
                    controller = CEMMPCController(CEMMPCConfig(
                        horizon=args.horizon, action_dim=args.n_joints, num_samples=args.num_samples, num_elites=args.num_elites,
                        elite_ratio=args.elite_ratio, cem_iters=args.cem_iters, init_std=args.init_std, min_std=args.min_std,
                        smoothing_alpha=args.smoothing_alpha, temporal_noise_alpha=args.temporal_noise_alpha,
                        reset_std_each_step=args.reset_std_each_step, uniform_sample_ratio=args.uniform_sample_ratio,
                        force_baseline_candidate=True, execute=args.cem_execute, seed=args.seed, device=str(device),
                        alternative_distance_scale=residual_max,
                    ), planner, env.joint_low, env.joint_high)
                    # CUDA's first rollout is a runtime initialisation artefact,
                    # not a representative asynchronous-plan delay.
                    if args.mpc_warmup_plans and bundle is not None:
                        generator_state = controller.generator.get_state()
                        for _ in range(args.mpc_warmup_plans):
                            controller.plan(anchor_state, anchor_command)
                        controller.generator.set_state(generator_state)
                        controller.reset()
                else:
                    controller.planner = planner
                result = controller.plan(anchor_state, anchor_command)
                planning_time, replanned, failure = float(result.planning_time), 1, int(result.failure)
                best, mean, baseline, selected, elite, selection = result.best_cost, result.mean_cost, result.baseline_cost, result.selected_cost, result.elite_mean_cost, result.selection_mode
                if result.failure:
                    event = "planner_failure"
                else:
                    packet = DelayedPlanPacket(
                        step, anchor, result.selected_residual_sequence.copy(),
                        result.selected_predicted_state_sequence.copy(), planning_time, args.multirate_mode,
                        result.branch_candidates, result.selected_q_ref_sequence.copy(),
                        (
                            np.clip(result.selected_action_sequence, -1.0, 1.0) * residual_max[None, :]
                        ).astype(np.float32),
                    )
                    if delay == 0:
                        active = packet
                        schedule_event = "packet_activated_zero_delay"
                    else:
                        pending[anchor] = packet
                        schedule_event = "packet_scheduled"
                    if planning_time > delay * control_dt:
                        schedule_event += (
                            ";oracle_overrun_scheduled"
                            if dynamics_backend == "mujoco_oracle"
                            else ";logical_wall_time_overrun"
                        )
                    event = (event + ";" if event else "") + schedule_event
                    if delay == 0:
                        (
                            nominal, age, planner_requested, plan_residual, predicted_feedback,
                            feedback_raw, feedback, requested_correction, command, executed,
                            projection_discrepancy,
                        ) = execution_for_step(step)
            delta = command - previous_command
            velocity, acceleration = delta / control_dt, (delta / control_dt - previous_velocity) / control_dt
            torque = env.compute_torque_components(command)
            rec["actual_states"].append(true_state.copy()); rec["observed_states"].append(state.copy()); rec["observation_noise"].append((state - true_state).astype(np.float32)); rec["q_des"].append(reference[step].copy()); rec["dq_des"].append(dq_reference[step].copy())
            if task_reference is not None:
                dp, dr = np.asarray(task_reference.task_positions_des[step], dtype=np.float32), np.asarray(task_reference.task_rotations_des[step], dtype=np.float32)
                ap, ar = api["site_pose"](env.model, env.data, args.ee_site_name); ap, ar = np.asarray(ap, dtype=np.float32), np.asarray(ar, dtype=np.float32)
                rec["desired_ee_positions"].append(dp); rec["desired_ee_rotations"].append(dr); rec["actual_ee_positions"].append(ap); rec["actual_ee_rotations"].append(ar)
                rec["ee_position_errors"].append(float(np.linalg.norm(ap - dp))); rec["ee_orientation_errors"].append(api["_orientation_error"](dr, ar)); rec["segment_ids"].append(int(task_reference.segment_ids[step])); rec["lap_ids"].append(int(task_reference.lap_ids[step]))
            external_force = api["_force_for_step"](robustness, step, execution_steps)
            true_state = env.step(command, external_force_world=external_force)
            state = env.get_observation()
            rec["next_states"].append(true_state.copy()); rec["external_force_world"].append(env.last_external_force_world.astype(np.float32).copy()); rec["external_generalized_force"].append(env.last_external_generalized_force.astype(np.float32).copy()); rec["actuator_q_ref"].append(command); rec["delta_q_ref"].append(delta); rec["command_velocity"].append(velocity); rec["command_acceleration"].append(acceleration)
            rec["planning_time"].append(0.0 if not np.isfinite(planning_time) else planning_time); rec["replan_time"].append(planning_time); rec["mpc_replanned"].append(replanned); rec["replan_deadline_miss"].append(int(np.isfinite(planning_time) and planning_time > delay * control_dt)); rec["control_step_wall_time"].append(api["time"].perf_counter() - started)
            rec["buffer_index"].append(age); rec["buffer_length"].append(args.horizon if active is not None else 0); rec["best_cost"].append(best); rec["mean_cost"].append(mean); rec["baseline_cost"].append(baseline); rec["selected_cost"].append(selected); rec["elite_mean_cost"].append(elite); rec["selection_mode"].append(selection); rec["failure_flags"].append(failure); rec["joint_limit_violation_flags"].append(0); rec["command_velocity_violation_flags"].append(int(np.any(np.abs(velocity) > physical_v + 1e-6))); rec["command_acceleration_violation_flags"].append(int(np.any(np.abs(acceleration) > physical_a + 1e-6)))
            residual_saturated = int(np.any(np.abs(planner_requested) >= 0.95 * residual_max))
            feedback_saturated = int(
                protocol.feedback and age >= 0 and np.any(np.abs(feedback_raw) >= feedback_max - 1e-8)
            )
            projection_active = int(np.any(np.abs(projection_discrepancy) > 1e-6))
            rec["nominal_q_ref"].append(nominal); rec["planner_requested_residual"].append(planner_requested); rec["buffered_residual"].append(plan_residual); rec["feedback_raw"].append(feedback_raw); rec["feedback_correction"].append(feedback); rec["requested_correction"].append(requested_correction); rec["executed_residual"].append(executed); rec["projection_discrepancy"].append(projection_discrepancy); rec["residual_saturated"].append(residual_saturated); rec["feedback_saturated"].append(feedback_saturated); rec["projection_active"].append(projection_active); rec["predicted_feedback_state"].append(predicted_feedback); rec["packet_age"].append(age); rec["packet_event"].append(event)
            for source, target in (("actuator_tau", "tau_actuator"), ("gravity_tau", "tau_gravity"), ("total_tau", "tau_total"), ("true_gravity_tau", "tau_gravity_true"), ("gravity_mismatch_tau", "tau_gravity_mismatch")):
                rec[target].append(torque[source].astype(np.float32))
            tracking = float(np.linalg.norm(true_state[: args.n_joints] - reference[step + 1])); rec["realized_tracking_error"].append(tracking)
            rows.append({"step": step, "controller_mode": "mpc", "tracking_error": tracking, "planning_time": planning_time, "replan_time": planning_time, "mpc_replanned": replanned, "replan_deadline_miss": rec["replan_deadline_miss"][-1], "multirate_mode": args.multirate_mode, "delay_protocol": protocol.name, "packet_event": event, "packet_age": age, "feedback_correction_norm": float(np.linalg.norm(feedback)), "executed_residual_norm": float(np.linalg.norm(executed)), "projection_discrepancy_norm": float(np.linalg.norm(projection_discrepancy)), "selection_mode": selection})
            previous_command, previous_velocity = command.copy(), velocity.astype(np.float32)
            commit_command_and_append_placeholder(states_history, command_history, command, state)
    finally:
        env.close()
        if oracle_env is not None:
            oracle_env.close()
    int_keys = {"mpc_replanned", "replan_deadline_miss", "buffer_index", "buffer_length", "failure_flags", "joint_limit_violation_flags", "command_velocity_violation_flags", "command_acceleration_violation_flags", "packet_age", "residual_saturated", "feedback_saturated", "projection_active", "segment_ids", "lap_ids"}
    string_keys = {"selection_mode", "packet_event"}
    arrays = {
        key: stack(value, dtype=(str if key in string_keys else np.int64 if key in int_keys else np.float32))
        for key, value in rec.items()
    }
    arrays.update({
        "controller_mode": np.asarray("mpc"), "mpc_policy": np.asarray("residual"), "cost_profile": np.asarray(args.cost_profile),
        "replan_interval_steps": np.asarray(args.replan_interval_steps, dtype=np.int64), "replan_deadline_s": np.asarray(delay * control_dt, dtype=np.float32),
        "multirate_mode": np.asarray(args.multirate_mode), "delay_protocol": np.asarray(protocol.name), "anticipation_delay_steps": np.asarray(delay, dtype=np.int64), "feedback_kq": np.asarray(args.feedback_kq, dtype=np.float32), "feedback_kdq": np.asarray(args.feedback_kdq, dtype=np.float32), "feedback_max": feedback_max, "residual_max": residual_max, "q_ref_velocity_limit": physical_v, "q_ref_acceleration_limit": physical_a,
        "dynamics_backend": np.asarray(dynamics_backend), "oracle_fixed_logical_delay": np.asarray(dynamics_backend == "mujoco_oracle"),
        "planner_solve_count": np.asarray(int(np.sum(np.asarray(rec["mpc_replanned"], dtype=np.int64))), dtype=np.int64),
        "planner_late_drop_count": np.asarray(sum("late_plan_dropped" in str(value) for value in rec["packet_event"]), dtype=np.int64),
        "recovery_active_flags": np.zeros(len(rec["actuator_q_ref"]), dtype=np.int64), "recovery_trigger_reasons": np.asarray([""] * len(rec["actuator_q_ref"])),
        "cem_reset_std_each_step": np.asarray(args.reset_std_each_step), "cem_uniform_sample_ratio": np.asarray(args.uniform_sample_ratio, dtype=np.float32), "cem_uniform_sample_count": np.asarray(int(round((args.num_samples - 2) * args.uniform_sample_ratio)), dtype=np.int64),
        "cem_num_samples": np.asarray(args.num_samples, dtype=np.int64), "cem_iters": np.asarray(args.cem_iters, dtype=np.int64),
        "cem_horizon": np.asarray(args.horizon, dtype=np.int64), "cem_seed": np.asarray(args.seed, dtype=np.int64),
        "ddq_des": stack([ddq_reference[i] for i in range(len(rec["q_des"]))]),
        **api["config_arrays"](robustness, env),
    })
    pulse_start, pulse_stop = robustness.pulse_window(execution_steps)
    arrays["force_pulse_start_step"] = np.asarray(pulse_start, dtype=np.int64)
    arrays["force_pulse_stop_step"] = np.asarray(pulse_stop, dtype=np.int64)
    arrays["oracle_wall_time_deadline_miss"] = (
        arrays["replan_deadline_miss"].copy()
        if dynamics_backend == "mujoco_oracle"
        else np.zeros_like(arrays["replan_deadline_miss"])
    )
    arrays["oracle_wall_time_deadline_miss_count"] = np.asarray(
        int(np.sum(arrays["oracle_wall_time_deadline_miss"])), dtype=np.int64
    )
    if task_reference is not None:
        arrays["execution_steps"] = np.asarray(execution_steps, dtype=np.int64)
    return {"arrays": arrays, "rows": rows, "failure_reasons": []}
