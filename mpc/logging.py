from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from mpc.utils import write_csv_rows


def save_mpc_run(save_dir: Path, arrays: dict[str, np.ndarray], rows: list[dict[str, Any]]) -> dict[str, Any]:
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(save_dir / "rollout.npz", **arrays)
    write_csv_rows(save_dir / "rollout.csv", rows)
    summary = _task_tracking_summary(arrays)
    if summary is not None:
        with (save_dir / "task_tracking_summary.json").open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)
    run_summary = build_run_summary(arrays, task_summary=summary)
    with (save_dir / "run_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(run_summary, handle, indent=2, sort_keys=True)
    plot_mpc_run(save_dir, arrays)
    print_run_summary(run_summary)
    return run_summary


def _finite_stats(values: np.ndarray) -> dict[str, float | int]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {"samples": 0, "mean": float("nan"), "p50": float("nan"), "p95": float("nan"), "p99": float("nan"), "max": float("nan")}
    return {
        "samples": int(finite.size),
        "mean": float(np.mean(finite)),
        "p50": float(np.percentile(finite, 50.0)),
        "p95": float(np.percentile(finite, 95.0)),
        "p99": float(np.percentile(finite, 99.0)),
        "max": float(np.max(finite)),
    }


def _sign_flip_rate(command_velocity: np.ndarray) -> dict[str, Any]:
    velocity = np.asarray(command_velocity, dtype=np.float64)
    if velocity.ndim != 2 or velocity.shape[0] < 2:
        return {"overall": float("nan"), "per_joint": []}
    threshold = np.maximum(np.percentile(np.abs(velocity), 20.0, axis=0), 1e-8)
    previous = velocity[:-1]
    current = velocity[1:]
    valid = (np.abs(previous) > threshold) & (np.abs(current) > threshold)
    flips = (previous * current) < 0.0
    per_joint = [float(np.mean(flips[:, joint][valid[:, joint]])) if np.any(valid[:, joint]) else float("nan") for joint in range(velocity.shape[1])]
    valid_all = valid.reshape(-1)
    return {
        "overall": float(np.mean(flips.reshape(-1)[valid_all])) if np.any(valid_all) else float("nan"),
        "per_joint": per_joint,
    }


def _per_joint_percentile(values: np.ndarray, percentile: float) -> list[float]:
    value = np.asarray(values, dtype=np.float64)
    if value.ndim != 2 or value.shape[0] == 0:
        return []
    return [float(item) for item in np.percentile(np.abs(value), percentile, axis=0)]


def _string_counts(values: np.ndarray) -> dict[str, int]:
    flattened = np.asarray(values).reshape(-1)
    counts: dict[str, int] = {}
    for value in flattened:
        text = str(value)
        if text:
            counts[text] = counts.get(text, 0) + 1
    return counts


def _replay_summary(arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    if "replay_q_error_norm" not in arrays or "replay_dq_error_norm" not in arrays:
        return {"status": "not_applicable"}
    q_error = np.asarray(arrays["replay_q_error_norm"], dtype=np.float64)
    dq_error = np.asarray(arrays["replay_dq_error_norm"], dtype=np.float64)
    if q_error.ndim != 2 or dq_error.shape != q_error.shape:
        return {"status": "invalid"}
    by_horizon = []
    for index in range(q_error.shape[1]):
        by_horizon.append(
            {
                "k": index + 1,
                "q_error": _finite_stats(q_error[:, index]),
                "dq_error": _finite_stats(dq_error[:, index]),
            }
        )
    first_mean = by_horizon[0]["q_error"]["mean"] if by_horizon else float("nan")
    terminal_mean = by_horizon[-1]["q_error"]["mean"] if by_horizon else float("nan")
    ratio = float(terminal_mean / first_mean) if np.isfinite(first_mean) and first_mean > 1e-12 else float("nan")
    return {
        "status": "available",
        "horizon": int(q_error.shape[1]),
        "valid_anchor_count": int(np.sum(np.isfinite(q_error[:, 0]))) if q_error.shape[1] else 0,
        "online_selected_next_q_error": _finite_stats(np.asarray(arrays.get("predicted_next_q_error", np.empty(0)))),
        "online_selected_next_dq_error": _finite_stats(np.asarray(arrays.get("predicted_next_dq_error", np.empty(0)))),
        "q_error_horizon_growth_ratio": ratio,
        "q_error_command_velocity_corr": float(np.asarray(arrays.get("replay_q_error_command_velocity_corr", np.nan)).reshape(-1)[0]),
        "q_error_command_acceleration_corr": float(np.asarray(arrays.get("replay_q_error_command_acceleration_corr", np.nan)).reshape(-1)[0]),
        "by_horizon": by_horizon,
    }


def build_run_summary(arrays: dict[str, np.ndarray], *, task_summary: dict[str, Any] | None = None) -> dict[str, Any]:
    actual_states = np.asarray(arrays.get("actual_states", np.empty((0,))))
    q_des = np.asarray(arrays.get("q_des", np.empty((0,))))
    n_joints = q_des.shape[1] if q_des.ndim == 2 else 0
    joint_tracking: dict[str, Any] = {"status": "not_available"}
    if actual_states.ndim == 2 and q_des.ndim == 2 and n_joints and actual_states.shape[0] and actual_states.shape[1] >= n_joints:
        length = min(actual_states.shape[0], q_des.shape[0])
        error = actual_states[:length, :n_joints] - q_des[:length]
        joint_tracking = {
            "position_rmse_rad": float(np.sqrt(np.mean(np.square(error)))),
            "max_position_error_rad": float(np.max(np.abs(error))),
            "final_position_error_inf_rad": float(np.max(np.abs(error[-1]))),
        }
    controller_mode = np.asarray(arrays.get("controller_mode", "unknown")).reshape(-1)
    recovery_trigger_counts = _string_counts(np.asarray(arrays.get("recovery_trigger_reasons", np.empty(0))))
    replan_time = np.asarray(arrays.get("replan_time", np.empty(0)), dtype=np.float64)
    if not replan_time.size:
        replan_time = np.asarray(arrays.get("planning_time", np.empty(0)), dtype=np.float64)
    replan_flags = np.asarray(arrays.get("mpc_replanned", np.empty(0)), dtype=np.int64)
    replan_count = int(np.sum(replan_flags != 0)) if replan_flags.size else int(np.sum(np.isfinite(replan_time)))
    deadline_misses = np.asarray(arrays.get("replan_deadline_miss", np.empty(0)), dtype=np.int64)
    deadline_miss_count = int(np.sum(deadline_misses != 0))
    deadline_s = float(np.asarray(arrays.get("replan_deadline_s", np.nan)).reshape(-1)[0])
    interval_steps = int(np.asarray(arrays.get("replan_interval_steps", 1)).reshape(-1)[0])
    multirate_mode = str(np.asarray(arrays.get("multirate_mode", "synchronous")).reshape(-1)[0])
    threaded_asap = multirate_mode == "threaded_asap"
    planner_solve_count = int(np.asarray(arrays.get("planner_solve_count", replan_count)).reshape(-1)[0])
    planner_late_drop_count = int(np.asarray(arrays.get("planner_late_drop_count", 0)).reshape(-1)[0])
    planner_rate = float(np.asarray(arrays.get("planner_actual_update_rate_hz", np.nan)).reshape(-1)[0])
    replanning = {
        "interval_steps": None if threaded_asap else interval_steps,
        "deadline_s": None if threaded_asap else deadline_s,
        "nominal_frequency_hz": float("nan") if threaded_asap else (float(1.0 / deadline_s) if np.isfinite(deadline_s) and deadline_s > 0.0 else float("nan")),
        "count": planner_solve_count if threaded_asap else replan_count,
        "deadline_miss_count": deadline_miss_count,
        "deadline_miss_rate": float(deadline_miss_count / planner_solve_count) if threaded_asap and planner_solve_count else (float(deadline_miss_count / replan_count) if replan_count else float("nan")),
        "control_deadline_miss_count": int(np.sum(np.asarray(arrays.get("control_deadline_miss", np.empty(0))) != 0)),
    }
    planner = {
        "solve_count": planner_solve_count,
        "actual_update_rate_hz": planner_rate,
        "late_drop_count": planner_late_drop_count,
        "late_drop_rate": float(planner_late_drop_count / planner_solve_count) if planner_solve_count else float("nan"),
        "dynamics_backend": str(np.asarray(arrays.get("dynamics_backend", "not_applicable")).reshape(-1)[0]),
        "oracle_fixed_logical_delay": bool(
            np.asarray(arrays.get("oracle_fixed_logical_delay", False)).reshape(-1)[0]
        ),
        "oracle_wall_time_deadline_miss_count": int(
            np.asarray(arrays.get("oracle_wall_time_deadline_miss_count", 0)).reshape(-1)[0]
        ),
        "end_to_end_latency_s": _finite_stats(np.asarray(arrays.get("planner_end_to_end_latency_s", np.empty(0)))),
        "packet_publish_deadline_s": float(np.asarray(arrays.get("packet_publish_deadline_s", np.nan)).reshape(-1)[0]),
    }
    return {
        "schema_version": 3,
        "controller_mode": str(controller_mode[0]) if controller_mode.size else "unknown",
        "mpc_policy": str(np.asarray(arrays.get("mpc_policy", "not_applicable")).reshape(-1)[0]),
        "cost_profile": str(np.asarray(arrays.get("cost_profile", "not_applicable")).reshape(-1)[0]),
        "recorded_steps": int(actual_states.shape[0]) if actual_states.ndim else 0,
        "robustness": {
            "payload_level": int(np.asarray(arrays.get("payload_level", 0)).reshape(-1)[0]),
            "actuator_gain_level": int(np.asarray(arrays.get("actuator_gain_level", 0)).reshape(-1)[0]),
            "force_pulse_level": int(np.asarray(arrays.get("force_pulse_level", 0)).reshape(-1)[0]),
            "observation_noise_level": int(np.asarray(arrays.get("observation_noise_level", 0)).reshape(-1)[0]),
            "payload_mass_kg": float(np.asarray(arrays.get("payload_mass_kg", 0.0)).reshape(-1)[0]),
            "actuator_kp_scale": float(np.asarray(arrays.get("actuator_kp_scale", 1.0)).reshape(-1)[0]),
            "actuator_kd_scale": float(np.asarray(arrays.get("actuator_kd_scale", 1.0)).reshape(-1)[0]),
            "force_pulse_n": float(np.asarray(arrays.get("force_pulse_n", 0.0)).reshape(-1)[0]),
            "observation_q_std_rad": float(np.asarray(arrays.get("observation_q_std_rad", 0.0)).reshape(-1)[0]),
            "observation_dq_std_rad_s": float(np.asarray(arrays.get("observation_dq_std_rad_s", 0.0)).reshape(-1)[0]),
        },
        "timing": {
            "control_step_wall_time_s": _finite_stats(np.asarray(arrays.get("control_step_wall_time", np.empty(0)))),
            "planning_time_s": _finite_stats(replan_time),
            "control_lateness_s": _finite_stats(np.asarray(arrays.get("control_lateness_s", np.empty(0)))),
            "actual_control_period_s": _finite_stats(np.asarray(arrays.get("actual_control_period_s", np.empty(0)))),
            "control_wakeup_lateness_s": _finite_stats(np.asarray(arrays.get("control_wakeup_lateness_s", np.empty(0)))),
            "control_start_jitter_s": _finite_stats(np.asarray(arrays.get("control_start_jitter_s", np.empty(0)))),
        },
        "replanning": replanning,
        "planner": planner,
        "cem_sampling": {
            "num_samples": int(np.asarray(arrays.get("cem_num_samples", 0)).reshape(-1)[0]),
            "iterations": int(np.asarray(arrays.get("cem_iters", 0)).reshape(-1)[0]),
            "horizon": int(np.asarray(arrays.get("cem_horizon", 0)).reshape(-1)[0]),
            "seed": int(np.asarray(arrays.get("cem_seed", 0)).reshape(-1)[0]),
            "reset_std_each_step": bool(np.asarray(arrays.get("cem_reset_std_each_step", False)).reshape(-1)[0]),
            "uniform_sample_ratio": float(np.asarray(arrays.get("cem_uniform_sample_ratio", 0.0)).reshape(-1)[0]),
            "uniform_sample_count": int(np.asarray(arrays.get("cem_uniform_sample_count", 0)).reshape(-1)[0]),
            "std_start_mean": _finite_stats(np.asarray(arrays.get("sampling_std_start_mean", np.empty(0)))),
            "std_end_mean": _finite_stats(np.asarray(arrays.get("sampling_std_end_mean", np.empty(0)))),
            "baseline_cost": _finite_stats(np.asarray(arrays.get("baseline_cost", np.empty(0)))),
        },
        "tracking": {"joint": joint_tracking, "task": None if task_summary is None else task_summary.get("overall")},
        "smoothness": {
            "command_velocity_p95_rad_s": _per_joint_percentile(np.asarray(arrays.get("command_velocity", np.empty((0,)))), 95.0),
            "command_acceleration_p95_rad_s2": _per_joint_percentile(np.asarray(arrays.get("command_acceleration", np.empty((0,)))), 95.0),
            "command_sign_flip_rate": _sign_flip_rate(np.asarray(arrays.get("command_velocity", np.empty((0,))))),
        },
        "actuator": {
            "torque_rms_nm": float(np.sqrt(np.mean(np.square(arrays["tau_actuator"])))) if np.asarray(arrays.get("tau_actuator", np.empty(0))).size else float("nan"),
            "torque_p95_nm": _per_joint_percentile(np.asarray(arrays.get("tau_actuator", np.empty((0,)))), 95.0),
            "torque_slew_p95_nm_per_step": _per_joint_percentile(np.diff(np.asarray(arrays.get("tau_actuator", np.empty((0,)))), axis=0), 95.0),
        },
        "safety": {
            "controller_failure_count": int(np.sum(np.asarray(arrays.get("failure_flags", np.empty(0))) != 0)),
            "joint_limit_violation_count": int(np.sum(np.asarray(arrays.get("joint_limit_violation_flags", np.empty(0))) != 0)),
            "command_velocity_violation_count": int(np.sum(np.asarray(arrays.get("command_velocity_violation_flags", np.empty(0))) != 0)),
            "command_acceleration_violation_count": int(np.sum(np.asarray(arrays.get("command_acceleration_violation_flags", np.empty(0))) != 0)),
            "recovery_active_step_count": int(np.sum(np.asarray(arrays.get("recovery_active_flags", np.empty(0))) != 0)),
            "recovery_trigger_count": int(sum(recovery_trigger_counts.values())),
            "recovery_trigger_counts": recovery_trigger_counts,
        },
        "residual": {
            "buffered_max_abs_rad": _per_joint_percentile(
                np.asarray(arrays.get("buffered_residual", np.empty((0,)))), 100.0
            ),
            "buffered_p95_abs_rad": _per_joint_percentile(
                np.asarray(arrays.get("buffered_residual", np.empty((0,)))), 95.0
            ),
            "max_abs_rad": _per_joint_percentile(np.asarray(arrays.get("executed_residual", np.empty((0,)))), 100.0),
            "p95_abs_rad": _per_joint_percentile(np.asarray(arrays.get("executed_residual", np.empty((0,)))), 95.0),
            "reanchor_delta_p95_abs_rad": _per_joint_percentile(
                np.asarray(arrays.get("residual_reanchor_delta", np.empty((0,)))), 95.0
            ),
            "buffer_modes": _string_counts(np.asarray(arrays.get("multirate_buffer_mode", np.empty(0)))),
            "feedback_p95_abs_rad": _per_joint_percentile(
                np.asarray(arrays.get("feedback_correction", np.empty((0,)))), 95.0
            ),
            "packet_events": _string_counts(np.asarray(arrays.get("packet_event", np.empty(0)))),
        },
        "model_replay": _replay_summary(arrays),
    }


def _fmt(value: float, *, digits: int = 3) -> str:
    return "n/a" if not np.isfinite(value) else f"{value:.{digits}g}"


def _fmt_ms(value_s: float) -> str:
    return "n/a" if not np.isfinite(value_s) else f"{1e3 * value_s:.1f} ms"


def _fmt_joint_values(values: list[float], unit: str) -> str:
    if not values:
        return "n/a"
    return "  ".join(f"J{index + 1}={_fmt(value):>7} {unit}" for index, value in enumerate(values))


def print_run_summary(summary: dict[str, Any]) -> None:
    """Print a compact, fixed-layout end-of-run diagnostic report."""
    timing = summary["timing"]
    sampling = summary["cem_sampling"]
    tracking = summary["tracking"]
    smoothness = summary["smoothness"]
    actuator = summary["actuator"]
    safety = summary["safety"]
    replay = summary["model_replay"]
    replanning = summary["replanning"]
    rule = "=" * 88
    print(f"\n{rule}\nCEM-MPC | END-OF-RUN REPORT\n{rule}")
    print(
        f"RUN       controller: {summary['controller_mode']:<12} "
        f"policy: {summary['mpc_policy']:<20} steps: {summary['recorded_steps']}"
    )
    print("-" * 88)
    print(
        "TIMING    "
        f"control [p50 / p95 / p99 / max] = {_fmt_ms(timing['control_step_wall_time_s']['p50']):>9} / "
        f"{_fmt_ms(timing['control_step_wall_time_s']['p95']):>9} / {_fmt_ms(timing['control_step_wall_time_s']['p99']):>9} / {_fmt_ms(timing['control_step_wall_time_s']['max']):>9}"
    )
    print(
        "          "
        f"planning [mean / p95]      = {_fmt_ms(timing['planning_time_s']['mean']):>9} / "
        f"{_fmt_ms(timing['planning_time_s']['p95']):>9}"
    )
    if replanning["count"]:
        print(
            "          "
            f"replan: {replanning['count']} @ {_fmt(replanning['nominal_frequency_hz'])} Hz  "
            f"deadline misses: {replanning['deadline_miss_count']} "
            f"({_fmt(100.0 * replanning['deadline_miss_rate'])}%)"
        )
    if replanning["control_deadline_miss_count"]:
        print(f"          control deadline misses: {replanning['control_deadline_miss_count']}")
    print(
        "CEM       "
        f"std reset: {str(sampling['reset_std_each_step']):<5}  "
        f"uniform: {100.0 * sampling['uniform_sample_ratio']:.0f}% ({sampling['uniform_sample_count']} samples)  "
        f"std start/end mean: {_fmt(sampling['std_start_mean']['mean'])} / {_fmt(sampling['std_end_mean']['mean'])}"
    )
    if sampling["baseline_cost"]["samples"]:
        print(f"          baseline cost [mean / p95] = {_fmt(sampling['baseline_cost']['mean'])} / {_fmt(sampling['baseline_cost']['p95'])}")
    joint = tracking["joint"]
    if joint.get("status") != "not_available":
        print(
            "TRACKING  "
            f"joint RMSE: {_fmt(joint['position_rmse_rad']):>8} rad   "
            f"max: {_fmt(joint['max_position_error_rad']):>8} rad   "
            f"final inf-norm: {_fmt(joint['final_position_error_inf_rad']):>8} rad"
        )
    if tracking["task"] is not None:
        task = tracking["task"]
        print(
            "          "
            f"TCP position RMSE: {_fmt(1e3 * task['position_rmse_m']):>7} mm   "
            f"max: {_fmt(1e3 * task['max_position_error_m']):>7} mm   "
            f"orientation RMSE: {_fmt(np.degrees(task['orientation_rmse_rad'])):>7} deg"
        )
    print(
        "COMMANDS  "
        f"sign-flip rate: {_fmt(smoothness['command_sign_flip_rate']['overall']):>7}   "
        "(successive non-trivial command-speed reversals)"
    )
    print(f"          speed p95: {_fmt_joint_values(smoothness['command_velocity_p95_rad_s'], 'rad/s')}")
    print(f"          accel p95: {_fmt_joint_values(smoothness['command_acceleration_p95_rad_s2'], 'rad/s^2')}")
    print(
        "ACTUATOR  "
        f"torque RMS: {_fmt(actuator['torque_rms_nm']):>7} Nm   "
        f"torque p95: {_fmt_joint_values(actuator['torque_p95_nm'], 'Nm')}"
    )
    print(
        "SAFETY    "
        f"planner failures: {safety['controller_failure_count']:<5} "
        f"joint-limit: {safety['joint_limit_violation_count']:<5} "
        f"command v/a flags: {safety['command_velocity_violation_count']}/{safety['command_acceleration_violation_count']}  "
        f"recovery triggers: {safety['recovery_trigger_count']}"
    )
    if safety["recovery_active_step_count"] or safety["recovery_trigger_counts"]:
        print(
            f"          recovery active steps: {safety['recovery_active_step_count']} "
            f"triggers: {safety['recovery_trigger_counts']}"
        )
    if replay["status"] == "available":
        first = replay["by_horizon"][0]
        terminal = replay["by_horizon"][-1]
        print(
            "MODEL     "
            f"actual-command replay anchors: {replay['valid_anchor_count']}   horizon: {replay['horizon']}"
        )
        print(
            "          "
            f"position error [k=1 / k=H]: {_fmt(1e3 * first['q_error']['mean']):>7} / "
            f"{_fmt(1e3 * terminal['q_error']['mean']):>7} mm   "
            f"growth: {_fmt(replay['q_error_horizon_growth_ratio']):>7}x"
        )
        print(
            "          "
            f"velocity error [k=1 / k=H]: {_fmt(first['dq_error']['mean']):>7} / "
            f"{_fmt(terminal['dq_error']['mean']):>7} rad/s"
        )
    else:
        print("MODEL     replay: not applicable (no learned MPC rollout)")
    print(rule)


def _task_arrays(arrays: dict[str, np.ndarray]) -> tuple[np.ndarray, ...] | None:
    required = (
        "desired_ee_positions",
        "desired_ee_rotations",
        "actual_ee_positions",
        "actual_ee_rotations",
        "ee_position_errors",
        "ee_orientation_errors",
        "segment_ids",
        "lap_ids",
    )
    if not all(name in arrays for name in required):
        return None

    values = tuple(np.asarray(arrays[name]) for name in required)
    desired_position, desired_rotation, actual_position, actual_rotation, *_ = values
    if (
        desired_position.ndim != 2
        or desired_position.shape[1:] != (3,)
        or actual_position.shape != desired_position.shape
        or desired_rotation.ndim != 3
        or desired_rotation.shape[1:] != (3, 3)
        or actual_rotation.shape != desired_rotation.shape
    ):
        return None
    lengths = [value.shape[0] for value in values]
    if not lengths or min(lengths) == 0:
        return None
    length = min(lengths)
    return tuple(value[:length] for value in values)


def _error_metrics(position_errors: np.ndarray, orientation_errors: np.ndarray) -> dict[str, float | int]:
    return {
        "samples": int(position_errors.shape[0]),
        "position_rmse_m": float(np.sqrt(np.mean(np.square(position_errors)))),
        "orientation_rmse_rad": float(np.sqrt(np.mean(np.square(orientation_errors)))),
        "max_position_error_m": float(np.max(position_errors)),
        "max_orientation_error_rad": float(np.max(orientation_errors)),
    }


def _metrics_by_id(
    identifiers: np.ndarray,
    position_errors: np.ndarray,
    orientation_errors: np.ndarray,
    *,
    exclude_negative_ids: bool = False,
) -> dict[str, dict[str, float | int]]:
    unique_ids = np.unique(identifiers)
    if exclude_negative_ids:
        unique_ids = unique_ids[unique_ids >= 0]
    summary: dict[str, dict[str, float | int]] = {}
    for identifier in unique_ids:
        mask = identifiers == identifier
        summary[str(int(identifier))] = _error_metrics(position_errors[mask], orientation_errors[mask])
    return summary


def _task_tracking_summary(arrays: dict[str, np.ndarray]) -> dict[str, Any] | None:
    task_arrays = _task_arrays(arrays)
    if task_arrays is None:
        return None
    (
        _desired_position,
        _desired_rotation,
        _actual_position,
        _actual_rotation,
        position_errors,
        orientation_errors,
        segment_ids,
        lap_ids,
    ) = task_arrays
    steps = position_errors.shape[0]
    summary: dict[str, Any] = {
        "recorded_steps": int(steps),
        "overall": _error_metrics(position_errors, orientation_errors),
        "segments": _metrics_by_id(segment_ids, position_errors, orientation_errors),
        "laps": _metrics_by_id(
            lap_ids,
            position_errors,
            orientation_errors,
            exclude_negative_ids=True,
        ),
        "final": {
            "tcp_position_error_m": float(position_errors[-1]),
            "tcp_orientation_error_rad": float(orientation_errors[-1]),
        },
    }
    controller_mode = np.asarray(arrays.get("controller_mode", np.empty((0,))))
    if controller_mode.size == 1:
        summary["controller_mode"] = str(controller_mode.reshape(-1)[0])

    actual_states = np.asarray(arrays.get("actual_states", np.empty((0,))))
    q_des = np.asarray(arrays.get("q_des", np.empty((0,))))
    if actual_states.ndim == 2 and q_des.ndim == 2 and actual_states.shape[1] >= q_des.shape[1]:
        joint_length = min(steps, actual_states.shape[0], q_des.shape[0])
        if joint_length:
            joint_error = actual_states[:joint_length, : q_des.shape[1]] - q_des[:joint_length]
            summary["joint_tracking"] = {
                "position_rmse_rad": float(np.sqrt(np.mean(np.square(joint_error)))),
                "max_position_error_rad": float(np.max(np.abs(joint_error))),
                "final_position_error_inf_rad": float(np.max(np.abs(joint_error[-1]))),
            }

    planning_time = np.asarray(arrays.get("planning_time", np.empty((0,))), dtype=np.float64)
    failure_flags = np.asarray(arrays.get("failure_flags", np.empty((0,))), dtype=np.float64)
    limit_flags = np.asarray(arrays.get("joint_limit_violation_flags", np.empty((0,))), dtype=np.float64)
    command_velocity_flags = np.asarray(arrays.get("command_velocity_violation_flags", np.empty((0,))), dtype=np.float64)
    command_acceleration_flags = np.asarray(arrays.get("command_acceleration_violation_flags", np.empty((0,))), dtype=np.float64)
    planning: dict[str, float | int] = {}
    if planning_time.size:
        planning["mean_planning_time_s"] = float(np.mean(planning_time))
        planning["max_planning_time_s"] = float(np.max(planning_time))
    if failure_flags.size:
        planning["failure_count"] = int(np.sum(failure_flags != 0.0))
        planning["failure_rate"] = float(np.mean(failure_flags != 0.0))
    if limit_flags.size:
        planning["joint_limit_violation_count"] = int(np.sum(limit_flags != 0.0))
        planning["joint_limit_violation_rate"] = float(np.mean(limit_flags != 0.0))
    if command_velocity_flags.size:
        planning["command_velocity_violation_count"] = int(np.sum(command_velocity_flags != 0.0))
        planning["command_velocity_violation_rate"] = float(np.mean(command_velocity_flags != 0.0))
    if command_acceleration_flags.size:
        planning["command_acceleration_violation_count"] = int(np.sum(command_acceleration_flags != 0.0))
        planning["command_acceleration_violation_rate"] = float(np.mean(command_acceleration_flags != 0.0))
    if planning:
        summary["planning"] = planning
    return summary


def _plane_projection(desired_position: np.ndarray, actual_position: np.ndarray) -> tuple[np.ndarray, np.ndarray, str, str]:
    """Project TCP paths into their dominant desired-trajectory plane when possible."""
    centered = desired_position - desired_position.mean(axis=0, keepdims=True)
    try:
        _, singular_values, right_vectors = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        singular_values = np.empty(0)
        right_vectors = np.empty((0, 3))
    if singular_values.size >= 2 and singular_values[1] > 1e-10:
        axes = right_vectors[:2]
        return centered @ axes.T, (actual_position - desired_position.mean(axis=0, keepdims=True)) @ axes.T, "plane axis 1 (m)", "plane axis 2 (m)"
    return desired_position[:, :2], actual_position[:, :2], "world x (m)", "world y (m)"


def _plot_group_tracking_summary(
    plt: Any,
    save_path: Path,
    group_ids: np.ndarray,
    position_errors: np.ndarray,
    orientation_errors: np.ndarray,
    title: str,
    exclude_negative_ids: bool = False,
) -> None:
    unique_ids = np.unique(group_ids)
    if exclude_negative_ids:
        unique_ids = unique_ids[unique_ids >= 0]
    if unique_ids.size == 0:
        return

    position_rmse = []
    orientation_rmse = []
    for group_id in unique_ids:
        mask = group_ids == group_id
        position_rmse.append(float(np.sqrt(np.mean(np.square(position_errors[mask])))))
        orientation_rmse.append(float(np.sqrt(np.mean(np.square(orientation_errors[mask])))))

    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    labels = [str(int(group_id)) for group_id in unique_ids]
    axes[0].bar(labels, position_rmse)
    axes[0].set_ylabel("TCP position RMSE (m)")
    axes[0].set_title(title)
    axes[1].bar(labels, orientation_rmse)
    axes[1].set_ylabel("TCP orientation RMSE (rad)")
    axes[1].set_xlabel("lap id" if exclude_negative_ids else "segment id")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def _plot_task_space_run(plt: Any, save_dir: Path, arrays: dict[str, np.ndarray]) -> None:
    task_arrays = _task_arrays(arrays)
    if task_arrays is None:
        return
    (
        desired_position,
        _desired_rotation,
        actual_position,
        _actual_rotation,
        position_errors,
        orientation_errors,
        segment_ids,
        lap_ids,
    ) = task_arrays
    time = np.arange(desired_position.shape[0])

    fig_3d = plt.figure(figsize=(8, 7))
    axis_3d = fig_3d.add_subplot(111, projection="3d")
    axis_3d.plot(*desired_position.T, label="desired TCP", linestyle="--")
    axis_3d.plot(*actual_position.T, label="actual TCP")
    axis_3d.set_xlabel("x (m)")
    axis_3d.set_ylabel("y (m)")
    axis_3d.set_zlabel("z (m)")
    axis_3d.legend()
    fig_3d.tight_layout()
    fig_3d.savefig(save_dir / "ee_trajectory_3d.png", dpi=150)
    plt.close(fig_3d)

    desired_projection, actual_projection, horizontal_label, vertical_label = _plane_projection(desired_position, actual_position)
    fig_projection, axis_projection = plt.subplots(figsize=(7, 6))
    axis_projection.plot(desired_projection[:, 0], desired_projection[:, 1], label="desired TCP", linestyle="--")
    axis_projection.plot(actual_projection[:, 0], actual_projection[:, 1], label="actual TCP")
    axis_projection.set_xlabel(horizontal_label)
    axis_projection.set_ylabel(vertical_label)
    axis_projection.axis("equal")
    axis_projection.legend()
    fig_projection.tight_layout()
    fig_projection.savefig(save_dir / "ee_xy_or_plane_projection.png", dpi=150)
    plt.close(fig_projection)

    fig_position, axes_position = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    for index, label in enumerate(("x", "y", "z")):
        axes_position[index].plot(time, desired_position[:, index], label="desired")
        axes_position[index].plot(time, actual_position[:, index], label="actual", linestyle="--")
        axes_position[index].set_ylabel(f"{label} (m)")
        axes_position[index].legend(fontsize=8)
    axes_position[-1].set_xlabel("step")
    fig_position.tight_layout()
    fig_position.savefig(save_dir / "ee_position_tracking.png", dpi=150)
    plt.close(fig_position)

    fig_position_error, axis_position_error = plt.subplots(figsize=(10, 4))
    axis_position_error.plot(time, position_errors)
    axis_position_error.set_xlabel("step")
    axis_position_error.set_ylabel("TCP position error (m)")
    fig_position_error.tight_layout()
    fig_position_error.savefig(save_dir / "ee_position_error.png", dpi=150)
    plt.close(fig_position_error)

    fig_orientation_error, axis_orientation_error = plt.subplots(figsize=(10, 4))
    axis_orientation_error.plot(time, orientation_errors)
    axis_orientation_error.set_xlabel("step")
    axis_orientation_error.set_ylabel("TCP orientation error (rad)")
    fig_orientation_error.tight_layout()
    fig_orientation_error.savefig(save_dir / "ee_orientation_error.png", dpi=150)
    plt.close(fig_orientation_error)

    _plot_group_tracking_summary(
        plt,
        save_dir / "segment_tracking_summary.png",
        segment_ids,
        position_errors,
        orientation_errors,
        title="TCP tracking by reference segment",
    )
    _plot_group_tracking_summary(
        plt,
        save_dir / "lap_tracking_summary.png",
        lap_ids,
        position_errors,
        orientation_errors,
        title="TCP tracking by shape lap",
        exclude_negative_ids=True,
    )


def plot_mpc_run(save_dir: Path, arrays: dict[str, np.ndarray]) -> None:
    # Rollout logging is batch/headless work.  Do not let Matplotlib select a
    # Tk backend: threaded_asap has a worker thread, and Tk objects destroyed
    # during interpreter shutdown otherwise emit "main thread is not in main
    # loop" errors even though the rollout itself completed successfully.
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    actual_states = arrays["actual_states"]
    q_des = arrays["q_des"]
    actuator_q_ref = arrays["actuator_q_ref"]
    nominal_q_ref = np.asarray(arrays.get("nominal_q_ref", np.empty((0,))))
    planning_time = np.asarray(arrays.get("replan_time", arrays["planning_time"]))
    best_cost = arrays["best_cost"]
    n_joints = q_des.shape[1]
    time = np.arange(q_des.shape[0])

    fig_q, axes_q = plt.subplots(n_joints, 1, figsize=(10, 2.0 * n_joints), sharex=True)
    fig_dq, axes_dq = plt.subplots(n_joints, 1, figsize=(10, 2.0 * n_joints), sharex=True)
    if n_joints == 1:
        axes_q = [axes_q]
        axes_dq = [axes_dq]
    for idx in range(n_joints):
        axes_q[idx].plot(time, actual_states[:, idx], label="actual_q")
        axes_q[idx].plot(time, q_des[:, idx], label="q_des", linestyle="--")
        if nominal_q_ref.shape == actuator_q_ref.shape:
            axes_q[idx].plot(time, nominal_q_ref[:, idx], label="q_nom", linestyle="-.")
        axes_q[idx].plot(time, actuator_q_ref[:, idx], label="actuator_q_ref", linestyle=":")
        axes_q[idx].set_ylabel(f"q{idx}")
        axes_q[idx].legend(fontsize=8)
        axes_dq[idx].plot(time, actual_states[:, n_joints + idx], label="actual_dq")
        axes_dq[idx].set_ylabel(f"dq{idx}")
        axes_dq[idx].legend(fontsize=8)
    axes_q[-1].set_xlabel("step")
    axes_dq[-1].set_xlabel("step")
    fig_q.tight_layout()
    fig_dq.tight_layout()
    fig_q.savefig(save_dir / "q_tracking.png", dpi=150)
    fig_dq.savefig(save_dir / "dq.png", dpi=150)
    plt.close(fig_q)
    plt.close(fig_dq)

    tracking_error = np.linalg.norm(actual_states[:, :n_joints] - q_des, axis=1)
    fig_err, ax_err = plt.subplots(figsize=(10, 4))
    ax_err.plot(time, tracking_error)
    ax_err.set_xlabel("step")
    ax_err.set_ylabel("||q - q_des||")
    fig_err.tight_layout()
    fig_err.savefig(save_dir / "tracking_error.png", dpi=150)
    plt.close(fig_err)

    fig_ctrl, axes_ctrl = plt.subplots(n_joints, 1, figsize=(10, 2.0 * n_joints), sharex=True)
    if n_joints == 1:
        axes_ctrl = [axes_ctrl]
    for idx in range(n_joints):
        axes_ctrl[idx].plot(time, actuator_q_ref[:, idx])
        axes_ctrl[idx].set_ylabel(f"q_ref{idx}")
    axes_ctrl[-1].set_xlabel("step")
    fig_ctrl.tight_layout()
    fig_ctrl.savefig(save_dir / "control.png", dpi=150)
    plt.close(fig_ctrl)

    fig_diag, axes_diag = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes_diag[0].plot(time, planning_time)
    axes_diag[0].set_ylabel("replan_time_s")
    axes_diag[1].plot(time, best_cost)
    axes_diag[1].set_ylabel("best_cost")
    axes_diag[1].set_xlabel("step")
    fig_diag.tight_layout()
    fig_diag.savefig(save_dir / "planning_diagnostics.png", dpi=150)
    plt.close(fig_diag)

    _plot_task_space_run(plt, save_dir, arrays)
