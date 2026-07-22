"""Pure reporting and resume helpers for robustness benchmark runners."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np

from scripts.experiment_utils.hashing import canonical_sha256, file_identity as common_file_identity, sha256_file
from scripts.experiment_utils.resume import load_completed_rollout as common_load_completed_rollout


def sha256(path: Path) -> str:
    return sha256_file(path)


def write_summary(path: Path, rows: list[dict[str, float | str | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _finite_mean(values: np.ndarray) -> float:
    return float(np.mean(values[np.isfinite(values)])) if np.any(np.isfinite(values)) else float("nan")


def _finite_percentile(values: np.ndarray, percentile: float) -> float:
    return float(np.percentile(values[np.isfinite(values)], percentile)) if np.any(np.isfinite(values)) else float("nan")


def robustness_summary(arrays: dict[str, np.ndarray]) -> dict[str, float | int]:
    scalar_int = lambda key: int(np.asarray(arrays.get(key, 0)).reshape(-1)[0])
    scalar_float = lambda key, default=0.0: float(np.asarray(arrays.get(key, default)).reshape(-1)[0])
    actual = np.asarray(arrays.get("actual_states", np.empty((0, 0))), dtype=np.float64)
    observed = np.asarray(arrays.get("observed_states", np.empty((0, 0))), dtype=np.float64)
    n_joints = actual.shape[1] // 2 if actual.ndim == 2 and actual.shape[1] else 0
    noise = observed - actual if observed.shape == actual.shape else np.empty((0, 0))
    position_error = np.asarray(arrays.get("ee_position_errors", np.empty(0)), dtype=np.float64)
    tracking = position_error if position_error.size else np.asarray(arrays.get("realized_tracking_error", np.empty(0)), dtype=np.float64)
    force_level, start, stop = scalar_int("force_pulse_level"), scalar_int("force_pulse_start_step"), scalar_int("force_pulse_stop_step")
    recovery_steps = peak = integrated = float("nan")
    if force_level and tracking.size and 0 < start < len(tracking):
        finite = np.where(np.isfinite(tracking), tracking, np.nan)
        baseline_values = finite[max(0, start - 50):start]
        baseline = float(np.nanmean(baseline_values)) if np.any(np.isfinite(baseline_values)) else 0.0
        threshold = max(1.5 * baseline, 0.005 if position_error.size else 0.05)
        post, full_post = finite[min(stop, len(finite)):], finite[start:]
        peak = float(np.nanmax(full_post)) if np.any(np.isfinite(full_post)) else float("nan")
        integrated = float(np.nansum(np.maximum(full_post - baseline, 0.0)) * 0.01)
        for offset in range(max(0, len(post) - 9)):
            window = post[offset:offset + 10]
            if len(window) == 10 and np.all(np.isfinite(window)) and np.all(window <= threshold):
                recovery_steps = float(offset)
                break
    residual = np.asarray(arrays.get("executed_residual", np.empty((0, 0))), dtype=np.float64)
    residual_max = np.asarray(arrays.get("residual_max", np.empty(0)), dtype=np.float64).reshape(-1)
    residual_saturation = float(np.mean(np.any(np.abs(residual) >= 0.95 * residual_max, axis=1))) if residual.ndim == 2 and residual.size and residual.shape[1] == residual_max.size else float("nan")
    feedback = np.asarray(arrays.get("feedback_correction", np.empty((0, 0))), dtype=np.float64)
    feedback_max = np.asarray(arrays.get("feedback_max", np.empty(0)), dtype=np.float64).reshape(-1)
    feedback_saturation = float(np.mean(np.any(np.abs(feedback) >= 0.95 * feedback_max, axis=1))) if feedback.ndim == 2 and feedback.size and feedback.shape[1] == feedback_max.size and np.any(feedback_max > 0) else float("nan")
    acceleration = np.asarray(arrays.get("command_acceleration", np.empty((0, 0))), dtype=np.float64)
    command = np.asarray(arrays.get("actuator_q_ref", np.empty((0, 0))), dtype=np.float64)
    packet_age = np.asarray(arrays.get("packet_age", np.empty(0)), dtype=np.float64)
    return {
        "payload_level": scalar_int("payload_level"), "actuator_gain_level": scalar_int("actuator_gain_level"),
        "force_pulse_level": force_level, "observation_noise_level": scalar_int("observation_noise_level"),
        "payload_mass_kg": scalar_float("payload_mass_kg"),
        "actuator_kp_scale": scalar_float("actuator_kp_scale", 1.0), "actuator_kd_scale": scalar_float("actuator_kd_scale", 1.0),
        "force_pulse_n": scalar_float("force_pulse_n"),
        "observation_q_std_rad": scalar_float("observation_q_std_rad"), "observation_dq_std_rad_s": scalar_float("observation_dq_std_rad_s"),
        "observation_q_rmse_rad": float(np.sqrt(np.mean(noise[:, :n_joints] ** 2))) if noise.size and n_joints else float("nan"),
        "observation_dq_rmse_rad_s": float(np.sqrt(np.mean(noise[:, n_joints:] ** 2))) if noise.size and n_joints else float("nan"),
        "force_peak_tracking_error": peak, "force_integrated_error_above_baseline": integrated,
        "force_recovery_steps": recovery_steps, "force_recovery_time_s": recovery_steps * 0.01 if np.isfinite(recovery_steps) else float("nan"),
        "residual_saturation_rate": residual_saturation, "feedback_saturation_rate": feedback_saturation,
        "feedback_rms_rad": float(np.sqrt(np.mean(feedback ** 2))) if feedback.size else float("nan"),
        "command_acceleration_rms_rad_s2": float(np.sqrt(np.mean(acceleration ** 2))) if acceleration.size else float("nan"),
        "command_total_variation_rad": float(np.sum(np.abs(np.diff(command, axis=0)))) if command.ndim == 2 and len(command) > 1 else float("nan"),
        "direct_ik_fallback_rate": float(np.mean(packet_age < 0.0)) if packet_age.size else float("nan"),
    }


def summarize(label: str, arrays: dict[str, np.ndarray]) -> dict[str, float | str | int]:
    tracking = np.asarray(arrays["realized_tracking_error"], dtype=np.float64)
    failures = np.asarray(arrays["failure_flags"], dtype=np.float64)
    planning = np.asarray(arrays["planning_time"], dtype=np.float64)
    replanned = np.asarray(arrays.get("mpc_replanned", np.empty(0)), dtype=bool)
    solve_planning = planning[replanned] if replanned.shape == planning.shape else planning
    actual = np.asarray(arrays.get("actual_states", np.empty((0, 0))), dtype=np.float64)
    desired = np.asarray(arrays.get("q_des", np.empty((0, 0))), dtype=np.float64)
    length = min(len(actual), len(desired)) if actual.ndim == desired.ndim == 2 else 0
    joint_rmse = float(np.sqrt(np.mean((actual[:length, :desired.shape[1]] - desired[:length]) ** 2))) if length and actual.shape[1] >= desired.shape[1] else float("nan")
    position = np.asarray(arrays.get("ee_position_errors", np.empty(0)), dtype=np.float64)
    orientation = np.asarray(arrays.get("ee_orientation_errors", np.empty(0)), dtype=np.float64)
    packet_age = np.asarray(arrays.get("packet_age", np.empty(0)), dtype=np.float64)
    solve_count = int(np.asarray(arrays.get("planner_solve_count", np.sum(replanned))).reshape(-1)[0])
    late_drop_count = int(np.asarray(arrays.get("planner_late_drop_count", 0)).reshape(-1)[0])
    result: dict[str, float | str | int] = {
        "label": label, "dynamics_backend": str(np.asarray(arrays.get("dynamics_backend", "not_applicable")).reshape(-1)[0]),
        "steps": int(len(tracking)), "tracking_error_mean": float(np.mean(tracking)) if len(tracking) else float("nan"),
        "tracking_error_final": float(tracking[-1]) if len(tracking) else float("nan"), "failure_rate": float(np.mean(failures)) if len(failures) else float("nan"),
        "planning_time_mean": _finite_mean(solve_planning), "joint_position_rmse_rad": joint_rmse,
        "tcp_position_rmse_m": float(np.sqrt(np.mean(position[np.isfinite(position)] ** 2))) if np.any(np.isfinite(position)) else float("nan"),
        "orientation_rmse_rad": float(np.sqrt(np.mean(orientation[np.isfinite(orientation)] ** 2))) if np.any(np.isfinite(orientation)) else float("nan"),
        "control_period_p99_s": _finite_percentile(np.asarray(arrays.get("actual_control_period_s", np.empty(0)), dtype=np.float64), 99),
        "control_compute_p99_s": _finite_percentile(np.asarray(arrays.get("control_step_wall_time", np.empty(0)), dtype=np.float64), 99),
        "control_deadline_miss_count": int(np.sum(np.asarray(arrays.get("control_deadline_miss", np.empty(0))) != 0)),
        "planner_solve_count": solve_count, "planner_update_rate_hz": float(np.asarray(arrays.get("planner_actual_update_rate_hz", np.nan)).reshape(-1)[0]),
        "planner_late_drop_rate": float(late_drop_count / solve_count) if solve_count else float("nan"),
        "active_packet_ratio": float(np.mean(packet_age >= 0.0)) if packet_age.size else float("nan"),
    }
    result.update(robustness_summary(arrays))
    return result


def paired_bootstrap(rows: list[dict[str, float | str | int]], samples: int, seed: int) -> dict[str, Any]:
    metrics = ("failure_rate", "tracking_error_mean", "joint_position_rmse_rad", "tcp_position_rmse_m", "force_peak_tracking_error", "force_recovery_time_s", "force_integrated_error_above_baseline", "command_acceleration_rms_rad_s2", "command_total_variation_rad")
    labels = sorted({str(row["label"]) for row in rows})
    by_key = {(str(row["label"]), str(row["case_id"])): row for row in rows}
    rng, comparisons = np.random.default_rng(seed), {}
    for left_index, left in enumerate(labels):
        for right in labels[left_index + 1:]:
            cases = sorted({case for label, case in by_key if label == left} & {case for label, case in by_key if label == right})
            report: dict[str, Any] = {"paired_cases": cases, "metrics": {}}
            for metric in metrics:
                delta = np.asarray([float(by_key[(right, case)][metric]) - float(by_key[(left, case)][metric]) for case in cases if np.isfinite(float(by_key[(left, case)][metric])) and np.isfinite(float(by_key[(right, case)][metric]))])
                if not len(delta):
                    report["metrics"][metric] = {"n": 0, "mean_delta_right_minus_left": float("nan"), "ci95": [float("nan"), float("nan")]}
                    continue
                draws = delta[rng.integers(0, len(delta), size=(samples, len(delta)))].mean(axis=1)
                report["metrics"][metric] = {"n": int(len(delta)), "mean_delta_right_minus_left": float(delta.mean()), "ci95": [float(np.quantile(draws, .025)), float(np.quantile(draws, .975))]}
            comparisons[f"{right}_minus_{left}"] = report
    return {"bootstrap_samples": samples, "seed": seed, "comparisons": comparisons}


def file_identity(value: object, resolve_path: Callable[[str], Path]) -> dict[str, object] | None:
    text = str(value or "").strip()
    if not text:
        return None
    path = resolve_path(text)
    if not path.is_file():
        return {"path": str(path), "exists": False}
    return common_file_identity(path)


def build_fingerprint(case_id: str, case: dict[str, object], controller: dict[str, object], run_args: argparse.Namespace, runner: Any) -> dict[str, object]:
    excluded = {"bootstrap_samples", "bootstrap_seed", "case_ids", "manifest", "resume", "save_dir", "label", "direct_ik_label"}
    config = {key: value for key, value in vars(run_args).items() if key not in excluded}
    robustness = runner._robustness_config(run_args)
    payload = {"schema_version": 1, "case_id": case_id, "case": case, "controller": controller, "run_config": config,
               "checkpoint": file_identity(controller.get("checkpoint"), runner.resolve_runtime_path),
               "normalizer": file_identity(controller.get("normalizer"), runner.resolve_runtime_path),
               "reference": file_identity(config.get("reference_file"), runner.resolve_runtime_path),
               "nominal_model": file_identity(robustness.nominal_model_xml, runner.resolve_runtime_path),
               "plant_model": file_identity(robustness.plant_model_xml, runner.resolve_runtime_path)}
    return {"sha256": canonical_sha256(payload), "payload": payload}


def load_completed_rollout(run_dir: Path, expected: dict[str, object]) -> dict[str, np.ndarray] | None:
    return common_load_completed_rollout(run_dir, expected)
