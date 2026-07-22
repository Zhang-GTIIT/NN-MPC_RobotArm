from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from copy import deepcopy
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.model_c._runtime import ROOT, ensure_import_paths, load_runner

ensure_import_paths()

import numpy as np

from mpc.logging import save_mpc_run


RUN_CEM_MPC = load_runner("model_c_evaluate_runner")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_model_spec(value: str) -> dict[str, object]:
    parts = [item.strip() for item in value.split(",")]
    if not parts or not parts[0] or "/" in parts[0] or "\\" in parts[0]:
        raise ValueError("--model_spec label must be a non-empty directory-safe name")
    if len(parts) == 2 and parts[1] == "direct_ik":
        return {"label": parts[0], "kind": "direct_ik", "dataset_path": ""}
    if len(parts) in {2, 4} and parts[1] == "oracle":
        spec: dict[str, object] = {"label": parts[0], "kind": "oracle", "dataset_path": ""}
        if len(parts) == 4:
            try:
                num_samples, cem_iters = int(parts[2]), int(parts[3])
            except ValueError as exc:
                raise ValueError("Oracle model spec must be label,oracle[,num_samples,cem_iters]") from exc
            if num_samples <= 0 or cem_iters <= 0:
                raise ValueError("Oracle num_samples and cem_iters must be positive")
            spec.update(num_samples=num_samples, cem_iters=cem_iters)
        return spec
    if len(parts) not in {4, 5}:
        raise ValueError(
            "--model_spec must be label,direct_ik; label,oracle[,num_samples,cem_iters]; "
            "or label,checkpoint,normalizer,model_type[,dataset_path], "
            f"got {value!r}"
        )
    return {
        "label": parts[0],
        "kind": "learned",
        "checkpoint": parts[1],
        "normalizer": parts[2],
        "model_type": parts[3],
        "dataset_path": "" if len(parts) == 4 else parts[4],
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = RUN_CEM_MPC.build_arg_parser()
    parser.description = "Evaluate Model A/B/C checkpoints with common CEM-MPC settings."
    for action in parser._actions:
        if action.dest in {"checkpoint", "normalizer"}:
            action.required = False
    parser.add_argument(
        "--model_spec",
        action="append",
        required=True,
        help="Repeat as label,direct_ik; label,oracle[,num_samples,cem_iters]; or learned-model fields.",
    )
    parser.add_argument("--manifest", default=None, help="JSON benchmark manifest with a cases list.")
    parser.add_argument("--allow_final_benchmark", action="store_true")
    parser.add_argument("--bootstrap_samples", type=int, default=2000)
    parser.add_argument("--bootstrap_seed", type=int, default=20260720)
    parser.add_argument("--case_ids", default=None, help="Optional comma-separated benchmark case IDs.")
    parser.add_argument("--resume", action="store_true", help="Reuse completed case/spec rollouts only when their fingerprint matches exactly.")
    parser.add_argument("--candidate_metrics", action="append", default=[], help="Optional label,path JSON produced by analyze_candidate_branches.py; repeatable.")
    parser.add_argument(
        "--include_no_feedback_ablation", action="store_true",
        help="Duplicate each learned model with feedback_kq=feedback_kdq=feedback_max=0.",
    )
    parser.set_defaults(save_dir="outputs/mpc/model_abc")
    return parser.parse_args(argv)


def expand_model_specs(
    model_specs: list[dict[str, object]], include_no_feedback_ablation: bool
) -> list[dict[str, object]]:
    """Add feedback-disabled learned baselines without mutating input specs."""
    expanded = [dict(spec) for spec in model_specs]
    if include_no_feedback_ablation:
        for spec in model_specs:
            if spec["kind"] == "learned":
                duplicate = dict(spec)
                duplicate["label"] = f"{spec['label']}_NoFeedback"
                duplicate["no_feedback"] = True
                expanded.append(duplicate)
    labels = [str(spec["label"]) for spec in expanded]
    if len(labels) != len(set(labels)):
        raise ValueError("Every --model_spec label must be unique, including generated NoFeedback labels")
    return expanded


def summarize(label: str, arrays: dict[str, np.ndarray], dataset_path: str) -> dict[str, float | str | int]:
    tracking = arrays["realized_tracking_error"]
    failures = arrays["failure_flags"]
    planning_time = arrays["planning_time"]
    predicted_q_error = arrays.get("predicted_next_q_error", np.empty(0))
    predicted_dq_error = arrays.get("predicted_next_dq_error", np.empty(0))
    replay_q_error = np.asarray(arrays.get("replay_q_error_norm", np.empty((0, 0))), dtype=np.float64)
    replay_dq_error = np.asarray(arrays.get("replay_dq_error_norm", np.empty((0, 0))), dtype=np.float64)
    replay_q_first = replay_q_error[:, 0] if replay_q_error.ndim == 2 and replay_q_error.shape[1] else np.empty(0)
    replay_q_terminal = replay_q_error[:, -1] if replay_q_error.ndim == 2 and replay_q_error.shape[1] else np.empty(0)
    replay_dq_first = replay_dq_error[:, 0] if replay_dq_error.ndim == 2 and replay_dq_error.shape[1] else np.empty(0)
    replay_dq_terminal = replay_dq_error[:, -1] if replay_dq_error.ndim == 2 and replay_dq_error.shape[1] else np.empty(0)
    finite_mean = lambda values: float(np.mean(values[np.isfinite(values)])) if np.any(np.isfinite(values)) else float("nan")
    finite_percentile = lambda values, q: float(np.percentile(values[np.isfinite(values)], q)) if np.any(np.isfinite(values)) else float("nan")
    replanned = np.asarray(arrays.get("mpc_replanned", np.empty(0)), dtype=bool)
    solve_mask = replanned if replanned.shape == planning_time.shape else np.isfinite(planning_time)
    solve_planning_time = planning_time[solve_mask] if solve_mask.size else planning_time
    best_cost = np.asarray(arrays.get("best_cost", np.empty(0)), dtype=np.float64)
    solve_best_cost = best_cost[solve_mask] if solve_mask.shape == best_cost.shape else best_cost
    actual_states = np.asarray(arrays.get("actual_states", np.empty((0, 0))), dtype=np.float64)
    q_des = np.asarray(arrays.get("q_des", np.empty((0, 0))), dtype=np.float64)
    joint_length = min(actual_states.shape[0], q_des.shape[0]) if actual_states.ndim == q_des.ndim == 2 else 0
    joint_rmse = float(np.sqrt(np.mean(np.square(actual_states[:joint_length, : q_des.shape[1]] - q_des[:joint_length])))) if joint_length and actual_states.shape[1] >= q_des.shape[1] else float("nan")
    position_error = np.asarray(arrays.get("ee_position_errors", np.empty(0)), dtype=np.float64)
    orientation_error = np.asarray(arrays.get("ee_orientation_errors", np.empty(0)), dtype=np.float64)
    packet_age = np.asarray(arrays.get("packet_age", np.empty(0)), dtype=np.float64)
    solve_count = int(np.asarray(arrays.get("planner_solve_count", np.sum(replanned))).reshape(-1)[0])
    late_drop_count = int(np.asarray(arrays.get("planner_late_drop_count", 0)).reshape(-1)[0])
    result: dict[str, float | str | int] = {
        "label": label,
        "dynamics_backend": str(np.asarray(arrays.get("dynamics_backend", "not_applicable")).reshape(-1)[0]),
        "dataset_path": dataset_path,
        "steps": int(len(tracking)),
        "tracking_error_mean": float(np.mean(tracking)) if len(tracking) else float("nan"),
        "tracking_error_final": float(tracking[-1]) if len(tracking) else float("nan"),
        "failure_rate": float(np.mean(failures)) if len(failures) else float("nan"),
        "planning_time_mean": finite_mean(solve_planning_time),
        "best_cost_mean": finite_mean(solve_best_cost),
        "predicted_next_q_error_mean": finite_mean(predicted_q_error),
        "predicted_next_dq_error_mean": finite_mean(predicted_dq_error),
        "tcp_position_rmse_m": float(np.sqrt(np.mean(np.square(position_error[np.isfinite(position_error)])))) if np.any(np.isfinite(position_error)) else float("nan"),
        "orientation_rmse_rad": float(np.sqrt(np.mean(np.square(orientation_error[np.isfinite(orientation_error)])))) if np.any(np.isfinite(orientation_error)) else float("nan"),
        "joint_position_rmse_rad": joint_rmse,
        "control_period_p99_s": finite_percentile(np.asarray(arrays.get("actual_control_period_s", np.empty(0)), dtype=np.float64), 99.0),
        "control_wakeup_lateness_p99_s": finite_percentile(np.asarray(arrays.get("control_wakeup_lateness_s", np.empty(0)), dtype=np.float64), 99.0),
        "control_compute_p99_s": finite_percentile(np.asarray(arrays.get("control_step_wall_time", np.empty(0)), dtype=np.float64), 99.0),
        "control_deadline_miss_count": int(np.sum(np.asarray(arrays.get("control_deadline_miss", np.empty(0))) != 0)),
        "planner_solve_count": solve_count,
        "planner_update_rate_hz": float(np.asarray(arrays.get("planner_actual_update_rate_hz", np.nan)).reshape(-1)[0]),
        "planner_late_drop_rate": float(late_drop_count / solve_count) if solve_count else float("nan"),
        "oracle_wall_time_deadline_miss_count": int(
            np.asarray(arrays.get("oracle_wall_time_deadline_miss_count", 0)).reshape(-1)[0]
        ),
        "active_packet_ratio": float(np.mean(packet_age >= 0.0)) if packet_age.size else float("nan"),
        "replay_q_error_k1_mean": finite_mean(replay_q_first),
        "replay_q_error_kH_mean": finite_mean(replay_q_terminal),
        "replay_dq_error_k1_mean": finite_mean(replay_dq_first),
        "replay_dq_error_kH_mean": finite_mean(replay_dq_terminal),
    }
    result.update(robustness_summary(arrays))
    return result


def robustness_summary(arrays: dict[str, np.ndarray]) -> dict[str, float | int]:
    """Summarise disturbance strength, recovery, noise, and command activity."""
    scalar_int = lambda key: int(np.asarray(arrays.get(key, 0)).reshape(-1)[0])
    scalar_float = lambda key, default=0.0: float(np.asarray(arrays.get(key, default)).reshape(-1)[0])
    actual = np.asarray(arrays.get("actual_states", np.empty((0, 0))), dtype=np.float64)
    observed = np.asarray(arrays.get("observed_states", np.empty((0, 0))), dtype=np.float64)
    n_joints = actual.shape[1] // 2 if actual.ndim == 2 and actual.shape[1] else 0
    noise = observed - actual if observed.shape == actual.shape else np.empty((0, 0))
    position_error = np.asarray(arrays.get("ee_position_errors", np.empty(0)), dtype=np.float64)
    tracking = position_error if position_error.size else np.asarray(arrays.get("realized_tracking_error", np.empty(0)), dtype=np.float64)
    start = scalar_int("force_pulse_start_step")
    stop = scalar_int("force_pulse_stop_step")
    force_level = scalar_int("force_pulse_level")
    recovery_steps = float("nan")
    peak_post = float("nan")
    integral_above_baseline = float("nan")
    if force_level and tracking.size and 0 < start < len(tracking):
        finite_tracking = np.where(np.isfinite(tracking), tracking, np.nan)
        baseline_slice = finite_tracking[max(0, start - 50) : start]
        baseline = float(np.nanmean(baseline_slice)) if np.any(np.isfinite(baseline_slice)) else 0.0
        threshold_floor = 0.005 if position_error.size else 0.05
        threshold = max(1.5 * baseline, threshold_floor)
        post = finite_tracking[min(stop, len(tracking)) :]
        peak_region = finite_tracking[start:]
        peak_post = float(np.nanmax(peak_region)) if np.any(np.isfinite(peak_region)) else float("nan")
        integral_above_baseline = float(np.nansum(np.maximum(peak_region - baseline, 0.0)) * 0.01)
        for offset in range(max(0, len(post) - 9)):
            window = post[offset : offset + 10]
            if len(window) == 10 and np.all(np.isfinite(window)) and np.all(window <= threshold):
                recovery_steps = float(offset)
                break
    residual = np.asarray(arrays.get("executed_residual", np.empty((0, 0))), dtype=np.float64)
    residual_max = np.asarray(arrays.get("residual_max", np.empty(0)), dtype=np.float64).reshape(-1)
    residual_saturation = (
        float(np.mean(np.any(np.abs(residual) >= 0.95 * residual_max, axis=1)))
        if residual.ndim == 2 and residual.size and residual.shape[1] == residual_max.size else float("nan")
    )
    feedback = np.asarray(arrays.get("feedback_correction", np.empty((0, 0))), dtype=np.float64)
    feedback_max = np.asarray(arrays.get("feedback_max", np.empty(0)), dtype=np.float64).reshape(-1)
    feedback_saturation = (
        float(np.mean(np.any(np.abs(feedback) >= 0.95 * feedback_max, axis=1)))
        if feedback.ndim == 2 and feedback.size and feedback.shape[1] == feedback_max.size and np.any(feedback_max > 0)
        else float("nan")
    )
    acceleration = np.asarray(arrays.get("command_acceleration", np.empty((0, 0))), dtype=np.float64)
    command = np.asarray(arrays.get("actuator_q_ref", np.empty((0, 0))), dtype=np.float64)
    packet_age = np.asarray(arrays.get("packet_age", np.empty(0)), dtype=np.float64)
    return {
        "payload_level": scalar_int("payload_level"),
        "actuator_gain_level": scalar_int("actuator_gain_level"),
        "force_pulse_level": force_level,
        "observation_noise_level": scalar_int("observation_noise_level"),
        "payload_mass_kg": scalar_float("payload_mass_kg"),
        "actuator_kp_scale": scalar_float("actuator_kp_scale", 1.0),
        "actuator_kd_scale": scalar_float("actuator_kd_scale", 1.0),
        "force_pulse_n": scalar_float("force_pulse_n"),
        "observation_q_std_rad": scalar_float("observation_q_std_rad"),
        "observation_dq_std_rad_s": scalar_float("observation_dq_std_rad_s"),
        "observation_q_rmse_rad": float(np.sqrt(np.mean(noise[:, :n_joints] ** 2))) if noise.size and n_joints else float("nan"),
        "observation_dq_rmse_rad_s": float(np.sqrt(np.mean(noise[:, n_joints:] ** 2))) if noise.size and n_joints else float("nan"),
        "force_peak_tracking_error": peak_post,
        "force_integrated_error_above_baseline": integral_above_baseline,
        "force_recovery_steps": recovery_steps,
        "force_recovery_time_s": recovery_steps * 0.01 if np.isfinite(recovery_steps) else float("nan"),
        "residual_saturation_rate": residual_saturation,
        "feedback_saturation_rate": feedback_saturation,
        "feedback_rms_rad": float(np.sqrt(np.mean(feedback**2))) if feedback.size else float("nan"),
        "command_acceleration_rms_rad_s2": float(np.sqrt(np.mean(acceleration**2))) if acceleration.size else float("nan"),
        "command_total_variation_rad": float(np.sum(np.abs(np.diff(command, axis=0)))) if command.ndim == 2 and len(command) > 1 else float("nan"),
        "direct_ik_fallback_rate": float(np.mean(packet_age < 0)) if packet_age.size else float("nan"),
    }


def write_summary(path: Path, rows: list[dict[str, float | str | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_candidate_metrics(values: list[str]) -> dict[str, object]:
    result: dict[str, object] = {}
    for value in values:
        label, separator, path_text = value.partition(",")
        if not separator or not label or not path_text:
            raise ValueError("--candidate_metrics must be label,path")
        path = RUN_CEM_MPC.resolve_runtime_path(path_text)
        if label in result:
            raise ValueError(f"Duplicate candidate metrics label: {label}")
        result[label] = json.loads(path.read_text(encoding="utf-8"))
    return result


def paired_bootstrap(rows: list[dict[str, float | str | int]], samples: int, seed: int) -> dict[str, object]:
    """Return paired, case-level bootstrap intervals for all model pairs.

    Lower is better for the reported metrics.  Pairing by benchmark case keeps
    reference, initial state and CEM seed fixed within each comparison.
    """
    labels = sorted({str(row["label"]) for row in rows})
    metrics = (
        "failure_rate", "tracking_error_mean", "joint_position_rmse_rad", "replay_q_error_kH_mean",
        "force_peak_tracking_error", "force_recovery_time_s", "force_integrated_error_above_baseline",
        "command_acceleration_rms_rad_s2", "command_total_variation_rad",
    )
    rng = np.random.default_rng(seed)
    output: dict[str, object] = {"bootstrap_samples": samples, "seed": seed, "comparisons": {}}
    by_label_case = {(str(row["label"]), str(row["case_id"])): row for row in rows}
    for left_index, left in enumerate(labels):
        for right in labels[left_index + 1 :]:
            case_ids = sorted(set(case for label, case in by_label_case if label == left) & set(case for label, case in by_label_case if label == right))
            comparison: dict[str, object] = {"paired_cases": case_ids, "metrics": {}}
            for metric in metrics:
                deltas = np.asarray([
                    float(by_label_case[(right, case)][metric]) - float(by_label_case[(left, case)][metric])
                    for case in case_ids
                    if np.isfinite(float(by_label_case[(left, case)][metric])) and np.isfinite(float(by_label_case[(right, case)][metric]))
                ], dtype=np.float64)
                if not len(deltas):
                    comparison["metrics"][metric] = {"n": 0, "mean_delta_right_minus_left": float("nan"), "ci95": [float("nan"), float("nan")]}
                    continue
                draws = deltas[rng.integers(0, len(deltas), size=(samples, len(deltas)))].mean(axis=1)
                comparison["metrics"][metric] = {
                    "n": int(len(deltas)), "mean_delta_right_minus_left": float(deltas.mean()),
                    "ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
                }
            output["comparisons"][f"{right}_minus_{left}"] = comparison
    return output


_FILE_IDENTITY_CACHE: dict[str, dict[str, object]] = {}


def _input_file_identity(value: object) -> dict[str, object] | None:
    text = str(value or "").strip()
    if not text:
        return None
    path = RUN_CEM_MPC.resolve_runtime_path(text)
    if not path.is_file():
        return {"path": str(path), "exists": False}
    cache_key = str(path.resolve())
    if cache_key not in _FILE_IDENTITY_CACHE:
        _FILE_IDENTITY_CACHE[cache_key] = {
            "path": cache_key, "size": path.stat().st_size, "sha256": sha256(path)
        }
    return dict(_FILE_IDENTITY_CACHE[cache_key])


def build_run_fingerprint(
    case_id: str,
    case: dict[str, object],
    spec: dict[str, object],
    run_args: argparse.Namespace,
) -> dict[str, object]:
    """Describe every behavior-affecting input used by one benchmark rollout."""
    excluded = {
        "allow_final_benchmark", "bootstrap_samples", "bootstrap_seed", "candidate_metrics",
        "case_ids", "include_no_feedback_ablation", "manifest", "model_spec", "resume", "save_dir",
    }
    run_config = {key: value for key, value in vars(run_args).items() if key not in excluded}
    robustness = RUN_CEM_MPC._robustness_config(run_args)
    payload = {
        "schema_version": 1,
        "case_id": case_id,
        "case": case,
        "model_spec": spec,
        "run_config": run_config,
        "checkpoint": _input_file_identity(spec.get("checkpoint")),
        "normalizer": _input_file_identity(spec.get("normalizer")),
        "reference": _input_file_identity(run_config.get("reference_file")),
        "nominal_model": _input_file_identity(robustness.nominal_model_xml),
        "plant_model": _input_file_identity(robustness.plant_model_xml),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return {"sha256": hashlib.sha256(canonical).hexdigest(), "payload": payload}


def load_completed_rollout(run_dir: Path, expected_fingerprint: dict[str, object]) -> dict[str, np.ndarray] | None:
    rollout_path = run_dir / "rollout.npz"
    fingerprint_path = run_dir / "run_fingerprint.json"
    if not rollout_path.exists() and not fingerprint_path.exists():
        return None
    if not rollout_path.exists() or not fingerprint_path.exists():
        raise RuntimeError(f"Incomplete resumable output at {run_dir}; move it aside or rerun without --resume")
    actual = json.loads(fingerprint_path.read_text(encoding="utf-8"))
    if actual.get("sha256") != expected_fingerprint.get("sha256"):
        raise RuntimeError(f"Resume fingerprint mismatch at {run_dir}; refusing to reuse a different experiment")
    with np.load(rollout_path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def main() -> None:
    args = parse_args()
    save_root = RUN_CEM_MPC.resolve_runtime_path(args.save_dir)
    manifest: dict[str, object] = {"cases": [{"id": "default"}]}
    if args.manifest:
        manifest_path = RUN_CEM_MPC.resolve_runtime_path(args.manifest)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("kind") == "final":
            marker = manifest.get("completion_marker")
            marker_path = manifest_path.parent / str(marker) if marker else None
            if not args.allow_final_benchmark or marker_path is None or not marker_path.exists():
                raise PermissionError("Final benchmark requires --allow_final_benchmark and its C2 completion marker")
    cases = manifest.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("Benchmark manifest must contain a non-empty cases list")
    requested_case_ids = None
    if args.case_ids:
        requested_case_ids = {value.strip() for value in args.case_ids.split(",") if value.strip()}
        if not requested_case_ids:
            raise ValueError("--case_ids must contain at least one non-empty case ID")
        available_case_ids = {str(case.get("id")) for case in cases if isinstance(case, dict)}
        missing_case_ids = requested_case_ids.difference(available_case_ids)
        if missing_case_ids:
            raise ValueError(f"Unknown --case_ids: {sorted(missing_case_ids)}")
        cases = [case for case in cases if isinstance(case, dict) and str(case.get("id")) in requested_case_ids]
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("Each benchmark case must be an object")
        expected_hash = case.get("reference_sha256")
        reference_file = case.get("run_args", case).get("reference_file")
        if expected_hash is not None:
            if not isinstance(reference_file, str) or sha256(Path(reference_file)) != expected_hash:
                raise ValueError(f"Benchmark reference hash mismatch for case {case.get('id')}")
    model_specs = expand_model_specs(
        [parse_model_spec(value) for value in args.model_spec], args.include_no_feedback_ablation
    )
    rows: list[dict[str, float | str | int]] = []
    robustness_overrides = {
        key: int(getattr(args, key))
        for key in ("payload_level", "actuator_gain_level", "force_pulse_level", "observation_noise_level")
    }
    for case_index, case in enumerate(cases):
        if not isinstance(case, dict):
            raise ValueError("Each benchmark case must be an object")
        case_id = str(case.get("id", f"case_{case_index:03d}"))
        for spec in model_specs:
            kind = str(spec["kind"])
            if kind == "direct_ik" and str(case.get("run_args", case).get("reference_mode", "")) != "task":
                # Direct IK is only defined on an IK-derived task reference;
                # retain it as a baseline on the task-space subset instead of
                # silently comparing a different controller on joint signals.
                continue
            run_args = deepcopy(args)
            for key, value in case.get("run_args", case).items():
                if key != "id" and hasattr(run_args, key):
                    setattr(run_args, key, value)
            for key, value in robustness_overrides.items():
                setattr(run_args, key, value)
            if kind == "direct_ik":
                run_args.controller_mode = "ik_direct"
                run_args.multirate_mode = "synchronous"
                run_args.dynamics_backend = "learned"
            elif kind == "oracle":
                run_args.controller_mode = "mpc"
                run_args.mpc_policy = "residual"
                run_args.multirate_mode = "virtual_asap"
                run_args.dynamics_backend = "mujoco_oracle"
                run_args.checkpoint = None
                run_args.normalizer = None
                if "num_samples" in spec:
                    run_args.num_samples = int(spec["num_samples"])
                    run_args.cem_iters = int(spec["cem_iters"])
            else:
                run_args.dynamics_backend = "learned"
                run_args.checkpoint = str(spec["checkpoint"])
                run_args.normalizer = str(spec["normalizer"])
                run_args.model_type = str(spec["model_type"])
                if spec.get("no_feedback"):
                    run_args.feedback_kq = 0.0
                    run_args.feedback_kdq = 0.0
                    run_args.feedback_max = "0"
            run_dir = save_root / case_id / str(spec["label"])
            run_args.save_dir = str(run_dir)
            fingerprint = build_run_fingerprint(case_id, case, spec, run_args)
            arrays = load_completed_rollout(run_dir, fingerprint) if args.resume else None
            if arrays is None:
                result = RUN_CEM_MPC.run_closed_loop_mpc(run_args)
                arrays = result["arrays"]
                save_mpc_run(run_dir, arrays, result["rows"])
                (run_dir / "run_fingerprint.json").write_text(
                    json.dumps(fingerprint, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
                )
            else:
                print(f"Reused completed rollout: {run_dir}")
            row = summarize(str(spec["label"]), arrays, str(spec["dataset_path"]))
            row["case_id"] = case_id
            rows.append(row)
            write_summary(save_root / "model_abc_summary.csv", rows)
    write_summary(save_root / "model_abc_summary.csv", rows)
    (save_root / "paired_bootstrap.json").write_text(
        json.dumps(paired_bootstrap(rows, args.bootstrap_samples, args.bootstrap_seed), indent=2, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    if args.candidate_metrics:
        (save_root / "candidate_metrics.json").write_text(
            json.dumps(load_candidate_metrics(args.candidate_metrics), indent=2, allow_nan=True) + "\n",
            encoding="utf-8",
        )
    print(f"Saved Model A/B/C summary to {save_root / 'model_abc_summary.csv'}")


if __name__ == "__main__":
    main()
