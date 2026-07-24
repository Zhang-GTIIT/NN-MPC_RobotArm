"""Evaluate four delay-aware learned-MPC protocols under fixed perturbations."""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from scripts.experiment_utils.hashing import canonical_sha256, file_identity
from scripts.experiment_utils.resume import load_completed_rollout, run_fingerprint
from scripts.paper_experiments.evaluation import summarize_arrays
from scripts.robustness import evaluate_direct_ik as direct
from scripts.robustness._evaluation import robustness_summary
from scripts.robustness._runtime import ROOT, ensure_import_paths, load_runner

ensure_import_paths()

from mpc.logging import save_mpc_run


RUNNER = load_runner("delay_aware_mpc_robustness_runner")
DEFAULT_CHECKPOINT = "dynamics_modeling/outputs/checkpoints/gru_20260717_182930/best_model.pt"
DEFAULT_NORMALIZER = "dynamics_modeling/outputs/checkpoints/gru_20260717_182930/normalizer.pt"
DEFAULT_DELAY_CALIBRATION = "outputs/robustness/timing/gru_20260717_182930.json"
METRICS = (
    "tcp_rmse_m",
    "tcp_p95_m",
    "orientation_rmse_rad",
    "joint_rmse_rad",
    "failure_rate",
    "planner_failure_step_rate",
    "fallback_step_rate",
    "projection_activation_rate",
    "feedback_saturation_rate",
    "command_acceleration_rms_rad_s2",
    "command_total_variation_rad",
    "actuator_torque_rms_nm",
    "joint_violation_rate",
    "velocity_violation_rate",
    "acceleration_violation_rate",
    "planner_hz",
    "late_packet_rate",
    "control_deadline_miss_count",
    "force_peak_tracking_error",
    "force_integrated_error_above_baseline",
    "force_recovery_time_s",
)
PAIR_METRICS = (
    "tcp_rmse_m",
    "tcp_p95_m",
    "orientation_rmse_rad",
    "joint_rmse_rad",
    "failure_rate",
    "fallback_step_rate",
    "command_acceleration_rms_rad_s2",
    "command_total_variation_rad",
    "actuator_torque_rms_nm",
    "force_peak_tracking_error",
    "force_integrated_error_above_baseline",
    "force_recovery_time_s",
)


@dataclass(frozen=True)
class Method:
    name: str
    multirate_mode: str
    delay_protocol: str
    uses_delay: bool


METHODS = (
    Method("IdealZeroDelay", "virtual_asap", "full", False),
    Method("NaiveDelayed", "virtual_asap", "naive_delayed", True),
    Method("VirtualDelayAware", "virtual_asap", "full", True),
    Method("ThreadedAsync", "threaded_asap", "full", True),
)
METHOD_BY_NAME = {method.name: method for method in METHODS}
COLORS = {
    "IdealZeroDelay": "#1b9e77",
    "NaiveDelayed": "#d95f02",
    "VirtualDelayAware": "#7570b3",
    "ThreadedAsync": "#e7298a",
}


@dataclass(frozen=True)
class ExperimentSpec:
    method: Method
    condition: direct.Condition
    case_id: str
    reference_type: str
    case: dict[str, Any]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = RUNNER.build_arg_parser()
    parser.description = "Evaluate ideal, naive-delayed, virtual delay-aware, and threaded asynchronous MPC robustness."
    parser.add_argument("--manifest", default="outputs/robustness/benchmark.json")
    parser.add_argument("--delay_calibration", default=DEFAULT_DELAY_CALIBRATION)
    parser.add_argument("--case_ids", default=direct.DEFAULT_CASE_IDS)
    parser.add_argument("--levels", default="0,3,6")
    parser.add_argument("--perturbations", default=",".join(direct.PERTURBATIONS))
    parser.add_argument("--methods", default=",".join(method.name for method in METHODS))
    parser.add_argument(
        "--seeds",
        default="",
        help="Optional comma-separated seeds. When provided, repeat every selected case for each seed.",
    )
    parser.add_argument("--bootstrap_samples", type=int, default=2000)
    parser.add_argument("--bootstrap_seed", type=int, default=20260722)
    parser.add_argument("--resume", action="store_true")
    parser.set_defaults(
        checkpoint=DEFAULT_CHECKPOINT,
        normalizer=DEFAULT_NORMALIZER,
        model_type="gru",
        history_len=16,
        device="cuda",
        save_dir="outputs/robustness/delay_aware_mpc_medium",
    )
    return parser.parse_args(argv)


def load_delay(path: Path) -> int:
    payload = json.loads(path.read_text(encoding="utf-8"))
    delay = int(payload.get("anticipation_delay_steps", -1))
    if delay <= 0:
        raise ValueError("Delay calibration must contain a positive anticipation_delay_steps")
    return delay


def build_specs(
    cases: Iterable[dict[str, Any]],
    conditions: Iterable[direct.Condition],
    methods: Iterable[Method],
) -> list[ExperimentSpec]:
    return [
        ExperimentSpec(
            method=method,
            condition=condition,
            case_id=str(case["id"]),
            reference_type=str(case.get("reference_type", "unknown")),
            case=case,
        )
        for method in methods
        for condition in conditions
        for case in cases
    ]


def _run_args(
    base: argparse.Namespace,
    spec: ExperimentSpec,
    run_dir: Path,
    delay_steps: int,
) -> argparse.Namespace:
    args = deepcopy(base)
    values = spec.case.get("run_args", spec.case)
    assert isinstance(values, dict)
    for key, value in values.items():
        if key != "id" and hasattr(args, key):
            setattr(args, key, value)
    args.controller_mode = "mpc"
    args.mpc_policy = "residual"
    args.dynamics_backend = "learned"
    args.checkpoint = base.checkpoint
    args.normalizer = base.normalizer
    args.model_type = "gru"
    args.history_len = 16
    args.device = base.device
    args.planner_projection = base.planner_projection
    args.planner_projection_backend = base.planner_projection_backend
    args.planner_projection_strategy = base.planner_projection_strategy
    args.reference_mode = "task"
    args.multirate_mode = spec.method.multirate_mode
    args.delay_protocol = spec.method.delay_protocol
    args.anticipation_delay_steps = delay_steps if spec.method.uses_delay else 0
    args.max_execution_steps = base.max_execution_steps
    args.visualize = False
    args.save_dir = str(run_dir)
    for argument, level in spec.condition.levels.items():
        setattr(args, argument, level)
    return args


def _fingerprint(
    spec: ExperimentSpec,
    args: argparse.Namespace,
    delay_calibration: Path,
) -> dict[str, Any]:
    excluded = {
        "bootstrap_samples",
        "bootstrap_seed",
        "case_ids",
        "delay_calibration",
        "levels",
        "manifest",
        "methods",
        "perturbations",
        "resume",
        "seeds",
        "save_dir",
    }
    config = {key: value for key, value in vars(args).items() if key not in excluded}
    reference = Path(str(args.reference_file))
    if not reference.is_absolute():
        reference = ROOT / reference
    robustness = RUNNER._robustness_config(args)
    return run_fingerprint(
        {
            "kind": "delay_aware_mpc_robustness",
            "method": spec.method.name,
            "case_id": spec.case_id,
            "reference_type": spec.reference_type,
            "condition": {
                "name": spec.condition.name,
                "perturbation": spec.condition.perturbation,
                "level": spec.condition.level,
                "levels": spec.condition.levels,
            },
            "run_config": config,
            "reference": file_identity(reference),
            "checkpoint": file_identity(RUNNER.resolve_runtime_path(args.checkpoint)),
            "normalizer": file_identity(RUNNER.resolve_runtime_path(args.normalizer)),
            "delay_calibration": file_identity(delay_calibration),
            "nominal_model": file_identity(robustness.nominal_model_xml),
            "plant_model": file_identity(robustness.plant_model_xml),
        }
    )


def summarize_mpc(spec: ExperimentSpec, arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    row = summarize_arrays(spec.method.name, arrays)
    row.update(robustness_summary(arrays))
    steps = max(int(row["steps"]), 1)
    command = np.asarray(arrays.get("actuator_q_ref", np.empty((0, 0))), dtype=np.float64)
    fallback = np.asarray(arrays.get("fallback_active", np.empty(0)), dtype=np.float64)
    if fallback.size:
        fallback_rate = float(np.mean(fallback != 0))
    else:
        packet_age = np.asarray(arrays.get("packet_age", np.empty(0)), dtype=np.float64)
        fallback_rate = float(np.mean(packet_age < 0)) if packet_age.size else float("nan")
    row.update(
        {
            "method": spec.method.name,
            "case_id": spec.case_id,
            "reference_type": spec.reference_type,
            "condition": spec.condition.name,
            "perturbation": spec.condition.perturbation,
            "level": spec.condition.level,
            "fallback_step_rate": fallback_rate,
            "joint_violation_rate": float(row["joint_violation_count"]) / steps,
            "velocity_violation_rate": float(row["velocity_violation_count"]) / steps,
            "acceleration_violation_rate": float(row["acceleration_violation_count"]) / steps,
            "command_total_variation_rad": (
                float(np.sum(np.abs(np.diff(command, axis=0))))
                if command.ndim == 2 and len(command) > 1
                else float("nan")
            ),
        }
    )
    return row


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _finite(values: Iterable[Any]) -> np.ndarray:
    result = np.asarray([float(value) for value in values], dtype=np.float64)
    return result[np.isfinite(result)]


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fields = ("method", "condition", "perturbation", "level", "reference_type")
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = tuple(row[field] for field in fields)
        groups.setdefault(key, []).append(row)
        pooled = tuple(row[field] if field != "reference_type" else "all" for field in fields)
        groups.setdefault(pooled, []).append(row)
    output: list[dict[str, Any]] = []
    for key, members in sorted(groups.items(), key=lambda item: tuple(map(str, item[0]))):
        aggregate = {field: value for field, value in zip(fields, key)}
        aggregate["n_cases"] = len(members)
        for metric in METRICS:
            values = _finite(member.get(metric, np.nan) for member in members)
            aggregate[f"{metric}_mean"] = float(np.mean(values)) if values.size else float("nan")
            aggregate[f"{metric}_std"] = float(np.std(values, ddof=1)) if values.size > 1 else 0.0 if values.size else float("nan")
            aggregate[f"{metric}_median"] = float(np.median(values)) if values.size else float("nan")
            aggregate[f"{metric}_p95"] = float(np.percentile(values, 95)) if values.size else float("nan")
        output.append(aggregate)
    return output


def _paired_ci(deltas: np.ndarray, samples: int, rng: np.random.Generator) -> dict[str, Any]:
    values = deltas[np.isfinite(deltas)]
    if not values.size:
        return {"n": 0, "mean_delta": float("nan"), "median_delta": float("nan"), "ci95": [float("nan"), float("nan")]}
    draws = values[rng.integers(0, len(values), size=(samples, len(values)))].mean(axis=1)
    return {
        "n": int(len(values)),
        "mean_delta": float(np.mean(values)),
        "median_delta": float(np.median(values)),
        "ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
    }


def paired_report(rows: list[dict[str, Any]], samples: int, seed: int) -> dict[str, Any]:
    by_key = {
        (str(row["method"]), str(row["condition"]), str(row["case_id"])): row
        for row in rows
    }
    methods = sorted({str(row["method"]) for row in rows})
    conditions = sorted({str(row["condition"]) for row in rows})
    perturbed = [condition for condition in conditions if condition != "nominal"]
    case_ids = sorted({str(row["case_id"]) for row in rows})
    rng = np.random.default_rng(seed)
    degradation: dict[str, Any] = {}
    for method in methods:
        for condition in perturbed:
            metrics = {}
            for metric in PAIR_METRICS:
                delta = np.asarray(
                    [
                        float(by_key[(method, condition, case_id)].get(metric, np.nan))
                        - float(by_key[(method, "nominal", case_id)].get(metric, np.nan))
                        for case_id in case_ids
                    ]
                )
                metrics[metric] = _paired_ci(delta, samples, rng)
            degradation[f"{method}:{condition}_minus_nominal"] = metrics
    method_comparison: dict[str, Any] = {}
    comparisons = (
        ("NaiveDelayed", "IdealZeroDelay"),
        ("VirtualDelayAware", "IdealZeroDelay"),
        ("ThreadedAsync", "IdealZeroDelay"),
        ("VirtualDelayAware", "NaiveDelayed"),
        ("ThreadedAsync", "VirtualDelayAware"),
    )
    for right, left in comparisons:
        if right not in methods or left not in methods:
            continue
        for condition in conditions:
            metrics = {}
            for metric in PAIR_METRICS:
                delta = np.asarray(
                    [
                        float(by_key[(right, condition, case_id)].get(metric, np.nan))
                        - float(by_key[(left, condition, case_id)].get(metric, np.nan))
                        for case_id in case_ids
                    ]
                )
                metrics[metric] = _paired_ci(delta, samples, rng)
            method_comparison[f"{condition}:{right}_minus_{left}"] = metrics
    return {
        "bootstrap_samples": samples,
        "seed": seed,
        "paired_degradation": degradation,
        "paired_method_comparison": method_comparison,
    }


def _bootstrap(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float, float]:
    finite = values[np.isfinite(values)]
    if not finite.size:
        return float("nan"), float("nan"), float("nan")
    draws = finite[rng.integers(0, len(finite), size=(1000, len(finite)))].mean(axis=1)
    return float(np.mean(finite)), float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def _members(
    rows: list[dict[str, Any]],
    method: str,
    perturbation: str,
    level: int,
) -> list[dict[str, Any]]:
    condition = "nominal" if level == 0 else f"{perturbation}_l{level}"
    return [row for row in rows if row["method"] == method and row["condition"] == condition]


def _metric_panels(
    plt: Any,
    rows: list[dict[str, Any]],
    metric: str,
    ylabel: str,
    destination: Path,
    seed: int,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    rng = np.random.default_rng(seed)
    methods = [method.name for method in METHODS if any(row["method"] == method.name for row in rows)]
    perturbations = [
        value for value in direct.PERTURBATIONS if any(row["perturbation"] == value for row in rows)
    ]
    for axis in axes.ravel()[len(perturbations):]:
        axis.set_visible(False)
    for axis, perturbation in zip(axes.ravel(), perturbations):
        levels = [0, *sorted({int(row["level"]) for row in rows if row["perturbation"] == perturbation})]
        for method in methods:
            stats = [
                _bootstrap(np.asarray([float(row.get(metric, np.nan)) for row in _members(rows, method, perturbation, level)]), rng)
                for level in levels
            ]
            means = np.asarray([item[0] for item in stats])
            axis.errorbar(
                levels,
                means,
                yerr=np.vstack((means - np.asarray([item[1] for item in stats]), np.asarray([item[2] for item in stats]) - means)),
                marker="o",
                capsize=2,
                linewidth=1.5,
                label=method,
                color=COLORS[method],
            )
        axis.set_title(perturbation.replace("_", " "))
        axis.set_xlabel("robustness level")
        axis.set_ylabel(ylabel)
        axis.set_xticks(levels)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def _reliability_plot(plt: Any, rows: list[dict[str, Any]], destination: Path, seed: int) -> None:
    metrics = (
        ("failure_rate", "failed-case rate"),
        ("fallback_step_rate", "fallback step rate"),
        ("late_packet_rate", "late packet rate"),
        ("control_deadline_miss_count", "deadline misses / case"),
    )
    methods = [method.name for method in METHODS if any(row["method"] == method.name for row in rows)]
    conditions = ["nominal", *(f"{value}_l6" for value in direct.PERTURBATIONS)]
    available = {str(row["condition"]) for row in rows}
    conditions = [condition for condition in conditions if condition in available]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    rng = np.random.default_rng(seed)
    x = np.arange(len(conditions))
    width = 0.8 / len(methods)
    for axis, (metric, ylabel) in zip(axes.ravel(), metrics):
        for index, method in enumerate(methods):
            values, errors = [], []
            for condition in conditions:
                members = [row for row in rows if row["method"] == method and row["condition"] == condition]
                mean, low, high = _bootstrap(np.asarray([float(row.get(metric, np.nan)) for row in members]), rng)
                values.append(mean)
                errors.append((mean - low, high - mean))
            positions = x - 0.4 + width / 2 + index * width
            axis.bar(positions, values, width, label=method, color=COLORS[method])
            axis.errorbar(positions, values, yerr=np.asarray(errors).T, fmt="none", color="black", capsize=2, linewidth=0.7)
        axis.set_xticks(x, [condition.replace("_", "\n") for condition in conditions])
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.25)
        axis.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def _force_plot(plt: Any, rows: list[dict[str, Any]], destination: Path, seed: int) -> None:
    levels = sorted({int(row["level"]) for row in rows if row["perturbation"] == "force_pulse"})
    if not levels:
        return
    metrics = (
        ("force_peak_tracking_error", "peak TCP error [m]"),
        ("force_integrated_error_above_baseline", "integrated excess [m·s]"),
        ("force_recovery_time_s", "recovery time [s]"),
    )
    methods = [method.name for method in METHODS if any(row["method"] == method.name for row in rows)]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    rng = np.random.default_rng(seed)
    for axis, (metric, ylabel) in zip(axes, metrics):
        for method in methods:
            stats = [
                _bootstrap(np.asarray([float(row.get(metric, np.nan)) for row in _members(rows, method, "force_pulse", level)]), rng)
                for level in levels
            ]
            means = np.asarray([item[0] for item in stats])
            axis.errorbar(
                levels,
                means,
                yerr=np.vstack((means - np.asarray([item[1] for item in stats]), np.asarray([item[2] for item in stats]) - means)),
                marker="o",
                capsize=2,
                label=method,
                color=COLORS[method],
            )
        axis.set_xlabel("force-pulse level")
        axis.set_ylabel(ylabel)
        axis.set_xticks(levels)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def _paired_method_plot(plt: Any, rows: list[dict[str, Any]], destination: Path) -> None:
    by_key = {
        (str(row["method"]), str(row["condition"]), str(row["case_id"])): row
        for row in rows
    }
    case_ids = sorted({str(row["case_id"]) for row in rows})
    conditions = ["nominal", *(f"{value}_l6" for value in direct.PERTURBATIONS)]
    available = {str(row["condition"]) for row in rows}
    conditions = [condition for condition in conditions if condition in available]
    available_methods = {str(row["method"]) for row in rows}
    comparisons = [
        comparison
        for comparison in (
        ("NaiveDelayed", "IdealZeroDelay"),
        ("VirtualDelayAware", "IdealZeroDelay"),
        ("ThreadedAsync", "IdealZeroDelay"),
        )
        if set(comparison).issubset(available_methods)
    ]
    if not comparisons:
        return
    fig, axes = plt.subplots(1, len(comparisons), figsize=(6 * len(comparisons), 5), sharey=True)
    axes = np.atleast_1d(axes)
    for axis, (right, left) in zip(axes, comparisons):
        data = [
            [
                float(by_key[(right, condition, case_id)]["tcp_rmse_m"])
                - float(by_key[(left, condition, case_id)]["tcp_rmse_m"])
                for case_id in case_ids
            ]
            for condition in conditions
        ]
        axis.boxplot(data, tick_labels=[condition.replace("_", "\n") for condition in conditions], showmeans=True)
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.set_title(f"{right} − {left}")
        axis.set_ylabel("paired TCP RMSE delta [m]")
        axis.grid(axis="y", alpha=0.25)
        axis.tick_params(axis="x", labelrotation=25, labelsize=8)
    fig.tight_layout()
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def write_plots(root: Path, rows: list[dict[str, Any]], seed: int) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    destination = root / "plots"
    destination.mkdir(parents=True, exist_ok=True)
    _metric_panels(plt, rows, "tcp_rmse_m", "TCP RMSE [m]", destination / "tcp_rmse_vs_perturbation.png", seed)
    _metric_panels(
        plt,
        rows,
        "orientation_rmse_rad",
        "orientation RMSE [rad]",
        destination / "orientation_rmse_vs_perturbation.png",
        seed + 1,
    )
    _metric_panels(
        plt,
        rows,
        "command_acceleration_rms_rad_s2",
        "command acceleration RMS [rad/s²]",
        destination / "command_smoothness_vs_perturbation.png",
        seed + 2,
    )
    _reliability_plot(plt, rows, destination / "reliability_and_timing.png", seed + 3)
    _force_plot(plt, rows, destination / "force_response_summary.png", seed + 4)
    _paired_method_plot(plt, rows, destination / "paired_method_tcp_delta.png")


def _write_or_validate_manifest(
    path: Path,
    source_manifest: Path,
    delay_calibration: Path,
    specs: list[ExperimentSpec],
    conditions: list[direct.Condition],
    args: argparse.Namespace,
    delay_steps: int,
) -> None:
    payload = {
        "schema_version": 1,
        "kind": "delay_aware_mpc_robustness",
        "source_manifest": file_identity(source_manifest),
        "checkpoint": file_identity(RUNNER.resolve_runtime_path(args.checkpoint)),
        "normalizer": file_identity(RUNNER.resolve_runtime_path(args.normalizer)),
        "delay_calibration": file_identity(delay_calibration),
        "delay_steps": delay_steps,
        "planner_projection": args.planner_projection,
        "planner_projection_backend": args.planner_projection_backend,
        "planner_projection_strategy": args.planner_projection_strategy,
        "case_ids": sorted({spec.case_id for spec in specs}),
        "methods": [
            {
                "name": method.name,
                "multirate_mode": method.multirate_mode,
                "delay_protocol": method.delay_protocol,
                "delay_steps": delay_steps if method.uses_delay else 0,
            }
            for method in METHODS
            if any(spec.method.name == method.name for spec in specs)
        ],
        "conditions": [
            {
                "name": condition.name,
                "perturbation": condition.perturbation,
                "level": condition.level,
                "levels": condition.levels,
                "physical_values": direct._physical_values(condition),
            }
            for condition in conditions
        ],
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.bootstrap_seed,
        "run_count": len(specs),
    }
    wrapped = {"sha256": canonical_sha256(payload), "payload": payload}
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("sha256") != wrapped["sha256"]:
            raise RuntimeError(f"Experiment manifest mismatch at {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(wrapped, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_one(
    base: argparse.Namespace,
    spec: ExperimentSpec,
    save_root: Path,
    delay_steps: int,
    delay_calibration: Path,
) -> dict[str, Any]:
    run_dir = save_root / spec.method.name / spec.condition.name / spec.case_id
    args = _run_args(base, spec, run_dir, delay_steps)
    fingerprint = _fingerprint(spec, args, delay_calibration)
    arrays = load_completed_rollout(run_dir, fingerprint) if base.resume else None
    if arrays is None:
        result = RUNNER.run_closed_loop_mpc(args)
        arrays = result["arrays"]
        save_mpc_run(run_dir, arrays, result["rows"], result.get("planner_events"))
        (run_dir / "run_fingerprint.json").write_text(
            json.dumps(fingerprint, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
    else:
        print(f"Reused completed rollout: {run_dir}")
    return summarize_mpc(spec, arrays)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/nn_mpc_matplotlib_cache")
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    if args.bootstrap_samples <= 0:
        raise ValueError("--bootstrap_samples must be positive")
    for argument in direct.LEVEL_ARGUMENTS.values():
        if int(getattr(args, argument)) != 0:
            raise ValueError(f"Use --levels/--perturbations instead of --{argument}")
    checkpoint = RUNNER.resolve_runtime_path(args.checkpoint)
    normalizer = RUNNER.resolve_runtime_path(args.normalizer)
    for path in (checkpoint, normalizer):
        if not path.is_file():
            raise FileNotFoundError(path)
    delay_path = RUNNER.resolve_runtime_path(args.delay_calibration)
    delay_steps = load_delay(delay_path)
    levels = direct._levels(args.levels)
    perturbations = direct._csv_values(args.perturbations, name="--perturbations")
    method_names = direct._csv_values(args.methods, name="--methods")
    unknown_methods = set(method_names).difference(METHOD_BY_NAME)
    if unknown_methods:
        raise ValueError(f"Unknown methods: {sorted(unknown_methods)}")
    methods = [METHOD_BY_NAME[name] for name in method_names]
    case_ids = direct._csv_values(args.case_ids, name="--case_ids")
    manifest_path = RUNNER.resolve_runtime_path(args.manifest)
    _, cases = direct.load_task_cases(manifest_path, case_ids)
    seed_values = (
        [int(value) for value in direct._csv_values(args.seeds, name="--seeds")]
        if args.seeds.strip()
        else []
    )
    if len(seed_values) != len(set(seed_values)):
        raise ValueError("--seeds must contain unique values")
    if seed_values:
        repeated_cases: list[dict[str, Any]] = []
        for case in cases:
            for seed in seed_values:
                repeated = deepcopy(case)
                repeated["id"] = f"{case['id']}_seed_{seed}"
                values = repeated.setdefault("run_args", {})
                assert isinstance(values, dict)
                values["seed"] = seed
                repeated_cases.append(repeated)
        cases = repeated_cases
    conditions = direct.build_conditions(levels, perturbations)
    specs = build_specs(cases, conditions, methods)
    save_root = RUNNER.resolve_runtime_path(args.save_dir)
    _write_or_validate_manifest(
        save_root / "experiment_manifest.json",
        manifest_path,
        delay_path,
        specs,
        conditions,
        args,
        delay_steps,
    )
    rows: list[dict[str, Any]] = []
    summary_path = save_root / "delay_aware_mpc_robustness_summary.csv"
    for index, spec in enumerate(specs, start=1):
        print(f"[{index}/{len(specs)}] {spec.method.name}/{spec.condition.name}/{spec.case_id}")
        rows.append(run_one(args, spec, save_root, delay_steps, delay_path))
        _write_csv(summary_path, rows)
    aggregates = aggregate_rows(rows)
    _write_csv(save_root / "delay_aware_mpc_robustness_aggregate.csv", aggregates)
    paired = paired_report(rows, args.bootstrap_samples, args.bootstrap_seed)
    (save_root / "paired_bootstrap.json").write_text(
        json.dumps(paired, indent=2, sort_keys=True, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    write_plots(save_root, rows, args.bootstrap_seed)
    print(f"Saved {len(rows)} delay-aware MPC robustness rows to {summary_path}")


if __name__ == "__main__":
    main()
