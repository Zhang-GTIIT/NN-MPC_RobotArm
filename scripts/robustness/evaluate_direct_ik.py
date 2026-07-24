"""Evaluate raw and physically projected DirectIK under robustness perturbations."""
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
from scripts.robustness._evaluation import robustness_summary
from scripts.robustness._runtime import ROOT, ensure_import_paths, load_runner

ensure_import_paths()

from mpc.logging import save_mpc_run
from mpc.robustness import (
    ACTUATOR_KP_SCALE,
    FORCE_PULSE_N,
    OBSERVATION_DQ_STD_RAD_S,
    OBSERVATION_Q_STD_RAD,
    PAYLOAD_MASS_KG,
)


RUNNER = load_runner("direct_ik_robustness_runner")
PERTURBATIONS = ("payload", "actuator_gain", "force_pulse", "observation_noise")
LEVEL_ARGUMENTS = {
    "payload": "payload_level",
    "actuator_gain": "actuator_gain_level",
    "force_pulse": "force_pulse_level",
    "observation_noise": "observation_noise_level",
}
DEFAULT_CASE_IDS = ",".join(
    f"{shape}_{index:02d}"
    for shape in ("circle", "figure8", "ellipse", "square")
    for index in range(3)
)
SUMMARY_METRICS = (
    "tcp_rmse_m",
    "tcp_p95_m",
    "orientation_rmse_rad",
    "joint_rmse_rad",
    "command_acceleration_rms_rad_s2",
    "command_acceleration_max_abs_rad_s2",
    "command_total_variation_rad",
    "actuator_torque_rms_nm",
    "joint_violation_rate",
    "velocity_violation_rate",
    "acceleration_violation_rate",
    "force_peak_tracking_error",
    "force_integrated_error_above_baseline",
    "force_recovery_time_s",
)
PAIR_METRICS = (
    "tcp_rmse_m",
    "tcp_p95_m",
    "orientation_rmse_rad",
    "joint_rmse_rad",
    "command_acceleration_rms_rad_s2",
    "command_total_variation_rad",
    "actuator_torque_rms_nm",
    "velocity_violation_rate",
    "acceleration_violation_rate",
    "force_peak_tracking_error",
    "force_integrated_error_above_baseline",
    "force_recovery_time_s",
)


@dataclass(frozen=True)
class Condition:
    name: str
    perturbation: str
    level: int
    levels: dict[str, int]


@dataclass(frozen=True)
class ExperimentSpec:
    projection: str
    condition: Condition
    case_id: str
    reference_type: str
    case: dict[str, Any]


def _csv_values(value: str, *, name: str) -> list[str]:
    values = [item.strip() for item in value.split(",") if item.strip()]
    if not values:
        raise ValueError(f"{name} must contain at least one value")
    if len(values) != len(set(values)):
        raise ValueError(f"{name} contains duplicate values")
    return values


def _levels(value: str) -> list[int]:
    try:
        levels = [int(item) for item in _csv_values(value, name="--levels")]
    except ValueError as exc:
        raise ValueError("--levels must be comma-separated integers in [0, 6]") from exc
    if any(level < 0 or level > 6 for level in levels):
        raise ValueError("--levels must be in [0, 6]")
    if 0 not in levels:
        raise ValueError("--levels must include 0 for the shared nominal baseline")
    return sorted(levels)


def build_conditions(levels: Iterable[int], perturbations: Iterable[str]) -> list[Condition]:
    level_values = sorted(set(int(level) for level in levels))
    perturbation_values = list(perturbations)
    unknown = set(perturbation_values).difference(PERTURBATIONS)
    if unknown:
        raise ValueError(f"Unknown perturbations: {sorted(unknown)}")
    zero = {argument: 0 for argument in LEVEL_ARGUMENTS.values()}
    output = [Condition("nominal", "nominal", 0, zero)]
    for perturbation in perturbation_values:
        argument = LEVEL_ARGUMENTS[perturbation]
        for level in level_values:
            if level == 0:
                continue
            values = dict(zero)
            values[argument] = level
            output.append(Condition(f"{perturbation}_l{level}", perturbation, level, values))
    return output


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_task_cases(manifest_path: Path, case_ids: Iterable[str]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("kind") != "model_a_robustness":
        raise ValueError("--manifest must be produced by build_benchmark_manifest.py")
    cases = manifest.get("cases")
    if not isinstance(cases, list):
        raise ValueError("Benchmark manifest has no cases list")
    by_id = {str(case.get("id")): case for case in cases if isinstance(case, dict)}
    requested = list(case_ids)
    missing = set(requested).difference(by_id)
    if missing:
        raise ValueError(f"Unknown --case_ids: {sorted(missing)}")
    selected: list[dict[str, Any]] = []
    for case_id in requested:
        case = by_id[case_id]
        values = case.get("run_args", case)
        if not isinstance(values, dict) or values.get("reference_mode") != "task":
            raise ValueError(f"DirectIK case must use a task-space reference: {case_id}")
        reference = Path(str(values.get("reference_file", "")))
        if not reference.is_absolute():
            reference = ROOT / reference
        if not reference.is_file():
            raise FileNotFoundError(f"Missing reference for {case_id}: {reference}")
        expected = case.get("reference_sha256")
        if expected is not None and _sha256(reference) != expected:
            raise ValueError(f"Reference hash mismatch for {case_id}")
        selected.append(case)
    return manifest, selected


def build_specs(
    cases: Iterable[dict[str, Any]],
    conditions: Iterable[Condition],
    projections: Iterable[str],
) -> list[ExperimentSpec]:
    output: list[ExperimentSpec] = []
    for projection in projections:
        if projection not in {"raw", "physical"}:
            raise ValueError(f"Unknown DirectIK projection: {projection}")
        for condition in conditions:
            for case in cases:
                output.append(
                    ExperimentSpec(
                        projection=projection,
                        condition=condition,
                        case_id=str(case["id"]),
                        reference_type=str(case.get("reference_type", "unknown")),
                        case=case,
                    )
                )
    return output


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = RUNNER.build_arg_parser()
    parser.description = "Evaluate DirectIK-only robustness with paired raw/physical command semantics."
    parser.add_argument("--manifest", default="outputs/robustness/benchmark.json")
    parser.add_argument("--case_ids", default=DEFAULT_CASE_IDS)
    parser.add_argument("--levels", default="0,3,6")
    parser.add_argument("--perturbations", default=",".join(PERTURBATIONS))
    parser.add_argument("--ik_command_projections", default="raw,physical")
    parser.add_argument(
        "--seeds",
        default="",
        help="Optional comma-separated seeds. When provided, repeat every selected case for each seed.",
    )
    parser.add_argument("--bootstrap_samples", type=int, default=2000)
    parser.add_argument("--bootstrap_seed", type=int, default=20260722)
    parser.add_argument("--resume", action="store_true")
    parser.set_defaults(save_dir="outputs/robustness/direct_ik_medium", device="cpu")
    return parser.parse_args(argv)


def _run_args(base: argparse.Namespace, spec: ExperimentSpec, run_dir: Path) -> argparse.Namespace:
    args = deepcopy(base)
    values = spec.case.get("run_args", spec.case)
    assert isinstance(values, dict)
    for key, value in values.items():
        if key != "id" and hasattr(args, key):
            setattr(args, key, value)
    args.controller_mode = "ik_direct"
    args.reference_mode = "task"
    args.multirate_mode = "synchronous"
    args.checkpoint = None
    args.normalizer = None
    args.device = "cpu"
    # Keep the explicit execution cap available for smoke tests. The frozen
    # benchmark normally stores None and must not erase this CLI override.
    args.max_execution_steps = base.max_execution_steps
    args.ik_preview_steps = 0
    args.ik_command_projection = spec.projection
    args.save_dir = str(run_dir)
    for argument, level in spec.condition.levels.items():
        setattr(args, argument, level)
    return args


def _fingerprint(spec: ExperimentSpec, args: argparse.Namespace) -> dict[str, Any]:
    excluded = {
        "bootstrap_samples",
        "bootstrap_seed",
        "case_ids",
        "ik_command_projections",
        "levels",
        "manifest",
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
            "kind": "direct_ik_robustness",
            "case_id": spec.case_id,
            "reference_type": spec.reference_type,
            "projection": spec.projection,
            "condition": {
                "name": spec.condition.name,
                "perturbation": spec.condition.perturbation,
                "level": spec.condition.level,
                "levels": spec.condition.levels,
            },
            "run_config": config,
            "reference": file_identity(reference),
            "nominal_model": file_identity(robustness.nominal_model_xml),
            "plant_model": file_identity(robustness.plant_model_xml),
        }
    )


def summarize_direct_ik(spec: ExperimentSpec, arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    row = summarize_arrays(f"DirectIK_{spec.projection}", arrays)
    row.update(robustness_summary(arrays))
    steps = max(int(row["steps"]), 1)
    command = np.asarray(arrays.get("actuator_q_ref", np.empty((0, 0))), dtype=np.float64)
    row.update(
        {
            "case_id": spec.case_id,
            "reference_type": spec.reference_type,
            "projection": spec.projection,
            "condition": spec.condition.name,
            "perturbation": spec.condition.perturbation,
            "level": spec.condition.level,
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
    array = np.asarray([float(value) for value in values], dtype=np.float64)
    return array[np.isfinite(array)]


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    group_fields = ("projection", "condition", "perturbation", "level", "reference_type")
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = tuple(row[field] for field in group_fields)
        groups.setdefault(key, []).append(row)
        pooled = tuple(row[field] if field != "reference_type" else "all" for field in group_fields)
        groups.setdefault(pooled, []).append(row)
    output: list[dict[str, Any]] = []
    for key, members in sorted(groups.items(), key=lambda item: tuple(map(str, item[0]))):
        result = {field: value for field, value in zip(group_fields, key)}
        result["n_cases"] = len(members)
        for metric in SUMMARY_METRICS:
            values = _finite(member.get(metric, np.nan) for member in members)
            result[f"{metric}_mean"] = float(np.mean(values)) if values.size else float("nan")
            result[f"{metric}_std"] = float(np.std(values, ddof=1)) if values.size > 1 else 0.0 if values.size else float("nan")
            result[f"{metric}_median"] = float(np.median(values)) if values.size else float("nan")
            result[f"{metric}_p95"] = float(np.percentile(values, 95)) if values.size else float("nan")
        output.append(result)
    return output


def _paired_ci(
    deltas: np.ndarray,
    samples: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    finite = deltas[np.isfinite(deltas)]
    if not finite.size:
        return {"n": 0, "mean_delta": float("nan"), "median_delta": float("nan"), "ci95": [float("nan"), float("nan")]}
    draws = finite[rng.integers(0, len(finite), size=(samples, len(finite)))].mean(axis=1)
    return {
        "n": int(len(finite)),
        "mean_delta": float(np.mean(finite)),
        "median_delta": float(np.median(finite)),
        "ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
    }


def paired_report(rows: list[dict[str, Any]], samples: int, seed: int) -> dict[str, Any]:
    by_key = {
        (str(row["projection"]), str(row["condition"]), str(row["case_id"])): row
        for row in rows
    }
    projections = sorted({str(row["projection"]) for row in rows})
    conditions = sorted({str(row["condition"]) for row in rows if row["condition"] != "nominal"})
    case_ids = sorted({str(row["case_id"]) for row in rows})
    rng = np.random.default_rng(seed)
    degradation: dict[str, Any] = {}
    for projection in projections:
        for condition in conditions:
            metrics: dict[str, Any] = {}
            for metric in PAIR_METRICS:
                deltas = np.asarray(
                    [
                        float(by_key[(projection, condition, case_id)].get(metric, np.nan))
                        - float(by_key[(projection, "nominal", case_id)].get(metric, np.nan))
                        for case_id in case_ids
                    ],
                    dtype=np.float64,
                )
                metrics[metric] = _paired_ci(deltas, samples, rng)
            degradation[f"{projection}:{condition}_minus_nominal"] = metrics
    projection_delta: dict[str, Any] = {}
    if {"raw", "physical"}.issubset(projections):
        for condition in ["nominal", *conditions]:
            metrics = {}
            for metric in PAIR_METRICS:
                deltas = np.asarray(
                    [
                        float(by_key[("physical", condition, case_id)].get(metric, np.nan))
                        - float(by_key[("raw", condition, case_id)].get(metric, np.nan))
                        for case_id in case_ids
                    ],
                    dtype=np.float64,
                )
                metrics[metric] = _paired_ci(deltas, samples, rng)
            projection_delta[f"{condition}:physical_minus_raw"] = metrics
    return {
        "bootstrap_samples": samples,
        "seed": seed,
        "paired_degradation": degradation,
        "paired_projection_delta": projection_delta,
    }


def _bootstrap_mean_ci(values: np.ndarray, rng: np.random.Generator, samples: int = 1000) -> tuple[float, float, float]:
    finite = values[np.isfinite(values)]
    if not finite.size:
        return float("nan"), float("nan"), float("nan")
    draws = finite[rng.integers(0, len(finite), size=(samples, len(finite)))].mean(axis=1)
    return float(np.mean(finite)), float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def _axis_level_rows(
    rows: list[dict[str, Any]],
    projection: str,
    perturbation: str,
    level: int,
) -> list[dict[str, Any]]:
    condition = "nominal" if level == 0 else f"{perturbation}_l{level}"
    return [row for row in rows if row["projection"] == projection and row["condition"] == condition]


def _metric_panels(
    plt: Any,
    rows: list[dict[str, Any]],
    metric: str,
    ylabel: str,
    destination: Path,
    seed: int,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=False)
    rng = np.random.default_rng(seed)
    projections = ("raw", "physical")
    colors = {"raw": "#d95f02", "physical": "#1b9e77"}
    present = [
        perturbation
        for perturbation in PERTURBATIONS
        if any(row["perturbation"] == perturbation for row in rows)
    ]
    for axis in axes.ravel()[len(present):]:
        axis.set_visible(False)
    for axis, perturbation in zip(axes.ravel(), present):
        levels = [0, *sorted({int(row["level"]) for row in rows if row["perturbation"] == perturbation})]
        for projection in projections:
            stats = [
                _bootstrap_mean_ci(
                    np.asarray(
                        [float(row.get(metric, np.nan)) for row in _axis_level_rows(rows, projection, perturbation, level)]
                    ),
                    rng,
                )
                for level in levels
            ]
            means = np.asarray([item[0] for item in stats])
            lower = means - np.asarray([item[1] for item in stats])
            upper = np.asarray([item[2] for item in stats]) - means
            axis.errorbar(levels, means, yerr=np.vstack((lower, upper)), marker="o", capsize=3, label=projection, color=colors[projection])
        axis.set_title(perturbation.replace("_", " "))
        axis.set_xlabel("robustness level")
        axis.set_ylabel(ylabel)
        axis.set_xticks(levels)
        axis.grid(alpha=0.25)
        axis.legend()
    fig.tight_layout()
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def _safety_plot(plt: Any, rows: list[dict[str, Any]], destination: Path, seed: int) -> None:
    metrics = (
        ("velocity_violation_rate", "velocity violation rate"),
        ("acceleration_violation_rate", "acceleration violation rate"),
        ("command_acceleration_rms_rad_s2", "command accel. RMS [rad/s²]"),
        ("command_total_variation_rad", "command total variation [rad]"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    rng = np.random.default_rng(seed)
    present = [
        perturbation
        for perturbation in PERTURBATIONS
        if any(row["perturbation"] == perturbation for row in rows)
    ]
    x = np.arange(len(present))
    width = 0.36
    for axis, (metric, ylabel) in zip(axes.ravel(), metrics):
        for offset, projection in ((-width / 2, "raw"), (width / 2, "physical")):
            values, errors = [], []
            for perturbation in present:
                maximum_level = max(int(row["level"]) for row in rows if row["perturbation"] == perturbation)
                members = _axis_level_rows(rows, projection, perturbation, maximum_level)
                mean, low, high = _bootstrap_mean_ci(
                    np.asarray([float(row.get(metric, np.nan)) for row in members]), rng
                )
                values.append(mean)
                errors.append((mean - low, high - mean))
            axis.bar(x + offset, values, width, label=projection)
            axis.errorbar(
                x + offset,
                values,
                yerr=np.asarray(errors).T,
                fmt="none",
                color="black",
                capsize=2,
                linewidth=0.8,
            )
        axis.set_xticks(x, [item.replace("_", "\n") for item in present])
        axis.set_ylabel(ylabel)
        axis.set_title("level 6")
        axis.grid(axis="y", alpha=0.25)
        axis.legend()
    fig.tight_layout()
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def _force_plot(plt: Any, rows: list[dict[str, Any]], destination: Path, seed: int) -> None:
    force_levels = sorted({int(row["level"]) for row in rows if row["perturbation"] == "force_pulse"})
    if not force_levels:
        return
    metrics = (
        ("force_peak_tracking_error", "peak TCP error [m]"),
        ("force_integrated_error_above_baseline", "integrated excess error [m·s]"),
        ("force_recovery_time_s", "recovery time [s]"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    rng = np.random.default_rng(seed)
    for axis, (metric, ylabel) in zip(axes, metrics):
        for projection, color in (("raw", "#d95f02"), ("physical", "#1b9e77")):
            stats = []
            for level in force_levels:
                members = _axis_level_rows(rows, projection, "force_pulse", level)
                stats.append(_bootstrap_mean_ci(np.asarray([float(row.get(metric, np.nan)) for row in members]), rng))
            means = np.asarray([item[0] for item in stats])
            axis.errorbar(
                force_levels,
                means,
                yerr=np.vstack((means - np.asarray([item[1] for item in stats]), np.asarray([item[2] for item in stats]) - means)),
                marker="o",
                capsize=3,
                label=projection,
                color=color,
            )
        axis.set_xticks(force_levels)
        axis.set_xlabel("force-pulse level")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
        axis.legend()
    fig.tight_layout()
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def _projection_delta_plot(plt: Any, rows: list[dict[str, Any]], destination: Path) -> None:
    projections = {str(row["projection"]) for row in rows}
    if not {"raw", "physical"}.issubset(projections):
        return
    by_key = {
        (str(row["projection"]), str(row["condition"]), str(row["case_id"])): row
        for row in rows
    }
    available = {str(row["condition"]) for row in rows}
    conditions = ["nominal", *sorted(available.difference({"nominal"}))]
    case_ids = sorted({str(row["case_id"]) for row in rows})
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for axis, metric, ylabel in (
        (axes[0], "tcp_rmse_m", "physical − raw TCP RMSE [m]"),
        (axes[1], "command_acceleration_rms_rad_s2", "physical − raw command accel. RMS"),
    ):
        data = [
            [
                float(by_key[("physical", condition, case_id)].get(metric, np.nan))
                - float(by_key[("raw", condition, case_id)].get(metric, np.nan))
                for case_id in case_ids
            ]
            for condition in conditions
        ]
        axis.boxplot(data, tick_labels=[item.replace("_", "\n") for item in conditions], showmeans=True)
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.25)
        axis.tick_params(axis="x", labelrotation=28, labelsize=8)
        for label in axis.get_xticklabels():
            label.set_horizontalalignment("right")
    fig.tight_layout()
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def _trajectory_plot(plt: Any, rows: list[dict[str, Any]], destination: Path) -> None:
    shapes = ("circle", "figure8", "ellipse", "square")
    available = {str(row["condition"]) for row in rows}
    conditions = [f"{item}_l6" for item in PERTURBATIONS if f"{item}_l6" in available]
    if not conditions:
        conditions = sorted(available.difference({"nominal"}))
    if not conditions:
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for axis, projection in zip(axes, ("raw", "physical")):
        matrix = np.full((len(shapes), len(conditions)), np.nan)
        for shape_index, shape in enumerate(shapes):
            for condition_index, condition in enumerate(conditions):
                values = _finite(
                    row["tcp_rmse_m"]
                    for row in rows
                    if row["projection"] == projection
                    and row["condition"] == condition
                    and row["reference_type"] == shape
                )
                matrix[shape_index, condition_index] = np.mean(values) if values.size else np.nan
        image = axis.imshow(matrix * 1000.0, aspect="auto", cmap="viridis")
        axis.set_title(projection)
        axis.set_xticks(range(len(conditions)), [item.replace("_l6", "").replace("_", "\n") for item in conditions])
        axis.set_yticks(range(len(shapes)), shapes)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                axis.text(j, i, f"{matrix[i, j] * 1000.0:.1f}", ha="center", va="center", color="white")
        fig.colorbar(image, ax=axis, label="TCP RMSE [mm]")
    fig.tight_layout()
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def write_plots(save_root: Path, rows: list[dict[str, Any]], seed: int) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    plot_dir = save_root / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    _metric_panels(plt, rows, "tcp_rmse_m", "TCP RMSE [m]", plot_dir / "tcp_rmse_vs_perturbation.png", seed)
    _metric_panels(
        plt,
        rows,
        "orientation_rmse_rad",
        "orientation RMSE [rad]",
        plot_dir / "orientation_rmse_vs_perturbation.png",
        seed + 1,
    )
    _safety_plot(plt, rows, plot_dir / "safety_and_smoothness.png", seed + 2)
    _force_plot(plt, rows, plot_dir / "force_response_summary.png", seed + 3)
    _projection_delta_plot(plt, rows, plot_dir / "raw_vs_physical_paired_delta.png")
    _trajectory_plot(plt, rows, plot_dir / "trajectory_type_level6_tcp_rmse.png")


def _physical_values(condition: Condition) -> dict[str, float]:
    level = condition.level
    return {
        "payload_mass_kg": PAYLOAD_MASS_KG[condition.levels["payload_level"]],
        "actuator_kp_scale": ACTUATOR_KP_SCALE[condition.levels["actuator_gain_level"]],
        "actuator_kd_scale": float(np.sqrt(ACTUATOR_KP_SCALE[condition.levels["actuator_gain_level"]])),
        "force_pulse_n": FORCE_PULSE_N[condition.levels["force_pulse_level"]],
        "observation_q_std_rad": OBSERVATION_Q_STD_RAD[condition.levels["observation_noise_level"]],
        "observation_dq_std_rad_s": OBSERVATION_DQ_STD_RAD_S[condition.levels["observation_noise_level"]],
        "reported_level": level,
    }


def _write_or_validate_manifest(
    path: Path,
    source_manifest: Path,
    specs: list[ExperimentSpec],
    conditions: list[Condition],
    args: argparse.Namespace,
) -> None:
    payload = {
        "schema_version": 1,
        "kind": "direct_ik_robustness",
        "source_manifest": file_identity(source_manifest),
        "case_ids": sorted({spec.case_id for spec in specs}),
        "projections": sorted({spec.projection for spec in specs}),
        "ik_preview_steps": 0,
        "conditions": [
            {
                "name": condition.name,
                "perturbation": condition.perturbation,
                "level": condition.level,
                "levels": condition.levels,
                "physical_values": _physical_values(condition),
            }
            for condition in conditions
        ],
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.bootstrap_seed,
        "seeds": sorted(
            {
                int(spec.case.get("run_args", {}).get("seed", -1))
                for spec in specs
            }
        ),
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


def run_one(base: argparse.Namespace, spec: ExperimentSpec, save_root: Path) -> dict[str, Any]:
    run_dir = save_root / spec.projection / spec.condition.name / spec.case_id
    args = _run_args(base, spec, run_dir)
    fingerprint = _fingerprint(spec, args)
    arrays = load_completed_rollout(run_dir, fingerprint) if base.resume else None
    if arrays is None:
        result = RUNNER.run_closed_loop_mpc(args)
        arrays = result["arrays"]
        save_mpc_run(run_dir, arrays, result["rows"])
        (run_dir / "run_fingerprint.json").write_text(
            json.dumps(fingerprint, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
    else:
        print(f"Reused completed rollout: {run_dir}")
    return summarize_direct_ik(spec, arrays)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/nn_mpc_matplotlib_cache")
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    if args.bootstrap_samples <= 0:
        raise ValueError("--bootstrap_samples must be positive")
    # The singular inherited robustness flags are intentionally not alternate
    # experiment inputs: using them would silently create combined conditions.
    for argument in LEVEL_ARGUMENTS.values():
        if int(getattr(args, argument)) != 0:
            raise ValueError(f"Use --levels/--perturbations instead of --{argument}")
    levels = _levels(args.levels)
    perturbations = _csv_values(args.perturbations, name="--perturbations")
    projections = _csv_values(args.ik_command_projections, name="--ik_command_projections")
    case_ids = _csv_values(args.case_ids, name="--case_ids")
    manifest_path = RUNNER.resolve_runtime_path(args.manifest)
    _, cases = load_task_cases(manifest_path, case_ids)
    seed_values = (
        [int(value) for value in _csv_values(args.seeds, name="--seeds")]
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
    conditions = build_conditions(levels, perturbations)
    specs = build_specs(cases, conditions, projections)
    save_root = RUNNER.resolve_runtime_path(args.save_dir)
    _write_or_validate_manifest(
        save_root / "experiment_manifest.json",
        manifest_path,
        specs,
        conditions,
        args,
    )
    rows: list[dict[str, Any]] = []
    summary_path = save_root / "direct_ik_robustness_summary.csv"
    for index, spec in enumerate(specs, start=1):
        print(f"[{index}/{len(specs)}] {spec.projection}/{spec.condition.name}/{spec.case_id}")
        rows.append(run_one(args, spec, save_root))
        _write_csv(summary_path, rows)
    aggregates = aggregate_rows(rows)
    _write_csv(save_root / "direct_ik_robustness_aggregate.csv", aggregates)
    paired = paired_report(rows, args.bootstrap_samples, args.bootstrap_seed)
    (save_root / "paired_bootstrap.json").write_text(
        json.dumps(paired, indent=2, sort_keys=True, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    write_plots(save_root, rows, args.bootstrap_seed)
    print(f"Saved {len(rows)} DirectIK robustness rows to {summary_path}")


if __name__ == "__main__":
    main()
