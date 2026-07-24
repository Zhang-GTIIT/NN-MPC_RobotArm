"""Evaluate H20 projection-off, exact eager/compiled, and two-stage MPC."""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np

from scripts.experiment_utils.hashing import file_identity
from scripts.experiment_utils.resume import load_completed_rollout, run_fingerprint
from scripts.paper_experiments.evaluation import summarize_arrays
from scripts.robustness._evaluation import robustness_summary
from scripts.robustness._runtime import ensure_import_paths, load_runner
from scripts.robustness.evaluate_direct_ik import load_task_cases

ensure_import_paths()

from mpc.logging import save_mpc_run


RUNNER = load_runner("planner_projection_h20_evaluation_runner")


@dataclass(frozen=True)
class Variant:
    name: str
    projection: str
    backend: str
    strategy: str


VARIANTS = (
    Variant("off", "off", "compiled", "full"),
    Variant("full_eager", "on", "eager", "full"),
    Variant("full_compiled", "on", "compiled", "full"),
    Variant("two_stage_compiled", "on", "compiled", "two_stage"),
)


def parse_args() -> argparse.Namespace:
    parser = RUNNER.build_arg_parser()
    parser.add_argument("--manifest", default="outputs/robustness_h20_d10/benchmark.json")
    parser.add_argument("--calibration_dir", default="outputs/planner_projection_h20_optimization/calibration")
    parser.add_argument("--case_ids", default="circle_00,figure8_00")
    parser.add_argument("--seeds", default="20260723,20260724,20260725")
    parser.add_argument("--resume", action="store_true")
    parser.set_defaults(
        checkpoint="dynamics_modeling/outputs/checkpoints/gru_20260717_182930/best_model.pt",
        normalizer="dynamics_modeling/outputs/checkpoints/gru_20260717_182930/normalizer.pt",
        model_type="gru",
        history_len=16,
        horizon=20,
        num_samples=128,
        cem_iters=2,
        rollout_batch_size=128,
        device="cuda",
        save_dir="outputs/planner_projection_h20_optimization/evaluation",
    )
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_args(
    base: argparse.Namespace,
    case: dict[str, Any],
    variant: Variant,
    mode: str,
    seed: int,
    delay: int,
    run_dir: Path,
) -> argparse.Namespace:
    args = deepcopy(base)
    for key, value in case["run_args"].items():
        if hasattr(args, key):
            setattr(args, key, value)
    args.checkpoint = base.checkpoint
    args.normalizer = base.normalizer
    args.model_type = "gru"
    args.history_len = 16
    args.device = base.device
    args.horizon = 20
    args.num_samples = 128
    args.cem_iters = 2
    args.rollout_batch_size = 128
    args.controller_mode = "mpc"
    args.mpc_policy = "residual"
    args.reference_mode = "task"
    args.multirate_mode = mode
    args.delay_protocol = "full"
    args.anticipation_delay_steps = delay
    args.planner_projection = variant.projection
    args.planner_projection_backend = variant.backend
    args.planner_projection_strategy = variant.strategy
    args.seed = seed
    args.max_execution_steps = 500
    args.visualize = False
    args.save_dir = str(run_dir)
    return args


def fingerprint(args: argparse.Namespace, case_id: str, variant: Variant, mode: str, seed: int, delay: int):
    reference = RUNNER.resolve_runtime_path(args.reference_file)
    config = dict(vars(args))
    config["resume"] = False
    return run_fingerprint(
        {
            "kind": "planner_projection_h20",
            "case_id": case_id,
            "variant": variant.__dict__,
            "mode": mode,
            "seed": seed,
            "delay": delay,
            "run_config": config,
            "reference": file_identity(reference),
            "checkpoint": file_identity(RUNNER.resolve_runtime_path(args.checkpoint)),
            "normalizer": file_identity(RUNNER.resolve_runtime_path(args.normalizer)),
        }
    )


def summary(arrays: dict[str, np.ndarray], case: dict[str, Any], variant: Variant, mode: str, seed: int, delay: int, evaluation_set: str):
    row = summarize_arrays(f"{mode}_{variant.name}", arrays)
    row.update(robustness_summary(arrays))
    fallback = np.asarray(arrays.get("fallback_active", np.empty(0)))
    packet_age = np.asarray(arrays.get("packet_age", np.empty(0)))
    row.update(
        {
            "case_id": str(case["id"]),
            "reference_type": str(case["reference_type"]),
            "variant": variant.name,
            "planner_projection": variant.projection,
            "planner_projection_backend": variant.backend,
            "planner_projection_strategy": variant.strategy,
            "mode": mode,
            "seed": seed,
            "delay_steps": delay,
            "evaluation_set": evaluation_set,
            "fallback_step_rate": float(np.mean(fallback != 0)) if fallback.size else float(np.mean(packet_age < 0)),
        }
    )
    return row


def plots(root: Path, rows: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    destination = root / "plots"
    destination.mkdir(parents=True, exist_ok=True)
    metrics = (
        ("tcp_rmse_m", 1000, "TCP RMSE [mm]"),
        ("joint_rmse_rad", 1, "Joint RMSE [rad]"),
        ("solve_p95_s", 1000, "Solve p95 [ms]"),
        ("e2e_p95_s", 1000, "E2E p95 [ms]"),
        ("late_packet_rate", 100, "Late packets [%]"),
        ("fallback_step_rate", 100, "Fallback steps [%]"),
    )
    variants = [variant.name for variant in VARIANTS]
    for evaluation_set in ("calibrated", "common_d"):
        members = [row for row in rows if row["evaluation_set"] == evaluation_set]
        if not members:
            continue
        fig, axes = plt.subplots(2, 3, figsize=(15, 9))
        for axis, (metric, scale, ylabel) in zip(axes.ravel(), metrics):
            data = [
                [float(row.get(metric, np.nan)) * scale for row in members if row["variant"] == variant]
                for variant in variants
            ]
            axis.boxplot(data, tick_labels=variants, showmeans=True)
            axis.set_ylabel(ylabel)
            axis.tick_params(axis="x", rotation=20)
            axis.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(destination / f"{evaluation_set}_summary.png", dpi=180)
        plt.close(fig)


def paired_report(rows: list[dict[str, Any]], samples: int = 10000) -> dict[str, Any]:
    rng = np.random.default_rng(20260723)
    metrics = (
        "tcp_rmse_m",
        "joint_rmse_rad",
        "orientation_rmse_rad",
        "solve_p95_s",
        "e2e_p95_s",
        "late_packet_rate",
        "fallback_step_rate",
        "planner_execution_qref_error_rms_rad",
    )
    report: dict[str, Any] = {"bootstrap_samples": samples, "comparisons": {}}
    for evaluation_set in ("calibrated", "common_d"):
        for mode in ("virtual_asap", "threaded_asap"):
            members = [
                row for row in rows
                if row["evaluation_set"] == evaluation_set and row["mode"] == mode
            ]
            baseline = {
                (row["case_id"], int(row["seed"])): row
                for row in members if row["variant"] == "off"
            }
            if not baseline:
                continue
            for variant in ("full_eager", "full_compiled", "two_stage_compiled"):
                comparison: dict[str, Any] = {}
                for metric in metrics:
                    deltas = np.asarray(
                        [
                            float(row.get(metric, np.nan))
                            - float(baseline[(row["case_id"], int(row["seed"]))].get(metric, np.nan))
                            for row in members
                            if row["variant"] == variant
                            and (row["case_id"], int(row["seed"])) in baseline
                        ]
                    )
                    deltas = deltas[np.isfinite(deltas)]
                    if deltas.size:
                        draws = deltas[
                            rng.integers(0, len(deltas), size=(samples, len(deltas)))
                        ].mean(axis=1)
                        comparison[metric] = {
                            "n": int(len(deltas)),
                            "mean_delta": float(np.mean(deltas)),
                            "ci95": [
                                float(np.quantile(draws, 0.025)),
                                float(np.quantile(draws, 0.975)),
                            ],
                        }
                    else:
                        comparison[metric] = {
                            "n": 0,
                            "mean_delta": float("nan"),
                            "ci95": [float("nan"), float("nan")],
                        }
                report["comparisons"][
                    f"{evaluation_set}:{mode}:{variant}_minus_off"
                ] = comparison
    return report


def main() -> None:
    args = parse_args()
    if args.horizon != 20:
        raise ValueError("H20 projection evaluation requires horizon=20")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/nn_mpc_matplotlib_cache")
    case_ids = [value.strip() for value in args.case_ids.split(",") if value.strip()]
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    manifest_path = RUNNER.resolve_runtime_path(args.manifest)
    _, cases = load_task_cases(manifest_path, case_ids)
    calibration_dir = RUNNER.resolve_runtime_path(args.calibration_dir)
    delays = {
        variant.name: int(json.loads((calibration_dir / f"{variant.name}.json").read_text())["anticipation_delay_steps"])
        for variant in VARIANTS
    }
    common_delay = max(delays.values())
    root = RUNNER.resolve_runtime_path(args.save_dir)
    root.mkdir(parents=True, exist_ok=True)
    experiment = {
        "schema_version": 1,
        "kind": "planner_projection_h20",
        "horizon": 20,
        "case_ids": case_ids,
        "seeds": seeds,
        "calibrated_delays": delays,
        "common_delay": common_delay,
        "variants": [variant.__dict__ for variant in VARIANTS],
        "manifest": file_identity(manifest_path),
    }
    (root / "experiment_manifest.json").write_text(json.dumps(experiment, indent=2, sort_keys=True) + "\n")
    rows: list[dict[str, Any]] = []
    jobs: list[tuple[str, str, Variant, int]] = []
    for variant in VARIANTS:
        jobs.append(("calibrated", "virtual_asap", variant, delays[variant.name]))
        jobs.append(("calibrated", "threaded_asap", variant, delays[variant.name]))
        jobs.append(("common_d", "threaded_asap", variant, common_delay))
    total = len(jobs) * len(cases) * len(seeds)
    index = 0
    for evaluation_set, mode, variant, delay in jobs:
        for case in cases:
            for seed in seeds:
                index += 1
                run_dir = root / "runs" / mode / variant.name / f"D{delay}" / str(case["id"]) / f"seed_{seed}"
                run_config = run_args(args, case, variant, mode, seed, delay, run_dir)
                fp = fingerprint(run_config, str(case["id"]), variant, mode, seed, delay)
                arrays = load_completed_rollout(run_dir, fp) if args.resume or run_dir.exists() else None
                print(f"[{index}/{total}] {evaluation_set}/{mode}/{variant.name}/D{delay}/{case['id']}/{seed}")
                if arrays is None:
                    result = RUNNER.run_closed_loop_mpc(run_config)
                    arrays = result["arrays"]
                    save_mpc_run(run_dir, arrays, result["rows"], result.get("planner_events"))
                    (run_dir / "run_fingerprint.json").write_text(json.dumps(fp, indent=2, sort_keys=True, default=str) + "\n")
                rows.append(summary(arrays, case, variant, mode, seed, delay, evaluation_set))
                write_csv(root / "planner_projection_h20_summary.csv", rows)
    grouped: list[dict[str, Any]] = []
    metrics = ("tcp_rmse_m", "joint_rmse_rad", "orientation_rmse_rad", "solve_p95_s", "e2e_p95_s", "planner_hz", "late_packet_rate", "fallback_step_rate", "planner_execution_qref_error_rms_rad", "projection_discrepancy_rms_rad", "failure_rate")
    for key in sorted({(row["evaluation_set"], row["mode"], row["variant"], row["delay_steps"]) for row in rows}):
        members = [row for row in rows if (row["evaluation_set"], row["mode"], row["variant"], row["delay_steps"]) == key]
        item: dict[str, Any] = dict(zip(("evaluation_set", "mode", "variant", "delay_steps"), key))
        item["n"] = len(members)
        for metric in metrics:
            values = np.asarray([float(row.get(metric, np.nan)) for row in members])
            values = values[np.isfinite(values)]
            item[f"{metric}_mean"] = float(np.mean(values)) if values.size else float("nan")
            item[f"{metric}_std"] = float(np.std(values, ddof=1)) if values.size > 1 else float("nan")
        grouped.append(item)
    write_csv(root / "planner_projection_h20_aggregate.csv", grouped)
    (root / "paired_bootstrap.json").write_text(
        json.dumps(paired_report(rows), indent=2, sort_keys=True, allow_nan=True) + "\n"
    )
    plots(root, rows)
    print(f"Saved {len(rows)} rows; common D={common_delay}; root={root}")


if __name__ == "__main__":
    main()
