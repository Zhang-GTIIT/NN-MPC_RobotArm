"""Paired planner-projection evaluation for virtual and threaded ASAP MPC."""
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

from scripts.experiment_utils.hashing import canonical_sha256, file_identity
from scripts.experiment_utils.resume import load_completed_rollout, run_fingerprint
from scripts.paper_experiments.evaluation import summarize_arrays
from scripts.robustness._evaluation import robustness_summary
from scripts.robustness._runtime import ROOT, ensure_import_paths, load_runner
from scripts.robustness.evaluate_direct_ik import load_task_cases

ensure_import_paths()

from mpc.logging import save_mpc_run


RUNNER = load_runner("planner_projection_v1_runner")
DEFAULT_MANIFEST = "outputs/robustness_d6/benchmark.json"
DEFAULT_DELAY = "outputs/robustness/timing/gru_20260717_182930_d6_override.json"
DEFAULT_CHECKPOINT = "dynamics_modeling/outputs/checkpoints/gru_20260717_182930/best_model.pt"
DEFAULT_NORMALIZER = "dynamics_modeling/outputs/checkpoints/gru_20260717_182930/normalizer.pt"
DEFAULT_CASES = "circle_00,figure8_00"
DEFAULT_SEEDS = "20260723,20260724,20260725"


@dataclass(frozen=True)
class Variant:
    mode: str
    multirate_mode: str
    planner_projection: str

    @property
    def label(self) -> str:
        return f"{self.mode}_{self.planner_projection}"


VARIANTS = (
    Variant("virtual_asap", "virtual_asap", "off"),
    Variant("virtual_asap", "virtual_asap", "on"),
    Variant("threaded_asap", "threaded_asap", "off"),
    Variant("threaded_asap", "threaded_asap", "on"),
)


def _csv_values(value: str) -> list[str]:
    values = [item.strip() for item in value.split(",") if item.strip()]
    if not values:
        raise ValueError("Expected at least one comma-separated value")
    return values


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = RUNNER.build_arg_parser()
    parser.description = "Compare planner projection on/off for virtual_asap and threaded_asap."
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--delay_calibration", default=DEFAULT_DELAY)
    parser.add_argument("--case_ids", default=DEFAULT_CASES)
    parser.add_argument("--seeds", default=DEFAULT_SEEDS)
    parser.add_argument("--resume", action="store_true")
    parser.set_defaults(
        checkpoint=DEFAULT_CHECKPOINT,
        normalizer=DEFAULT_NORMALIZER,
        model_type="gru",
        history_len=16,
        device="cuda",
        save_dir="outputs/planner_projection_v1_test",
    )
    return parser.parse_args(argv)


def _delay_steps(path: Path) -> int:
    delay = int(json.loads(path.read_text(encoding="utf-8"))["anticipation_delay_steps"])
    if delay <= 0:
        raise ValueError("anticipation_delay_steps must be positive")
    return delay


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _run_args(
    base: argparse.Namespace,
    case: dict[str, Any],
    variant: Variant,
    seed: int,
    delay: int,
    run_dir: Path,
) -> argparse.Namespace:
    args = deepcopy(base)
    values = case.get("run_args", case)
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
    args.reference_mode = "task"
    args.multirate_mode = variant.multirate_mode
    args.delay_protocol = "full"
    args.anticipation_delay_steps = delay
    args.planner_projection = variant.planner_projection
    args.seed = seed
    args.visualize = False
    args.save_dir = str(run_dir)
    return args


def _fingerprint(
    args: argparse.Namespace,
    case_id: str,
    variant: Variant,
    seed: int,
    delay_path: Path,
) -> dict[str, Any]:
    excluded = {"case_ids", "delay_calibration", "manifest", "resume", "save_dir", "seeds"}
    reference = RUNNER.resolve_runtime_path(args.reference_file)
    robustness = RUNNER._robustness_config(args)
    return run_fingerprint(
        {
            "kind": "planner_projection_v1",
            "case_id": case_id,
            "variant": variant.label,
            "seed": seed,
            "run_config": {
                key: value for key, value in vars(args).items() if key not in excluded
            },
            "reference": file_identity(reference),
            "checkpoint": file_identity(RUNNER.resolve_runtime_path(args.checkpoint)),
            "normalizer": file_identity(RUNNER.resolve_runtime_path(args.normalizer)),
            "delay_calibration": file_identity(delay_path),
            "nominal_model": file_identity(robustness.nominal_model_xml),
            "plant_model": file_identity(robustness.plant_model_xml),
        }
    )


def _summarize(
    arrays: dict[str, np.ndarray],
    case_id: str,
    reference_type: str,
    variant: Variant,
    seed: int,
) -> dict[str, Any]:
    row = summarize_arrays(variant.label, arrays)
    row.update(robustness_summary(arrays))
    fallback = np.asarray(arrays.get("fallback_active", np.empty(0)))
    packet_age = np.asarray(arrays.get("packet_age", np.empty(0)))
    row.update(
        {
            "case_id": case_id,
            "reference_type": reference_type,
            "seed": seed,
            "mode": variant.mode,
            "multirate_mode": variant.multirate_mode,
            "planner_projection": variant.planner_projection,
            "variant": variant.label,
            "fallback_step_rate": (
                float(np.mean(fallback != 0))
                if fallback.size
                else float(np.mean(packet_age < 0)) if packet_age.size else float("nan")
            ),
        }
    )
    return row


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics = (
        "tcp_rmse_m",
        "tcp_p95_m",
        "orientation_rmse_rad",
        "joint_rmse_rad",
        "projection_discrepancy_rms_rad",
        "planner_execution_qref_error_rms_rad",
        "projection_activation_rate",
        "planner_hz",
        "solve_p95_s",
        "e2e_p95_s",
        "late_packet_rate",
        "fallback_step_rate",
        "control_deadline_miss_count",
        "failure_rate",
    )
    output: list[dict[str, Any]] = []
    for mode in ("virtual_asap", "threaded_asap"):
        for projection in ("off", "on"):
            members = [
                row for row in rows
                if row["mode"] == mode and row["planner_projection"] == projection
            ]
            aggregate: dict[str, Any] = {
                "mode": mode,
                "planner_projection": projection,
                "n": len(members),
            }
            for metric in metrics:
                values = np.asarray([float(row.get(metric, np.nan)) for row in members])
                values = values[np.isfinite(values)]
                aggregate[f"{metric}_mean"] = float(np.mean(values)) if values.size else float("nan")
                aggregate[f"{metric}_std"] = (
                    float(np.std(values, ddof=1)) if values.size > 1 else 0.0
                    if values.size else float("nan")
                )
            output.append(aggregate)
    return output


def _paired(rows: list[dict[str, Any]], samples: int = 10000) -> dict[str, Any]:
    by_key = {
        (str(row["mode"]), str(row["planner_projection"]), str(row["case_id"]), int(row["seed"])): row
        for row in rows
    }
    rng = np.random.default_rng(20260723)
    report: dict[str, Any] = {}
    metrics = (
        "tcp_rmse_m",
        "joint_rmse_rad",
        "orientation_rmse_rad",
        "projection_discrepancy_rms_rad",
        "planner_execution_qref_error_rms_rad",
        "planner_hz",
        "solve_p95_s",
        "e2e_p95_s",
        "late_packet_rate",
        "fallback_step_rate",
    )
    pairs = sorted({(str(row["case_id"]), int(row["seed"])) for row in rows})
    for mode in ("virtual_asap", "threaded_asap"):
        mode_report: dict[str, Any] = {}
        for metric in metrics:
            delta = np.asarray(
                [
                    float(by_key[(mode, "on", case_id, seed)].get(metric, np.nan))
                    - float(by_key[(mode, "off", case_id, seed)].get(metric, np.nan))
                    for case_id, seed in pairs
                ],
                dtype=np.float64,
            )
            delta = delta[np.isfinite(delta)]
            if delta.size:
                draws = delta[rng.integers(0, len(delta), size=(samples, len(delta)))].mean(axis=1)
                mode_report[metric] = {
                    "n": len(delta),
                    "mean_on_minus_off": float(np.mean(delta)),
                    "median_on_minus_off": float(np.median(delta)),
                    "ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
                    "on_better_count": int(np.sum(delta < 0)),
                }
            else:
                mode_report[metric] = {
                    "n": 0,
                    "mean_on_minus_off": float("nan"),
                    "median_on_minus_off": float("nan"),
                    "ci95": [float("nan"), float("nan")],
                    "on_better_count": 0,
                }
        report[mode] = mode_report
    return {"bootstrap_samples": samples, "modes": report}


def _plots(root: Path, rows: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    plot_dir = root / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    colors = {"off": "#4c78a8", "on": "#f58518"}
    metrics = (
        ("tcp_rmse_m", 1000.0, "TCP RMSE [mm]"),
        ("joint_rmse_rad", 1.0, "Joint RMSE [rad]"),
        ("planner_execution_qref_error_rms_rad", 1000.0, "Planner–execution mismatch [mrad]"),
        ("solve_p95_s", 1000.0, "Solve p95 [ms]"),
        ("e2e_p95_s", 1000.0, "E2E p95 [ms]"),
        ("late_packet_rate", 100.0, "Late packet rate [%]"),
    )
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    labels = ["virtual off", "virtual on", "threaded off", "threaded on"]
    selections = [
        ("virtual_asap", "off"), ("virtual_asap", "on"),
        ("threaded_asap", "off"), ("threaded_asap", "on"),
    ]
    for axis, (metric, scale, ylabel) in zip(axes.ravel(), metrics):
        data = [
            [float(row.get(metric, np.nan)) * scale for row in rows
             if row["mode"] == mode and row["planner_projection"] == projection]
            for mode, projection in selections
        ]
        axis.boxplot(data, tick_labels=labels, showmeans=True)
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.25)
        axis.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(plot_dir / "projection_performance_summary.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    pairs = sorted({(str(row["case_id"]), int(row["seed"])) for row in rows})
    lookup = {
        (str(row["mode"]), str(row["planner_projection"]), str(row["case_id"]), int(row["seed"])): row
        for row in rows
    }
    for axis, mode in zip(axes, ("virtual_asap", "threaded_asap")):
        off = np.asarray([lookup[(mode, "off", case_id, seed)]["tcp_rmse_m"] for case_id, seed in pairs]) * 1000
        on = np.asarray([lookup[(mode, "on", case_id, seed)]["tcp_rmse_m"] for case_id, seed in pairs]) * 1000
        for index in range(len(pairs)):
            axis.plot((0, 1), (off[index], on[index]), color="#777777", alpha=0.65)
        axis.scatter(np.zeros(len(off)), off, color=colors["off"], label="off")
        axis.scatter(np.ones(len(on)), on, color=colors["on"], label="on")
        axis.set_xticks((0, 1), ("off", "on"))
        axis.set_title(mode)
        axis.set_ylabel("TCP RMSE [mm]")
        axis.grid(alpha=0.25)
        axis.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / "paired_tcp_rmse_on_off.png", dpi=180)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/nn_mpc_matplotlib_cache")
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    manifest_path = RUNNER.resolve_runtime_path(args.manifest)
    delay_path = RUNNER.resolve_runtime_path(args.delay_calibration)
    delay = _delay_steps(delay_path)
    case_ids = _csv_values(args.case_ids)
    seeds = [int(value) for value in _csv_values(args.seeds)]
    if len(set(seeds)) != len(seeds):
        raise ValueError("--seeds must be unique")
    _, cases = load_task_cases(manifest_path, case_ids)
    root = RUNNER.resolve_runtime_path(args.save_dir)
    specs = [
        (case, variant, seed)
        for variant in VARIANTS
        for case in cases
        for seed in seeds
    ]
    payload = {
        "schema_version": 1,
        "kind": "planner_projection_v1",
        "source_manifest": file_identity(manifest_path),
        "delay_calibration": file_identity(delay_path),
        "delay_steps": delay,
        "checkpoint": file_identity(RUNNER.resolve_runtime_path(args.checkpoint)),
        "normalizer": file_identity(RUNNER.resolve_runtime_path(args.normalizer)),
        "case_ids": case_ids,
        "seeds": seeds,
        "variants": [variant.__dict__ | {"label": variant.label} for variant in VARIANTS],
        "run_count": len(specs),
    }
    manifest = {"sha256": canonical_sha256(payload), "payload": payload}
    manifest_file = root / "experiment_manifest.json"
    if manifest_file.exists():
        if json.loads(manifest_file.read_text(encoding="utf-8")).get("sha256") != manifest["sha256"]:
            raise RuntimeError(f"Experiment manifest mismatch at {manifest_file}")
    else:
        root.mkdir(parents=True, exist_ok=True)
        manifest_file.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    rows: list[dict[str, Any]] = []
    summary = root / "planner_projection_summary.csv"
    for index, (case, variant, seed) in enumerate(specs, start=1):
        case_id = str(case["id"])
        reference_type = str(case["reference_type"])
        run_dir = root / variant.mode / f"projection_{variant.planner_projection}" / case_id / f"seed_{seed}"
        run_args = _run_args(args, case, variant, seed, delay, run_dir)
        fingerprint = _fingerprint(run_args, case_id, variant, seed, delay_path)
        arrays = load_completed_rollout(run_dir, fingerprint) if args.resume else None
        print(f"[{index}/{len(specs)}] {variant.label}/{case_id}/seed={seed}")
        if arrays is None:
            result = RUNNER.run_closed_loop_mpc(run_args)
            arrays = result["arrays"]
            save_mpc_run(run_dir, arrays, result["rows"], result.get("planner_events"))
            (run_dir / "run_fingerprint.json").write_text(
                json.dumps(fingerprint, indent=2, sort_keys=True, default=str) + "\n",
                encoding="utf-8",
            )
        rows.append(_summarize(arrays, case_id, reference_type, variant, seed))
        _write_csv(summary, rows)

    _write_csv(root / "planner_projection_aggregate.csv", _aggregate(rows))
    paired = _paired(rows)
    (root / "paired_on_minus_off.json").write_text(
        json.dumps(paired, indent=2, sort_keys=True, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    _plots(root, rows)
    print(f"Saved {len(rows)} planner-projection runs to {root}")


if __name__ == "__main__":
    main()
