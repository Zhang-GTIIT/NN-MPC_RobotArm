"""One reproducible entry point for the ROBIO delay-aware MPC experiments."""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import mujoco
import numpy as np
import torch

from mpc.logging import save_mpc_run
from mpc.reference import finite_difference_dq, generate_joint_reference
from mpc.reference_pipeline import ReferenceConfig, build_reference, save_reference_bundle
from scripts.experiment_utils import (
    environment_snapshot, file_identity, load_completed_rollout, load_json,
    paired_bootstrap_rows, run_fingerprint, write_immutable_json,
)
from scripts.paper_experiments.evaluation import aggregate_rows, latency_recovery, summarize_arrays, write_csv, write_json
from scripts.robustness._runtime import ROOT, load_runner


RUNNER = load_runner("paper_delay_aware_runner")
DEFAULT_ROOT = ROOT / "outputs" / "paper_delay_aware"
CHECKPOINT = ROOT / "outputs" / "checkpoints" / "gru_20260720_202923" / "best_model.pt"
NORMALIZER = ROOT / "outputs" / "checkpoints" / "gru_20260720_202923" / "normalizer.pt"
MODEL_XML = ROOT / "dynamics_modeling" / "ABB_IRB2400.xml"
TRAJECTORIES = ("circle", "figure8", "fast_ellipse", "rounded_square")


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def parser() -> argparse.ArgumentParser:
    top = argparse.ArgumentParser(description=__doc__)
    top.add_argument("--output-root", default=str(DEFAULT_ROOT))
    sub = top.add_subparsers(dest="command", required=True)

    calibration = sub.add_parser("generate-calibration-reference")
    calibration.add_argument("--steps", type=int, default=2200)
    calibration.add_argument("--overwrite", action="store_true")

    delay = sub.add_parser("calibrate-delay")
    delay.add_argument("--samples", type=int, default=500)
    delay.add_argument("--provisional-delay", type=int, default=10)
    delay.add_argument("--guard-ms", type=float, default=5.0)
    delay.add_argument("--smoke", action="store_true")

    references = sub.add_parser("generate-references")
    references.add_argument("--overwrite", action="store_true")

    preview = sub.add_parser("calibrate-preview")
    preview.add_argument("--preview-values", default="0,1,2,3,4")
    preview.add_argument("--smoke", action="store_true")

    validation = sub.add_parser("validate-model")
    validation.add_argument("--num-rollouts", type=int, default=20)
    validation.add_argument("--rollout-len", type=int, default=200)

    manifest = sub.add_parser("build-manifest")
    manifest.add_argument("--allow-dirty", action="store_true")
    manifest.add_argument("--profile", choices=["paper", "smoke"], default="paper")

    run = sub.add_parser("run")
    run.add_argument("--manifest", default=None)
    run.add_argument("--suite", choices=["main", "ablation", "delay_sweep", "preview", "oracle", "all"], default="main")
    run.add_argument("--resume", action="store_true")
    run.add_argument("--case-limit", type=int, default=None)

    summary = sub.add_parser("summarize")
    summary.add_argument("--suite", choices=["main", "ablation", "delay_sweep", "preview", "oracle", "smoke", "all"], default="all")
    summary.add_argument("--bootstrap-samples", type=int, default=10000)

    sub.add_parser("smoke")
    return top


def _require_models() -> None:
    for path in (CHECKPOINT, NORMALIZER, MODEL_XML):
        if not path.is_file():
            raise FileNotFoundError(path)


def generate_calibration_reference(output: Path, steps: int, overwrite: bool) -> Path:
    if steps < 100:
        raise ValueError("--steps must be at least 100")
    destination = output / "calibration" / "joint_chirp.npz"
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {destination}")
    model = mujoco.MjModel.from_xml_path(str(MODEL_XML))
    q = generate_joint_reference(
        "chirp", np.zeros(model.nu, dtype=np.float32), model.jnt_range[: model.nu, 0],
        model.jnt_range[: model.nu, 1], steps, 0.01, seed=20260722,
    )
    padding = 64
    q = np.concatenate([q, np.repeat(q[-1:], padding, axis=0)])
    dq = finite_difference_dq(q, 0.01); ddq = finite_difference_dq(dq, 0.01)
    dq[steps - 1 :] = 0.0; ddq[steps - 1 :] = 0.0
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, q_des=q, dq_des=dq, ddq_des=ddq,
                        execution_steps=np.asarray(steps), control_dt=np.asarray(0.01, dtype=np.float32))
    return destination


def _base_args() -> argparse.Namespace:
    args = RUNNER.parse_args([])
    args.checkpoint = str(CHECKPOINT); args.normalizer = str(NORMALIZER)
    args.model_type = "gru"; args.history_len = 16; args.device = "cuda"
    args.horizon = 20; args.num_samples = 128; args.cem_iters = 2
    args.replan_interval_steps = 5; args.mpc_warmup_plans = 1
    args.controller_mode = "mpc"; args.mpc_policy = "residual"
    args.delay_protocol = "full"; args.dynamics_backend = "learned"
    args.visualize = False; args.settle_steps = 50
    return args


def calibrate_delay(output: Path, samples: int, provisional: int, guard_ms: float, smoke: bool) -> Path:
    if samples <= 0 or provisional <= 0 or guard_ms < 0:
        raise ValueError("samples/provisional delay must be positive and guard non-negative")
    path = output / "calibration" / ("delay_smoke.json" if smoke else "delay.json")
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite immutable calibration: {path}")
    reference = output / "calibration" / "joint_chirp.npz"
    if not reference.is_file():
        raise FileNotFoundError("Run generate-calibration-reference first")
    args = _base_args()
    args.multirate_mode = "threaded_asap"; args.anticipation_delay_steps = provisional
    args.planner_guard_ms = 0.0; args.reference_mode = "joint_file"; args.reference_file = str(reference)
    args.max_execution_steps = 40 if smoke else None
    args.horizon = 3 if smoke else args.horizon; args.num_samples = 8 if smoke else args.num_samples
    args.cem_iters = 1 if smoke else args.cem_iters
    collected: list[float] = []
    episode = 0
    target = min(samples, 3) if smoke else samples
    while len(collected) < target:
        args.seed = episode
        result = RUNNER.run_closed_loop_mpc(deepcopy(args))["arrays"]
        values = np.asarray(result.get("planner_end_to_end_latency_s", np.empty(0)), dtype=np.float64)
        collected.extend(values[np.isfinite(values)].tolist())
        episode += 1
        if smoke and episode >= 3 and not collected:
            raise RuntimeError("Threaded smoke produced no E2E latency samples")
    values = np.asarray(collected[:target])
    p95 = float(np.percentile(values, 95))
    delay = int(math.ceil((p95 + guard_ms / 1000.0) / 0.01))
    write_immutable_json(path, {
        "definition": "ceil((P95(snapshot_to_publication)+guard)/control_dt)",
        "samples": values.tolist(), "p50_s": float(np.percentile(values, 50)), "p95_s": p95,
        "guard_ms": guard_ms, "control_dt_s": 0.01, "anticipation_delay_steps": delay,
        "provisional_delay_steps": provisional, "source": "planner_end_to_end_latency_s",
    })
    return path


def _reference_configs() -> dict[str, ReferenceConfig]:
    common = dict(repeat_count=3, safe_departure_mode="auto", start_hold_duration=0.5,
                  joint_departure_duration=2.0, approach_duration=2.0, return_duration=2.0,
                  joint_return_duration=2.0, final_hold_duration=0.5)
    return {
        "circle": ReferenceConfig(shape_name="circle", lap_duration=3.0, circle_radius=0.05, **common),
        "figure8": ReferenceConfig(shape_name="figure8", lap_duration=3.0, figure8_axis_a=0.05, figure8_axis_b=0.03, **common),
        "fast_ellipse": ReferenceConfig(shape_name="ellipse", lap_duration=2.0, ellipse_axis_a=0.055, ellipse_axis_b=0.03, **common),
        "rounded_square": ReferenceConfig(shape_name="rounded_square", lap_duration=3.0, square_half_side=0.03, rounded_square_corner_radius=0.008, **common),
        "preview_calibration": ReferenceConfig(shape_name="ellipse", repeat_count=1, lap_duration=4.0,
            ellipse_axis_a=0.035, ellipse_axis_b=0.02, plane_axis_u=(1.0, 0.0, 0.0),
            plane_axis_v=(0.0, 0.0, 1.0), safe_departure_mode="auto"),
    }


def generate_references(output: Path, overwrite: bool) -> None:
    delay_path = output / "calibration" / "delay.json"
    if not delay_path.is_file():
        raise FileNotFoundError("Run calibrate-delay first")
    delay = int(load_json(delay_path)["anticipation_delay_steps"])
    model = mujoco.MjModel.from_xml_path(str(MODEL_XML))
    records: list[dict[str, Any]] = []
    for label, config in _reference_configs().items():
        destination = output / "references" / label
        path = destination / "reference.npz"
        if path.exists() and not overwrite:
            raise FileExistsError(f"Refusing to overwrite {path}")
        bundle = build_reference(config, model, np.zeros(model.nu), 0.01, 20, max(delay, 8, 4))
        saved = save_reference_bundle(bundle, destination)
        records.append({"label": label, "file": file_identity(saved), "config": bundle.metadata["config"]})
    manifest = output / "references" / "manifest.json"
    if manifest.exists() and overwrite:
        manifest.unlink()
    write_immutable_json(manifest, {"schema_version": 1, "references": records})


def calibrate_preview(output: Path, values: list[int], smoke: bool) -> Path:
    path = output / "calibration" / ("preview_smoke.json" if smoke else "preview.json")
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite immutable calibration: {path}")
    reference = output / "references" / "preview_calibration" / "reference.npz"
    if not reference.is_file():
        raise FileNotFoundError("Run generate-references first")
    rows: list[dict[str, Any]] = []
    for preview in values:
        if preview < 0:
            raise ValueError("preview values must be non-negative")
        args = _base_args(); args.controller_mode = "ik_direct"; args.multirate_mode = "synchronous"
        args.checkpoint = None; args.normalizer = None; args.reference_mode = "task"; args.reference_file = str(reference)
        args.ik_preview_steps = preview; args.max_execution_steps = 10 if smoke else None
        result = RUNNER.run_closed_loop_mpc(args)
        run_dir = output / "calibration" / "preview" / f"p{preview}"
        save_mpc_run(run_dir, result["arrays"], result["rows"])
        row = summarize_arrays(f"PreviewIK_{preview}", result["arrays"]); row["preview_steps"] = preview; rows.append(row)
    selected = min(rows, key=lambda row: (float(row["tcp_rmse_m"]), int(row["preview_steps"])))
    write_csv(output / "calibration" / "preview" / "summary.csv", rows)
    write_immutable_json(path, {"candidate_steps": values, "selected_steps": int(selected["preview_steps"]), "criterion": "minimum calibration TCP RMSE; ties choose smaller preview"})
    return path


def validate_model(output: Path, num_rollouts: int, rollout_len: int) -> None:
    _require_models()
    destination = output / "diagnostics" / "gru_validation"
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Refusing to overwrite frozen model validation: {destination}")
    command = [
        sys.executable, str(ROOT / "dynamics_modeling" / "scripts" / "eval_dynamics.py"),
        "--checkpoint", str(CHECKPOINT), "--normalizer", str(NORMALIZER),
        "--model_type", "gru", "--history_len", "16",
        "--num_rollouts", str(num_rollouts), "--rollout_len", str(rollout_len),
        "--horizons", "1,5,10,20", "--teacher_forcing", "--seed", "20260730",
        "--save_dir", str(destination),
    ]
    subprocess.run(command, cwd=ROOT, check=True)
    files = sorted(destination.glob("evaluation_rollout_*.npz"))
    write_json(destination / "validation_manifest.json", {
        "description": "Post-checkpoint frozen evaluation rollouts; not used for training",
        "checkpoint": file_identity(CHECKPOINT), "normalizer": file_identity(NORMALIZER),
        "rollouts": [file_identity(path) for path in files], "seed": 20260730,
        "horizons": [1, 5, 10, 20], "one_step_name": "ground_truth_history_one_step",
        "divergence": "nonfinite, q outside joint bounds by >0.05 rad, or |dq|>25 rad/s",
    })


def build_manifest(output: Path, allow_dirty: bool, profile: str) -> Path:
    _require_models()
    delay_file = output / "calibration" / ("delay_smoke.json" if profile == "smoke" else "delay.json")
    preview_file = output / "calibration" / ("preview_smoke.json" if profile == "smoke" else "preview.json")
    for path in (delay_file, preview_file, output / "references" / "manifest.json"):
        if not path.is_file():
            raise FileNotFoundError(path)
    environment = environment_snapshot(ROOT)
    if environment["git_dirty"] and not allow_dirty:
        raise RuntimeError("Formal paper manifest requires a clean Git worktree; use --allow-dirty only for smoke")
    base = vars(_base_args()).copy()
    for key in ("save_dir", "seed", "reference_file", "reference_mode", "controller_mode", "multirate_mode"):
        base.pop(key, None)
    references = {name: file_identity(output / "references" / name / "reference.npz") for name in (*TRAJECTORIES, "preview_calibration")}
    payload = {
        "schema_version": 2, "kind": "paper_delay_aware", "profile": profile,
        "environment": environment, "checkpoint": file_identity(CHECKPOINT), "normalizer": file_identity(NORMALIZER),
        "model_xml": file_identity(MODEL_XML), "delay_calibration": load_json(delay_file),
        "preview_calibration": load_json(preview_file), "base_run_args": base, "references": references,
        "paired_cem_seeds": [0, 1, 2, 3, 4], "delay_sweep_seeds": [0, 1, 2],
        "delay_sweep_steps": [0, 2, 4, 6, 8], "bootstrap_seed": 20260722,
    }
    path = output / "manifests" / f"{profile}.json"
    write_immutable_json(path, payload)
    return path


def _case(label: str, trajectory: str, seed: int, mode: str, protocol: str, delay: int, **extra: Any) -> dict[str, Any]:
    return {"label": label, "trajectory": trajectory, "seed": seed, "multirate_mode": mode,
            "delay_protocol": protocol, "delay_steps": delay, **extra}


def suite_cases(manifest: dict[str, Any], suite: str) -> list[dict[str, Any]]:
    delay = int(manifest["delay_calibration"]["anticipation_delay_steps"])
    seeds = [int(value) for value in manifest["paired_cem_seeds"]]
    cases: list[dict[str, Any]] = []
    if suite == "main":
        for trajectory in TRAJECTORIES:
            cases.append(_case("DirectIK", trajectory, 0, "synchronous", "full", 0, controller="ik_direct"))
            for seed in seeds:
                cases.extend([
                    _case("IdealZeroDelay", trajectory, seed, "virtual_asap", "full", 0),
                    _case("NaiveDelayed", trajectory, seed, "virtual_asap", "naive_delayed", delay),
                    _case("FullVirtual", trajectory, seed, "virtual_asap", "full", delay),
                    _case("ThreadedASAP", trajectory, seed, "threaded_asap", "full", delay),
                ])
    elif suite == "ablation":
        labels = (("FullVirtual", "full"), ("NoFutureAlignment", "no_future_alignment"),
                  ("NoReanchor", "no_reanchor"), ("NoFeedback", "no_feedback"))
        for trajectory in TRAJECTORIES:
            for seed in seeds:
                cases.extend(_case(label, trajectory, seed, "virtual_asap", protocol, delay) for label, protocol in labels)
    elif suite == "delay_sweep":
        for trajectory in ("circle", "fast_ellipse"):
            for seed in manifest["delay_sweep_seeds"]:
                for imposed in manifest["delay_sweep_steps"]:
                    cases.append(_case(f"Full_D{imposed}", trajectory, int(seed), "virtual_asap", "full", int(imposed)))
                    cases.append(_case(f"Naive_D{imposed}", trajectory, int(seed), "virtual_asap", "naive_delayed", int(imposed)))
    elif suite == "preview":
        preview = int(manifest["preview_calibration"]["selected_steps"])
        for trajectory in TRAJECTORIES:
            cases.append(_case("PreviewIK", trajectory, 0, "synchronous", "full", 0, controller="ik_direct", preview_steps=preview))
    elif suite == "oracle":
        for trajectory in ("circle", "fast_ellipse"):
            for seed in (0, 1, 2):
                cases.append(_case("LearnedFull", trajectory, seed, "virtual_asap", "full", delay))
                cases.append(_case("OracleUpperBound", trajectory, seed, "virtual_asap", "full", delay, dynamics_backend="mujoco_oracle"))
    else:
        raise ValueError(suite)
    return cases


def _run_case(output: Path, manifest: dict[str, Any], case: dict[str, Any], resume: bool, suite: str) -> dict[str, Any]:
    args = RUNNER.parse_args([])
    for key, value in manifest["base_run_args"].items():
        if hasattr(args, key): setattr(args, key, value)
    reference = Path(manifest["references"][case["trajectory"]]["path"])
    args.reference_mode = "task"; args.reference_file = str(reference); args.seed = int(case["seed"])
    args.multirate_mode = case["multirate_mode"]; args.delay_protocol = case["delay_protocol"]
    args.anticipation_delay_steps = int(case["delay_steps"]); args.controller_mode = case.get("controller", "mpc")
    args.ik_preview_steps = int(case.get("preview_steps", 0)); args.dynamics_backend = case.get("dynamics_backend", "learned")
    if args.controller_mode == "ik_direct": args.checkpoint = args.normalizer = None
    fingerprint_case = {key: value for key, value in case.items() if key != "label"}
    payload = {"manifest_commit": manifest["environment"]["git_commit"], "case": fingerprint_case,
               "run_args": {key: value for key, value in vars(args).items() if key != "save_dir"},
               "reference": file_identity(reference), "checkpoint": None if not args.checkpoint else manifest["checkpoint"],
               "normalizer": None if not args.normalizer else manifest["normalizer"]}
    fingerprint = run_fingerprint(payload)
    run_dir = output / "runs" / "cache" / fingerprint["sha256"]
    arrays = load_completed_rollout(run_dir, fingerprint)
    if arrays is None:
        args.save_dir = str(run_dir)
        result = RUNNER.run_closed_loop_mpc(args); arrays = result["arrays"]
        save_mpc_run(run_dir, arrays, result["rows"])
        write_json(run_dir / "run_fingerprint.json", fingerprint)
    elif resume:
        print(f"Reused completed rollout: {run_dir}")
    return {**case, "suite": suite, "fingerprint": fingerprint["sha256"], "run_dir": str(run_dir)}


def run_suite(output: Path, manifest_path: Path, suite: str, resume: bool, case_limit: int | None) -> None:
    manifest = load_json(manifest_path)
    suites = ("main", "ablation", "delay_sweep", "preview", "oracle") if suite == "all" else (suite,)
    for name in suites:
        cases = suite_cases(manifest, name)
        if case_limit is not None: cases = cases[:case_limit]
        entries = [_run_case(output, manifest, case, resume, name) for case in cases]
        write_json(output / "runs" / "indexes" / f"{name}.json", {"suite": name, "entries": entries})
        summarize(output, name, 10000)


def summarize(output: Path, suite: str, bootstrap_samples: int) -> None:
    suites = ("main", "ablation", "delay_sweep", "preview", "oracle", "smoke") if suite == "all" else (suite,)
    for name in suites:
        index_path = output / "runs" / "indexes" / f"{name}.json"
        if not index_path.is_file():
            continue
        entries = load_json(index_path)["entries"]
        rows: list[dict[str, Any]] = []
        for entry in entries:
            with np.load(Path(entry["run_dir"]) / "rollout.npz", allow_pickle=False) as archive:
                arrays = {key: np.asarray(archive[key]) for key in archive.files}
            row = summarize_arrays(str(entry["label"]), arrays)
            row.update({key: entry[key] for key in ("trajectory", "seed", "label")})
            row["case_id"] = f"{entry['trajectory']}:{entry['seed']}"
            rows.append(row)
        write_csv(output / "summaries" / f"{name}.csv", rows)
        groups = ("label", "trajectory") if name != "delay_sweep" else ("delay_protocol", "trajectory", "delay_steps")
        write_csv(output / "summaries" / f"{name}_aggregate.csv", aggregate_rows(rows, groups))
        if name == "main":
            comparisons = {
                "FullVirtual_minus_NaiveDelayed": paired_bootstrap_rows(
                    rows, left="NaiveDelayed", right="FullVirtual",
                    metrics=("tcp_rmse_m", "tcp_p95_m", "orientation_rmse_rad", "failure_rate"),
                    samples=bootstrap_samples, seed=20260722),
                "ThreadedASAP_minus_NaiveDelayed": paired_bootstrap_rows(
                    rows, left="NaiveDelayed", right="ThreadedASAP",
                    metrics=("tcp_rmse_m", "tcp_p95_m", "orientation_rmse_rad", "failure_rate"),
                    samples=bootstrap_samples, seed=20260723),
            }
            write_json(output / "summaries" / "main_paired_bootstrap.json", comparisons)
            write_json(output / "summaries" / "latency_recovery.json", latency_recovery(rows))
        elif name == "ablation":
            comparisons = {}
            for index, variant in enumerate(("NoFutureAlignment", "NoReanchor", "NoFeedback")):
                comparisons[f"{variant}_minus_FullVirtual"] = paired_bootstrap_rows(
                    rows, left="FullVirtual", right=variant,
                    metrics=(
                        "tcp_rmse_m", "tcp_p95_m", "orientation_rmse_rad",
                        "failure_rate", "projection_discrepancy_rms_rad",
                    ),
                    samples=bootstrap_samples, seed=20260722 + index,
                )
            write_json(output / "summaries" / "ablation_paired_bootstrap.json", comparisons)


def smoke(output: Path) -> None:
    _require_models()
    reference = ROOT / "outputs" / "references" / "circle_3laps" / "reference.npz"
    if not reference.is_file(): raise FileNotFoundError(reference)
    base = _base_args(); base.reference_mode = "task"; base.reference_file = str(reference)
    cuda_available = bool(torch.cuda.is_available())
    if not cuda_available:
        base.device = "cpu"
    base.horizon = 3; base.num_samples = 8; base.cem_iters = 1; base.max_execution_steps = 3
    base.replan_interval_steps = 1; base.anticipation_delay_steps = 1; base.mpc_warmup_plans = 0
    specs = [
        ("IdealZeroDelay", "virtual_asap", "full", 0, "mpc"),
        ("FullVirtual", "virtual_asap", "full", 1, "mpc"),
        ("NaiveDelayed", "virtual_asap", "naive_delayed", 1, "mpc"),
        ("NoFutureAlignment", "virtual_asap", "no_future_alignment", 1, "mpc"),
        ("NoReanchor", "virtual_asap", "no_reanchor", 1, "mpc"),
        ("NoFeedback", "virtual_asap", "no_feedback", 1, "mpc"),
        ("DirectIK", "synchronous", "full", 0, "ik_direct"),
    ]
    if cuda_available:
        specs.append(("ThreadedASAP", "threaded_asap", "full", 1, "mpc"))
    entries: list[dict[str, Any]] = []
    for label, mode, protocol, delay, controller in specs:
        args = deepcopy(base); args.multirate_mode = mode; args.delay_protocol = protocol
        args.anticipation_delay_steps = delay; args.controller_mode = controller
        if controller == "ik_direct": args.checkpoint = args.normalizer = None
        payload = {"smoke": True, "label": label, "args": {key: value for key, value in vars(args).items() if key != "save_dir"}}
        fingerprint = run_fingerprint(payload); run_dir = output / "smoke" / label
        result = RUNNER.run_closed_loop_mpc(args); save_mpc_run(run_dir, result["arrays"], result["rows"])
        write_json(run_dir / "run_fingerprint.json", fingerprint)
        entries.append({"label": label, "trajectory": "circle_smoke", "seed": 0, "suite": "smoke", "fingerprint": fingerprint["sha256"], "run_dir": str(run_dir)})
    write_json(output / "runs" / "indexes" / "smoke.json", {"suite": "smoke", "entries": entries})
    write_json(output / "smoke" / "environment_status.json", {
        "cuda_available": cuda_available,
        "threaded_smoke": "completed" if cuda_available else "skipped_cuda_unavailable",
    })
    summarize(output, "smoke", 100)


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv); output = resolve(args.output_root)
    output.mkdir(parents=True, exist_ok=True)
    (output / "logs").mkdir(parents=True, exist_ok=True)
    if args.command == "generate-calibration-reference":
        print(generate_calibration_reference(output, args.steps, args.overwrite))
    elif args.command == "calibrate-delay":
        print(calibrate_delay(output, args.samples, args.provisional_delay, args.guard_ms, args.smoke))
    elif args.command == "generate-references": generate_references(output, args.overwrite)
    elif args.command == "calibrate-preview":
        values = [int(value) for value in args.preview_values.split(",") if value.strip()]
        print(calibrate_preview(output, values, args.smoke))
    elif args.command == "validate-model": validate_model(output, args.num_rollouts, args.rollout_len)
    elif args.command == "build-manifest": print(build_manifest(output, args.allow_dirty, args.profile))
    elif args.command == "run":
        manifest = resolve(args.manifest) if args.manifest else output / "manifests" / "paper.json"
        run_suite(output, manifest, args.suite, args.resume, args.case_limit)
    elif args.command == "summarize": summarize(output, args.suite, args.bootstrap_samples)
    elif args.command == "smoke": smoke(output)


if __name__ == "__main__":
    main()
