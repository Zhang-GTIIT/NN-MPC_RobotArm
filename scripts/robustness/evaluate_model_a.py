"""Evaluate Model-A threaded MPC and Direct IK under fixed robustness cases."""
from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.robustness._evaluation import (
    build_fingerprint, load_completed_rollout, paired_bootstrap, sha256, summarize, write_summary,
)
from scripts.robustness._runtime import ensure_import_paths, load_runner

ensure_import_paths()

from mpc.logging import save_mpc_run


RUNNER = load_runner("model_a_robustness_runner")
ROBUSTNESS_LEVELS = ("payload_level", "actuator_gain_level", "force_pulse_level", "observation_noise_level")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = RUNNER.build_arg_parser()
    parser.description = "Evaluate frozen Model-A threaded MPC and Direct IK under robustness perturbations."
    parser.add_argument("--manifest", default="outputs/robustness/benchmark.json")
    parser.add_argument("--label", default="ModelA_MPC")
    parser.add_argument("--direct_ik_label", default="DirectIK")
    parser.add_argument("--bootstrap_samples", type=int, default=2000)
    parser.add_argument("--bootstrap_seed", type=int, default=20260722)
    parser.add_argument("--case_ids", default=None, help="Optional comma-separated immutable case IDs.")
    parser.add_argument("--resume", action="store_true")
    parser.set_defaults(save_dir="outputs/robustness/runs")
    return parser.parse_args(argv)


def selected_cases(manifest: dict[str, object], case_ids: str | None) -> list[dict[str, object]]:
    cases = manifest.get("cases")
    if not isinstance(cases, list) or not cases or not all(isinstance(case, dict) for case in cases):
        raise ValueError("Benchmark manifest must contain a non-empty object cases list")
    selected = list(cases)
    if case_ids:
        requested = {value.strip() for value in case_ids.split(",") if value.strip()}
        if not requested:
            raise ValueError("--case_ids must contain at least one case ID")
        available = {str(case.get("id")) for case in selected}
        missing = requested - available
        if missing:
            raise ValueError(f"Unknown --case_ids: {sorted(missing)}")
        selected = [case for case in selected if str(case.get("id")) in requested]
    for case in selected:
        run_args = case.get("run_args", case)
        reference = run_args.get("reference_file") if isinstance(run_args, dict) else None
        if not isinstance(reference, str) or not Path(reference).is_file():
            raise FileNotFoundError(f"Missing reference for benchmark case {case.get('id')}")
        expected = case.get("reference_sha256")
        if expected is not None and sha256(Path(reference)) != expected:
            raise ValueError(f"Benchmark reference hash mismatch for case {case.get('id')}")
    return selected


def run_one(args: argparse.Namespace, case: dict[str, object], controller: dict[str, object], save_root: Path) -> dict[str, float | str | int]:
    case_id = str(case["id"])
    run_args = deepcopy(args)
    values = case.get("run_args", case)
    assert isinstance(values, dict)
    for key, value in values.items():
        if key != "id" and hasattr(run_args, key):
            setattr(run_args, key, value)
    # CLI perturbation levels are experiment-level variables, never manifest defaults.
    for key in ROBUSTNESS_LEVELS:
        setattr(run_args, key, int(getattr(args, key)))
    kind = str(controller["kind"])
    if kind == "model_a":
        run_args.controller_mode = "mpc"
        run_args.mpc_policy = "residual"
        run_args.dynamics_backend = "learned"
        run_args.multirate_mode = "threaded_asap"
        run_args.checkpoint = str(controller["checkpoint"])
        run_args.normalizer = str(controller["normalizer"])
        run_args.model_type = str(controller["model_type"])
    else:
        run_args.controller_mode = "ik_direct"
        run_args.dynamics_backend = "learned"
        run_args.multirate_mode = "synchronous"
        run_args.checkpoint = None
        run_args.normalizer = None
    label = str(controller["label"])
    run_dir = save_root / case_id / label
    run_args.save_dir = str(run_dir)
    fingerprint = build_fingerprint(case_id, case, controller, run_args, RUNNER)
    arrays = load_completed_rollout(run_dir, fingerprint) if args.resume else None
    if arrays is None:
        result = RUNNER.run_closed_loop_mpc(run_args)
        arrays = result["arrays"]
        save_mpc_run(run_dir, arrays, result["rows"])
        (run_dir / "run_fingerprint.json").write_text(json.dumps(fingerprint, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    else:
        print(f"Reused completed rollout: {run_dir}")
    row = summarize(label, arrays)
    row["case_id"] = case_id
    row["reference_type"] = str(case.get("reference_type", "unknown"))
    row["control_protocol"] = "threaded_asap" if kind == "model_a" else "synchronous_direct_ik"
    return row


def main() -> None:
    args = parse_args()
    if args.bootstrap_samples <= 0:
        raise ValueError("--bootstrap_samples must be positive")
    if not args.checkpoint or not args.normalizer:
        raise ValueError("--checkpoint and --normalizer are required for Model-A robustness evaluation")
    manifest_path = RUNNER.resolve_runtime_path(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("kind") != "model_a_robustness":
        raise ValueError("--manifest must be produced by scripts/robustness/build_benchmark_manifest.py")
    cases = selected_cases(manifest, args.case_ids)
    controllers = [
        {"label": args.label, "kind": "model_a", "checkpoint": args.checkpoint, "normalizer": args.normalizer, "model_type": args.model_type},
        {"label": args.direct_ik_label, "kind": "direct_ik"},
    ]
    if len({str(item["label"]) for item in controllers}) != len(controllers):
        raise ValueError("--label and --direct_ik_label must differ")
    save_root = RUNNER.resolve_runtime_path(args.save_dir)
    rows: list[dict[str, float | str | int]] = []
    for case in cases:
        values = case.get("run_args", case)
        assert isinstance(values, dict)
        for controller in controllers:
            if controller["kind"] == "direct_ik" and values.get("reference_mode") != "task":
                continue
            rows.append(run_one(args, case, controller, save_root))
            write_summary(save_root / "model_a_robustness_summary.csv", rows)
    write_summary(save_root / "model_a_robustness_summary.csv", rows)
    (save_root / "paired_bootstrap.json").write_text(json.dumps(paired_bootstrap(rows, args.bootstrap_samples, args.bootstrap_seed), indent=2, allow_nan=True) + "\n", encoding="utf-8")
    print(f"Saved Model-A robustness summary to {save_root / 'model_a_robustness_summary.csv'}")


if __name__ == "__main__":
    main()
