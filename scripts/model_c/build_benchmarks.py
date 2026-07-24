"""Create immutable, fully specified Model-C development/final manifests."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.model_c._runtime import ROOT, load_runner
DEFAULT_TYPES = ("multi_joint_sine", "waypoint", "chirp", "circle", "figure8", "ellipse", "square")


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _runner_defaults() -> dict[str, Any]:
    module = load_runner("model_c_benchmark_runner")
    defaults = vars(module.build_arg_parser().parse_args([]))
    excluded = {"checkpoint", "normalizer", "model_type", "history_len", "device", "seed", "save_dir", "reference_mode", "reference_file", "episode_len", "horizon", "anticipation_delay_steps", "multirate_mode", "payload_level", "actuator_gain_level", "force_pulse_level", "observation_noise_level"}
    return {key: value for key, value in defaults.items() if key not in excluded}


def _reference_paths_from_manifest(path: Path) -> set[tuple[Path, str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        (Path(case["run_args"]["reference_file"]).resolve(), str(case["reference_sha256"]))
        for case in payload.get("cases", []) if "reference_sha256" in case
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a fixed development or final benchmark manifest.")
    parser.add_argument("--kind", choices=("development", "final"), required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--cases_per_type", type=int, default=10)
    parser.add_argument("--task_reference_dir", required=True, help="Contains immutable <type>_<index>/reference.npz bundles for every benchmark type.")
    parser.add_argument("--completion_marker", default="C2_COMPLETE")
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--delay", type=int, required=True)
    parser.add_argument("--disjoint_from", default=None, help="Required for final: development manifest whose reference files/hashes must not overlap.")
    args = parser.parse_args()
    if args.cases_per_type <= 0 or args.delay <= 0:
        raise ValueError("--cases_per_type and --delay must be positive")
    output = _resolve(args.output_path)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite immutable benchmark manifest: {output}")
    if args.kind == "final" and not args.disjoint_from:
        raise ValueError("Final benchmark requires --disjoint_from DEVELOPMENT_MANIFEST")
    previous = set() if not args.disjoint_from else _reference_paths_from_manifest(_resolve(args.disjoint_from))
    task_dir = _resolve(args.task_reference_dir)
    locked = _runner_defaults()
    locked.update({"n_joints": 6, "controller_mode": "mpc", "mpc_policy": "residual", "num_samples": 128, "cem_iters": 2, "replan_interval_steps": 5, "rollout_batch_size": 128})
    cases: list[dict[str, object]] = []
    for type_index, reference_type in enumerate(DEFAULT_TYPES):
        for ordinal in range(args.cases_per_type):
            reference_path = task_dir / f"{reference_type}_{ordinal:02d}" / "reference.npz"
            if not reference_path.exists():
                raise FileNotFoundError(f"Missing immutable benchmark reference: {reference_path}")
            reference_hash = _sha256(reference_path)
            if (reference_path.resolve(), reference_hash) in previous or any(reference_hash == known_hash for _, known_hash in previous):
                raise ValueError(f"Benchmark reference overlaps --disjoint_from: {reference_path}")
            run_args: dict[str, object] = dict(locked)
            run_args.update({
                "seed": args.seed + type_index * 10_000 + ordinal,
                "episode_len": 500, "horizon": args.horizon, "anticipation_delay_steps": args.delay,
                "multirate_mode": "virtual_asap", "reference_file": str(reference_path),
                "reference_mode": "joint_file" if reference_type in {"multi_joint_sine", "waypoint", "chirp"} else "task",
            })
            cases.append({"id": f"{reference_type}_{ordinal:02d}", "reference_type": reference_type,
                          "reference_sha256": reference_hash, "run_args": run_args})
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema_version": 2, "kind": args.kind, "seed": args.seed, "cases": cases,
               "locked_controller_config": locked, "completion_marker": args.completion_marker if args.kind == "final" else None,
               "note": "Immutable manifest: evaluator verifies all reference hashes and manifest values override CLI controller settings."}
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {len(cases)} immutable {args.kind} cases to {output}")


if __name__ == "__main__":
    main()
