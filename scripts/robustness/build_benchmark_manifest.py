"""Build one immutable threaded-asap benchmark manifest for Model-A robustness."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.robustness._evaluation import sha256
from scripts.robustness._runtime import ROOT, load_runner


TYPES = ("multi_joint_sine", "waypoint", "chirp", "circle", "figure8", "ellipse", "square")
TASK_TYPES = {"circle", "figure8", "ellipse", "square"}


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def runner_defaults() -> dict[str, Any]:
    runner = load_runner("robustness_manifest_runner")
    defaults = vars(runner.build_arg_parser().parse_args([]))
    excluded = {"checkpoint", "normalizer", "model_type", "history_len", "device", "seed", "save_dir", "reference_mode", "reference_file", "episode_len", "horizon", "anticipation_delay_steps", "multirate_mode", "payload_level", "actuator_gain_level", "force_pulse_level", "observation_noise_level"}
    return {key: value for key, value in defaults.items() if key not in excluded}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build immutable Model-A threaded robustness benchmark manifest.")
    parser.add_argument("--output_path", default="outputs/robustness/benchmark.json")
    parser.add_argument("--reference_dir", default="outputs/robustness/references")
    parser.add_argument("--cases_per_type", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--horizon", type=int, default=25)
    parser.add_argument("--delay", required=True, type=int)
    args = parser.parse_args()
    if args.cases_per_type <= 0 or args.horizon <= 0 or args.delay <= 0:
        raise ValueError("--cases_per_type, --horizon, and --delay must be positive")
    output, references = resolve(args.output_path), resolve(args.reference_dir)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite immutable robustness manifest: {output}")
    locked = runner_defaults()
    locked.update(n_joints=6, controller_mode="mpc", mpc_policy="residual", multirate_mode="threaded_asap", num_samples=128, cem_iters=2, replan_interval_steps=5, rollout_batch_size=128)
    cases: list[dict[str, object]] = []
    for type_index, reference_type in enumerate(TYPES):
        for ordinal in range(args.cases_per_type):
            reference = references / f"{reference_type}_{ordinal:02d}" / "reference.npz"
            if not reference.is_file():
                raise FileNotFoundError(f"Missing benchmark reference: {reference}")
            run_args = dict(locked)
            run_args.update(seed=args.seed + type_index * 10_000 + ordinal, episode_len=500, horizon=args.horizon,
                            anticipation_delay_steps=args.delay, reference_file=str(reference),
                            reference_mode="task" if reference_type in TASK_TYPES else "joint_file")
            cases.append({"id": f"{reference_type}_{ordinal:02d}", "reference_type": reference_type,
                          "reference_sha256": sha256(reference), "run_args": run_args})
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema_version": 1, "kind": "model_a_robustness", "seed": args.seed, "cases": cases,
               "locked_controller_config": locked,
               "note": "Immutable Model-A robustness benchmark; MPC runs threaded_asap and Direct IK runs synchronous."}
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {len(cases)} immutable Model-A robustness cases to {output}")


if __name__ == "__main__":
    main()
