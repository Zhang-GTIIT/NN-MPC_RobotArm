from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DYNAMICS_ROOT = ROOT / "learned_mujoco_dynamics"
for path in (ROOT, DYNAMICS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import numpy as np


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze MPC query OOD statistics against each model's training dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--pair",
        action="append",
        required=True,
        help="Repeat as label,mpc_rollout_npz,training_dataset_npz.",
    )
    parser.add_argument("--baseline_dataset", default=None, type=str)
    parser.add_argument("--save_csv", default="outputs/mpc/ood_summary.csv", type=str)
    parser.add_argument("--z_threshold", default=3.0, type=float)
    return parser.parse_args(argv)


def parse_pair(value: str) -> tuple[str, Path, Path]:
    parts = [item.strip() for item in value.split(",")]
    if len(parts) != 3:
        raise ValueError(f"--pair must be label,mpc_rollout_npz,training_dataset_npz, got {value!r}")
    return parts[0], resolve_runtime_path(parts[1]), resolve_runtime_path(parts[2])


def resolve_runtime_path(path: str) -> Path:
    expanded = Path(path).expanduser()
    if expanded.is_absolute():
        return expanded
    root_path = ROOT / expanded
    if root_path.exists() or (expanded.parts and expanded.parts[0] == "learned_mujoco_dynamics"):
        return root_path
    dynamics_path = DYNAMICS_ROOT / expanded
    if dynamics_path.exists():
        return dynamics_path
    if expanded.parts and expanded.parts[0] == "outputs":
        return root_path
    return root_path


def load_training_distribution(dataset_path: Path) -> dict[str, np.ndarray]:
    with np.load(dataset_path) as data:
        states = np.asarray(data["states"], dtype=np.float64)
        actions = np.asarray(data["actions"], dtype=np.float64)
    return {
        "state_mean": states.mean(axis=0),
        "state_std": states.std(axis=0).clip(min=1e-8),
        "action_mean": actions.mean(axis=0),
        "action_std": actions.std(axis=0).clip(min=1e-8),
    }


def summarize_against_distribution(
    label: str,
    rollout_path: Path,
    dataset_label: str,
    distribution: dict[str, np.ndarray],
    z_threshold: float,
) -> dict[str, float | str | int]:
    with np.load(rollout_path) as rollout:
        states = np.asarray(rollout["actual_states"], dtype=np.float64)
        actions = np.asarray(rollout["actuator_q_ref"], dtype=np.float64)
        failure_flags = np.asarray(rollout["failure_flags"], dtype=np.float64) if "failure_flags" in rollout.files else np.zeros(len(states))
        gap = np.asarray(rollout["predicted_real_error_gap"], dtype=np.float64) if "predicted_real_error_gap" in rollout.files else np.empty(0)

    state_z = np.abs((states - distribution["state_mean"]) / distribution["state_std"])
    action_z = np.abs((actions - distribution["action_mean"]) / distribution["action_std"])
    return {
        "label": label,
        "dataset_label": dataset_label,
        "rollout_path": str(rollout_path),
        "samples": int(len(states)),
        "state_z_mean": float(np.mean(state_z)) if state_z.size else float("nan"),
        "state_z_max": float(np.max(state_z)) if state_z.size else float("nan"),
        "state_ood_fraction": float(np.mean(np.any(state_z > z_threshold, axis=1))) if state_z.size else float("nan"),
        "action_z_mean": float(np.mean(action_z)) if action_z.size else float("nan"),
        "action_z_max": float(np.max(action_z)) if action_z.size else float("nan"),
        "action_ood_fraction": float(np.mean(np.any(action_z > z_threshold, axis=1))) if action_z.size else float("nan"),
        "failure_rate": float(np.mean(failure_flags)) if len(failure_flags) else float("nan"),
        "predicted_real_error_gap_mean": float(np.mean(gap)) if len(gap) else float("nan"),
    }


def write_rows(path: Path, rows: list[dict[str, float | str | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    rows: list[dict[str, float | str | int]] = []
    baseline_distribution = None if args.baseline_dataset is None else load_training_distribution(resolve_runtime_path(args.baseline_dataset))
    for pair_text in args.pair:
        label, rollout_path, dataset_path = parse_pair(pair_text)
        own_distribution = load_training_distribution(dataset_path)
        rows.append(summarize_against_distribution(label, rollout_path, "own_training_dataset", own_distribution, args.z_threshold))
        if baseline_distribution is not None:
            rows.append(summarize_against_distribution(label, rollout_path, "baseline_dataset", baseline_distribution, args.z_threshold))
    save_csv = resolve_runtime_path(args.save_csv)
    write_rows(save_csv, rows)
    print(f"Saved OOD summary to {save_csv}")


if __name__ == "__main__":
    main()
