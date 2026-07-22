"""Shard-based virtual-ASAP MPC-induced data collection.

This collector deliberately owns persistence; it never uses the legacy
read-concatenate-rewrite ``--append`` path.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.model_c._runtime import ROOT, ensure_import_paths, load_runner

ensure_import_paths()

import numpy as np
import mujoco
import torch

from mpc.model_c.branches import (
    evaluate_branch_cost,
    execute_exact_action_branch,
    execute_open_loop_action_branch,
    project_activation_q_ref_sequence,
)
from neural_dynamics.rollout import rollout_dynamics_batch
from mpc.reference_pipeline import ReferenceConfig, build_reference, save_reference_bundle


RUNNER = load_runner("model_c_collect_runner")
ROLE_BITS = {"selected": 1, "baseline": 2, "alternative_elite": 4}
ACTIVATION_REASON_IDS = {"": 0, "joint_limit": 1, "velocity": 2, "acceleration": 3}
REFERENCE_MODE_IDS = {"multi_joint_sine": 0, "waypoint": 1, "chirp": 2, "circle": 3, "figure8": 4}
BRANCH_KIND_IDS = {"strict_exact_action": 0, "activation_projected": 1}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = RUNNER.build_arg_parser()
    parser.description = "Collect immutable Model-C MPC-induced shards."
    parser.add_argument("--output_dir", default="outputs/model_c/round1", type=str)
    parser.add_argument("--episodes", default=800, type=int)
    parser.add_argument("--shard_main_episodes", default=25, type=int)
    parser.add_argument("--branch_horizon", default=25, type=int)
    parser.add_argument("--branch_targets", default="0,50,100,150,200,250,300,350,400,450", type=str)
    parser.add_argument("--trajectory_counts", default="multi_joint_sine:320,waypoint:240,chirp:120,circle:80,figure8:40")
    parser.add_argument("--timing_json", default=None, type=str)
    parser.add_argument("--round_name", default="round1", type=str)
    parser.set_defaults(
        model_type="gru", history_len=16, horizon=25, num_samples=128, cem_iters=2,
        replan_interval_steps=5, multirate_mode="virtual_asap", episode_len=500,
        force_baseline_candidate=True,
    )
    return parser.parse_args(argv)


def _stack(records: list[np.ndarray], dtype: np.dtype[Any]) -> np.ndarray:
    return np.asarray(records, dtype=dtype)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_savez(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.stem}.", suffix=".tmp", delete=False) as file:
        temporary = Path(file.name)
        np.savez_compressed(file, **arrays)
    temporary.replace(path)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.stem}.", suffix=".tmp", mode="w", encoding="utf-8", delete=False) as file:
        json.dump(value, file, indent=2, sort_keys=True)
        file.write("\n")
        temporary = Path(file.name)
    temporary.replace(path)


def _parse_counts(value: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in value.split(","):
        name, separator, count = item.strip().partition(":")
        if not separator or name not in REFERENCE_MODE_IDS or int(count) <= 0:
            raise ValueError("trajectory_counts must contain positive name:count entries for supported modes")
        counts[name] = int(count)
    return counts


def _task_reference(args: argparse.Namespace, name: str, episode_id: int, output_dir: Path) -> Path:
    """Generate one validated, exactly-500-step collection task reference."""
    rng = np.random.default_rng(10_000 + episode_id)
    planes = [((1., 0., 0.), (0., 1., 0.)), ((1., 0., 0.), (0., 0., 1.)), ((0., 1., 0.), (0., 0., 1.))]
    axis_u, axis_v = planes[int(rng.integers(len(planes)))]
    if rng.integers(2):
        axis_v = tuple(-value for value in axis_v)
    config = ReferenceConfig(
        shape_name=name, repeat_count=1, collection_only=True, safe_departure_mode="always",
        start_hold_duration=0.2, joint_departure_duration=0.79, approach_duration=0.79,
        lap_duration=2.99, shape_end_hold_duration=0.2,
        center_offset=tuple(rng.uniform(-0.01, 0.01, size=3)), plane_axis_u=axis_u, plane_axis_v=axis_v,
        circle_radius=float(rng.uniform(0.02, 0.04)), figure8_axis_a=float(rng.uniform(0.02, 0.04)),
        figure8_axis_b=float(rng.uniform(0.012, 0.025)),
    )
    model = mujoco.MjModel.from_xml_path(str(RUNNER.resolve_runtime_path(args.model_xml)))
    bundle = build_reference(config, model, np.zeros(args.n_joints), 0.01, args.horizon, args.anticipation_delay_steps)
    if bundle.execution_steps != 500 or int(np.sum(bundle.segment_ids[:500] == 3)) != 300:
        raise RuntimeError(f"Collection task reference must have 500 execution steps and 300 loop steps, got {bundle.execution_steps}")
    path = output_dir / "references" / f"{name}_{episode_id:05d}"
    path.mkdir(parents=True, exist_ok=True)
    return save_reference_bundle(bundle, path)


def _write_shard(output_dir: Path, shard_index: int, records: dict[str, list[np.ndarray]], branches: list[dict[str, Any]]) -> None:
    transition_path = output_dir / f"transitions_{shard_index:05d}.npz"
    branch_path = output_dir / f"branches_{shard_index:05d}.npz"
    integer_fields = {
        "episode_id", "split_group_id", "source_id", "branch_id", "reference_mode_id", "branch_kind_id",
        "control_step", "valid_target", "context_only", "recovery_active", "packet_fallback",
    }
    arrays = {key: _stack(values, np.int64 if key in integer_fields else np.float32) for key, values in records.items()}
    _atomic_savez(transition_path, **arrays)
    if branches:
        arrays = {
            "branch_id": np.asarray([item["branch_id"] for item in branches], dtype=np.int64),
            "candidate_group_id": np.asarray([item["candidate_group_id"] for item in branches], dtype=np.int64),
            "branch_kind_id": np.asarray([item["branch_kind_id"] for item in branches], dtype=np.int8),
            "parent_main_episode_id": np.asarray([item["parent_main_episode_id"] for item in branches], dtype=np.int64),
            "activation_step": np.asarray([item["activation_step"] for item in branches], dtype=np.int64),
            "target_step": np.asarray([item["target_step"] for item in branches], dtype=np.int64),
            "role_mask": np.asarray([item["role_mask"] for item in branches], dtype=np.int8),
            "activation_infeasible": np.asarray([item["activation_infeasible"] for item in branches], dtype=np.int8),
            "activation_infeasible_reason_id": np.asarray([item["activation_infeasible_reason_id"] for item in branches], dtype=np.int8),
            "predicted_anchor_state": np.asarray([item["predicted_anchor_state"] for item in branches], dtype=np.float32),
            "actual_activation_state": np.asarray([item["actual_activation_state"] for item in branches], dtype=np.float32),
            "actual_previous_command": np.asarray([item["actual_previous_command"] for item in branches], dtype=np.float32),
            "actual_previous_velocity": np.asarray([item["actual_previous_velocity"] for item in branches], dtype=np.float32),
            "planned_q_ref_sequence": np.asarray([item["planned_q_ref_sequence"] for item in branches], dtype=np.float32),
            "executed_q_ref_sequence": np.asarray([item["executed_q_ref_sequence"] for item in branches], dtype=np.float32),
            "predicted_state_sequence": np.asarray([item["predicted_state_sequence"] for item in branches], dtype=np.float32),
            "realized_state_sequence": np.asarray([item["realized_state_sequence"] for item in branches], dtype=np.float32),
            "predicted_cost": np.asarray([item["predicted_cost"] for item in branches], dtype=np.float32),
            "realized_cost": np.asarray([item["realized_cost"] for item in branches], dtype=np.float32),
            "anchor_prediction_error": np.asarray([item["anchor_prediction_error"] for item in branches], dtype=np.float32),
            "q_des_sequence": np.asarray([item["q_des_sequence"] for item in branches], dtype=np.float32),
            "dq_des_sequence": np.asarray([item["dq_des_sequence"] for item in branches], dtype=np.float32),
        }
        _atomic_savez(branch_path, **arrays)


def main() -> None:
    args = parse_args()
    robustness_levels = {
        name: int(getattr(args, name, 0))
        for name in ("payload_level", "actuator_gain_level", "force_pulse_level", "observation_noise_level")
    }
    if any(robustness_levels.values()):
        raise ValueError(f"Model-C collection forbids robustness perturbations: {robustness_levels}")
    if args.multirate_mode != "virtual_asap" or args.episode_len != 500:
        raise ValueError("Model-C collection requires --multirate_mode virtual_asap and --episode_len 500")
    expected = {"model_type": "gru", "history_len": 16, "horizon": 25, "branch_horizon": 25, "num_samples": 128, "cem_iters": 2, "replan_interval_steps": 5}
    mismatch = {name: (getattr(args, name), value) for name, value in expected.items() if getattr(args, name) != value}
    if mismatch:
        raise ValueError(f"Model-C collection is fixed to {expected}; received incompatible values {mismatch}")
    output_dir = RUNNER.resolve_runtime_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    targets = [int(value) for value in args.branch_targets.split(",") if value.strip()]
    if any(value < 0 or value >= args.episode_len for value in targets):
        raise ValueError("branch_targets must lie in [0, episode_len)")
    counts = _parse_counts(args.trajectory_counts)
    if sum(counts.values()) != args.episodes:
        raise ValueError(f"trajectory_counts sum={sum(counts.values())} must equal --episodes={args.episodes}")
    schedule = [name for name, count in counts.items() for _ in range(count)]
    np.random.default_rng(args.seed).shuffle(schedule)
    if args.timing_json:
        timing_path = RUNNER.resolve_runtime_path(args.timing_json)
        args.anticipation_delay_steps = int(json.loads(timing_path.read_text(encoding="utf-8"))["anticipation_delay_steps"])
    physical_velocity = RUNNER._parse_joint_vector(
        args.command_velocity_physical_limit, args.n_joints, "command_velocity_physical_limit"
    )
    physical_acceleration = RUNNER._parse_joint_vector(
        args.command_acceleration_physical_limit, args.n_joints, "command_acceleration_physical_limit"
    )
    manifest_path = output_dir / "manifest.json"
    base_seed = int(args.seed)
    runtime_paths = {name: RUNNER.resolve_runtime_path(getattr(args, name)) for name in ("checkpoint", "normalizer", "model_xml")}
    if any(not path.exists() for path in runtime_paths.values()):
        raise FileNotFoundError(f"Collection input missing: {runtime_paths}")
    collection_config = {name: value for name, value in vars(args).items() if name not in {"reference_file", "activation_observer"}}
    collection_config["seed"] = base_seed
    fingerprint_value = json.dumps(collection_config, sort_keys=True, default=str)
    manifest: dict[str, Any] = {
        "schema_version": 3, "round_name": args.round_name, "episodes": args.episodes,
        "branch_targets": targets, "role_bits": ROLE_BITS, "reference_mode_ids": REFERENCE_MODE_IDS,
        "branch_kind_ids": BRANCH_KIND_IDS,
        "activation_reason_ids": ACTIVATION_REASON_IDS,
        "trajectory_counts": counts, "anticipation_delay_steps": args.anticipation_delay_steps, "schedule": schedule,
        "collection_config": collection_config, "collection_config_sha256": hashlib.sha256(fingerprint_value.encode()).hexdigest(),
        "input_sha256": {name: _sha256(path) for name, path in runtime_paths.items()},
        "timing_json_sha256": None if not args.timing_json else _sha256(RUNNER.resolve_runtime_path(args.timing_json)),
        "completed_shards": [],
    }
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if int(manifest.get("schema_version", 0)) != 3:
            raise ValueError("Existing collection uses the legacy strict-only branch schema; choose a new output directory")
        if manifest.get("collection_config_sha256") != hashlib.sha256(fingerprint_value.encode()).hexdigest():
            raise ValueError("Existing collection manifest has a different configuration; choose a new output directory")
        if manifest.get("input_sha256") != {name: _sha256(path) for name, path in runtime_paths.items()}:
            raise ValueError("Existing collection manifest has different checkpoint/normalizer/model inputs")
    completed = set(int(value) for value in manifest.get("completed_shards", []))
    for shard_start in range(0, args.episodes, args.shard_main_episodes):
        shard_index = shard_start // args.shard_main_episodes
        if shard_index in completed:
            continue
        records: dict[str, list[np.ndarray]] = {key: [] for key in (
        "states actions next_states episode_id split_group_id source_id valid_target context_only branch_id branch_kind_id reference_mode_id".split()
        )}
        for key in ("control_step", "tracking_error", "residual_fraction", "recovery_active", "packet_fallback"):
            records[key] = []
        branch_records: list[dict[str, Any]] = []
        next_branch_id = shard_start * len(targets) * 6
        for main_episode_id in range(shard_start, min(args.episodes, shard_start + args.shard_main_episodes)):
            args.seed = int(base_seed + main_episode_id)
            mode = schedule[main_episode_id]
            args.reference_mode = "task" if mode in {"circle", "figure8"} else mode
            args.reference_file = str(_task_reference(args, mode, main_episode_id, output_dir)) if args.reference_mode == "task" else None
            target_cursor = 0

            def observer(**kwargs: Any) -> None:
                nonlocal target_cursor, next_branch_id
                step = int(kwargs["step"])
                if target_cursor >= len(targets) or step < targets[target_cursor]:
                    return
                # A valid 16-token GRU input is 15 genuine pre-activation
                # tokens plus the first counterfactual token at activation.
                # Do not consume an overdue target until that history exists.
                states_history = np.asarray(kwargs["states_history"], dtype=np.float32)
                command_history = np.asarray(kwargs["command_history"], dtype=np.float32)
                candidates = tuple(kwargs["packet"].branch_candidates)
                if len(states_history) < args.history_len or len(command_history) < args.history_len or not candidates:
                    return
                target_index = target_cursor
                target_step = targets[target_index]
                target_cursor += 1
                full_state = kwargs["env"].capture_full_state()
                candidate_group_id = main_episode_id * len(targets) + target_index

                def append_sidecar(branch: Any, *, branch_id: int, branch_kind: str) -> None:
                    branch_records.append({
                        "branch_id": branch_id, "candidate_group_id": candidate_group_id,
                        "branch_kind_id": BRANCH_KIND_IDS[branch_kind],
                        "parent_main_episode_id": main_episode_id, "activation_step": step, "target_step": target_step,
                        "role_mask": sum(ROLE_BITS[role] for role in branch.role_mask),
                        "activation_infeasible": int(branch.activation_infeasible),
                        "activation_infeasible_reason_id": ACTIVATION_REASON_IDS[branch.infeasible_reason],
                        "predicted_anchor_state": branch.predicted_state_sequence[0],
                        "actual_activation_state": branch.actual_activation_state,
                        "actual_previous_command": branch.actual_previous_command,
                        "actual_previous_velocity": branch.actual_previous_velocity,
                        "planned_q_ref_sequence": branch.planned_q_ref_sequence,
                        "executed_q_ref_sequence": (
                            np.full_like(branch.planned_q_ref_sequence, np.nan)
                            if branch.activation_infeasible else branch.executed_q_ref_sequence
                        ),
                        "predicted_state_sequence": branch.predicted_state_sequence,
                        "realized_state_sequence": (
                            np.full_like(branch.predicted_state_sequence, np.nan)
                            if branch.activation_infeasible else branch.realized_state_sequence
                        ),
                        "predicted_cost": branch.predicted_cost,
                        "realized_cost": branch.realized_cost,
                        "anchor_prediction_error": float(np.linalg.norm(branch.predicted_state_sequence[0] - branch.actual_activation_state)),
                        "q_des_sequence": kwargs["q_des_sequence"], "dq_des_sequence": kwargs["dq_des_sequence"],
                    })

                # Strict exact-action branches retain the original CEM
                # prediction/action semantics.  They are diagnostics only:
                # even feasible strict branches never enter Model-C training.
                for candidate in candidates:
                    strict_id = next_branch_id; next_branch_id += 1
                    branch = execute_exact_action_branch(
                        parent_env=kwargs["env"], full_state=full_state, candidate=candidate,
                        actual_state=kwargs["state"], previous_command=kwargs["previous_command"],
                        previous_velocity=kwargs["previous_velocity"], joint_limit_margin=args.joint_limit_margin,
                        velocity_limit=physical_velocity, acceleration_limit=physical_acceleration,
                        q_des=kwargs["q_des_sequence"], dq_des=kwargs["dq_des_sequence"], cost_config=kwargs["cost_config"],
                    )
                    append_sidecar(branch, branch_id=strict_id, branch_kind="strict_exact_action")

                # Activation-projected branches are the only branch samples
                # used for Model-C training.  Projection starts at the real
                # activation command, then prediction and MuJoCo execution
                # consume exactly the same executable sequence.
                projected_candidates: list[tuple[tuple[str, ...], np.ndarray, np.ndarray]] = []
                for candidate in candidates:
                    projected = project_activation_q_ref_sequence(
                        candidate.q_ref_sequence, kwargs["previous_command"], kwargs["previous_velocity"],
                        kwargs["env"].joint_low, kwargs["env"].joint_high, args.joint_limit_margin,
                        physical_velocity, physical_acceleration, kwargs["env"].control_dt,
                    )
                    for index, (roles, planned, existing) in enumerate(projected_candidates):
                        if np.allclose(projected, existing, atol=1e-6, rtol=0.0):
                            projected_candidates[index] = (tuple(dict.fromkeys((*roles, *candidate.role_mask))), planned, existing)
                            break
                    else:
                        projected_candidates.append((tuple(candidate.role_mask), np.asarray(candidate.q_ref_sequence, dtype=np.float32), projected))

                context_states = states_history[-args.history_len:-1]
                context_actions = command_history[-args.history_len:-1]
                context_next_states = states_history[-(args.history_len - 1):]
                if len(context_states) != 15 or len(context_actions) != 15:
                    return
                bundle = kwargs["dynamics_bundle"]
                history = RUNNER.build_history_tensor(list(states_history), list(command_history), bundle.history_len, bundle.device)
                for roles, planned, projected in projected_candidates:
                    projected_id = next_branch_id; next_branch_id += 1
                    with torch.no_grad():
                        predicted = rollout_dynamics_batch(
                            model=bundle.model, normalizer=bundle.normalizer, model_type=bundle.model_type,
                            initial_history=history,
                            future_q_ref=torch.as_tensor(projected, dtype=torch.float32, device=bundle.device).unsqueeze(0),
                            state_dim=bundle.state_dim, target_mode=bundle.target_mode, control_dt=bundle.control_dt,
                        )[0].detach().cpu().numpy().astype(np.float32)
                    predicted_cost = evaluate_branch_cost(
                        states=predicted, actions=projected, q_des=kwargs["q_des_sequence"], dq_des=kwargs["dq_des_sequence"],
                        previous_command=kwargs["previous_command"], previous_velocity=kwargs["previous_velocity"],
                        parent_env=kwargs["env"], cost_config=kwargs["cost_config"],
                    )
                    branch = execute_open_loop_action_branch(
                        parent_env=kwargs["env"], full_state=full_state, planned_q_ref_sequence=planned,
                        executed_q_ref_sequence=projected, predicted_state_sequence=predicted, predicted_cost=predicted_cost,
                        role_mask=roles, actual_state=kwargs["state"], previous_command=kwargs["previous_command"],
                        previous_velocity=kwargs["previous_velocity"], q_des=kwargs["q_des_sequence"],
                        dq_des=kwargs["dq_des_sequence"], cost_config=kwargs["cost_config"],
                    )
                    append_sidecar(branch, branch_id=projected_id, branch_kind="activation_projected")
                    states = np.concatenate([context_states, branch.realized_state_sequence[:-1]], axis=0)
                    actions = np.concatenate([context_actions, branch.executed_q_ref_sequence], axis=0)
                    next_states = np.concatenate([context_next_states, branch.realized_state_sequence[1:]], axis=0)
                    if len(states) != 40 or len(actions) != 40 or len(next_states) != 40:
                        raise RuntimeError("A Model-C branch must serialize exactly 15 context + 25 branch transitions")
                    for index in range(len(states)):
                        records["states"].append(states[index]); records["actions"].append(actions[index]); records["next_states"].append(next_states[index])
                        records["episode_id"].append(np.asarray(projected_id)); records["split_group_id"].append(np.asarray(main_episode_id))
                        records["source_id"].append(np.asarray(1)); records["valid_target"].append(np.asarray(index >= 15, dtype=np.int8)); records["context_only"].append(np.asarray(index < 15, dtype=np.int8))
                        records["branch_id"].append(np.asarray(projected_id)); records["branch_kind_id"].append(np.asarray(BRANCH_KIND_IDS["activation_projected"])); records["reference_mode_id"].append(np.asarray(REFERENCE_MODE_IDS[mode]))
                        records["control_step"].append(np.asarray(-1)); records["tracking_error"].append(np.asarray(np.nan)); records["residual_fraction"].append(np.asarray(np.nan)); records["recovery_active"].append(np.asarray(0)); records["packet_fallback"].append(np.asarray(0))

            result = RUNNER.run_closed_loop_mpc(args, activation_observer=observer)
            arrays = result["arrays"]
            count = len(arrays["actual_states"])
            for index in range(count):
                records["states"].append(arrays["actual_states"][index]); records["actions"].append(arrays["actuator_q_ref"][index]); records["next_states"].append(arrays["next_states"][index])
                residual = np.asarray(arrays["executed_residual"][index], dtype=np.float32)
                residual_max = np.asarray(arrays["residual_max"], dtype=np.float32)
                event = str(arrays["packet_event"][index])
                records["episode_id"].append(np.asarray(main_episode_id)); records["split_group_id"].append(np.asarray(main_episode_id)); records["source_id"].append(np.asarray(0)); records["valid_target"].append(np.asarray(1)); records["context_only"].append(np.asarray(0)); records["branch_id"].append(np.asarray(-1)); records["branch_kind_id"].append(np.asarray(-1)); records["reference_mode_id"].append(np.asarray(REFERENCE_MODE_IDS[mode])); records["control_step"].append(np.asarray(index)); records["tracking_error"].append(np.asarray(arrays["realized_tracking_error"][index])); records["residual_fraction"].append(np.asarray(np.max(np.abs(residual) / np.maximum(residual_max, 1e-8)))); records["recovery_active"].append(np.asarray(arrays["recovery_active_flags"][index])); records["packet_fallback"].append(np.asarray(int("late_plan_dropped" in event or "planner_failure" in event)))
        _write_shard(output_dir, shard_index, records, branch_records)
        manifest.setdefault("completed_shards", []).append(shard_index)
        _atomic_json(manifest_path, manifest)


if __name__ == "__main__":
    main()
