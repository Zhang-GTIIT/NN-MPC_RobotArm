from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DYNAMICS_ROOT = ROOT / "learned_mujoco_dynamics"
for path in (ROOT, DYNAMICS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import numpy as np
import torch

from learned_dynamics.mujoco_env import MuJoCoArmEnv
from learned_dynamics.parallel_collector import reset_safe_workspace
from learned_dynamics.paths import DEFAULT_MODEL_XML
from learned_dynamics.rollout import load_dynamics_bundle
from learned_dynamics.train_utils import set_seed
from mpc.cem_controller import CEMMPCConfig, CEMMPCController
from mpc.cost_functions import JointSpaceCostConfig
from mpc.logging import save_mpc_run
from mpc.planner_rollout import LearnedDynamicsPlanner, PlannerRolloutConfig
from mpc.reference import finite_difference_dq, generate_joint_reference
from mpc.utils import build_history_tensor


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


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run learned CEM-MPC in closed-loop MuJoCo simulation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model_xml", default=DEFAULT_MODEL_XML, type=str)
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--normalizer", required=True, type=str)
    parser.add_argument("--model_type", choices=["mlp", "gru", "transformer"], default="transformer")
    parser.add_argument("--history_len", default=None, type=int)
    parser.add_argument("--n_joints", default=6, type=int)
    parser.add_argument("--device", default="auto", type=str)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--episode_len", default=200, type=int)
    parser.add_argument("--settle_steps", default=50, type=int)
    parser.add_argument("--reference_mode", choices=["hold", "step", "joint_sine", "multi_joint_sine"], default="multi_joint_sine")
    parser.add_argument("--reference_amplitude", default=0.15, type=float)
    parser.add_argument("--save_dir", default="outputs/mpc/cem_run", type=str)
    parser.add_argument("--fail_on_limit_violation", action="store_true")

    parser.add_argument("--horizon", default=20, type=int)
    parser.add_argument("--num_samples", default=1024, type=int)
    parser.add_argument("--num_elites", default=None, type=int)
    parser.add_argument("--elite_ratio", default=0.08, type=float)
    parser.add_argument("--cem_iters", default=4, type=int)
    parser.add_argument("--init_std", default=0.12, type=float)
    parser.add_argument("--min_std", default=0.01, type=float)
    parser.add_argument("--smoothing_alpha", default=0.2, type=float)
    parser.add_argument("--temporal_noise_alpha", default=0.8, type=float)
    parser.add_argument("--rollout_batch_size", default=256, type=int)

    parser.add_argument("--ref_mode", choices=["delta", "absolute"], default="delta")
    parser.add_argument("--delta_base", choices=["previous_q_ref", "current_q"], default="previous_q_ref")
    parser.add_argument("--delta_q_ref_max", default=0.08, type=float)
    parser.add_argument("--q_ref_rate_limit", default=0.08, type=float)
    parser.add_argument("--delta_rate_limit", default=None, type=float)
    parser.add_argument("--joint_limit_margin", default=0.02, type=float)

    parser.add_argument("--w_q", default=1.0, type=float)
    parser.add_argument("--w_dq", default=0.01, type=float)
    parser.add_argument("--w_u", default=0.001, type=float)
    parser.add_argument("--w_du", default=0.001, type=float)
    parser.add_argument("--w_terminal", default=1.0, type=float)
    parser.add_argument("--w_joint_limit", default=10.0, type=float)
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_arg_parser().parse_args(argv)


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def _stack_records(records: list[np.ndarray], dtype: np.dtype = np.float32) -> np.ndarray:
    if not records:
        return np.empty((0,), dtype=dtype)
    return np.asarray(records, dtype=dtype)


def run_closed_loop_mpc(args: argparse.Namespace) -> dict[str, Any]:
    if args.episode_len <= 0:
        raise ValueError(f"episode_len must be positive, got {args.episode_len}")
    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = resolve_device(args.device)
    bundle = load_dynamics_bundle(
        checkpoint_path=resolve_runtime_path(args.checkpoint),
        normalizer_path=resolve_runtime_path(args.normalizer),
        model_type=args.model_type,
        n_joints=args.n_joints,
        device=device,
        history_len=args.history_len,
    )
    env = MuJoCoArmEnv(str(resolve_runtime_path(args.model_xml)), n_joints=args.n_joints, seed=args.seed)

    states_history: list[np.ndarray] = []
    q_ref_history: list[np.ndarray] = []
    actual_states: list[np.ndarray] = []
    next_states: list[np.ndarray] = []
    selected_q_refs: list[np.ndarray] = []
    selected_delta_q_refs: list[np.ndarray] = []
    q_des_records: list[np.ndarray] = []
    dq_des_records: list[np.ndarray] = []
    planning_times: list[float] = []
    best_costs: list[float] = []
    elite_costs: list[float] = []
    failures: list[int] = []
    failure_reasons: list[str] = []
    joint_limit_violations: list[int] = []
    realized_tracking_errors: list[float] = []
    predicted_real_error_gaps: list[float] = []
    torque_records: dict[str, list[np.ndarray]] = {"tau_actuator": [], "tau_gravity": [], "tau_total": []}
    rows: list[dict[str, Any]] = []

    try:
        state = reset_safe_workspace(env, rng, args.n_joints)
        previous_q_ref = np.asarray(state[: args.n_joints], dtype=np.float32).copy()
        for _ in range(args.settle_steps):
            state = env.step(previous_q_ref)
        states_history.append(state.copy())
        q_ref_history.append(previous_q_ref.copy())

        reference = generate_joint_reference(
            args.reference_mode,
            state[: args.n_joints],
            env.joint_low + args.joint_limit_margin,
            env.joint_high - args.joint_limit_margin,
            args.episode_len + args.horizon + 1,
            bundle.control_dt,
            seed=args.seed,
            amplitude=args.reference_amplitude,
        )
        dq_reference = finite_difference_dq(reference, bundle.control_dt)
        cost_config = JointSpaceCostConfig(
            w_q=args.w_q,
            w_dq=args.w_dq,
            w_u=args.w_u,
            w_du=args.w_du,
            w_terminal=args.w_terminal,
            w_joint_limit=args.w_joint_limit,
        )
        rollout_config = PlannerRolloutConfig(
            mode=args.ref_mode,
            delta_base=args.delta_base,
            delta_q_ref_max=args.delta_q_ref_max,
            q_ref_rate_limit=args.q_ref_rate_limit,
            delta_rate_limit=args.delta_rate_limit,
            joint_limit_margin=args.joint_limit_margin,
            rollout_batch_size=args.rollout_batch_size,
        )
        controller: CEMMPCController | None = None

        for step_idx in range(args.episode_len):
            initial_history = build_history_tensor(states_history, q_ref_history, bundle.history_len, device)
            planner = LearnedDynamicsPlanner(
                model=bundle.model,
                normalizer=bundle.normalizer,
                model_type=bundle.model_type,
                state_dim=bundle.state_dim,
                target_mode=bundle.target_mode,
                control_dt=bundle.control_dt,
                initial_history=initial_history,
                q_des=torch.as_tensor(reference[step_idx + 1 : step_idx + 1 + args.horizon], dtype=torch.float32, device=device),
                dq_des=torch.as_tensor(dq_reference[step_idx + 1 : step_idx + 1 + args.horizon], dtype=torch.float32, device=device),
                previous_q_ref=torch.as_tensor(previous_q_ref, dtype=torch.float32, device=device),
                joint_low=torch.as_tensor(env.joint_low, dtype=torch.float32, device=device),
                joint_high=torch.as_tensor(env.joint_high, dtype=torch.float32, device=device),
                cost_config=cost_config,
                rollout_config=rollout_config,
            )
            if controller is None:
                controller = CEMMPCController(
                    config=CEMMPCConfig(
                        horizon=args.horizon,
                        action_dim=args.n_joints,
                        num_samples=args.num_samples,
                        num_elites=args.num_elites,
                        elite_ratio=args.elite_ratio,
                        cem_iters=args.cem_iters,
                        init_std=args.init_std,
                        min_std=args.min_std,
                        smoothing_alpha=args.smoothing_alpha,
                        temporal_noise_alpha=args.temporal_noise_alpha,
                        seed=args.seed,
                        device=str(device),
                    ),
                    planner=planner,
                    joint_low=env.joint_low,
                    joint_high=env.joint_high,
                )
            else:
                controller.planner = planner

            result = controller.plan(current_state=state, previous_q_ref=previous_q_ref)
            q_ref_command = result.q_ref.astype(np.float32)
            torque = env.compute_torque_components(q_ref_command)
            actual_states.append(state.copy())
            q_des_records.append(reference[step_idx].copy())
            dq_des_records.append(dq_reference[step_idx].copy())
            selected_q_refs.append(q_ref_command.copy())
            selected_delta_q_refs.append(result.delta_q_ref.copy())
            planning_times.append(result.planning_time)
            best_costs.append(result.best_cost)
            elite_costs.append(result.elite_mean_cost)
            failures.append(int(result.failure))
            failure_reasons.append(result.failure_reason)
            for key, target_key in (("actuator_tau", "tau_actuator"), ("gravity_tau", "tau_gravity"), ("total_tau", "tau_total")):
                torque_records[target_key].append(torque[key].astype(np.float32))

            try:
                state = env.step(q_ref_command)
                joint_limit_violations.append(0)
            except RuntimeError:
                joint_limit_violations.append(1)
                if args.fail_on_limit_violation:
                    raise
                break

            next_states.append(state.copy())
            previous_q_ref = q_ref_command.copy()
            states_history.append(state.copy())
            q_ref_history.append(previous_q_ref.copy())
            realized_error = float(np.linalg.norm(state[: args.n_joints] - reference[step_idx + 1]))
            realized_tracking_errors.append(realized_error)
            predicted_real_error_gaps.append(float(result.best_cost - realized_error))
            rows.append(
                {
                    "step": step_idx,
                    "tracking_error": realized_error,
                    "planning_time": result.planning_time,
                    "best_cost": result.best_cost,
                    "elite_mean_cost": result.elite_mean_cost,
                    "failure": int(result.failure),
                    "failure_reason": result.failure_reason,
                    "joint_limit_violation": joint_limit_violations[-1],
                    "predicted_real_error_gap": predicted_real_error_gaps[-1],
                }
            )
    finally:
        env.close()

    arrays: dict[str, np.ndarray] = {
        "actual_states": _stack_records(actual_states),
        "next_states": _stack_records(next_states),
        "q_des": _stack_records(q_des_records),
        "dq_des": _stack_records(dq_des_records),
        "actuator_q_ref": _stack_records(selected_q_refs),
        "delta_q_ref": _stack_records(selected_delta_q_refs),
        "planning_time": np.asarray(planning_times, dtype=np.float32),
        "best_cost": np.asarray(best_costs, dtype=np.float32),
        "elite_mean_cost": np.asarray(elite_costs, dtype=np.float32),
        "failure_flags": np.asarray(failures, dtype=np.int64),
        "joint_limit_violation_flags": np.asarray(joint_limit_violations, dtype=np.int64),
        "realized_tracking_error": np.asarray(realized_tracking_errors, dtype=np.float32),
        "predicted_real_error_gap": np.asarray(predicted_real_error_gaps, dtype=np.float32),
        **{key: _stack_records(value) for key, value in torque_records.items()},
    }
    return {"arrays": arrays, "rows": rows, "failure_reasons": failure_reasons}


def main() -> None:
    args = parse_args()
    result = run_closed_loop_mpc(args)
    save_dir = resolve_runtime_path(args.save_dir)
    save_mpc_run(save_dir, result["arrays"], result["rows"])
    print(f"Saved CEM-MPC rollout to {save_dir}")


if __name__ == "__main__":
    main()
