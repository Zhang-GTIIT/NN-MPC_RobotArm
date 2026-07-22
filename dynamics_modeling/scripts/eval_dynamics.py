from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import torch

from neural_dynamics.mujoco_env import MuJoCoArmEnv
from neural_dynamics.normalization import StandardNormalizer
from neural_dynamics.parallel_collector import (
    MOTION_MODE_NAMES,
    generate_q_ref_sequence,
    parse_action_std,
    reset_safe_workspace,
)
from neural_dynamics.paths import DEFAULT_MODEL_XML, resolve_project_path
from neural_dynamics.train_utils import build_model, load_checkpoint, set_seed
from neural_dynamics.integration import reconstruct_next_state


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate learned dynamics against MuJoCo rollouts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model_xml", default=DEFAULT_MODEL_XML, type=str, help="MuJoCo XML/MJCF model path")
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--normalizer", required=True, type=str)
    parser.add_argument("--model_type", choices=["mlp", "gru", "transformer"], required=True)
    parser.add_argument("--n_joints", default=6, type=int)
    parser.add_argument("--history_len", default=1, type=int)
    parser.add_argument("--rollout_len", default=200, type=int)
    parser.add_argument("--num_rollouts", default=3, type=int)
    parser.add_argument("--save_dir", default="outputs/figures", type=str)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--action_std", default=0.3, type=str)
    parser.add_argument("--settle_steps", default=50, type=int, help="Steps to hold q_ref=q after reset before evaluation")
    parser.add_argument("--warmup_steps", default=0, type=int)
    parser.add_argument("--horizons", default="1,5,10,20,50,200", type=str)
    parser.add_argument("--teacher_forcing", action="store_true")
    return parser.parse_args(argv)


def parse_horizons(value: str) -> list[int]:
    horizons = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not horizons:
        raise ValueError("At least one horizon must be provided")
    if any(horizon <= 0 for horizon in horizons):
        raise ValueError(f"Horizons must be positive integers, got {horizons}")
    return sorted(set(horizons))


def state_labels(n_joints: int) -> list[str]:
    return [f"q{idx}" for idx in range(n_joints)] + [f"dq{idx}" for idx in range(n_joints)]


def per_dimension_rmse(truth: np.ndarray, pred: np.ndarray) -> np.ndarray:
    if truth.shape != pred.shape:
        raise ValueError(f"truth and pred must have same shape, got {truth.shape} and {pred.shape}")
    if truth.ndim != 2:
        raise ValueError(f"truth and pred must be rank-2 arrays, got ndim={truth.ndim}")
    return np.sqrt(np.mean(np.square(truth - pred), axis=0))


def build_sequence_history(entries: list[np.ndarray], current_index: int, history_len: int) -> list[np.ndarray]:
    if history_len <= 0:
        raise ValueError(f"history_len must be positive, got {history_len}")
    if not entries:
        raise ValueError("entries must not be empty")
    if current_index < 0 or current_index >= len(entries):
        raise IndexError(f"current_index={current_index} out of range for entries length={len(entries)}")
    start = max(0, current_index - history_len + 1)
    history = entries[start : current_index + 1]
    if len(history) < history_len:
        history = [history[0]] * (history_len - len(history)) + history
    return history


def resolve_history_len(model_type: str, requested_history_len: int, config: dict) -> int:
    if model_type == "mlp":
        return 1
    if requested_history_len != 1:
        return requested_history_len
    checkpoint_history_len = config.get("history_len")
    if checkpoint_history_len is not None:
        return int(checkpoint_history_len)
    return requested_history_len


def predict_next_state(
    model: torch.nn.Module,
    normalizer: StandardNormalizer,
    model_type: str,
    state: np.ndarray,
    action: np.ndarray,
    history: list[np.ndarray],
    state_dim: int,
    device: torch.device,
    target_mode: str = "delta_state",
    control_dt: float = 0.01,
) -> np.ndarray:
    with torch.no_grad():
        if model_type == "mlp":
            s = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            a = torch.as_tensor(action, dtype=torch.float32, device=device).unsqueeze(0)
            x = normalizer.normalize_single_input(s, a)
        else:
            seq = torch.as_tensor(np.stack(history, axis=0), dtype=torch.float32, device=device).unsqueeze(0)
            x = normalizer.normalize_sequence_input(seq, state_dim)
        pred_delta_norm = model(x)
        pred_delta = normalizer.denormalize_delta(pred_delta_norm).squeeze(0).cpu().numpy()
    state_tensor = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
    target_tensor = torch.as_tensor(pred_delta, dtype=torch.float32).unsqueeze(0)
    next_state = reconstruct_next_state(state_tensor, target_tensor, target_mode, control_dt, state_dim // 2)
    return next_state.squeeze(0).numpy().astype(np.float32)


def plot_rollout(
    truth: np.ndarray,
    pred: np.ndarray,
    n_joints: int,
    rollout_idx: int,
    save_dir: Path,
    prefix: str = "rollout",
    error_title: str = "Open-loop prediction error",
) -> None:
    time = np.arange(truth.shape[0])
    fig_q, axes_q = plt.subplots(n_joints, 1, figsize=(10, 2.0 * n_joints), sharex=True)
    fig_dq, axes_dq = plt.subplots(n_joints, 1, figsize=(10, 2.0 * n_joints), sharex=True)
    if n_joints == 1:
        axes_q = [axes_q]
        axes_dq = [axes_dq]
    for idx in range(n_joints):
        axes_q[idx].plot(time, truth[:, idx], label="mujoco")
        axes_q[idx].plot(time, pred[:, idx], label="learned", linestyle="--")
        axes_q[idx].set_ylabel(f"q{idx}")
        axes_q[idx].legend()
        dq_idx = n_joints + idx
        axes_dq[idx].plot(time, truth[:, dq_idx], label="mujoco")
        axes_dq[idx].plot(time, pred[:, dq_idx], label="learned", linestyle="--")
        axes_dq[idx].set_ylabel(f"dq{idx}")
        axes_dq[idx].legend()
    axes_q[-1].set_xlabel("step")
    axes_dq[-1].set_xlabel("step")
    fig_q.tight_layout()
    fig_dq.tight_layout()
    fig_q.savefig(save_dir / f"{prefix}_{rollout_idx:03d}_q.png", dpi=150)
    fig_dq.savefig(save_dir / f"{prefix}_{rollout_idx:03d}_dq.png", dpi=150)
    plt.close(fig_q)
    plt.close(fig_dq)

    error = np.linalg.norm(truth - pred, axis=1)
    fig_err, ax = plt.subplots(figsize=(10, 4))
    ax.plot(time, error)
    ax.set_xlabel("step")
    ax.set_ylabel("state L2 error")
    ax.set_title(error_title)
    fig_err.tight_layout()
    fig_err.savefig(save_dir / f"{prefix}_{rollout_idx:03d}_error.png", dpi=150)
    plt.close(fig_err)


def plot_torque_components(
    torque_components: dict[str, np.ndarray],
    n_joints: int,
    rollout_idx: int,
    save_dir: Path,
    prefix: str = "rollout",
) -> None:
    time = np.arange(torque_components["total_tau"].shape[0])
    fig, axes = plt.subplots(n_joints, 1, figsize=(10, 2.0 * n_joints), sharex=True)
    if n_joints == 1:
        axes = [axes]
    for idx in range(n_joints):
        axes[idx].plot(time, torque_components["total_tau"][:, idx], label="total_tau")
        axes[idx].plot(time, torque_components["actuator_tau"][:, idx], label="actuator_tau", linestyle="--")
        axes[idx].plot(time, torque_components["gravity_tau"][:, idx], label="gravity_tau", linestyle=":")
        axes[idx].set_ylabel(f"tau{idx} Nm")
        axes[idx].grid(True, alpha=0.3)
        axes[idx].legend(fontsize=8)
    axes[-1].set_xlabel("step")
    fig.tight_layout()
    fig.savefig(save_dir / f"{prefix}_{rollout_idx:03d}_torque.png", dpi=150)
    plt.close(fig)


def write_metric_rows(path: Path, fieldnames: list[str], rows: list[dict[str, float | int | str]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_prediction(
    truth: np.ndarray,
    pred: np.ndarray,
    labels: list[str],
    *,
    joint_low: np.ndarray | None = None,
    joint_high: np.ndarray | None = None,
    velocity_limit: float = 25.0,
) -> dict[str, float]:
    errors = truth - pred
    n_joints = truth.shape[1] // 2
    row: dict[str, float] = {
        "rmse": float(np.sqrt(np.mean(np.square(errors)))),
        "max_l2": float(np.max(np.linalg.norm(errors, axis=1))),
        "q_rmse": float(np.sqrt(np.mean(np.square(errors[:, :n_joints])))),
        "dq_rmse": float(np.sqrt(np.mean(np.square(errors[:, n_joints:])))),
    }
    invalid = ~np.all(np.isfinite(pred), axis=1)
    if joint_low is not None and joint_high is not None:
        low = np.asarray(joint_low, dtype=np.float64).reshape(1, -1)
        high = np.asarray(joint_high, dtype=np.float64).reshape(1, -1)
        invalid |= np.any(pred[:, :n_joints] < low - 0.05, axis=1)
        invalid |= np.any(pred[:, :n_joints] > high + 0.05, axis=1)
    invalid |= np.any(np.abs(pred[:, n_joints:]) > float(velocity_limit), axis=1)
    row["divergence_rate"] = float(np.mean(invalid))
    rmse = per_dimension_rmse(truth, pred)
    mse = np.mean(np.square(errors), axis=0)
    truth_var = np.var(truth, axis=0)
    truth_std = np.std(truth, axis=0)
    pred_std = np.std(pred, axis=0)
    nmse = mse / np.maximum(truth_var, 1e-12)
    amp_ratio = pred_std / np.maximum(truth_std, 1e-12)
    for label, value in zip(labels, rmse):
        row[f"{label}_rmse"] = float(value)
    for label, value in zip(labels, nmse):
        row[f"{label}_nmse"] = float(value)
    for label, value in zip(labels, 1.0 - nmse):
        row[f"{label}_r2"] = float(value)
    for label, value in zip(labels, amp_ratio):
        row[f"{label}_amp_ratio"] = float(value)
    return row


def collect_truth_rollout(
    env: MuJoCoArmEnv,
    rng: np.random.Generator,
    n_joints: int,
    total_steps: int,
    action_std: float | np.ndarray,
    settle_steps: int = 50,
    mode_id: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    true_states, actions, true_next_states, _torque = collect_truth_rollout_with_torque(
        env,
        rng,
        n_joints,
        total_steps,
        action_std,
        settle_steps=settle_steps,
        mode_id=mode_id,
    )
    return true_states, actions, true_next_states


def action_std_array(action_std: float | np.ndarray, n_joints: int) -> np.ndarray:
    if np.isscalar(action_std):
        return np.full(n_joints, float(action_std), dtype=np.float32)
    array = np.asarray(action_std, dtype=np.float32)
    if array.shape != (n_joints,):
        raise ValueError(f"action_std must be scalar or shape ({n_joints},), got {array.shape}")
    return array


def collect_truth_rollout_with_torque(
    env: MuJoCoArmEnv,
    rng: np.random.Generator,
    n_joints: int,
    total_steps: int,
    action_std: float | np.ndarray,
    settle_steps: int = 50,
    mode_id: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    if settle_steps < 0:
        raise ValueError(f"settle_steps must be non-negative, got {settle_steps}")
    true_states: list[np.ndarray] = []
    true_next_states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    torque_records: dict[str, list[np.ndarray]] = {
        "total_tau": [],
        "actuator_tau": [],
        "gravity_tau": [],
    }
    state = reset_safe_workspace(env, rng, n_joints)
    q_ref = np.asarray(state[:n_joints], dtype=np.float32).copy()
    for _ in range(settle_steps):
        state = env.step(q_ref)
    q_ref_sequence = generate_q_ref_sequence(
        rng,
        q_ref,
        env.action_low,
        env.action_high,
        total_steps,
        action_std_array(action_std, n_joints),
        mode_id % len(MOTION_MODE_NAMES),
    )
    for q_ref in q_ref_sequence:
        torque = env.compute_torque_components(q_ref)
        for key in torque_records:
            torque_records[key].append(torque[key])
        actions.append(q_ref.copy())
        true_states.append(state.copy())
        state = env.step(q_ref)
        true_next_states.append(state.copy())
    return (
        np.asarray(true_states, dtype=np.float32),
        np.asarray(actions, dtype=np.float32),
        np.asarray(true_next_states, dtype=np.float32),
        {key: np.asarray(value, dtype=np.float64) for key, value in torque_records.items()},
    )


def predict_open_loop_segment(
    model: torch.nn.Module,
    normalizer: StandardNormalizer,
    model_type: str,
    true_states: np.ndarray,
    actions: np.ndarray,
    start_index: int,
    rollout_len: int,
    history_len: int,
    state_dim: int,
    device: torch.device,
    target_mode: str = "delta_state",
    control_dt: float = 0.01,
    record_next_states: bool = False,
) -> np.ndarray:
    pred_states: list[np.ndarray] = []
    history_entries = [
        np.concatenate([true_states[idx], actions[idx]]).astype(np.float32)
        for idx in range(start_index)
    ]
    pred_state = true_states[start_index].copy()
    for step_idx in range(rollout_len):
        action_index = start_index + step_idx
        action_i = actions[action_index]
        history_entries.append(np.concatenate([pred_state, action_i]).astype(np.float32))
        history = build_sequence_history(history_entries, len(history_entries) - 1, history_len)
        if not record_next_states:
            pred_states.append(pred_state.copy())
        pred_state = predict_next_state(
            model, normalizer, model_type, pred_state, action_i, history, state_dim, device, target_mode, control_dt
        )
        if record_next_states:
            pred_states.append(pred_state.copy())
    return np.asarray(pred_states, dtype=np.float32)


def predict_teacher_forcing(
    model: torch.nn.Module,
    normalizer: StandardNormalizer,
    model_type: str,
    true_states: np.ndarray,
    actions: np.ndarray,
    history_len: int,
    state_dim: int,
    device: torch.device,
    target_mode: str = "delta_state",
    control_dt: float = 0.01,
) -> np.ndarray:
    history_entries = [
        np.concatenate([state, action]).astype(np.float32)
        for state, action in zip(true_states, actions)
    ]
    predictions: list[np.ndarray] = []
    for idx, (state, action) in enumerate(zip(true_states, actions)):
        history = build_sequence_history(history_entries, idx, history_len)
        predictions.append(
            predict_next_state(
                model,
                normalizer,
                model_type,
                state,
                action,
                history,
                state_dim,
                device,
                target_mode,
                control_dt,
            )
        )
    return np.asarray(predictions, dtype=np.float32)


def main() -> None:
    args = parse_args()
    if args.model_type == "mlp":
        args.history_len = 1
    if args.rollout_len <= 0 or args.num_rollouts <= 0:
        raise ValueError("rollout_len and num_rollouts must be positive")
    if args.warmup_steps < 0:
        raise ValueError(f"warmup_steps must be non-negative, got {args.warmup_steps}")
    if args.settle_steps < 0:
        raise ValueError(f"settle_steps must be non-negative, got {args.settle_steps}")
    action_std = parse_action_std(args.action_std, args.n_joints)
    horizons = parse_horizons(args.horizons)
    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dim = 2 * args.n_joints
    action_dim = args.n_joints

    checkpoint = load_checkpoint(Path(args.checkpoint), map_location=device)
    config = checkpoint.get("config", {})
    if config and int(config.get("state_dim", state_dim)) != state_dim:
        raise ValueError(f"Checkpoint state_dim={config.get('state_dim')} does not match n_joints={args.n_joints}")
    target_mode = str(config.get("target_mode", "delta_state")) if isinstance(config, dict) else "delta_state"
    output_dim = int(config.get("output_dim", state_dim)) if isinstance(config, dict) else state_dim
    control_dt = float(config.get("control_dt", 0.01)) if isinstance(config, dict) else 0.01
    args.history_len = resolve_history_len(args.model_type, args.history_len, config if isinstance(config, dict) else {})
    model = build_model(args.model_type, state_dim, action_dim, args.history_len, output_dim=output_dim).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    normalizer = StandardNormalizer.load(Path(args.normalizer), map_location=device)

    env = MuJoCoArmEnv(str(resolve_project_path(args.model_xml, ROOT)), n_joints=args.n_joints, seed=args.seed)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    labels = state_labels(args.n_joints)
    horizon_rows: list[dict[str, float | int | str]] = []
    teacher_rows: list[dict[str, float | int | str]] = []
    try:
        for rollout_idx in range(args.num_rollouts):
            total_steps = args.warmup_steps + max(args.rollout_len, max(horizons))
            true_states_all, actions, true_next_states_all, torque_all = collect_truth_rollout_with_torque(
                env,
                rng,
                args.n_joints,
                total_steps,
                action_std,
                settle_steps=args.settle_steps,
                mode_id=rollout_idx,
            )
            true_states = true_states_all[args.warmup_steps : args.warmup_steps + args.rollout_len]
            np.savez_compressed(
                save_dir / f"evaluation_rollout_{rollout_idx:03d}.npz",
                true_states=true_states_all,
                actions=actions,
                true_next_states=true_next_states_all,
            )
            torque_components = {
                key: value[args.warmup_steps : args.warmup_steps + args.rollout_len]
                for key, value in torque_all.items()
            }
            pred_states = predict_open_loop_segment(
                model,
                normalizer,
                args.model_type,
                true_states_all,
                actions,
                args.warmup_steps,
                args.rollout_len,
                args.history_len,
                state_dim,
                device,
                target_mode,
                control_dt,
            )

            plot_rollout(
                true_states,
                pred_states,
                args.n_joints,
                rollout_idx,
                save_dir,
            )
            plot_torque_components(torque_components, args.n_joints, rollout_idx, save_dir)
            print(f"saved rollout figures for rollout {rollout_idx} to {save_dir}")

            for horizon in horizons:
                horizon_truth = true_next_states_all[args.warmup_steps : args.warmup_steps + horizon]
                horizon_pred = predict_open_loop_segment(
                    model,
                    normalizer,
                    args.model_type,
                    true_states_all,
                    actions,
                    args.warmup_steps,
                    horizon,
                    args.history_len,
                    state_dim,
                    device,
                    target_mode,
                    control_dt,
                    record_next_states=True,
                )
                horizon_rows.append(
                    {
                        "rollout": rollout_idx,
                        "mode": "open_loop",
                        "horizon": horizon,
                        "action_std": args.action_std,
                        "warmup_steps": args.warmup_steps,
                        **summarize_prediction(
                            horizon_truth, horizon_pred, labels,
                            joint_low=env.joint_low, joint_high=env.joint_high,
                        ),
                    }
                )

            if args.teacher_forcing:
                teacher_pred_next = predict_teacher_forcing(
                    model,
                    normalizer,
                    args.model_type,
                    true_states_all,
                    actions,
                    args.history_len,
                    state_dim,
                    device,
                    target_mode,
                    control_dt,
                )
                teacher_truth_next = true_next_states_all[args.warmup_steps : args.warmup_steps + args.rollout_len]
                teacher_plot_pred = teacher_pred_next[args.warmup_steps : args.warmup_steps + args.rollout_len]
                plot_rollout(
                    teacher_truth_next,
                    teacher_plot_pred,
                    args.n_joints,
                    rollout_idx,
                    save_dir,
                    prefix="teacher_forcing",
                    error_title="Teacher-forcing one-step prediction error",
                )
                teacher_rows.append(
                    {
                        "rollout": rollout_idx,
                        "mode": "teacher_forcing",
                        "horizon": 1,
                        "action_std": args.action_std,
                        "warmup_steps": args.warmup_steps,
                        **summarize_prediction(
                            true_next_states_all, teacher_pred_next, labels,
                            joint_low=env.joint_low, joint_high=env.joint_high,
                        ),
                    }
                )
    finally:
        env.close()

    metric_fieldnames = [
        "rollout",
        "mode",
        "horizon",
        "action_std",
        "warmup_steps",
        "rmse",
        "max_l2",
        "q_rmse",
        "dq_rmse",
        "divergence_rate",
        *[f"{label}_rmse" for label in labels],
        *[f"{label}_nmse" for label in labels],
        *[f"{label}_r2" for label in labels],
        *[f"{label}_amp_ratio" for label in labels],
    ]
    write_metric_rows(save_dir / "horizon_metrics.csv", metric_fieldnames, horizon_rows)
    write_metric_rows(save_dir / "teacher_forcing_metrics.csv", metric_fieldnames, teacher_rows)


if __name__ == "__main__":
    main()
