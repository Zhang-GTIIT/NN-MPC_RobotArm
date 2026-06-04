from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from learned_dynamics.dataset import load_npz_dataset, load_rollout_npz_dataset, split_dataset
from learned_dynamics.integration import reconstruct_next_state
from learned_dynamics.normalization import StandardNormalizer
from learned_dynamics.train_utils import (
    build_model,
    load_checkpoint,
    require_resume_checkpoint,
    save_checkpoint,
    save_yaml,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train learned MuJoCo dynamics.")
    parser.add_argument("--data_path", required=True, type=str)
    parser.add_argument("--model_type", choices=["mlp", "gru", "transformer"], default="transformer", type=str)
    parser.add_argument("--history_len", default=1, type=int)
    parser.add_argument("--batch_size", default=1024, type=int)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--save_dir", default="outputs/checkpoints", type=str)
    parser.add_argument("--seed", default=10, type=int)
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--resume_checkpoint", default=None, type=str)
    parser.add_argument("--init_from_checkpoint", default=None, type=str)
    parser.add_argument("--q_weight", default=1.0, type=float)
    parser.add_argument("--dq_weight", default=1.0, type=float)
    parser.add_argument("--q_extra_weights", default=None, type=str)
    parser.add_argument("--dq_extra_weights", default=None, type=str)
    parser.add_argument("--target_mode", choices=["delta_state", "delta_dq"], default="delta_state", type=str)
    parser.add_argument("--control_dt", default=0.01, type=float)
    parser.add_argument("--loss_type", choices=["mse", "huber"], default="mse", type=str)
    parser.add_argument("--huber_delta", default=1.0, type=float)
    parser.add_argument("--rollout_loss_steps", default=1, type=int)
    parser.add_argument("--rollout_loss_weight", default=0.0, type=float)
    parser.add_argument("--rollout_loss_discount", default=1.0, type=float)
    parser.add_argument("--train_sample_stride", default=1, type=int)
    parser.add_argument("--val_sample_stride", default=1, type=int)
    return parser.parse_args()


def validate_checkpoint_config(checkpoint: dict, expected: dict) -> None:
    config = checkpoint.get("config", {})
    if not isinstance(config, dict):
        raise ValueError("Checkpoint config must be a mapping")
    for key in ("model_type", "history_len", "state_dim", "action_dim"):
        if key not in config:
            continue
        if config[key] != expected[key]:
            raise ValueError(
                f"Checkpoint {key}={config[key]!r} does not match current {key}={expected[key]!r}"
            )


def checkpoint_metadata(
    checkpoint_type: str,
    epoch: int,
    train_loss: float,
    val_loss: float,
    best_val: float,
    args: argparse.Namespace,
    best_rollout: float | None = None,
) -> dict:
    return {
        "checkpoint_type": checkpoint_type,
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "best_val": best_val,
        "best_rollout": best_rollout,
        "init_from_checkpoint": args.init_from_checkpoint,
        "resume_checkpoint": args.resume_checkpoint,
    }


def normalize_batch(
    x: torch.Tensor,
    y: torch.Tensor,
    normalizer: StandardNormalizer,
    state_dim: int,
    model_type: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if model_type == "mlp":
        x_norm = normalizer.normalize_single_input(x[:, :state_dim], x[:, state_dim:])
    else:
        x_norm = normalizer.normalize_sequence_input(x, state_dim)
    y_norm = normalizer.normalize_delta(y)
    return x_norm, y_norm


def parse_extra_weights(value: str | None, n_joints: int, name: str) -> torch.Tensor | None:
    if value is None:
        return None
    weights = [float(item.strip()) for item in value.split(",") if item.strip()]
    if len(weights) != n_joints:
        raise ValueError(f"Expected {n_joints} values for {name}, got {len(weights)}: {value!r}")
    if any(weight < 0 for weight in weights):
        raise ValueError(f"{name} values must be non-negative, got {value!r}")
    return torch.tensor(weights, dtype=torch.float32)


def weighted_delta_loss(
    pred_norm: torch.Tensor,
    target_norm: torch.Tensor,
    n_joints: int,
    q_weight: float,
    dq_weight: float,
    q_extra_weights: torch.Tensor | None,
    dq_extra_weights: torch.Tensor | None,
    loss_type: str = "mse",
    huber_delta: float = 1.0,
) -> torch.Tensor:
    if pred_norm.shape != target_norm.shape:
        raise ValueError(f"pred_norm and target_norm must have same shape, got {pred_norm.shape} and {target_norm.shape}")
    if pred_norm.ndim != 2:
        raise ValueError(f"pred_norm and target_norm must be rank-2 tensors, got ndim={pred_norm.ndim}")
    if loss_type not in {"mse", "huber"}:
        raise ValueError(f"loss_type must be 'mse' or 'huber', got {loss_type!r}")
    if pred_norm.shape[1] not in {n_joints, 2 * n_joints}:
        raise ValueError(f"Expected target dimension {n_joints} or {2 * n_joints}, got {pred_norm.shape[1]}")

    def elementwise_loss(error: torch.Tensor) -> torch.Tensor:
        if loss_type == "mse":
            return torch.square(error)
        abs_error = torch.abs(error)
        quadratic = torch.minimum(abs_error, torch.as_tensor(huber_delta, device=error.device, dtype=error.dtype))
        linear = abs_error - quadratic
        return 0.5 * torch.square(quadratic) + huber_delta * linear

    if pred_norm.shape[1] == n_joints:
        dq_error = elementwise_loss(pred_norm - target_norm)
        if dq_extra_weights is not None:
            dq_error = dq_error * dq_extra_weights.to(device=pred_norm.device, dtype=pred_norm.dtype)
        return dq_weight * torch.sum(dq_error)

    q_error = elementwise_loss(pred_norm[:, :n_joints] - target_norm[:, :n_joints])
    dq_error = elementwise_loss(pred_norm[:, n_joints:] - target_norm[:, n_joints:])
    if q_extra_weights is not None:
        q_error = q_error * q_extra_weights.to(device=pred_norm.device, dtype=pred_norm.dtype)
    if dq_extra_weights is not None:
        dq_error = dq_error * dq_extra_weights.to(device=pred_norm.device, dtype=pred_norm.dtype)
    q_loss = torch.sum(q_error)
    dq_loss = torch.sum(dq_error)
    return q_weight * q_loss + dq_weight * dq_loss


def rollout_state_loss(
    pred_states: torch.Tensor,
    true_states: torch.Tensor,
    normalizer: StandardNormalizer,
    discount: float = 1.0,
) -> torch.Tensor:
    if pred_states.shape != true_states.shape:
        raise ValueError(f"pred_states and true_states must have same shape, got {pred_states.shape} and {true_states.shape}")
    if pred_states.ndim != 3:
        raise ValueError(f"pred_states and true_states must be rank-3 tensors, got ndim={pred_states.ndim}")
    if discount <= 0:
        raise ValueError(f"discount must be positive, got {discount}")
    pred_norm = normalizer.normalize_state(pred_states)
    true_norm = normalizer.normalize_state(true_states)
    step_error = torch.sum(torch.square(pred_norm - true_norm), dim=-1)
    weights = torch.pow(
        torch.as_tensor(discount, device=pred_states.device, dtype=pred_states.dtype),
        torch.arange(pred_states.shape[1], device=pred_states.device, dtype=pred_states.dtype),
    )
    return torch.sum(step_error * weights.unsqueeze(0))


def predict_rollout_states(
    model: nn.Module,
    normalizer: StandardNormalizer,
    model_type: str,
    initial_input: torch.Tensor,
    rollout_actions: torch.Tensor,
    state_dim: int,
    target_mode: str,
    control_dt: float,
) -> torch.Tensor:
    if rollout_actions.ndim != 3:
        raise ValueError(f"rollout_actions must have shape [batch, steps, action_dim], got {rollout_actions.shape}")
    if model_type == "mlp":
        pred_state = initial_input[:, :state_dim]
        history = None
    else:
        pred_state = initial_input[:, -1, :state_dim]
        history = initial_input
    pred_states: list[torch.Tensor] = []
    for step_idx in range(rollout_actions.shape[1]):
        action_i = rollout_actions[:, step_idx]
        if model_type == "mlp":
            model_input = normalizer.normalize_single_input(pred_state, action_i)
        else:
            if history is None:
                raise RuntimeError("Sequence model rollout expected history tensor")
            model_input = normalizer.normalize_sequence_input(history, state_dim)
        pred_target = normalizer.denormalize_delta(model(model_input))
        pred_state = reconstruct_next_state(
            pred_state,
            pred_target,
            target_mode,
            control_dt,
            state_dim // 2,
        )
        pred_states.append(pred_state)
        if model_type != "mlp" and step_idx + 1 < rollout_actions.shape[1]:
            next_action = rollout_actions[:, step_idx + 1]
            next_entry = torch.cat([pred_state, next_action], dim=-1).unsqueeze(1)
            history = torch.cat([history[:, 1:], next_entry], dim=1)
    return torch.stack(pred_states, dim=1)


def format_epoch_header(epoch: int, train_loss: float, val_loss: float, is_best: bool) -> str:
    best_marker = "  ★ best" if is_best else ""
    return f"\n[Epoch {epoch:04d}]  train_loss={train_loss:.6f}  val_loss={val_loss:.6f}{best_marker}"


def format_delta_rmse_table(train_rmse: torch.Tensor, val_rmse: torch.Tensor, n_joints: int) -> str:
    t = train_rmse.detach().cpu().tolist()
    v = val_rmse.detach().cpu().tolist()
    lines = [
        f"{'Joint':>5} | {'train_q':>10} {'train_dq':>10} | {'val_q':>10} {'val_dq':>10}",
        f"{'-' * 5}-+-{'-' * 10}-{'-' * 10}-+-{'-' * 10}-{'-' * 10}",
    ]
    for i in range(n_joints):
        lines.append(
            f"{i:>5} | {t[i]:>10.6f} {t[n_joints + i]:>10.6f} | {v[i]:>10.6f} {v[n_joints + i]:>10.6f}"
        )
    return "\n".join(lines)


def format_delta_rmse(name: str, rmse: torch.Tensor, n_joints: int) -> str:
    values = rmse.detach().cpu().tolist()
    labels = [f"q{i}" for i in range(n_joints)] + [f"dq{i}" for i in range(n_joints)]
    parts = [f"{name}_{label}_rmse={value:.6f}" for label, value in zip(labels, values)]
    return " ".join(parts)


def fit_normalizer_with_progress(
    normalizer: StandardNormalizer,
    states: torch.Tensor,
    actions: torch.Tensor,
    deltas: torch.Tensor,
) -> None:
    with tqdm(total=6, desc="fit normalizer", unit="stat") as progress:
        normalizer.state_mean = states.mean(dim=0)
        progress.update()
        normalizer.state_std = states.std(dim=0, unbiased=False).clamp_min(normalizer.eps)
        progress.update()
        normalizer.action_mean = actions.mean(dim=0)
        progress.update()
        normalizer.action_std = actions.std(dim=0, unbiased=False).clamp_min(normalizer.eps)
        progress.update()
        normalizer.delta_mean = deltas.mean(dim=0)
        progress.update()
        normalizer.delta_std = deltas.std(dim=0, unbiased=False).clamp_min(normalizer.eps)
        progress.update()


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    normalizer: StandardNormalizer,
    state_dim: int,
    model_type: str,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    use_amp: bool = False,
    q_weight: float = 1.0,
    dq_weight: float = 1.0,
    q_extra_weights: torch.Tensor | None = None,
    dq_extra_weights: torch.Tensor | None = None,
    target_mode: str = "delta_state",
    control_dt: float = 0.01,
    loss_type: str = "mse",
    huber_delta: float = 1.0,
    rollout_loss_weight: float = 0.0,
    rollout_loss_discount: float = 1.0,
) -> tuple[float, torch.Tensor, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_rollout_loss = 0.0
    total_samples = 0
    squared_error_sum = torch.zeros(state_dim, dtype=torch.float64)
    progress = tqdm(loader, desc="train" if training else "val", leave=False)
    for batch in progress:
        if len(batch) == 2:
            x, y = batch
            rollout_actions = None
            rollout_next_states = None
        elif len(batch) == 4:
            x, y, rollout_actions, rollout_next_states = batch
        else:
            raise ValueError(f"Expected batch with 2 or 4 tensors, got {len(batch)}")
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        if rollout_actions is not None:
            rollout_actions = rollout_actions.to(device, non_blocking=True)
        if rollout_next_states is not None:
            rollout_next_states = rollout_next_states.to(device, non_blocking=True)
        x_norm, y_norm = normalize_batch(x, y, normalizer, state_dim, model_type)

        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                pred = model(x_norm)
                one_step_loss = weighted_delta_loss(
                    pred,
                    y_norm,
                    state_dim // 2,
                    q_weight,
                    dq_weight,
                    q_extra_weights,
                    dq_extra_weights,
                    loss_type,
                    huber_delta,
                )
                rollout_loss_value = torch.zeros((), dtype=one_step_loss.dtype, device=one_step_loss.device)
                if (
                    rollout_loss_weight > 0.0
                    and rollout_actions is not None
                    and rollout_next_states is not None
                    and rollout_actions.shape[1] > 1
                ):
                    rollout_pred_states = predict_rollout_states(
                        model,
                        normalizer,
                        model_type,
                        x,
                        rollout_actions,
                        state_dim,
                        target_mode,
                        control_dt,
                    )
                    rollout_loss_value = rollout_state_loss(
                        rollout_pred_states,
                        rollout_next_states,
                        normalizer,
                        rollout_loss_discount,
                    )
                loss = one_step_loss + rollout_loss_weight * rollout_loss_value
            if training:
                if scaler is not None and use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        pred_target = normalizer.denormalize_delta(pred.detach())
        current_state = x[:, :state_dim] if model_type == "mlp" else x[:, -1, :state_dim]
        pred_next = reconstruct_next_state(current_state, pred_target, target_mode, control_dt, state_dim // 2)
        if target_mode == "delta_state":
            true_next = current_state + y
        else:
            true_next = reconstruct_next_state(current_state, y, target_mode, control_dt, state_dim // 2)
        batch_squared_error = torch.square(pred_next - true_next).sum(dim=0).detach().cpu().double()
        squared_error_sum += batch_squared_error
        batch_samples = int(y.shape[0])
        total_loss += float(loss.detach().cpu())
        total_rollout_loss += float(rollout_loss_value.detach().cpu())
        total_samples += batch_samples
        progress.set_postfix(loss=total_loss / max(total_samples, 1))
    rmse = torch.sqrt(squared_error_sum / max(total_samples, 1)).float()
    return total_loss / max(total_samples, 1), rmse, total_rollout_loss / max(total_samples, 1)


def main() -> None:
    args = parse_args()
    if args.resume_checkpoint and args.init_from_checkpoint:
        raise ValueError("--resume_checkpoint and --init_from_checkpoint cannot be used together")
    if args.model_type == "mlp":
        args.history_len = 1
    if args.history_len <= 0:
        raise ValueError(f"history_len must be positive, got {args.history_len}")
    if args.epochs <= 0:
        raise ValueError(f"epochs must be positive, got {args.epochs}")
    if args.control_dt <= 0:
        raise ValueError(f"control_dt must be positive, got {args.control_dt}")
    if args.huber_delta <= 0:
        raise ValueError(f"huber_delta must be positive, got {args.huber_delta}")
    if args.rollout_loss_steps <= 0:
        raise ValueError(f"rollout_loss_steps must be positive, got {args.rollout_loss_steps}")
    if args.rollout_loss_weight < 0:
        raise ValueError(f"rollout_loss_weight must be non-negative, got {args.rollout_loss_weight}")
    if args.rollout_loss_discount <= 0:
        raise ValueError(f"rollout_loss_discount must be positive, got {args.rollout_loss_discount}")
    if args.train_sample_stride <= 0:
        raise ValueError(f"train_sample_stride must be positive, got {args.train_sample_stride}")
    if args.val_sample_stride <= 0:
        raise ValueError(f"val_sample_stride must be positive, got {args.val_sample_stride}")
    if args.q_weight < 0 or args.dq_weight < 0:
        raise ValueError(f"q_weight and dq_weight must be non-negative, got {args.q_weight}, {args.dq_weight}")
    if args.q_weight == 0 and args.dq_weight == 0:
        raise ValueError("At least one of q_weight or dq_weight must be positive")
    use_rollout_loss = bool(args.rollout_loss_steps > 1 and args.rollout_loss_weight > 0.0)

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}", flush=True)
    if use_rollout_loss:
        dataset = load_rollout_npz_dataset(
            Path(args.data_path),
            args.model_type,
            args.history_len,
            target_mode=args.target_mode,
            rollout_steps=args.rollout_loss_steps,
        )
    else:
        dataset = load_npz_dataset(Path(args.data_path), args.model_type, args.history_len, target_mode=args.target_mode)
    q_extra_weights = parse_extra_weights(args.q_extra_weights, dataset.state_dim // 2, "q_extra_weights")
    dq_extra_weights = parse_extra_weights(args.dq_extra_weights, dataset.state_dim // 2, "dq_extra_weights")
    print(
        f"loaded dataset: samples={len(dataset)} state_dim={dataset.state_dim} "
        f"action_dim={dataset.action_dim} history_len={args.history_len}",
        flush=True,
    )
    if q_extra_weights is not None:
        print(f"q_extra_weights={q_extra_weights.tolist()}", flush=True)
    if dq_extra_weights is not None:
        print(f"dq_extra_weights={dq_extra_weights.tolist()}", flush=True)
    train_set, val_set = split_dataset(
        dataset,
        val_fraction=0.1,
        seed=args.seed,
        train_sample_stride=args.train_sample_stride,
        val_sample_stride=args.val_sample_stride,
        show_progress=True,
    )
    print(
        f"split dataset: train={len(train_set)} val={len(val_set)} "
        f"train_sample_stride={args.train_sample_stride} val_sample_stride={args.val_sample_stride}",
        flush=True,
    )

    print("fitting normalizer", flush=True)
    deltas = dataset.next_states - dataset.states
    if args.target_mode == "delta_dq":
        deltas = deltas[:, dataset.state_dim // 2 :]
    normalizer = StandardNormalizer()
    fit_normalizer_with_progress(normalizer, dataset.states, dataset.actions, deltas)
    del deltas
    print("normalizer ready", flush=True)

    model = build_model(
        args.model_type,
        dataset.state_dim,
        dataset.action_dim,
        args.history_len,
        output_dim=dataset.target_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    use_amp = bool(args.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    config = {
        "model_type": args.model_type,
        "state_dim": dataset.state_dim,
        "action_dim": dataset.action_dim,
        "output_dim": dataset.target_dim,
        "history_len": args.history_len,
        "target_mode": args.target_mode,
        "control_dt": args.control_dt,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "seed": args.seed,
        "q_weight": args.q_weight,
        "dq_weight": args.dq_weight,
        "q_extra_weights": None if q_extra_weights is None else q_extra_weights.tolist(),
        "dq_extra_weights": None if dq_extra_weights is None else dq_extra_weights.tolist(),
        "loss_type": args.loss_type,
        "huber_delta": args.huber_delta,
        "rollout_loss_steps": args.rollout_loss_steps,
        "rollout_loss_weight": args.rollout_loss_weight,
        "rollout_loss_discount": args.rollout_loss_discount,
        "train_sample_stride": args.train_sample_stride,
        "val_sample_stride": args.val_sample_stride,
    }

    start_epoch = 1
    best_val = float("inf")
    best_rollout = float("inf")

    if args.resume_checkpoint:
        checkpoint = require_resume_checkpoint(load_checkpoint(Path(args.resume_checkpoint), map_location=device))
        validate_checkpoint_config(checkpoint, config)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        for group in optimizer.param_groups:
            group["lr"] = args.lr
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        metadata = checkpoint["metadata"]
        start_epoch = int(metadata["epoch"]) + 1
        best_val = float(metadata["best_val"])
        best_rollout = float(metadata.get("best_rollout", float("inf")))
        save_dir = Path(args.resume_checkpoint).expanduser().resolve().parent
    else:
        if args.init_from_checkpoint:
            checkpoint = load_checkpoint(Path(args.init_from_checkpoint), map_location=device)
            validate_checkpoint_config(checkpoint, config)
            model.load_state_dict(checkpoint["model_state_dict"])
        run_name = f"{args.model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        save_dir = Path(args.save_dir) / run_name

    if start_epoch > args.epochs:
        raise ValueError(
            f"resume checkpoint is already at epoch {start_epoch - 1}, "
            f"but --epochs={args.epochs}. Increase --epochs to continue training."
        )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory and device.type == "cuda",
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory and device.type == "cuda",
    )

    save_dir.mkdir(parents=True, exist_ok=True)
    save_yaml(save_dir / "config.yaml", config)
    normalizer.save(save_dir / "normalizer.pt")

    for epoch in range(start_epoch, args.epochs + 1):
        train_loss, train_rmse, train_rollout_loss = run_epoch(
            model,
            train_loader,
            normalizer,
            dataset.state_dim,
            args.model_type,
            device,
            optimizer,
            scaler,
            use_amp,
            args.q_weight,
            args.dq_weight,
            q_extra_weights,
            dq_extra_weights,
            args.target_mode,
            args.control_dt,
            args.loss_type,
            args.huber_delta,
            args.rollout_loss_weight,
            args.rollout_loss_discount,
        )
        val_loss, val_rmse, val_rollout_loss = run_epoch(
            model,
            val_loader,
            normalizer,
            dataset.state_dim,
            args.model_type,
            device,
            q_weight=args.q_weight,
            dq_weight=args.dq_weight,
            q_extra_weights=q_extra_weights,
            dq_extra_weights=dq_extra_weights,
            target_mode=args.target_mode,
            control_dt=args.control_dt,
            loss_type=args.loss_type,
            huber_delta=args.huber_delta,
            rollout_loss_weight=args.rollout_loss_weight,
            rollout_loss_discount=args.rollout_loss_discount,
        )
        is_best = val_loss < best_val
        is_best_rollout = bool(use_rollout_loss and val_rollout_loss < best_rollout)
        if is_best:
            best_val = val_loss
        if is_best_rollout:
            best_rollout = val_rollout_loss
        print(format_epoch_header(epoch, train_loss, val_loss, is_best))
        if use_rollout_loss:
            print(
                f"rollout_loss train={train_rollout_loss:.6f} val={val_rollout_loss:.6f} "
                f"steps={args.rollout_loss_steps} weight={args.rollout_loss_weight:.6f}",
                flush=True,
            )
        print(format_delta_rmse_table(train_rmse, val_rmse, dataset.state_dim // 2))
        if is_best:
            save_checkpoint(
                save_dir / "best_model.pt",
                model,
                config,
                optimizer=optimizer,
                scaler=scaler,
                metadata=checkpoint_metadata("best", epoch, train_loss, val_loss, best_val, args, best_rollout),
            )
            print(f"  ✓ saved best checkpoint: {save_dir / 'best_model.pt'}")
        if is_best_rollout:
            save_checkpoint(
                save_dir / "best_rollout_model.pt",
                model,
                config,
                optimizer=optimizer,
                scaler=scaler,
                metadata=checkpoint_metadata("best_rollout", epoch, train_loss, val_loss, best_val, args, best_rollout),
            )
            print(f"  ✓ saved best rollout checkpoint: {save_dir / 'best_rollout_model.pt'}")
        save_checkpoint(
            save_dir / "latest_model.pt",
            model,
            config,
            optimizer=optimizer,
            scaler=scaler,
            metadata=checkpoint_metadata("latest", epoch, train_loss, val_loss, best_val, args, best_rollout),
        )
        print(f"  → saved latest checkpoint: {save_dir / 'latest_model.pt'}")


if __name__ == "__main__":
    main()
