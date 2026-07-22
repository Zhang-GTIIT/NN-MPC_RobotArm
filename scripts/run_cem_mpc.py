from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DYNAMICS_ROOT = ROOT / "dynamics_modeling"
for path in (ROOT, DYNAMICS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import numpy as np
import torch

from neural_dynamics.mujoco_env import MuJoCoArmEnv
from neural_dynamics.paths import DEFAULT_MODEL_XML
from neural_dynamics.rollout import load_dynamics_bundle, rollout_dynamics_batch
from neural_dynamics.train_utils import set_seed
from mpc.cem_controller import CEMMPCConfig, CEMMPCController
from mpc.constraints import project_nominal_q_ref_sequence
from mpc.delay_aware import DelayedPlanPacket, corrected_direct_ik_command, feedback_correction
from mpc.cost_functions import JointSpaceCostConfig
from mpc.kinematics_utils import site_pose
from mpc.logging import save_mpc_run
from mpc.delay_aware_runner import run as run_delay_aware_virtual
from mpc.asap_runner import run as run_threaded_asap
from mpc.planner_rollout import LearnedDynamicsPlanner, PlannerRolloutConfig, reanchor_residual_command
from mpc.reference import finite_difference_dq, generate_joint_reference
from mpc.reference_pipeline import ReferenceBundle, load_reference_bundle
from mpc.recovery import residual_recovery_reason
from mpc.replay_diagnostics import replay_executed_commands
from mpc.robustness import RobustnessConfig, config_arrays, resolve_robustness_config
from mpc.utils import build_history_tensor
from mpc.history import commit_command_and_append_placeholder


MPC_HOME_Q = np.zeros(6, dtype=np.float32)
# Conservative MuJoCo command-planning caps, not ABB hardware ratings.
# Each acceleration cap is five times its matching speed cap, giving a 0.2 s
# ramp from rest to the cap before reference-specific auto calibration.
DEFAULT_JOINT_VELOCITY_LIMIT = (1.0, 1.0, 1.0, 2.0, 2.0, 2.5)
DEFAULT_JOINT_ACCELERATION_LIMIT = (5.0, 5.0, 5.0, 10.0, 10.0, 12.5)
DEFAULT_RESIDUAL_MAX = (0.12, 0.10, 0.12, 0.15, 0.15, 0.20)
DEFAULT_SERVO_SCALE = (0.08, 0.07, 0.08, 0.04, 0.025, 0.05)
COST_TERM_NAMES = (
    "q_tracking",
    "dq_tracking",
    "residual",
    "servo",
    "residual_velocity",
    "residual_acceleration",
    "first",
    "qref_velocity",
    "qref_acceleration",
    "terminal",
    "joint_limit",
    "dq_limit",
    "torque",
    "delta_torque",
    "total",
)


def resolve_runtime_path(path: str) -> Path:
    expanded = Path(path).expanduser()
    if expanded.is_absolute():
        return expanded
    root_path = ROOT / expanded
    if root_path.exists() or (expanded.parts and expanded.parts[0] == "dynamics_modeling"):
        return root_path
    dynamics_path = DYNAMICS_ROOT / expanded
    if dynamics_path.exists():
        return dynamics_path
    if expanded.parts and expanded.parts[0] == "outputs":
        return root_path
    return root_path


def _robustness_config(args: argparse.Namespace) -> RobustnessConfig:
    return resolve_robustness_config(args, resolve_runtime_path)


def _build_control_env(args: argparse.Namespace, *, seed: int | None = None) -> MuJoCoArmEnv:
    config = _robustness_config(args)
    environment_seed = args.seed if seed is None else seed
    return MuJoCoArmEnv(
        str(config.plant_model_xml), n_joints=args.n_joints, seed=environment_seed,
        gravity_compensation_model_xml=str(config.nominal_model_xml),
        actuator_kp_scale=config.actuator_kp_scale,
        actuator_kd_scale=config.actuator_kd_scale,
        observation_q_noise_std=config.observation_q_std_rad,
        observation_dq_noise_std=config.observation_dq_std_rad_s,
        observation_seed=environment_seed + 104729,
    )


def _force_for_step(config: RobustnessConfig, step: int, execution_steps: int) -> np.ndarray:
    start, stop = config.pulse_window(execution_steps)
    return config.force_world() if start <= step < stop else np.zeros(3, dtype=np.float32)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run learned CEM-MPC in closed-loop MuJoCo simulation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model_xml", default=DEFAULT_MODEL_XML, type=str)
    parser.add_argument("--payload_level", choices=range(7), default=0, type=int)
    parser.add_argument("--actuator_gain_level", choices=range(7), default=0, type=int)
    parser.add_argument("--force_pulse_level", choices=range(7), default=0, type=int)
    parser.add_argument("--observation_noise_level", choices=range(7), default=0, type=int)
    parser.add_argument("--checkpoint", default=None, type=str, help="Dynamics checkpoint. Required for --controller_mode mpc.")
    parser.add_argument("--normalizer", default=None, type=str, help="Dynamics normalizer. Required for --controller_mode mpc.")
    parser.add_argument("--model_type", choices=["mlp", "gru", "transformer"], default="transformer")
    parser.add_argument(
        "--dynamics_backend",
        choices=["learned", "mujoco_oracle"],
        default="learned",
        help="State rollout backend. mujoco_oracle is an offline virtual-ASAP upper bound and does not require a checkpoint.",
    )
    parser.add_argument("--history_len", default=None, type=int)
    parser.add_argument("--n_joints", default=6, type=int)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--episode_len", default=200, type=int)
    parser.add_argument(
        "--max_execution_steps",
        default=None,
        type=int,
        help="Optional cap on executed control steps, including task-reference runs.",
    )
    parser.add_argument("--settle_steps", default=50, type=int)
    parser.add_argument(
        "--reference_mode",
        choices=["hold", "step", "joint_sine", "multi_joint_sine", "waypoint", "chirp", "joint_file", "task"],
        default="multi_joint_sine",
    )
    parser.add_argument(
        "--reference_file",
        default=None,
        type=str,
        help="Immutable reference .npz file. Required for --reference_mode task or joint_file.",
    )
    parser.add_argument("--ee_site_name", default="ee_site", type=str)
    parser.add_argument("--reference_amplitude", default=0.15, type=float)
    parser.add_argument("--save_dir", default="outputs/mpc/cem_run", type=str)
    parser.add_argument("--fail_on_limit_violation", action="store_true")
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Open a MuJoCo viewer and replay the closed-loop rollout at the simulation control period.",
    )
    parser.add_argument(
        "--controller_mode",
        choices=["mpc", "ik_direct"],
        default="mpc",
        help="mpc uses learned CEM-MPC; ik_direct sends the validated task-space IK q_des directly to position actuators.",
    )
    parser.add_argument(
        "--ik_preview_steps",
        default=0,
        type=int,
        help="Fixed task-reference preview for Direct/Preview IK; calibrated once and frozen across test trajectories.",
    )

    parser.add_argument("--horizon", default=20, type=int)
    parser.add_argument(
        "--replan_interval_steps",
        default=5,
        type=int,
        help="Fixed replan interval for synchronous/virtual modes; threaded_asap replans whenever its worker finishes.",
    )
    parser.add_argument(
        "--multirate_mode",
        choices=["synchronous", "virtual_asap", "virtual_smooth", "threaded_asap"],
        default="threaded_asap",
        help="Default threaded_asap runs wall-clock 100 Hz control with a CUDA planner thread; virtual_asap is the deterministic ablation mode.",
    )
    parser.add_argument(
        "--anticipation_delay_steps",
        default=6,
        type=int,
        help="Expected planner-to-activation delay in 100 Hz steps. The default 6-step delay matches the measured GRU planning latency.",
    )
    parser.add_argument(
        "--delay_protocol",
        choices=["full", "naive_delayed", "no_future_alignment", "no_reanchor", "no_feedback"],
        default="full",
        help="Canonical fixed-delay causal variant. Threaded deployment only supports full.",
    )
    parser.add_argument("--planner_guard_ms", default=5.0, type=float, help="threaded_asap drops a packet published within this many ms of its activation deadline.")
    parser.add_argument("--planner_min_interval_ms", default=0.0, type=float, help="Minimum delay between threaded planner launches; zero means strict ASAP.")
    parser.add_argument(
        "--asap_history_mode",
        choices=["aligned", "legacy_shifted"],
        default="aligned",
        help="Threaded ASAP ablation: use training-aligned [x_t,u_t] history or the former one-step-shifted history.",
    )
    parser.add_argument(
        "--asap_snapshot_mode",
        choices=["tick_start", "post_step_legacy"],
        default="tick_start",
        help="Threaded ASAP ablation: publish physical tick-start states or reproduce the former post-step snapshot phase.",
    )
    parser.add_argument("--feedback_kq", default=0.30, type=float, help="ASAP/tube position-feedback gain.")
    parser.add_argument("--feedback_kdq", default=0.015, type=float, help="ASAP/tube velocity-feedback gain in seconds.")
    parser.add_argument(
        "--feedback_max",
        default="0.015",
        type=str,
        help="Maximum absolute feedback correction in rad: scalar or one value per joint.",
    )
    parser.add_argument(
        "--mpc_warmup_plans",
        default=1,
        type=int,
        help="Discarded CEM plans before the first control command, used to warm up CUDA execution.",
    )
    parser.add_argument(
        "--mpc_policy",
        choices=["residual", "legacy_acceleration"],
        default="residual",
        help="Default residual MPC anchors commands to an executable IK nominal; legacy_acceleration reproduces the former unanchored action space.",
    )
    parser.add_argument("--num_samples", default=128, type=int)
    parser.add_argument("--num_elites", default=None, type=int)
    parser.add_argument("--elite_ratio", default=0.08, type=float)
    parser.add_argument("--cem_iters", default=2, type=int)
    parser.add_argument(
        "--init_std",
        default=0.5,
        type=float,
        help="Initial CEM standard deviation in normalized action units.",
    )
    parser.add_argument(
        "--min_std",
        default=0.25,
        type=float,
        help="Warm-start floor for the CEM standard deviation in normalized action units.",
    )
    parser.add_argument("--smoothing_alpha", default=0.2, type=float)
    parser.add_argument("--temporal_noise_alpha", default=0.8, type=float)
    parser.add_argument(
        "--reset_std_each_step",
        action="store_true",
        help="Reset CEM std to init_std at every MPC step while retaining the shifted mean warm start.",
    )
    parser.add_argument(
        "--uniform_sample_ratio",
        default=0.15,
        type=float,
        help="Fraction of non-forced CEM candidates sampled uniformly from [-1, 1].",
    )
    parser.add_argument("--rollout_batch_size", default=128, type=int)
    parser.add_argument(
        "--cem_execute",
        choices=["mean", "best", "lowest_cost"],
        default="lowest_cost",
        help="Action selected from the CEM mean, best sample, or their lowest-cost comparison with the residual baseline.",
    )

    parser.add_argument(
        "--q_ref_velocity_limit",
        default="auto",
        type=str,
        help="Command velocity limit in rad/s: auto, one scalar, or n_joints comma-separated values.",
    )
    parser.add_argument(
        "--q_ref_acceleration_limit",
        default="auto",
        type=str,
        help="Command acceleration limit in rad/s^2: auto, one scalar, or n_joints comma-separated values.",
    )
    parser.add_argument(
        "--command_velocity_physical_limit",
        default=",".join(map(str, DEFAULT_JOINT_VELOCITY_LIMIT)),
        type=str,
        help="Upper MuJoCo command-planning speed cap in rad/s; scalar or comma-separated per-joint values.",
    )
    parser.add_argument(
        "--command_acceleration_physical_limit",
        default=",".join(map(str, DEFAULT_JOINT_ACCELERATION_LIMIT)),
        type=str,
        help="Upper MuJoCo command-planning acceleration cap in rad/s^2; scalar or comma-separated values.",
    )
    parser.add_argument(
        "--state_velocity_limit",
        default=",".join(map(str, DEFAULT_JOINT_VELOCITY_LIMIT)),
        type=str,
        help="Predicted joint-speed soft-limit in rad/s; scalar or comma-separated per-joint values.",
    )
    parser.add_argument("--joint_limit_margin", default=0.02, type=float)
    parser.add_argument(
        "--residual_max",
        default=",".join(map(str, DEFAULT_RESIDUAL_MAX)),
        type=str,
        help="Residual q_ref bound in rad: scalar or comma-separated per joint. Used by --mpc_policy residual.",
    )
    parser.add_argument("--temporal_discount", default=0.95, type=float)
    parser.add_argument("--barrier_max_weight", default=2.0, type=float)
    parser.add_argument(
        "--servo_scale",
        default=",".join(map(str, DEFAULT_SERVO_SCALE)),
        type=str,
        help="Servo proxy scale in rad: scalar or comma-separated per joint.",
    )
    parser.add_argument("--recovery_error_ratio", default=1.25, type=float)
    parser.add_argument(
        "--recovery_min_tracking_error",
        default=0.05,
        type=float,
        help="Minimum joint-space L2 tracking error in rad before growth can trigger recovery.",
    )
    parser.add_argument(
        "--recovery_residual_fraction",
        default=0.95,
        type=float,
        help="Residual-bound fraction sustained before recovery can trigger; command-limit saturation is diagnostic only.",
    )
    parser.add_argument("--recovery_consecutive_steps", default=3, type=int)
    parser.add_argument("--recovery_cooldown_steps", default=5, type=int)

    parser.add_argument("--w_q", default=1.0, type=float)
    parser.add_argument("--w_dq", default=0.10, type=float)
    parser.add_argument("--w_residual", default=0.20, type=float)
    parser.add_argument("--w_servo", default=0.05, type=float)
    parser.add_argument("--w_residual_velocity", default=0.05, type=float)
    parser.add_argument("--w_residual_acceleration", default=0.02, type=float)
    parser.add_argument("--w_first", default=0.20, type=float)
    # The two qref smoothness weights are retained solely for legacy_acceleration.
    parser.add_argument("--w_qref_velocity", default=0.05, type=float)
    parser.add_argument("--w_qref_acceleration", default=0.02, type=float)
    parser.add_argument("--w_terminal", default=0.0, type=float)
    parser.add_argument("--w_joint_limit", default=10.0, type=float)
    parser.add_argument("--w_dq_limit", default=5.0, type=float)
    parser.add_argument("--w_tau", default=0.02, type=float, help="Used only by --cost_profile actuator_aware.")
    parser.add_argument("--w_delta_tau", default=0.02, type=float, help="Used only by --cost_profile actuator_aware.")
    parser.add_argument("--cost_profile", choices=["blackbox", "actuator_aware"], default="blackbox")
    parser.add_argument("--joint_limit_safe_margin", default=0.08, type=float)
    parser.add_argument("--joint_limit_temp", default=0.02, type=float)
    parser.add_argument("--dq_limit_temp", default=0.1, type=float)
    parser.add_argument("--velocity_cost_mode", choices=["track", "damping"], default="track")
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


def _parse_joint_vector(value: str, n_joints: int, name: str, *, allow_auto: bool = False) -> np.ndarray | None:
    text = str(value).strip().lower()
    if allow_auto and text == "auto":
        return None
    try:
        values = np.asarray([float(item) for item in text.split(",")], dtype=np.float32)
    except ValueError as exc:
        raise ValueError(f"{name} must be 'auto', one positive scalar, or {n_joints} comma-separated values") from exc
    if values.size == 1:
        values = np.full(n_joints, float(values[0]), dtype=np.float32)
    if values.shape != (n_joints,) or not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError(f"{name} must contain {n_joints} finite positive values")
    return values


def _reference_calibration(
    reference: np.ndarray,
    dq_reference: np.ndarray,
    ddq_reference: np.ndarray,
    physical_velocity_limit: np.ndarray,
    physical_acceleration_limit: np.ndarray,
) -> dict[str, np.ndarray]:
    q_scale = np.clip(
        0.1 * (np.percentile(reference, 95.0, axis=0) - np.percentile(reference, 5.0, axis=0)),
        0.04,
        0.08,
    )
    dq_scale = np.maximum(np.percentile(np.abs(dq_reference), 99.0, axis=0), 0.25)
    velocity_limit = np.clip(
        3.0 * np.percentile(np.abs(dq_reference), 99.0, axis=0),
        0.05 * physical_velocity_limit,
        physical_velocity_limit,
    )
    acceleration_limit = np.clip(
        3.0 * np.percentile(np.abs(ddq_reference), 99.0, axis=0),
        0.05 * physical_acceleration_limit,
        physical_acceleration_limit,
    )
    return {
        "q_tracking_scale": q_scale.astype(np.float32),
        "dq_tracking_scale": dq_scale.astype(np.float32),
        "q_ref_velocity_limit": velocity_limit.astype(np.float32),
        "q_ref_acceleration_limit": acceleration_limit.astype(np.float32),
    }


def _validate_task_reference(bundle: ReferenceBundle, n_joints: int, future_steps: int) -> None:
    """Validate the reference invariants needed by the selected online controller."""
    q_des = np.asarray(bundle.q_des)
    dq_des = np.asarray(bundle.dq_des)
    if q_des.ndim != 2 or q_des.shape[1] != n_joints:
        raise ValueError(f"Task reference q_des must have shape [N, {n_joints}], got {q_des.shape}")
    if dq_des.shape != q_des.shape:
        raise ValueError(f"Task reference dq_des must match q_des shape {q_des.shape}, got {dq_des.shape}")
    if bundle.execution_steps <= 0:
        raise ValueError(f"Task reference execution_steps must be positive, got {bundle.execution_steps}")
    minimum_length = int(bundle.execution_steps) + int(future_steps) + 1
    if q_des.shape[0] < minimum_length:
        raise ValueError(
            "Task reference is too short for the requested controller look-ahead: "
            f"need at least execution_steps + future_steps + 1 = {minimum_length} points, got {q_des.shape[0]}"
        )

    expected_task_shapes = {
        "task_positions_des": (q_des.shape[0], 3),
        "task_rotations_des": (q_des.shape[0], 3, 3),
        "segment_ids": (q_des.shape[0],),
        "lap_ids": (q_des.shape[0],),
    }
    for name, expected_shape in expected_task_shapes.items():
        value = getattr(bundle, name, None)
        if value is None or np.asarray(value).shape != expected_shape:
            actual_shape = None if value is None else np.asarray(value).shape
            raise ValueError(f"Task reference {name} must have shape {expected_shape}, got {actual_shape}")


def _load_task_reference(args: argparse.Namespace) -> ReferenceBundle:
    if not args.reference_file:
        raise ValueError("--reference_file is required when --reference_mode task")
    # Virtual delay-aware execution shortens the executable prefix near the
    # end of a reference, rather than requiring a separately padded file.
    # The runner applies that prefix truncation before issuing a future plan.
    future_steps = args.horizon if args.controller_mode == "mpc" else int(args.ik_preview_steps)
    if args.controller_mode == "mpc" and args.multirate_mode in {"virtual_asap", "virtual_smooth"}:
        future_steps += int(args.anticipation_delay_steps)
    bundle = load_reference_bundle(
        resolve_runtime_path(args.reference_file),
        expected_n_joints=args.n_joints,
        min_horizon=future_steps,
    )
    _validate_task_reference(bundle, args.n_joints, future_steps)
    return bundle


def _load_joint_file_reference(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    if not args.reference_file:
        raise ValueError("--reference_file is required when --reference_mode joint_file")
    path = resolve_runtime_path(args.reference_file)
    with np.load(path, allow_pickle=False) as archive:
        required = {"q_des", "dq_des", "ddq_des", "execution_steps"}
        missing = required.difference(archive.files)
        if missing:
            raise KeyError(f"Joint benchmark reference {path} is missing {sorted(missing)}")
        q_des = np.asarray(archive["q_des"], dtype=np.float32)
        dq_des = np.asarray(archive["dq_des"], dtype=np.float32)
        ddq_des = np.asarray(archive["ddq_des"], dtype=np.float32)
        execution_steps = int(np.asarray(archive["execution_steps"]).item())
    if q_des.ndim != 2 or q_des.shape[1] != args.n_joints or dq_des.shape != q_des.shape or ddq_des.shape != q_des.shape:
        raise ValueError(f"Joint benchmark reference {path} has incompatible q/dq/ddq shapes")
    future_steps = args.horizon + (int(args.anticipation_delay_steps) if args.multirate_mode in {"virtual_asap", "virtual_smooth"} else 0)
    if execution_steps <= 0 or q_des.shape[0] < execution_steps + future_steps + 1:
        raise ValueError("Joint benchmark reference lacks horizon plus delay padding")
    if not np.all(np.isfinite(q_des)) or not np.all(np.isfinite(dq_des)) or not np.all(np.isfinite(ddq_des)):
        raise ValueError("Joint benchmark reference contains non-finite values")
    return q_des, dq_des, ddq_des, execution_steps


def _reference_for_run(
    args: argparse.Namespace,
    state: np.ndarray,
    env: MuJoCoArmEnv,
    control_dt: float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, ReferenceBundle | None]:
    """Return the reference arrays and the number of closed-loop control steps."""
    if args.reference_mode == "task":
        bundle = _load_task_reference(args)
        expected_initial_q = MPC_HOME_Q[: args.n_joints]
        if not np.allclose(bundle.q_des[0], expected_initial_q, atol=1e-6, rtol=0.0):
            raise ValueError(
                "Task reference must start at the fixed MPC home pose "
                f"{expected_initial_q.tolist()}, got {np.asarray(bundle.q_des[0]).tolist()}"
            )
        return (
            np.asarray(bundle.q_des, dtype=np.float32),
            np.asarray(bundle.dq_des, dtype=np.float32),
            np.asarray(bundle.ddq_des, dtype=np.float32),
            int(bundle.execution_steps),
            bundle,
        )

    if args.reference_mode == "joint_file":
        q_des, dq_des, ddq_des, execution_steps = _load_joint_file_reference(args)
        return q_des, dq_des, ddq_des, execution_steps, None

    if control_dt is None:
        raise RuntimeError("Joint-space references require a loaded MPC dynamics bundle")
    reference = generate_joint_reference(
        args.reference_mode,
        state[: args.n_joints],
        env.joint_low + args.joint_limit_margin,
        env.joint_high - args.joint_limit_margin,
        args.episode_len + args.horizon + 1,
        control_dt,
        seed=args.seed,
        amplitude=args.reference_amplitude,
    )
    dq_reference = finite_difference_dq(reference, control_dt)
    ddq_reference = finite_difference_dq(dq_reference, control_dt)
    return reference, dq_reference, ddq_reference, args.episode_len, None


def _orientation_error(desired_rotation: np.ndarray, actual_rotation: np.ndarray) -> float:
    """Return the geodesic orientation error in radians."""
    relative_rotation = np.asarray(desired_rotation, dtype=np.float64) @ np.asarray(actual_rotation, dtype=np.float64).T
    cosine = np.clip((np.trace(relative_rotation) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.arccos(cosine))


def _launch_mujoco_viewer(env: MuJoCoArmEnv) -> Any:
    """Open the optional passive viewer only for interactive local runs."""
    try:
        from mujoco import viewer

        return viewer.launch_passive(env.model, env.data)
    except Exception as exc:
        raise RuntimeError(
            "Could not open the MuJoCo viewer. Run with a local graphical display or remove --visualize."
        ) from exc


def run_closed_loop_mpc(args: argparse.Namespace, *, activation_observer: Any | None = None) -> dict[str, Any]:
    if activation_observer is not None:
        setattr(args, "activation_observer", activation_observer)
    if args.controller_mode == "ik_direct" and args.reference_mode != "task":
        raise ValueError("--controller_mode ik_direct requires --reference_mode task with a validated IK reference")
    if args.ik_preview_steps < 0:
        raise ValueError("--ik_preview_steps must be non-negative")
    if args.controller_mode != "ik_direct" and args.ik_preview_steps:
        raise ValueError("--ik_preview_steps is only valid with --controller_mode ik_direct")
    if args.anticipation_delay_steps < 0:
        raise ValueError("--anticipation_delay_steps must be non-negative")
    if args.reference_mode != "task" and args.episode_len <= 0:
        raise ValueError(f"episode_len must be positive, got {args.episode_len}")
    if args.controller_mode == "mpc" and args.horizon <= 0:
        raise ValueError(f"horizon must be positive, got {args.horizon}")
    if args.replan_interval_steps <= 0:
        raise ValueError("replan_interval_steps must be positive")
    if args.controller_mode == "mpc" and args.replan_interval_steps > args.horizon:
        raise ValueError("replan_interval_steps must not exceed horizon")
    if args.mpc_warmup_plans < 0:
        raise ValueError("mpc_warmup_plans must be non-negative")
    if args.max_execution_steps is not None and args.max_execution_steps <= 0:
        raise ValueError("max_execution_steps must be positive when provided")
    if not 0.0 <= args.uniform_sample_ratio <= 1.0:
        raise ValueError("uniform_sample_ratio must be in [0, 1]")
    if not 0.0 < args.temporal_discount <= 1.0:
        raise ValueError("temporal_discount must be in (0, 1]")
    if args.barrier_max_weight < 0.0:
        raise ValueError("barrier_max_weight must be non-negative")
    if args.recovery_consecutive_steps <= 0 or args.recovery_cooldown_steps < 0:
        raise ValueError("recovery step counts must be positive (cooldown may be zero)")
    if not 0.0 < args.recovery_residual_fraction <= 1.0:
        raise ValueError("recovery_residual_fraction must be in (0, 1]")
    if args.recovery_error_ratio <= 1.0:
        raise ValueError("recovery_error_ratio must be greater than 1")
    if args.recovery_min_tracking_error < 0.0:
        raise ValueError("recovery_min_tracking_error must be non-negative")
    if args.planner_guard_ms < 0.0 or args.planner_min_interval_ms < 0.0:
        raise ValueError("planner_guard_ms and planner_min_interval_ms must be non-negative")
    dynamics_backend = getattr(args, "dynamics_backend", "learned")
    robustness = _robustness_config(args)
    if robustness.enabled and dynamics_backend == "mujoco_oracle":
        raise ValueError("Robustness perturbations are only defined for learned MPC and Direct IK, not mujoco_oracle")
    if dynamics_backend == "mujoco_oracle":
        if args.controller_mode != "mpc" or args.mpc_policy != "residual":
            raise ValueError("mujoco_oracle requires --controller_mode mpc --mpc_policy residual")
        if args.multirate_mode != "virtual_asap":
            raise ValueError("mujoco_oracle is an offline upper bound and only supports --multirate_mode virtual_asap")
    if args.multirate_mode == "threaded_asap":
        if args.delay_protocol != "full":
            raise ValueError("threaded_asap only supports --delay_protocol full")
        return run_threaded_asap(args, globals())
    if args.multirate_mode != "synchronous":
        return run_delay_aware_virtual(args, globals())
    set_seed(args.seed)
    device = resolve_device(args.device)
    bundle = None
    if args.controller_mode == "mpc" and dynamics_backend == "learned":
        if not args.checkpoint or not args.normalizer:
            raise ValueError("--checkpoint and --normalizer are required when --controller_mode mpc")
        bundle = load_dynamics_bundle(
            checkpoint_path=resolve_runtime_path(args.checkpoint),
            normalizer_path=resolve_runtime_path(args.normalizer),
            model_type=args.model_type,
            n_joints=args.n_joints,
            device=device,
            history_len=args.history_len,
        )
    env = _build_control_env(args)

    states_history: list[np.ndarray] = []
    q_ref_history: list[np.ndarray] = []
    actual_states: list[np.ndarray] = []
    observed_states: list[np.ndarray] = []
    observation_noise: list[np.ndarray] = []
    next_states: list[np.ndarray] = []
    selected_q_refs: list[np.ndarray] = []
    selected_delta_q_refs: list[np.ndarray] = []
    command_velocities: list[np.ndarray] = []
    command_accelerations: list[np.ndarray] = []
    command_velocity_violations: list[int] = []
    command_acceleration_violations: list[int] = []
    q_des_records: list[np.ndarray] = []
    dq_des_records: list[np.ndarray] = []
    planning_times: list[float] = []
    replan_times: list[float] = []
    mpc_replanned_flags: list[int] = []
    buffer_indices: list[int] = []
    buffer_lengths: list[int] = []
    replan_deadline_miss_flags: list[int] = []
    control_step_wall_times: list[float] = []
    best_costs: list[float] = []
    mean_costs: list[float] = []
    baseline_costs: list[float] = []
    selected_costs: list[float] = []
    elite_costs: list[float] = []
    selection_modes: list[str] = []
    failures: list[int] = []
    failure_reasons: list[str] = []
    joint_limit_violations: list[int] = []
    realized_tracking_errors: list[float] = []
    predicted_next_q_errors: list[float] = []
    predicted_next_dq_errors: list[float] = []
    sampling_std_start_means: list[float] = []
    sampling_std_end_means: list[float] = []
    nominal_q_refs: list[np.ndarray] = []
    buffered_residuals: list[np.ndarray] = []
    executed_residuals: list[np.ndarray] = []
    residual_reanchor_deltas: list[np.ndarray] = []
    multirate_buffer_modes: list[str] = []
    recovery_active_flags: list[int] = []
    recovery_trigger_reasons: list[str] = []
    cost_term_records: dict[str, list[float]] = {name: [] for name in COST_TERM_NAMES}
    torque_records: dict[str, list[np.ndarray]] = {
        "tau_actuator": [], "tau_gravity": [], "tau_total": [],
        "tau_gravity_true": [], "tau_gravity_mismatch": [],
    }
    external_force_world_records: list[np.ndarray] = []
    external_generalized_force_records: list[np.ndarray] = []
    desired_ee_positions: list[np.ndarray] = []
    desired_ee_rotations: list[np.ndarray] = []
    actual_ee_positions: list[np.ndarray] = []
    actual_ee_rotations: list[np.ndarray] = []
    ee_position_errors: list[float] = []
    ee_orientation_errors: list[float] = []
    segment_ids: list[int] = []
    lap_ids: list[int] = []
    rows: list[dict[str, Any]] = []
    task_reference: ReferenceBundle | None = None
    execution_steps = args.episode_len
    viewer: Any | None = None

    try:
        if args.n_joints > MPC_HOME_Q.shape[0]:
            raise ValueError(f"MPC home pose supports at most {MPC_HOME_Q.shape[0]} joints, got {args.n_joints}")
        true_state = env.reset_to_configuration(MPC_HOME_Q[: args.n_joints])
        previous_q_ref = np.asarray(true_state[: args.n_joints], dtype=np.float32).copy()
        previous_q_ref_velocity = np.zeros(args.n_joints, dtype=np.float32)
        previous_residual = np.zeros(args.n_joints, dtype=np.float32)
        previous_residual_velocity = np.zeros(args.n_joints, dtype=np.float32)
        recovery_remaining = 0
        residual_saturation_streak = 0
        for _ in range(args.settle_steps):
            true_state = env.step(previous_q_ref)
        state = env.get_observation()
        states_history.append(state.copy())
        q_ref_history.append(previous_q_ref.copy())

        reference, dq_reference, ddq_reference, execution_steps, task_reference = _reference_for_run(
            args=args,
            state=true_state,
            env=env,
            control_dt=bundle.control_dt if bundle is not None else None,
        )
        if args.max_execution_steps is not None:
            execution_steps = min(execution_steps, args.max_execution_steps)
        cost_config = None
        rollout_config = None
        physical_velocity_limit = _parse_joint_vector(
            args.command_velocity_physical_limit, args.n_joints, "command_velocity_physical_limit"
        )
        physical_acceleration_limit = _parse_joint_vector(
            args.command_acceleration_physical_limit, args.n_joints, "command_acceleration_physical_limit"
        )
        command_velocity_limit_for_log = physical_velocity_limit
        command_acceleration_limit_for_log = physical_acceleration_limit
        if args.controller_mode == "mpc":
            state_velocity_limit = _parse_joint_vector(args.state_velocity_limit, args.n_joints, "state_velocity_limit")
            residual_max = _parse_joint_vector(args.residual_max, args.n_joints, "residual_max")
            servo_scale = _parse_joint_vector(args.servo_scale, args.n_joints, "servo_scale")
            calibration = _reference_calibration(
                reference,
                dq_reference,
                ddq_reference,
                physical_velocity_limit,
                physical_acceleration_limit,
            )
            q_ref_velocity_limit = _parse_joint_vector(
                args.q_ref_velocity_limit, args.n_joints, "q_ref_velocity_limit", allow_auto=True
            )
            q_ref_acceleration_limit = _parse_joint_vector(
                args.q_ref_acceleration_limit, args.n_joints, "q_ref_acceleration_limit", allow_auto=True
            )
            if q_ref_velocity_limit is None:
                q_ref_velocity_limit = calibration["q_ref_velocity_limit"]
            if q_ref_acceleration_limit is None:
                q_ref_acceleration_limit = calibration["q_ref_acceleration_limit"]
            command_velocity_limit_for_log = q_ref_velocity_limit
            command_acceleration_limit_for_log = q_ref_acceleration_limit
            actuator_kp = actuator_kd = torque_scale = delta_torque_scale = None
            w_tau = w_delta_tau = 0.0
            if args.cost_profile == "actuator_aware":
                actuator_kp, actuator_kd = env.nominal_position_actuator_gains
                torque_scale = np.maximum(actuator_kp * q_ref_velocity_limit * env.control_dt, 1.0).astype(np.float32)
                delta_torque_scale = np.maximum(
                    actuator_kp * q_ref_acceleration_limit * (env.control_dt**2), 0.5
                ).astype(np.float32)
                w_tau = args.w_tau
                w_delta_tau = args.w_delta_tau
            cost_config = JointSpaceCostConfig(
                cost_mode="residual" if args.mpc_policy == "residual" else "legacy",
                w_q=args.w_q,
                w_dq=args.w_dq,
                w_residual=args.w_residual,
                w_servo=args.w_servo,
                w_residual_velocity=args.w_residual_velocity,
                w_residual_acceleration=args.w_residual_acceleration,
                w_first=args.w_first,
                w_qref_velocity=args.w_qref_velocity,
                w_qref_acceleration=args.w_qref_acceleration,
                w_terminal=args.w_terminal,
                w_joint_limit=args.w_joint_limit,
                w_dq_limit=args.w_dq_limit,
                q_tracking_scale=torch.as_tensor(calibration["q_tracking_scale"], dtype=torch.float32, device=device),
                dq_tracking_scale=torch.as_tensor(calibration["dq_tracking_scale"], dtype=torch.float32, device=device),
                residual_scale=torch.as_tensor(0.5 * residual_max, dtype=torch.float32, device=device),
                servo_scale=torch.as_tensor(servo_scale, dtype=torch.float32, device=device),
                residual_velocity_scale=torch.as_tensor(residual_max / bundle.control_dt, dtype=torch.float32, device=device),
                residual_acceleration_scale=torch.as_tensor(
                    residual_max / (bundle.control_dt**2), dtype=torch.float32, device=device
                ),
                qref_velocity_scale=torch.as_tensor(q_ref_velocity_limit, dtype=torch.float32, device=device),
                qref_acceleration_scale=torch.as_tensor(q_ref_acceleration_limit, dtype=torch.float32, device=device),
                temporal_discount=args.temporal_discount,
                barrier_max_weight=args.barrier_max_weight,
                state_velocity_limit=torch.as_tensor(state_velocity_limit, dtype=torch.float32, device=device),
                joint_limit_safe_margin=args.joint_limit_safe_margin,
                joint_limit_temp=args.joint_limit_temp,
                dq_limit_temp=args.dq_limit_temp,
                control_dt=bundle.control_dt,
                velocity_cost_mode=args.velocity_cost_mode,
                w_tau=w_tau,
                w_delta_tau=w_delta_tau,
                actuator_kp=None if actuator_kp is None else torch.as_tensor(actuator_kp, dtype=torch.float32, device=device),
                actuator_kd=None if actuator_kd is None else torch.as_tensor(actuator_kd, dtype=torch.float32, device=device),
                torque_scale=None if torque_scale is None else torch.as_tensor(torque_scale, dtype=torch.float32, device=device),
                delta_torque_scale=None
                if delta_torque_scale is None
                else torch.as_tensor(delta_torque_scale, dtype=torch.float32, device=device),
            )
            rollout_config = PlannerRolloutConfig(
                mpc_policy=args.mpc_policy,
                q_ref_velocity_limit=torch.as_tensor(q_ref_velocity_limit, dtype=torch.float32, device=device),
                q_ref_acceleration_limit=torch.as_tensor(q_ref_acceleration_limit, dtype=torch.float32, device=device),
                residual_max=torch.as_tensor(residual_max, dtype=torch.float32, device=device),
                joint_limit_margin=args.joint_limit_margin,
                rollout_batch_size=args.rollout_batch_size,
            )
        controller: CEMMPCController | None = None
        # ``command_buffer`` is retained for the old absolute-command action
        # space only.  Residual MPC caches corrections and re-anchors them at
        # every 100 Hz actuator command below.
        command_buffer = np.empty((0, args.n_joints), dtype=np.float32)
        residual_buffer = np.empty((0, args.n_joints), dtype=np.float32)
        command_buffer_index = 0
        command_buffer_plan_length = 0
        warmup_completed = False
        viewer_deadline = 0.0
        if args.visualize:
            viewer = _launch_mujoco_viewer(env)
            viewer.sync()
            viewer_deadline = time.perf_counter() + env.control_dt

        def reanchor_buffered_residual(buffered_residual: np.ndarray, nominal: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool]:
            if rollout_config is None or bundle is None:
                raise RuntimeError("residual re-anchoring requires initialized MPC rollout configuration")
            command, executed, feasible = reanchor_residual_command(
                buffered_residual=torch.as_tensor(buffered_residual, dtype=torch.float32, device=device),
                nominal_q_ref=torch.as_tensor(nominal, dtype=torch.float32, device=device),
                residual_max=rollout_config.residual_max,
                previous_q_ref=torch.as_tensor(previous_q_ref, dtype=torch.float32, device=device),
                previous_q_ref_velocity=torch.as_tensor(previous_q_ref_velocity, dtype=torch.float32, device=device),
                joint_low=torch.as_tensor(env.joint_low, dtype=torch.float32, device=device),
                joint_high=torch.as_tensor(env.joint_high, dtype=torch.float32, device=device),
                joint_limit_margin=rollout_config.joint_limit_margin,
                q_ref_velocity_limit=rollout_config.q_ref_velocity_limit,
                q_ref_acceleration_limit=rollout_config.q_ref_acceleration_limit,
                control_dt=bundle.control_dt,
            )
            return (
                command.detach().cpu().numpy().astype(np.float32),
                executed.detach().cpu().numpy().astype(np.float32),
                feasible,
            )

        for step_idx in range(execution_steps):
            if viewer is not None and not viewer.is_running():
                break
            control_step_started = time.perf_counter()
            recovery_active = 0
            recovery_trigger_reason = ""
            buffered_residual = np.zeros(args.n_joints, dtype=np.float32)
            multirate_buffer_mode = "not_applicable"
            nominal_command = np.asarray(reference[step_idx + 1], dtype=np.float32).copy()
            if args.controller_mode == "mpc":
                if bundle is None or cost_config is None or rollout_config is None:
                    raise RuntimeError("MPC controller dependencies were not initialized")
                future_q_des = torch.as_tensor(
                    reference[step_idx + 1 : step_idx + 1 + args.horizon], dtype=torch.float32, device=device
                )
                if args.mpc_policy == "residual":
                    nominal_q_ref = project_nominal_q_ref_sequence(
                        future_q_des,
                        previous_q_ref=torch.as_tensor(previous_q_ref, dtype=torch.float32, device=device),
                        previous_q_ref_velocity=torch.as_tensor(previous_q_ref_velocity, dtype=torch.float32, device=device),
                        control_dt=bundle.control_dt,
                        velocity_limit=rollout_config.q_ref_velocity_limit,
                        acceleration_limit=rollout_config.q_ref_acceleration_limit,
                        joint_low=torch.as_tensor(env.joint_low, dtype=torch.float32, device=device),
                        joint_high=torch.as_tensor(env.joint_high, dtype=torch.float32, device=device),
                        joint_limit_margin=rollout_config.joint_limit_margin,
                    )
                    nominal_command = nominal_q_ref[0].detach().cpu().numpy().astype(np.float32)
                else:
                    nominal_q_ref = None
                if recovery_remaining > 0 and args.mpc_policy == "residual":
                    recovery_active = 1
                    recovery_remaining -= 1
                    command_buffer = np.empty((0, args.n_joints), dtype=np.float32)
                    residual_buffer = np.empty((0, args.n_joints), dtype=np.float32)
                    command_buffer_index = command_buffer_plan_length = 0
                    q_ref_command = nominal_command.copy()
                    delta_q_ref = q_ref_command - previous_q_ref
                    planning_time = 0.0
                    replan_time = float("nan")
                    mpc_replanned = 0
                    buffer_index = -1
                    buffer_length = 0
                    replan_deadline_miss = 0
                    best_cost = mean_cost = baseline_cost = selected_cost = elite_mean_cost = float("nan")
                    selection_mode = "recovery_nominal"
                    failure = 0
                    failure_reason = ""
                    selected_cost_terms = {}
                    predicted_next_state = np.full(2 * args.n_joints, np.nan, dtype=np.float32)
                    sampling_std_start_mean = sampling_std_end_mean = float("nan")
                    multirate_buffer_mode = "recovery_nominal"
                elif (
                    (args.mpc_policy == "residual" and command_buffer_index < residual_buffer.shape[0])
                    or (args.mpc_policy == "legacy_acceleration" and command_buffer_index < command_buffer.shape[0])
                ):
                    buffer_index = command_buffer_index
                    buffer_length = command_buffer_plan_length
                    if args.mpc_policy == "residual":
                        buffered_residual = residual_buffer[command_buffer_index].copy()
                        q_ref_command, _, residual_feasible = reanchor_buffered_residual(buffered_residual, nominal_command)
                        if not residual_feasible:
                            buffered_residual.fill(0.0)
                            q_ref_command = nominal_command.copy()
                            selection_mode = "buffered_residual_nominal_fallback"
                        else:
                            selection_mode = "buffered_residual_reanchored"
                        multirate_buffer_mode = "residual_reanchored"
                    else:
                        q_ref_command = command_buffer[command_buffer_index].copy()
                        selection_mode = "buffered"
                        multirate_buffer_mode = "absolute_q_ref_legacy"
                    command_buffer_index += 1
                    delta_q_ref = q_ref_command - previous_q_ref
                    planning_time = 0.0
                    replan_time = float("nan")
                    mpc_replanned = 0
                    replan_deadline_miss = 0
                    best_cost = mean_cost = baseline_cost = selected_cost = elite_mean_cost = float("nan")
                    failure = 0
                    failure_reason = ""
                    selected_cost_terms = {}
                    predicted_next_state = np.full(2 * args.n_joints, np.nan, dtype=np.float32)
                    sampling_std_start_mean = sampling_std_end_mean = float("nan")
                else:
                    initial_history = build_history_tensor(states_history, q_ref_history, bundle.history_len, device)
                    planner = LearnedDynamicsPlanner(
                        model=bundle.model,
                        normalizer=bundle.normalizer,
                        model_type=bundle.model_type,
                        state_dim=bundle.state_dim,
                        target_mode=bundle.target_mode,
                        control_dt=bundle.control_dt,
                        initial_history=initial_history,
                        q_des=future_q_des,
                        dq_des=torch.as_tensor(
                            dq_reference[step_idx + 1 : step_idx + 1 + args.horizon], dtype=torch.float32, device=device
                        ),
                        nominal_q_ref=nominal_q_ref,
                        previous_q_ref=torch.as_tensor(previous_q_ref, dtype=torch.float32, device=device),
                        previous_q_ref_velocity=torch.as_tensor(previous_q_ref_velocity, dtype=torch.float32, device=device),
                        previous_residual=None
                        if args.mpc_policy == "legacy_acceleration"
                        else torch.as_tensor(previous_residual, dtype=torch.float32, device=device),
                        previous_residual_velocity=None
                        if args.mpc_policy == "legacy_acceleration"
                        else torch.as_tensor(previous_residual_velocity, dtype=torch.float32, device=device),
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
                                reset_std_each_step=args.reset_std_each_step,
                                uniform_sample_ratio=args.uniform_sample_ratio,
                                force_baseline_candidate=args.mpc_policy == "residual",
                                execute=args.cem_execute,
                                seed=args.seed,
                                device=str(device),
                            ),
                            planner=planner,
                            joint_low=env.joint_low,
                            joint_high=env.joint_high,
                        )
                    else:
                        controller.planner = planner
                    if not warmup_completed:
                        if args.mpc_warmup_plans:
                            generator_state = controller.generator.get_state()
                            for _ in range(args.mpc_warmup_plans):
                                controller.plan(current_state=state, previous_q_ref=previous_q_ref)
                            controller.generator.set_state(generator_state)
                            controller.reset()
                        warmup_completed = True
                    result = controller.plan(current_state=state, previous_q_ref=previous_q_ref)
                    q_ref_command = result.q_ref.astype(np.float32)
                    delta_q_ref = result.delta_q_ref.astype(np.float32)
                    planning_time = float(result.planning_time)
                    replan_time = planning_time
                    mpc_replanned = 1
                    buffer_index = 0
                    replan_deadline_miss = int(planning_time > args.replan_interval_steps * bundle.control_dt)
                    best_cost = float(result.best_cost)
                    mean_cost = float(result.mean_cost)
                    baseline_cost = float(result.baseline_cost)
                    selected_cost = float(result.selected_cost)
                    elite_mean_cost = float(result.elite_mean_cost)
                    selection_mode = result.selection_mode
                    failure = int(result.failure)
                    failure_reason = result.failure_reason
                    selected_cost_terms = result.cost_terms
                    predicted_next_state = result.predicted_next_state
                    sampling_std_start_mean = float(getattr(result, "sampling_std_start_mean", float("nan")))
                    sampling_std_end_mean = float(getattr(result, "sampling_std_end_mean", float("nan")))
                    if result.failure and args.mpc_policy == "residual":
                        command_buffer = np.empty((0, args.n_joints), dtype=np.float32)
                        residual_buffer = np.empty((0, args.n_joints), dtype=np.float32)
                        command_buffer_index = command_buffer_plan_length = 0
                        buffer_length = 0
                        q_ref_command = nominal_command.copy()
                        delta_q_ref = q_ref_command - previous_q_ref
                        selection_mode = "planner_failure_nominal"
                        recovery_trigger_reason = "planner_failure"
                        recovery_remaining = args.recovery_cooldown_steps
                        controller.reset()
                        multirate_buffer_mode = "planner_failure_nominal"
                    else:
                        if args.mpc_policy == "residual":
                            selected_residual_sequence = np.asarray(
                                getattr(result, "selected_residual_sequence", np.empty((0, args.n_joints))), dtype=np.float32
                            )
                            if selected_residual_sequence.shape != (args.horizon, args.n_joints):
                                raise RuntimeError("CEM result has an invalid selected_residual_sequence shape")
                            buffer_length = min(
                                args.replan_interval_steps, selected_residual_sequence.shape[0], execution_steps - step_idx
                            )
                            residual_buffer = selected_residual_sequence[:buffer_length].copy()
                            command_buffer = np.empty((0, args.n_joints), dtype=np.float32)
                            buffered_residual = residual_buffer[0].copy()
                            q_ref_command, _, residual_feasible = reanchor_buffered_residual(buffered_residual, nominal_command)
                            if not residual_feasible:
                                buffered_residual.fill(0.0)
                                residual_buffer[0].fill(0.0)
                                q_ref_command = nominal_command.copy()
                                selection_mode = "replan_residual_nominal_fallback"
                            multirate_buffer_mode = "residual_reanchored"
                        else:
                            selected_sequence = np.asarray(
                                getattr(result, "selected_q_ref_sequence", result.q_ref[None, :]), dtype=np.float32
                            )
                            if selected_sequence.shape != (args.horizon, args.n_joints):
                                raise RuntimeError("CEM result has an invalid selected_q_ref_sequence shape")
                            buffer_length = min(args.replan_interval_steps, selected_sequence.shape[0], execution_steps - step_idx)
                            command_buffer = selected_sequence[:buffer_length].copy()
                            residual_buffer = np.empty((0, args.n_joints), dtype=np.float32)
                            q_ref_command = command_buffer[0].copy()
                            multirate_buffer_mode = "absolute_q_ref_legacy"
                        command_buffer_index = 1
                        command_buffer_plan_length = buffer_length
                        delta_q_ref = q_ref_command - previous_q_ref
            else:
                q_ref_command = np.asarray(
                    reference[step_idx + 1 + int(args.ik_preview_steps)], dtype=np.float32
                ).copy()
                delta_q_ref = q_ref_command - previous_q_ref
                planning_time = 0.0
                replan_time = float("nan")
                mpc_replanned = 0
                buffer_index = -1
                buffer_length = 0
                replan_deadline_miss = 0
                best_cost = float("nan")
                mean_cost = float("nan")
                baseline_cost = float("nan")
                selected_cost = float("nan")
                elite_mean_cost = float("nan")
                selection_mode = "not_applicable"
                failure = 0
                failure_reason = "not_applicable"
                selected_cost_terms = {}
                predicted_next_state = np.full(2 * args.n_joints, np.nan, dtype=np.float32)
                sampling_std_start_mean = float("nan")
                sampling_std_end_mean = float("nan")

            torque = env.compute_torque_components(q_ref_command)
            actual_states.append(true_state.copy())
            observed_states.append(state.copy())
            observation_noise.append((state - true_state).astype(np.float32))
            q_des_records.append(reference[step_idx].copy())
            dq_des_records.append(dq_reference[step_idx].copy())
            if args.controller_mode == "mpc" and args.mpc_policy == "residual":
                executed_residual = (q_ref_command - nominal_command).astype(np.float32)
                residual_reanchor_delta = (executed_residual - buffered_residual).astype(np.float32)
                nominal_q_refs.append(nominal_command.copy())
                buffered_residuals.append(buffered_residual.copy())
                executed_residuals.append(executed_residual.copy())
                residual_reanchor_deltas.append(residual_reanchor_delta.copy())
                multirate_buffer_modes.append(multirate_buffer_mode)
            else:
                executed_residual = np.zeros(args.n_joints, dtype=np.float32)
            if task_reference is not None:
                desired_position = np.asarray(task_reference.task_positions_des[step_idx], dtype=np.float32)
                desired_rotation = np.asarray(task_reference.task_rotations_des[step_idx], dtype=np.float32)
                actual_position, actual_rotation = site_pose(env.model, env.data, args.ee_site_name)
                actual_position = np.asarray(actual_position, dtype=np.float32)
                actual_rotation = np.asarray(actual_rotation, dtype=np.float32)
                desired_ee_positions.append(desired_position)
                desired_ee_rotations.append(desired_rotation)
                actual_ee_positions.append(actual_position)
                actual_ee_rotations.append(actual_rotation)
                ee_position_errors.append(float(np.linalg.norm(actual_position - desired_position)))
                ee_orientation_errors.append(_orientation_error(desired_rotation, actual_rotation))
                segment_ids.append(int(task_reference.segment_ids[step_idx]))
                lap_ids.append(int(task_reference.lap_ids[step_idx]))
            selected_q_refs.append(q_ref_command.copy())
            selected_delta_q_refs.append(delta_q_ref.copy())
            command_velocity = delta_q_ref / env.control_dt
            command_acceleration = (command_velocity - previous_q_ref_velocity) / env.control_dt
            command_velocities.append(command_velocity.astype(np.float32))
            command_accelerations.append(command_acceleration.astype(np.float32))
            command_velocity_violations.append(int(np.any(np.abs(command_velocity) > command_velocity_limit_for_log + 1e-6)))
            command_acceleration_violations.append(
                int(np.any(np.abs(command_acceleration) > command_acceleration_limit_for_log + 1e-6))
            )
            planning_times.append(planning_time)
            replan_times.append(replan_time)
            mpc_replanned_flags.append(mpc_replanned)
            buffer_indices.append(buffer_index)
            buffer_lengths.append(buffer_length)
            replan_deadline_miss_flags.append(replan_deadline_miss)
            best_costs.append(best_cost)
            mean_costs.append(mean_cost)
            baseline_costs.append(baseline_cost)
            selected_costs.append(selected_cost)
            elite_costs.append(elite_mean_cost)
            selection_modes.append(selection_mode)
            failures.append(failure)
            failure_reasons.append(failure_reason)
            for name in COST_TERM_NAMES:
                cost_term_records[name].append(float(selected_cost_terms.get(name, np.nan)))
            for key, target_key in (
                ("actuator_tau", "tau_actuator"), ("gravity_tau", "tau_gravity"),
                ("total_tau", "tau_total"), ("true_gravity_tau", "tau_gravity_true"),
                ("gravity_mismatch_tau", "tau_gravity_mismatch"),
            ):
                torque_records[target_key].append(torque[key].astype(np.float32))

            # Commit the command to the current state before advancing the
            # simulator, matching training tokens [x_t, u_t].
            q_ref_history[-1] = q_ref_command.copy()
            external_force = _force_for_step(robustness, step_idx, execution_steps)
            try:
                true_state = env.step(q_ref_command, external_force_world=external_force)
                state = env.get_observation()
                joint_limit_violations.append(0)
            except RuntimeError:
                joint_limit_violations.append(1)
                if args.fail_on_limit_violation:
                    raise
                break
            external_force_world_records.append(env.last_external_force_world.astype(np.float32).copy())
            external_generalized_force_records.append(env.last_external_generalized_force.astype(np.float32).copy())
            control_step_wall_times.append(time.perf_counter() - control_step_started)

            if (
                args.controller_mode == "mpc"
                and controller is not None
                and command_buffer_plan_length > 0
                and command_buffer_index >= command_buffer_plan_length
            ):
                controller.advance_after_execution(command_buffer_plan_length)
                command_buffer = np.empty((0, args.n_joints), dtype=np.float32)
                residual_buffer = np.empty((0, args.n_joints), dtype=np.float32)
                command_buffer_index = command_buffer_plan_length = 0

            if viewer is not None:
                if viewer.is_running():
                    viewer.sync()
                    remaining = viewer_deadline - time.perf_counter()
                    if remaining > 0.0:
                        time.sleep(remaining)
                    viewer_deadline += env.control_dt

            next_states.append(true_state.copy())
            if args.controller_mode == "mpc" and np.all(np.isfinite(predicted_next_state)):
                predicted_next_q_errors.append(float(np.linalg.norm(predicted_next_state[: args.n_joints] - true_state[: args.n_joints])))
                predicted_next_dq_errors.append(
                    float(np.linalg.norm(predicted_next_state[args.n_joints :] - true_state[args.n_joints :]))
                )
            else:
                predicted_next_q_errors.append(float("nan"))
                predicted_next_dq_errors.append(float("nan"))
            sampling_std_start_means.append(sampling_std_start_mean)
            sampling_std_end_means.append(sampling_std_end_mean)
            previous_q_ref_velocity = (q_ref_command - previous_q_ref) / env.control_dt
            previous_q_ref = q_ref_command.copy()
            commit_command_and_append_placeholder(states_history, q_ref_history, q_ref_command, state)
            realized_error = float(np.linalg.norm(true_state[: args.n_joints] - reference[step_idx + 1]))
            realized_tracking_errors.append(realized_error)
            if args.controller_mode == "mpc" and args.mpc_policy == "residual":
                previous_residual_velocity = (executed_residual - previous_residual) / env.control_dt
                previous_residual = executed_residual.copy()
                residual_saturated = bool(
                    np.any(np.abs(executed_residual) >= args.recovery_residual_fraction * residual_max)
                )
                residual_saturation_streak = residual_saturation_streak + 1 if residual_saturated else 0
                if not recovery_trigger_reason and not recovery_active:
                    recovery_trigger_reason = residual_recovery_reason(
                        realized_tracking_errors,
                        residual_saturation_streak=residual_saturation_streak,
                        consecutive_steps=args.recovery_consecutive_steps,
                        error_ratio=args.recovery_error_ratio,
                        min_tracking_error=args.recovery_min_tracking_error,
                        recovery_active=False,
                    )
                    if recovery_trigger_reason:
                        recovery_remaining = args.recovery_cooldown_steps
                        if controller is not None:
                            controller.reset()
                        command_buffer = np.empty((0, args.n_joints), dtype=np.float32)
                        residual_buffer = np.empty((0, args.n_joints), dtype=np.float32)
                        command_buffer_index = command_buffer_plan_length = 0
                recovery_trigger_reasons.append(recovery_trigger_reason)
                recovery_active_flags.append(recovery_active)
            else:
                recovery_trigger_reasons.append("")
                recovery_active_flags.append(0)
            row: dict[str, Any] = {
                "step": step_idx,
                "controller_mode": args.controller_mode,
                "tracking_error": realized_error,
                "planning_time": planning_time,
                "replan_time": replan_time,
                "mpc_replanned": mpc_replanned,
                "buffer_index": buffer_index,
                "buffer_length": buffer_length,
                "replan_deadline_miss": replan_deadline_miss,
                "control_step_wall_time": control_step_wall_times[-1],
                "best_cost": best_cost,
                "mean_cost": mean_cost,
                "baseline_cost": baseline_cost,
                "selected_cost": selected_cost,
                "elite_mean_cost": elite_mean_cost,
                "selection_mode": selection_mode,
                "failure": failure,
                "failure_reason": failure_reason,
                "joint_limit_violation": joint_limit_violations[-1],
                "command_velocity_violation": command_velocity_violations[-1],
                "command_acceleration_violation": command_acceleration_violations[-1],
                "predicted_next_q_error": predicted_next_q_errors[-1],
                "predicted_next_dq_error": predicted_next_dq_errors[-1],
                "sampling_std_start_mean": sampling_std_start_means[-1],
                "sampling_std_end_mean": sampling_std_end_means[-1],
                "recovery_active": recovery_active_flags[-1],
                "recovery_trigger_reason": recovery_trigger_reasons[-1],
                "multirate_buffer_mode": multirate_buffer_mode,
                "buffered_residual_norm": float(np.linalg.norm(buffered_residual)),
                "executed_residual_norm": float(np.linalg.norm(executed_residual)),
                "residual_reanchor_delta_norm": float(np.linalg.norm(executed_residual - buffered_residual)),
            }
            row.update({f"cost_{name}": cost_term_records[name][-1] for name in COST_TERM_NAMES})
            if task_reference is not None:
                row.update(
                    {
                        "ee_position_error": ee_position_errors[-1],
                        "ee_orientation_error": ee_orientation_errors[-1],
                        "segment_id": segment_ids[-1],
                        "lap_id": lap_ids[-1],
                    }
                )
            rows.append(row)
    finally:
        if viewer is not None:
            viewer.close()
        env.close()

    replay_arrays: dict[str, np.ndarray] = {}
    if args.controller_mode == "mpc" and bundle is not None and isinstance(bundle.model, torch.nn.Module):
        replay_arrays = replay_executed_commands(
            model=bundle.model,
            normalizer=bundle.normalizer,
            model_type=bundle.model_type,
            state_dim=bundle.state_dim,
            target_mode=bundle.target_mode,
            control_dt=bundle.control_dt,
            history_len=bundle.history_len,
            states_history=states_history,
            q_ref_history=q_ref_history,
            executed_q_ref=selected_q_refs,
            horizon=args.horizon,
            device=device,
            rollout_batch_size=args.rollout_batch_size,
            command_velocity=command_velocities,
            command_acceleration=command_accelerations,
        )

    arrays: dict[str, np.ndarray] = {
        "controller_mode": np.asarray(args.controller_mode),
        "ik_preview_steps": np.asarray(args.ik_preview_steps, dtype=np.int64),
        "delay_protocol": np.asarray(args.delay_protocol),
        "anticipation_delay_steps": np.asarray(args.anticipation_delay_steps, dtype=np.int64),
        "actual_states": _stack_records(actual_states),
        "observed_states": _stack_records(observed_states),
        "observation_noise": _stack_records(observation_noise),
        "next_states": _stack_records(next_states),
        "q_des": _stack_records(q_des_records),
        "dq_des": _stack_records(dq_des_records),
        "ddq_des": _stack_records([ddq_reference[idx] for idx in range(len(q_des_records))]),
        "actuator_q_ref": _stack_records(selected_q_refs),
        "delta_q_ref": _stack_records(selected_delta_q_refs),
        "command_velocity": _stack_records(command_velocities),
        "command_acceleration": _stack_records(command_accelerations),
        "planning_time": np.asarray(planning_times, dtype=np.float32),
        "replan_time": np.asarray(replan_times, dtype=np.float32),
        "mpc_replanned": np.asarray(mpc_replanned_flags, dtype=np.int64),
        "buffer_index": np.asarray(buffer_indices, dtype=np.int64),
        "buffer_length": np.asarray(buffer_lengths, dtype=np.int64),
        "replan_deadline_miss": np.asarray(replan_deadline_miss_flags, dtype=np.int64),
        "replan_deadline_s": np.asarray(args.replan_interval_steps * env.control_dt, dtype=np.float32),
        "control_step_wall_time": np.asarray(control_step_wall_times, dtype=np.float32),
        "best_cost": np.asarray(best_costs, dtype=np.float32),
        "mean_cost": np.asarray(mean_costs, dtype=np.float32),
        "baseline_cost": np.asarray(baseline_costs, dtype=np.float32),
        "selected_cost": np.asarray(selected_costs, dtype=np.float32),
        "elite_mean_cost": np.asarray(elite_costs, dtype=np.float32),
        "selection_mode": np.asarray(selection_modes),
        "failure_flags": np.asarray(failures, dtype=np.int64),
        "joint_limit_violation_flags": np.asarray(joint_limit_violations, dtype=np.int64),
        "command_velocity_violation_flags": np.asarray(command_velocity_violations, dtype=np.int64),
        "command_acceleration_violation_flags": np.asarray(command_acceleration_violations, dtype=np.int64),
        "realized_tracking_error": np.asarray(realized_tracking_errors, dtype=np.float32),
        "predicted_next_q_error": np.asarray(predicted_next_q_errors, dtype=np.float32),
        "predicted_next_dq_error": np.asarray(predicted_next_dq_errors, dtype=np.float32),
        "sampling_std_start_mean": np.asarray(sampling_std_start_means, dtype=np.float32),
        "sampling_std_end_mean": np.asarray(sampling_std_end_means, dtype=np.float32),
        "nominal_q_ref": _stack_records(nominal_q_refs),
        "buffered_residual": _stack_records(buffered_residuals),
        "executed_residual": _stack_records(executed_residuals),
        "residual_reanchor_delta": _stack_records(residual_reanchor_deltas),
        "multirate_buffer_mode": np.asarray(multirate_buffer_modes),
        "recovery_active_flags": np.asarray(recovery_active_flags, dtype=np.int64),
        "recovery_trigger_reasons": np.asarray(recovery_trigger_reasons),
        "external_force_world": _stack_records(external_force_world_records),
        "external_generalized_force": _stack_records(external_generalized_force_records),
        "cem_reset_std_each_step": np.asarray(args.reset_std_each_step),
        "cem_init_std": np.asarray(args.init_std, dtype=np.float32),
        "cem_min_std": np.asarray(args.min_std, dtype=np.float32),
        "cem_execute": np.asarray(args.cem_execute),
        "cem_force_baseline_candidate": np.asarray(args.mpc_policy == "residual"),
        "cem_uniform_sample_ratio": np.asarray(args.uniform_sample_ratio, dtype=np.float32),
        "cem_uniform_sample_count": np.asarray(
            int(
                round(
                    (args.num_samples - (2 if args.mpc_policy == "residual" else 0)) * args.uniform_sample_ratio
                )
            )
            if args.controller_mode == "mpc"
            else 0,
            dtype=np.int64,
        ),
        "replan_interval_steps": np.asarray(args.replan_interval_steps, dtype=np.int64),
        "mpc_warmup_plans": np.asarray(args.mpc_warmup_plans, dtype=np.int64),
        "mpc_policy": np.asarray(args.mpc_policy),
        "cost_profile": np.asarray(args.cost_profile),
        "cost_temporal_discount": np.asarray(args.temporal_discount, dtype=np.float32),
        "cost_barrier_max_weight": np.asarray(args.barrier_max_weight, dtype=np.float32),
        "cost_w_q": np.asarray(args.w_q, dtype=np.float32),
        "cost_w_dq": np.asarray(args.w_dq, dtype=np.float32),
        "cost_w_residual": np.asarray(args.w_residual, dtype=np.float32),
        "cost_w_servo": np.asarray(args.w_servo, dtype=np.float32),
        "cost_w_residual_velocity": np.asarray(args.w_residual_velocity, dtype=np.float32),
        "cost_w_residual_acceleration": np.asarray(args.w_residual_acceleration, dtype=np.float32),
        "cost_w_first": np.asarray(args.w_first, dtype=np.float32),
        "cost_w_terminal": np.asarray(args.w_terminal, dtype=np.float32),
        "cost_w_joint_limit": np.asarray(args.w_joint_limit, dtype=np.float32),
        "cost_w_dq_limit": np.asarray(args.w_dq_limit, dtype=np.float32),
        "recovery_error_ratio": np.asarray(args.recovery_error_ratio, dtype=np.float32),
        "recovery_min_tracking_error": np.asarray(args.recovery_min_tracking_error, dtype=np.float32),
        "recovery_residual_fraction": np.asarray(args.recovery_residual_fraction, dtype=np.float32),
        "recovery_consecutive_steps": np.asarray(args.recovery_consecutive_steps, dtype=np.int64),
        "recovery_cooldown_steps": np.asarray(args.recovery_cooldown_steps, dtype=np.int64),
        "q_tracking_scale": np.asarray(calibration["q_tracking_scale"], dtype=np.float32) if args.controller_mode == "mpc" else np.empty((0,), dtype=np.float32),
        "dq_tracking_scale": np.asarray(calibration["dq_tracking_scale"], dtype=np.float32) if args.controller_mode == "mpc" else np.empty((0,), dtype=np.float32),
        "q_ref_velocity_limit": np.asarray(q_ref_velocity_limit, dtype=np.float32) if args.controller_mode == "mpc" else np.empty((0,), dtype=np.float32),
        "q_ref_acceleration_limit": np.asarray(q_ref_acceleration_limit, dtype=np.float32) if args.controller_mode == "mpc" else np.empty((0,), dtype=np.float32),
        "residual_max": np.asarray(residual_max, dtype=np.float32) if args.controller_mode == "mpc" else np.empty((0,), dtype=np.float32),
        "servo_scale": np.asarray(servo_scale, dtype=np.float32) if args.controller_mode == "mpc" else np.empty((0,), dtype=np.float32),
        **{f"cost_{name}": np.asarray(values, dtype=np.float32) for name, values in cost_term_records.items()},
        **replay_arrays,
        **{key: _stack_records(value) for key, value in torque_records.items()},
        **config_arrays(robustness, env),
    }
    pulse_start, pulse_stop = robustness.pulse_window(execution_steps)
    arrays["force_pulse_start_step"] = np.asarray(pulse_start, dtype=np.int64)
    arrays["force_pulse_stop_step"] = np.asarray(pulse_stop, dtype=np.int64)
    if task_reference is not None:
        arrays.update(
            {
                "desired_ee_positions": _stack_records(desired_ee_positions),
                "desired_ee_rotations": _stack_records(desired_ee_rotations),
                "actual_ee_positions": _stack_records(actual_ee_positions),
                "actual_ee_rotations": _stack_records(actual_ee_rotations),
                "ee_position_errors": np.asarray(ee_position_errors, dtype=np.float32),
                "ee_orientation_errors": np.asarray(ee_orientation_errors, dtype=np.float32),
                "segment_ids": _stack_records(segment_ids, dtype=np.int64),
                "lap_ids": _stack_records(lap_ids, dtype=np.int64),
                "execution_steps": np.asarray(execution_steps, dtype=np.int64),
            }
        )
    return {"arrays": arrays, "rows": rows, "failure_reasons": failure_reasons}


def main() -> None:
    args = parse_args()
    result = run_closed_loop_mpc(args)
    save_dir = resolve_runtime_path(args.save_dir)
    save_mpc_run(save_dir, result["arrays"], result["rows"])
    print(f"Saved CEM-MPC rollout to {save_dir}")


if __name__ == "__main__":
    main()
