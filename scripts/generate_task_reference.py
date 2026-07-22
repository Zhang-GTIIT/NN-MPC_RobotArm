"""Generate and validate an offline TCP trajectory reference for CEM-MPC."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DYNAMICS_ROOT = ROOT / "dynamics_modeling"
for import_path in (ROOT, DYNAMICS_ROOT):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

import mujoco
import numpy as np

from mpc.ik_solver import IKConfig
from mpc.reference_pipeline import (
    ReferenceConfig,
    build_reference,
    reference_summary,
    save_reference_bundle,
)
from mpc.task_space_reference import SEGMENT_NAMES


def resolve_runtime_path(value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return ROOT / path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a validated ABB IRB2400 task-space reference.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model_xml", default="dynamics_modeling/ABB_IRB2400.xml")
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--shape", choices=["circle", "ellipse", "figure8", "square", "rounded_square"], default="circle")
    parser.add_argument("--repeat_count", type=int, default=3)
    parser.add_argument("--control_dt", type=float, default=0.01)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--lookahead_steps", type=int, default=0, help="Additional delayed-MPC reference padding steps.")
    parser.add_argument("--ee_site_name", default="ee_site")

    parser.add_argument("--start_hold_duration", type=float, default=0.5)
    parser.add_argument("--joint_departure_duration", type=float, default=2.0)
    parser.add_argument("--approach_duration", type=float, default=2.0)
    parser.add_argument("--lap_duration", type=float, default=4.0)
    parser.add_argument("--return_duration", type=float, default=2.0)
    parser.add_argument("--joint_return_duration", type=float, default=2.0)
    parser.add_argument("--final_hold_duration", type=float, default=0.5)
    parser.add_argument("--collection_only", action="store_true")
    parser.add_argument("--shape_end_hold_duration", type=float, default=0.2)

    parser.add_argument("--center_mode", choices=["relative", "absolute"], default="relative")
    parser.add_argument("--center_offset", type=float, nargs=3, metavar=("X", "Y", "Z"), default=(0.0, 0.0, 0.0))
    parser.add_argument("--plane_axis_u", type=float, nargs=3, metavar=("X", "Y", "Z"), default=(0.0, 1.0, 0.0))
    parser.add_argument("--plane_axis_v", type=float, nargs=3, metavar=("X", "Y", "Z"), default=(0.0, 0.0, 1.0))
    parser.add_argument("--fixed_orientation", choices=["initial", "safe"], default="safe")
    parser.add_argument("--circle_radius", type=float, default=0.1)
    parser.add_argument("--ellipse_axis_a", type=float, default=0.04)
    parser.add_argument("--ellipse_axis_b", type=float, default=0.025)
    parser.add_argument("--figure8_axis_a", type=float, default=0.035)
    parser.add_argument("--figure8_axis_b", type=float, default=0.02)
    parser.add_argument("--square_half_side", type=float, default=0.025)
    parser.add_argument("--rounded_square_corner_radius", type=float, default=0.006)

    parser.add_argument("--safe_departure_mode", choices=["auto", "always", "never"], default="auto")
    parser.add_argument("--safe_sigma_threshold", type=float, default=0.10)
    parser.add_argument("--safe_search_samples", type=int, default=2048)
    parser.add_argument("--safe_search_seed", type=int, default=0)
    parser.add_argument("--safe_joint_limit_margin", type=float, default=0.05)

    parser.add_argument("--ik_damping", type=float, default=1e-2)
    parser.add_argument("--ik_max_iterations", type=int, default=100)
    parser.add_argument("--ik_position_tolerance", type=float, default=1e-4)
    parser.add_argument("--ik_orientation_tolerance", type=float, default=1e-3)
    parser.add_argument("--ik_max_joint_step", type=float, default=0.05)
    parser.add_argument("--ik_step_gain", type=float, default=0.5)
    parser.add_argument("--ik_orientation_weight", type=float, default=0.3)
    parser.add_argument("--ik_joint_limit_margin", type=float, default=0.05)
    parser.add_argument("--ik_sigma_warning", type=float, default=0.02)
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_arg_parser().parse_args(argv)


def _reference_config_from_args(args: argparse.Namespace) -> ReferenceConfig:
    ik_config = IKConfig(
        max_iterations=args.ik_max_iterations,
        position_tolerance=args.ik_position_tolerance,
        orientation_tolerance=args.ik_orientation_tolerance,
        damping=args.ik_damping,
        step_gain=args.ik_step_gain,
        max_joint_step=args.ik_max_joint_step,
        orientation_weight=args.ik_orientation_weight,
        joint_limit_margin=args.ik_joint_limit_margin,
        sigma_warning=args.ik_sigma_warning,
    )
    return ReferenceConfig(
        shape_name=args.shape,
        repeat_count=args.repeat_count,
        start_hold_duration=args.start_hold_duration,
        joint_departure_duration=args.joint_departure_duration,
        approach_duration=args.approach_duration,
        lap_duration=args.lap_duration,
        return_duration=args.return_duration,
        joint_return_duration=args.joint_return_duration,
        final_hold_duration=args.final_hold_duration,
        collection_only=args.collection_only,
        shape_end_hold_duration=args.shape_end_hold_duration,
        center_mode=args.center_mode,
        center_offset=tuple(args.center_offset),
        plane_axis_u=tuple(args.plane_axis_u),
        plane_axis_v=tuple(args.plane_axis_v),
        fixed_orientation=args.fixed_orientation,
        ee_site_name=args.ee_site_name,
        circle_radius=args.circle_radius,
        ellipse_axis_a=args.ellipse_axis_a,
        ellipse_axis_b=args.ellipse_axis_b,
        figure8_axis_a=args.figure8_axis_a,
        figure8_axis_b=args.figure8_axis_b,
        square_half_side=args.square_half_side,
        rounded_square_corner_radius=args.rounded_square_corner_radius,
        safe_departure_mode=args.safe_departure_mode,
        safe_sigma_threshold=args.safe_sigma_threshold,
        safe_search_samples=args.safe_search_samples,
        safe_search_seed=args.safe_search_seed,
        safe_joint_limit_margin=args.safe_joint_limit_margin,
        ik_config=ik_config,
    )


def _save_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=True) + "\n", encoding="utf-8")


def _segment_boundaries(segment_ids: np.ndarray) -> np.ndarray:
    if segment_ids.size <= 1:
        return np.empty(0, dtype=np.int64)
    return np.flatnonzero(np.diff(segment_ids) != 0) + 1


def plot_reference_diagnostics(save_dir: Path, bundle) -> None:
    """Save non-interactive diagnostic plots alongside ``reference.npz``."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    time = bundle.time
    n_joints = bundle.n_joints
    task_mask = np.isin(bundle.segment_ids, (2, 3, 4))
    boundaries = _segment_boundaries(bundle.segment_ids)

    if bundle.task_positions_des is not None:
        figure = plt.figure(figsize=(8, 7))
        axes = figure.add_subplot(111, projection="3d")
        axes.plot(
            bundle.task_positions_des[task_mask, 0],
            bundle.task_positions_des[task_mask, 1],
            bundle.task_positions_des[task_mask, 2],
            linewidth=1.5,
        )
        axes.set_xlabel("x [m]")
        axes.set_ylabel("y [m]")
        axes.set_zlabel("z [m]")
        axes.set_title("Desired TCP trajectory")
        figure.tight_layout()
        figure.savefig(save_dir / "task_trajectory_3d.png", dpi=150)
        plt.close(figure)

    def plot_joint_array(values: np.ndarray, title: str, ylabel: str, filename: str) -> None:
        figure, axes = plt.subplots(n_joints, 1, figsize=(10, 1.8 * n_joints), sharex=True)
        if n_joints == 1:
            axes = [axes]
        for index, axis in enumerate(axes):
            axis.plot(time, values[:, index], linewidth=1.0)
            for boundary in boundaries:
                axis.axvline(time[boundary], color="0.75", linewidth=0.6)
            axis.set_ylabel(f"{ylabel}{index + 1}")
        axes[-1].set_xlabel("time [s]")
        figure.suptitle(title)
        figure.tight_layout()
        figure.savefig(save_dir / filename, dpi=150)
        plt.close(figure)

    plot_joint_array(bundle.q_des, "Joint reference", "q", "joint_reference.png")
    plot_joint_array(bundle.dq_des, "Joint velocity reference", "dq", "joint_velocity.png")
    plot_joint_array(bundle.ddq_des, "Joint acceleration reference", "ddq", "joint_acceleration.png")

    def plot_scalar(values: np.ndarray | None, title: str, ylabel: str, filename: str) -> None:
        if values is None:
            return
        figure, axis = plt.subplots(figsize=(10, 3.5))
        axis.plot(time, values, linewidth=1.0)
        for boundary in boundaries:
            axis.axvline(time[boundary], color="0.75", linewidth=0.6)
        axis.set_title(title)
        axis.set_xlabel("time [s]")
        axis.set_ylabel(ylabel)
        figure.tight_layout()
        figure.savefig(save_dir / filename, dpi=150)
        plt.close(figure)

    plot_scalar(bundle.ik_position_errors, "IK position error", "error [m]", "ik_position_error.png")
    plot_scalar(bundle.ik_orientation_errors, "IK orientation error", "error [rad]", "ik_orientation_error.png")
    plot_scalar(bundle.ik_iterations, "IK iterations", "iterations", "ik_iterations.png")
    plot_scalar(bundle.ik_sigma_min, "TCP Jacobian minimum singular value", "sigma_min", "jacobian_sigma_min.png")
    plot_scalar(bundle.ik_joint_limit_margins, "Joint limit margin", "margin [rad]", "joint_limit_margin.png")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = _reference_config_from_args(args)
    model_path = resolve_runtime_path(args.model_xml)
    save_dir = resolve_runtime_path(args.save_dir)
    model = mujoco.MjModel.from_xml_path(str(model_path))
    bundle = build_reference(
        config=config,
        model=model,
        initial_q=np.zeros(model.nu, dtype=np.float64),
        control_dt=args.control_dt,
        horizon=args.horizon,
        lookahead_steps=args.lookahead_steps,
    )
    save_dir.mkdir(parents=True, exist_ok=True)
    reference_path = save_reference_bundle(bundle, save_dir)
    _save_json(save_dir / "config.json", bundle.metadata["config"])
    _save_json(save_dir / "summary.json", reference_summary(bundle))
    plot_reference_diagnostics(save_dir, bundle)
    segment_names = [SEGMENT_NAMES.get(int(value), str(int(value))) for value in np.unique(bundle.segment_ids)]
    print(
        f"Saved validated {args.shape} reference to {reference_path} "
        f"(execution_steps={bundle.execution_steps}, reference_length={bundle.reference_length}, "
        f"segments={segment_names})"
    )


if __name__ == "__main__":
    main()
