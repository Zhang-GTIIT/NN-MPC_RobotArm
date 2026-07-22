"""Generate independent fixed references for Model-A robustness evaluation."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.robustness._runtime import ROOT, ensure_import_paths

ensure_import_paths()

import mujoco
import numpy as np

from mpc.reference import finite_difference_dq, generate_joint_reference
from mpc.reference_pipeline import ReferenceConfig, build_reference, save_reference_bundle


TASK_TYPES = ("circle", "figure8", "ellipse", "square")
JOINT_TYPES = ("multi_joint_sine", "waypoint", "chirp")


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate fixed Model-A robustness benchmark references.")
    parser.add_argument("--output_dir", default="outputs/robustness/references")
    parser.add_argument("--model_xml", default="dynamics_modeling/ABB_IRB2400.xml")
    parser.add_argument("--cases_per_type", type=int, default=10)
    parser.add_argument("--horizon", type=int, default=25)
    parser.add_argument("--delay", required=True, type=int)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.cases_per_type <= 0 or args.horizon <= 0 or args.delay <= 0:
        raise ValueError("--cases_per_type, --horizon, and --delay must be positive")
    output = resolve(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    model = mujoco.MjModel.from_xml_path(str(resolve(args.model_xml)))
    rng = np.random.default_rng(args.seed)
    joint_low = np.asarray(model.jnt_range[:model.nu, 0], dtype=np.float32)
    joint_high = np.asarray(model.jnt_range[:model.nu, 1], dtype=np.float32)
    generated: list[dict[str, object]] = []
    padding = args.horizon + args.delay + 1
    for type_index, shape in enumerate(JOINT_TYPES):
        for index in range(args.cases_per_type):
            destination, path = output / f"{shape}_{index:02d}", output / f"{shape}_{index:02d}" / "reference.npz"
            if path.exists() and not args.overwrite:
                generated.append({"type": shape, "index": index, "path": str(path), "reused": True})
                continue
            seed = args.seed * 1_000_000 + 100_000 + type_index * 10_000 + index
            execution = generate_joint_reference(shape, np.zeros(model.nu, dtype=np.float32), joint_low, joint_high, 500, 0.01, seed=seed)
            q_des = np.concatenate((execution, np.repeat(execution[-1:], padding, axis=0)))
            dq_des, ddq_des = finite_difference_dq(q_des, .01), None
            ddq_des = finite_difference_dq(dq_des, .01)
            dq_des[499:] = 0.0
            ddq_des[499:] = 0.0
            destination.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, q_des=q_des, dq_des=dq_des, ddq_des=ddq_des,
                                execution_steps=np.asarray(500, dtype=np.int64), control_dt=np.asarray(.01, dtype=np.float32))
            generated.append({"type": shape, "index": index, "path": str(path), "seed": seed})
    planes = [((1., 0., 0.), (0., 1., 0.)), ((1., 0., 0.), (0., 0., 1.)), ((0., 1., 0.), (0., 0., 1.))]
    for shape in TASK_TYPES:
        for index in range(args.cases_per_type):
            destination, path = output / f"{shape}_{index:02d}", output / f"{shape}_{index:02d}" / "reference.npz"
            if path.exists() and not args.overwrite:
                generated.append({"type": shape, "index": index, "path": str(path), "reused": True})
                continue
            axis_u, axis_v = planes[int(rng.integers(len(planes)))]
            if rng.integers(2):
                axis_v = tuple(-value for value in axis_v)
            shape_kwargs: dict[str, float] = {}
            if shape == "circle":
                shape_kwargs["circle_radius"] = float(rng.uniform(.045, .060))
            elif shape == "figure8":
                shape_kwargs.update(figure8_axis_a=float(rng.uniform(.045, .060)), figure8_axis_b=float(rng.uniform(.026, .035)))
            elif shape == "ellipse":
                shape_kwargs.update(ellipse_axis_a=float(rng.uniform(.035, .055)), ellipse_axis_b=float(rng.uniform(.018, .032)))
            else:
                shape_kwargs["square_half_side"] = float(rng.uniform(.020, .032))
            config = ReferenceConfig(shape_name=shape, repeat_count=1, collection_only=True, safe_departure_mode="always",
                                     start_hold_duration=.2, joint_departure_duration=.79, approach_duration=.79,
                                     lap_duration=2.96 if shape == "square" else 2.99, shape_end_hold_duration=.2,
                                     center_offset=tuple(rng.uniform(-.02, .02, size=3)), plane_axis_u=axis_u, plane_axis_v=axis_v,
                                     **shape_kwargs)
            bundle = build_reference(config, model, np.zeros(model.nu), .01, args.horizon, args.delay)
            if bundle.execution_steps != 500:
                raise RuntimeError(f"{shape}_{index:02d} has {bundle.execution_steps} steps, expected 500")
            destination.mkdir(parents=True, exist_ok=True)
            generated.append({"type": shape, "index": index, "path": str(save_reference_bundle(bundle, destination)), "config": bundle.metadata["config"]})
    (output / "manifest.json").write_text(json.dumps({"seed": args.seed, "cases": generated}, indent=2) + "\n", encoding="utf-8")
    print(f"Generated/reused {len(generated)} robustness references in {output}")


if __name__ == "__main__":
    main()
