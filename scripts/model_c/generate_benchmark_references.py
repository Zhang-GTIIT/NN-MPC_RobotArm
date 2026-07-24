"""Generate deterministic, immutable task-space references for Model-C benchmarks."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.model_c._runtime import ROOT, ensure_import_paths

ensure_import_paths()

import mujoco
import numpy as np

from mpc.reference_pipeline import ReferenceConfig, build_reference, save_reference_bundle
from mpc.reference import finite_difference_dq, generate_joint_reference

TASK_TYPES = ("circle", "figure8", "ellipse", "square")
JOINT_TYPES = ("multi_joint_sine", "waypoint", "chirp")


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate fixed ID/OOD task-space benchmark references.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_xml", default="dynamics_modeling/ABB_IRB2400.xml")
    parser.add_argument("--cases_per_type", type=int, default=10)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--delay", type=int, required=True)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.cases_per_type <= 0:
        raise ValueError("--cases_per_type must be positive")
    output = _resolve(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    model = mujoco.MjModel.from_xml_path(str(_resolve(args.model_xml)))
    rng = np.random.default_rng(args.seed)
    manifest: list[dict[str, object]] = []
    planes = [((1., 0., 0.), (0., 1., 0.)), ((1., 0., 0.), (0., 0., 1.)), ((0., 1., 0.), (0., 0., 1.))]
    joint_low = np.asarray(model.jnt_range[: model.nu, 0], dtype=np.float32)
    joint_high = np.asarray(model.jnt_range[: model.nu, 1], dtype=np.float32)
    for shape in JOINT_TYPES:
        for index in range(args.cases_per_type):
            destination = output / f"{shape}_{index:02d}"
            reference_path = destination / "reference.npz"
            if reference_path.exists() and not args.overwrite:
                manifest.append({"shape": shape, "index": index, "path": str(reference_path), "reused": True})
                continue
            execution_steps = 500
            padding = args.horizon + args.delay + 1
            # Keep the two immutable benchmark sets disjoint.  Adding the
            # set seed to a small per-case offset is unsafe: final(seed+1,
            # case 0) would then reproduce development(seed, case 1).
            # Allocate a non-overlapping seed block to each benchmark set.
            joint_seed = int(args.seed) * 1_000_000 + 100_000 + index + 10_000 * JOINT_TYPES.index(shape)
            q_execution = generate_joint_reference(
                shape, np.zeros(model.nu, dtype=np.float32), joint_low, joint_high,
                execution_steps, 0.01, seed=joint_seed,
            )
            q_des = np.concatenate([q_execution, np.repeat(q_execution[-1:], padding, axis=0)], axis=0)
            dq_des = finite_difference_dq(q_des, 0.01)
            ddq_des = finite_difference_dq(dq_des, 0.01)
            dq_des[execution_steps - 1 :] = 0.0
            ddq_des[execution_steps - 1 :] = 0.0
            destination.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(reference_path, q_des=q_des, dq_des=dq_des, ddq_des=ddq_des,
                                execution_steps=np.asarray(execution_steps, dtype=np.int64), control_dt=np.asarray(0.01, dtype=np.float32))
            manifest.append({"shape": shape, "index": index, "path": str(reference_path), "seed": joint_seed})
    for shape in TASK_TYPES:
        for index in range(args.cases_per_type):
            destination = output / f"{shape}_{index:02d}"
            reference_path = destination / "reference.npz"
            if reference_path.exists() and not args.overwrite:
                manifest.append({"shape": shape, "index": index, "path": str(reference_path), "reused": True})
                continue
            axis_u, axis_v = planes[int(rng.integers(len(planes)))]
            if rng.integers(2):
                axis_v = tuple(-item for item in axis_v)
            # Circle/figure-8 are ID trajectory types with *unseen* parameter
            # ranges.  Ellipse/square are held-out OOD types throughout C data
            # collection, not merely different random seeds.
            kwargs: dict[str, float] = {}
            if shape == "circle":
                kwargs["circle_radius"] = float(rng.uniform(0.045, 0.060))
            elif shape == "figure8":
                kwargs["figure8_axis_a"] = float(rng.uniform(0.045, 0.060))
                kwargs["figure8_axis_b"] = float(rng.uniform(0.026, 0.035))
            elif shape == "ellipse":
                kwargs["ellipse_axis_a"] = float(rng.uniform(0.035, 0.055))
                kwargs["ellipse_axis_b"] = float(rng.uniform(0.018, 0.032))
            else:
                kwargs["square_half_side"] = float(rng.uniform(0.020, 0.032))
            config = ReferenceConfig(
                shape_name=shape, repeat_count=1, collection_only=True, safe_departure_mode="always",
                start_hold_duration=0.2, joint_departure_duration=0.79, approach_duration=0.79,
                # Square construction stores both endpoints at four corners;
                # 2.96 s produces four 75-sample edges (300 loop samples).
                lap_duration=2.96 if shape == "square" else 2.99, shape_end_hold_duration=0.2,
                center_offset=tuple(rng.uniform(-0.02, 0.02, size=3)), plane_axis_u=axis_u, plane_axis_v=axis_v,
                **kwargs,
            )
            bundle = build_reference(config, model, np.zeros(model.nu), 0.01, args.horizon, args.delay)
            if bundle.execution_steps != 500:
                raise RuntimeError(f"{shape}_{index:02d} has {bundle.execution_steps} execution steps, expected 500")
            destination.mkdir(parents=True, exist_ok=True)
            path = save_reference_bundle(bundle, destination)
            manifest.append({"shape": shape, "index": index, "path": str(path), "config": bundle.metadata["config"]})
    (output / "manifest.json").write_text(json.dumps({"seed": args.seed, "cases": manifest}, indent=2) + "\n", encoding="utf-8")
    print(f"Generated/reused {len(manifest)} task references in {output}")


if __name__ == "__main__":
    main()
