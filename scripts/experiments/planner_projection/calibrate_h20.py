"""Calibrate real threaded E2E delay for one H20 planner-projection variant."""
from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np

from scripts.experiment_utils.hashing import file_identity
from scripts.robustness._runtime import ensure_import_paths, load_runner
from scripts.robustness.evaluate_direct_ik import load_task_cases

ensure_import_paths()


RUNNER = load_runner("planner_projection_h20_calibration_runner")


def parse_args() -> argparse.Namespace:
    parser = RUNNER.build_arg_parser()
    parser.add_argument("--manifest", default="outputs/robustness_h20_d10/benchmark.json")
    parser.add_argument("--case_ids", default="circle_00,figure8_00")
    parser.add_argument("--plans", default=500, type=int)
    parser.add_argument("--calibration_delay", default=10, type=int)
    parser.add_argument("--output_path", required=True)
    parser.set_defaults(
        checkpoint="dynamics_modeling/outputs/checkpoints/gru_20260717_182930/best_model.pt",
        normalizer="dynamics_modeling/outputs/checkpoints/gru_20260717_182930/normalizer.pt",
        model_type="gru",
        history_len=16,
        horizon=20,
        num_samples=128,
        cem_iters=2,
        rollout_batch_size=128,
        multirate_mode="threaded_asap",
        delay_protocol="full",
        device="cuda",
        max_execution_steps=500,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.plans <= 0 or args.calibration_delay <= 0:
        raise ValueError("plans and calibration_delay must be positive")
    if args.horizon != 20:
        raise ValueError("planner-projection calibration is frozen to horizon=20")
    case_ids = [value.strip() for value in args.case_ids.split(",") if value.strip()]
    manifest_path = RUNNER.resolve_runtime_path(args.manifest)
    _, cases = load_task_cases(manifest_path, case_ids)
    solve: list[float] = []
    e2e: list[float] = []
    episodes = 0
    late_count = 0
    while len(e2e) < args.plans:
        case = cases[episodes % len(cases)]
        run_args = deepcopy(args)
        for key, value in case["run_args"].items():
            if hasattr(run_args, key):
                setattr(run_args, key, value)
        run_args.checkpoint = args.checkpoint
        run_args.normalizer = args.normalizer
        run_args.model_type = "gru"
        run_args.history_len = 16
        run_args.device = args.device
        run_args.horizon = 20
        run_args.controller_mode = "mpc"
        run_args.mpc_policy = "residual"
        run_args.reference_mode = "task"
        run_args.multirate_mode = "threaded_asap"
        run_args.delay_protocol = "full"
        run_args.anticipation_delay_steps = args.calibration_delay
        run_args.max_execution_steps = 500
        run_args.planner_projection = args.planner_projection
        run_args.planner_projection_backend = args.planner_projection_backend
        run_args.planner_projection_strategy = args.planner_projection_strategy
        run_args.seed = 20260723 + episodes
        run_args.visualize = False
        result = RUNNER.run_closed_loop_mpc(run_args)
        arrays = result["arrays"]
        replanned = np.asarray(arrays["mpc_replanned"], dtype=bool)
        solve_values = np.asarray(arrays["planning_time"], dtype=np.float64)
        e2e_values = np.asarray(arrays["planner_end_to_end_latency_s"], dtype=np.float64)
        valid = replanned & np.isfinite(e2e_values)
        solve.extend(solve_values[valid & np.isfinite(solve_values)].tolist())
        e2e.extend(e2e_values[valid].tolist())
        late = np.asarray(arrays["packet_late_dropped"], dtype=np.int64)
        late_count += int(np.sum(late[valid] != 0))
        episodes += 1
        print(f"episode={episodes} collected={min(len(e2e), args.plans)}/{args.plans}")
    solve_array = np.asarray(solve[: args.plans], dtype=np.float64)
    e2e_array = np.asarray(e2e[: args.plans], dtype=np.float64)
    guard_s = float(args.planner_guard_ms) / 1000.0
    control_dt = 0.01
    delay = int(np.ceil((float(np.percentile(e2e_array, 95)) + guard_s) / control_dt))

    def stats(values: np.ndarray) -> dict[str, float]:
        return {
            "p50_s": float(np.percentile(values, 50)),
            "p95_s": float(np.percentile(values, 95)),
            "p99_s": float(np.percentile(values, 99)),
            "max_s": float(np.max(values)),
        }

    payload = {
        "schema_version": 1,
        "kind": "threaded_e2e_delay_calibration",
        "horizon": 20,
        "num_samples": args.num_samples,
        "cem_iters": args.cem_iters,
        "plans": args.plans,
        "episodes": episodes,
        "case_ids": case_ids,
        "planner_projection": args.planner_projection,
        "planner_projection_backend": args.planner_projection_backend,
        "planner_projection_strategy": args.planner_projection_strategy,
        "calibration_delay_steps": args.calibration_delay,
        "solve_latency": stats(solve_array),
        "end_to_end_latency": stats(e2e_array),
        "guard_s": guard_s,
        "control_dt_s": control_dt,
        "anticipation_delay_steps": delay,
        "calibration_late_count": late_count,
        "manifest": file_identity(manifest_path),
        "checkpoint": file_identity(RUNNER.resolve_runtime_path(args.checkpoint)),
        "normalizer": file_identity(RUNNER.resolve_runtime_path(args.normalizer)),
    }
    output = RUNNER.resolve_runtime_path(args.output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"E2E p95={payload['end_to_end_latency']['p95_s'] * 1e3:.2f} ms; "
        f"D={delay}; saved {output}"
    )


if __name__ == "__main__":
    main()
