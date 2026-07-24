"""Measure virtual CEM planning latency and freeze an anticipation delay."""
from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.model_c._runtime import load_runner

RUNNER = load_runner("model_c_delay_calibration_runner")


def parse_args() -> argparse.Namespace:
    parser = RUNNER.build_arg_parser()
    parser.add_argument("--plans", default=500, type=int)
    parser.add_argument("--output_path", default="outputs/model_c/timing.json")
    parser.set_defaults(model_type="gru", history_len=16, horizon=20, num_samples=128, cem_iters=2, replan_interval_steps=5, multirate_mode="virtual_asap", episode_len=500)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.plans <= 0:
        raise ValueError("--plans must be positive")
    times: list[float] = []
    run_index = 0
    while len(times) < args.plans:
        run_args = deepcopy(args)
        run_args.seed = args.seed + run_index
        result = RUNNER.run_closed_loop_mpc(run_args)
        replanned = np.asarray(result["arrays"]["mpc_replanned"], dtype=bool)
        values = np.asarray(result["arrays"]["planning_time"], dtype=np.float64)
        times.extend(values[replanned & np.isfinite(values)].tolist())
        run_index += 1
    values = np.asarray(times[: args.plans], dtype=np.float64)
    p95 = float(np.percentile(values, 95.0))
    delay = int(np.ceil((p95 + 0.005) / 0.01))
    output = RUNNER.resolve_runtime_path(args.output_path); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"plans": len(values), "planning_time_p50_s": float(np.percentile(values, 50)), "planning_time_p95_s": p95, "guard_s": 0.005, "control_dt_s": 0.01, "anticipation_delay_steps": delay}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"p95={p95 * 1e3:.2f} ms; anticipation_delay_steps={delay}; saved {output}")


if __name__ == "__main__":
    main()
