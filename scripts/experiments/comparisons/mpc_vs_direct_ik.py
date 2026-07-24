#!/usr/bin/env python3
"""Create a paired MPC-versus-Direct-IK comparison from robustness summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


METRICS = ("tcp_rmse_m", "joint_rmse_rad", "orientation_rmse_rad")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mpc_csv", type=Path, required=True)
    parser.add_argument("--direct_ik_csv", type=Path, required=True)
    parser.add_argument("--save_dir", type=Path, required=True)
    parser.add_argument("--bootstrap_samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260723)
    return parser.parse_args()


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator, samples: int) -> list[float]:
    draws = rng.choice(values, size=(samples, len(values)), replace=True).mean(axis=1)
    return [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]


def main() -> None:
    args = parse_args()
    args.save_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = args.save_dir / "plots"
    plot_dir.mkdir(exist_ok=True)

    mpc = pd.read_csv(args.mpc_csv)
    direct = pd.read_csv(args.direct_ik_csv)
    direct = direct[direct["projection"] == "raw"].copy()
    direct["method"] = "DirectIK"
    direct["label"] = "DirectIK"
    methods = ["DirectIK", "IdealZeroDelay", "NaiveDelayed", "VirtualDelayAware", "ThreadedAsync"]
    combined = pd.concat([direct, mpc], ignore_index=True, sort=False)

    aggregate = (
        combined.groupby(["method", "reference_type"], sort=False)[list(METRICS)]
        .mean()
        .reset_index()
    )
    pooled = combined.groupby("method", sort=False)[list(METRICS)].mean().reset_index()
    pooled["reference_type"] = "pooled"
    aggregate = pd.concat([aggregate, pooled], ignore_index=True)
    aggregate["method"] = pd.Categorical(aggregate["method"], methods, ordered=True)
    aggregate = aggregate.sort_values(["reference_type", "method"]).reset_index(drop=True)
    aggregate.to_csv(args.save_dir / "mpc_vs_direct_ik_summary.csv", index=False)

    rng = np.random.default_rng(args.seed)
    comparisons: dict[str, object] = {
        "pairing": "case_id",
        "baseline": "DirectIK",
        "bootstrap_samples": args.bootstrap_samples,
        "comparisons": {},
    }
    direct_by_case = direct.set_index("case_id")
    for method in methods[1:]:
        subset = mpc[mpc["method"] == method].set_index("case_id")
        case_ids = sorted(set(direct_by_case.index) & set(subset.index))
        method_result: dict[str, object] = {"pairs": len(case_ids), "metrics": {}}
        for metric in METRICS:
            delta = (
                subset.loc[case_ids, metric].to_numpy(dtype=float)
                - direct_by_case.loc[case_ids, metric].to_numpy(dtype=float)
            )
            method_result["metrics"][metric] = {
                "mean_delta": float(delta.mean()),
                "ci95": bootstrap_ci(delta, rng, args.bootstrap_samples),
                "relative_delta_percent": float(
                    100.0 * delta.mean() / direct_by_case.loc[case_ids, metric].mean()
                ),
            }
        comparisons["comparisons"][method] = method_result
    (args.save_dir / "mpc_vs_direct_ik_paired.json").write_text(
        json.dumps(comparisons, indent=2) + "\n", encoding="utf-8"
    )

    plot_data = aggregate[aggregate["reference_type"].isin(["circle", "figure8", "pooled"])]
    fig, axis = plt.subplots(figsize=(10, 5))
    x = np.arange(3)
    width = 0.16
    for index, method in enumerate(methods):
        rows = plot_data[plot_data["method"] == method].set_index("reference_type")
        values = [rows.loc[item, "tcp_rmse_m"] * 1000.0 for item in ("circle", "figure8", "pooled")]
        axis.bar(x + (index - 2) * width, values, width, label=method)
    axis.set_xticks(x, ["Circle", "Figure-8", "Pooled"])
    axis.set_ylabel("TCP RMSE [mm]")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(ncols=2)
    fig.tight_layout()
    fig.savefig(plot_dir / "mpc_vs_direct_ik_tcp_rmse.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
