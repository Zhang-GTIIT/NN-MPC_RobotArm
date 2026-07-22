from __future__ import annotations

from typing import Any

import numpy as np


def paired_bootstrap_rows(
    rows: list[dict[str, Any]], *, left: str, right: str, metrics: tuple[str, ...],
    samples: int, seed: int, key_field: str = "case_id", label_field: str = "label",
) -> dict[str, Any]:
    if samples <= 0:
        raise ValueError("samples must be positive")
    indexed = {(str(row[label_field]), str(row[key_field])): row for row in rows}
    cases = sorted({key for label, key in indexed if label == left} & {key for label, key in indexed if label == right})
    rng = np.random.default_rng(seed)
    report: dict[str, Any] = {"left": left, "right": right, "paired_cases": cases, "metrics": {}}
    for metric in metrics:
        delta = np.asarray([
            float(indexed[(right, case)][metric]) - float(indexed[(left, case)][metric])
            for case in cases
            if np.isfinite(float(indexed[(left, case)].get(metric, np.nan)))
            and np.isfinite(float(indexed[(right, case)].get(metric, np.nan)))
        ], dtype=np.float64)
        if not delta.size:
            report["metrics"][metric] = {"n": 0, "mean_delta_right_minus_left": float("nan"), "ci95": [float("nan"), float("nan")]}
            continue
        draws = delta[rng.integers(0, delta.size, size=(samples, delta.size))].mean(axis=1)
        report["metrics"][metric] = {
            "n": int(delta.size), "mean_delta_right_minus_left": float(delta.mean()),
            "ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
        }
    return report
