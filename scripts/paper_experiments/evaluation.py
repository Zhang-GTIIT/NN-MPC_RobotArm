from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


def _finite(values: Any) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    return array[np.isfinite(array)]


def _percentile(values: Any, q: float) -> float:
    finite = _finite(values)
    return float(np.percentile(finite, q)) if finite.size else float("nan")


def _rms(values: Any) -> float:
    finite = _finite(values)
    return float(np.sqrt(np.mean(np.square(finite)))) if finite.size else float("nan")


def summarize_arrays(label: str, arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    actual = np.asarray(arrays.get("actual_states", np.empty((0, 0))), dtype=np.float64)
    desired = np.asarray(arrays.get("q_des", np.empty((0, 0))), dtype=np.float64)
    n = min(len(actual), len(desired)) if actual.ndim == desired.ndim == 2 else 0
    joint_error = actual[:n, : desired.shape[1]] - desired[:n] if n and actual.shape[1] >= desired.shape[1] else np.empty(0)
    tcp = _finite(arrays.get("ee_position_errors", np.empty(0)))
    planning = np.asarray(arrays.get("planning_time", np.empty(0)), dtype=np.float64)
    replanned = np.asarray(arrays.get("mpc_replanned", np.empty(0)), dtype=bool)
    solves = planning[replanned] if replanned.shape == planning.shape else planning
    packet_age = np.asarray(arrays.get("packet_age", np.empty(0)), dtype=np.float64)
    solve_count = int(np.asarray(arrays.get("planner_solve_count", np.sum(replanned))).reshape(-1)[0])
    late_count = int(np.asarray(arrays.get("planner_late_drop_count", 0)).reshape(-1)[0])
    failure_flags = np.asarray(arrays.get("failure_flags", np.empty(0))) != 0
    scalar = lambda key, default="": np.asarray(arrays.get(key, default)).reshape(-1)[0]
    return {
        "label": label,
        "delay_protocol": str(scalar("delay_protocol", "not_applicable")),
        "multirate_mode": str(scalar("multirate_mode", "synchronous")),
        "delay_steps": int(scalar("anticipation_delay_steps", 0)),
        "steps": int(len(np.asarray(arrays.get("actuator_q_ref", np.empty(0))))),
        "tcp_rmse_m": float(np.sqrt(np.mean(np.square(tcp)))) if tcp.size else float("nan"),
        "tcp_p95_m": float(np.percentile(tcp, 95)) if tcp.size else float("nan"),
        "orientation_rmse_rad": _rms(arrays.get("ee_orientation_errors", np.empty(0))),
        "joint_rmse_rad": _rms(joint_error),
        # A row is one trajectory-seed case.  Keep the paper failure metric
        # binary and expose transient planner fallbacks separately; 10 ms
        # ticks are not independent trials.
        "failure_rate": float(np.any(failure_flags)) if failure_flags.size else 0.0,
        "planner_failure_step_rate": float(np.mean(failure_flags)) if failure_flags.size else 0.0,
        "command_acceleration_rms_rad_s2": _rms(arrays.get("command_acceleration", np.empty(0))),
        "actuator_torque_rms_nm": _rms(arrays.get("tau_actuator", np.empty(0))),
        "total_torque_rms_nm": _rms(arrays.get("tau_total", np.empty(0))),
        "residual_rms_rad": _rms(arrays.get("executed_residual", np.empty(0))),
        "projection_discrepancy_rms_rad": _rms(arrays.get("projection_discrepancy", np.empty(0))),
        "projection_activation_rate": float(np.mean(np.asarray(arrays.get("projection_active", np.empty(0))) != 0)) if np.asarray(arrays.get("projection_active", np.empty(0))).size else float("nan"),
        "residual_saturation_rate": float(np.mean(np.asarray(arrays.get("residual_saturated", np.empty(0))) != 0)) if np.asarray(arrays.get("residual_saturated", np.empty(0))).size else float("nan"),
        "feedback_saturation_rate": float(np.mean(np.asarray(arrays.get("feedback_saturated", np.empty(0))) != 0)) if np.asarray(arrays.get("feedback_saturated", np.empty(0))).size else float("nan"),
        "joint_violation_count": int(np.sum(np.asarray(arrays.get("joint_limit_violation_flags", np.empty(0))) != 0)),
        "velocity_violation_count": int(np.sum(np.asarray(arrays.get("command_velocity_violation_flags", np.empty(0))) != 0)),
        "acceleration_violation_count": int(np.sum(np.asarray(arrays.get("command_acceleration_violation_flags", np.empty(0))) != 0)),
        "solve_p50_s": _percentile(solves, 50), "solve_p95_s": _percentile(solves, 95), "solve_p99_s": _percentile(solves, 99),
        "e2e_p50_s": _percentile(arrays.get("planner_end_to_end_latency_s", np.empty(0)), 50),
        "e2e_p95_s": _percentile(arrays.get("planner_end_to_end_latency_s", np.empty(0)), 95),
        "e2e_p99_s": _percentile(arrays.get("planner_end_to_end_latency_s", np.empty(0)), 99),
        "planner_hz": float(scalar("planner_actual_update_rate_hz", np.nan)),
        "late_packet_rate": float(late_count / solve_count) if solve_count else float("nan"),
        "active_packet_ratio": float(np.mean(packet_age >= 0)) if packet_age.size else float("nan"),
        "control_compute_p99_s": _percentile(arrays.get("control_step_wall_time", np.empty(0)), 99),
        "control_period_p99_s": _percentile(arrays.get("actual_control_period_s", np.empty(0)), 99),
        "wakeup_lateness_p99_s": _percentile(arrays.get("control_wakeup_lateness_s", np.empty(0)), 99),
        "start_jitter_p99_s": _percentile(arrays.get("control_start_jitter_s", np.empty(0)), 99),
        "control_deadline_miss_count": int(np.sum(np.asarray(arrays.get("control_deadline_miss", np.empty(0))) != 0)),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_rows(rows: list[dict[str, Any]], group_fields: tuple[str, ...]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(tuple(row.get(field) for field in group_fields), []).append(row)
    metrics = (
        "tcp_rmse_m", "tcp_p95_m", "orientation_rmse_rad", "joint_rmse_rad",
        "failure_rate", "planner_failure_step_rate",
        "command_acceleration_rms_rad_s2", "actuator_torque_rms_nm",
        "projection_discrepancy_rms_rad", "projection_activation_rate",
        "residual_saturation_rate", "feedback_saturation_rate", "e2e_p95_s",
        "planner_hz", "late_packet_rate", "control_deadline_miss_count",
    )
    output: list[dict[str, Any]] = []
    for key, members in sorted(groups.items(), key=lambda item: tuple(map(str, item[0]))):
        aggregate = {field: value for field, value in zip(group_fields, key)}
        aggregate["n_cases"] = len(members)
        for metric in metrics:
            values = np.asarray([float(member.get(metric, np.nan)) for member in members], dtype=np.float64)
            finite = values[np.isfinite(values)]
            aggregate[f"{metric}_mean"] = float(np.mean(finite)) if finite.size else float("nan")
            aggregate[f"{metric}_std"] = float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0 if finite.size == 1 else float("nan")
        output.append(aggregate)
    return output


def latency_recovery(rows: list[dict[str, Any]], epsilon_m: float = 1e-6) -> dict[str, Any]:
    grouped: dict[tuple[str, int], dict[str, float]] = {}
    for row in rows:
        key = (str(row.get("trajectory")), int(row.get("seed", -1)))
        grouped.setdefault(key, {})[str(row["label"])] = float(row["tcp_rmse_m"])
    values: list[float] = []
    for methods in grouped.values():
        if {"NaiveDelayed", "FullVirtual", "IdealZeroDelay"}.issubset(methods):
            denominator = methods["NaiveDelayed"] - methods["IdealZeroDelay"]
            if denominator > epsilon_m:
                values.append((methods["NaiveDelayed"] - methods["FullVirtual"]) / denominator)
    return {"epsilon_m": epsilon_m, "n": len(values), "mean": float(np.mean(values)) if values else float("nan"), "values": values}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=True, default=str) + "\n", encoding="utf-8")
