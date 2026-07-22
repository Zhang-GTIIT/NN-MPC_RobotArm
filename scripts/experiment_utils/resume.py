from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts.experiment_utils.hashing import canonical_sha256


def run_fingerprint(payload: dict[str, Any], *, schema_version: int = 2) -> dict[str, Any]:
    wrapped = {"schema_version": schema_version, **payload}
    return {"sha256": canonical_sha256(wrapped), "payload": wrapped}


def load_completed_rollout(run_dir: Path, expected: dict[str, Any]) -> dict[str, np.ndarray] | None:
    rollout = run_dir / "rollout.npz"
    fingerprint = run_dir / "run_fingerprint.json"
    if not rollout.exists() and not fingerprint.exists():
        return None
    if not rollout.exists() or not fingerprint.exists():
        raise RuntimeError(f"Incomplete resumable output at {run_dir}")
    actual = json.loads(fingerprint.read_text(encoding="utf-8"))
    if actual.get("sha256") != expected.get("sha256"):
        raise RuntimeError(f"Resume fingerprint mismatch at {run_dir}")
    with np.load(rollout, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}
