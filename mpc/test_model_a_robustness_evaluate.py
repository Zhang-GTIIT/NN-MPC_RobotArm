from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "test_model_a_robustness_evaluate", ROOT / "scripts" / "robustness" / "evaluate_model_a.py"
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ModelARobustnessEvaluatorTests(unittest.TestCase):
    def test_only_task_cases_receive_direct_ik_and_mpc_is_threaded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            joint_ref, task_ref = root / "joint.npz", root / "task.npz"
            np.savez_compressed(joint_ref, value=np.asarray([1], dtype=np.int64))
            np.savez_compressed(task_ref, value=np.asarray([2], dtype=np.int64))
            checkpoint, normalizer = root / "model.pt", root / "normalizer.pt"
            checkpoint.write_bytes(b"checkpoint")
            normalizer.write_bytes(b"normalizer")
            manifest = root / "benchmark.json"
            manifest.write_text(json.dumps({"kind": "model_a_robustness", "cases": [
                {"id": "joint_00", "reference_type": "multi_joint_sine", "reference_sha256": digest(joint_ref), "run_args": {"reference_mode": "joint_file", "reference_file": str(joint_ref)}},
                {"id": "circle_00", "reference_type": "circle", "reference_sha256": digest(task_ref), "run_args": {"reference_mode": "task", "reference_file": str(task_ref)}},
            ]}), encoding="utf-8")
            arrays = {
                "realized_tracking_error": np.asarray([.1], dtype=np.float32),
                "failure_flags": np.asarray([0], dtype=np.int64), "planning_time": np.asarray([.01], dtype=np.float32),
                "mpc_replanned": np.asarray([1], dtype=np.int64), "actual_states": np.zeros((1, 12), dtype=np.float32),
                "q_des": np.zeros((1, 6), dtype=np.float32), "dynamics_backend": np.asarray("learned"),
            }
            called: list[object] = []

            def fake_run(args):
                called.append(args)
                return {"arrays": arrays, "rows": []}

            def fake_save(path, saved_arrays, rows):
                del rows
                Path(path).mkdir(parents=True, exist_ok=True)
                np.savez_compressed(Path(path) / "rollout.npz", **saved_arrays)

            argv = ["evaluate_model_a.py", "--manifest", str(manifest), "--checkpoint", str(checkpoint),
                    "--normalizer", str(normalizer), "--device", "cpu", "--save_dir", str(root / "runs")]
            with mock.patch.object(sys, "argv", argv), \
                 mock.patch.object(MODULE.RUNNER, "run_closed_loop_mpc", side_effect=fake_run), \
                 mock.patch.object(MODULE, "save_mpc_run", side_effect=fake_save):
                MODULE.main()

            self.assertEqual(len(called), 3)
            mpc_calls = [item for item in called if item.controller_mode == "mpc"]
            ik_calls = [item for item in called if item.controller_mode == "ik_direct"]
            self.assertEqual(len(mpc_calls), 2)
            self.assertEqual(len(ik_calls), 1)
            self.assertTrue(all(item.multirate_mode == "threaded_asap" for item in mpc_calls))
            self.assertEqual(ik_calls[0].multirate_mode, "synchronous")
            self.assertEqual(ik_calls[0].reference_mode, "task")
            summary = (root / "runs" / "model_a_robustness_summary.csv").read_text(encoding="utf-8")
            self.assertIn("ModelA_MPC", summary)
            self.assertIn("DirectIK", summary)

    def test_model_c_multi_spec_and_feedback_flags_are_not_exposed(self) -> None:
        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                MODULE.parse_args(["--model_spec", "C1,x,y,gru"])
            with self.assertRaises(SystemExit):
                MODULE.parse_args(["--include_no_feedback_ablation"])


if __name__ == "__main__":
    unittest.main()
