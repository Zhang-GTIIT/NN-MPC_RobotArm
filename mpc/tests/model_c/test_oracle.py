from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
DYNAMICS_ROOT = ROOT / "dynamics_modeling"
for path in (ROOT, DYNAMICS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from neural_dynamics.mujoco_env import MuJoCoArmEnv
from mpc.cost_functions import JointSpaceCostConfig
from mpc.model_c.oracle import MuJoCoOraclePlanner
from mpc.planner_rollout import PlannerRolloutConfig


def load_script_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RUN_CEM_MPC = load_script_module("oracle_test_run_cem_mpc", ROOT / "scripts" / "run_cem_mpc.py")
EVALUATE = load_script_module("oracle_test_evaluate", ROOT / "scripts" / "model_c" / "evaluate.py")


class OracleMPCTests(unittest.TestCase):
    model_xml = str(DYNAMICS_ROOT / "ABB_IRB2400.xml")

    def _planner(self, env: MuJoCoArmEnv, snapshot: np.ndarray, horizon: int = 2) -> MuJoCoOraclePlanner:
        n = env.n_joints
        device = torch.device("cpu")
        vector = torch.ones(n, dtype=torch.float32, device=device)
        zeros = torch.zeros(n, dtype=torch.float32, device=device)
        nominal = torch.zeros((horizon, n), dtype=torch.float32, device=device)
        cost = JointSpaceCostConfig(
            cost_mode="residual",
            q_tracking_scale=0.05 * vector,
            dq_tracking_scale=0.5 * vector,
            residual_scale=0.05 * vector,
            servo_scale=0.1 * vector,
            residual_velocity_scale=5.0 * vector,
            residual_acceleration_scale=500.0 * vector,
            state_velocity_limit=10.0 * vector,
            control_dt=env.control_dt,
        )
        rollout = PlannerRolloutConfig(
            mpc_policy="residual",
            q_ref_velocity_limit=10.0 * vector,
            q_ref_acceleration_limit=1000.0 * vector,
            residual_max=0.05 * vector,
            project_residual_kinematics=False,
        )
        return MuJoCoOraclePlanner(
            env=env,
            anchor_snapshot=snapshot,
            q_des=nominal,
            dq_des=nominal,
            nominal_q_ref=nominal,
            previous_q_ref=zeros,
            previous_q_ref_velocity=zeros,
            previous_residual=zeros,
            previous_residual_velocity=zeros,
            joint_low=torch.as_tensor(env.joint_low),
            joint_high=torch.as_tensor(env.joint_high),
            cost_config=cost,
            rollout_config=rollout,
        )

    def test_oracle_candidates_restore_identical_anchor_and_match_direct_steps(self) -> None:
        parent = MuJoCoArmEnv(self.model_xml, n_joints=6, seed=3)
        clone = MuJoCoArmEnv(self.model_xml, n_joints=6, seed=3)
        direct = MuJoCoArmEnv(self.model_xml, n_joints=6, seed=3)
        try:
            parent.reset_to_configuration(np.zeros(6, dtype=np.float32))
            snapshot = parent.capture_full_state()
            parent_before = parent.get_state().copy()
            planner = self._planner(clone, snapshot)
            candidates = torch.zeros((2, 2, 6), dtype=torch.float32)
            candidates[1, :, 0] = 0.5

            first = planner.evaluate(candidates)
            second = planner.evaluate(candidates)

            np.testing.assert_array_equal(parent.get_state(), parent_before)
            np.testing.assert_allclose(first["pred_states"].numpy(), second["pred_states"].numpy(), atol=0.0, rtol=0.0)
            np.testing.assert_allclose(first["pred_states"][:, 0].numpy(), np.repeat(parent_before[None], 2, axis=0))
            direct.restore_full_state(snapshot)
            expected = [direct.get_state()]
            for command in first["q_ref_sequences"][1].numpy():
                expected.append(direct.step(command))
            np.testing.assert_allclose(first["pred_states"][1].numpy(), np.asarray(expected), atol=1e-7, rtol=0.0)
            self.assertTrue(torch.all(torch.isfinite(first["costs"])))
        finally:
            parent.close()
            clone.close()
            direct.close()

    def test_oracle_specs_parse_locked_and_high_budget_forms(self) -> None:
        self.assertEqual(EVALUATE.parse_model_spec("Oracle,oracle")["kind"], "oracle")
        high = EVALUATE.parse_model_spec("OracleHighBudget,oracle,512,4")
        self.assertEqual((high["num_samples"], high["cem_iters"]), (512, 4))
        with self.assertRaises(ValueError):
            EVALUATE.parse_model_spec("Oracle,oracle,0,4")

    def test_resume_reuses_only_an_exact_fingerprint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            np.savez_compressed(run_dir / "rollout.npz", value=np.asarray([1.0], dtype=np.float32))
            fingerprint = {"sha256": "expected", "payload": {"case_id": "case_00"}}
            (run_dir / "run_fingerprint.json").write_text(json.dumps(fingerprint), encoding="utf-8")
            loaded = EVALUATE.load_completed_rollout(run_dir, fingerprint)
            self.assertIsNotNone(loaded)
            np.testing.assert_array_equal(loaded["value"], np.asarray([1.0], dtype=np.float32))
            with self.assertRaisesRegex(RuntimeError, "fingerprint mismatch"):
                EVALUATE.load_completed_rollout(run_dir, {"sha256": "different"})

    def test_evaluator_runs_oracle_spec_and_resumes_incrementally(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = root / "manifest.json"
            output_dir = root / "evaluation"
            manifest_path.write_text(
                json.dumps({"kind": "development", "cases": [{
                    "id": "case_00", "run_args": {"reference_mode": "joint_file", "num_samples": 7, "cem_iters": 2}
                }]}),
                encoding="utf-8",
            )
            arrays = {
                "realized_tracking_error": np.asarray([0.1], dtype=np.float32),
                "failure_flags": np.asarray([0], dtype=np.int64),
                "planning_time": np.asarray([0.2], dtype=np.float32),
                "mpc_replanned": np.asarray([1], dtype=np.int64),
                "actual_states": np.zeros((1, 12), dtype=np.float32),
                "q_des": np.zeros((1, 6), dtype=np.float32),
                "dynamics_backend": np.asarray("mujoco_oracle"),
            }

            def fake_save(run_dir, saved_arrays, rows):
                del rows
                Path(run_dir).mkdir(parents=True, exist_ok=True)
                np.savez_compressed(Path(run_dir) / "rollout.npz", **saved_arrays)

            argv = [
                "evaluate_model_abc.py", "--manifest", str(manifest_path), "--save_dir", str(output_dir),
                "--device", "cpu", "--model_spec", "Oracle,oracle",
            ]
            with mock.patch.object(sys, "argv", argv), \
                 mock.patch.object(EVALUATE, "save_mpc_run", side_effect=fake_save), \
                 mock.patch.object(EVALUATE.RUN_CEM_MPC, "run_closed_loop_mpc", return_value={"arrays": arrays, "rows": []}) as run:
                EVALUATE.main()
                called_args = run.call_args.args[0]
                self.assertEqual(called_args.dynamics_backend, "mujoco_oracle")
                self.assertEqual((called_args.num_samples, called_args.cem_iters), (7, 2))

            with mock.patch.object(sys, "argv", [*argv, "--resume"]), \
                 mock.patch.object(EVALUATE.RUN_CEM_MPC, "run_closed_loop_mpc") as resumed_run:
                EVALUATE.main()
                resumed_run.assert_not_called()
            self.assertTrue((output_dir / "model_abc_summary.csv").exists())
            self.assertTrue((output_dir / "case_00" / "Oracle" / "run_fingerprint.json").exists())

    def test_oracle_overrun_is_scheduled_at_fixed_logical_delay(self) -> None:
        args = RUN_CEM_MPC.parse_args([
            "--dynamics_backend", "mujoco_oracle", "--device", "cpu",
            "--planner_projection", "off", "--planner_projection_strategy", "full",
            "--multirate_mode", "virtual_asap",
            "--reference_mode", "multi_joint_sine", "--episode_len", "12",
            "--max_execution_steps", "2", "--settle_steps", "1", "--horizon", "3",
            "--num_samples", "3", "--cem_iters", "1", "--replan_interval_steps", "1",
            "--anticipation_delay_steps", "1", "--mpc_warmup_plans", "0",
        ])
        from mpc.cem_controller import CEMMPCController

        original_plan = CEMMPCController.plan

        def forced_overrun(controller, *plan_args, **plan_kwargs):
            result = original_plan(controller, *plan_args, **plan_kwargs)
            result.planning_time = 1.0
            return result

        with mock.patch.object(CEMMPCController, "plan", forced_overrun):
            result = RUN_CEM_MPC.run_closed_loop_mpc(args)
        events = result["arrays"]["packet_event"].tolist()
        self.assertIn("oracle_overrun_scheduled", events[0])
        self.assertIn("packet_activated", events[1])
        self.assertEqual(result["arrays"]["planner_late_drop_count"].item(), 0)
        self.assertEqual(result["arrays"]["replan_deadline_miss"].tolist(), [1, 1])

    def test_oracle_rejects_non_virtual_asap_modes(self) -> None:
        args = RUN_CEM_MPC.parse_args([
            "--dynamics_backend", "mujoco_oracle",
            "--planner_projection", "off", "--planner_projection_strategy", "full",
            "--multirate_mode", "threaded_asap",
        ])
        with self.assertRaisesRegex(ValueError, "only supports"):
            RUN_CEM_MPC.run_closed_loop_mpc(args)


if __name__ == "__main__":
    unittest.main()
