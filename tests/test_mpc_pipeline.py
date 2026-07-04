from __future__ import annotations

import math
import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
DYNAMICS_ROOT = ROOT / "learned_mujoco_dynamics"
for path in (ROOT, DYNAMICS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from learned_dynamics.integration import reconstruct_next_state


def load_script_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RUN_CEM_MPC = load_script_module("local_run_cem_mpc_for_tests", ROOT / "scripts" / "run_cem_mpc.py")
ANALYZE_OOD_MPC = load_script_module("local_analyze_ood_mpc_for_tests", ROOT / "scripts" / "analyze_ood_mpc.py")


class IdentityNormalizer:
    def normalize_single_input(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return torch.cat([states, actions], dim=-1)

    def normalize_sequence_input(self, sequence: torch.Tensor, state_dim: int) -> torch.Tensor:
        return sequence

    def denormalize_delta(self, deltas: torch.Tensor) -> torch.Tensor:
        return deltas


class ConstantDeltaDQModel(torch.nn.Module):
    def __init__(self, n_joints: int, delta: float) -> None:
        super().__init__()
        self.n_joints = n_joints
        self.delta = float(delta)
        self.seen_inputs: list[torch.Tensor] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.seen_inputs.append(x.detach().cpu())
        return torch.full((x.shape[0], self.n_joints), self.delta, dtype=x.dtype, device=x.device)


class MPCPipelineTests(unittest.TestCase):
    def test_runtime_outputs_resolve_to_top_level_outputs(self) -> None:
        resolved = RUN_CEM_MPC.resolve_runtime_path("outputs/mpc/cem_run")

        self.assertEqual(resolved, ROOT / "outputs" / "mpc" / "cem_run")

    def test_runtime_legacy_dynamics_outputs_stay_addressable(self) -> None:
        resolved = RUN_CEM_MPC.resolve_runtime_path(
            "learned_mujoco_dynamics/outputs/checkpoints_transformer/transformer_20260606_154206/best_model.pt"
        )

        self.assertEqual(
            resolved,
            ROOT / "learned_mujoco_dynamics" / "outputs" / "checkpoints_transformer" / "transformer_20260606_154206" / "best_model.pt",
        )

    def test_runtime_bare_model_xml_resolves_to_dynamics_root(self) -> None:
        resolved = RUN_CEM_MPC.resolve_runtime_path("ABB_IRB2400.xml")

        self.assertEqual(resolved, DYNAMICS_ROOT / "ABB_IRB2400.xml")

    def test_ood_outputs_resolve_to_top_level_outputs(self) -> None:
        resolved = ANALYZE_OOD_MPC.resolve_runtime_path("outputs/mpc/ood_summary.csv")

        self.assertEqual(resolved, ROOT / "outputs" / "mpc" / "ood_summary.csv")

    def test_delta_q_ref_is_converted_to_cumulative_absolute_actuator_targets(self) -> None:
        from mpc.planner_rollout import construct_actuator_q_ref_sequence

        previous_q_ref = torch.tensor([0.5, -0.25], dtype=torch.float32)
        delta_q_ref = torch.tensor(
            [
                [[0.1, 0.2], [0.2, -0.1], [-0.05, 0.3]],
                [[-0.1, 0.0], [0.0, 0.1], [0.2, -0.2]],
            ],
            dtype=torch.float32,
        )
        joint_low = torch.full((2,), -10.0)
        joint_high = torch.full((2,), 10.0)

        q_ref = construct_actuator_q_ref_sequence(
            candidate_sequence=delta_q_ref,
            current_q=torch.zeros(2),
            previous_q_ref=previous_q_ref,
            mode="delta",
            delta_base="previous_q_ref",
            joint_low=joint_low,
            joint_high=joint_high,
            joint_limit_margin=0.0,
        )

        expected = torch.tensor(
            [
                [[0.6, -0.05], [0.8, -0.15], [0.75, 0.15]],
                [[0.4, -0.25], [0.4, -0.15], [0.6, -0.35]],
            ],
            dtype=torch.float32,
        )
        self.assertTrue(torch.allclose(q_ref, expected))

    def test_rollout_dynamics_batch_uses_absolute_q_ref_and_matches_unchunked(self) -> None:
        from learned_dynamics.rollout import rollout_dynamics_batch

        n_joints = 2
        state_dim = 2 * n_joints
        history = torch.zeros((4, 3, state_dim + n_joints), dtype=torch.float32)
        future_q_ref = torch.randn((4, 5, n_joints), dtype=torch.float32)
        model_full = ConstantDeltaDQModel(n_joints=n_joints, delta=0.1)
        model_chunked = ConstantDeltaDQModel(n_joints=n_joints, delta=0.1)

        full = rollout_dynamics_batch(
            model=model_full,
            normalizer=IdentityNormalizer(),
            model_type="transformer",
            initial_history=history,
            future_q_ref=future_q_ref,
            state_dim=state_dim,
            target_mode="delta_dq",
            control_dt=0.01,
            rollout_batch_size=None,
        )
        chunked = rollout_dynamics_batch(
            model=model_chunked,
            normalizer=IdentityNormalizer(),
            model_type="transformer",
            initial_history=history,
            future_q_ref=future_q_ref,
            state_dim=state_dim,
            target_mode="delta_dq",
            control_dt=0.01,
            rollout_batch_size=2,
        )

        self.assertTrue(torch.allclose(full, chunked))
        first_model_input = model_full.seen_inputs[0]
        self.assertTrue(torch.allclose(first_model_input[:, -1, state_dim:], future_q_ref[:, 0]))
        second_model_input = model_full.seen_inputs[1]
        self.assertTrue(torch.allclose(second_model_input[:, -1, state_dim:], future_q_ref[:, 1]))

    def test_rollout_dynamics_batch_matches_shared_delta_dq_reconstruction(self) -> None:
        from learned_dynamics.rollout import rollout_dynamics_batch

        n_joints = 2
        state_dim = 2 * n_joints
        initial_state = torch.tensor([[1.0, -1.0, 0.2, -0.1]], dtype=torch.float32)
        initial_action = torch.zeros((1, n_joints), dtype=torch.float32)
        history = torch.cat([initial_state, initial_action], dim=-1).unsqueeze(1)
        future_q_ref = torch.zeros((1, 1, n_joints), dtype=torch.float32)
        model = ConstantDeltaDQModel(n_joints=n_joints, delta=0.3)

        predicted = rollout_dynamics_batch(
            model=model,
            normalizer=IdentityNormalizer(),
            model_type="transformer",
            initial_history=history,
            future_q_ref=future_q_ref,
            state_dim=state_dim,
            target_mode="delta_dq",
            control_dt=0.05,
        )
        expected_next = reconstruct_next_state(
            initial_state,
            torch.full((1, n_joints), 0.3),
            "delta_dq",
            0.05,
            n_joints,
        )

        self.assertTrue(torch.allclose(predicted[:, 1], expected_next))

    def test_cem_invalid_cost_falls_back_to_previous_q_ref(self) -> None:
        from mpc.cem_controller import CEMMPCController, CEMMPCConfig

        class InvalidPlanner:
            def evaluate(self, candidate_delta_q_ref: torch.Tensor) -> dict[str, torch.Tensor]:
                return {
                    "costs": torch.full((candidate_delta_q_ref.shape[0],), math.nan),
                    "q_ref_sequences": torch.zeros_like(candidate_delta_q_ref),
                }

        previous_q_ref = np.array([0.1, -0.2], dtype=np.float32)
        controller = CEMMPCController(
            config=CEMMPCConfig(
                horizon=3,
                action_dim=2,
                num_samples=8,
                num_elites=2,
                cem_iters=1,
                init_std=0.2,
                min_std=0.01,
                seed=3,
            ),
            planner=InvalidPlanner(),
            joint_low=np.full(2, -1.0, dtype=np.float32),
            joint_high=np.full(2, 1.0, dtype=np.float32),
        )

        result = controller.plan(current_state=np.zeros(4, dtype=np.float32), previous_q_ref=previous_q_ref)

        self.assertTrue(result.failure)
        self.assertTrue(np.allclose(result.q_ref, previous_q_ref))
        self.assertTrue(np.allclose(result.delta_q_ref, np.zeros(2, dtype=np.float32)))


if __name__ == "__main__":
    unittest.main()
