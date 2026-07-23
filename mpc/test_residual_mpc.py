from __future__ import annotations

import unittest
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
DYNAMICS_ROOT = ROOT / "dynamics_modeling"
if str(DYNAMICS_ROOT) not in sys.path:
    sys.path.insert(0, str(DYNAMICS_ROOT))

from mpc.cem_controller import CEMMPCConfig, CEMMPCController
from mpc.constraints import project_nominal_q_ref_sequence, project_position_command_sequence
from mpc.cost_functions import JointSpaceCostConfig, joint_space_tracking_cost
from mpc.logging import build_run_summary
from mpc.planner_rollout import construct_residual_q_ref_sequence, reanchor_residual_command
from mpc.delay_aware import corrected_direct_ik_command, corrected_direct_ik_command_np, feedback_correction
from mpc.asap_shared import LatestSnapshotStore, PacketFallbackStateMachine, PlanPacketStore, PlannerResultStore
from mpc.asap_types import ASAPPlanPacket, PlannerResultEvent, PlanningSnapshot
from mpc.recovery import residual_recovery_reason


class ResidualConstraintTests(unittest.TestCase):
    def test_nominal_projection_preserves_a_feasible_direct_reference(self) -> None:
        desired = torch.tensor([[0.10], [0.20], [0.30]], dtype=torch.float32)
        nominal = project_nominal_q_ref_sequence(
            desired,
            previous_q_ref=torch.tensor([0.0]),
            previous_q_ref_velocity=torch.tensor([0.0]),
            control_dt=0.1,
            velocity_limit=torch.tensor([2.0]),
            acceleration_limit=torch.tensor([20.0]),
            joint_low=torch.tensor([-1.0]),
            joint_high=torch.tensor([1.0]),
        )
        torch.testing.assert_close(nominal, desired)

    def test_position_projection_respects_velocity_and_acceleration(self) -> None:
        projected = project_position_command_sequence(
            torch.full((1, 4, 1), 1.0),
            previous_q_ref=torch.tensor([0.0]),
            previous_q_ref_velocity=torch.tensor([0.0]),
            control_dt=0.1,
            velocity_limit=torch.tensor([0.5]),
            acceleration_limit=torch.tensor([1.0]),
            joint_low=torch.tensor([-2.0]),
            joint_high=torch.tensor([2.0]),
        )[0, :, 0]
        velocity = torch.diff(torch.cat([torch.zeros(1), projected])) / 0.1
        acceleration = torch.diff(torch.cat([torch.zeros(1), velocity])) / 0.1
        self.assertTrue(bool(torch.all(torch.abs(velocity) <= 0.5 + 1e-6)))
        self.assertTrue(bool(torch.all(torch.abs(acceleration) <= 1.0 + 1e-6)))

    def test_zero_residual_is_the_executable_nominal_baseline(self) -> None:
        nominal = torch.tensor([[0.02], [0.05]], dtype=torch.float32)
        q_ref, residual, projected_offset, feasible = construct_residual_q_ref_sequence(
            torch.zeros((2, 2, 1)),
            nominal_q_ref=nominal,
            residual_max=torch.tensor([0.1]),
            previous_q_ref=torch.tensor([0.0]),
            previous_q_ref_velocity=torch.tensor([0.0]),
            joint_low=torch.tensor([-1.0]),
            joint_high=torch.tensor([1.0]),
            joint_limit_margin=0.0,
            q_ref_velocity_limit=torch.tensor([1.0]),
            q_ref_acceleration_limit=torch.tensor([10.0]),
            control_dt=0.1,
        )
        torch.testing.assert_close(q_ref, nominal.unsqueeze(0).expand_as(q_ref))
        torch.testing.assert_close(residual, torch.zeros_like(residual))
        torch.testing.assert_close(projected_offset, torch.zeros_like(projected_offset))
        self.assertTrue(bool(torch.all(feasible)))

    def test_legacy_projected_offset_bound_is_optional(self) -> None:
        common = dict(
            candidate_normalized_residual=torch.zeros((1, 1, 1)),
            nominal_q_ref=torch.tensor([[0.8]]), residual_max=torch.tensor([0.1]),
            previous_q_ref=torch.tensor([0.0]), previous_q_ref_velocity=torch.tensor([0.0]),
            joint_low=torch.tensor([-1.0]), joint_high=torch.tensor([1.0]), joint_limit_margin=0.0,
            q_ref_velocity_limit=torch.tensor([0.1]), q_ref_acceleration_limit=torch.tensor([0.1]),
            control_dt=0.01,
        )
        self.assertTrue(bool(construct_residual_q_ref_sequence(**common)[3][0]))
        self.assertFalse(bool(construct_residual_q_ref_sequence(
            **common, enforce_projected_offset_bound=True,
        )[3][0]))

    def test_cached_residual_is_reanchored_and_reprojected(self) -> None:
        command, executed, feasible = reanchor_residual_command(
            buffered_residual=torch.tensor([0.08]),
            nominal_q_ref=torch.tensor([0.20]),
            residual_max=torch.tensor([0.10]),
            previous_q_ref=torch.tensor([0.0]),
            previous_q_ref_velocity=torch.tensor([0.0]),
            joint_low=torch.tensor([-1.0]),
            joint_high=torch.tensor([1.0]),
            joint_limit_margin=0.0,
            q_ref_velocity_limit=torch.tensor([2.0]),
            q_ref_acceleration_limit=torch.tensor([20.0]),
            control_dt=0.1,
        )
        self.assertTrue(feasible)
        torch.testing.assert_close(command, torch.tensor([0.20]))
        torch.testing.assert_close(executed, torch.tensor([0.0]))

    def test_reanchoring_zero_residual_preserves_online_nominal(self) -> None:
        command, executed, feasible = reanchor_residual_command(
            buffered_residual=torch.tensor([0.0]),
            nominal_q_ref=torch.tensor([0.10]),
            residual_max=torch.tensor([0.10]),
            previous_q_ref=torch.tensor([0.0]),
            previous_q_ref_velocity=torch.tensor([0.0]),
            joint_low=torch.tensor([-1.0]),
            joint_high=torch.tensor([1.0]),
            joint_limit_margin=0.0,
            q_ref_velocity_limit=torch.tensor([2.0]),
            q_ref_acceleration_limit=torch.tensor([20.0]),
            control_dt=0.1,
        )
        self.assertTrue(feasible)
        torch.testing.assert_close(command, torch.tensor([0.10]))
        torch.testing.assert_close(executed, torch.tensor([0.0]))

    def test_zero_correction_returns_to_nominal_through_projection(self) -> None:
        command, correction = corrected_direct_ik_command(
            nominal_q_des=torch.tensor([0.30]),
            correction=torch.tensor([0.0]),
            previous_q_ref=torch.tensor([0.0]),
            previous_q_ref_velocity=torch.tensor([0.0]),
            joint_low=torch.tensor([-1.0]),
            joint_high=torch.tensor([1.0]),
            joint_limit_margin=0.0,
            velocity_limit=torch.tensor([0.1]),
            acceleration_limit=torch.tensor([0.1]),
            control_dt=0.01,
        )
        self.assertLessEqual(float(command[0]), 1.1e-5)
        torch.testing.assert_close(correction, command - torch.tensor([0.30]))

    def test_feedback_has_the_planned_tracking_sign_and_is_bounded(self) -> None:
        correction = feedback_correction(
            predicted_state=np.array([0.4, 1.0], dtype=np.float32),
            measured_state=np.array([0.1, 0.0], dtype=np.float32),
            kq=0.5,
            kdq=0.1,
            max_abs=np.array([0.05], dtype=np.float32),
        )
        np.testing.assert_allclose(correction, np.array([0.05], dtype=np.float32))

    def test_numpy_direct_ik_projection_limits_zero_and_nonzero_corrections(self) -> None:
        command, residual = corrected_direct_ik_command_np(
            nominal_q_des=np.array([0.3], dtype=np.float32), correction=np.array([0.0], dtype=np.float32),
            previous_q_ref=np.array([0.0], dtype=np.float32), previous_q_ref_velocity=np.array([0.0], dtype=np.float32),
            joint_low=np.array([-1.0], dtype=np.float32), joint_high=np.array([1.0], dtype=np.float32),
            joint_limit_margin=0.0, velocity_limit=np.array([0.1], dtype=np.float32),
            acceleration_limit=np.array([0.1], dtype=np.float32), control_dt=0.01,
        )
        self.assertLessEqual(float(command[0]), 1.1e-5)
        np.testing.assert_allclose(residual, command - np.array([0.3], dtype=np.float32), atol=1e-7)
        command, _ = corrected_direct_ik_command_np(
            np.array([1.0], dtype=np.float32), np.array([1.0], dtype=np.float32), np.array([0.0], dtype=np.float32),
            np.array([0.0], dtype=np.float32), np.array([-2.0], dtype=np.float32), np.array([2.0], dtype=np.float32),
            0.0, np.array([0.5], dtype=np.float32), np.array([1.0], dtype=np.float32), 0.1,
        )
        self.assertLessEqual(float(command[0]), 0.011)


class ASAPStoreTests(unittest.TestCase):
    def _packet(self, plan_id: int, activation: int) -> ASAPPlanPacket:
        return ASAPPlanPacket(plan_id=plan_id, launch_step=0, launch_time_ns=0, activation_step=activation,
            activation_time_ns=activation * 10, publish_time_ns=1, residual_sequence=np.full((3, 1), plan_id, dtype=np.float32),
            predicted_state_sequence=np.zeros((4, 2), dtype=np.float32), planning_time_s=0.01,
            anchor_state=np.zeros(2, dtype=np.float32), selection_mode="best", selected_cost=1.0)

    def test_latest_snapshot_coalesces_and_copies_arrays(self) -> None:
        store = LatestSnapshotStore()
        first = PlanningSnapshot(0, 0, 0, np.zeros((1, 2), dtype=np.float32), np.zeros((1, 1), dtype=np.float32), np.zeros(1, dtype=np.float32), np.zeros(1, dtype=np.float32), np.zeros(1, dtype=np.float32), np.zeros(1, dtype=np.float32), np.zeros(1, dtype=np.float32), np.zeros(1, dtype=np.float32), ())
        second = PlanningSnapshot(1, 1, 1, np.ones((1, 2), dtype=np.float32), np.ones((1, 1), dtype=np.float32), np.ones(1, dtype=np.float32), np.ones(1, dtype=np.float32), np.ones(1, dtype=np.float32), np.ones(1, dtype=np.float32), np.ones(1, dtype=np.float32), np.ones(1, dtype=np.float32), ())
        store.publish(first); store.publish(second)
        received = store.wait_for_newer(-1, 0.0)
        self.assertIsNotNone(received)
        assert received is not None
        self.assertEqual(received.request_id, 1)
        second.states_history.fill(9.0)
        self.assertEqual(float(received.states_history[0, 0]), 1.0)

    def test_packet_activation_uses_latest_due_packet(self) -> None:
        store = PlanPacketStore()
        store.publish(self._packet(1, 2)); store.publish(self._packet(2, 3))
        active = store.activate_due(3, 30)
        self.assertIsNotNone(active)
        assert active is not None
        self.assertEqual(active.plan_id, 2)

    def test_planner_result_events_are_delivered_once_in_order(self) -> None:
        store = PlannerResultStore()
        for result_id, result_type in ((0, "planner_failure"), (1, "success_published")):
            store.publish(PlannerResultEvent(
                result_id=result_id, request_id=result_id + 10, result_type=result_type,
                reason_code="all_costs_invalid" if result_id == 0 else "", reason_detail="",
                plan_id=-1 if result_id == 0 else 4, planning_time_s=0.01,
                end_to_end_latency_s=0.02, candidate_count=8, valid_candidate_count=4,
            ))
        self.assertEqual([event.result_id for event in store.drain()], [0, 1])
        self.assertEqual(store.drain(), [])

    def test_packet_gap_state_machine_distinguishes_startup_failure_and_recovery(self) -> None:
        machine = PacketFallbackStateMachine()
        startup = machine.update(None)
        self.assertEqual(startup.state, "STARTUP_NO_PACKET")
        self.assertEqual(startup.packet_expired_event, 0)
        first = machine.update(4)
        self.assertEqual(first.first_packet_activated_event, 1)
        self.assertEqual(first.fallback_ended_event, 1)
        machine.observe_result("planner_failure")
        self.assertEqual(machine.update(4).state, "PACKET_ACTIVE")
        gap = machine.update(None)
        self.assertEqual(gap.state, "GAP_AFTER_PLANNER_FAILURE")
        self.assertEqual(gap.packet_expired_event, 1)
        self.assertEqual(gap.fallback_started_event, 1)
        self.assertEqual(machine.expiration_count, 1)
        replacement = machine.update(5)
        self.assertEqual(replacement.state, "PACKET_ACTIVE")
        self.assertEqual(replacement.fallback_ended_event, 1)
        self.assertEqual(machine.last_blocking_result_type, "")


class ResidualCostTests(unittest.TestCase):
    def _config(self) -> JointSpaceCostConfig:
        return JointSpaceCostConfig(
            q_tracking_scale=torch.tensor([1.0]),
            dq_tracking_scale=torch.tensor([1.0]),
            residual_scale=torch.tensor([0.1]),
            servo_scale=torch.tensor([1.0]),
            residual_velocity_scale=torch.tensor([1.0]),
            residual_acceleration_scale=torch.tensor([1.0]),
            state_velocity_limit=torch.tensor([10.0]),
            control_dt=0.1,
            joint_limit_safe_margin=0.01,
            joint_limit_temp=0.01,
            dq_limit_temp=0.01,
        )

    def test_zero_residual_has_no_anchor_or_smoothness_cost(self) -> None:
        pred = torch.zeros((1, 3, 2))
        cost, terms = joint_space_tracking_cost(
            pred_states=pred,
            q_des=torch.zeros((2, 1)),
            dq_des=torch.zeros((2, 1)),
            actuator_q_ref=torch.zeros((1, 2, 1)),
            nominal_q_ref=torch.zeros((2, 1)),
            requested_residual=torch.zeros((1, 2, 1)),
            previous_q_ref=torch.zeros(1),
            previous_q_ref_velocity=torch.zeros(1),
            previous_residual=torch.zeros(1),
            previous_residual_velocity=torch.zeros(1),
            joint_low=torch.tensor([-1.0]),
            joint_high=torch.tensor([1.0]),
            config=self._config(),
            return_terms=True,
        )
        self.assertLess(float(cost[0]), 1e-6)
        for name in ("residual", "servo", "residual_velocity", "residual_acceleration", "first", "terminal"):
            self.assertLess(float(terms[name][0]), 1e-6)

    def test_predicted_hard_joint_violation_is_rejected(self) -> None:
        pred = torch.zeros((1, 3, 2))
        pred[:, 1, 0] = 1.1
        cost = joint_space_tracking_cost(
            pred_states=pred,
            q_des=torch.zeros((2, 1)),
            dq_des=torch.zeros((2, 1)),
            actuator_q_ref=torch.zeros((1, 2, 1)),
            nominal_q_ref=torch.zeros((2, 1)),
            requested_residual=torch.zeros((1, 2, 1)),
            previous_q_ref=torch.zeros(1),
            previous_q_ref_velocity=torch.zeros(1),
            previous_residual=torch.zeros(1),
            previous_residual_velocity=torch.zeros(1),
            joint_low=torch.tensor([-1.0]),
            joint_high=torch.tensor([1.0]),
            config=self._config(),
        )
        self.assertTrue(bool(torch.isinf(cost[0])))

    def test_residual_cost_is_independent_of_projection_lag(self) -> None:
        requested = torch.tensor([[[0.02], [0.03]]])
        common = dict(
            pred_states=torch.zeros((1, 3, 2)), q_des=torch.zeros((2, 1)), dq_des=torch.zeros((2, 1)),
            requested_residual=requested, nominal_q_ref=torch.zeros((2, 1)),
            previous_q_ref=torch.zeros(1), previous_q_ref_velocity=torch.zeros(1),
            previous_residual=torch.zeros(1), previous_residual_velocity=torch.zeros(1),
            joint_low=torch.tensor([-1.0]), joint_high=torch.tensor([1.0]), config=self._config(),
            return_terms=True,
        )
        _, first = joint_space_tracking_cost(actuator_q_ref=requested.clone(), **common)
        _, second = joint_space_tracking_cost(actuator_q_ref=requested + 0.10, **common)
        for name in ("residual", "residual_velocity", "residual_acceleration", "first"):
            torch.testing.assert_close(first[name], second[name])
        self.assertFalse(torch.allclose(first["servo"], second["servo"]))

    def test_diagnostic_projected_offset_can_drive_residual_cost(self) -> None:
        requested = torch.zeros((1, 2, 1))
        projected_offset = torch.tensor([[[0.02], [0.03]]])
        _, terms = joint_space_tracking_cost(
            pred_states=torch.zeros((1, 3, 2)), q_des=torch.zeros((2, 1)), dq_des=torch.zeros((2, 1)),
            actuator_q_ref=projected_offset, requested_residual=requested,
            residual_cost_sequence=projected_offset, nominal_q_ref=torch.zeros((2, 1)),
            previous_q_ref=torch.zeros(1), previous_q_ref_velocity=torch.zeros(1),
            previous_residual=torch.zeros(1), previous_residual_velocity=torch.zeros(1),
            joint_low=torch.tensor([-1.0]), joint_high=torch.tensor([1.0]), config=self._config(),
            return_terms=True,
        )
        self.assertGreater(float(terms["residual"][0]), 0.0)


class _BaselinePlanner:
    def evaluate(self, action: torch.Tensor) -> dict[str, torch.Tensor]:
        costs = torch.mean(torch.square(action), dim=(1, 2))
        batch, horizon, dim = action.shape
        return {
            "costs": costs,
            "q_ref_sequences": action.clone(),
            "cost_terms": {"total": costs},
            "pred_states": torch.zeros((batch, horizon + 1, 2 * dim)),
        }


class CEMBaselineTests(unittest.TestCase):
    def test_zero_residual_is_forced_and_selected_when_optimal(self) -> None:
        controller = CEMMPCController(
            CEMMPCConfig(
                horizon=3,
                action_dim=1,
                num_samples=8,
                cem_iters=2,
                init_std=0.5,
                min_std=0.25,
                uniform_sample_ratio=0.25,
                force_baseline_candidate=True,
                execute="lowest_cost",
                seed=7,
            ),
            planner=_BaselinePlanner(),
            joint_low=np.array([-1.0], dtype=np.float32),
            joint_high=np.array([1.0], dtype=np.float32),
        )
        samples = controller._sample_population(controller.mean, controller.std)
        self.assertEqual(tuple(samples.shape), (8, 3, 1))
        self.assertTrue(bool(torch.allclose(samples[0], torch.zeros_like(samples[0]))))
        result = controller.plan(np.zeros(2, dtype=np.float32), np.zeros(1, dtype=np.float32))
        self.assertEqual(result.selection_mode, "baseline")
        self.assertAlmostEqual(result.baseline_cost, 0.0)
        np.testing.assert_allclose(result.q_ref, 0.0)
        self.assertEqual(result.selected_q_ref_sequence.shape, (3, 1))
        self.assertEqual(result.selected_residual_sequence.shape, (3, 1))
        self.assertEqual(result.selected_predicted_state_sequence.shape, (4, 2))
        self.assertEqual(result.selected_action_sequence.shape, (3, 1))
        np.testing.assert_allclose(result.selected_action_sequence, 0.0)
        np.testing.assert_allclose(result.selected_residual_sequence, 0.0)
        np.testing.assert_allclose(result.q_ref, result.selected_q_ref_sequence[0])
        controller.mean = torch.tensor([[1.0], [2.0], [3.0]])
        controller.advance_after_execution(2)
        torch.testing.assert_close(controller.mean, torch.tensor([[2.0], [3.0], [3.0]]))
        controller.mean = torch.tensor([[1.0], [2.0], [3.0]])
        controller.shift_warm_start(2)
        torch.testing.assert_close(controller.mean, torch.tensor([[3.0], [3.0], [3.0]]))
        controller.mean.fill_(1.0)
        controller.reset()
        self.assertTrue(bool(torch.allclose(controller.mean, torch.zeros_like(controller.mean))))


class RecoveryTests(unittest.TestCase):
    def test_command_limit_saturation_is_not_a_recovery_condition(self) -> None:
        """Only the supplied failure signals may trigger recovery."""
        reason = residual_recovery_reason(
            [0.10, 0.10, 0.10, 0.10],
            residual_saturation_streak=0,
            consecutive_steps=3,
            error_ratio=1.25,
            min_tracking_error=0.05,
            recovery_active=False,
        )
        self.assertEqual(reason, "")

    def test_sustained_error_growth_and_residual_saturation_trigger_recovery(self) -> None:
        common = dict(
            consecutive_steps=3,
            error_ratio=1.25,
            min_tracking_error=0.05,
            recovery_active=False,
        )
        self.assertEqual(
            residual_recovery_reason([0.05, 0.06, 0.07, 0.08], residual_saturation_streak=0, **common),
            "tracking_error_growth",
        )
        self.assertEqual(
            residual_recovery_reason([0.10, 0.10], residual_saturation_streak=3, **common),
            "residual_saturation",
        )

    def test_run_summary_reports_total_recovery_trigger_count(self) -> None:
        summary = build_run_summary(
            {
                "recovery_active_flags": np.array([0, 1, 1, 0]),
                "recovery_trigger_reasons": np.array(
                    ["tracking_error_growth", "", "", "residual_saturation"]
                ),
            }
        )
        safety = summary["safety"]
        self.assertEqual(safety["recovery_trigger_count"], 2)
        self.assertEqual(safety["recovery_active_step_count"], 2)

    def test_run_summary_uses_only_replanning_steps_for_planning_timing(self) -> None:
        summary = build_run_summary(
            {
                "planning_time": np.array([0.04, 0.0, 0.0, 0.06]),
                "replan_time": np.array([0.04, np.nan, np.nan, 0.06]),
                "mpc_replanned": np.array([1, 0, 0, 1]),
                "replan_deadline_miss": np.array([0, 0, 0, 1]),
                "replan_interval_steps": np.array(5),
                "replan_deadline_s": np.array(0.05),
            }
        )
        self.assertAlmostEqual(summary["timing"]["planning_time_s"]["mean"], 0.05)
        self.assertEqual(summary["replanning"]["count"], 2)
        self.assertEqual(summary["replanning"]["deadline_miss_count"], 1)


if __name__ == "__main__":
    unittest.main()
