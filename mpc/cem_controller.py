from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Protocol

import numpy as np
import torch


class PlannerProtocol(Protocol):
    def evaluate(self, candidate_action: torch.Tensor) -> dict[str, torch.Tensor]:
        ...


@dataclass(frozen=True)
class CEMMPCConfig:
    horizon: int
    action_dim: int
    num_samples: int = 1024
    num_elites: int | None = None
    elite_ratio: float = 0.08
    cem_iters: int = 4
    init_std: float = 0.5
    min_std: float = 0.05
    smoothing_alpha: float = 0.2
    temporal_noise_alpha: float = 0.8
    reset_std_each_step: bool = True
    uniform_sample_ratio: float = 0.0
    force_baseline_candidate: bool = False
    seed: int = 0
    device: str = "cpu"
    execute: str = "lowest_cost"
    alternative_distance_scale: np.ndarray | None = None


@dataclass
class CEMCandidate:
    """One projected counterfactual plan retained for offline validation."""

    role_mask: tuple[str, ...]
    q_ref_sequence: np.ndarray
    residual_sequence: np.ndarray
    predicted_state_sequence: np.ndarray
    predicted_cost: float
    cost_terms: dict[str, float]


@dataclass
class CEMMPCResult:
    q_ref: np.ndarray
    delta_q_ref: np.ndarray
    best_cost: float
    mean_cost: float
    baseline_cost: float
    selected_cost: float
    elite_mean_cost: float
    selection_mode: str
    planning_time: float
    failure: bool
    failure_reason: str
    best_sequence: np.ndarray
    selected_action_sequence: np.ndarray
    selected_q_ref_sequence: np.ndarray
    selected_residual_sequence: np.ndarray
    cost_terms: dict[str, float]
    predicted_next_state: np.ndarray
    selected_predicted_state_sequence: np.ndarray
    sampling_std_start_mean: float
    sampling_std_end_mean: float
    branch_candidates: tuple[CEMCandidate, ...] = ()


class CEMMPCController:
    def __init__(
        self,
        config: CEMMPCConfig,
        planner: PlannerProtocol,
        joint_low: np.ndarray,
        joint_high: np.ndarray,
    ) -> None:
        if config.horizon <= 0 or config.action_dim <= 0:
            raise ValueError("horizon and action_dim must be positive")
        if config.num_samples <= 0 or config.cem_iters <= 0:
            raise ValueError("num_samples and cem_iters must be positive")
        if config.execute not in {"mean", "best", "lowest_cost"}:
            raise ValueError(f"execute must be 'mean', 'best' or 'lowest_cost', got {config.execute!r}")
        if not 0.0 <= config.uniform_sample_ratio <= 1.0:
            raise ValueError("uniform_sample_ratio must be in [0, 1]")
        if config.init_std <= 0.0 or config.min_std <= 0.0:
            raise ValueError("init_std and min_std must be positive")
        if config.force_baseline_candidate and config.num_samples < 3:
            raise ValueError("force_baseline_candidate requires at least three samples")
        self.config = config
        self.planner = planner
        self.device = torch.device(config.device)
        self.generator = torch.Generator(device=self.device).manual_seed(config.seed)
        self.mean = torch.zeros((config.horizon, config.action_dim), dtype=torch.float32, device=self.device)
        self.initial_std = torch.full_like(self.mean, float(config.init_std))
        self.std = self.initial_std.clone()
        self.joint_low = torch.as_tensor(joint_low, dtype=torch.float32, device=self.device)
        self.joint_high = torch.as_tensor(joint_high, dtype=torch.float32, device=self.device)
        if config.alternative_distance_scale is None:
            self.alternative_distance_scale = torch.ones(config.action_dim, dtype=torch.float32, device=self.device)
        else:
            scale = torch.as_tensor(config.alternative_distance_scale, dtype=torch.float32, device=self.device)
            if scale.shape != (config.action_dim,) or not bool(torch.all(torch.isfinite(scale))) or bool(torch.any(scale <= 0)):
                raise ValueError("alternative_distance_scale must contain one finite positive value per action dimension")
            self.alternative_distance_scale = scale

    @property
    def num_elites(self) -> int:
        if self.config.num_elites is not None:
            return max(1, min(int(self.config.num_elites), self.config.num_samples))
        return max(1, min(self.config.num_samples, int(round(self.config.num_samples * self.config.elite_ratio))))

    @property
    def uniform_sample_count(self) -> int:
        stochastic_count = self.config.num_samples - (2 if self.config.force_baseline_candidate else 0)
        return int(round(stochastic_count * float(self.config.uniform_sample_ratio)))

    def _sample_temporal_noise(self, count: int) -> torch.Tensor:
        noise = torch.randn(
            (count, self.config.horizon, self.config.action_dim),
            generator=self.generator,
            device=self.device,
        )
        alpha = float(self.config.temporal_noise_alpha)
        if alpha <= 0:
            return noise
        for step_idx in range(1, self.config.horizon):
            noise[:, step_idx] = alpha * noise[:, step_idx - 1] + (1.0 - alpha) * noise[:, step_idx]
        return noise

    def _sample_population(self, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        """Mix local Gaussian candidates with global uniform exploration samples."""
        forced: list[torch.Tensor] = []
        stochastic_count = self.config.num_samples
        if self.config.force_baseline_candidate:
            forced = [torch.zeros_like(mean).unsqueeze(0), mean.unsqueeze(0)]
            stochastic_count -= 2
        uniform_count = int(round(stochastic_count * float(self.config.uniform_sample_ratio)))
        gaussian_count = stochastic_count - uniform_count
        samples: list[torch.Tensor] = []
        if gaussian_count:
            gaussian = mean.unsqueeze(0) + self._sample_temporal_noise(gaussian_count) * std.unsqueeze(0)
            samples.append(torch.clamp(gaussian, min=-1.0, max=1.0))
        if uniform_count:
            uniform = 2.0 * torch.rand(
                (uniform_count, self.config.horizon, self.config.action_dim),
                generator=self.generator,
                device=self.device,
            ) - 1.0
            samples.append(uniform)
        return torch.cat([*forced, *samples], dim=0)

    def reset(self) -> None:
        """Discard a stale warm start after the online safety monitor recovers."""
        self.mean.zero_()
        self.std = self.initial_std.clone()

    def advance_after_execution(self, executed_steps: int) -> None:
        """Advance the CEM warm start after executing a buffered plan prefix.

        ``plan`` without an explicit anchor shift already discards one step;
        legacy multi-rate callers discard the remaining prefix here.
        """
        if executed_steps <= 1:
            return
        self.shift_warm_start(int(executed_steps) - 1)

    def shift_warm_start(self, shift_steps: int) -> None:
        """Align the saved unshifted sequence to a later planning anchor."""
        if shift_steps <= 0:
            return
        shift = min(int(shift_steps), self.config.horizon)
        tail = self.mean[-1:].expand(shift, -1)
        self.mean = torch.cat([self.mean[shift:], tail], dim=0).detach().clone()

    def _shifted_warm_start(self, shift_steps: int) -> torch.Tensor:
        """Return an aligned mean without mutating controller state."""
        if shift_steps <= 0:
            return self.mean.clone()
        shift = min(int(shift_steps), self.config.horizon)
        tail = self.mean[-1:].expand(shift, -1)
        return torch.cat([self.mean[shift:], tail], dim=0).detach().clone()

    def _fallback(self, previous_q_ref: np.ndarray, start_time: float, reason: str) -> CEMMPCResult:
        previous = np.asarray(previous_q_ref, dtype=np.float32)
        return CEMMPCResult(
            q_ref=previous.copy(),
            delta_q_ref=np.zeros(self.config.action_dim, dtype=np.float32),
            best_cost=float("inf"),
            mean_cost=float("nan"),
            baseline_cost=float("nan"),
            selected_cost=float("inf"),
            elite_mean_cost=float("inf"),
            selection_mode="previous_q_ref_fallback",
            planning_time=perf_counter() - start_time,
            failure=True,
            failure_reason=reason,
            best_sequence=np.zeros((self.config.horizon, self.config.action_dim), dtype=np.float32),
            selected_action_sequence=np.zeros((self.config.horizon, self.config.action_dim), dtype=np.float32),
            selected_q_ref_sequence=np.repeat(previous[None, :], self.config.horizon, axis=0),
            selected_residual_sequence=np.zeros((self.config.horizon, self.config.action_dim), dtype=np.float32),
            cost_terms={},
            predicted_next_state=np.full(2 * self.config.action_dim, np.nan, dtype=np.float32),
            selected_predicted_state_sequence=np.full(
                (self.config.horizon + 1, 2 * self.config.action_dim), np.nan, dtype=np.float32
            ),
            sampling_std_start_mean=float("nan"),
            sampling_std_end_mean=float("nan"),
        )

    def _valid_q_ref_sequence(self, sequence: torch.Tensor, batch_size: int) -> bool:
        expected_shape = (batch_size, self.config.horizon, self.config.action_dim)
        return sequence.shape == expected_shape and bool(torch.all(torch.isfinite(sequence)))

    def _diagnostics_from_evaluation(
        self,
        evaluation: dict[str, torch.Tensor],
        index: int,
    ) -> tuple[dict[str, float], np.ndarray]:
        terms: dict[str, float] = {}
        for name, values in evaluation.get("cost_terms", {}).items():
            if isinstance(values, torch.Tensor) and values.ndim == 1 and values.shape[0] > index:
                terms[name] = float(values[index].detach().cpu())
        pred_states = evaluation.get("pred_states")
        if isinstance(pred_states, torch.Tensor) and pred_states.ndim == 3 and pred_states.shape[0] > index and pred_states.shape[1] > 1:
            predicted_next_state = pred_states[index, 1].detach().cpu().numpy().astype(np.float32)
        else:
            predicted_next_state = np.full(2 * self.config.action_dim, np.nan, dtype=np.float32)
        return terms, predicted_next_state

    def _evaluate_sequence(
        self, sequence: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float], np.ndarray, torch.Tensor] | None:
        """Return one constrained action sequence's rollout cost when valid."""
        try:
            evaluation = self.planner.evaluate(sequence.unsqueeze(0))
        except RuntimeError:
            return None
        costs = evaluation.get("costs")
        q_ref_sequences = evaluation.get("q_ref_sequences")
        if costs is None or q_ref_sequences is None:
            return None
        costs = costs.to(self.device)
        q_ref_sequences = q_ref_sequences.to(self.device)
        if costs.shape != (1,) or not bool(torch.isfinite(costs[0])):
            return None
        if not self._valid_q_ref_sequence(q_ref_sequences, batch_size=1):
            return None
        residual_sequences = evaluation.get("residual_sequences", sequence.unsqueeze(0))
        residual_sequences = residual_sequences.to(self.device)
        if residual_sequences.shape != q_ref_sequences.shape or not bool(torch.all(torch.isfinite(residual_sequences))):
            return None
        terms, predicted_next_state = self._diagnostics_from_evaluation(evaluation, 0)
        pred_state_sequence = evaluation.get("pred_states")
        if (
            not isinstance(pred_state_sequence, torch.Tensor)
            or pred_state_sequence.shape != (1, self.config.horizon + 1, 2 * self.config.action_dim)
            or not bool(torch.all(torch.isfinite(pred_state_sequence)))
        ):
            return None
        return costs[0], q_ref_sequences[0], residual_sequences[0], terms, predicted_next_state, pred_state_sequence[0]

    def plan(self, current_state: np.ndarray, previous_q_ref: np.ndarray, *, warm_start_shift_steps: int | None = None) -> CEMMPCResult:
        del current_state
        start_time = perf_counter()
        asynchronous_anchor = warm_start_shift_steps is not None
        mean = self._shifted_warm_start(warm_start_shift_steps) if asynchronous_anchor else self.mean.clone()
        std = self.initial_std.clone() if self.config.reset_std_each_step else self.std.clone()
        sampling_std_start_mean = float(std.mean().detach().cpu())
        best_sequence = None
        best_q_ref_sequence = None
        best_residual_sequence = None
        best_cost_terms: dict[str, float] = {}
        best_predicted_next_state = np.full(2 * self.config.action_dim, np.nan, dtype=np.float32)
        best_predicted_state_sequence: torch.Tensor | None = None
        best_cost = torch.as_tensor(float("inf"), device=self.device)
        elite_mean_cost = torch.as_tensor(float("inf"), device=self.device)
        final_samples: torch.Tensor | None = None
        final_evaluation: dict[str, torch.Tensor] | None = None
        final_elite_indices: torch.Tensor | None = None

        try:
            for _ in range(self.config.cem_iters):
                samples = self._sample_population(mean, std)
                evaluation = self.planner.evaluate(samples)
                costs = evaluation["costs"].to(self.device)
                if costs.ndim != 1 or costs.shape[0] != self.config.num_samples:
                    return self._fallback(previous_q_ref, start_time, "invalid_cost_shape")
                valid = torch.isfinite(costs)
                if not bool(torch.any(valid)):
                    return self._fallback(previous_q_ref, start_time, "all_costs_invalid")
                safe_costs = torch.where(valid, costs, torch.full_like(costs, float("inf")))
                elite_indices = torch.topk(safe_costs, k=self.num_elites, largest=False).indices
                elites = samples[elite_indices]
                elite_costs = safe_costs[elite_indices]
                elite_mean_cost = elite_costs.mean()
                candidate_best_cost, best_local = torch.min(safe_costs, dim=0)
                if candidate_best_cost < best_cost:
                    best_cost = candidate_best_cost
                    best_sequence = samples[best_local].detach().clone()
                    if "q_ref_sequences" in evaluation:
                        best_q_ref_sequence = evaluation["q_ref_sequences"][best_local].detach().clone()
                    residual_sequences = evaluation.get("residual_sequences", samples)
                    if residual_sequences.shape == samples.shape:
                        best_residual_sequence = residual_sequences[best_local].detach().clone()
                    best_cost_terms, best_predicted_next_state = self._diagnostics_from_evaluation(evaluation, int(best_local))
                    best_predicted_state_sequence = evaluation["pred_states"][best_local].detach().clone()
                new_mean = torch.clamp(elites.mean(dim=0), min=-1.0, max=1.0)
                new_std = elites.std(dim=0, unbiased=False).clamp_min(float(self.config.min_std))
                alpha = float(self.config.smoothing_alpha)
                mean = alpha * mean + (1.0 - alpha) * new_mean
                std = alpha * std + (1.0 - alpha) * new_std
                final_samples = samples.detach().clone()
                final_evaluation = evaluation
                final_elite_indices = elite_indices.detach().clone()
        except RuntimeError as exc:
            return self._fallback(previous_q_ref, start_time, f"planner_runtime_error:{exc}")

        if best_sequence is None or best_q_ref_sequence is None or best_residual_sequence is None or best_predicted_state_sequence is None:
            return self._fallback(previous_q_ref, start_time, "no_valid_sequence")
        if (
            not torch.all(torch.isfinite(best_sequence))
            or not self._valid_q_ref_sequence(best_q_ref_sequence.unsqueeze(0), batch_size=1)
            or best_residual_sequence.shape != best_sequence.shape
            or not bool(torch.all(torch.isfinite(best_residual_sequence)))
        ):
            return self._fallback(previous_q_ref, start_time, "invalid_selected_action")

        mean_cost = float("nan")
        baseline_cost = float("nan")
        selected_raw_sequence = best_sequence
        selected_q_ref_sequence = best_q_ref_sequence
        selected_residual_sequence = best_residual_sequence
        selected_cost = best_cost
        selection_mode = "best"
        selected_cost_terms = best_cost_terms
        selected_predicted_next_state = best_predicted_next_state
        selected_predicted_state_sequence = best_predicted_state_sequence
        mean_evaluation = None
        if self.config.execute in {"mean", "lowest_cost"}:
            mean_evaluation = self._evaluate_sequence(mean)
            if mean_evaluation is not None:
                mean_cost_tensor, mean_q_ref_sequence, mean_residual_sequence, mean_cost_terms, mean_predicted_next_state, mean_predicted_state_sequence = mean_evaluation
                mean_cost = float(mean_cost_tensor.detach().cpu())
                if self.config.execute == "mean":
                    selected_raw_sequence = mean
                    selected_q_ref_sequence = mean_q_ref_sequence
                    selected_residual_sequence = mean_residual_sequence
                    selected_cost = mean_cost_tensor
                    selection_mode = "mean"
                    selected_cost_terms = mean_cost_terms
                    selected_predicted_next_state = mean_predicted_next_state
                    selected_predicted_state_sequence = mean_predicted_state_sequence
            else:
                if self.config.execute == "mean":
                    selection_mode = "best_fallback_invalid_mean"

        baseline_evaluation = None
        if self.config.force_baseline_candidate:
            baseline_evaluation = self._evaluate_sequence(torch.zeros_like(mean))
            if baseline_evaluation is not None:
                baseline_cost = float(baseline_evaluation[0].detach().cpu())

        if self.config.execute == "lowest_cost":
            candidates: list[tuple[str, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float], np.ndarray, torch.Tensor]] = [
                (
                    "best",
                    best_sequence,
                    best_cost,
                    best_q_ref_sequence,
                    best_residual_sequence,
                    best_cost_terms,
                    best_predicted_next_state,
                    best_predicted_state_sequence,
                )
            ]
            if mean_evaluation is not None:
                mean_cost_tensor, mean_q_ref_sequence, mean_residual_sequence, mean_cost_terms, mean_predicted_next_state, mean_predicted_state_sequence = mean_evaluation
                mean_cost = float(mean_cost_tensor.detach().cpu())
                candidates.append(
                    ("mean", mean, mean_cost_tensor, mean_q_ref_sequence, mean_residual_sequence, mean_cost_terms, mean_predicted_next_state, mean_predicted_state_sequence)
                )
            if baseline_evaluation is not None:
                (
                    baseline_cost_tensor,
                    baseline_q_ref_sequence,
                    baseline_residual_sequence,
                    baseline_cost_terms,
                    baseline_predicted_next_state,
                    baseline_predicted_state_sequence,
                ) = baseline_evaluation
                candidates.append(
                    (
                        "baseline",
                        torch.zeros_like(mean),
                        baseline_cost_tensor,
                        baseline_q_ref_sequence,
                        baseline_residual_sequence,
                        baseline_cost_terms,
                        baseline_predicted_next_state,
                        baseline_predicted_state_sequence,
                    )
                )
            # Equal costs should prefer the deterministic baseline, then mean,
            # over a sampled action.  This makes the direct nominal fallback
            # stable instead of depending on population ordering.
            preference = {"baseline": 0, "mean": 1, "best": 2}
            (
                selected_name,
                selected_raw_sequence,
                selected_cost,
                selected_q_ref_sequence,
                selected_residual_sequence,
                selected_cost_terms,
                selected_predicted_next_state,
                selected_predicted_state_sequence,
            ) = min(
                candidates, key=lambda item: (float(item[2].detach().cpu()), preference[item[0]])
            )
            selection_mode = selected_name

        saved_mean = selected_raw_sequence if asynchronous_anchor else torch.cat(
            [selected_raw_sequence[1:], selected_raw_sequence[-1:].clone()], dim=0
        )
        self.mean = torch.clamp(saved_mean, min=-1.0, max=1.0).detach().clone()
        sampling_std_end_mean = float(std.mean().detach().cpu())
        self.std = self.initial_std.clone() if self.config.reset_std_each_step else std.detach().clone()
        selected_q_ref = selected_q_ref_sequence[0].detach().cpu().numpy().astype(np.float32)
        previous = np.asarray(previous_q_ref, dtype=np.float32)
        selected_delta = (selected_q_ref - previous).astype(np.float32)
        candidate_specs: list[tuple[tuple[str, ...], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]]] = [
            (
                ("selected",),
                selected_q_ref_sequence,
                selected_residual_sequence,
                selected_predicted_state_sequence,
                selected_cost,
                selected_cost_terms,
            )
        ]
        if baseline_evaluation is not None:
            baseline_cost_tensor, baseline_q_ref, baseline_residual, baseline_terms, _, baseline_predicted = baseline_evaluation
            duplicate = torch.allclose(baseline_q_ref, selected_q_ref_sequence, atol=1e-6, rtol=0.0)
            if duplicate:
                candidate_specs[0] = (
                    ("selected", "baseline"), selected_q_ref_sequence, selected_residual_sequence,
                    selected_predicted_state_sequence, selected_cost, selected_cost_terms,
                )
            else:
                candidate_specs.append((("baseline",), baseline_q_ref, baseline_residual, baseline_predicted, baseline_cost_tensor, baseline_terms))

        if final_samples is not None and final_evaluation is not None and final_elite_indices is not None:
            final_q = final_evaluation.get("q_ref_sequences")
            final_residual = final_evaluation.get("residual_sequences")
            final_predicted = final_evaluation.get("pred_states")
            final_costs = final_evaluation.get("costs")
            if all(isinstance(value, torch.Tensor) for value in (final_q, final_residual, final_predicted, final_costs)):
                distances = torch.sqrt(torch.mean(torch.square((final_q[final_elite_indices] - selected_q_ref_sequence) / self.alternative_distance_scale), dim=(1, 2)))
                valid_distinct = distances > 1e-6
                if bool(torch.any(valid_distinct)):
                    masked = torch.where(valid_distinct, distances, torch.full_like(distances, -1.0))
                    alternative_index = final_elite_indices[torch.argmax(masked)]
                    alternative_q = final_q[alternative_index]
                    if not any(torch.allclose(alternative_q, spec[1], atol=1e-6, rtol=0.0) for spec in candidate_specs):
                        alternative_terms, _ = self._diagnostics_from_evaluation(final_evaluation, int(alternative_index))
                        candidate_specs.append((
                            ("alternative_elite",), alternative_q, final_residual[alternative_index],
                            final_predicted[alternative_index], final_costs[alternative_index], alternative_terms,
                        ))

        branch_candidates = tuple(
            CEMCandidate(
                role_mask=roles,
                q_ref_sequence=q_ref.detach().cpu().numpy().astype(np.float32),
                residual_sequence=residual.detach().cpu().numpy().astype(np.float32),
                predicted_state_sequence=predicted.detach().cpu().numpy().astype(np.float32),
                predicted_cost=float(cost.detach().cpu()),
                cost_terms=terms,
            )
            for roles, q_ref, residual, predicted, cost, terms in candidate_specs
        )
        return CEMMPCResult(
            q_ref=selected_q_ref,
            delta_q_ref=selected_delta,
            best_cost=float(best_cost.detach().cpu()),
            mean_cost=mean_cost,
            baseline_cost=baseline_cost,
            selected_cost=float(selected_cost.detach().cpu()),
            elite_mean_cost=float(elite_mean_cost.detach().cpu()),
            selection_mode=selection_mode,
            planning_time=perf_counter() - start_time,
            failure=False,
            failure_reason="",
            best_sequence=best_sequence.detach().cpu().numpy().astype(np.float32),
            selected_action_sequence=selected_raw_sequence.detach().cpu().numpy().astype(np.float32),
            selected_q_ref_sequence=selected_q_ref_sequence.detach().cpu().numpy().astype(np.float32),
            selected_residual_sequence=selected_residual_sequence.detach().cpu().numpy().astype(np.float32),
            cost_terms=selected_cost_terms,
            predicted_next_state=selected_predicted_next_state,
            selected_predicted_state_sequence=selected_predicted_state_sequence.detach().cpu().numpy().astype(np.float32),
            sampling_std_start_mean=sampling_std_start_mean,
            sampling_std_end_mean=sampling_std_end_mean,
            branch_candidates=branch_candidates,
        )
