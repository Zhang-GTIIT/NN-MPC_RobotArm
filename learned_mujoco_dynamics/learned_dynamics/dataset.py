from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, Subset, random_split
from tqdm import tqdm


class DynamicsDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        next_states: np.ndarray,
        model_type: str = "mlp",
        history_len: int = 1,
        episode_ids: np.ndarray | None = None,
        target_mode: str = "delta_state",
    ) -> None:
        if model_type not in {"mlp", "gru", "transformer"}:
            raise ValueError(f"model_type must be one of mlp/gru/transformer, got {model_type!r}")
        if target_mode not in {"delta_state", "delta_dq"}:
            raise ValueError(f"target_mode must be 'delta_state' or 'delta_dq', got {target_mode!r}")
        if history_len <= 0:
            raise ValueError(f"history_len must be positive, got {history_len}")
        if states.ndim != 2 or actions.ndim != 2 or next_states.ndim != 2:
            raise ValueError("states, actions, and next_states must all be rank-2 arrays")
        if len(states) != len(actions) or len(states) != len(next_states):
            raise ValueError(
                f"Array length mismatch: states={len(states)}, actions={len(actions)}, next_states={len(next_states)}"
            )
        if states.shape != next_states.shape:
            raise ValueError(f"states and next_states must have same shape, got {states.shape} and {next_states.shape}")
        if model_type != "mlp" and len(states) < history_len:
            raise ValueError(f"Need at least history_len={history_len} samples, got {len(states)}")
        if episode_ids is not None:
            if episode_ids.ndim != 1:
                raise ValueError(f"episode_ids must be rank-1, got shape {episode_ids.shape}")
            if len(episode_ids) != len(states):
                raise ValueError(f"episode_ids length={len(episode_ids)} does not match samples={len(states)}")

        self.states = torch.as_tensor(states, dtype=torch.float32)
        self.actions = torch.as_tensor(actions, dtype=torch.float32)
        self.next_states = torch.as_tensor(next_states, dtype=torch.float32)
        self.model_type = model_type
        self.history_len = history_len
        self.target_mode = target_mode
        self.episode_ids = None if episode_ids is None else torch.as_tensor(episode_ids, dtype=torch.long)
        self.sequence_indices = self._build_sequence_indices()

    def _build_sequence_indices(self) -> np.ndarray | None:
        if self.model_type == "mlp" or self.episode_ids is None:
            return None
        episode_ids = self.episode_ids.cpu().numpy()
        boundaries = np.flatnonzero(np.diff(episode_ids) != 0) + 1
        run_starts = np.concatenate(([0], boundaries))
        run_ends = np.concatenate((boundaries, [len(episode_ids)]))
        valid_runs = (run_ends - run_starts) >= self.history_len
        starts_by_run = [
            np.arange(start, end - self.history_len + 1, dtype=np.int64)
            for start, end in zip(run_starts[valid_runs], run_ends[valid_runs])
        ]
        if not starts_by_run:
            raise ValueError(
                f"No valid sequence windows for history_len={self.history_len}; "
                "each episode must contain at least history_len samples."
            )
        return np.concatenate(starts_by_run)

    @property
    def state_dim(self) -> int:
        return int(self.states.shape[1])

    @property
    def action_dim(self) -> int:
        return int(self.actions.shape[1])

    @property
    def target_dim(self) -> int:
        if self.target_mode == "delta_dq":
            return self.state_dim // 2
        return self.state_dim

    def _target_at(self, index: int) -> torch.Tensor:
        delta = self.next_states[index] - self.states[index]
        if self.target_mode == "delta_dq":
            return delta[self.state_dim // 2 :]
        return delta

    def __len__(self) -> int:
        if self.model_type == "mlp":
            return int(self.states.shape[0])
        if self.sequence_indices is not None:
            return len(self.sequence_indices)
        # Backward compatibility for v1 npz files with no episode_ids.
        return int(self.states.shape[0] - self.history_len + 1)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.model_type == "mlp":
            x = torch.cat([self.states[index], self.actions[index]], dim=0)
            y = self._target_at(index)
            return x, y

        start = self.sequence_indices[index] if self.sequence_indices is not None else index
        end = start + self.history_len
        x = torch.cat([self.states[start:end], self.actions[start:end]], dim=-1)
        y = self._target_at(end - 1)
        return x, y


class RolloutDynamicsDataset(DynamicsDataset):
    def __init__(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        next_states: np.ndarray,
        model_type: str = "mlp",
        history_len: int = 1,
        episode_ids: np.ndarray | None = None,
        target_mode: str = "delta_state",
        rollout_steps: int = 1,
    ) -> None:
        if rollout_steps <= 0:
            raise ValueError(f"rollout_steps must be positive, got {rollout_steps}")
        self.rollout_steps = int(rollout_steps)
        super().__init__(
            states,
            actions,
            next_states,
            model_type=model_type,
            history_len=history_len,
            episode_ids=episode_ids,
            target_mode=target_mode,
        )

    def _build_sequence_indices(self) -> np.ndarray | None:
        if self.model_type == "mlp" and self.episode_ids is None:
            max_start = int(self.states.shape[0] - self.rollout_steps)
            if max_start < 0:
                raise ValueError(
                    f"No valid rollout windows for rollout_steps={self.rollout_steps}; "
                    f"dataset has {self.states.shape[0]} samples."
                )
            return np.arange(0, max_start + 1, dtype=np.int64)

        if self.episode_ids is None:
            max_start = int(self.states.shape[0] - self.history_len - self.rollout_steps + 1)
            if max_start < 0:
                raise ValueError(
                    f"No valid rollout windows for history_len={self.history_len}, "
                    f"rollout_steps={self.rollout_steps}; dataset has {self.states.shape[0]} samples."
                )
            return np.arange(0, max_start + 1, dtype=np.int64)

        episode_ids = self.episode_ids.cpu().numpy()
        boundaries = np.flatnonzero(np.diff(episode_ids) != 0) + 1
        run_starts = np.concatenate(([0], boundaries))
        run_ends = np.concatenate((boundaries, [len(episode_ids)]))
        min_window = 1 if self.model_type == "mlp" else self.history_len
        valid_runs = (run_ends - run_starts) >= (min_window + self.rollout_steps)
        starts_by_run = [
            np.arange(start, end - min_window - self.rollout_steps + 2, dtype=np.int64)
            for start, end in zip(run_starts[valid_runs], run_ends[valid_runs])
        ]
        if not starts_by_run:
            raise ValueError(
                f"No valid rollout windows for history_len={self.history_len}, "
                f"rollout_steps={self.rollout_steps}; each episode must contain enough samples."
            )
        return np.concatenate(starts_by_run)

    def __len__(self) -> int:
        if self.sequence_indices is None:
            raise RuntimeError("RolloutDynamicsDataset expected sequence_indices to be initialized")
        return len(self.sequence_indices)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.sequence_indices is None:
            raise RuntimeError("RolloutDynamicsDataset expected sequence_indices to be initialized")
        start = int(self.sequence_indices[index])
        if self.model_type == "mlp":
            current_index = start
            x = torch.cat([self.states[current_index], self.actions[current_index]], dim=0)
        else:
            end = start + self.history_len
            current_index = end - 1
            x = torch.cat([self.states[start:end], self.actions[start:end]], dim=-1)
        y = self._target_at(current_index)
        rollout_end = current_index + self.rollout_steps
        rollout_actions = self.actions[current_index:rollout_end]
        rollout_next_states = self.next_states[current_index:rollout_end]
        return x, y, rollout_actions, rollout_next_states


def load_npz_dataset(
    data_path: Path,
    model_type: str,
    history_len: int,
    target_mode: str = "delta_state",
) -> DynamicsDataset:
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset file does not exist: {data_path}")
    data = np.load(data_path)
    required = {"states", "actions", "next_states"}
    missing = required.difference(data.files)
    if missing:
        raise KeyError(f"Dataset file {data_path} is missing arrays: {sorted(missing)}")
    episode_ids = data["episode_ids"] if "episode_ids" in data.files else None
    return DynamicsDataset(
        data["states"],
        data["actions"],
        data["next_states"],
        model_type,
        history_len,
        episode_ids,
        target_mode=target_mode,
    )


def load_rollout_npz_dataset(
    data_path: Path,
    model_type: str,
    history_len: int,
    target_mode: str = "delta_state",
    rollout_steps: int = 1,
) -> RolloutDynamicsDataset:
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset file does not exist: {data_path}")
    data = np.load(data_path)
    required = {"states", "actions", "next_states"}
    missing = required.difference(data.files)
    if missing:
        raise KeyError(f"Dataset file {data_path} is missing arrays: {sorted(missing)}")
    episode_ids = data["episode_ids"] if "episode_ids" in data.files else None
    return RolloutDynamicsDataset(
        data["states"],
        data["actions"],
        data["next_states"],
        model_type,
        history_len,
        episode_ids,
        target_mode=target_mode,
        rollout_steps=rollout_steps,
    )


def split_dataset(
    dataset: DynamicsDataset,
    val_fraction: float = 0.1,
    seed: int = 0,
    train_sample_stride: int = 1,
    val_sample_stride: int = 1,
    show_progress: bool = False,
) -> Tuple[Subset[DynamicsDataset], Subset[DynamicsDataset]]:
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")
    if train_sample_stride <= 0:
        raise ValueError(f"train_sample_stride must be positive, got {train_sample_stride}")
    if val_sample_stride <= 0:
        raise ValueError(f"val_sample_stride must be positive, got {val_sample_stride}")

    def apply_sample_stride(indices: np.ndarray, stride: int) -> np.ndarray:
        if stride == 1:
            return indices.astype(np.int64, copy=False)
        if dataset.episode_ids is None or dataset.sequence_indices is None:
            return indices[::stride].astype(np.int64, copy=False)
        sample_episode_ids = dataset.episode_ids.cpu().numpy()[dataset.sequence_indices]
        strided_by_episode = [
            episode_indices[::stride]
            for episode_id in np.unique(sample_episode_ids[indices])
            for episode_indices in [indices[sample_episode_ids[indices] == episode_id]]
        ]
        if not strided_by_episode:
            return indices[:1].astype(np.int64, copy=False)
        return np.concatenate(strided_by_episode).astype(np.int64, copy=False)

    def split_sequence_indices_by_episode() -> tuple[np.ndarray, np.ndarray] | None:
        if dataset.episode_ids is None or dataset.sequence_indices is None:
            return None
        episode_ids = dataset.episode_ids.cpu().numpy()
        boundaries = np.flatnonzero(np.diff(episode_ids) != 0) + 1
        run_starts = np.concatenate(([0], boundaries))
        run_ends = np.concatenate((boundaries, [len(episode_ids)]))
        run_episode_ids = episode_ids[run_starts]
        if len(run_episode_ids) < 2:
            return None

        rng = np.random.default_rng(seed)
        shuffled = np.unique(run_episode_ids)
        rng.shuffle(shuffled)
        val_episode_count = min(len(shuffled) - 1, max(1, int(len(shuffled) * val_fraction)))
        val_episodes = set(int(episode_id) for episode_id in shuffled[:val_episode_count])
        sequence_indices = dataset.sequence_indices
        train_parts: list[np.ndarray] = []
        val_parts: list[np.ndarray] = []
        runs = zip(run_starts, run_ends, run_episode_ids)
        if show_progress:
            runs = tqdm(list(runs), desc="split episodes", unit="episode")
        for run_start, run_end, episode_id in runs:
            sample_start = int(np.searchsorted(sequence_indices, run_start, side="left"))
            sample_end = int(np.searchsorted(sequence_indices, run_end, side="left"))
            if sample_start >= sample_end:
                continue
            stride = val_sample_stride if int(episode_id) in val_episodes else train_sample_stride
            indices = np.arange(sample_start, sample_end, stride, dtype=np.int64)
            if int(episode_id) in val_episodes:
                val_parts.append(indices)
            else:
                train_parts.append(indices)
        if not train_parts or not val_parts:
            return None
        return np.concatenate(train_parts), np.concatenate(val_parts)

    val_size = max(1, int(len(dataset) * val_fraction))
    train_size = len(dataset) - val_size
    if train_size <= 0:
        raise ValueError(f"Dataset too small to split: len={len(dataset)}, val_fraction={val_fraction}")
    if dataset.episode_ids is not None and dataset.sequence_indices is not None:
        episode_split = split_sequence_indices_by_episode()
        if episode_split is not None:
            train_indices, val_indices = episode_split
            return Subset(dataset, train_indices), Subset(dataset, val_indices)
    generator = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(dataset, [train_size, val_size], generator=generator)
    train_indices = apply_sample_stride(np.asarray(train_set.indices, dtype=np.int64), train_sample_stride)
    val_indices = apply_sample_stride(np.asarray(val_set.indices, dtype=np.int64), val_sample_stride)
    return Subset(dataset, train_indices), Subset(dataset, val_indices)
