from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
DYNAMICS_ROOT = ROOT / "dynamics_modeling"
if str(DYNAMICS_ROOT) not in sys.path:
    sys.path.insert(0, str(DYNAMICS_ROOT))

from neural_dynamics.dataset import RolloutDynamicsDataset, split_dataset


class ModelCDatasetTests(unittest.TestCase):
    def test_40_row_branch_has_six_valid_gru_rollout_windows(self) -> None:
        # 15 context-only rows + 25 valid transitions.  The current target is
        # the final history row, so L-H-R+2 = 40-16-20+2 = 6.
        values = np.zeros((40, 12), dtype=np.float32)
        actions = np.zeros((40, 6), dtype=np.float32)
        dataset = RolloutDynamicsDataset(
            values, actions, values, model_type="gru", history_len=16,
            episode_ids=np.zeros(40, dtype=np.int64),
            split_group_ids=np.zeros(40, dtype=np.int64),
            valid_target=np.r_[np.zeros(15, dtype=np.int8), np.ones(25, dtype=np.int8)],
            target_mode="delta_dq", rollout_steps=20,
        )
        self.assertEqual(len(dataset), 6)

    def test_parent_split_group_keeps_main_and_branch_together(self) -> None:
        # Main episode ids and branch ids deliberately differ, while parent
        # split_group ids bind them into the same train/validation partition.
        length = 40
        values = np.zeros((length * 4, 12), dtype=np.float32)
        actions = np.zeros((length * 4, 6), dtype=np.float32)
        episodes = np.repeat(np.array([0, 100, 1, 101]), length)
        groups = np.repeat(np.array([0, 0, 1, 1]), length)
        dataset = RolloutDynamicsDataset(
            values, actions, values, model_type="gru", history_len=16,
            episode_ids=episodes, split_group_ids=groups,
            valid_target=np.ones(len(values), dtype=np.int8), target_mode="delta_dq", rollout_steps=20,
        )
        train, validation = split_dataset(dataset, val_fraction=0.5, seed=7)
        sequence_starts = dataset.sequence_indices
        train_groups = set(groups[sequence_starts[np.asarray(train.indices)]])
        validation_groups = set(groups[sequence_starts[np.asarray(validation.indices)]])
        self.assertTrue(train_groups.isdisjoint(validation_groups))
        self.assertEqual(train_groups | validation_groups, {0, 1})


if __name__ == "__main__":
    unittest.main()
