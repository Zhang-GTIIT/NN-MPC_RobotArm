from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[3]


class ModelCDatasetBuilderTests(unittest.TestCase):
    def test_selection_manifest_keeps_complete_selected_branch_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            # A main trajectory plus two contiguous 40-row counterfactual
            # branches, mirroring the collector's training shard layout.
            count = 120
            states = np.zeros((count, 12), dtype=np.float32)
            actions = np.zeros((count, 6), dtype=np.float32)
            branch_id = np.r_[np.full(40, 10), np.full(40, 11), np.full(40, -1)]
            np.savez_compressed(
                directory / "transitions_00000.npz", states=states, actions=actions, next_states=states,
                episode_id=np.r_[np.full(40, 10), np.full(40, 11), np.zeros(40)],
                split_group_id=np.zeros(count), source_id=np.ones(count), branch_id=branch_id,
                branch_kind_id=np.r_[np.ones(80), -np.ones(40)],
                valid_target=np.r_[np.r_[np.zeros(15), np.ones(25)], np.r_[np.zeros(15), np.ones(25)], np.ones(40)],
                context_only=np.r_[np.ones(15), np.zeros(25), np.ones(15), np.zeros(25), np.zeros(40)],
            )
            selection = directory / "selection.json"
            selection.write_text(json.dumps({"input_dir": str(directory.resolve()), "branch_kind_id": 1, "selected_branch_ids": [11]}), encoding="utf-8")
            output = directory / "C2.npz"
            subprocess.run(
                [sys.executable, str(ROOT / "scripts" / "model_c" / "build_dataset.py"), "--input", str(directory),
                 "--selection_manifest", str(selection), "--output_path", str(output)],
                check=True, cwd=ROOT,
            )
            with np.load(output) as data:
                self.assertEqual(data["states"].shape[0], 40)
                self.assertEqual(np.unique(data["episode_ids"]).tolist(), [0])
                self.assertEqual(int(np.sum(data["valid_target"])), 25)


if __name__ == "__main__":
    unittest.main()
