from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location("model_c_round2_selection", ROOT / "scripts" / "model_c" / "select_round2_cases.py")
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class Round2SelectionTests(unittest.TestCase):
    def test_nonexclusive_labels_do_not_duplicate_selected_branches(self) -> None:
        rows = []
        for branch_id in range(20):
            rows.append({
                "branch_id": branch_id, "parent_main_episode_id": branch_id // 2,
                "activation_step": (branch_id // 2) * 10, "role_mask": 1 if branch_id % 2 == 0 else 2,
                "model_error": float(branch_id), "predicted_cost": float(branch_id % 3),
                "realized_cost": float((branch_id + 1) % 3), "tracking_error": float(branch_id),
                "residual_fraction": 0.95 if branch_id % 4 == 0 else 0.1,
                "recovery_active": float(branch_id % 5 == 0), "packet_fallback": 0.0,
            })
        MODULE._label_rows(rows)
        selected, counts = MODULE._select(rows, 10, MODULE.DEFAULT_WEIGHTS)
        ids = [row["branch_id"] for row in selected]
        self.assertEqual(len(ids), 10)
        self.assertEqual(len(ids), len(set(ids)))
        self.assertGreater(sum(counts.values()), 0)
        self.assertTrue(any(row["ranking_flip"] for row in rows))


if __name__ == "__main__":
    unittest.main()
