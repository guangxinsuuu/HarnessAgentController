from __future__ import annotations

import random
import unittest

from cgr_exp.analysis.replay_fairness import run_replay_fairness_matrix
from cgr_exp.datasets.level1_assignment import generate_assignment_fixture
from cgr_exp.datasets.level1_setcover import generate_setcover_fixture


class ReplayFairnessMatrixTest(unittest.TestCase):
    def test_replay_matrix_uses_same_replay_hash_for_all_policies(self) -> None:
        task = generate_assignment_fixture(random.Random(41), "fairness", 4, 3)
        rows = run_replay_fairness_matrix(task=task, seed=41, policies=("dfs", "random", "best_bound", "tree_gnn", "graph_gnn"), node_budget=1000, proposal_budget=3)
        self.assertEqual({row["policy"] for row in rows}, {"dfs", "random", "best_bound", "tree_gnn", "graph_gnn"})
        replay_hashes = {row["replay_hash"] for row in rows}
        self.assertEqual(len(replay_hashes), 1)
        self.assertTrue(all(row["proposal_calls"] <= 1000 for row in rows))
        self.assertTrue(all(row["same_proposal_support"] for row in rows))
        self.assertTrue(all(row["best_objective"] == task.exhaustive_oracle() for row in rows))

    def test_replay_matrix_supports_setcover_tasks(self) -> None:
        task = generate_setcover_fixture(random.Random(42), "fairness-setcover", universe_size=5, set_count=6)
        rows = run_replay_fairness_matrix(task=task, seed=42, policies=("dfs", "best_bound"), node_budget=1000, proposal_budget=4)
        self.assertEqual({row["policy"] for row in rows}, {"dfs", "best_bound"})
        self.assertTrue(all(row["best_objective"] == task.exhaustive_oracle() for row in rows))


if __name__ == "__main__":
    unittest.main()
