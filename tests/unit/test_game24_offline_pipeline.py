from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from cgr_exp.datasets.game24 import Game24Action, Game24Task, game24_feature_vector
from cgr_exp.game24_experiments import (
    FORBIDDEN_GAME24_INFERENCE_FEATURES,
    generate_game24_ranking_pairs,
    game24_split_manifest,
    generate_game24_trajectories,
    run_game24_frontier_evaluation,
    run_game24_offline_experiment,
)
from cgr_exp.proposers.replay import ReplayProposer, build_replay_from_task
from cgr_exp.search.dfs import DFSPolicy
from cgr_exp.search.oracle_policy import OraclePolicy


class Game24OfflinePipelineTest(unittest.TestCase):
    def test_exact_oracle_reachability_and_min_future_expansions(self) -> None:
        solvable = Game24Task(task_id="solvable", numbers=(4, 5, 6, 10))
        unsolvable = Game24Task(task_id="unsolvable", numbers=(1, 1, 1, 1))
        self.assertTrue(solvable.can_reach_24(solvable.root_node()))
        self.assertFalse(unsolvable.can_reach_24(unsolvable.root_node()))
        self.assertEqual(solvable.min_future_expansions(solvable.root_node()), 3)
        next_node = solvable.apply(solvable.root_node(), Game24Action(left=10, right=4, operator="-"))
        self.assertEqual(solvable.min_future_expansions(next_node), 2)

    def test_oracle_labels_do_not_enter_inference_features(self) -> None:
        task = Game24Task(task_id="features", numbers=(4, 5, 6, 10))
        feature_names, _ = game24_feature_vector(task, task.root_node())
        self.assertFalse(set(feature_names).intersection(FORBIDDEN_GAME24_INFERENCE_FEATURES))

    def test_trajectory_records_oracle_supervision_as_label_only(self) -> None:
        task = Game24Task(task_id="trace", numbers=(4, 5, 6, 10))
        rows = generate_game24_trajectories([task], node_budget=50)
        self.assertTrue(rows)
        root = rows[0]
        self.assertIn("can_reach_24", root.label)
        self.assertIn("minimum_future_expansions", root.label)
        self.assertNotIn("can_reach_24", root.inference_features)
        self.assertNotIn("minimum_future_expansions", root.inference_features)

    def test_replay_proposal_set_is_independent_of_search_policy(self) -> None:
        task = Game24Task(task_id="replay", numbers=(4, 5, 6, 10))
        replay = build_replay_from_task(task, k=20)
        root = task.root_node()
        dfs_actions = ReplayProposer(replay).propose(task, root, 20)
        oracle_actions = ReplayProposer(replay).propose(task, root, 20)
        self.assertEqual([a.to_dict() for a in dfs_actions], [a.to_dict() for a in oracle_actions])
        self.assertNotEqual(DFSPolicy().score(root), OraclePolicy(task).score(root))

    def test_ranking_pairs_prefer_lower_remaining_cost_then_reachable(self) -> None:
        task = Game24Task(task_id="pairs", numbers=(4, 5, 6, 10))
        pairs = generate_game24_ranking_pairs([task], node_budget=50)
        self.assertTrue(pairs)
        for pair in pairs[:10]:
            self.assertLess(pair.positive_label["sort_key"], pair.negative_label["sort_key"])

    def test_offline_experiment_writes_baseline_and_model_tables(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            metrics = run_game24_offline_experiment(Path(tmp), limit=12, seed=9)
            table = (Path(tmp) / "game24_offline_table.csv").read_text(encoding="utf-8")
            frontier_table = (Path(tmp) / "game24_frontier_metrics.csv").read_text(encoding="utf-8")
            ablation_table = (Path(tmp) / "game24_label_ablation.csv").read_text(encoding="utf-8")
            self.assertIn("Random", table)
            self.assertIn("DFS", table)
            self.assertIn("Oracle", table)
            self.assertIn("MLP", table)
            self.assertIn("Tree-GNN", table)
            self.assertIn("Graph-GNN", table)
            self.assertIn("selection_regret_mean", frontier_table)
            self.assertIn("frontier_top1_accuracy", frontier_table)
            self.assertIn("reachable", ablation_table)
            self.assertIn("minimum_future_expansions", ablation_table)
            self.assertIn("frontier_relative", ablation_table)
            self.assertGreater(metrics["pair_count"], 0)
            self.assertEqual(metrics["learned_ranker"]["training"], "pairwise oracle ranking pairs from exact Game24 trajectories")
            self.assertIn("frontier_metrics", metrics)
            self.assertIn("label_ablation", metrics)
            self.assertLessEqual(metrics["policies"]["Oracle"]["nodes_mean"], metrics["policies"]["Random"]["nodes_mean"])
            self.assertLessEqual(metrics["policies"]["MLP"]["nodes_mean"], metrics["policies"]["Random"]["nodes_mean"])
            self.assertLessEqual(metrics["policies"]["Tree-GNN"]["nodes_mean"], metrics["policies"]["Random"]["nodes_mean"])
            self.assertLessEqual(metrics["policies"]["Graph-GNN"]["nodes_mean"], metrics["policies"]["Random"]["nodes_mean"])

    def test_frontier_evaluation_reports_selection_regret_and_top1(self) -> None:
        task = Game24Task(task_id="frontier", numbers=(4, 5, 6, 10))
        metrics = run_game24_frontier_evaluation(task, OraclePolicy(task), seed=3, node_budget=50)
        self.assertEqual(metrics["solved"], True)
        self.assertEqual(metrics["selection_regret_mean"], 0.0)
        self.assertEqual(metrics["frontier_top1_accuracy"], 1.0)
        self.assertGreaterEqual(metrics["ranking_accuracy"], 0.0)
        self.assertLessEqual(metrics["ranking_accuracy"], 1.0)

    def test_official_split_windows_match_tot_style_indices_when_available(self) -> None:
        manifest = game24_split_manifest(1000)
        self.assertEqual(manifest["train"][0], "game24-0000")
        self.assertEqual(manifest["train"][-1], "game24-0699")
        self.assertEqual(manifest["validation"][0], "game24-0700")
        self.assertEqual(manifest["development"][0], "game24-0800")
        self.assertEqual(manifest["official_final_test"][0], "game24-0900")
        all_ids = [task_id for ids in manifest.values() for task_id in ids]
        self.assertEqual(len(all_ids), len(set(all_ids)))


if __name__ == "__main__":
    unittest.main()
