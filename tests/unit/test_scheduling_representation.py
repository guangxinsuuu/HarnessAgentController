from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from cgr_exp.scheduling_micro import SCHEDULING_FEATURE_NAMES, generate_scheduling_micro_instances
from cgr_exp.scheduling_representation import (
    GRAPH_FEATURE_NAMES,
    TREE_FEATURE_NAMES,
    run_scheduling_representation_ablation,
    scheduling_representation_vector,
)


class SchedulingRepresentationTest(unittest.TestCase):
    def test_representation_vectors_add_tree_and_graph_features(self) -> None:
        task = generate_scheduling_micro_instances(train_count=1, validation_count=0, test_count=0, seed=21)["train"][0]
        root = task.root_node()
        action = task.candidate_actions(root, 1)[0]
        child = task.apply(root, action)
        mlp = scheduling_representation_vector(task, child, "mlp")
        tree = scheduling_representation_vector(task, child, "tree_gnn")
        graph = scheduling_representation_vector(task, child, "graph_gnn")
        self.assertEqual(len(mlp), len(SCHEDULING_FEATURE_NAMES))
        self.assertEqual(len(tree), len(TREE_FEATURE_NAMES))
        self.assertEqual(len(graph), len(GRAPH_FEATURE_NAMES))
        self.assertGreater(len(tree), len(mlp))
        self.assertGreater(len(graph), len(tree))

    def test_representation_ablation_writes_required_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            metrics = run_scheduling_representation_ablation(
                Path(tmp),
                seed=22,
                train_count=8,
                validation_count=3,
                test_count=4,
                job_count=5,
                machine_count=2,
                node_budget=70,
                action_budget=6,
                records_per_instance=12,
                rank_epochs=5,
            )
            self.assertEqual(metrics["benchmark"], "scheduling_representation_ablation")
            self.assertEqual(set(metrics["pair_counts"]), {"mlp", "tree_gnn", "graph_gnn"})
            root = Path(tmp)
            self.assertTrue((root / "scheduling_representation_table.csv").exists())
            self.assertTrue((root / "scheduling_representation_metrics.json").exists())
            saved = json.loads((root / "scheduling_representation_metrics.json").read_text(encoding="utf-8"))
            self.assertTrue({"MLP", "Tree-GNN", "Graph-GNN", "Oracle"}.issubset(saved["methods"]))


if __name__ == "__main__":
    unittest.main()
