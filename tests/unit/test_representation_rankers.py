from __future__ import annotations

import unittest

from cgr_exp.datasets.level1_assignment import AssignmentTask, make_node
from cgr_exp.graph.graph_builder import build_graph_edges
from cgr_exp.models.graph_gnn import GraphGNNRanker
from cgr_exp.models.history_transformer import HistoryTransformerRanker
from cgr_exp.models.mlp_ranker import MLPRanker
from cgr_exp.models.tree_gnn import TreeGNNRanker


def fixture_nodes():
    task = AssignmentTask(
        task_id="ranker-fixture",
        durations={"T0": 2, "T1": 3},
        resources=("R0", "R1"),
        allowed={"T0": ("R0", "R1"), "T1": ("R0", "R1")},
    )
    root = task.root_node()
    actions = task.candidate_actions(root, 2)
    child_a = make_node(task, {actions[0].task: actions[0].resource}, root.node_id, actions[0], "child-a", None)
    child_b = make_node(task, {actions[1].task: actions[1].resource}, root.node_id, actions[1], "child-b", None)
    duplicate_a = child_a.with_updates(node_id="child-a-duplicate")
    return [root, child_a, child_b, duplicate_a]


class RepresentationRankerTest(unittest.TestCase):
    def test_all_rankers_score_every_node(self) -> None:
        nodes = fixture_nodes()
        edges = build_graph_edges(nodes, {"sibling_failure": True, "shared_constraint": True, "equivalent_state": True})
        for ranker in (MLPRanker(), HistoryTransformerRanker(), TreeGNNRanker(), GraphGNNRanker()):
            scores = ranker.score_nodes(nodes, {"edges": edges})
            self.assertEqual(set(scores), {n.node_id for n in nodes})
            self.assertTrue(all(isinstance(v, float) for v in scores.values()))

    def test_graph_ranker_can_use_extra_edges_beyond_tree(self) -> None:
        nodes = fixture_nodes()
        tree_edges = build_graph_edges(nodes, {})
        graph_edges = build_graph_edges(nodes, {"sibling_failure": True, "shared_constraint": True, "equivalent_state": True})
        tree_scores = TreeGNNRanker().score_nodes(nodes, {"edges": tree_edges})
        graph_scores = GraphGNNRanker().score_nodes(nodes, {"edges": graph_edges})
        self.assertNotEqual(tree_scores, graph_scores)


if __name__ == "__main__":
    unittest.main()

