from __future__ import annotations

import random
import unittest

from cgr_exp.datasets.level1_assignment import AssignmentTask, generate_assignment_fixture, make_node
from cgr_exp.graph.edge_types import VALID_EDGE_TYPES
from cgr_exp.graph.graph_builder import build_graph_edges
from cgr_exp.graph.tree_builder import build_tree_edges
from cgr_exp.search.best_bound import BestBoundPolicy
from cgr_exp.search.controller import run_frontier_search
from cgr_exp.schemas import SearchNode
from cgr_exp.verification.level1_verifier import Level1AssignmentVerifier


def trace_nodes(result) -> list[SearchNode]:
    out = []
    for event in result.trace:
        if event.get("event") == "expand":
            out.append(SearchNode.from_dict(event["node"]))
    return out


class GraphConstructionTest(unittest.TestCase):
    def test_tree_edges_are_parent_child_only_with_valid_endpoints(self) -> None:
        task = generate_assignment_fixture(random.Random(12), "graph-tree", 4, 3)
        result = run_frontier_search(task, BestBoundPolicy(), Level1AssignmentVerifier(seed=1), 100, 3, seed=1)
        nodes = trace_nodes(result)
        node_ids = {n.node_id for n in nodes}
        edges = build_tree_edges(nodes)
        self.assertTrue(edges)
        for edge in edges:
            self.assertIn(edge.edge_type, VALID_EDGE_TYPES)
            self.assertEqual(edge.edge_type, "parent_child")
            self.assertIn(edge.source, node_ids)
            self.assertIn(edge.target, node_ids)
            self.assertTrue(edge.reason)

    def test_graph_edges_include_requested_extra_types(self) -> None:
        task = AssignmentTask(
            task_id="graph-full",
            durations={"T0": 2, "T1": 3},
            resources=("R0", "R1"),
            allowed={"T0": ("R0", "R1"), "T1": ("R0", "R1")},
        )
        root = task.root_node()
        actions = task.candidate_actions(root, 3)
        child_a = make_node(task, {actions[0].task: actions[0].resource}, root.node_id, actions[0], "child-a", None)
        child_b = make_node(task, {actions[1].task: actions[1].resource}, root.node_id, actions[1], "child-b", None)
        duplicate_a = child_a.with_updates(node_id="child-a-duplicate")
        nodes = [root, child_a, child_b, duplicate_a]
        edges = build_graph_edges(
            nodes,
            config={
                "sibling_failure": True,
                "shared_constraint": True,
                "equivalent_state": True,
                "tool_observation": False,
            },
        )
        edge_types = {e.edge_type for e in edges}
        self.assertIn("parent_child", edge_types)
        self.assertIn("sibling_failure", edge_types)
        self.assertIn("shared_constraint", edge_types)
        self.assertIn("equivalent_state", edge_types)
        for edge in edges:
            self.assertIn(edge.edge_type, VALID_EDGE_TYPES)
            self.assertTrue(edge.reason)


if __name__ == "__main__":
    unittest.main()
