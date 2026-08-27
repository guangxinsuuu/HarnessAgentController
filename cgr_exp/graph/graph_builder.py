from __future__ import annotations

from itertools import combinations

from cgr_exp.graph.edges import GraphEdge
from cgr_exp.graph.tree_builder import build_tree_edges


def build_graph_edges(nodes, config=None):
    config = config or {}
    edges = list(build_tree_edges(nodes))

    if config.get("sibling_failure"):
        by_parent = {}
        for node in nodes:
            if node.parent_id is not None:
                by_parent.setdefault(node.parent_id, []).append(node)
        for siblings in by_parent.values():
            for a, b in combinations(siblings, 2):
                edges.append(GraphEdge(a.node_id, b.node_id, "sibling_failure", "nodes share the same expanded parent"))

    if config.get("shared_constraint"):
        for a, b in combinations(nodes, 2):
            shared = set(a.verified_constraints).intersection(b.verified_constraints)
            if shared:
                edges.append(GraphEdge(a.node_id, b.node_id, "shared_constraint", f"shared verified constraint {sorted(shared)[0]}"))

    if config.get("equivalent_state"):
        seen = {}
        for node in nodes:
            key = repr(sorted(node.structured_state.get("assignments", {}).items()))
            if key in seen:
                edges.append(GraphEdge(seen[key], node.node_id, "equivalent_state", "same structured assignment map"))
            else:
                seen[key] = node.node_id

    if config.get("tool_observation"):
        by_observation = {}
        for node in nodes:
            if node.observation:
                by_observation.setdefault(repr(sorted(node.observation.items())), []).append(node)
        for group in by_observation.values():
            for a, b in combinations(group, 2):
                edges.append(GraphEdge(a.node_id, b.node_id, "tool_observation", "same observed tool outcome"))

    return edges
