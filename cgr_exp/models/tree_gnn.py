from __future__ import annotations


class TreeGNNRanker:
    """Tree message-passing baseline using parent-child edges only."""

    name = "tree-gnn"

    def score_nodes(self, nodes, search_context):
        node_map = {node.node_id: node for node in nodes}
        child_counts = {node.node_id: 0 for node in nodes}
        for edge in search_context.get("edges", []):
            if edge.edge_type == "parent_child" and edge.source in child_counts:
                child_counts[edge.source] += 1
        scores = {}
        for node in nodes:
            lower = node.lower_bound if node.lower_bound is not None else 1e6
            parent_bonus = 0.0
            if node.parent_id in node_map:
                parent_lower = node_map[node.parent_id].lower_bound
                if parent_lower is not None:
                    parent_bonus = max(0.0, parent_lower - lower)
            scores[node.node_id] = float(-lower + 0.04 * parent_bonus - 0.01 * child_counts[node.node_id])
        return scores
