from __future__ import annotations


class GraphGNNRanker:
    """Graph message-passing baseline using non-tree relation edges when available."""

    name = "graph-gnn"

    def score_nodes(self, nodes, search_context):
        relation_counts = {node.node_id: 0 for node in nodes}
        shared_counts = {node.node_id: 0 for node in nodes}
        equivalent_counts = {node.node_id: 0 for node in nodes}
        for edge in search_context.get("edges", []):
            for endpoint in (edge.source, edge.target):
                if endpoint not in relation_counts:
                    continue
                if edge.edge_type != "parent_child":
                    relation_counts[endpoint] += 1
                if edge.edge_type == "shared_constraint":
                    shared_counts[endpoint] += 1
                if edge.edge_type == "equivalent_state":
                    equivalent_counts[endpoint] += 1
        scores = {}
        for node in nodes:
            lower = node.lower_bound if node.lower_bound is not None else 1e6
            scores[node.node_id] = float(
                -lower
                + 0.02 * relation_counts[node.node_id]
                + 0.03 * shared_counts[node.node_id]
                - 0.04 * equivalent_counts[node.node_id]
            )
        return scores
