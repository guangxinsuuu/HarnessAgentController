from __future__ import annotations


class MLPRanker:
    """Dependency-free node-local scorer used as the MLP baseline in smoke runs."""

    name = "node-local-mlp"

    def score_nodes(self, nodes, search_context):
        del search_context
        scores = {}
        for node in nodes:
            lower = node.lower_bound if node.lower_bound is not None else 1e6
            depth = node.universal_features.get("depth", float(node.depth))
            branch = node.universal_features.get("branch_factor", 0.0)
            scores[node.node_id] = float(-lower + 0.05 * depth - 0.01 * branch)
        return scores


__all__ = ["MLPRanker"]
