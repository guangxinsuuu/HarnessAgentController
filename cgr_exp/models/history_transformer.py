from __future__ import annotations


class HistoryTransformerRanker:
    """Path/history baseline: uses ancestor depth and bound-improvement hints."""

    name = "history-path"

    def score_nodes(self, nodes, search_context):
        del search_context
        scores = {}
        for node in nodes:
            lower = node.lower_bound if node.lower_bound is not None else 1e6
            ancestor_len = node.universal_features.get("ancestor_length", float(node.depth))
            uncertain = node.verifier_features.get("uncertain_constraint_count") or 0.0
            scores[node.node_id] = float(-lower + 0.03 * ancestor_len - 0.02 * uncertain)
        return scores
