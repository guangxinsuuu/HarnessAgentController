from __future__ import annotations

from cgr_exp.features import state_features


class LearnedRankerPolicy:
    def __init__(self, ranker, problem=None):
        self.ranker = ranker
        self.problem = problem

    def score(self, node) -> float:
        if hasattr(self.ranker, "score_node"):
            return float(self.ranker.score_node(node))
        if self.problem is None:
            raise ValueError("problem is required when ranker consumes legacy feature vectors")
        return float(self.ranker.score(state_features(self.problem, node)))
