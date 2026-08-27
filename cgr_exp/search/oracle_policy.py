from __future__ import annotations


class OraclePolicy:
    def __init__(self, task):
        self.task = task

    def score(self, node) -> float:
        return -float(self.task.best_descendant_objective(node))
