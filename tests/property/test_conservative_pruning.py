from __future__ import annotations

import random
import unittest

from cgr_exp.datasets.level1_assignment import generate_assignment_fixture
from cgr_exp.search.controller import run_frontier_search
from cgr_exp.search.dfs import DFSPolicy
from cgr_exp.search.random_policy import RandomPolicy
from cgr_exp.search.best_bound import BestBoundPolicy
from cgr_exp.verification.level1_verifier import Level1AssignmentVerifier


class ConservativePruningPropertyTest(unittest.TestCase):
    def test_false_prune_zero_on_random_tiny_conservative_instances(self) -> None:
        for seed in range(20):
            task = generate_assignment_fixture(random.Random(seed), task_id=f"tiny-{seed}", job_count=4, resource_count=3)
            result = run_frontier_search(
                task=task,
                policy=BestBoundPolicy(),
                verifier=Level1AssignmentVerifier(false_negative_rate=0.25, false_positive_rate=0.0, seed=seed),
                node_budget=10_000,
                proposal_budget=3,
                seed=seed,
            )
            self.assertEqual(result.false_prune_count, 0)
            self.assertEqual(result.best_objective, task.exhaustive_oracle())

    def test_unlimited_ordering_invariance_under_same_support(self) -> None:
        task = generate_assignment_fixture(random.Random(123), task_id="order", job_count=4, resource_count=3)
        policies = [DFSPolicy(), RandomPolicy(seed=1), RandomPolicy(seed=2), BestBoundPolicy()]
        objectives = []
        for policy in policies:
            result = run_frontier_search(
                task=task,
                policy=policy,
                verifier=Level1AssignmentVerifier(seed=99),
                node_budget=10_000,
                proposal_budget=3,
                seed=99,
            )
            objectives.append(result.best_objective)
        self.assertEqual(objectives, [task.exhaustive_oracle()] * len(policies))


if __name__ == "__main__":
    unittest.main()

