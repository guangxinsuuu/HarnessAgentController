from __future__ import annotations

import random
import unittest

from cgr_exp.budgets import BudgetState, BudgetExceeded
from cgr_exp.datasets.level1_assignment import generate_assignment_fixture
from cgr_exp.proposers.replay import ReplayProposer, build_replay_from_task
from cgr_exp.search.best_bound import BestBoundPolicy
from cgr_exp.search.controller import run_frontier_search
from cgr_exp.search.dfs import DFSPolicy
from cgr_exp.search.random_policy import RandomPolicy
from cgr_exp.verification.level1_verifier import Level1AssignmentVerifier


class BudgetAndReplayTest(unittest.TestCase):
    def test_budget_state_charges_calls_separately(self) -> None:
        budget = BudgetState(proposal_call_budget=2, evaluation_call_budget=1, generated_token_budget=10)
        budget.charge_proposal(input_tokens=3, output_tokens=4)
        budget.charge_evaluation(input_tokens=2, output_tokens=1)
        self.assertEqual(budget.cost.proposal_calls, 1)
        self.assertEqual(budget.cost.evaluation_calls, 1)
        self.assertEqual(budget.cost.input_tokens, 5)
        self.assertEqual(budget.cost.output_tokens, 5)
        with self.assertRaises(BudgetExceeded):
            budget.charge_evaluation()

    def test_controller_respects_proposal_call_budget(self) -> None:
        task = generate_assignment_fixture(random.Random(31), "budget", 5, 3)
        result = run_frontier_search(
            task=task,
            policy=BestBoundPolicy(),
            verifier=Level1AssignmentVerifier(seed=2),
            node_budget=1000,
            proposal_budget=3,
            seed=2,
            proposal_call_budget=2,
        )
        self.assertLessEqual(result.cost.proposal_calls, 2)
        self.assertTrue(result.budget_exhausted)

    def test_replay_proposer_gives_same_candidates_across_policies(self) -> None:
        task = generate_assignment_fixture(random.Random(32), "replay", 4, 3)
        replay = build_replay_from_task(task, k=3)
        root_id = task.root_node().node_id
        first = [a.to_dict() for a in ReplayProposer(replay).propose(task, task.root_node(), 3)]
        second = [a.to_dict() for a in ReplayProposer(replay).propose(task, task.root_node(), 3)]
        self.assertEqual(first, second)
        self.assertIn(root_id, replay)

        objectives = []
        for policy in (DFSPolicy(), RandomPolicy(seed=99), BestBoundPolicy()):
            result = run_frontier_search(
                task=task,
                policy=policy,
                verifier=Level1AssignmentVerifier(seed=3),
                node_budget=1000,
                proposal_budget=3,
                seed=3,
                proposer=ReplayProposer(replay),
            )
            objectives.append(result.best_objective)
        self.assertEqual(objectives, [task.exhaustive_oracle()] * 3)


if __name__ == "__main__":
    unittest.main()

