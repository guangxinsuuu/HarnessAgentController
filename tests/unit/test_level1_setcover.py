from __future__ import annotations

import random
import unittest

from cgr_exp.datasets.level1_setcover import generate_setcover_fixture
from cgr_exp.proposers.replay import ReplayProposer, build_replay_from_task
from cgr_exp.search.best_bound import BestBoundPolicy
from cgr_exp.search.controller import run_frontier_search
from cgr_exp.verification.level1_setcover_verifier import Level1SetCoverVerifier


class Level1SetCoverTest(unittest.TestCase):
    def test_setcover_fixture_has_exact_oracle_and_sound_root_bound(self) -> None:
        task = generate_setcover_fixture(random.Random(1), "setcover-1", universe_size=5, set_count=6)
        optimum = task.exhaustive_oracle()
        root = task.root_node()
        self.assertGreaterEqual(optimum, 0.0)
        self.assertLessEqual(root.lower_bound, optimum)
        self.assertTrue(root.missingness_mask["semantic_features"])

    def test_setcover_branch_children_preserve_typed_actions(self) -> None:
        task = generate_setcover_fixture(random.Random(2), "setcover-2", universe_size=5, set_count=6)
        root = task.root_node()
        actions = task.candidate_actions(root, 3)
        self.assertTrue(actions)
        child = task.apply(root, actions[0])
        self.assertEqual(child.parent_id, root.node_id)
        self.assertEqual(child.proposal_action.action_type, "choose_set")
        self.assertIn(actions[0].set_id, child.structured_state["chosen_sets"])

    def test_best_descendant_from_partial_cover_matches_oracle_completion(self) -> None:
        task = generate_setcover_fixture(random.Random(3), "setcover-3", universe_size=5, set_count=6)
        root = task.root_node()
        partial = task.apply(root, task.candidate_actions(root, 2)[0])
        self.assertLessEqual(task.best_descendant_objective(partial), task.exhaustive_oracle() + sum(task.costs.values()))
        self.assertGreaterEqual(task.best_descendant_objective(partial), partial.lower_bound)

    def test_setcover_runs_through_generic_search_and_replay(self) -> None:
        task = generate_setcover_fixture(random.Random(4), "setcover-4", universe_size=5, set_count=6)
        replay = build_replay_from_task(task, k=4)
        result = run_frontier_search(
            task=task,
            policy=BestBoundPolicy(),
            verifier=Level1SetCoverVerifier(),
            node_budget=1000,
            proposal_budget=4,
            seed=4,
            proposer=ReplayProposer(replay),
        )
        self.assertEqual(result.best_objective, task.exhaustive_oracle())
        self.assertEqual(result.false_prune_count, 0)
        self.assertIn(task.root_node().node_id, replay)


if __name__ == "__main__":
    unittest.main()
