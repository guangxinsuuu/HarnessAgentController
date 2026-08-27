from __future__ import annotations

import random
import unittest

from cgr_exp.datasets.level1_assignment import AssignmentTask
from cgr_exp.datasets.level1_scheduling import ScheduleAction, SchedulingTask, generate_scheduling_fixture
from cgr_exp.proposers.replay import ReplayProposer, build_replay_from_task
from cgr_exp.schemas import SearchNode
from cgr_exp.search.best_bound import BestBoundPolicy
from cgr_exp.search.controller import run_frontier_search
from cgr_exp.verification.level1_scheduling_verifier import Level1SchedulingVerifier


class Level1SchedulingTest(unittest.TestCase):
    def test_fixture_is_distinct_scheduling_family_with_sound_bound(self) -> None:
        task = generate_scheduling_fixture(random.Random(3), "sched-1", job_count=5, machine_count=2)
        root = task.root_node()
        optimum = task.exhaustive_oracle()
        self.assertIsInstance(task, SchedulingTask)
        self.assertNotIsInstance(task, AssignmentTask)
        self.assertEqual(root.metadata["family"], "scheduling")
        self.assertLessEqual(root.lower_bound, optimum)
        self.assertTrue(task.ready_jobs(root.structured_state["scheduled"]))

    def test_search_reaches_oracle_on_fixture_replay(self) -> None:
        task = generate_scheduling_fixture(random.Random(4), "sched-2", job_count=5, machine_count=2)
        result = run_frontier_search(
            task=task,
            policy=BestBoundPolicy(),
            verifier=Level1SchedulingVerifier(),
            node_budget=200,
            proposal_budget=4,
            seed=5,
            proposer=ReplayProposer(build_replay_from_task(task, 4)),
        )
        self.assertEqual(result.best_objective, task.exhaustive_oracle())
        self.assertEqual(result.false_prune_count, 0)
        self.assertGreaterEqual(result.pruned_nodes, 0)

    def test_certified_pruning_can_be_disabled_for_ablation(self) -> None:
        task = generate_scheduling_fixture(random.Random(6), "sched-prune", job_count=5, machine_count=2)
        replay = ReplayProposer(build_replay_from_task(task, 4))
        with_pruning = run_frontier_search(
            task=task,
            policy=BestBoundPolicy(),
            verifier=Level1SchedulingVerifier(),
            node_budget=200,
            proposal_budget=4,
            seed=6,
            proposer=replay,
            certified_pruning=True,
        )
        without_pruning = run_frontier_search(
            task=task,
            policy=BestBoundPolicy(),
            verifier=Level1SchedulingVerifier(),
            node_budget=200,
            proposal_budget=4,
            seed=6,
            proposer=replay,
            certified_pruning=False,
        )
        self.assertEqual(with_pruning.false_prune_count, 0)
        self.assertEqual(without_pruning.pruned_nodes, 0)
        self.assertGreaterEqual(without_pruning.expanded_nodes, with_pruning.expanded_nodes)

    def test_schedule_action_round_trips_through_search_node(self) -> None:
        action = ScheduleAction(job="J1", machine="M0")
        node = generate_scheduling_fixture(random.Random(5), "sched-3", job_count=3, machine_count=2).root_node()
        child = node.with_updates(proposal_action=action)
        restored = SearchNode.from_dict(child.to_dict())
        self.assertEqual(type(restored.proposal_action), ScheduleAction)
        self.assertEqual(restored.to_dict(), child.to_dict())


if __name__ == "__main__":
    unittest.main()
