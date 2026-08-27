from __future__ import annotations

import unittest

from cgr_exp.certification.prune import PruneDecision, should_prune
from cgr_exp.schemas import CompletionStatus, CostLedger, FeasibleStatus, SearchNode


def node(lower_bound, uncertain=None) -> SearchNode:
    return SearchNode(
        node_id="n",
        parent_id=None,
        depth=0,
        task_id="t",
        semantic_state={},
        structured_state={},
        verified_constraints=[],
        uncertain_constraints=uncertain or [],
        lower_bound=lower_bound,
        incumbent_at_creation=10.0,
        objective_if_complete=None,
        feasible_status=FeasibleStatus.UNKNOWN,
        completion_status=CompletionStatus.PARTIAL,
        proposal_action=None,
        observation=None,
        universal_features={},
        verifier_features={"lower_bound": lower_bound},
        semantic_features=None,
        missingness_mask={"lower_bound": lower_bound is None},
        accumulated_expensive_cost=CostLedger(),
        metadata={},
    )


class CertificationTest(unittest.TestCase):
    def test_missing_bound_does_not_prune_as_zero(self) -> None:
        decision = should_prune(node(None), incumbent=0.0)
        self.assertEqual(decision, PruneDecision(False, "no-certificate"))

    def test_uncertain_constraints_do_not_affect_pruning(self) -> None:
        base = should_prune(node(5.0), incumbent=4.0)
        changed_uncertain = should_prune(node(5.0, uncertain=["noisy-hard-looking-constraint"]), incumbent=4.0)
        self.assertEqual(base, changed_uncertain)

    def test_learned_score_argument_has_no_pruning_authority(self) -> None:
        decision = should_prune(node(3.0), incumbent=4.0, learned_score=-10**9, memory_score=10**9)
        self.assertFalse(decision.prune)

    def test_bound_and_infeasibility_prune_with_reasons(self) -> None:
        self.assertEqual(should_prune(node(5.0), incumbent=4.0).reason, "lower-bound")
        infeasible = node(None)
        infeasible = infeasible.with_updates(feasible_status=FeasibleStatus.INFEASIBLE)
        self.assertEqual(should_prune(infeasible, incumbent=4.0).reason, "verified-infeasible")


if __name__ == "__main__":
    unittest.main()

