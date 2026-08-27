from __future__ import annotations

import random
import unittest

from cgr_exp.datasets.level1_assignment import generate_assignment_fixture
from cgr_exp.datasets.level2_renderer import render_level2_variants
from cgr_exp.experiments import PRESETS, _level2_proposal_diagnostics
from cgr_exp.verification.level2_verifier import Level2Verifier
from cgr_exp.proposers.simulator import DeterministicSimulatorProposer
from cgr_exp.analysis.proposal_analysis import proposal_metrics_for_typed_actions
from cgr_exp.schemas import AssignAction


class Level2RenderingTest(unittest.TestCase):
    def test_render_variants_preserve_hidden_task(self) -> None:
        task = generate_assignment_fixture(random.Random(7), "hidden-7", job_count=4, resource_count=3)
        variants = render_level2_variants(task)
        self.assertGreaterEqual(len(variants), 5)
        hidden_ids = {v.hidden_task_id for v in variants}
        self.assertEqual(hidden_ids, {task.task_id})
        names = {v.perturbation for v in variants}
        self.assertTrue({"canonical", "paraphrase", "entity_rename", "irrelevant_context", "mild_ambiguity"}.issubset(names))
        renamed = next(v for v in variants if v.perturbation == "entity_rename")
        self.assertNotEqual(renamed.entity_map["tasks"]["T0"], "T0")
        self.assertEqual(renamed.hidden_task, task)


class TypedProposalMetricsTest(unittest.TestCase):
    def test_proposal_recall_counts_only_valid_useful_typed_actions(self) -> None:
        useful = {("T1", "R2")}
        actions = [
            AssignAction(task="T1", resource="R2"),
            AssignAction(task="T1", resource="R2"),
            AssignAction(task="", resource="R2"),
            AssignAction(task="T1", resource="R9"),
        ]
        metrics = proposal_metrics_for_typed_actions(actions, useful_actions=useful, known_tasks={"T1"}, known_resources={"R2"})
        self.assertEqual(metrics["proposal_recall_at_k"], 1.0)
        self.assertEqual(metrics["valid_useful_action_count"], 2)
        self.assertEqual(metrics["invalid_action_count"], 2)
        self.assertGreater(metrics["duplicate_action_rate"], 0.0)

    def test_deterministic_simulator_reproducible_with_same_seed(self) -> None:
        task = generate_assignment_fixture(random.Random(8), "sim-8", job_count=4, resource_count=3)
        node = task.root_node()
        a = DeterministicSimulatorProposer(seed=10, useful_action_recall=0.9, invalid_action_prob=0.2, duplicate_prob=0.2)
        b = DeterministicSimulatorProposer(seed=10, useful_action_recall=0.9, invalid_action_prob=0.2, duplicate_prob=0.2)
        self.assertEqual(
            [x.to_dict() for x in a.propose(task, node, 5, budget_state=None)],
            [x.to_dict() for x in b.propose(task, node, 5, budget_state=None)],
        )


class Level2VerifierTest(unittest.TestCase):
    def test_level2_verifier_resolves_renamed_entities_to_hidden_task(self) -> None:
        task = generate_assignment_fixture(random.Random(9), "hidden-9", job_count=4, resource_count=3)
        rendering = next(v for v in render_level2_variants(task) if v.perturbation == "entity_rename")
        root = task.root_node()
        canonical_task = next(iter(task.durations))
        canonical_resource = task.allowed[canonical_task][0]
        action = AssignAction(task=rendering.entity_map["tasks"][canonical_task], resource=rendering.entity_map["resources"][canonical_resource])
        result = Level2Verifier().verify(rendering, root, action)
        self.assertTrue(result.feasible)
        self.assertIn(f"assign:{canonical_task}:{canonical_resource}", result.verified_constraints)
        self.assertEqual(result.verification_evidence["resolved_action"]["task"], canonical_task)
        self.assertEqual(result.verification_evidence["raw_action"]["task"], action.task)

    def test_level2_verifier_keeps_preference_uncertain_not_certified(self) -> None:
        task = generate_assignment_fixture(random.Random(10), "hidden-10", job_count=4, resource_count=3)
        rendering = next(v for v in render_level2_variants(task) if v.perturbation == "mild_ambiguity")
        root = task.root_node()
        canonical_task = next(iter(task.durations))
        canonical_resource = task.allowed[canonical_task][0]
        result = Level2Verifier().verify(rendering, root, AssignAction(task=canonical_task, resource=canonical_resource))
        self.assertTrue(result.feasible)
        self.assertIn("preference:load_balance", result.uncertain_constraints)
        self.assertNotIn("prefer:load_balance", result.verified_constraints)

    def test_level2_verifier_rejects_unknown_rendered_entity(self) -> None:
        task = generate_assignment_fixture(random.Random(11), "hidden-11", job_count=4, resource_count=3)
        rendering = render_level2_variants(task)[0]
        result = Level2Verifier().verify(rendering, task.root_node(), AssignAction(task="missing-task", resource="R0"))
        self.assertFalse(result.feasible)
        self.assertIn("unknown-rendered-entity", result.violations)

    def test_level2_proposal_diagnostics_include_verifier_outcome(self) -> None:
        rows = _level2_proposal_diagnostics(PRESETS["smoke"])
        self.assertTrue(rows)
        self.assertTrue(all("verifier_backend" in row for row in rows))
        self.assertTrue(any(row["uncertain_constraints"] > 0 for row in rows if row["perturbation"] == "mild_ambiguity"))


if __name__ == "__main__":
    unittest.main()
