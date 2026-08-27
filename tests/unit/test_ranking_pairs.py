from __future__ import annotations

import unittest

from cgr_exp.features import FEATURE_NAMES
from cgr_exp.problems import Problem, State, generate_suite
from cgr_exp.proposers import SolverProposer
from cgr_exp.trace_data import collect_states
from cgr_exp.training.pairs import FORBIDDEN_INFERENCE_FEATURES, make_labeled_ranking_pairs
from cgr_exp.verifier import Verifier


class RankingPairTest(unittest.TestCase):
    def test_future_oracle_labels_are_not_in_inference_features(self) -> None:
        forbidden = set(FORBIDDEN_INFERENCE_FEATURES)
        self.assertFalse(forbidden.intersection(FEATURE_NAMES))

    def test_labeled_pairs_keep_future_values_only_in_metadata(self) -> None:
        problem = Problem(pid="train-0000", durations=(10, 6, 6), machine_count=2, allowed=((0, 1), (0,), (0, 1)), family="balanced")
        root = State.root(problem, proposer_kind="solver")
        samples = [
            (problem, root),
            (problem, root.apply(problem, (0, 0), verified=True)),
            (problem, root.apply(problem, (0, 1), verified=True)),
        ]
        records = make_labeled_ranking_pairs(samples, max_pairs_per_problem=10, seed=14, source_split="train")
        self.assertTrue(records)
        first = records[0]
        self.assertEqual(len(first.positive_features), len(FEATURE_NAMES))
        self.assertEqual(len(first.negative_features), len(FEATURE_NAMES))
        self.assertIn("positive_best_descendant_objective", first.label_metadata)
        self.assertIn("negative_best_descendant_objective", first.label_metadata)
        self.assertNotIn("positive_best_descendant_objective", first.feature_names)
        self.assertTrue(all(value.startswith("train-") for value in first.trace_ids))

    def test_labeled_pairs_are_deterministic_for_same_seed(self) -> None:
        problems = generate_suite(15, 2, ("sparse",), (4, 4), (2, 2), "train")
        samples = collect_states(problems, SolverProposer(), lambda seed: Verifier(seed=seed), 20, 3, 16, "solver")
        a = make_labeled_ranking_pairs(samples, 10, seed=17, source_split="train")
        b = make_labeled_ranking_pairs(samples, 10, seed=17, source_split="train")
        self.assertEqual([row.to_dict() for row in a], [row.to_dict() for row in b])


if __name__ == "__main__":
    unittest.main()
