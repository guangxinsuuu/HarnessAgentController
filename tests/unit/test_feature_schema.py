from __future__ import annotations

import random
import unittest

from cgr_exp.datasets.level1_assignment import generate_assignment_fixture
from cgr_exp.datasets.level1_setcover import generate_setcover_fixture
from cgr_exp.datasets.level3_tool_env import make_customer_support_fixture
from cgr_exp.feature_schema import FEATURE_SCHEMA, validate_node_features
from cgr_exp.training.pairs import FORBIDDEN_INFERENCE_FEATURES


class SharedFeatureSchemaTest(unittest.TestCase):
    def test_schema_declares_required_groups_and_excludes_future_labels(self) -> None:
        self.assertIn("depth", FEATURE_SCHEMA.universal)
        self.assertIn("branch_factor", FEATURE_SCHEMA.universal)
        self.assertIn("lower_bound", FEATURE_SCHEMA.verifier)
        self.assertIn("verified_constraint_count", FEATURE_SCHEMA.verifier)
        self.assertFalse(set(FEATURE_SCHEMA.all_inference_features()).intersection(FORBIDDEN_INFERENCE_FEATURES))

    def test_level_nodes_validate_with_missing_masks(self) -> None:
        assignment = generate_assignment_fixture(random.Random(1), "feature-assignment", job_count=4, resource_count=3).root_node()
        setcover = generate_setcover_fixture(random.Random(2), "feature-setcover", universe_size=5, set_count=6).root_node()
        level3 = make_customer_support_fixture().root_node("case-1")
        for node in (assignment, setcover, level3):
            with self.subTest(node=node.node_id):
                validate_node_features(node)
        self.assertTrue(level3.missingness_mask["lower_bound"])
        self.assertIsNone(level3.verifier_features["lower_bound"])

    def test_validator_rejects_unmasked_missing_required_feature(self) -> None:
        node = make_customer_support_fixture().root_node("case-1").with_updates(missingness_mask={"semantic_features": True})
        with self.assertRaises(ValueError):
            validate_node_features(node)


if __name__ == "__main__":
    unittest.main()
