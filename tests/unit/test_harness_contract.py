from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from cgr_exp.harness_contract import audit_canonical_contract_manifest, canonical_contract_manifest, canonical_feature_names
from cgr_exp.harness_ranker import GenericHarnessRanker, run_workflow_source_validation
from cgr_exp.tau3_contract_adapter import tau3_adapter_manifest, tau3_candidate_to_canonical
from cgr_exp.workflow_oracle import collect_workflow_oracle_records, make_workflow_ranking_pairs
from cgr_exp.workflow_source_generator import generate_workflow_source_tasks, source_generator_manifest


class HarnessContractTest(unittest.TestCase):
    def test_contract_manifest_separates_runtime_features_from_labels(self) -> None:
        manifest = canonical_contract_manifest()
        audit = audit_canonical_contract_manifest(manifest)
        self.assertTrue(audit["ready"], audit["blockers"])
        self.assertFalse(manifest["oracle_labels_in_inference_features"])
        self.assertFalse(manifest["learned_score_authorizes_pruning"])
        self.assertFalse(manifest["tool_names_used_as_features"])
        self.assertIn("verified_progress_known_mask", manifest["runtime_known_mask_features"])
        self.assertTrue(set(manifest["runtime_known_mask_features"]).issubset(set(manifest["runtime_features"])))
        self.assertFalse(set(manifest["runtime_features"]) & set(manifest["training_only_labels"]))
        self.assertFalse(set(canonical_feature_names()) & set(manifest["training_only_labels"]))
        self.assertFalse(set(canonical_feature_names()) & set(manifest["evaluation_only_metrics"]))
        self.assertGreater(len(canonical_feature_names()), len(manifest["runtime_features"]))

    def test_workflow_source_generator_uses_anonymous_tools_and_oracle_pairs(self) -> None:
        tasks = generate_workflow_source_tasks(6, seed=3, anonymize_tools=True)
        self.assertGreaterEqual(len({task.family for task in tasks}), 5)
        self.assertTrue(all(tool.name.startswith("source_tool_") for task in tasks for tool in task.tools))
        anonymous_manifest = source_generator_manifest(seed=3, count=6, anonymize_tools=True)
        anonymous_manifest_other_seed = source_generator_manifest(seed=99, count=20, anonymize_tools=True)
        named_manifest = source_generator_manifest(seed=3, count=6, anonymize_tools=False)
        self.assertTrue(anonymous_manifest["anonymized_tool_names"])
        self.assertFalse(named_manifest["anonymized_tool_names"])
        self.assertEqual(anonymous_manifest["distribution_hash"], anonymous_manifest_other_seed["distribution_hash"])
        self.assertNotEqual(anonymous_manifest["distribution_hash"], named_manifest["distribution_hash"])
        self.assertIn("anonymous", anonymous_manifest["parameter_distribution"]["tool_names"])
        self.assertIn("family_action", named_manifest["parameter_distribution"]["tool_names"])
        records = []
        for task in tasks:
            records.extend(collect_workflow_oracle_records(task, max_states=50))
        pairs = make_workflow_ranking_pairs(records)
        self.assertGreater(len(records), 0)
        self.assertGreater(len(pairs), 0)

    def test_workflow_source_validation_writes_frozen_zero_target_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            manifest = run_workflow_source_validation(out, train_count=10, validation_count=5, seed=5, epochs=3)
            self.assertTrue((out / "canonical_harness_source_validation.csv").exists())
            self.assertTrue((out / "canonical_harness_tool_identity_stress.csv").exists())
            self.assertTrue((out / "canonical_harness_contract_coverage_stress.csv").exists())
            self.assertTrue((out / "canonical_harness_feature_coverage.csv").exists())
            self.assertTrue((out / "generic_harness_ranker.json").exists())
            saved = json.loads((out / "canonical_harness_source_validation.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["benchmark"], manifest["benchmark"])
            self.assertFalse(saved["tau3_train_trajectories_used"])
            self.assertFalse(saved["oracle_labels_in_inference_features"])
            self.assertFalse(saved["tool_names_used_as_features"])
            self.assertTrue(saved["controller_checkpoint_frozen"])
            self.assertTrue(saved["source_generator"]["source_generator_distribution_fixed_before_target_test_evaluation"])
            self.assertFalse(saved["source_generator"]["tau3_heldout_test_seen_before_freeze"])
            self.assertTrue(saved["contract_audit"]["ready"], saved["contract_audit"]["blockers"])
            self.assertTrue(saved["tool_identity_generalization"]["ready"])
            self.assertEqual(saved["tool_identity_generalization"]["top1_absolute_delta"], 0.0)
            self.assertTrue(saved["contract_coverage_stress"]["ready"])
            self.assertIn("black_box_contract_top1_accuracy", saved["contract_coverage_stress"])
            self.assertIn("canonical_feature_coverage", saved)
            self.assertIn("train", saved["canonical_feature_coverage"])
            self.assertIn("verified_progress_known_mask", saved["canonical_feature_coverage"]["train"]["per_feature_availability"])
            self.assertGreater(saved["train_pair_count"], 0)

    def test_workflow_source_validation_can_train_action_intent_controller(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            manifest = run_workflow_source_validation(
                out,
                train_count=8,
                validation_count=4,
                seed=6,
                epochs=2,
                max_states_per_task=60,
                action_intent_training=True,
            )
            saved = json.loads((out / "canonical_harness_source_validation.json").read_text(encoding="utf-8"))
            self.assertTrue(manifest["action_intent_training"])
            self.assertEqual(saved["source_action_abstraction"], "available_action_intent")
            self.assertGreater(saved["train_pair_count"], 0)
            self.assertTrue((out / "generic_harness_ranker.json").exists())

    def test_generic_harness_ranker_rejects_bad_checkpoint_dimension(self) -> None:
        with self.assertRaisesRegex(ValueError, "weight dimension mismatch"):
            GenericHarnessRanker.from_json(json.dumps({"weights": [0.0], "bias": 0.0, "feature_names": canonical_feature_names()}))

    def test_tau3_adapter_outputs_canonical_features_without_oracle_or_tool_identity(self) -> None:
        class Tool:
            def __init__(self, name):
                self.name = name

        class Call:
            def __init__(self, name, arguments):
                self.name = name
                self.arguments = arguments

        class Message:
            def __init__(self, content=None, tool_calls=None):
                self.content = content
                self.tool_calls = tool_calls

        class ToolMessage:
            def __init__(self, content):
                self.content = content
                self.tool_calls = None

        history = [
            Message(content="Need order ORD123. yes, approved to proceed."),
            Message(tool_calls=[Call("get_order", {"order_id": "ORD123"})]),
            ToolMessage(content="Order ORD123 found. Customer identity verified."),
        ]
        candidate = Message(tool_calls=[Call("update_order", {"order_id": "ORD123"})])
        canonical = tau3_candidate_to_canonical(
            candidate_id="c0",
            message_history=history,
            tools=[Tool("get_order"), Tool("update_order")],
            candidate=candidate,
            remaining_turns=8,
            max_turns=10,
            remaining_tool_budget=7,
            max_tool_budget=10,
        )
        manifest = tau3_adapter_manifest()
        self.assertEqual(len(canonical.feature_vector()), len(canonical_feature_names()))
        self.assertEqual(canonical.action.action_is_write, 1.0)
        self.assertEqual(canonical.state.verified_progress_known_mask, 1.0)
        self.assertEqual(canonical.state.remaining_requirement_known_mask, 1.0)
        self.assertGreater(canonical.state.verified_progress_frac, 0.0)
        self.assertLess(canonical.state.remaining_requirement_frac, 1.0)
        self.assertFalse(manifest["oracle_visible_to_controller"])
        self.assertFalse(manifest["oracle_used_for_candidate_generation"])
        self.assertEqual(manifest["adapter_revision"], "Tau3AdapterV2-progress-predicates")
        self.assertIn("required_read_completed", manifest["progress_predicates"])
        self.assertEqual(manifest["tool_identity_features"], "none; names are used only for read/write class normalization")


if __name__ == "__main__":
    unittest.main()
