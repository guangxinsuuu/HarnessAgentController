from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from cgr_exp.harness_contract import audit_canonical_contract_manifest, canonical_contract_manifest, canonical_feature_names
from cgr_exp.harness_ranker import run_workflow_source_validation
from cgr_exp.tau3_contract_adapter import tau3_adapter_manifest, tau3_candidate_to_canonical
from cgr_exp.workflow_oracle import collect_workflow_oracle_records, make_workflow_ranking_pairs
from cgr_exp.workflow_source_generator import generate_workflow_source_tasks


class HarnessContractTest(unittest.TestCase):
    def test_contract_manifest_separates_runtime_features_from_labels(self) -> None:
        manifest = canonical_contract_manifest()
        audit = audit_canonical_contract_manifest(manifest)
        self.assertTrue(audit["ready"], audit["blockers"])
        self.assertFalse(manifest["oracle_labels_in_inference_features"])
        self.assertFalse(manifest["learned_score_authorizes_pruning"])
        self.assertFalse(manifest["tool_names_used_as_features"])
        self.assertFalse(set(manifest["runtime_features"]) & set(manifest["training_only_labels"]))
        self.assertGreater(len(canonical_feature_names()), len(manifest["runtime_features"]))

    def test_workflow_source_generator_uses_anonymous_tools_and_oracle_pairs(self) -> None:
        tasks = generate_workflow_source_tasks(6, seed=3, anonymize_tools=True)
        self.assertGreaterEqual(len({task.family for task in tasks}), 5)
        self.assertTrue(all(tool.name.startswith("source_tool_") for task in tasks for tool in task.tools))
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
            self.assertTrue((out / "generic_harness_ranker.json").exists())
            saved = json.loads((out / "canonical_harness_source_validation.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["benchmark"], manifest["benchmark"])
            self.assertFalse(saved["tau3_train_trajectories_used"])
            self.assertFalse(saved["oracle_labels_in_inference_features"])
            self.assertFalse(saved["tool_names_used_as_features"])
            self.assertTrue(saved["controller_checkpoint_frozen"])
            self.assertGreater(saved["train_pair_count"], 0)

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

        history = [Message(content="Need order ORD123"), Message(tool_calls=[Call("get_order", {"order_id": "ORD123"})])]
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
        self.assertFalse(manifest["oracle_visible_to_controller"])
        self.assertFalse(manifest["oracle_used_for_candidate_generation"])
        self.assertEqual(manifest["tool_identity_features"], "none; names are used only for read/write class normalization")


if __name__ == "__main__":
    unittest.main()

