from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from cgr_exp.q2_method_contracts import Q2_METHOD_CONTRACTS, q2_method_contract
from cgr_exp.q2_implementation_matrix import (
    Q2_LOCAL_IMPLEMENTATION_PATHS,
    Q2_REQUIRED_ARTIFACTS,
    build_q2_implementation_matrix,
)
from cgr_exp.q2_source_evidence import audit_q2_source_evidence, q2_source_evidence, q2_source_evidence_methods
from cgr_exp.table5_q2 import TABLE5_METHODS


class Q2MethodContractsTest(unittest.TestCase):
    def test_contracts_cover_every_table5_row(self) -> None:
        missing = [method for method, _setting, _source, _kind in TABLE5_METHODS if method not in Q2_METHOD_CONTRACTS]
        self.assertEqual(missing, [])

    def test_contracts_include_scope_and_non_claims(self) -> None:
        for method, _setting, _source, _kind in TABLE5_METHODS:
            contract = q2_method_contract(method)
            self.assertTrue(contract["reference"], method)
            self.assertTrue(contract["claim_scope"], method)
            self.assertIn("official_reproduction", contract, method)
            self.assertTrue(contract["required_table_wording"], method)
            self.assertTrue(contract["implemented_scope"], method)
            self.assertTrue(contract["source_primitives"], method)
            self.assertTrue(contract["non_claims"], method)

    def test_rest_mcts_contract_is_explicitly_style_only(self) -> None:
        contract = q2_method_contract("PRM-guided MCTS (ReST-MCTS-style)")
        self.assertEqual(contract["claim_scope"], "style_component_adaptation")
        self.assertFalse(contract["official_reproduction"])
        self.assertIn("style", contract["required_table_wording"])
        self.assertIn("backup along the selected path", contract["implemented_scope"])
        self.assertIn("MCTS backpropagation", contract["source_primitives"])
        self.assertIn("not full ReST-MCTS iterative self-training", contract["non_claims"])

    def test_best_of_n_contract_discloses_verifier_selection(self) -> None:
        contract = q2_method_contract("Best-of-N")
        self.assertEqual(contract["required_table_wording"], "Best-of-N + verifier")
        self.assertIn("lowest verified objective selects the final sample", contract["implemented_scope"])
        self.assertIn("best sample selection", contract["source_primitives"])
        self.assertIn("strong verifier-selected variant; table notes must disclose this", contract["non_claims"])

    def test_tot_contract_tracks_official_bfs_entrypoints(self) -> None:
        contract = q2_method_contract("Tree of Thoughts")
        self.assertIn("method_generate", contract["source_primitives"])
        self.assertIn("method_evaluate", contract["source_primitives"])
        self.assertIn("method_select", contract["source_primitives"])
        self.assertIn("breadth-first search", contract["source_primitives"])

    def test_external_baselines_are_not_marked_as_official_full_reproductions(self) -> None:
        for method, _setting, _source, source_kind in TABLE5_METHODS:
            contract = q2_method_contract(method)
            if source_kind in {"structured", "native", "scheduling"} and contract["claim_scope"] != "ours":
                self.assertFalse(contract["official_reproduction"], method)

    def test_paper_or_github_baselines_have_source_urls(self) -> None:
        source_required = {
            "CP-SAT",
            "Direct / CoT",
            "Self-Consistency",
            "Best-of-N",
            "Tree of Thoughts",
            "Target-PRM priority",
            "PRM-guided MCTS (ReST-MCTS-style)",
        }
        for method in source_required:
            self.assertTrue(q2_method_contract(method)["source_url"], method)

    def test_solver_rows_are_marked_as_ours(self) -> None:
        for method in ("Solver-trained priority", "Solver-trained priority + certified pruning"):
            contract = q2_method_contract(method)
            self.assertEqual(contract["claim_scope"], "ours")
            self.assertTrue(contract["official_reproduction"])

    def test_source_alignment_matrix_mentions_core_external_sources(self) -> None:
        text = Path("docs/q2_source_alignment_matrix.md").read_text(encoding="utf-8")
        for needle in (
            "https://github.com/google/or-tools",
            "https://github.com/princeton-nlp/tree-of-thought-llm",
            "https://github.com/openai/prm800k",
            "https://github.com/THUDM/ReST-MCTS",
            "https://arxiv.org/abs/2408.03314",
            "method_generate",
            "method_evaluate",
            "method_select",
            "step-level correctness labels",
            "selection, expansion, simulation, and backpropagation",
        ):
            self.assertIn(needle, text)

    def test_source_evidence_covers_external_baselines(self) -> None:
        required = {
            "CP-SAT",
            "Direct / CoT",
            "Self-Consistency",
            "Best-of-N",
            "Tree of Thoughts",
            "Target-PRM priority",
            "PRM-guided MCTS (ReST-MCTS-style)",
        }
        self.assertEqual(required - set(q2_source_evidence_methods()), set())

    def test_source_evidence_matches_contract_urls_and_primitives(self) -> None:
        for method in q2_source_evidence_methods():
            evidence = q2_source_evidence(method)
            contract = q2_method_contract(method)
            self.assertEqual(evidence["source_url"], contract["source_url"], method)
            self.assertTrue(evidence["source_files"], method)
            self.assertTrue(evidence["official_source_anchors"], method)
            self.assertTrue(evidence["official_primitives"], method)
            self.assertTrue(evidence["local_promotion_gates"], method)
            declared = {str(item) for item in evidence["official_primitives"]}
            required = {str(item) for item in contract["source_primitives"]}
            self.assertTrue(required & declared, method)

    def test_source_alignment_matrix_points_to_machine_readable_evidence(self) -> None:
        text = Path("docs/q2_source_alignment_matrix.md").read_text(encoding="utf-8")
        self.assertIn("cgr_exp/q2_source_evidence.py", text)

    def test_source_evidence_audit_writes_ready_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            manifest = audit_q2_source_evidence(out)
            self.assertTrue(manifest["all_external_sources_ready"])
            self.assertTrue((out / "q2_source_evidence_audit.csv").exists())
            saved = json.loads((out / "q2_source_evidence_audit.json").read_text(encoding="utf-8"))
            self.assertTrue(saved["all_external_sources_ready"])
            by_method = {row["Method"]: row for row in saved["rows"]}
            self.assertEqual(by_method["Tree of Thoughts"]["source_url"], "https://github.com/princeton-nlp/tree-of-thought-llm")
            self.assertIn("src/tot/methods/bfs.py", by_method["Tree of Thoughts"]["source_files"])
            self.assertIn("method_generate", by_method["Tree of Thoughts"]["official_source_anchors"])
            self.assertIn("process-supervision", by_method["Target-PRM priority"]["official_source_anchors"])

    def test_implementation_matrix_covers_table_rows_with_local_code_and_artifacts(self) -> None:
        table_methods = {method for method, _setting, _source, _kind in TABLE5_METHODS}
        self.assertEqual(table_methods - set(Q2_LOCAL_IMPLEMENTATION_PATHS), set())
        self.assertEqual(table_methods - set(Q2_REQUIRED_ARTIFACTS), set())
        for method in table_methods:
            for path in Q2_LOCAL_IMPLEMENTATION_PATHS[method]:
                self.assertTrue(Path(path).exists(), f"{method}: {path}")
            self.assertTrue(Q2_REQUIRED_ARTIFACTS[method], method)

    def test_implementation_matrix_writes_traceable_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            manifest = build_q2_implementation_matrix(out)
            self.assertTrue(manifest["all_table_rows_covered"])
            self.assertTrue((out / "q2_implementation_matrix.csv").exists())
            saved = json.loads((out / "q2_implementation_matrix.json").read_text(encoding="utf-8"))
            by_method = {row["Method"]: row for row in saved["rows"]}
            self.assertIn("cgr_exp/scheduling_tot_q2.py", by_method["Tree of Thoughts"]["Local code paths"])
            self.assertIn("src/tot/methods/bfs.py", by_method["Tree of Thoughts"]["Official source files"])
            self.assertIn("generation, evaluation", by_method["Tree of Thoughts"]["Official source anchors"])
            self.assertIn("method_generate", by_method["Tree of Thoughts"]["Official primitives"])
            self.assertIn("target_prm_ranker.json", by_method["Target-PRM priority"]["Required result artifacts"])
            self.assertIn("not full ReST-MCTS iterative self-training", by_method["PRM-guided MCTS (ReST-MCTS-style)"]["Non-claims"])


if __name__ == "__main__":
    unittest.main()
