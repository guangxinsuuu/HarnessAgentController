from __future__ import annotations

import unittest
from pathlib import Path

from cgr_exp.q2_method_contracts import Q2_METHOD_CONTRACTS, q2_method_contract
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
            "method_generate",
            "method_evaluate",
            "method_select",
            "step-level correctness labels",
            "selection, expansion, simulation, and backpropagation",
        ):
            self.assertIn(needle, text)


if __name__ == "__main__":
    unittest.main()
