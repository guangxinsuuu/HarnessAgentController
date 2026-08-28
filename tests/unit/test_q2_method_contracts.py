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
            self.assertTrue(contract["implemented_scope"], method)
            self.assertTrue(contract["non_claims"], method)

    def test_rest_mcts_contract_is_explicitly_style_only(self) -> None:
        contract = q2_method_contract("PRM-guided MCTS (ReST-MCTS-style)")
        self.assertIn("backup along the selected path", contract["implemented_scope"])
        self.assertIn("not full ReST-MCTS iterative self-training", contract["non_claims"])

    def test_best_of_n_contract_discloses_verifier_selection(self) -> None:
        contract = q2_method_contract("Best-of-N")
        self.assertIn("lowest verified objective selects the final sample", contract["implemented_scope"])
        self.assertIn("strong verifier-selected variant; table notes must disclose this", contract["non_claims"])

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
