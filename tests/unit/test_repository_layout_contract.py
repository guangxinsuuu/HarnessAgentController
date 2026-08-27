from __future__ import annotations

import importlib
import unittest


class RepositoryLayoutContractTest(unittest.TestCase):
    def test_specified_modules_are_importable(self) -> None:
        modules = [
            "cgr_exp.verification.contract",
            "cgr_exp.verification.conservative",
            "cgr_exp.verification.corruption",
            "cgr_exp.search.frontier",
            "cgr_exp.search.oracle_policy",
            "cgr_exp.search.learned_policy",
        ]
        for module in modules:
            with self.subTest(module=module):
                importlib.import_module(module)

    def test_specified_policy_and_verification_symbols_exist(self) -> None:
        contract = importlib.import_module("cgr_exp.verification.contract")
        conservative = importlib.import_module("cgr_exp.verification.conservative")
        corruption = importlib.import_module("cgr_exp.verification.corruption")
        frontier = importlib.import_module("cgr_exp.search.frontier")
        oracle_policy = importlib.import_module("cgr_exp.search.oracle_policy")
        learned_policy = importlib.import_module("cgr_exp.search.learned_policy")
        self.assertTrue(hasattr(contract, "VerifierProtocol"))
        self.assertTrue(hasattr(conservative, "conservative_should_prune"))
        self.assertTrue(hasattr(corruption, "VerificationCorruptionConfig"))
        self.assertTrue(hasattr(frontier, "Frontier"))
        self.assertTrue(hasattr(oracle_policy, "OraclePolicy"))
        self.assertTrue(hasattr(learned_policy, "LearnedRankerPolicy"))


if __name__ == "__main__":
    unittest.main()
