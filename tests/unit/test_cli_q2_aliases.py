from __future__ import annotations

import subprocess
import sys
import unittest

from cgr_exp.cli import COMMANDS


class CliQ2AliasesTest(unittest.TestCase):
    def test_table4_q2_aliases_are_registered(self) -> None:
        self.assertIn("build-table4-q2", COMMANDS)
        self.assertIn("audit-table4-q2", COMMANDS)
        self.assertIn("build-table5-q2", COMMANDS)
        self.assertIn("audit-table5-q2", COMMANDS)

    def test_table4_and_legacy_table5_aliases_expose_same_q2_arguments(self) -> None:
        expected_flags = (
            "--solver-prior-dir",
            "--expected-test-count",
            "--tot-dir",
            "--target-prm-dir",
            "--direct-dir",
            "--self-consistency-dir",
            "--best-of-n-dir",
            "--rest-mcts-dir",
            "--solver-certification-dir",
            "--structured-dir",
        )
        for command in ("build-table4-q2", "build-table5-q2", "audit-table4-q2", "audit-table5-q2"):
            result = subprocess.run(
                [sys.executable, "-m", "cgr_exp.cli", command, "--help"],
                check=False,
                text=True,
                capture_output=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            for flag in expected_flags:
                self.assertIn(flag, result.stdout, command)


if __name__ == "__main__":
    unittest.main()
