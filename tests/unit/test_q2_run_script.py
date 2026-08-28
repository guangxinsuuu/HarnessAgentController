from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path


class Q2RunScriptTest(unittest.TestCase):
    def test_q2_table4_script_is_syntax_valid_and_key_safe(self) -> None:
        script = Path("scripts/run_q2_table4.sh")
        text = script.read_text(encoding="utf-8")
        self.assertNotIn("sk-", text)
        self.assertIn("preflight", text)
        self.assertIn("missing CGR_LLM_API_KEY", text)
        self.assertIn("run-scheduling-native-q2", text)
        self.assertIn("run-scheduling-llm-pilot", text)
        self.assertIn("run-scheduling-tot-q2", text)
        self.assertIn("build-table4-q2", text)
        self.assertIn("audit-table4-q2", text)
        self.assertIn("audit-q2-source-evidence", text)
        self.assertIn("SOURCE_AUDIT_DIR", text)
        self.assertIn("source-audit", text)
        self.assertIn("source_audit", text)
        self.assertIn("run_native &", text)
        self.assertIn("run_learned &", text)
        self.assertIn("run_tot &", text)
        result = subprocess.run(["zsh", "-n", str(script)], check=False, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_q2_table4_preflight_runs_without_key_or_network(self) -> None:
        script = Path("scripts/run_q2_table4.sh")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            env = {
                **os.environ,
                "STRUCTURED_DIR": str(root / "structured"),
                "NATIVE_DIR": str(root / "native"),
                "LEARNED_DIR": str(root / "learned"),
                "TOT_DIR": str(root / "tot"),
                "TABLE_DIR": str(root / "table"),
                "AUDIT_DIR": str(root / "audit"),
                "SOURCE_AUDIT_DIR": str(root / "source_audit"),
                "TEST_COUNT": "1",
            }
            env.pop("CGR_LLM_API_KEY", None)
            result = subprocess.run(
                ["zsh", str(script), "preflight"],
                check=False,
                text=True,
                capture_output=True,
                env=env,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("llm_key: missing", result.stdout)
            self.assertIn("source_audit_dir:", result.stdout)
            self.assertTrue((root / "audit" / "table5_q2_readiness.csv").exists())
            self.assertTrue((root / "source_audit" / "q2_source_evidence_audit.csv").exists())


if __name__ == "__main__":
    unittest.main()
