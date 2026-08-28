from __future__ import annotations

import subprocess
import unittest
from pathlib import Path


class Q2RunScriptTest(unittest.TestCase):
    def test_q2_table4_script_is_syntax_valid_and_key_safe(self) -> None:
        script = Path("scripts/run_q2_table4.sh")
        text = script.read_text(encoding="utf-8")
        self.assertNotIn("sk-", text)
        self.assertIn("missing CGR_LLM_API_KEY", text)
        self.assertIn("run-scheduling-native-q2", text)
        self.assertIn("run-scheduling-llm-pilot", text)
        self.assertIn("run-scheduling-tot-q2", text)
        self.assertIn("build-table4-q2", text)
        self.assertIn("audit-table4-q2", text)
        self.assertIn("run_native &", text)
        self.assertIn("run_learned &", text)
        self.assertIn("run_tot &", text)
        result = subprocess.run(["zsh", "-n", str(script)], check=False, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
