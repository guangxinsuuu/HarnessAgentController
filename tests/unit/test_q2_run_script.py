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
        self.assertIn("require_cpsat", text)
        self.assertIn("ortools.sat.python", text)
        self.assertIn("missing OR-Tools CP-SAT", text)
        self.assertIn("run-scheduling-native-q2", text)
        self.assertIn("run-scheduling-llm-pilot", text)
        self.assertIn("run-scheduling-tot-q2", text)
        self.assertIn("build-table4-q2", text)
        self.assertIn("audit-table4-q2", text)
        self.assertIn("audit-q2-source-evidence", text)
        self.assertIn("verify-q2-source-refs", text)
        self.assertIn("build-q2-implementation-matrix", text)
        self.assertIn("SOURCE_AUDIT_DIR", text)
        self.assertIn("SOURCE_REF_DIR", text)
        self.assertIn("IMPLEMENTATION_MATRIX_DIR", text)
        self.assertIn("q2_structured_100_seed0_runtime_v2", text)
        self.assertIn("q2_native_llm_baselines_100_seed0_sc_v2", text)
        self.assertIn("q2_learned_search_100_seed0_rest_gate_v2", text)
        self.assertIn("q2_tot_bfs_100_seed0_tot_gate_v2", text)
        self.assertIn("q2_table_current_seed0_protocol_v2", text)
        self.assertIn("source-audit", text)
        self.assertIn("source-ref-verify", text)
        self.assertIn("implementation-matrix", text)
        self.assertIn("source_audit", text)
        self.assertIn("source_ref_verify", text)
        self.assertIn("implementation_matrix", text)
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
                "SOURCE_REF_DIR": str(root / "source_refs"),
                "IMPLEMENTATION_MATRIX_DIR": str(root / "matrix"),
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
            self.assertIn("ortools_cp_sat:", result.stdout)
            self.assertIn("source_audit_dir:", result.stdout)
            self.assertIn("source_ref_dir:", result.stdout)
            self.assertIn("implementation_matrix_dir:", result.stdout)
            self.assertTrue((root / "audit" / "table5_q2_readiness.csv").exists())
            self.assertTrue((root / "source_audit" / "q2_source_evidence_audit.csv").exists())
            self.assertTrue((root / "source_refs" / "q2_source_ref_verification.csv").exists())
            self.assertTrue((root / "matrix" / "q2_implementation_matrix.csv").exists())

    def test_q2_runbook_uses_current_protocol_directories(self) -> None:
        text = Path("docs/q2_runbook.md").read_text(encoding="utf-8")
        self.assertIn("PYTHON_BIN=.venv/bin/python", text)
        self.assertIn("ortools.sat.python.cp_model", text)
        self.assertIn("q2_structured_100_seed0_runtime_v2", text)
        self.assertIn("q2_native_llm_baselines_100_seed0_sc_v2", text)
        self.assertIn("q2_learned_search_100_seed0_rest_gate_v2", text)
        self.assertIn("q2_tot_bfs_100_seed0_tot_gate_v2", text)
        self.assertIn("q2_table_current_seed0_protocol_v2_audit", text)
        self.assertIn("q2_source_ref_verification", text)
        self.assertNotIn("protocol_v2_protocol_v2", text)


if __name__ == "__main__":
    unittest.main()
