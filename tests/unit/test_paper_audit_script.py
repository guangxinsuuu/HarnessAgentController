from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
import stat
from pathlib import Path


class PaperAuditScriptTest(unittest.TestCase):
    def test_paper_audit_script_is_syntax_valid_and_key_safe(self) -> None:
        script = Path("scripts/run_paper_audits.sh")
        text = script.read_text(encoding="utf-8")
        mode = script.stat().st_mode
        self.assertTrue(mode & stat.S_IXUSR)
        self.assertNotIn("sk-", text)
        self.assertIn("audit_q2", text)
        self.assertIn("audit_q4", text)
        self.assertIn("audit_certification", text)
        self.assertIn("verify-q2-source-refs", text)
        self.assertIn("verify-q4-tau-source-refs", text)
        self.assertIn("audit-certification-tables", text)
        self.assertIn("audit-certification-authority-source", text)
        self.assertIn("build-current-experiment-status", text)
        self.assertIn("llm_key: not required", text)
        self.assertIn("STATUS_REPORT_DIR", text)
        self.assertIn('STATUS_REPORT_DIR="${STATUS_REPORT_DIR:-runs/current_experiment_status}"', text)
        self.assertIn("Q2_REMAINING_RUNS_DIR", text)
        self.assertIn("build-q2-remaining-runs", text)
        self.assertIn("Q2_RUN_PREFLIGHT_DIR", text)
        self.assertIn("build-q2-run-preflight", text)
        self.assertIn("Q2_LOCAL_PRIMITIVE_AUDIT_DIR", text)
        self.assertIn("audit-q2-local-primitives", text)
        self.assertIn("--q2-local-primitives", text)
        self.assertIn("backfill-scheduling-llm-pilot-audit-artifacts", text)
        self.assertIn("Q4_ADAPTER_CONTRACT_DIR", text)
        self.assertIn("build-q4-tau-adapter-contract", text)
        self.assertIn("q4_adapter_contract_dir", text)
        self.assertIn("audit-secret-safety", text)
        self.assertIn("secret_audit_dir", text)
        self.assertIn("CANDIDATE_WORKERS", text)
        self.assertIn("CANDIDATE_RETRIES", text)
        self.assertIn("CANDIDATE_RETRY_BACKOFF", text)
        self.assertIn('TAU3_TABLE_BASE_DIR="${TAU3_TABLE_BASE_DIR:-runs/table10_tau3}"', text)
        self.assertNotIn("runs/table10_tau3_canonical", text)
        self.assertIn('TAU3_SOLVER_RETAIL_DIR="${TAU3_SOLVER_RETAIL_DIR:-${TAU3_TABLE_BASE_DIR}/solver_retail}"', text)
        self.assertIn("TAU3_CANONICAL_RETAIL_DIR", text)
        self.assertIn('"${TAU3_CANONICAL_RETAIL_DIR}"', text)
        self.assertIn("certification_authority_source_audit_dir", text)
        self.assertIn("--certification-authority-source", text)
        self.assertIn("prefer_existing_dir", text)
        self.assertIn("prefer_existing_artifact_dir", text)
        self.assertIn("scheduling_llm_pilot_table.csv", text)
        self.assertIn("scheduling_table5_prm_rest_cert_100_seed0", text)
        self.assertIn("scheduling_missing_gpt_frontier_rank_100_seed0", text)
        result = subprocess.run(["zsh", "-n", str(script)], check=False, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_paper_audit_script_runs_without_key_or_network(self) -> None:
        script = Path("scripts/run_paper_audits.sh")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            env = {
                **os.environ,
                "TEST_COUNT": "1",
                "Q2_STRUCTURED_DIR": str(root / "missing_q2_structured"),
                "Q2_NATIVE_DIR": str(root / "missing_q2_native"),
                "Q2_LEARNED_DIR": str(root / "missing_q2_learned"),
                "Q2_TOT_DIR": str(root / "missing_q2_tot"),
                "Q2_TABLE_AUDIT_DIR": str(root / "q2_table_audit"),
                "Q2_SOURCE_AUDIT_DIR": str(root / "q2_source_audit"),
                "Q2_SOURCE_REF_DIR": str(root / "q2_source_refs"),
                "Q2_IMPLEMENTATION_MATRIX_DIR": str(root / "q2_matrix"),
                "Q2_LOCAL_PRIMITIVE_AUDIT_DIR": str(root / "q2_local_primitives"),
                "Q2_REMAINING_RUNS_DIR": str(root / "q2_remaining"),
                "Q2_RUN_PREFLIGHT_DIR": str(root / "q2_run_preflight"),
                "Q4_TABLE_AUDIT_DIR": str(root / "q4_table_audit"),
                "Q4_SOURCE_AUDIT_DIR": str(root / "q4_source_audit"),
                "Q4_SOURCE_REF_DIR": str(root / "q4_source_refs"),
                "Q4_ADAPTER_CONTRACT_DIR": str(root / "q4_contract"),
                "Q4_NATIVE_DIR": str(root / "missing_q4_native"),
                "Q4_BEST_OF_N_DIR": str(root / "missing_q4_best"),
                "Q4_VALUE_GUIDED_DIR": str(root / "missing_q4_value"),
                "Q4_TARGET_PRM_DIR": str(root / "missing_q4_prm"),
                "Q4_SOLVER_CONTROLLER_DIR": str(root / "missing_q4_solver"),
                "CERTIFICATION_AUDIT_DIR": str(root / "cert_audit"),
                "CERTIFICATION_AUTHORITY_SOURCE_AUDIT_DIR": str(root / "cert_authority_source"),
                "CERTIFICATION_DIR": str(root / "missing_cert"),
                "AUTHORITY_DIR": str(root / "missing_authority"),
                "STATUS_REPORT_DIR": str(root / "status"),
                "SECRET_AUDIT_DIR": str(root / "secret_audit"),
                "TAU3_TABLE_BASE_DIR": str(root / "tau3_base"),
            }
            env.pop("CGR_LLM_API_KEY", None)
            result = subprocess.run(
                ["zsh", str(script), "all"],
                check=False,
                text=True,
                capture_output=True,
                env=env,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("llm_key: not required", result.stdout)
            self.assertIn("q2_local_primitive_audit_dir:", result.stdout)
            self.assertIn("q2_run_preflight_dir:", result.stdout)
            self.assertIn("q4_adapter_contract_dir:", result.stdout)
            self.assertIn("tau3_canonical_retail_dir:", result.stdout)
            self.assertIn("tau3_canonical_airline_dir:", result.stdout)
            self.assertIn("tau3_candidate_workers:", result.stdout)
            self.assertIn("tau3_candidate_retries:", result.stdout)
            self.assertIn("tau3_candidate_retry_backoff:", result.stdout)
            self.assertIn("secret_audit_dir:", result.stdout)
            self.assertTrue((root / "q2_table_audit" / "table5_q2_readiness.csv").exists())
            self.assertTrue((root / "q2_source_audit" / "q2_source_evidence_audit.csv").exists())
            self.assertTrue((root / "q2_source_refs" / "q2_source_ref_verification.csv").exists())
            self.assertTrue((root / "q2_matrix" / "q2_implementation_matrix.csv").exists())
            self.assertTrue((root / "q2_local_primitives" / "q2_local_primitive_audit.csv").exists())
            self.assertTrue((root / "q2_remaining" / "q2_remaining_runs.csv").exists())
            self.assertTrue((root / "q2_run_preflight" / "q2_run_preflight.csv").exists())
            self.assertTrue((root / "q4_table_audit" / "q4_tau_readiness.csv").exists())
            self.assertTrue((root / "q4_source_audit" / "q4_tau_source_evidence_audit.csv").exists())
            self.assertTrue((root / "q4_source_refs" / "q4_tau_source_ref_verification.csv").exists())
            self.assertTrue((root / "q4_contract" / "q4_tau_adapter_contract.csv").exists())
            self.assertTrue((root / "cert_audit" / "certification_tables_readiness.csv").exists())
            self.assertTrue((root / "cert_authority_source" / "certification_authority_source_audit.csv").exists())
            self.assertTrue((root / "secret_audit" / "secret_safety_audit.csv").exists())
            self.assertTrue((root / "status" / "current_experiment_status.md").exists())
            with (root / "tau3_base" / "table10_readiness" / "table10_tau3_readiness.csv").open(newline="", encoding="utf-8") as handle:
                self.assertEqual(sum(1 for _ in handle) - 1, 10)

    def test_current_status_doc_records_audit_outputs_and_pending_tau(self) -> None:
        text = Path("docs/current_experiment_status.md").read_text(encoding="utf-8")
        self.assertIn("scripts/run_paper_audits.sh all", text)
        self.assertIn("q2_table_current_seed0_protocol_v2_audit/table5_q2_readiness.csv", text)
        self.assertIn("q4_tau_table_current_audit/q4_tau_readiness.csv", text)
        self.assertIn("q4_tau_adapter_contract/q4_tau_adapter_contract.md", text)
        self.assertIn("docs/q2_table4_official_implementation_plan.md", text)
        self.assertIn("q2_local_primitive_audit/q2_local_primitive_audit.csv", text)
        self.assertIn("q2_run_preflight/q2_run_preflight.md", text)
        self.assertIn("secret_safety_audit/secret_safety_audit.csv", text)
        self.assertIn("certification_tables_readiness/certification_tables_readiness.csv", text)
        self.assertIn("certification_authority_source_audit/certification_authority_source_audit.csv", text)
        self.assertIn("All Q4 tau/tau2 rows remain pending", text)
        self.assertIn("Only rows marked ready", text)

    def test_tau3_runbook_documents_ten_row_paper_audit(self) -> None:
        text = Path("docs/tau3_runbook.md").read_text(encoding="utf-8")
        self.assertIn("scripts/run_paper_audits.sh tau3", text)
        self.assertIn("10 maintained-protocol Table10 rows", text)
        self.assertIn("native retail and airline", text)
        self.assertIn("Best-of-N retail and airline", text)
        self.assertIn("legacy solver-controller retail and airline", text)
        self.assertIn("canonical solver-controller retail and airline", text)
        self.assertIn("certified canonical solver-controller retail and airline", text)


if __name__ == "__main__":
    unittest.main()
