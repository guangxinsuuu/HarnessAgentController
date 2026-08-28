from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from cgr_exp.certification_table_readiness import audit_certification_tables
from cgr_exp.scheduling_certification import run_scheduling_certification_experiment
from cgr_exp.table6_authority import run_table6_authority_experiment


class CertificationTableReadinessTest(unittest.TestCase):
    def test_audit_accepts_complete_clean_and_robustness_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cert = root / "cert"
            auth = root / "auth"
            run_scheduling_certification_experiment(
                cert,
                seed=5,
                train_count=8,
                validation_count=3,
                test_count=3,
                job_count=5,
                machine_count=2,
                node_budget=45,
                action_budget=5,
                records_per_instance=8,
                rank_epochs=3,
                perturbation="canonical",
            )
            run_table6_authority_experiment(
                auth,
                seed=5,
                train_count=8,
                validation_count=3,
                test_count=3,
                job_count=5,
                machine_count=2,
                node_budget=45,
                action_budget=5,
                records_per_instance=8,
                rank_epochs=3,
                corruption_seeds=(0,),
            )
            manifest = audit_certification_tables(root / "audit", certification_dir=cert, authority_dir=auth, expected_test_count=3)
            self.assertTrue(manifest["all_rows_ready"])

    def test_audit_keeps_missing_artifacts_pending(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest = audit_certification_tables(Path(tmp) / "audit", expected_test_count=3)
            self.assertFalse(manifest["all_rows_ready"])
            blockers = {row["Table"]: row["result_blockers"] for row in manifest["rows"]}
            self.assertIn("artifact missing", blockers["Table 6 clean certification"])
            self.assertIn("artifact missing", blockers["Table 7 grounding robustness"])

    def test_audit_rejects_partial_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cert = root / "cert"
            run_scheduling_certification_experiment(
                cert,
                seed=5,
                train_count=8,
                validation_count=3,
                test_count=2,
                job_count=5,
                machine_count=2,
                node_budget=45,
                action_budget=5,
                records_per_instance=8,
                rank_epochs=3,
                perturbation="canonical",
            )
            manifest = audit_certification_tables(root / "audit", certification_dir=cert, expected_test_count=3)
            row = next(row for row in manifest["rows"] if row["Table"] == "Table 6 clean certification")
            self.assertFalse(row["result_ready"])
            self.assertIn("partial task coverage: 2/3", row["result_blockers"])


if __name__ == "__main__":
    unittest.main()
