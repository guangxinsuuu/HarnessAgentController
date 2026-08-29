from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from cgr_exp.certification_authority_audit import audit_certification_authority_source
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
            source_audit = root / "authority_source"
            audit_certification_authority_source(source_audit)
            manifest = audit_certification_tables(
                root / "audit",
                certification_dir=cert,
                authority_dir=auth,
                expected_test_count=3,
                authority_source_audit=source_audit / "certification_authority_source_audit.csv",
            )
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

    def test_audit_rejects_authority_rows_without_required_runtime_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            auth = root / "auth"
            auth.mkdir()
            (auth / "table6_authority_metrics.json").write_text(
                json.dumps(
                    {
                        "benchmark": "table6_authority_separation",
                        "certification_protocol": {
                            "pruning_authority": "verified lower-bound/feasibility logic only",
                            "learned_score_authority": "ranking only",
                            "learned_score_can_prune": False,
                            "uncertain_constraints_can_prune": False,
                            "oracle_labels_in_inference_features": False,
                        },
                        "table7_robustness_conditions": [
                            "Clean certified",
                            "Verified FN 1%",
                            "Verified FN 20%",
                            "Verified FP 0.5%",
                            "Verified FP 5%",
                            "Uncertain-only 20%",
                            "Learned-score pruning clean",
                        ],
                    }
                ),
                encoding="utf-8",
            )
            _write_csv(
                auth / "table7_grounding_robustness.csv",
                [
                    {
                        "Condition": "Clean certified",
                        "Semantic false prune": "0",
                        "Verified FP count": "0",
                        "Verified infeasible rejections": "0",
                        "Learned-score prunes": "0",
                    },
                    {
                        "Condition": "Uncertain-only 20%",
                        "Semantic false prune": "0",
                        "Verified FP count": "1",
                        "Verified infeasible rejections": "1",
                        "Learned-score prunes": "0",
                    },
                    {
                        "Condition": "Learned-score pruning clean",
                        "Semantic false prune": "0",
                        "Verified FP count": "0",
                        "Verified infeasible rejections": "0",
                        "Learned-score prunes": "0",
                    },
                ],
            )
            _write_csv(auth / "table6_authority_per_instance.csv", [{"task_id": "t0"}])
            manifest = audit_certification_tables(root / "audit", authority_dir=auth, expected_test_count=1)
            row = next(row for row in manifest["rows"] if row["Table"] == "Table 7 grounding robustness")
            self.assertFalse(row["result_ready"])
            self.assertIn("Uncertain-only 20% row must not convert uncertain constraints", row["result_blockers"])
            self.assertIn("Uncertain-only 20% row must not reject actions", row["result_blockers"])
            self.assertIn("Learned-score pruning clean row must trigger positive learned-score prunes", row["result_blockers"])

    def test_audit_rejects_failed_authority_source_audit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            failed = root / "failed_authority_source.csv"
            _write_csv(
                failed,
                [
                    {
                        "Check": "learned_score_has_no_pruning_authority",
                        "authority_source_ready": "False",
                        "blockers": "forbidden authority anchors present: if learned_score",
                    }
                ],
            )
            manifest = audit_certification_tables(
                root / "audit",
                expected_test_count=1,
                authority_source_audit=failed,
            )
            for row in manifest["rows"]:
                self.assertFalse(row["result_ready"])
                self.assertIn("certification authority source audit failed", row["result_blockers"])


def _write_csv(path: Path, rows) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    unittest.main()
