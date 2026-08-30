from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from cgr_exp.harness_contract import canonical_contract_manifest, canonical_feature_names
from cgr_exp.cli import COMMANDS


class CliQ4TauTest(unittest.TestCase):
    def test_q4_tau_commands_are_registered(self) -> None:
        self.assertIn("build-table-q4-tau", COMMANDS)
        self.assertIn("audit-table-q4-tau", COMMANDS)
        self.assertIn("summarize-table10-tau3-progress", COMMANDS)
        self.assertIn("audit-table10-tau3-smoke", COMMANDS)
        self.assertIn("audit-q4-tau-source-evidence", COMMANDS)
        self.assertIn("verify-q4-tau-source-refs", COMMANDS)

    def test_table10_tau3_run_preflight_defaults_to_runner_base_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "preflight"
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "cgr_exp.cli",
                    "build-table10-tau3-run-preflight",
                    "--output-dir",
                    str(out),
                    "--expected-test-count",
                    "1",
                    "--overwrite",
                ],
                check=False,
                text=True,
                capture_output=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            manifest = json.loads((out / "table10_tau3_run_preflight.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["base_dir"], "runs/table10_tau3")

    def test_q4_tau_commands_expose_artifact_inputs(self) -> None:
        expected_flags = (
            "--expected-test-count",
            "--native-dir",
            "--best-of-n-dir",
            "--value-guided-dir",
            "--target-prm-dir",
            "--solver-controller-dir",
        )
        for command in ("build-table-q4-tau", "audit-table-q4-tau"):
            result = subprocess.run(
                [sys.executable, "-m", "cgr_exp.cli", command, "--help"],
                check=False,
                text=True,
                capture_output=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            for flag in expected_flags:
                self.assertIn(flag, result.stdout, command)

    def test_q4_tau_audit_command_writes_pending_manifest_without_native_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "audit"
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "cgr_exp.cli",
                    "audit-table-q4-tau",
                    "--output-dir",
                    str(out),
                    "--expected-test-count",
                    "3",
                    "--overwrite",
                ],
                check=False,
                text=True,
                capture_output=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            manifest = json.loads((out / "command_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["expected_test_count"], 3)
            self.assertEqual(manifest["all_rows_result_ready"], False)
            rows = _read_csv(out / "q4_tau_readiness.csv")
            self.assertTrue(rows)
            self.assertTrue(all(row["result_ready"] == "False" for row in rows))
            self.assertTrue(all("artifact missing" in row["result_blockers"] for row in rows))

    def test_q4_tau_source_evidence_cli_writes_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "source_evidence"
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "cgr_exp.cli",
                    "audit-q4-tau-source-evidence",
                    "--output-dir",
                    str(out),
                    "--overwrite",
                ],
                check=False,
                text=True,
                capture_output=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            manifest = json.loads((out / "command_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["all_sources_ready"], True)
            self.assertTrue((out / "q4_tau_source_evidence_audit.csv").exists())

    def test_q4_tau_source_ref_verifier_cli_writes_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "source_refs"
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "cgr_exp.cli",
                    "verify-q4-tau-source-refs",
                    "--output-dir",
                    str(out),
                    "--overwrite",
                ],
                check=False,
                text=True,
                capture_output=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            manifest = json.loads((out / "command_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["all_static_refs_ready"], True)
            self.assertTrue((out / "q4_tau_source_ref_verification.csv").exists())

    def test_tau3_smoke_gate_cli_can_fail_on_not_ready_after_writing_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            row = root / "solver_canonical_smoke_retail"
            row.mkdir()
            out = root / "gate"
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "cgr_exp.cli",
                    "audit-table10-tau3-smoke",
                    "--output-dir",
                    str(out),
                    "--expected-smoke-count",
                    "2",
                    "--row-dirs",
                    str(row),
                    "--fail-on-not-ready",
                    "--overwrite",
                ],
                check=False,
                text=True,
                capture_output=True,
            )
            self.assertEqual(result.returncode, 1)
            manifest = json.loads((out / "table10_tau3_smoke_gate.json").read_text(encoding="utf-8"))
            self.assertFalse(manifest["all_smoke_rows_ready"])
            self.assertTrue((out / "table10_tau3_smoke_gate.csv").exists())

    def test_tau3_trace_audit_cli_manifest_reports_scoring_modes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trace = root / "trace.jsonl"
            out = root / "audit"
            contract = canonical_contract_manifest()
            rows = [
                {
                    "scoring_mode": mode,
                    "canonical_contract": contract["contract"],
                    "canonical_feature_schema_hash": contract["feature_schema_hash"],
                    "canonical_feature_count": len(canonical_feature_names()),
                    "ranker_checkpoint_loaded": True,
                    "candidate_scores": [{"candidate_index": 0, "score": score}],
                }
                for mode, score in (
                    ("canonical_ranker", 1.0),
                    ("greedy_content_passthrough", None),
                    ("fallback_greedy", None),
                )
            ]
            trace.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "cgr_exp.cli",
                    "audit-tau3-controller-trace",
                    "--output-dir",
                    str(out),
                    "--trace-path",
                    str(trace),
                    "--overwrite",
                ],
                check=False,
                text=True,
                capture_output=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            manifest = json.loads((out / "command_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["canonical_ranker_events"], 1)
            self.assertEqual(manifest["greedy_content_passthrough_events"], 1)
            self.assertEqual(manifest["fallback_greedy_events"], 1)


def _read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


if __name__ == "__main__":
    unittest.main()
