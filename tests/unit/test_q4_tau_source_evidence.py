from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from cgr_exp.q4_tau_source_evidence import audit_q4_tau_source_evidence, q4_tau_source_evidence_summary
from cgr_exp.q4_tau_source_ref_verifier import verify_q4_tau_source_refs
from cgr_exp.q4_tau_adapter_contract import build_q4_tau_adapter_contract


class Q4TauSourceEvidenceTest(unittest.TestCase):
    def test_source_evidence_covers_tau_and_tau2_with_pinned_refs(self) -> None:
        summary = q4_tau_source_evidence_summary()
        self.assertEqual(set(summary["benchmarks"]), {"tau-bench", "tau2-bench"})
        self.assertTrue(summary["all_sources_ready"])
        by_benchmark = {row["Benchmark"]: row for row in summary["rows"]}
        self.assertEqual(by_benchmark["tau-bench"]["source_revision"], "59a200c6d575d595120f1cb70fea53cef0632f6b")
        self.assertEqual(by_benchmark["tau2-bench"]["source_revision"], "a2c024725189473d2d7cea3a5cfdbcc67478e41f")
        self.assertIn("README.md", by_benchmark["tau-bench"]["source_files"])
        self.assertIn("tool-call", by_benchmark["tau2-bench"]["official_primitives"])

    def test_source_evidence_audit_writes_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            manifest = audit_q4_tau_source_evidence(out)
            self.assertTrue(manifest["all_sources_ready"])
            self.assertTrue((out / "q4_tau_source_evidence_audit.csv").exists())
            saved = json.loads((out / "q4_tau_source_evidence_audit.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["benchmarks"], ["tau-bench", "tau2-bench"])

    def test_source_ref_verifier_writes_static_readiness_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            manifest = verify_q4_tau_source_refs(out)
            self.assertTrue(manifest["all_static_refs_ready"])
            self.assertTrue((out / "q4_tau_source_ref_verification.csv").exists())
            saved = json.loads((out / "q4_tau_source_ref_verification.json").read_text(encoding="utf-8"))
            by_benchmark = {row["Benchmark"]: row for row in saved["rows"]}
            self.assertGreater(by_benchmark["tau-bench"]["raw_url_count"], 0)
            self.assertGreater(by_benchmark["tau2-bench"]["symbol_ref_count"], 0)

    def test_source_ref_verifier_can_run_online_check_with_injected_urlopen(self) -> None:
        seen_urls = []

        class FakeResponse:
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self) -> bytes:
                return b"dynamic conversations between a user\nEach domain specifies:\n"

        def fake_urlopen(request, timeout=0):
            seen_urls.append(request.full_url)
            return FakeResponse()

        with tempfile.TemporaryDirectory() as tmp:
            manifest = verify_q4_tau_source_refs(Path(tmp), online=True, timeout=3.0, urlopen=fake_urlopen)
            self.assertTrue(manifest["all_online_refs_ready"])
            self.assertTrue(any("sierra-research/tau-bench/commit" in url for url in seen_urls))
            self.assertTrue(any("raw.githubusercontent.com/sierra-research/tau2-bench" in url for url in seen_urls))
            by_benchmark = {row["Benchmark"]: row for row in manifest["rows"]}
            self.assertEqual(by_benchmark["tau-bench"]["raw_anchor_match_count"], 1)
            self.assertEqual(by_benchmark["tau2-bench"]["raw_anchor_match_count"], 1)
            locations = json.loads(by_benchmark["tau-bench"]["raw_anchor_locations"])
            self.assertTrue(any("dynamic conversations between a user" in key for key in locations))
            self.assertIn(1, next(iter(locations.values())))

    def test_source_ref_verifier_rejects_online_raw_anchor_mismatch(self) -> None:
        class FakeResponse:
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self) -> bytes:
                return b"reachable page without the pinned benchmark anchor"

        def fake_urlopen(request, timeout=0):
            del request, timeout
            return FakeResponse()

        with tempfile.TemporaryDirectory() as tmp:
            manifest = verify_q4_tau_source_refs(Path(tmp), online=True, timeout=3.0, urlopen=fake_urlopen)
            self.assertFalse(manifest["all_online_refs_ready"])
            by_benchmark = {row["Benchmark"]: row for row in manifest["rows"]}
            self.assertIn("missing raw content anchor", by_benchmark["tau-bench"]["online_errors"])

    def test_adapter_contract_writes_native_protocol_requirements(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            manifest = build_q4_tau_adapter_contract(out, expected_test_count=7)
            self.assertEqual(manifest["expected_test_count"], 7)
            self.assertEqual(manifest["adapter_status"]["status"], "stub")
            self.assertIn("official_task_verifier", manifest["native_protocol_required"])
            self.assertIn("q4_tau_manifest.json", manifest["required_artifacts"])
            by_method = {row["Method"]: row for row in manifest["rows"]}
            self.assertIn("zero_shot_target_tool_transfer", by_method["Solver-derived controller"]["method_primitives_required"])
            self.assertIn("benchmark-compatible selection", by_method["Best-of-N"]["method_requirements"])
            text = (out / "q4_tau_adapter_contract.md").read_text(encoding="utf-8")
            self.assertIn("synthetic", text)
            self.assertIn("must not populate Q4 rows", text)
            self.assertTrue((out / "q4_tau_adapter_contract.csv").exists())


if __name__ == "__main__":
    unittest.main()
