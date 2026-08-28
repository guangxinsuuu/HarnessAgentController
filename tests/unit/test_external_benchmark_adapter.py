from __future__ import annotations

import unittest
from pathlib import Path

from cgr_exp.datasets.external_tool_benchmark import ExternalToolBenchmarkAdapter


class ExternalBenchmarkAdapterTest(unittest.TestCase):
    def test_external_adapter_is_explicit_stub_without_network_requirement(self) -> None:
        adapter = ExternalToolBenchmarkAdapter(name="tau-bench-like", enabled=False)
        self.assertFalse(adapter.is_available())
        self.assertEqual(adapter.to_manifest()["status"], "stub")
        self.assertEqual(adapter.to_manifest()["requires_network_for_smoke"], False)
        self.assertEqual(adapter.to_manifest()["result_ready"], False)
        self.assertEqual(adapter.to_manifest()["source_url"], "https://github.com/sierra-research/tau-bench")
        self.assertEqual(adapter.to_manifest()["preferred_current_source_url"], "https://github.com/sierra-research/tau2-bench")
        self.assertIn("native stateful tool-agent", adapter.to_manifest()["native_protocol_reference"])
        self.assertIn("no local native tau-bench runner implemented", adapter.to_manifest()["implementation_scope"])
        with self.assertRaises(RuntimeError):
            adapter.load_tasks(split="test")

    def test_tau_source_alignment_doc_records_stub_boundary(self) -> None:
        text = Path("docs/q4_tau_source_alignment.md").read_text(encoding="utf-8")
        self.assertIn("https://github.com/sierra-research/tau-bench", text)
        self.assertIn("https://github.com/sierra-research/tau2-bench", text)
        self.assertIn("not tau-bench", text)
        self.assertIn("result_ready = false", text)
        self.assertIn("no native runner implemented", text)


if __name__ == "__main__":
    unittest.main()
