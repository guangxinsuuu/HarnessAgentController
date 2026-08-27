from __future__ import annotations

import unittest

from cgr_exp.datasets.external_tool_benchmark import ExternalToolBenchmarkAdapter


class ExternalBenchmarkAdapterTest(unittest.TestCase):
    def test_external_adapter_is_explicit_stub_without_network_requirement(self) -> None:
        adapter = ExternalToolBenchmarkAdapter(name="tau-bench-like", enabled=False)
        self.assertFalse(adapter.is_available())
        self.assertEqual(adapter.to_manifest()["status"], "stub")
        self.assertEqual(adapter.to_manifest()["requires_network_for_smoke"], False)
        with self.assertRaises(RuntimeError):
            adapter.load_tasks(split="test")


if __name__ == "__main__":
    unittest.main()
