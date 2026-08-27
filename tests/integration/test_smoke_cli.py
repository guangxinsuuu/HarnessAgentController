from __future__ import annotations

import json
import math
import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REQUIRED = [
    "config.json",
    "environment.json",
    "metrics.json",
    "metrics_by_instance.jsonl",
    "primary_table.csv",
    "transfer_table.csv",
    "ranking_table.csv",
    "certification_table.csv",
    "proposal_table.csv",
    "budget_table.csv",
    "budget_protocol.json",
    "domain_shift_table.csv",
    "external_benchmark_table.csv",
    "trace_sample.jsonl",
    "train_history.jsonl",
    "family_split_manifest.json",
    "ranking_label_schema.json",
    "reproducibility_manifest.json",
    "stdout.log",
    "manifest.json",
]


class SmokeCliTest(unittest.TestCase):
    def test_smoke_cli_writes_required_artifacts_without_nan(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "smoke"
            cmd = [sys.executable, "-m", "cgr_exp.run_experiment", "--preset", "smoke", "--out", str(out), "--seed", "5"]
            subprocess.run(cmd, check=True, cwd=Path(__file__).resolve().parents[2], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            for name in REQUIRED:
                self.assertTrue((out / name).exists(), name)
            metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
            self.assertGreater(metrics["solver_trace_count"], 0)
            self.assertGreater(metrics["level2_trace_count"], 0)
            self.assertIn("zero_shot", metrics["transfer"])
            self.assertEqual(metrics["config"]["feature_schema_version"], "shared-node-v1")
            self.assertIn("frontier_size_at_creation", metrics["shared_feature_schema"]["universal"])
            self.assertIn("normalized_bound_gap", metrics["shared_feature_schema"]["verifier"])
            self.assertIn("lower_bound", metrics["shared_feature_schema"]["mask_keys"])
            self.assertGreaterEqual(metrics["domain_classifier"]["heldout_accuracy"], 0.0)
            self.assertGreaterEqual(metrics["domain_classifier"]["auc"], 0.0)
            self.assertEqual(metrics["domain_classifier"]["proxy_formula"], "2 * (2 * heldout_accuracy - 1), clipped to [0, 2]")
            with (out / "primary_table.csv").open(encoding="utf-8") as f:
                primary_fields = next(csv.reader(f))
            self.assertEqual(primary_fields, ["Method", "Success", "Gap", "Violations", "False prune", "LLM calls", "Tokens", "Nodes", "Latency"])
            with (out / "transfer_table.csv").open(encoding="utf-8") as f:
                transfer_fields = next(csv.reader(f))
            self.assertEqual(transfer_fields, ["Setting", "Target trajectories used", "Ranking accuracy", "Success", "Expanded nodes", "LLM calls"])
            with (out / "certification_table.csv").open(encoding="utf-8") as f:
                certification_fields = next(csv.reader(f))
            self.assertEqual(certification_fields, ["setting", "verified precision", "verified recall", "bound slack", "false prune", "nodes", "success"])
            with (out / "domain_shift_table.csv").open(encoding="utf-8") as f:
                domain_rows = list(csv.DictReader(f))
            self.assertNotEqual(domain_rows[0]["domain_classifier_accuracy"], "--")
            self.assertNotEqual(domain_rows[0]["auc"], "--")
            with (out / "llm_adapter_table.csv").open(encoding="utf-8") as f:
                llm_adapter_rows = list(csv.DictReader(f))
            http_rows = [row for row in llm_adapter_rows if row["setting"] == "openai-compatible-http-client"]
            self.assertEqual(http_rows[0]["network_required_for_smoke"], "False")
            protocol = json.loads((out / "budget_protocol.json").read_text(encoding="utf-8"))
            self.assertEqual(protocol["fairness_rule"], "same task, proposer, branching factor, budget, and information access")
            self.assertEqual(protocol["proposal_branching_factor_k"], metrics["config"]["proposal_budget"])
            self.assertTrue(protocol["all_methods_share_test_instances"])
            self.assertIn("proposal_call_budget", protocol["primary_budgets"])

            def walk(obj):
                if isinstance(obj, float):
                    self.assertFalse(math.isnan(obj))
                elif isinstance(obj, dict):
                    for value in obj.values():
                        walk(value)
                elif isinstance(obj, list):
                    for value in obj:
                        walk(value)

            walk(metrics)


if __name__ == "__main__":
    unittest.main()
