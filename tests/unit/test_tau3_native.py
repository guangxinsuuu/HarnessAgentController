from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from cgr_exp.tau3_native import Tau3RunConfig, aggregate_tau3_official_results, build_tau2_run_command, build_table10_tau3_results


class Tau3NativeTest(unittest.TestCase):
    def test_builds_official_tau2_cli_command(self) -> None:
        cfg = Tau3RunConfig(
            output_dir=Path("runs/x"),
            domain="retail",
            num_tasks=5,
            num_trials=4,
            max_concurrency=6,
            workers=2,
            python_bin="/tmp/python",
            timeout=300,
        )
        command = build_tau2_run_command(cfg, save_to="out/results.json")
        self.assertEqual(command[:4], ["/tmp/python", "-m", "tau2.cli", "run"])
        self.assertIn("--domain", command)
        self.assertIn("retail", command)
        self.assertIn("--task-split-name", command)
        self.assertIn("base", command)
        self.assertIn("--num-trials", command)
        self.assertIn("4", command)
        self.assertIn("--max-concurrency", command)
        self.assertIn("6", command)
        self.assertIn("--workers", command)
        self.assertIn("2", command)
        self.assertIn("--auto-resume", command)
        self.assertIn("--enforce-communication-protocol", command)
        save_to = command[command.index("--save-to") + 1]
        self.assertFalse(save_to.endswith("results.json"))

    def test_aggregates_official_results_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = root / "tau"
            official = out / "official_tau2_results"
            official.mkdir(parents=True)
            (official / "results.json").write_text(
                json.dumps(
                    {
                        "info": {"environment_info": {"domain_name": "airline"}},
                        "tasks": [{"id": "0"}, {"id": "1"}],
                        "simulations": [
                            _sim("0", 0, 1.0),
                            _sim("0", 1, 1.0),
                            _sim("0", 2, 0.0),
                            _sim("0", 3, 0.0),
                            _sim("1", 0, 0.0),
                            _sim("1", 1, 0.0),
                            _sim("1", 2, 0.0),
                            _sim("1", 3, 0.0),
                        ],
                    }
                ),
                encoding="utf-8",
            )
            manifest = aggregate_tau3_official_results(out, domain="airline", expected_trials=4)
            row = manifest["row"]
            self.assertEqual(row["Domain"], "airline")
            self.assertAlmostEqual(row["pass^1"], 0.25)
            self.assertAlmostEqual(row["pass^4"], 0.0)
            self.assertEqual(row["task_count"], 2)
            self.assertEqual(row["simulation_count"], 8)
            self.assertTrue((out / "q4_tau_row.json").exists())
            self.assertTrue((out / "q4_tau_per_instance.csv").exists())

    def test_aggregate_skips_directory_named_results_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / "official_tau2_results" / "results.json").mkdir(parents=True)
            manifest = aggregate_tau3_official_results(out, method="Native tool-calling rerun", domain="retail")
            self.assertEqual(manifest["row"]["status"], "official results missing")

    def test_build_table10_from_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            retail = root / "retail"
            airline = root / "airline"
            retail.mkdir()
            airline.mkdir()
            _write_row(retail, "retail")
            _write_row(airline, "airline")
            manifest = build_table10_tau3_results(root / "table", [retail, airline], expected_test_count=100)
            self.assertEqual(len(manifest["rows"]), 2)
            self.assertTrue((root / "table" / "table10_tau3_results.csv").exists())
            self.assertIn("partial: 2/100", (root / "table" / "table10_tau3_results.csv").read_text(encoding="utf-8"))


def _sim(task_id: str, trial: int, reward: float) -> dict:
    return {
        "id": f"{task_id}-{trial}",
        "task_id": task_id,
        "trial": trial,
        "duration": 1.5,
        "termination_reason": "user_stop",
        "reward_info": {"reward": reward},
        "messages": [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"name": "x", "arguments": {}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            },
            {"role": "user", "content": "ok", "usage": {"prompt_tokens": 4, "completion_tokens": 2}},
        ],
    }


def _write_row(path: Path, domain: str) -> None:
    (path / "q4_tau_row.json").write_text(
        json.dumps(
            {
                "Domain": domain,
                "Method": "Native tool-calling rerun",
                "pass^1": 0.5,
                "pass^4": 0.75,
                "violations": 0.0,
                "tool_calls": 3.0,
                "tokens": 123.0,
                "wall_clock_seconds": 4.0,
                "status": "complete",
                "task_count": 2,
                "official_results": "x",
            }
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    unittest.main()
