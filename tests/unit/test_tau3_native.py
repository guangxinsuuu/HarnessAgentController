from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cgr_exp.tau3_native import (
    Tau3RunConfig,
    aggregate_tau3_official_results,
    audit_table10_tau3_readiness,
    audit_tau3_controller_trace,
    build_tau3_best_of_n_from_native,
    build_tau2_run_command,
    build_table10_tau3_results,
    build_table10_tau3_run_preflight,
    run_tau3_native,
)


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

    def test_dry_run_labels_canonical_and_certified_controller_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            canonical = run_tau3_native(
                Tau3RunConfig(
                    output_dir=root / "canonical",
                    domain="retail",
                    method="solver",
                    num_tasks=1,
                    num_trials=1,
                    dry_run=True,
                    solver_ranker_checkpoint=root / "ranker.json",
                )
            )
            certified = run_tau3_native(
                Tau3RunConfig(
                    output_dir=root / "certified",
                    domain="retail",
                    method="solver",
                    num_tasks=1,
                    num_trials=1,
                    dry_run=True,
                    solver_ranker_checkpoint=root / "ranker.json",
                    enable_certified_filter=True,
                )
            )
            self.assertEqual(canonical["row"]["Method"], "Solver-derived canonical controller")
            self.assertEqual(certified["row"]["Method"], "Solver-derived canonical controller + certified filter")

    def test_existing_complete_results_skip_native_rerun(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            official = root / "native" / "official_tau2_results"
            official.mkdir(parents=True)
            (official / "results.json").write_text(
                json.dumps({"info": {"environment_info": {"domain_name": "retail"}}, "simulations": [_sim("0", 0, 1.0), _sim("1", 0, 0.0)]}),
                encoding="utf-8",
            )
            with (
                patch("cgr_exp.tau3_native.tau3_preflight", return_value={"ready": True, "blockers": []}),
                patch("cgr_exp.tau3_native._tau2_revision", return_value="rev"),
                patch("cgr_exp.tau3_native.subprocess.run") as run_mock,
            ):
                result = run_tau3_native(Tau3RunConfig(output_dir=root / "native", domain="retail", num_tasks=2, num_trials=1))
            self.assertEqual(result["row"]["task_count"], 2)
            run_mock.assert_not_called()

    def test_existing_partial_results_resume_native_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            official = root / "native" / "official_tau2_results"
            official.mkdir(parents=True)
            (official / "results.json").write_text(
                json.dumps({"info": {"environment_info": {"domain_name": "retail"}}, "simulations": [_sim("0", 0, 1.0)]}),
                encoding="utf-8",
            )
            completed = subprocess.CompletedProcess(args=["tau2"], returncode=0, stdout="ok", stderr="")
            with (
                patch("cgr_exp.tau3_native.tau3_preflight", return_value={"ready": True, "blockers": []}),
                patch("cgr_exp.tau3_native._tau2_revision", return_value="rev"),
                patch("cgr_exp.tau3_native.subprocess.run", return_value=completed) as run_mock,
            ):
                run_tau3_native(Tau3RunConfig(output_dir=root / "native", domain="retail", num_tasks=2, num_trials=1))
            run_mock.assert_called_once()
            command_manifest = json.loads((root / "native" / "tau3_native_command.json").read_text(encoding="utf-8"))
            self.assertTrue(command_manifest["resume_from_partial_results"])
            self.assertEqual(command_manifest["existing_task_count"], 1)
            self.assertEqual(command_manifest["expected_task_count"], 2)

    def test_builds_best_of_n_from_native_rollouts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            native = root / "native"
            official = native / "official_tau2_results"
            official.mkdir(parents=True)
            (official / "results.json").write_text(
                json.dumps(
                    {
                        "info": {"environment_info": {"domain_name": "retail"}},
                        "simulations": [
                            _sim("0", 0, 0.0),
                            _sim("0", 1, 1.0),
                            _sim("1", 0, 0.0),
                            _sim("1", 1, 0.0),
                        ],
                    }
                ),
                encoding="utf-8",
            )
            manifest = build_tau3_best_of_n_from_native(root / "best", native, n=2, method="Best-of-2")
            row = manifest["row"]
            self.assertEqual(row["Method"], "Best-of-2")
            self.assertAlmostEqual(row["pass^1"], 0.5)
            self.assertEqual(row["task_count"], 2)
            self.assertEqual(row["simulation_count"], 4)
            self.assertAlmostEqual(row["tokens"], 42.0)
            self.assertAlmostEqual(row["tool_calls"], 2.0)
            self.assertIn("benchmark_compatible_selection", manifest["manifest"]["method_protocols"]["Best-of-2"]["method_primitives"])
            self.assertTrue((root / "best" / "q4_tau_table.csv").exists())

    def test_audits_tau3_controller_trace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trace = root / "tau3_controller_trace.jsonl"
            trace.write_text(
                json.dumps(
                    {
                        "event": "tau3_controller_selection",
                        "scoring_mode": "canonical_ranker",
                        "ranker_checkpoint_loaded": True,
                        "certified_filter_enabled": True,
                        "certified_pruned_count": 1,
                        "oracle_visible_to_controller": False,
                        "oracle_used_for_candidate_generation": False,
                        "oracle_used_for_filtering": False,
                        "tool_names_used_as_features": False,
                        "candidate_scores": [
                            {"candidate_index": 0, "score": 0.1, "action_type": "read"},
                            {"candidate_index": 1, "score": 0.2, "action_type": "write"},
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            manifest = audit_tau3_controller_trace(root / "audit", trace)
            self.assertEqual(manifest["event_count"], 1)
            self.assertEqual(manifest["canonical_ranker_events"], 1)
            self.assertEqual(manifest["ranker_checkpoint_loaded_events"], 1)
            self.assertEqual(manifest["certified_filter_enabled_events"], 1)
            self.assertEqual(manifest["certified_pruned_candidates"], 1)
            self.assertAlmostEqual(manifest["certified_prune_rate"], 0.5)
            self.assertFalse(manifest["oracle_visible_to_controller"])
            self.assertFalse(manifest["tool_names_used_as_features"])
            self.assertTrue((root / "audit" / "tau3_controller_trace_audit.csv").exists())

    def test_audits_table10_tau3_readiness_with_canonical_trace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            native = root / "native"
            canonical = root / "canonical"
            native.mkdir()
            canonical.mkdir()
            _write_row(native, "retail", method="Native tool-calling rerun", task_count=2, status="complete")
            _write_row(canonical, "retail", method="Solver-derived canonical controller", task_count=2, status="complete")
            (canonical / "tau3_native_command.json").write_text(json.dumps({"solver_ranker_checkpoint": "ranker.json"}), encoding="utf-8")
            (canonical / "tau3_controller_trace.jsonl").write_text(
                json.dumps(
                    {
                        "scoring_mode": "canonical_ranker",
                        "ranker_checkpoint_loaded": True,
                        "certified_filter_enabled": False,
                        "certified_pruned_count": 0,
                        "oracle_visible_to_controller": False,
                        "oracle_used_for_candidate_generation": False,
                        "oracle_used_for_filtering": False,
                        "tool_names_used_as_features": False,
                        "candidate_scores": [{"candidate_index": 0, "score": 1.0}],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            readiness = audit_table10_tau3_readiness(root / "audit", [native, canonical], expected_test_count=2)
            self.assertTrue(readiness["all_rows_result_ready"])
            rows = {row["Method"]: row for row in readiness["audit_rows"]}
            self.assertTrue(rows["Solver-derived canonical controller"]["result_ready"])
            self.assertEqual(rows["Solver-derived canonical controller"]["source_artifact"], str(canonical))
            self.assertEqual(rows["Solver-derived canonical controller"]["source_policy"], "official tau3 maintained protocol")

    def test_audits_table10_tau3_readiness_rejects_missing_canonical_trace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            canonical = root / "canonical"
            canonical.mkdir()
            _write_row(canonical, "retail", method="Solver-derived canonical controller", task_count=1, status="complete")
            (canonical / "tau3_native_command.json").write_text(json.dumps({"solver_ranker_checkpoint": "ranker.json"}), encoding="utf-8")
            readiness = audit_table10_tau3_readiness(root / "audit", [canonical], expected_test_count=2)
            self.assertFalse(readiness["all_rows_result_ready"])
            self.assertIn("retail: Solver-derived canonical controller", readiness["pending_or_partial_rows"])
            blockers = readiness["audit_rows"][0]["result_blockers"]
            self.assertIn("partial: 1/2", blockers)
            self.assertIn("controller trace missing", blockers)

    def test_audits_table10_tau3_readiness_labels_missing_expected_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            readiness = audit_table10_tau3_readiness(
                root / "audit",
                [
                    root / "native_retail",
                    root / "best_of_4_airline",
                    root / "solver_canonical_certified_retail",
                ],
                expected_test_count=2,
            )
            rows = {(row["Domain"], row["Method"]): row for row in readiness["audit_rows"]}
            self.assertIn(("retail", "Native tool-calling rerun"), rows)
            self.assertIn(("airline", "Best-of-N"), rows)
            self.assertIn(("retail", "Solver-derived canonical controller + certified filter"), rows)
            self.assertIn("retail: Native tool-calling rerun", readiness["pending_or_partial_rows"])
            self.assertEqual(rows[("airline", "Best-of-N")]["source_artifact"], str(root / "best_of_4_airline"))

    def test_builds_table10_tau3_run_preflight(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = build_table10_tau3_run_preflight(
                root / "preflight",
                base_dir=root / "runs",
                expected_test_count=3,
                best_of_n=4,
                num_trials=4,
                max_concurrency=2,
                tau_python_bin="/tmp/python",
            )
            groups = {row["run_group"]: row for row in manifest["rows"]}
            self.assertIn("full", groups)
            self.assertIn("native_full", groups)
            self.assertIn("solver_canonical_full", groups)
            self.assertIn("solver_canonical_certified_full", groups)
            self.assertTrue(groups["full"]["api_required"])
            self.assertIn("build_and_audit", groups)
            self.assertTrue(groups["native_full"]["api_required"])
            self.assertFalse(groups["build_and_audit"]["api_required"])
            self.assertTrue(manifest["parallel_domains"])
            self.assertFalse(manifest["parallel_method_groups"])
            self.assertEqual(groups["native_full"]["effective_max_concurrency"], 4)
            self.assertEqual(groups["full"]["effective_max_concurrency"], 4)
            self.assertEqual(manifest["max_api_backed_effective_concurrency"], 4)
            self.assertEqual(manifest["recommended_next_run_group"], "native_full")
            self.assertIn("scripts/run_table10_tau3.sh native-full", manifest["recommended_next_command"])
            self.assertIn("scripts/run_table10_tau3.sh full", groups["full"]["command"])
            self.assertIn("PARALLEL_METHOD_GROUPS=0", groups["full"]["command"])
            self.assertIn("scripts/run_table10_tau3.sh native-full", groups["native_full"]["command"])
            self.assertIn("PARALLEL_DOMAINS=1", groups["native_full"]["command"])
            self.assertTrue((root / "preflight" / "table10_tau3_run_preflight.csv").exists())
            self.assertTrue((root / "preflight" / "table10_tau3_run_preflight.md").exists())

    def test_table10_preflight_recommends_next_missing_group(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "runs"
            for row_dir, domain, method in (
                (base / "native_retail", "retail", "Native tool-calling rerun"),
                (base / "native_airline", "airline", "Native tool-calling rerun"),
                (base / "best_of_4_retail", "retail", "Best-of-N"),
                (base / "best_of_4_airline", "airline", "Best-of-N"),
            ):
                row_dir.mkdir(parents=True)
                _write_row(row_dir, domain, method=method, task_count=3, status="complete")
            manifest = build_table10_tau3_run_preflight(root / "preflight", base_dir=base, expected_test_count=3)
            self.assertEqual(manifest["recommended_next_run_group"], "legacy_solver_full")
            self.assertIn("scripts/run_table10_tau3.sh solver-full", manifest["recommended_next_command"])

    def test_table10_preflight_reports_parallel_method_group_concurrency(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = build_table10_tau3_run_preflight(
                root / "preflight",
                base_dir=root / "runs",
                expected_test_count=3,
                max_concurrency=2,
                parallel_domains=True,
                parallel_method_groups=True,
            )
            groups = {row["run_group"]: row for row in manifest["rows"]}
            self.assertEqual(groups["full"]["effective_max_concurrency"], 16)
            self.assertEqual(groups["native_full"]["effective_max_concurrency"], 4)
            self.assertEqual(groups["build_and_audit"]["effective_max_concurrency"], 0)
            self.assertEqual(manifest["max_api_backed_effective_concurrency"], 16)


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


def _write_row(path: Path, domain: str, method: str = "Native tool-calling rerun", task_count: int = 2, status: str = "complete") -> None:
    (path / "q4_tau_row.json").write_text(
        json.dumps(
            {
                "Domain": domain,
                "Method": method,
                "pass^1": 0.5,
                "pass^4": 0.75,
                "violations": 0.0,
                "tool_calls": 3.0,
                "tokens": 123.0,
                "wall_clock_seconds": 4.0,
                "status": status,
                "task_count": task_count,
                "official_results": "x",
            }
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    unittest.main()
