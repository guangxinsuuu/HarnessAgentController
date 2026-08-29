from __future__ import annotations

import json
import os
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
    def test_table10_tau3_script_guards_api_backed_runs(self) -> None:
        script = Path("scripts/run_table10_tau3.sh")
        text = script.read_text(encoding="utf-8")
        self.assertIn("require_tau3_api_key", text)
        self.assertIn("CGR_LLM_API_KEY", text)
        self.assertIn('export OPENAI_API_KEY="${CGR_LLM_API_KEY}"', text)
        self.assertIn("missing OPENAI_API_KEY or CGR_LLM_API_KEY", text)
        self.assertIn("run_domain_pair() {\n  require_tau3_api_key", text)
        self.assertIn("CANDIDATE_WORKERS", text)
        self.assertIn("source-controller-smoke", text)
        self.assertIn("run_source_controller_smoke", text)
        result = subprocess.run(["zsh", "-n", str(script)], check=False, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_table10_tau3_source_controller_smoke_is_local(self) -> None:
        script = Path("scripts/run_table10_tau3.sh")
        with tempfile.TemporaryDirectory() as tmp:
            env = {
                **os.environ,
                "PYTHON_BIN": "/usr/bin/true",
                "BASE_DIR": str(Path(tmp) / "tau3"),
            }
            env.pop("OPENAI_API_KEY", None)
            env.pop("CGR_LLM_API_KEY", None)
            result = subprocess.run(["zsh", str(script), "source-controller-smoke"], check=False, text=True, capture_output=True, env=env)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_table10_tau3_api_guard_fails_before_api_run_without_key(self) -> None:
        script = Path("scripts/run_table10_tau3.sh")
        with tempfile.TemporaryDirectory() as tmp:
            env = {
                **os.environ,
                "PYTHON_BIN": "/usr/bin/true",
                "BASE_DIR": str(Path(tmp) / "tau3"),
                "NUM_TASKS": "1",
            }
            env.pop("OPENAI_API_KEY", None)
            env.pop("CGR_LLM_API_KEY", None)
            result = subprocess.run(["zsh", str(script), "native-full"], check=False, text=True, capture_output=True, env=env)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("missing OPENAI_API_KEY or CGR_LLM_API_KEY", result.stderr)

    def test_table10_tau3_api_guard_accepts_cgr_key_alias(self) -> None:
        script = Path("scripts/run_table10_tau3.sh")
        with tempfile.TemporaryDirectory() as tmp:
            env = {
                **os.environ,
                "PYTHON_BIN": "/usr/bin/true",
                "BASE_DIR": str(Path(tmp) / "tau3"),
                "NUM_TASKS": "1",
                "CGR_LLM_API_KEY": "test-key",
            }
            env.pop("OPENAI_API_KEY", None)
            result = subprocess.run(["zsh", str(script), "native-full"], check=False, text=True, capture_output=True, env=env)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_tau3_runbook_documents_current_execution_protocol(self) -> None:
        text = Path("docs/tau3_runbook.md").read_text(encoding="utf-8")
        self.assertIn("scripts/run_table10_tau3.sh preflight-runs", text)
        self.assertIn("scripts/run_table10_tau3.sh source-controller", text)
        self.assertIn("scripts/run_table10_tau3.sh source-controller-smoke", text)
        self.assertIn("scripts/run_table10_tau3.sh solver-canonical-smoke", text)
        self.assertIn("scripts/run_table10_tau3.sh solver-canonical-certified-smoke", text)
        self.assertIn("scripts/run_table10_tau3.sh full", text)
        self.assertIn("scripts/run_table10_tau3.sh build", text)
        self.assertIn("source_controller_seed${seed}/generic_harness_ranker.json", text)
        self.assertIn("OPENAI_API_KEY", text)
        self.assertIn("CGR_LLM_API_KEY", text)
        self.assertIn("CANDIDATE_WORKERS=4", text)
        self.assertIn("tool-identity stress passed", text)
        self.assertIn("contract-coverage stress passed", text)
        self.assertIn("result_ready=True", text)

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
            csv_text = (root / "table" / "table10_tau3_results.csv").read_text(encoding="utf-8")
            tex_text = (root / "table" / "table10_tau3_results.tex").read_text(encoding="utf-8")
            self.assertIn("LLM calls", csv_text)
            self.assertIn("Wall-clock", csv_text)
            self.assertIn("partial: 2/100", csv_text)
            self.assertIn("LLM calls", tex_text)
            self.assertIn("Wall-clock", tex_text)

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
                    candidate_workers=3,
                )
            )
            self.assertEqual(canonical["row"]["Method"], "Solver-derived canonical controller")
            self.assertEqual(certified["row"]["Method"], "Solver-derived canonical controller + certified filter")
            command_manifest = json.loads((root / "certified" / "tau3_native_command.json").read_text(encoding="utf-8"))
            self.assertEqual(command_manifest["candidate_workers"], 3)
            reported_metrics = canonical["manifest"]["native_protocol"]["reported_metrics"]
            self.assertIn("llm_calls", reported_metrics)
            self.assertIn("wall_clock_seconds", reported_metrics)

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
            self.assertEqual(row["selector"], "official_reward_best_of_n")
            self.assertTrue(row["selector_full_coverage"])
            self.assertIn("benchmark_compatible_selection", manifest["manifest"]["method_protocols"]["Best-of-2"]["method_primitives"])
            self.assertTrue((root / "best" / "q4_tau_table.csv").exists())
            per_instance = (root / "best" / "q4_tau_per_instance.csv").read_text(encoding="utf-8")
            self.assertIn("selected_rollout_id", per_instance)
            self.assertIn("candidate_rewards", per_instance)
            self.assertIn("rollout_coverage_ready", per_instance)

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
            source = root / "source"
            source.mkdir()
            _write_source_manifest(source, tau3_train_trajectories_used=False)
            (canonical / "tau3_native_command.json").write_text(
                json.dumps(
                    {
                        "solver_ranker_checkpoint": str(source / "generic_harness_ranker.json"),
                        "source_controller_protocol": {
                            "manifest_found": True,
                            "controller_checkpoint_frozen": True,
                            "tau3_train_trajectories_used": False,
                            "oracle_labels_in_inference_features": False,
                            "tool_names_used_as_features": False,
                            "source_generator_distribution_fixed_before_target_test_evaluation": True,
                            "tau3_heldout_test_seen_before_freeze": False,
                            "source_generator_uses_tau3_trajectories": False,
                            "source_generator_anonymized_tool_names": True,
                            "tool_identity_generalization_ready": True,
                            "tool_identity_generalization_top1_delta": 0.0,
                            "contract_coverage_stress_ready": True,
                            "contract_coverage_black_box_top1_delta_from_full": 0.0,
                        },
                    }
                ),
                encoding="utf-8",
            )
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
            canonical_row = rows["Solver-derived canonical controller"]
            self.assertTrue(canonical_row["source_generator_freeze_ready"])
            self.assertTrue(canonical_row["source_generator_anonymized_tools"])
            self.assertTrue(canonical_row["tool_identity_generalization_ready"])
            self.assertTrue(canonical_row["contract_coverage_stress_ready"])
            self.assertEqual(canonical_row["canonical_ranker_events"], 1)
            self.assertIn("source generator frozen before target test", canonical_row["notes"])
            self.assertIn("tool-identity stress delta=0.0", canonical_row["notes"])
            self.assertIn("contract coverage black-box delta=0.0", canonical_row["notes"])

    def test_audits_table10_tau3_readiness_rejects_missing_compute_accounting(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            native = root / "native"
            native.mkdir()
            _write_row(native, "retail", method="Native tool-calling rerun", task_count=2, status="complete")
            row = json.loads((native / "q4_tau_row.json").read_text(encoding="utf-8"))
            row["llm_calls"] = ""
            row["tokens"] = ""
            (native / "q4_tau_row.json").write_text(json.dumps(row), encoding="utf-8")
            readiness = audit_table10_tau3_readiness(root / "audit", [native], expected_test_count=2)
            self.assertFalse(readiness["all_rows_result_ready"])
            blockers = readiness["audit_rows"][0]["result_blockers"]
            self.assertIn("LLM call accounting missing", blockers)
            self.assertIn("token accounting missing", blockers)

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
            self.assertIn("source controller manifest missing", blockers)

    def test_audits_table10_tau3_readiness_rejects_source_controller_leakage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            canonical = root / "solver_canonical_retail"
            source = root / "source"
            canonical.mkdir()
            source.mkdir()
            _write_row(canonical, "retail", method="Solver-derived canonical controller", task_count=2, status="complete")
            _write_source_manifest(source, tau3_train_trajectories_used=True)
            (canonical / "tau3_native_command.json").write_text(json.dumps({"solver_ranker_checkpoint": str(source / "generic_harness_ranker.json")}), encoding="utf-8")
            (canonical / "tau3_controller_trace.jsonl").write_text(
                json.dumps(
                    {
                        "scoring_mode": "canonical_ranker",
                        "ranker_checkpoint_loaded": True,
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
            readiness = audit_table10_tau3_readiness(root / "audit", [canonical], expected_test_count=2)
            self.assertFalse(readiness["all_rows_result_ready"])
            self.assertIn("source controller used tau3 train trajectories", readiness["audit_rows"][0]["result_blockers"])

    def test_audits_table10_tau3_readiness_rejects_source_generator_target_tuning(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            canonical = root / "solver_canonical_retail"
            source = root / "source"
            canonical.mkdir()
            source.mkdir()
            _write_row(canonical, "retail", method="Solver-derived canonical controller", task_count=2, status="complete")
            _write_source_manifest(source, tau3_train_trajectories_used=False)
            manifest_path = source / "canonical_harness_source_validation.json"
            source_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            source_manifest["source_generator"]["source_generator_distribution_fixed_before_target_test_evaluation"] = False
            source_manifest["source_generator"]["tau3_heldout_test_seen_before_freeze"] = True
            source_manifest["source_generator"]["anonymized_tool_names"] = False
            manifest_path.write_text(json.dumps(source_manifest), encoding="utf-8")
            (canonical / "tau3_native_command.json").write_text(json.dumps({"solver_ranker_checkpoint": str(source / "generic_harness_ranker.json")}), encoding="utf-8")
            (canonical / "tau3_controller_trace.jsonl").write_text(
                json.dumps(
                    {
                        "scoring_mode": "canonical_ranker",
                        "ranker_checkpoint_loaded": True,
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
            readiness = audit_table10_tau3_readiness(root / "audit", [canonical], expected_test_count=2)
            blockers = readiness["audit_rows"][0]["result_blockers"]
            self.assertIn("tau3 heldout test influenced source generator freeze", blockers)
            self.assertIn("source generator freeze evidence missing", blockers)
            self.assertIn("source generator tool-name anonymization evidence missing", blockers)

    def test_audits_table10_tau3_readiness_rejects_missing_contract_coverage_stress(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            canonical = root / "solver_canonical_retail"
            source = root / "source"
            canonical.mkdir()
            source.mkdir()
            _write_row(canonical, "retail", method="Solver-derived canonical controller", task_count=2, status="complete")
            _write_source_manifest(source, tau3_train_trajectories_used=False)
            manifest_path = source / "canonical_harness_source_validation.json"
            source_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            source_manifest["contract_coverage_stress"]["ready"] = False
            manifest_path.write_text(json.dumps(source_manifest), encoding="utf-8")
            (canonical / "tau3_native_command.json").write_text(json.dumps({"solver_ranker_checkpoint": str(source / "generic_harness_ranker.json")}), encoding="utf-8")
            (canonical / "tau3_controller_trace.jsonl").write_text(
                json.dumps(
                    {
                        "scoring_mode": "canonical_ranker",
                        "ranker_checkpoint_loaded": True,
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
            readiness = audit_table10_tau3_readiness(root / "audit", [canonical], expected_test_count=2)
            self.assertIn("contract coverage stress missing or failed", readiness["audit_rows"][0]["result_blockers"])

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

    def test_audits_table10_tau3_readiness_rejects_best_of_n_without_selector_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            best = root / "best_of_4_retail"
            best.mkdir()
            _write_row(best, "retail", method="Best-of-N", task_count=2, status="complete")
            readiness = audit_table10_tau3_readiness(root / "audit", [best], expected_test_count=2)
            self.assertFalse(readiness["all_rows_result_ready"])
            blockers = readiness["audit_rows"][0]["result_blockers"]
            self.assertIn("Best-of-N selector evidence missing", blockers)
            self.assertIn("Best-of-N rollout coverage incomplete", blockers)

    def test_builds_table10_tau3_run_preflight(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = build_table10_tau3_run_preflight(
                root / "preflight",
                base_dir=root / "runs",
                expected_test_count=3,
                best_of_n=4,
                num_trials=4,
                seed=2,
                max_concurrency=2,
                candidate_workers=3,
                tau_python_bin="/tmp/python",
            )
            groups = {row["run_group"]: row for row in manifest["rows"]}
            self.assertIn("source_controller", groups)
            self.assertIn("full", groups)
            self.assertIn("native_full", groups)
            self.assertIn("solver_canonical_full", groups)
            self.assertIn("solver_canonical_certified_full", groups)
            self.assertFalse(groups["source_controller"]["api_required"])
            self.assertEqual(groups["source_controller"]["effective_max_concurrency"], 0)
            self.assertTrue(groups["full"]["api_required"])
            self.assertIn("build_and_audit", groups)
            self.assertTrue(groups["native_full"]["api_required"])
            self.assertFalse(groups["build_and_audit"]["api_required"])
            self.assertTrue(manifest["parallel_domains"])
            self.assertFalse(manifest["parallel_method_groups"])
            self.assertEqual(groups["native_full"]["effective_max_concurrency"], 4)
            self.assertEqual(groups["solver_canonical_full"]["effective_max_concurrency"], 12)
            self.assertEqual(groups["full"]["effective_max_concurrency"], 12)
            self.assertEqual(manifest["max_api_backed_effective_concurrency"], 12)
            self.assertEqual(manifest["candidate_workers"], 3)
            self.assertEqual(manifest["seed"], 2)
            self.assertEqual(manifest["recommended_next_run_group"], "source_controller")
            self.assertIn("source_controller_seed2", groups["source_controller"]["source_controller_checkpoint"])
            self.assertIn("SEED=2", groups["source_controller"]["command"])
            self.assertIn("scripts/run_table10_tau3.sh source-controller", manifest["recommended_next_command"])
            self.assertIn("scripts/run_table10_tau3.sh full", groups["full"]["command"])
            self.assertIn("SEED=2", groups["full"]["command"])
            self.assertIn("CANDIDATE_WORKERS=3", groups["full"]["command"])
            self.assertIn("PARALLEL_METHOD_GROUPS=0", groups["full"]["command"])
            self.assertIn("scripts/run_table10_tau3.sh native-full", groups["native_full"]["command"])
            self.assertIn("PARALLEL_DOMAINS=1", groups["native_full"]["command"])
            self.assertTrue((root / "preflight" / "table10_tau3_run_preflight.csv").exists())
            self.assertTrue((root / "preflight" / "table10_tau3_run_preflight.md").exists())

    def test_table10_preflight_recommends_next_missing_group(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "runs"
            source = base / "source_controller_seed0"
            source.mkdir(parents=True)
            _write_source_manifest(source, tau3_train_trajectories_used=False)
            for row_dir, domain, method in (
                (base / "native_retail", "retail", "Native tool-calling rerun"),
                (base / "native_airline", "airline", "Native tool-calling rerun"),
                (base / "best_of_4_retail", "retail", "Best-of-N"),
                (base / "best_of_4_airline", "airline", "Best-of-N"),
            ):
                row_dir.mkdir(parents=True)
                _write_row(row_dir, domain, method=method, task_count=3, status="complete")
                if method == "Best-of-N":
                    row = json.loads((row_dir / "q4_tau_row.json").read_text(encoding="utf-8"))
                    row["selector"] = "official_reward_best_of_n"
                    row["selector_full_coverage"] = True
                    (row_dir / "q4_tau_row.json").write_text(json.dumps(row), encoding="utf-8")
            manifest = build_table10_tau3_run_preflight(root / "preflight", base_dir=base, expected_test_count=3)
            self.assertEqual(manifest["recommended_next_run_group"], "legacy_solver_full")
            self.assertIn("scripts/run_table10_tau3.sh solver-full", manifest["recommended_next_command"])

    def test_table10_preflight_recommends_native_after_source_controller_ready(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "runs"
            source = base / "source_controller_seed0"
            source.mkdir(parents=True)
            _write_source_manifest(source, tau3_train_trajectories_used=False)
            manifest = build_table10_tau3_run_preflight(root / "preflight", base_dir=base, expected_test_count=3)
            groups = {row["run_group"]: row for row in manifest["rows"]}
            self.assertTrue(groups["source_controller"]["ready"])
            self.assertEqual(manifest["recommended_next_run_group"], "native_full")
            self.assertIn("scripts/run_table10_tau3.sh native-full", manifest["recommended_next_command"])

    def test_table10_preflight_reports_parallel_method_group_concurrency(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = build_table10_tau3_run_preflight(
                root / "preflight",
                base_dir=root / "runs",
                expected_test_count=3,
                max_concurrency=2,
                candidate_workers=3,
                parallel_domains=True,
                parallel_method_groups=True,
            )
            groups = {row["run_group"]: row for row in manifest["rows"]}
            self.assertEqual(groups["full"]["effective_max_concurrency"], 40)
            self.assertEqual(groups["native_full"]["effective_max_concurrency"], 4)
            self.assertEqual(groups["solver_canonical_full"]["effective_max_concurrency"], 12)
            self.assertEqual(groups["build_and_audit"]["effective_max_concurrency"], 0)
            self.assertEqual(manifest["max_api_backed_effective_concurrency"], 40)


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
                "llm_calls": 7.0,
                "tokens": 123.0,
                "wall_clock_seconds": 4.0,
                "status": status,
                "task_count": task_count,
                "official_results": "x",
            }
        ),
        encoding="utf-8",
    )


def _write_source_manifest(path: Path, *, tau3_train_trajectories_used: bool) -> None:
    (path / "generic_harness_ranker.json").write_text(json.dumps({"weights": [0.0], "bias": 0.0}), encoding="utf-8")
    (path / "canonical_harness_source_validation.json").write_text(
        json.dumps(
            {
                "benchmark": "canonical_harness_source_workflow_validation",
                "controller_checkpoint_frozen": True,
                "adapter_changes_checkpoint_stays_frozen": True,
                "tau3_train_trajectories_used": tau3_train_trajectories_used,
                "oracle_labels_in_inference_features": False,
                "learned_score_authorizes_pruning": False,
                "tool_names_used_as_features": False,
                "source_generator": {
                    "tau3_trajectories_used": False,
                    "tau3_heldout_test_seen_before_freeze": False,
                    "source_generator_distribution_fixed_before_target_test_evaluation": True,
                    "anonymized_tool_names": True,
                },
                "tool_identity_generalization": {
                    "ready": True,
                    "top1_absolute_delta": 0.0,
                },
                "contract_coverage_stress": {
                    "ready": True,
                    "black_box_contract_top1_delta_from_full": 0.0,
                },
                "train_count": 3,
                "validation_count": 2,
                "seed": 0,
            }
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    unittest.main()
