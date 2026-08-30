from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cgr_exp.harness_contract import canonical_contract_manifest, canonical_feature_names
from cgr_exp.tau3_native import (
    Tau3RunConfig,
    _run_tau2_solver_controller,
    aggregate_tau3_official_results,
    audit_table10_tau3_readiness,
    audit_table10_tau3_smoke_gate,
    audit_tau3_controller_trace,
    build_tau3_best_of_n_from_native,
    build_tau2_run_command,
    build_table10_tau3_results,
    build_table10_tau3_run_preflight,
    run_tau3_native,
    summarize_table10_tau3_progress,
)
from cgr_exp.tau3_contract_adapter import tau3_adapter_manifest


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
        self.assertIn("ensure_source_controller", text)
        self.assertIn("progress)", text)
        self.assertIn("FORCE_SOURCE_CONTROLLER", text)
        self.assertIn("audit_controller_trace_pair solver_canonical_smoke", text)
        self.assertIn("audit_controller_trace_pair solver_canonical_certified_smoke", text)
        self.assertIn("audit_smoke_gate_pair solver_canonical_smoke", text)
        self.assertIn("audit_smoke_gate_pair solver_canonical_certified_smoke", text)
        self.assertIn("solver-canonical-suite", text)
        self.assertIn("run_canonical_suite()", text)
        self.assertIn('run_domain_pair "${NUM_TASKS}" solver_canonical solver', text)
        self.assertIn('run_domain_pair "${NUM_TASKS}" solver_canonical_certified solver', text)
        self.assertIn("--fail-on-not-ready", text)
        self.assertIn("refresh_status()", text)
        self.assertIn("run_source_controller() {", text)
        self.assertIn("summarize_progress", text)
        self.assertIn("build_run_preflight", text)
        self.assertIn('PYTHON_BIN="/opt/anaconda3/bin/python"', text)
        self.assertIn('TAU_PYTHON_BIN="${TAU_PYTHON_BIN:-${PYTHON_BIN}}"', text)
        result = subprocess.run(["zsh", "-n", str(script)], check=False, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_table10_tau3_canonical_smoke_reuses_existing_source_checkpoint(self) -> None:
        script = Path("scripts/run_table10_tau3.sh")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "tau3" / "source_controller_seed0" / "generic_harness_ranker.json"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_text("{}", encoding="utf-8")
            env = {
                **os.environ,
                "PYTHON_BIN": "/usr/bin/true",
                "TAU_PYTHON_BIN": "/usr/bin/true",
                "BASE_DIR": str(root / "tau3"),
                "NUM_TASKS": "1",
                "OPENAI_API_KEY": "test-key",
            }
            result = subprocess.run(["zsh", str(script), "solver-canonical-smoke"], check=False, text=True, capture_output=True, env=env)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("using existing source controller", result.stdout)

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
        self.assertIn("preflight should recommend", text)
        self.assertIn("scripts/run_table10_tau3.sh solver-canonical-certified-smoke", text)
        self.assertIn("scripts/run_table10_tau3.sh solver-canonical-suite", text)
        self.assertIn("scripts/run_table10_tau3.sh full", text)
        self.assertIn("scripts/run_table10_tau3.sh build", text)
        self.assertIn("source_controller_seed${seed}/generic_harness_ranker.json", text)
        self.assertIn("OPENAI_API_KEY", text)
        self.assertIn("CGR_LLM_API_KEY", text)
        self.assertIn("If `PYTHON_BIN` is not set", text)
        self.assertIn("TAU_PYTHON_BIN` defaults", text)
        self.assertIn("CANDIDATE_WORKERS=4", text)
        self.assertIn("CANDIDATE_RETRIES=2", text)
        self.assertIn("candidate_sample_failures", text)
        self.assertIn("table10_tau3_smoke_gate.json", text)
        self.assertIn("all_smoke_rows_ready=True", text)
        self.assertIn("refresh `progress/` and `run_preflight/`", text)
        self.assertIn("Both source-controller targets refresh", text)
        self.assertIn("FORCE_SOURCE_CONTROLLER=1", text)
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
            per_instance = (out / "q4_tau_per_instance.csv").read_text(encoding="utf-8")
            self.assertIn("task_success_count", per_instance)
            self.assertIn("task_best_reward", per_instance)
            self.assertIn("task_pass^1", per_instance)
            self.assertIn("task_pass^4", per_instance)
            self.assertIn("task_termination_reasons", per_instance)

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
            missing = root / "solver_canonical_certified_airline"
            manifest = build_table10_tau3_results(root / "table", [retail, airline, missing], expected_test_count=100)
            self.assertEqual(len(manifest["rows"]), 3)
            self.assertTrue((root / "table" / "table10_tau3_results.csv").exists())
            csv_text = (root / "table" / "table10_tau3_results.csv").read_text(encoding="utf-8")
            tex_text = (root / "table" / "table10_tau3_results.tex").read_text(encoding="utf-8")
            self.assertIn("LLM calls", csv_text)
            self.assertIn("Wall-clock", csv_text)
            self.assertIn("partial: 2/100", csv_text)
            self.assertIn("Solver-derived canonical controller + certified filter", csv_text)
            self.assertIn("missing: q4_tau_row.json", csv_text)
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
            self.assertEqual(command_manifest["tau3_adapter_protocol"]["adapter"], "Tau3AdapterV1")
            self.assertFalse(command_manifest["tau3_adapter_protocol"]["uses_tau3_train_trajectories"])
            self.assertFalse(command_manifest["tau3_adapter_protocol"]["oracle_visible_to_controller"])
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
            self.assertTrue(result["row"]["reused_existing_results"])
            row_artifact = json.loads((root / "native" / "q4_tau_row.json").read_text(encoding="utf-8"))
            self.assertTrue(row_artifact["reused_existing_results"])
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
            self.assertTrue(command_manifest["resume_safe"])
            self.assertFalse(command_manifest["force_rerun"])
            self.assertTrue(command_manifest["overwrite_is_output_dir_guard_only"])
            self.assertIn("--force is required", command_manifest["resume_note"])
            self.assertEqual(command_manifest["existing_task_count"], 1)
            self.assertEqual(command_manifest["expected_task_count"], 2)

    def test_failed_tau2_run_writes_pending_row_for_progress(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            failed = subprocess.CompletedProcess(args=["tau2"], returncode=7, stdout="out", stderr="bad")
            with (
                patch("cgr_exp.tau3_native.tau3_preflight", return_value={"ready": True, "blockers": []}),
                patch("cgr_exp.tau3_native._tau2_revision", return_value="rev"),
                patch("cgr_exp.tau3_native.subprocess.run", return_value=failed),
            ):
                with self.assertRaisesRegex(RuntimeError, "exit code 7"):
                    run_tau3_native(Tau3RunConfig(output_dir=root / "native", domain="retail", num_tasks=2, num_trials=1))
            row = json.loads((root / "native" / "q4_tau_row.json").read_text(encoding="utf-8"))
            command_manifest = json.loads((root / "native" / "tau3_native_command.json").read_text(encoding="utf-8"))
            self.assertEqual(row["status"], "tau2_failed:7")
            self.assertEqual(row["task_count"], 0)
            self.assertIn("tau3_native_stderr.log", row["notes"])
            self.assertEqual((root / "native" / "tau3_native_stderr.log").read_text(encoding="utf-8"), "bad")
            self.assertEqual(command_manifest["returncode"], 7)

    def test_solver_controller_python_api_injects_canonical_runtime_args(self) -> None:
        class FakeResults:
            def save(self, path: Path, format: str = "json") -> None:
                Path(path).mkdir(parents=True, exist_ok=True)
                (Path(path) / "results.json").write_text(json.dumps({"simulations": []}), encoding="utf-8")

        captured = {}

        def fake_run_domain(run_config):
            captured["run_config"] = run_config
            return FakeResults()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "ranker.json"
            checkpoint.write_text("{}", encoding="utf-8")
            config = Tau3RunConfig(
                output_dir=root / "solver",
                domain="retail",
                method="solver",
                num_tasks=1,
                num_trials=1,
                solver_ranker_checkpoint=checkpoint,
                enable_certified_filter=True,
                candidate_workers=5,
                candidate_retries=3,
                candidate_retry_backoff=0.25,
                max_steps=123,
            )
            with patch("tau2.runner.run_domain", side_effect=fake_run_domain):
                result = _run_tau2_solver_controller(config, root / "official")
        self.assertEqual(result.returncode, 0)
        run_config = captured["run_config"]
        self.assertEqual(run_config.agent, "cgr_solver_controller")
        args = run_config.llm_args_agent
        self.assertEqual(args["cgr_ranker_checkpoint"], str(checkpoint))
        self.assertTrue(args["cgr_enable_certified_filter"])
        self.assertEqual(args["cgr_candidate_workers"], 5)
        self.assertEqual(args["cgr_candidate_retries"], 3)
        self.assertEqual(args["cgr_candidate_retry_backoff"], 0.25)
        self.assertEqual(args["cgr_max_turns"], 123)
        self.assertEqual(args["cgr_max_tool_budget"], 123)
        self.assertTrue(str(args["cgr_controller_trace_path"]).endswith("tau3_controller_trace.jsonl"))

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
            self.assertIn("first_reward", per_instance)
            self.assertIn("best_reward", per_instance)
            self.assertIn("any_success", per_instance)
            self.assertIn("rollout_coverage_ready", per_instance)

    def test_audits_tau3_controller_trace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trace = root / "tau3_controller_trace.jsonl"
            contract = canonical_contract_manifest()
            trace.write_text(
                json.dumps(
                    {
                        "event": "tau3_controller_selection",
                        "scoring_mode": "canonical_ranker",
                        "canonical_contract": contract["contract"],
                        "canonical_feature_schema_hash": contract["feature_schema_hash"],
                        "canonical_feature_count": len(canonical_feature_names()),
                        "ranker_checkpoint_loaded": True,
                        "certified_filter_enabled": True,
                        "certified_pruned_count": 1,
                        "candidate_sample_failures": 1,
                        "oracle_visible_to_controller": False,
                        "oracle_used_for_candidate_generation": False,
                        "oracle_used_for_filtering": False,
                        "tool_names_used_as_features": False,
                        "tool_name_use_policy": "coarse_action_type_normalization_only",
                        "candidate_scores": [
                            {"candidate_index": 0, "score": 0.1, "action_type": "read"},
                            {"candidate_index": 1, "score": 0.2, "action_type": "write"},
                        ],
                    }
                )
                + "\n"
                + json.dumps(
                    {
                        "event": "tau3_controller_selection",
                        "scoring_mode": "greedy_content_passthrough",
                        "canonical_contract": contract["contract"],
                        "canonical_feature_schema_hash": contract["feature_schema_hash"],
                        "canonical_feature_count": len(canonical_feature_names()),
                        "ranker_checkpoint_loaded": True,
                        "candidate_scores": [{"candidate_index": 0, "score": None, "action_type": "respond"}],
                    }
                )
                + "\n"
                + json.dumps(
                    {
                        "event": "tau3_controller_selection",
                        "scoring_mode": "fallback_greedy",
                        "canonical_contract": contract["contract"],
                        "canonical_feature_schema_hash": contract["feature_schema_hash"],
                        "canonical_feature_count": len(canonical_feature_names()),
                        "ranker_checkpoint_loaded": True,
                        "candidate_scores": [{"candidate_index": 0, "score": None, "action_type": "terminate"}],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            manifest = audit_tau3_controller_trace(root / "audit", trace)
            self.assertEqual(manifest["event_count"], 3)
            self.assertEqual(manifest["canonical_ranker_events"], 1)
            self.assertEqual(manifest["greedy_content_passthrough_events"], 1)
            self.assertEqual(manifest["fallback_greedy_events"], 1)
            self.assertEqual(manifest["ranker_checkpoint_loaded_events"], 3)
            self.assertEqual(manifest["certified_filter_enabled_events"], 1)
            self.assertEqual(manifest["certified_pruned_candidates"], 1)
            self.assertEqual(manifest["candidate_sample_failures"], 1)
            self.assertAlmostEqual(manifest["certified_prune_rate"], 0.25)
            self.assertFalse(manifest["oracle_visible_to_controller"])
            self.assertFalse(manifest["tool_names_used_as_features"])
            self.assertEqual(manifest["tool_name_use_policy"], "coarse_action_type_normalization_only")
            self.assertEqual(manifest["canonical_contract"], "CanonicalHarnessContractV1")
            self.assertEqual(manifest["canonical_feature_schema_hash"], contract["feature_schema_hash"])
            self.assertTrue(manifest["canonical_feature_schema_match"])
            self.assertTrue(manifest["canonical_feature_count_match"])
            self.assertTrue((root / "audit" / "tau3_controller_trace_audit.csv").exists())
            csv_text = (root / "audit" / "tau3_controller_trace_audit.csv").read_text(encoding="utf-8")
            self.assertIn("candidate_sample_failures", csv_text.splitlines()[0])
            self.assertIn("greedy_content_passthrough_events", csv_text.splitlines()[0])
            self.assertIn("fallback_greedy_events", csv_text.splitlines()[0])
            self.assertIn(",1,", csv_text)

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
                        "tau3_adapter_protocol": tau3_adapter_manifest(),
                        "source_controller_protocol": _source_protocol_fields(),
                    }
                ),
                encoding="utf-8",
            )
            contract = canonical_contract_manifest()
            (canonical / "tau3_controller_trace.jsonl").write_text(
                json.dumps(
                    {
                        "scoring_mode": "canonical_ranker",
                        "canonical_contract": contract["contract"],
                        "canonical_feature_schema_hash": contract["feature_schema_hash"],
                        "canonical_feature_count": len(canonical_feature_names()),
                        "ranker_checkpoint_loaded": True,
                        "certified_filter_enabled": False,
                        "certified_pruned_count": 0,
                        "candidate_sample_failures": 2,
                        "oracle_visible_to_controller": False,
                        "oracle_used_for_candidate_generation": False,
                        "oracle_used_for_filtering": False,
                        "tool_names_used_as_features": False,
                        "tool_name_use_policy": "coarse_action_type_normalization_only",
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
            self.assertEqual(canonical_row["source_generator_distribution_hash"], "source-hash")
            self.assertTrue(canonical_row["source_checkpoint_loadable"])
            self.assertEqual(canonical_row["source_checkpoint_feature_count"], len(canonical_feature_names()))
            self.assertTrue(canonical_row["source_checkpoint_feature_count_match"])
            self.assertTrue(canonical_row["source_checkpoint_schema_match"])
            self.assertTrue(canonical_row["source_generator_anonymized_tools"])
            self.assertTrue(canonical_row["tool_identity_generalization_ready"])
            self.assertTrue(canonical_row["contract_coverage_stress_ready"])
            self.assertTrue(canonical_row["source_contract_audit_ready"])
            self.assertEqual(canonical_row["source_contract_audit_blockers"], "")
            self.assertEqual(canonical_row["tau3_adapter"], "Tau3AdapterV1")
            self.assertTrue(canonical_row["tau3_adapter_train_trajectories_hidden"])
            self.assertTrue(canonical_row["tau3_adapter_oracle_hidden"])
            self.assertEqual(canonical_row["tool_name_use_policy"], "coarse_action_type_normalization_only")
            self.assertEqual(canonical_row["canonical_ranker_events"], 1)
            self.assertEqual(canonical_row["ranker_checkpoint_loaded_events"], 1)
            self.assertTrue(canonical_row["canonical_feature_schema_match"])
            self.assertTrue(canonical_row["canonical_feature_count_match"])
            self.assertEqual(canonical_row["candidate_sample_failures"], 2)
            self.assertIn("source generator frozen before target test", canonical_row["notes"])
            self.assertIn("source generator hash=source-hash", canonical_row["notes"])
            self.assertIn("tool-identity stress delta=0.0", canonical_row["notes"])
            self.assertIn("candidate sample failures after retries=2", canonical_row["notes"])
            self.assertIn("contract coverage black-box delta=0.0", canonical_row["notes"])
            self.assertIn("tool-name policy=coarse_action_type_normalization_only", canonical_row["notes"])

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

    def test_table10_readiness_rejects_failed_tau2_row(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            row_dir = root / "solver_retail"
            row_dir.mkdir()
            _write_row(row_dir, "retail", method="Solver-derived controller", task_count=0, status="tau2_failed:7")
            readiness = audit_table10_tau3_readiness(root / "audit", [row_dir], expected_test_count=2)
            self.assertFalse(readiness["all_rows_result_ready"])
            self.assertEqual(readiness["pending_or_partial_rows"], ["retail: Solver-derived controller"])
            row = readiness["audit_rows"][0]
            self.assertFalse(row["result_ready"])
            self.assertIn("tau2_failed:7", row["result_blockers"])
            self.assertIn("partial: 0/2", row["result_blockers"])

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

    def test_audits_table10_tau3_readiness_rejects_missing_canonical_schema_trace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            canonical = root / "solver_canonical_retail"
            source = root / "source"
            canonical.mkdir()
            source.mkdir()
            _write_row(canonical, "retail", method="Solver-derived canonical controller", task_count=2, status="complete")
            _write_source_manifest(source, tau3_train_trajectories_used=False)
            (canonical / "tau3_native_command.json").write_text(
                json.dumps(
                    {
                        "solver_ranker_checkpoint": str(source / "generic_harness_ranker.json"),
                        "tau3_adapter_protocol": tau3_adapter_manifest(),
                        "source_controller_protocol": _source_protocol_fields(),
                    }
                ),
                encoding="utf-8",
            )
            (canonical / "tau3_controller_trace.jsonl").write_text(
                json.dumps(
                    {
                        "scoring_mode": "canonical_ranker",
                        "ranker_checkpoint_loaded": True,
                        "candidate_scores": [{"candidate_index": 0, "score": 1.0}],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            readiness = audit_table10_tau3_readiness(root / "audit", [canonical], expected_test_count=2)
            self.assertFalse(readiness["all_rows_result_ready"])
            blockers = readiness["audit_rows"][0]["result_blockers"]
            self.assertIn("canonical feature schema hash missing or mismatched", blockers)
            self.assertIn("canonical feature count missing or mismatched", blockers)

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

    def test_audits_table10_tau3_smoke_gate_for_canonical_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            source.mkdir()
            _write_source_manifest(source, tau3_train_trajectories_used=False)
            contract = canonical_contract_manifest()
            row_dirs = []
            for domain in ("retail", "airline"):
                row_dir = root / f"solver_canonical_smoke_{domain}"
                row_dir.mkdir()
                row_dirs.append(row_dir)
                _write_row(row_dir, domain, method="Solver-derived canonical controller", task_count=2, status="complete")
                (row_dir / "tau3_native_command.json").write_text(
                    json.dumps(
                        {
                            "solver_ranker_checkpoint": str(source / "generic_harness_ranker.json"),
                            "tau3_adapter_protocol": tau3_adapter_manifest(),
                            "source_controller_protocol": _source_protocol_fields(),
                        }
                    ),
                    encoding="utf-8",
                )
                (row_dir / "tau3_controller_trace.jsonl").write_text(
                    json.dumps(
                        {
                            "scoring_mode": "canonical_ranker",
                            "canonical_contract": contract["contract"],
                            "canonical_feature_schema_hash": contract["feature_schema_hash"],
                            "canonical_feature_count": len(canonical_feature_names()),
                            "ranker_checkpoint_loaded": True,
                            "oracle_visible_to_controller": False,
                            "oracle_used_for_candidate_generation": False,
                            "oracle_used_for_filtering": False,
                            "tool_names_used_as_features": False,
                            "tool_name_use_policy": "coarse_action_type_normalization_only",
                            "candidate_scores": [{"candidate_index": 0, "score": 1.0}],
                        }
                    )
                    + "\n",
                    encoding="utf-8",
                )
            manifest = audit_table10_tau3_smoke_gate(root / "gate", row_dirs, expected_smoke_count=2)
            self.assertTrue(manifest["all_smoke_rows_ready"])
            self.assertEqual(manifest["pending_or_failed_rows"], [])
            self.assertIn("greedy_content_passthrough_events", manifest["rows"][0])
            self.assertIn("fallback_greedy_events", manifest["rows"][0])
            self.assertIn("candidate_sample_failures", manifest["rows"][0])
            self.assertIn("tool_name_use_policy", manifest["rows"][0])
            self.assertTrue((root / "gate" / "table10_tau3_smoke_gate.csv").exists())
            self.assertTrue((root / "gate" / "readiness" / "table10_tau3_readiness.csv").exists())
            gate_csv = (root / "gate" / "table10_tau3_smoke_gate.csv").read_text(encoding="utf-8")
            self.assertIn("greedy_content_passthrough_events", gate_csv.splitlines()[0])
            self.assertIn("fallback_greedy_events", gate_csv.splitlines()[0])
            self.assertIn("candidate_sample_failures", gate_csv.splitlines()[0])
            self.assertIn("tool_name_use_policy", gate_csv.splitlines()[0])

    def test_audits_table10_tau3_smoke_gate_rejects_incomplete_trace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            row_dir = root / "solver_canonical_smoke_retail"
            row_dir.mkdir()
            _write_row(row_dir, "retail", method="Solver-derived canonical controller", task_count=2, status="complete")
            manifest = audit_table10_tau3_smoke_gate(root / "gate", [row_dir], expected_smoke_count=2)
            self.assertFalse(manifest["all_smoke_rows_ready"])
            self.assertIn("retail: Solver-derived canonical controller", manifest["pending_or_failed_rows"])
            self.assertIn("controller trace missing", manifest["rows"][0]["result_blockers"])

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
                candidate_retries=5,
                candidate_retry_backoff=0.25,
                python_bin="/tmp/runner-python",
                tau_python_bin="/tmp/python",
            )
            groups = {row["run_group"]: row for row in manifest["rows"]}
            self.assertIn("source_controller", groups)
            self.assertIn("full", groups)
            self.assertIn("solver_canonical_smoke", groups)
            self.assertIn("solver_canonical_certified_smoke", groups)
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
            self.assertEqual(groups["solver_canonical_smoke"]["effective_max_concurrency"], 12)
            self.assertEqual(groups["solver_canonical_full"]["effective_max_concurrency"], 12)
            self.assertEqual(groups["full"]["effective_max_concurrency"], 12)
            self.assertEqual(groups["solver_canonical_smoke"]["estimated_api_rows"], 2)
            self.assertEqual(groups["solver_canonical_smoke"]["estimated_tau2_simulations"], 16)
            self.assertEqual(groups["native_full"]["estimated_api_rows"], 2)
            self.assertEqual(groups["native_full"]["estimated_tau2_simulations"], 24)
            self.assertEqual(groups["solver_canonical_full"]["estimated_api_rows"], 2)
            self.assertEqual(groups["solver_canonical_full"]["estimated_tau2_simulations"], 24)
            self.assertEqual(groups["solver_canonical_full"]["candidate_worker_multiplier"], 3)
            self.assertEqual(groups["full"]["estimated_api_rows"], 8)
            self.assertEqual(groups["full"]["estimated_tau2_simulations"], 96)
            self.assertEqual(manifest["max_api_backed_effective_concurrency"], 12)
            self.assertEqual(manifest["candidate_workers"], 3)
            self.assertEqual(manifest["candidate_retries"], 5)
            self.assertEqual(manifest["candidate_retry_backoff"], 0.25)
            self.assertEqual(manifest["python_bin"], "/tmp/runner-python")
            self.assertEqual(manifest["seed"], 2)
            self.assertEqual(manifest["recommended_next_run_group"], "source_controller")
            self.assertEqual(manifest["recommended_next_summary"]["recommended_next_run_group"], "source_controller")
            self.assertEqual(manifest["recommended_next_summary"]["estimated_tau2_simulations"], 0)
            self.assertIn("source_controller_seed2", groups["source_controller"]["source_controller_checkpoint"])
            self.assertIn("SEED=2", groups["source_controller"]["command"])
            self.assertIn("scripts/run_table10_tau3.sh source-controller", manifest["recommended_next_command"])
            self.assertIn("scripts/run_table10_tau3.sh full", groups["full"]["command"])
            self.assertIn("scripts/run_table10_tau3.sh solver-canonical-smoke", groups["solver_canonical_smoke"]["command"])
            self.assertIn("SEED=2", groups["full"]["command"])
            self.assertIn("PYTHON_BIN=/tmp/runner-python", groups["full"]["command"])
            self.assertIn("CANDIDATE_WORKERS=3", groups["full"]["command"])
            self.assertIn("CANDIDATE_RETRIES=5", groups["full"]["command"])
            self.assertIn("CANDIDATE_RETRY_BACKOFF=0.25", groups["full"]["command"])
            self.assertIn("PARALLEL_METHOD_GROUPS=0", groups["full"]["command"])
            self.assertIn("scripts/run_table10_tau3.sh native-full", groups["native_full"]["command"])
            self.assertIn("PARALLEL_DOMAINS=1", groups["native_full"]["command"])
            self.assertTrue((root / "preflight" / "table10_tau3_run_preflight.csv").exists())
            self.assertTrue((root / "preflight" / "table10_tau3_run_preflight.md").exists())
            self.assertTrue((root / "preflight" / "next_tau3_run.json").exists())
            self.assertTrue((root / "preflight" / "next_tau3_run.md").exists())
            next_text = (root / "preflight" / "next_tau3_run.md").read_text(encoding="utf-8")
            self.assertIn("Run group: `source_controller`", next_text)
            self.assertIn("Estimated tau2 simulations: `0`", next_text)

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
            self.assertEqual(manifest["recommended_next_run_group"], "solver_canonical_smoke")
            groups = {row["run_group"]: row for row in manifest["rows"]}
            native_methods = groups["native_full"]["methods"]
            self.assertIn("retail: Native tool-calling rerun", native_methods)
            self.assertIn("airline: Best-of-N", native_methods)
            self.assertIn("scripts/run_table10_tau3.sh solver-canonical-smoke", manifest["recommended_next_command"])

    def test_table10_preflight_recommends_native_after_source_controller_ready(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "runs"
            source = base / "source_controller_seed0"
            source.mkdir(parents=True)
            _write_source_manifest(source, tau3_train_trajectories_used=False)
            with patch.dict(os.environ, {}, clear=True):
                manifest = build_table10_tau3_run_preflight(root / "preflight", base_dir=base, expected_test_count=3)
            groups = {row["run_group"]: row for row in manifest["rows"]}
            self.assertTrue(groups["source_controller"]["ready"])
            self.assertEqual(manifest["recommended_next_run_group"], "solver_canonical_smoke")
            self.assertEqual(manifest["recommended_next_summary"]["recommended_next_run_group"], "solver_canonical_smoke")
            self.assertEqual(manifest["recommended_next_summary"]["estimated_tau2_simulations"], 16)
            self.assertFalse(manifest["api_key_present"])
            self.assertTrue(manifest["recommended_next_requires_api_key"])
            self.assertFalse(manifest["recommended_next_api_key_ready"])
            self.assertEqual(manifest["accepted_api_key_envs"], ["OPENAI_API_KEY", "CGR_LLM_API_KEY"])
            self.assertIn("scripts/run_table10_tau3.sh solver-canonical-smoke", manifest["recommended_next_command"])
            text = (root / "preflight" / "table10_tau3_run_preflight.md").read_text(encoding="utf-8")
            next_text = (root / "preflight" / "next_tau3_run.md").read_text(encoding="utf-8")
            self.assertIn("API key present: `False`", text)
            self.assertIn("Set one of `OPENAI_API_KEY, CGR_LLM_API_KEY`", next_text)
            self.assertIn("Present API key envs: `none`", text)
            self.assertIn("Warning: recommended next run needs one of `OPENAI_API_KEY, CGR_LLM_API_KEY`", text)
            self.assertIn("Resume policy:", text)
            self.assertIn("`--force` is the opt-in switch", text)

    def test_table10_preflight_rejects_bad_source_checkpoint_dimension(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "runs"
            source = base / "source_controller_seed0"
            source.mkdir(parents=True)
            _write_source_manifest(source, tau3_train_trajectories_used=False)
            (source / "generic_harness_ranker.json").write_text(json.dumps({"weights": [0.0], "bias": 0.0}), encoding="utf-8")
            manifest = build_table10_tau3_run_preflight(root / "preflight", base_dir=base, expected_test_count=3)
            groups = {row["run_group"]: row for row in manifest["rows"]}
            self.assertFalse(groups["source_controller"]["ready"])
            self.assertIn("source controller checkpoint feature dimension mismatch", groups["source_controller"]["blockers"])
            self.assertEqual(manifest["recommended_next_run_group"], "source_controller")

    def test_table10_preflight_rejects_bad_source_contract_audit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "runs"
            source = base / "source_controller_seed0"
            source.mkdir(parents=True)
            _write_source_manifest(source, tau3_train_trajectories_used=False)
            manifest_path = source / "canonical_harness_source_validation.json"
            source_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            source_manifest["contract_audit"] = {"ready": False, "blockers": ["training-only labels appear in runtime features: success"]}
            manifest_path.write_text(json.dumps(source_manifest), encoding="utf-8")
            manifest = build_table10_tau3_run_preflight(root / "preflight", base_dir=base, expected_test_count=3)
            groups = {row["run_group"]: row for row in manifest["rows"]}
            self.assertFalse(groups["source_controller"]["ready"])
            self.assertIn("source contract audit failed", groups["source_controller"]["blockers"])
            self.assertIn("training-only labels appear in runtime features: success", groups["source_controller"]["blockers"])
            self.assertEqual(manifest["recommended_next_run_group"], "source_controller")

    def test_table10_preflight_recommends_certified_smoke_after_canonical_smoke_ready(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "runs"
            source = base / "source_controller_seed0"
            source.mkdir(parents=True)
            _write_source_manifest(source, tau3_train_trajectories_used=False)
            for domain in ("retail", "airline"):
                _write_canonical_solver_row(
                    base / f"solver_canonical_smoke_{domain}",
                    domain,
                    source_dir=source,
                    method="Solver-derived canonical controller",
                )
            manifest = build_table10_tau3_run_preflight(root / "preflight", base_dir=base, expected_test_count=3)
            self.assertEqual(manifest["recommended_next_run_group"], "solver_canonical_certified_smoke")
            self.assertEqual(manifest["recommended_next_summary"]["recommended_next_run_group"], "solver_canonical_certified_smoke")
            self.assertIn("scripts/run_table10_tau3.sh solver-canonical-certified-smoke", manifest["recommended_next_command"])

    def test_table10_preflight_recommends_native_full_after_smokes_ready(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "runs"
            source = base / "source_controller_seed0"
            source.mkdir(parents=True)
            _write_source_manifest(source, tau3_train_trajectories_used=False)
            for suffix, method in (
                ("solver_canonical_smoke", "Solver-derived canonical controller"),
                ("solver_canonical_certified_smoke", "Solver-derived canonical controller + certified filter"),
            ):
                for domain in ("retail", "airline"):
                    _write_canonical_solver_row(base / f"{suffix}_{domain}", domain, source_dir=source, method=method)
            manifest = build_table10_tau3_run_preflight(root / "preflight", base_dir=base, expected_test_count=3)
            self.assertEqual(manifest["recommended_next_run_group"], "native_full")
            self.assertEqual(manifest["recommended_next_summary"]["recommended_next_run_group"], "native_full")
            self.assertIn("scripts/run_table10_tau3.sh native-full", manifest["recommended_next_command"])

    def test_table10_preflight_accepts_cgr_llm_api_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "runs"
            source = base / "source_controller_seed0"
            source.mkdir(parents=True)
            _write_source_manifest(source, tau3_train_trajectories_used=False)
            with patch.dict(os.environ, {"CGR_LLM_API_KEY": "test-key"}, clear=True):
                manifest = build_table10_tau3_run_preflight(root / "preflight", base_dir=base, expected_test_count=3)
            self.assertTrue(manifest["api_key_present"])
            self.assertEqual(manifest["present_api_key_envs"], ["CGR_LLM_API_KEY"])
            self.assertTrue(manifest["recommended_next_api_key_ready"])
            text = (root / "preflight" / "table10_tau3_run_preflight.md").read_text(encoding="utf-8")
            self.assertIn("Present API key envs: `CGR_LLM_API_KEY`", text)
            self.assertNotIn("test-key", text)

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
            self.assertEqual(groups["solver_canonical_smoke"]["effective_max_concurrency"], 12)
            self.assertEqual(groups["solver_canonical_full"]["effective_max_concurrency"], 12)
            self.assertEqual(groups["build_and_audit"]["effective_max_concurrency"], 0)
            self.assertEqual(manifest["max_api_backed_effective_concurrency"], 40)

    def test_table10_preflight_caps_effective_concurrency_by_work(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = build_table10_tau3_run_preflight(
                root / "preflight",
                base_dir=root / "runs",
                expected_test_count=1,
                num_trials=1,
                max_concurrency=20,
                candidate_workers=10,
                parallel_domains=True,
                parallel_method_groups=False,
            )
            groups = {row["run_group"]: row for row in manifest["rows"]}
            self.assertEqual(groups["solver_canonical_smoke"]["raw_effective_max_concurrency"], 400)
            self.assertEqual(groups["solver_canonical_smoke"]["estimated_tau2_simulations"], 4)
            self.assertEqual(groups["solver_canonical_smoke"]["effective_max_concurrency"], 4)
            self.assertEqual(groups["native_full"]["raw_effective_max_concurrency"], 40)
            self.assertEqual(groups["native_full"]["estimated_tau2_simulations"], 2)
            self.assertEqual(groups["native_full"]["effective_max_concurrency"], 2)
            self.assertEqual(groups["full"]["raw_effective_max_concurrency"], 400)
            self.assertEqual(groups["full"]["estimated_tau2_simulations"], 8)
            self.assertEqual(groups["full"]["effective_max_concurrency"], 8)
            text = (root / "preflight" / "table10_tau3_run_preflight.md").read_text(encoding="utf-8")
            self.assertIn("Raw effective max concurrency: `400`", text)
            self.assertIn("Effective max concurrency: `4`", text)

    def test_summarizes_table10_tau3_progress_from_partial_results(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "runs"
            official = base / "native_retail" / "official_tau2_results"
            official.mkdir(parents=True)
            (official / "results.json").write_text(
                json.dumps(
                    {
                        "info": {"environment_info": {"domain_name": "retail"}},
                        "simulations": [
                            _sim("0", 0, 1.0),
                            _sim("0", 1, 0.0),
                            _sim("1", 0, 1.0),
                        ],
                    }
                ),
                encoding="utf-8",
            )
            manifest = summarize_table10_tau3_progress(root / "progress", base_dir=base, expected_test_count=4, best_of_n=4)
            rows = {(row["Domain"], row["Method"]): row for row in manifest["rows"]}
            native = rows[("retail", "Native tool-calling rerun")]
            self.assertEqual(native["status"], "partial_unaggregated")
            self.assertEqual(native["runtime_kind"], "api_native")
            self.assertTrue(native["api_required"])
            self.assertFalse(native["derived_from_native"])
            self.assertEqual(native["expected_api_simulation_count"], 16)
            self.assertEqual(native["task_count"], 2)
            self.assertEqual(native["simulation_count"], 3)
            self.assertAlmostEqual(native["task_progress"], 0.5)
            self.assertAlmostEqual(native["simulation_progress"], 3 / 16)
            best = rows[("retail", "Best-of-N")]
            self.assertEqual(best["runtime_kind"], "derived_from_native")
            self.assertFalse(best["api_required"])
            self.assertTrue(best["derived_from_native"])
            self.assertEqual(best["expected_api_simulation_count"], 0)
            self.assertTrue((root / "progress" / "table10_tau3_progress.csv").exists())
            text = (root / "progress" / "table10_tau3_progress.md").read_text(encoding="utf-8")
            self.assertIn("Table10 Tau3 Progress", text)
            self.assertIn("2/4", text)
            self.assertIn("derived_from_native", text)
            self.assertIn("LLM calls", text)
            self.assertIn("Tool calls", text)
            self.assertIn("Tokens", text)
            self.assertIn("Wall-clock", text)
            self.assertIn("21", text)
            self.assertIn("2.00", text)
            self.assertIn("1.00", text)
            self.assertIn("1.50", text)

    def test_summarizes_pending_tau3_artifact_without_results_as_rerunnable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "runs"
            solver = base / "solver_retail"
            solver.mkdir(parents=True)
            _write_row(solver, "retail", method="Solver-derived controller", task_count=0, status="preflight_failed: missing key")
            manifest = summarize_table10_tau3_progress(root / "progress", base_dir=base, expected_test_count=4, best_of_n=4)
            rows = {(row["Domain"], row["Method"]): row for row in manifest["rows"]}
            solver_row = rows[("retail", "Solver-derived controller")]
            self.assertEqual(solver_row["runtime_kind"], "api_solver_controller")
            self.assertTrue(solver_row["api_required"])
            self.assertFalse(solver_row["derived_from_native"])
            self.assertEqual(solver_row["expected_api_simulation_count"], 16)
            self.assertTrue(solver_row["q4_row_present"])
            self.assertFalse(solver_row["official_results_present"])
            self.assertTrue(solver_row["pending_without_official_results"])
            self.assertTrue(solver_row["rerun_safe"])
            self.assertIn("rerun same row", solver_row["note"])
            text = (root / "progress" / "table10_tau3_progress.md").read_text(encoding="utf-8")
            self.assertIn("pending artifact only", text)

    def test_progress_surfaces_failed_tau3_row_notes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "runs"
            solver = base / "solver_retail"
            solver.mkdir(parents=True)
            _write_row(solver, "retail", method="Solver-derived controller", task_count=0, status="tau2_failed:7")
            row = json.loads((solver / "q4_tau_row.json").read_text(encoding="utf-8"))
            row["notes"] = "tau2 official run failed; see tau3_native_stderr.log"
            (solver / "q4_tau_row.json").write_text(json.dumps(row), encoding="utf-8")
            manifest = summarize_table10_tau3_progress(root / "progress", base_dir=base, expected_test_count=4, best_of_n=4)
            rows = {(row["Domain"], row["Method"]): row for row in manifest["rows"]}
            solver_row = rows[("retail", "Solver-derived controller")]
            self.assertEqual(solver_row["status"], "tau2_failed:7")
            self.assertIn("tau3_native_stderr.log", solver_row["note"])
            text = (root / "progress" / "table10_tau3_progress.md").read_text(encoding="utf-8")
            self.assertIn("tau3_native_stderr.log", text)


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
    (path / "generic_harness_ranker.json").write_text(
        json.dumps({"weights": [0.0] * len(canonical_feature_names()), "bias": 0.0, "feature_names": canonical_feature_names()}),
        encoding="utf-8",
    )
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
                "contract": canonical_contract_manifest(),
                "contract_audit": {"ready": True, "blockers": []},
                "source_generator": {
                    "distribution_hash": "source-hash",
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


def _source_protocol_fields() -> dict:
    return {
        "manifest_found": True,
        "checkpoint_found": True,
        "checkpoint_loadable": True,
        "checkpoint_feature_count": len(canonical_feature_names()),
        "checkpoint_feature_count_match": True,
        "checkpoint_schema_match": True,
        "controller_checkpoint_frozen": True,
        "tau3_train_trajectories_used": False,
        "oracle_labels_in_inference_features": False,
        "tool_names_used_as_features": False,
        "source_generator_distribution_fixed_before_target_test_evaluation": True,
        "source_generator_distribution_hash": "source-hash",
        "tau3_heldout_test_seen_before_freeze": False,
        "source_generator_uses_tau3_trajectories": False,
        "source_generator_anonymized_tool_names": True,
        "tool_identity_generalization_ready": True,
        "tool_identity_generalization_top1_delta": 0.0,
        "contract_coverage_stress_ready": True,
        "contract_coverage_black_box_top1_delta_from_full": 0.0,
        "contract_audit_ready": True,
        "contract_audit_blockers": "",
    }


def _write_canonical_solver_row(
    row_dir: Path,
    domain: str,
    *,
    source_dir: Path,
    method: str,
    task_count: int = 2,
) -> None:
    row_dir.mkdir(parents=True, exist_ok=True)
    _write_row(row_dir, domain, method=method, task_count=task_count, status="complete")
    (row_dir / "tau3_native_command.json").write_text(
        json.dumps(
            {
                "solver_ranker_checkpoint": str(source_dir / "generic_harness_ranker.json"),
                "tau3_adapter_protocol": tau3_adapter_manifest(),
                "source_controller_protocol": _source_protocol_fields(),
            }
        ),
        encoding="utf-8",
    )
    contract = canonical_contract_manifest()
    (row_dir / "tau3_controller_trace.jsonl").write_text(
        json.dumps(
            {
                "scoring_mode": "canonical_ranker",
                "canonical_contract": contract["contract"],
                "canonical_feature_schema_hash": contract["feature_schema_hash"],
                "canonical_feature_count": len(canonical_feature_names()),
                "ranker_checkpoint_loaded": True,
                "oracle_visible_to_controller": False,
                "oracle_used_for_candidate_generation": False,
                "oracle_used_for_filtering": False,
                "tool_names_used_as_features": False,
                "certified_filter_enabled": "certified filter" in method,
                "candidate_scores": [{"candidate_index": 0, "score": 1.0}],
            }
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    unittest.main()
