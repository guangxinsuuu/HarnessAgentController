from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cgr_exp.q2_method_contracts import q2_method_contracts
from cgr_exp.q2_source_evidence import q2_source_evidence_for_methods
from cgr_exp.q2_source_evidence import Q2_SOURCE_EVIDENCE
from cgr_exp.table5_q2 import TABLE5_METHODS, audit_table5_q2_readiness, build_table5_q2_results


class Table5Q2Test(unittest.TestCase):
    def test_readiness_blocks_rows_when_local_primitive_audit_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            local_primitives = root / "q2_local_primitive_audit.csv"
            _write_csv(
                local_primitives,
                [
                    {
                        "Method": method,
                        "local_primitives_ready": "False" if method == "Direct / CoT" else "True",
                        "missing_paths": "",
                        "missing_anchors": "first_valid" if method == "Direct / CoT" else "",
                        "missing_primitive_mappings": "single reasoning path" if method == "Direct / CoT" else "",
                        "missing_primitive_anchors": "final answer extraction: generate_scheduling_solution" if method == "Direct / CoT" else "",
                    }
                    for method, _setting, _source, _kind in TABLE5_METHODS
                ],
            )
            manifest = audit_table5_q2_readiness(
                root / "out",
                solver_prior_dir=root / "missing",
                q2_local_primitives=local_primitives,
            )
            audit_rows = {row["Method"]: row for row in manifest["readiness"]["audit_rows"]}
            self.assertIn("local primitive audit failed", audit_rows["Direct / CoT"]["result_blockers"])
            self.assertIn("missing anchors: first_valid", audit_rows["Direct / CoT"]["result_blockers"])
            self.assertIn("missing primitive mappings: single reasoning path", audit_rows["Direct / CoT"]["result_blockers"])
            self.assertIn("missing primitive anchors: final answer extraction", audit_rows["Direct / CoT"]["result_blockers"])
            self.assertNotIn("local primitive audit failed", audit_rows["Self-Consistency"]["result_blockers"])

    def test_builder_keeps_missing_native_rows_pending_and_extracts_completed_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            solver = root / "solver"
            solver.mkdir()
            _write_csv(
                solver / "scheduling_llm_pilot_table.csv",
                [
                    {
                        "Policy": "Solver-trained MLP",
                        "success": "0.89",
                        "gap": "0.019335",
                        "nodes_mean": "80",
                        "proposal_calls_mean": "80",
                        "value_calls": "0",
                        "proposal_tokens": "20196491",
                        "value_tokens": "0",
                        "total_tokens": "20196491",
                        "on_demand_total_tokens": "5000",
                    }
                ],
            )
            _write_csv(
                solver / "scheduling_llm_pilot_per_instance.csv",
                [{"Policy": "Solver-trained MLP", "task_id": f"t{i}"} for i in range(100)],
            )
            _write_csv(solver / "timing_metrics.csv", [{"component": "proposal_cache", "wall_clock_seconds": "3600", "call_count": "1"}])
            (solver / "scheduling_llm_pilot_ranker.json").write_text("{}", encoding="utf-8")
            _write_learned_metrics(solver, ["Solver-trained priority"], ["Solver-trained MLP"])
            _write_solver_audit_artifacts(solver, train_ids=["train-0"], validation_ids=["validation-0"], test_ids=[f"t{i}" for i in range(100)])

            gpt = root / "gpt"
            gpt.mkdir()
            _write_csv(
                gpt / "scheduling_llm_pilot_table.csv",
                [
                    {
                        "Policy": "GPT-frontier-rank / ToT-style",
                        "success": "0.61",
                        "gap": "0.08",
                        "nodes_mean": "80",
                        "proposal_calls_mean": "80",
                        "value_calls": "8000",
                        "proposal_tokens": "20196491",
                        "value_tokens": "4000000",
                        "total_tokens": "24196491",
                    }
                ],
            )
            _write_csv(
                gpt / "scheduling_llm_pilot_per_instance.csv",
                [{"Policy": "GPT-frontier-rank / ToT-style", "task_id": f"t{i}"} for i in range(100)],
            )

            out = root / "out"
            manifest = build_table5_q2_results(out, solver_prior_dir=solver, gpt_frontier_rank_dir=gpt)
            rows = _read_csv(out / "table5_q2_results.csv")
            by_method = {row["Method"]: row for row in rows}
            self.assertEqual(by_method["Solver-trained priority"]["Status"], "complete")
            self.assertEqual(by_method["Solver-trained priority"]["Success"], "0.8900")
            self.assertEqual(by_method["Solver-trained priority"]["LLM calls"], "8000")
            self.assertEqual(by_method["Solver-trained priority"]["Tokens"], "5.0K")
            self.assertEqual(by_method["Solver-trained priority"]["Wall-clock"], "1.00h")
            self.assertEqual(by_method["Solver-trained priority"]["Claim scope"], "ours")
            self.assertEqual(by_method["Tree of Thoughts"]["Claim scope"], "native_adaptation")
            self.assertEqual(by_method["Best-of-N"]["Required wording"], "Best-of-N + verifier")
            self.assertEqual(by_method["Best-of-N"]["Source URL"], "https://arxiv.org/abs/2408.03314")
            self.assertEqual(by_method["Tree of Thoughts"]["Source URL"], "https://github.com/princeton-nlp/tree-of-thought-llm")
            self.assertIn("on-demand per-policy accounting", by_method["Solver-trained priority"]["notes"])
            self.assertEqual(by_method["Tree of Thoughts"]["Status"], "pending")
            self.assertIn("native runner pending", by_method["Tree of Thoughts"]["notes"])
            self.assertEqual(by_method["Direct / CoT"]["Status"], "pending")
            tex = (out / "table5_q2_results.tex").read_text(encoding="utf-8")
            self.assertIn("Task-specific optimization references", tex)
            self.assertIn("Frozen and online LLM inference", tex)
            self.assertIn("Target-domain learned search", tex)
            self.assertIn("Ours -- solver-to-LLM zero-shot transfer", tex)
            self.assertIn("Best-of-$N$ is verifier-selected", tex)
            self.assertIn("full iterative self-training pipeline", tex)
            self.assertEqual(manifest["strict_pending_policy"], True)
            self.assertIn("sections", manifest)
            saved = json.loads((out / "table5_q2_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["table"], "Table 4")
            self.assertEqual(saved["legacy_artifact_prefix"], "table5_q2")
            self.assertEqual(
                saved["method_alignment"]["Best-of-N"]["source_url"],
                "https://arxiv.org/abs/2408.03314",
            )
            self.assertIn(
                "method_generate",
                saved["method_alignment"]["Tree of Thoughts"]["source_primitives"],
            )
            self.assertEqual(
                saved["source_evidence"]["Tree of Thoughts"]["source_files"],
                ["src/tot/methods/bfs.py"],
            )
            self.assertTrue(saved["source_ref_verification"]["all_static_refs_ready"])
            self.assertFalse(saved["source_ref_verification"]["online_checked"])
            self.assertIn("Tree of Thoughts", saved["source_ref_verification"]["external_methods"])
            self.assertIn(
                "declared sample_count coverage per held-out task",
                saved["source_evidence"]["Best-of-N"]["local_promotion_gates"],
            )

    def test_builder_marks_incomplete_checkpoints_as_partial(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            solver = root / "solver"
            solver.mkdir()
            _write_csv(
                solver / "scheduling_llm_pilot_table.csv",
                [
                    {
                        "Policy": "Solver-trained MLP",
                        "success": "1.0",
                        "gap": "0.0",
                        "nodes_mean": "80",
                        "proposal_calls_mean": "80",
                        "value_calls": "0",
                        "proposal_tokens": "1000",
                        "value_tokens": "0",
                        "total_tokens": "1000",
                    }
                ],
            )
            _write_csv(
                solver / "scheduling_llm_pilot_per_instance.csv",
                [{"Policy": "Solver-trained MLP", "task_id": f"t{i}"} for i in range(4)],
            )
            out = root / "out"
            build_table5_q2_results(out, solver_prior_dir=solver, expected_test_count=100)
            by_method = {row["Method"]: row for row in _read_csv(out / "table5_q2_results.csv")}
            self.assertEqual(by_method["Solver-trained priority"]["Status"], "partial: 4/100")
            self.assertIn("partial checkpoint", by_method["Solver-trained priority"]["notes"])

    def test_builder_reads_native_row_json_when_available(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            native = root / "direct"
            native.mkdir()
            (native / "table5_native_row.json").write_text(
                json.dumps(
                    {
                        "success": 0.5,
                        "gap": 0.1,
                        "llm_calls": 100,
                        "tokens": 1200000,
                        "wall_clock_seconds": 61,
                        "status": "complete",
                    }
                ),
                encoding="utf-8",
            )
            solver = root / "missing_solver"
            out = root / "out"
            build_table5_q2_results(out, solver_prior_dir=solver, direct_dir=native)
            by_method = {row["Method"]: row for row in _read_csv(out / "table5_q2_results.csv")}
            self.assertEqual(by_method["Direct / CoT"]["Status"], "complete")
            self.assertEqual(by_method["Direct / CoT"]["Tokens"], "1.20M")

    def test_builder_rejects_native_row_json_with_wrong_method(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            native = root / "wrong"
            native.mkdir()
            (native / "table5_native_row.json").write_text(
                json.dumps(
                    {
                        "Method": "Self-Consistency",
                        "success": 0.8,
                        "gap": 0.1,
                        "llm_calls": 800,
                        "tokens": 1200000,
                        "wall_clock_seconds": 61,
                        "status": "complete",
                    }
                ),
                encoding="utf-8",
            )
            out = root / "out"
            build_table5_q2_results(out, solver_prior_dir=root / "missing_solver", direct_dir=native)
            by_method = {row["Method"]: row for row in _read_csv(out / "table5_q2_results.csv")}
            self.assertEqual(by_method["Direct / CoT"]["Status"], "pending")

    def test_builder_does_not_use_native_artifact_for_learned_search_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake_prm = root / "fake_prm"
            fake_prm.mkdir()
            (fake_prm / "table5_native_row.json").write_text(
                json.dumps(
                    {
                        "Method": "Target-PRM priority",
                        "success": 0.99,
                        "gap": 0.0,
                        "llm_calls": 1,
                        "tokens": 1,
                        "wall_clock_seconds": 1,
                        "status": "complete",
                        "source_policy": "native",
                    }
                ),
                encoding="utf-8",
            )
            out = root / "out"
            build_table5_q2_results(out, solver_prior_dir=root / "missing", target_prm_dir=fake_prm)
            by_method = {row["Method"]: row for row in _read_csv(out / "table5_q2_results.csv")}
            self.assertEqual(by_method["Target-PRM priority"]["Status"], "pending")
            self.assertIn("native runner pending", by_method["Target-PRM priority"]["notes"])

    def test_builder_reads_native_method_from_aggregate_table(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            native = root / "native"
            native.mkdir()
            _write_csv(
                native / "scheduling_native_q2_table.csv",
                [
                    {
                        "Method": "Direct / CoT",
                        "success": "0.4",
                        "gap": "0.2",
                        "llm_calls": "100",
                        "tokens": "1000",
                        "wall_clock_seconds": "10",
                        "status": "complete",
                        "source_policy": "native",
                        "notes": "direct",
                    },
                    {
                        "Method": "Best-of-N",
                        "success": "0.7",
                        "gap": "0.05",
                        "llm_calls": "800",
                        "tokens": "9000",
                        "wall_clock_seconds": "12",
                        "status": "complete",
                        "source_policy": "native",
                        "notes": "best",
                    },
                ],
            )
            out = root / "out"
            build_table5_q2_results(out, solver_prior_dir=root / "missing", direct_dir=native, best_of_n_dir=native)
            by_method = {row["Method"]: row for row in _read_csv(out / "table5_q2_results.csv")}
            self.assertEqual(by_method["Direct / CoT"]["Success"], "0.4000")
            self.assertEqual(by_method["Best-of-N"]["Success"], "0.7000")
            self.assertEqual(by_method["Best-of-N"]["LLM calls"], "800")

    def test_builder_reads_native_method_subdir_after_incremental_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            native = root / "native"
            direct_dir = native / "direct_cot"
            direct_dir.mkdir(parents=True)
            (direct_dir / "table5_native_row.json").write_text(
                json.dumps(
                    {
                        "Method": "Direct / CoT",
                        "success": 0.45,
                        "gap": 0.2,
                        "llm_calls": 100,
                        "tokens": 1000,
                        "wall_clock_seconds": 10,
                        "status": "complete",
                        "source_policy": "native",
                        "native_runtime_sample_count": 1,
                        "native_runtime_selection_events": 1,
                        "native_runtime_valid_samples": 1,
                        "native_runtime_parse_failures": 0,
                    }
                ),
                encoding="utf-8",
            )
            _write_csv(
                native / "scheduling_native_q2_table.csv",
                [
                    {
                        "Method": "Self-Consistency",
                        "success": "0.6",
                        "gap": "0.15",
                        "llm_calls": "800",
                        "tokens": "9000",
                        "wall_clock_seconds": "20",
                        "status": "complete",
                        "source_policy": "native",
                    }
                ],
            )
            out = root / "out"
            build_table5_q2_results(out, solver_prior_dir=root / "missing", direct_dir=native, self_consistency_dir=native)
            by_method = {row["Method"]: row for row in _read_csv(out / "table5_q2_results.csv")}
            self.assertEqual(by_method["Direct / CoT"]["Status"], "complete")
            self.assertEqual(by_method["Direct / CoT"]["Success"], "0.4500")
            self.assertIn("direct_cot/table5_native_row.json", by_method["Direct / CoT"]["source_artifact"])
            self.assertEqual(by_method["Self-Consistency"]["Success"], "0.6000")

    def test_readiness_audit_checks_native_samples_at_incremental_run_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            native = root / "native"
            direct_dir = native / "direct_cot"
            direct_dir.mkdir(parents=True)
            (direct_dir / "table5_native_row.json").write_text(
                json.dumps(
                    {
                        "Method": "Direct / CoT",
                        "success": 0.45,
                        "gap": 0.2,
                        "llm_calls": 100,
                        "tokens": 1000,
                        "wall_clock_seconds": 10,
                        "status": "complete",
                        "source_policy": "native",
                        "native_runtime_sample_count": 1,
                        "native_runtime_selection_events": 1,
                        "native_runtime_valid_samples": 1,
                        "native_runtime_parse_failures": 0,
                        "native_runtime_reasoning_samples": 1,
                    }
                ),
                encoding="utf-8",
            )
            (native / "scheduling_native_q2_samples.jsonl").write_text(
                json.dumps({"Method": "Direct / CoT", "task_id": "t0", "reasoning_present": True, "reasoning_step_count": 1}) + "\n",
                encoding="utf-8",
            )
            _write_native_manifest(native, ["Direct / CoT"], test_count=1)
            out = root / "out"
            audit_table5_q2_readiness(out, solver_prior_dir=root / "missing", direct_dir=native, expected_test_count=1)
            audit_rows = {row["Method"]: row for row in _read_csv(out / "table5_q2_readiness.csv")}
            self.assertEqual(audit_rows["Direct / CoT"]["result_ready"], "True")
            self.assertNotIn("scheduling_native_q2_samples.jsonl", audit_rows["Direct / CoT"]["result_blockers"])

    def test_readiness_audit_rejects_native_without_runtime_counts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            native = root / "native"
            native.mkdir()
            (native / "table5_native_row.json").write_text(
                json.dumps(
                    {
                        "Method": "Direct / CoT",
                        "success": 0.45,
                        "gap": 0.2,
                        "llm_calls": 1,
                        "tokens": 1000,
                        "wall_clock_seconds": 10,
                        "status": "complete",
                        "source_policy": "native",
                    }
                ),
                encoding="utf-8",
            )
            (native / "scheduling_native_q2_samples.jsonl").write_text(
                json.dumps({"Method": "Direct / CoT", "task_id": "t0"}) + "\n",
                encoding="utf-8",
            )
            _write_native_manifest(native, ["Direct / CoT"], test_count=1)
            out = root / "out"
            audit_table5_q2_readiness(out, solver_prior_dir=root / "missing", direct_dir=native, expected_test_count=1)
            audit_rows = {row["Method"]: row for row in _read_csv(out / "table5_q2_readiness.csv")}
            self.assertEqual(audit_rows["Direct / CoT"]["result_ready"], "False")
            self.assertIn("must record native runtime samples", audit_rows["Direct / CoT"]["result_blockers"])
            self.assertIn("must record one native selection event per task", audit_rows["Direct / CoT"]["result_blockers"])

    def test_readiness_audit_rejects_native_samples_with_partial_task_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            native = root / "native"
            native.mkdir()
            (native / "table5_native_row.json").write_text(
                json.dumps(
                    {
                        "Method": "Direct / CoT",
                        "success": 0.45,
                        "gap": 0.2,
                        "llm_calls": 100,
                        "tokens": 1000,
                        "wall_clock_seconds": 10,
                        "status": "complete",
                        "source_policy": "native",
                    }
                ),
                encoding="utf-8",
            )
            (native / "scheduling_native_q2_samples.jsonl").write_text(
                "".join(json.dumps({"Method": "Direct / CoT", "task_id": f"t{i}"}) + "\n" for i in range(2)),
                encoding="utf-8",
            )
            out = root / "out"
            audit_table5_q2_readiness(out, solver_prior_dir=root / "missing", direct_dir=native, expected_test_count=3)
            audit_rows = {row["Method"]: row for row in _read_csv(out / "table5_q2_readiness.csv")}
            self.assertEqual(audit_rows["Direct / CoT"]["result_ready"], "False")
            self.assertIn("partial task coverage: 2/3", audit_rows["Direct / CoT"]["result_blockers"])

    def test_readiness_audit_rejects_self_consistency_single_sample_protocol(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            native = root / "native"
            native.mkdir()
            (native / "table5_native_row.json").write_text(
                json.dumps(
                    {
                        "Method": "Self-Consistency",
                        "success": 0.45,
                        "gap": 0.2,
                        "llm_calls": 1,
                        "tokens": 1000,
                        "wall_clock_seconds": 10,
                        "status": "complete",
                        "source_policy": "native",
                    }
                ),
                encoding="utf-8",
            )
            (native / "scheduling_native_q2_samples.jsonl").write_text(
                json.dumps({"Method": "Self-Consistency", "task_id": "t0", "sample_index": 0}) + "\n",
                encoding="utf-8",
            )
            (native / "scheduling_native_q2_manifest.json").write_text(
                json.dumps(
                    {
                        "splits": {"test": 1},
                        "native_protocol": {
                            "methods": {
                                "Self-Consistency": {
                                    "sample_count": 1,
                                    "selection_rule": "first_valid",
                                    "oracle_future_labels_exposed": False,
                                    "verifier_role": "validate complete schedules for metrics",
                                }
                            }
                        },
                        "method_alignment": q2_method_contracts(["Self-Consistency"]),
                    }
                ),
                encoding="utf-8",
            )
            out = root / "out"
            audit_table5_q2_readiness(out, solver_prior_dir=root / "missing", self_consistency_dir=native, expected_test_count=1)
            audit_rows = {row["Method"]: row for row in _read_csv(out / "table5_q2_readiness.csv")}
            self.assertEqual(audit_rows["Self-Consistency"]["result_ready"], "False")
            self.assertIn("must declare multiple independent samples", audit_rows["Self-Consistency"]["result_blockers"])
            self.assertIn("selection_rule=majority_job_order_then_first_valid", audit_rows["Self-Consistency"]["result_blockers"])

    def test_readiness_audit_rejects_best_of_n_below_declared_sample_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            native = root / "native"
            native.mkdir()
            (native / "table5_native_row.json").write_text(
                json.dumps(
                    {
                        "Method": "Best-of-N",
                        "success": 0.45,
                        "gap": 0.2,
                        "llm_calls": 1,
                        "tokens": 1000,
                        "wall_clock_seconds": 10,
                        "status": "complete",
                        "source_policy": "native",
                    }
                ),
                encoding="utf-8",
            )
            (native / "scheduling_native_q2_samples.jsonl").write_text(
                json.dumps({"Method": "Best-of-N", "task_id": "t0", "sample_index": 0}) + "\n",
                encoding="utf-8",
            )
            (native / "scheduling_native_q2_manifest.json").write_text(
                json.dumps(
                    {
                        "splits": {"test": 1},
                        "native_protocol": {
                            "methods": {
                                "Best-of-N": {
                                    "sample_count": 4,
                                    "selection_rule": "best_verified_objective",
                                    "oracle_future_labels_exposed": False,
                                    "verifier_role": "filter invalid complete schedules and select minimum verified objective",
                                }
                            }
                        },
                        "method_alignment": q2_method_contracts(["Best-of-N"]),
                    }
                ),
                encoding="utf-8",
            )
            out = root / "out"
            audit_table5_q2_readiness(out, solver_prior_dir=root / "missing", best_of_n_dir=native, expected_test_count=1)
            audit_rows = {row["Method"]: row for row in _read_csv(out / "table5_q2_readiness.csv")}
            self.assertEqual(audit_rows["Best-of-N"]["result_ready"], "False")
            self.assertIn("sample coverage per task below declared sample_count: 1/4", audit_rows["Best-of-N"]["result_blockers"])

    def test_readiness_audit_rejects_native_manifest_without_method_protocol(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            native = root / "native"
            native.mkdir()
            (native / "table5_native_row.json").write_text(
                json.dumps(
                    {
                        "Method": "Best-of-N",
                        "success": 0.45,
                        "gap": 0.2,
                        "llm_calls": 100,
                        "tokens": 1000,
                        "wall_clock_seconds": 10,
                        "status": "complete",
                        "source_policy": "native",
                    }
                ),
                encoding="utf-8",
            )
            (native / "scheduling_native_q2_samples.jsonl").write_text(
                json.dumps({"Method": "Best-of-N", "task_id": "t0"}) + "\n",
                encoding="utf-8",
            )
            (native / "scheduling_native_q2_manifest.json").write_text(
                json.dumps({"splits": {"test": 1}, "native_protocol": {"methods": {}}}),
                encoding="utf-8",
            )
            out = root / "out"
            audit_table5_q2_readiness(out, solver_prior_dir=root / "missing", best_of_n_dir=native, expected_test_count=1)
            audit_rows = {row["Method"]: row for row in _read_csv(out / "table5_q2_readiness.csv")}
            self.assertEqual(audit_rows["Best-of-N"]["result_ready"], "False")
            self.assertIn("missing native protocol for Best-of-N", audit_rows["Best-of-N"]["result_blockers"])

    def test_readiness_audit_rejects_native_samples_without_matching_method(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            native = root / "native"
            native.mkdir()
            (native / "table5_native_row.json").write_text(
                json.dumps(
                    {
                        "Method": "Direct / CoT",
                        "success": 0.45,
                        "gap": 0.2,
                        "llm_calls": 100,
                        "tokens": 1000,
                        "wall_clock_seconds": 10,
                        "status": "complete",
                    }
                ),
                encoding="utf-8",
            )
            (native / "scheduling_native_q2_samples.jsonl").write_text(json.dumps({"Method": "Self-Consistency"}) + "\n", encoding="utf-8")
            out = root / "out"
            audit_table5_q2_readiness(out, solver_prior_dir=root / "missing", direct_dir=native)
            audit_rows = {row["Method"]: row for row in _read_csv(out / "table5_q2_readiness.csv")}
            self.assertEqual(audit_rows["Direct / CoT"]["result_ready"], "False")
            self.assertIn("missing method rows", audit_rows["Direct / CoT"]["result_blockers"])

    def test_readiness_audit_reports_malformed_jsonl_without_crashing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            native = root / "native"
            native.mkdir()
            (native / "table5_native_row.json").write_text(
                json.dumps(
                    {
                        "Method": "Direct / CoT",
                        "success": 0.45,
                        "gap": 0.2,
                        "llm_calls": 100,
                        "tokens": 1000,
                        "wall_clock_seconds": 10,
                        "status": "complete",
                    }
                ),
                encoding="utf-8",
            )
            (native / "scheduling_native_q2_samples.jsonl").write_text("{bad json\n", encoding="utf-8")
            out = root / "out"
            audit_table5_q2_readiness(out, solver_prior_dir=root / "missing", direct_dir=native)
            audit_rows = {row["Method"]: row for row in _read_csv(out / "table5_q2_readiness.csv")}
            self.assertEqual(audit_rows["Direct / CoT"]["result_ready"], "False")
            self.assertIn("malformed jsonl", audit_rows["Direct / CoT"]["result_blockers"])

    def test_readiness_audit_reports_malformed_native_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            native = root / "native"
            native.mkdir()
            (native / "table5_native_row.json").write_text(
                json.dumps(
                    {
                        "Method": "Direct / CoT",
                        "success": 0.45,
                        "gap": 0.2,
                        "llm_calls": 100,
                        "tokens": 1000,
                        "wall_clock_seconds": 10,
                        "status": "complete",
                    }
                ),
                encoding="utf-8",
            )
            (native / "scheduling_native_q2_samples.jsonl").write_text(
                json.dumps({"Method": "Direct / CoT", "task_id": "t0"}) + "\n",
                encoding="utf-8",
            )
            (native / "scheduling_native_q2_manifest.json").write_text("{bad json\n", encoding="utf-8")
            out = root / "out"
            audit_table5_q2_readiness(out, solver_prior_dir=root / "missing", direct_dir=native, expected_test_count=1)
            audit_rows = {row["Method"]: row for row in _read_csv(out / "table5_q2_readiness.csv")}
            self.assertEqual(audit_rows["Direct / CoT"]["result_ready"], "False")
            self.assertIn("scheduling_native_q2_manifest.json malformed json", audit_rows["Direct / CoT"]["result_blockers"])

    def test_builder_reads_structured_reference_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            structured = root / "structured"
            structured.mkdir()
            _write_csv(
                structured / "scheduling_structured_q2_table.csv",
                [
                    {
                        "Method": "CP-SAT",
                        "Transfer setting": "Structured-input solver",
                        "success": "",
                        "gap": "",
                        "llm_calls": "0",
                        "tokens": "0",
                        "wall_clock_seconds": "0.01",
                        "status": "partial: 0/100",
                        "notes": "dependency missing",
                    },
                    {
                        "Method": "EST / list scheduling heuristic",
                        "Transfer setting": "Structured-input heuristic",
                        "success": "0.25",
                        "gap": "0.4",
                        "llm_calls": "0",
                        "tokens": "0",
                        "wall_clock_seconds": "0.02",
                        "status": "complete",
                        "notes": "heuristic",
                    },
                ],
            )
            out = root / "out"
            build_table5_q2_results(out, solver_prior_dir=root / "missing", structured_dir=structured)
            by_method = {row["Method"]: row for row in _read_csv(out / "table5_q2_results.csv")}
            self.assertEqual(by_method["CP-SAT"]["Success"], "--")
            self.assertEqual(by_method["CP-SAT"]["Tokens"], "0")
            self.assertEqual(by_method["EST / list scheduling heuristic"]["Success"], "0.2500")
            self.assertEqual(by_method["EST / list scheduling heuristic"]["Transfer setting"], "Structured-input heuristic")

    def test_readiness_audit_requires_structured_metrics_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            structured = root / "structured"
            structured.mkdir()
            _write_csv(
                structured / "scheduling_structured_q2_table.csv",
                [
                    {
                        "Method": "CP-SAT",
                        "success": "1.0",
                        "gap": "0.0",
                        "wall_clock_seconds": "0.01",
                        "status": "complete",
                    }
                ],
            )
            out = root / "out"
            audit_table5_q2_readiness(out, solver_prior_dir=root / "missing", structured_dir=structured)
            audit_rows = {row["Method"]: row for row in _read_csv(out / "table5_q2_readiness.csv")}
            self.assertEqual(audit_rows["CP-SAT"]["result_ready"], "False")
            self.assertIn("scheduling_structured_q2_metrics.json", audit_rows["CP-SAT"]["result_blockers"])

    def test_readiness_audit_rejects_structured_metrics_with_llm_calls(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            structured = root / "structured"
            structured.mkdir()
            _write_csv(
                structured / "scheduling_structured_q2_table.csv",
                [
                    {
                        "Method": "CP-SAT",
                        "success": "1.0",
                        "gap": "0.0",
                        "wall_clock_seconds": "0.01",
                        "status": "complete",
                    }
                ],
            )
            (structured / "scheduling_structured_q2_metrics.json").write_text(
                json.dumps(
                    {
                        "structured_input": True,
                        "llm_calls": 1,
                        "tokens": 0,
                        "method_alignment": {"CP-SAT": {}},
                    }
                ),
                encoding="utf-8",
            )
            out = root / "out"
            audit_table5_q2_readiness(out, solver_prior_dir=root / "missing", structured_dir=structured)
            audit_rows = {row["Method"]: row for row in _read_csv(out / "table5_q2_readiness.csv")}
            self.assertEqual(audit_rows["CP-SAT"]["result_ready"], "False")
            self.assertIn("structured reference must have llm_calls = 0", audit_rows["CP-SAT"]["result_blockers"])

    def test_readiness_audit_rejects_cpsat_without_runtime_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            structured = root / "structured"
            structured.mkdir()
            _write_csv(
                structured / "scheduling_structured_q2_table.csv",
                [
                    {
                        "Method": "CP-SAT",
                        "success": "1.0",
                        "gap": "0.0",
                        "llm_calls": "0",
                        "tokens": "0",
                        "wall_clock_seconds": "0.01",
                        "status": "complete",
                    }
                ],
            )
            _write_structured_metrics_without_table_patch(structured, ["CP-SAT"], test_count=1)
            out = root / "out"
            audit_table5_q2_readiness(out, solver_prior_dir=root / "missing", structured_dir=structured, expected_test_count=1)
            audit_rows = {row["Method"]: row for row in _read_csv(out / "table5_q2_readiness.csv")}
            self.assertEqual(audit_rows["CP-SAT"]["result_ready"], "False")
            self.assertIn("structured_runtime_solver=ortools_cp_sat", audit_rows["CP-SAT"]["result_blockers"])

    def test_readiness_audit_rejects_est_without_dispatch_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            structured = root / "structured"
            structured.mkdir()
            _write_csv(
                structured / "scheduling_structured_q2_table.csv",
                [
                    {
                        "Method": "EST / list scheduling heuristic",
                        "success": "0.5",
                        "gap": "0.2",
                        "llm_calls": "0",
                        "tokens": "0",
                        "wall_clock_seconds": "0.01",
                        "status": "complete",
                    }
                ],
            )
            _write_structured_metrics_without_table_patch(structured, ["EST / list scheduling heuristic"], test_count=1)
            out = root / "out"
            audit_table5_q2_readiness(out, solver_prior_dir=root / "missing", structured_dir=structured, expected_test_count=1)
            audit_rows = {row["Method"]: row for row in _read_csv(out / "table5_q2_readiness.csv")}
            self.assertEqual(audit_rows["EST / list scheduling heuristic"]["result_ready"], "False")
            self.assertIn("structured_runtime_heuristic=earliest_start_list_scheduling", audit_rows["EST / list scheduling heuristic"]["result_blockers"])

    def test_readiness_audit_rejects_mismatched_heldout_task_sets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            structured = root / "structured"
            structured.mkdir()
            _write_csv(
                structured / "scheduling_structured_q2_table.csv",
                [
                    {
                        "Method": "CP-SAT",
                        "success": "1.0",
                        "gap": "0.0",
                        "wall_clock_seconds": "0.01",
                        "status": "complete",
                    }
                ],
            )
            _write_csv(
                structured / "scheduling_structured_q2_per_instance.csv",
                [
                    {"Method": "CP-SAT", "task_id": "structured-0"},
                    {"Method": "CP-SAT", "task_id": "structured-1"},
                ],
            )
            _write_structured_metrics(structured, ["CP-SAT"], test_count=2)

            native = root / "native"
            native.mkdir()
            (native / "table5_native_row.json").write_text(
                json.dumps(
                    {
                        "Method": "Direct / CoT",
                        "success": 0.5,
                        "gap": 0.1,
                        "llm_calls": 2,
                        "tokens": 200,
                        "wall_clock_seconds": 2,
                        "status": "complete",
                        "source_policy": "native",
                    }
                ),
                encoding="utf-8",
            )
            _write_csv(
                native / "scheduling_native_q2_per_instance.csv",
                [
                    {"Method": "Direct / CoT", "task_id": "native-0"},
                    {"Method": "Direct / CoT", "task_id": "native-1"},
                ],
            )
            (native / "scheduling_native_q2_samples.jsonl").write_text(
                "".join(json.dumps({"Method": "Direct / CoT", "task_id": f"native-{idx}"}) + "\n" for idx in range(2)),
                encoding="utf-8",
            )
            _write_native_manifest(native, ["Direct / CoT"], test_count=2)

            out = root / "out"
            audit_table5_q2_readiness(out, solver_prior_dir=root / "missing", structured_dir=structured, direct_dir=native, expected_test_count=2)
            audit_rows = {row["Method"]: row for row in _read_csv(out / "table5_q2_readiness.csv")}
            self.assertEqual(audit_rows["CP-SAT"]["result_ready"], "False")
            self.assertEqual(audit_rows["Direct / CoT"]["result_ready"], "False")
            self.assertIn("split mismatch", audit_rows["CP-SAT"]["result_blockers"])
            self.assertIn("split mismatch", audit_rows["Direct / CoT"]["result_blockers"])

    def test_builder_reads_native_tot_row_when_shared_artifact_absent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tot = root / "tot"
            tot.mkdir()
            (tot / "table5_native_row.json").write_text(
                json.dumps(
                    {
                        "Method": "Tree of Thoughts",
                        "success": 0.6,
                        "gap": 0.07,
                        "llm_calls": 1200,
                        "tokens": 23000000,
                        "wall_clock_seconds": 300,
                        "status": "complete",
                        "source_policy": "native_tot_bfs",
                        "tot_generate_events": 1,
                        "tot_evaluate_events": 1,
                        "tot_select_events": 1,
                        "tot_value_sample_count": 1,
                    }
                ),
                encoding="utf-8",
            )
            out = root / "out"
            build_table5_q2_results(out, solver_prior_dir=root / "missing", gpt_frontier_rank_dir=tot)
            by_method = {row["Method"]: row for row in _read_csv(out / "table5_q2_results.csv")}
            self.assertEqual(by_method["Tree of Thoughts"]["Status"], "complete")
            self.assertEqual(by_method["Tree of Thoughts"]["Tokens"], "23.00M")
            self.assertEqual(by_method["Tree of Thoughts"]["source_policy"], "native_tot_bfs")

    def test_builder_rejects_controller_diagnostic_as_native_tot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tot = root / "tot"
            tot.mkdir()
            (tot / "table5_native_row.json").write_text(
                json.dumps(
                    {
                        "Method": "Tree of Thoughts",
                        "success": 0.6,
                        "gap": 0.07,
                        "llm_calls": 1200,
                        "tokens": 23000000,
                        "wall_clock_seconds": 300,
                        "status": "complete",
                        "source_policy": "GPT-frontier-rank / ToT-style",
                    }
                ),
                encoding="utf-8",
            )
            out = root / "out"
            build_table5_q2_results(out, solver_prior_dir=root / "missing", gpt_frontier_rank_dir=tot)
            by_method = {row["Method"]: row for row in _read_csv(out / "table5_q2_results.csv")}
            self.assertEqual(by_method["Tree of Thoughts"]["Status"], "pending")

    def test_readiness_audit_rejects_tot_trace_without_finish_event(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tot = root / "tot"
            tot.mkdir()
            (tot / "table5_native_row.json").write_text(
                json.dumps(
                    {
                        "Method": "Tree of Thoughts",
                        "success": 0.6,
                        "gap": 0.07,
                        "llm_calls": 1200,
                        "tokens": 23000000,
                        "wall_clock_seconds": 300,
                        "status": "complete",
                        "source_policy": "native_tot_bfs",
                        "tot_generate_events": 1,
                        "tot_evaluate_events": 1,
                        "tot_select_events": 1,
                        "tot_value_sample_count": 1,
                    }
                ),
                encoding="utf-8",
            )
            (tot / "scheduling_tot_q2_trace.jsonl").write_text(json.dumps({"event": "generate"}) + "\n", encoding="utf-8")
            out = root / "out"
            audit_table5_q2_readiness(out, solver_prior_dir=root / "missing", gpt_frontier_rank_dir=tot)
            audit_rows = {row["Method"]: row for row in _read_csv(out / "table5_q2_readiness.csv")}
            self.assertEqual(audit_rows["Tree of Thoughts"]["result_ready"], "False")
            self.assertIn("missing finish event", audit_rows["Tree of Thoughts"]["result_blockers"])

    def test_readiness_audit_rejects_tot_trace_with_partial_finish_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tot = root / "tot"
            tot.mkdir()
            (tot / "table5_native_row.json").write_text(
                json.dumps(
                    {
                        "Method": "Tree of Thoughts",
                        "success": 0.6,
                        "gap": 0.07,
                        "llm_calls": 1200,
                        "tokens": 23000000,
                        "wall_clock_seconds": 300,
                        "status": "complete",
                        "source_policy": "native_tot_bfs",
                        "tot_generate_events": 1,
                        "tot_evaluate_events": 1,
                        "tot_select_events": 1,
                        "tot_value_sample_count": 1,
                    }
                ),
                encoding="utf-8",
            )
            (tot / "scheduling_tot_q2_trace.jsonl").write_text(
                "".join(json.dumps({"event": "finish", "rendering_id": f"r{i}"}) + "\n" for i in range(2)),
                encoding="utf-8",
            )
            out = root / "out"
            audit_table5_q2_readiness(out, solver_prior_dir=root / "missing", gpt_frontier_rank_dir=tot, expected_test_count=3)
            audit_rows = {row["Method"]: row for row in _read_csv(out / "table5_q2_readiness.csv")}
            self.assertEqual(audit_rows["Tree of Thoughts"]["result_ready"], "False")
            self.assertIn("partial finish coverage: 2/3", audit_rows["Tree of Thoughts"]["result_blockers"])

    def test_readiness_audit_rejects_tot_trace_without_generate_evaluate_select_events(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tot = root / "tot"
            tot.mkdir()
            (tot / "table5_native_row.json").write_text(
                json.dumps(
                    {
                        "Method": "Tree of Thoughts",
                        "success": 0.6,
                        "gap": 0.07,
                        "llm_calls": 1200,
                        "tokens": 23000000,
                        "wall_clock_seconds": 300,
                        "status": "complete",
                        "source_policy": "native_tot_bfs",
                    }
                ),
                encoding="utf-8",
            )
            (tot / "scheduling_tot_q2_trace.jsonl").write_text(
                json.dumps({"event": "finish", "rendering_id": "r0"}) + "\n",
                encoding="utf-8",
            )
            _write_tot_manifest(tot, test_count=1)
            out = root / "out"
            audit_table5_q2_readiness(out, solver_prior_dir=root / "missing", gpt_frontier_rank_dir=tot, expected_test_count=1)
            audit_rows = {row["Method"]: row for row in _read_csv(out / "table5_q2_readiness.csv")}
            self.assertEqual(audit_rows["Tree of Thoughts"]["result_ready"], "False")
            self.assertIn("missing generate events", audit_rows["Tree of Thoughts"]["result_blockers"])
            self.assertIn("missing evaluate events", audit_rows["Tree of Thoughts"]["result_blockers"])
            self.assertIn("missing select events", audit_rows["Tree of Thoughts"]["result_blockers"])

    def test_readiness_audit_accepts_tot_trace_with_generate_evaluate_select_events(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tot = root / "tot"
            tot.mkdir()
            (tot / "table5_native_row.json").write_text(
                json.dumps(
                    {
                        "Method": "Tree of Thoughts",
                        "success": 0.6,
                        "gap": 0.07,
                        "llm_calls": 1200,
                        "tokens": 23000000,
                        "wall_clock_seconds": 300,
                        "status": "complete",
                        "source_policy": "native_tot_bfs",
                        "tot_generate_events": 1,
                        "tot_evaluate_events": 1,
                        "tot_select_events": 1,
                        "tot_value_sample_count": 1,
                    }
                ),
                encoding="utf-8",
            )
            (tot / "scheduling_tot_q2_trace.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps({"event": "generate", "rendering_id": "r0"}),
                        json.dumps(
                            {
                                "event": "evaluate_select",
                                "rendering_id": "r0",
                                "candidate_ids": ["c0"],
                                "selected_ids": ["c0"],
                                "averaged_scores": [1.0],
                                "value_samples": [{"sample_index": 0, "scores": [1.0]}],
                            }
                        ),
                        json.dumps({"event": "finish", "rendering_id": "r0"}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            _write_tot_manifest(tot, test_count=1)
            out = root / "out"
            audit_table5_q2_readiness(out, solver_prior_dir=root / "missing", gpt_frontier_rank_dir=tot, expected_test_count=1)
            audit_rows = {row["Method"]: row for row in _read_csv(out / "table5_q2_readiness.csv")}
            self.assertEqual(audit_rows["Tree of Thoughts"]["result_ready"], "True")

    def test_readiness_audit_rejects_tot_value_scores_not_matched_to_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tot = root / "tot"
            tot.mkdir()
            (tot / "table5_native_row.json").write_text(
                json.dumps(
                    {
                        "Method": "Tree of Thoughts",
                        "success": 0.6,
                        "gap": 0.07,
                        "llm_calls": 1200,
                        "tokens": 23000000,
                        "wall_clock_seconds": 300,
                        "status": "complete",
                        "source_policy": "native_tot_bfs",
                        "tot_generate_events": 1,
                        "tot_evaluate_events": 1,
                        "tot_select_events": 1,
                        "tot_value_sample_count": 1,
                    }
                ),
                encoding="utf-8",
            )
            (tot / "scheduling_tot_q2_trace.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps({"event": "generate", "rendering_id": "r0"}),
                        json.dumps(
                            {
                                "event": "evaluate_select",
                                "rendering_id": "r0",
                                "candidate_ids": ["c0", "c1"],
                                "selected_ids": ["c0"],
                                "averaged_scores": [1.0],
                                "value_samples": [{"sample_index": 0, "scores": [1.0]}],
                            }
                        ),
                        json.dumps({"event": "finish", "rendering_id": "r0"}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            _write_tot_manifest(tot, test_count=1)
            out = root / "out"
            audit_table5_q2_readiness(out, solver_prior_dir=root / "missing", gpt_frontier_rank_dir=tot, expected_test_count=1)
            audit_rows = {row["Method"]: row for row in _read_csv(out / "table5_q2_readiness.csv")}
            self.assertEqual(audit_rows["Tree of Thoughts"]["result_ready"], "False")
            self.assertIn("averaged_scores must match candidate_ids", audit_rows["Tree of Thoughts"]["result_blockers"])
            self.assertIn("value sample scores must match candidate_ids", audit_rows["Tree of Thoughts"]["result_blockers"])

    def test_readiness_audit_rejects_tot_without_runtime_protocol_counts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tot = root / "tot"
            tot.mkdir()
            (tot / "table5_native_row.json").write_text(
                json.dumps(
                    {
                        "Method": "Tree of Thoughts",
                        "success": 0.6,
                        "gap": 0.07,
                        "llm_calls": 1200,
                        "tokens": 23000000,
                        "wall_clock_seconds": 300,
                        "status": "complete",
                        "source_policy": "native_tot_bfs",
                    }
                ),
                encoding="utf-8",
            )
            (tot / "scheduling_tot_q2_trace.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps({"event": "generate", "rendering_id": "r0"}),
                        json.dumps(
                            {
                                "event": "evaluate_select",
                                "rendering_id": "r0",
                                "candidate_ids": ["c0"],
                                "selected_ids": ["c0"],
                                "averaged_scores": [1.0],
                                "value_samples": [{"sample_index": 0, "scores": [1.0]}],
                            }
                        ),
                        json.dumps({"event": "finish", "rendering_id": "r0"}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            _write_tot_manifest(tot, test_count=1)
            out = root / "out"
            audit_table5_q2_readiness(out, solver_prior_dir=root / "missing", gpt_frontier_rank_dir=tot, expected_test_count=1)
            audit_rows = {row["Method"]: row for row in _read_csv(out / "table5_q2_readiness.csv")}
            self.assertEqual(audit_rows["Tree of Thoughts"]["result_ready"], "False")
            self.assertIn("must record positive generate events", audit_rows["Tree of Thoughts"]["result_blockers"])
            self.assertIn("must record positive value sample count", audit_rows["Tree of Thoughts"]["result_blockers"])

    def test_readiness_audit_rejects_tot_manifest_without_official_entrypoints(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tot = root / "tot"
            tot.mkdir()
            (tot / "table5_native_row.json").write_text(
                json.dumps(
                    {
                        "Method": "Tree of Thoughts",
                        "success": 0.6,
                        "gap": 0.07,
                        "llm_calls": 1200,
                        "tokens": 23000000,
                        "wall_clock_seconds": 300,
                        "status": "complete",
                        "source_policy": "native_tot_bfs",
                    }
                ),
                encoding="utf-8",
            )
            (tot / "scheduling_tot_q2_trace.jsonl").write_text(
                json.dumps({"event": "finish", "rendering_id": "r0"}) + "\n",
                encoding="utf-8",
            )
            (tot / "scheduling_tot_q2_manifest.json").write_text(
                json.dumps({"splits": {"test": 1}, "native_tot_protocol": {"official_entrypoints": ["method_generate"]}}),
                encoding="utf-8",
            )
            out = root / "out"
            audit_table5_q2_readiness(out, solver_prior_dir=root / "missing", gpt_frontier_rank_dir=tot, expected_test_count=1)
            audit_rows = {row["Method"]: row for row in _read_csv(out / "table5_q2_readiness.csv")}
            self.assertEqual(audit_rows["Tree of Thoughts"]["result_ready"], "False")
            self.assertIn("missing official BFS entrypoints", audit_rows["Tree of Thoughts"]["result_blockers"])

    def test_readiness_audit_rejects_tot_manifest_without_greedy_selection_details(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tot = root / "tot"
            tot.mkdir()
            (tot / "table5_native_row.json").write_text(
                json.dumps(
                    {
                        "Method": "Tree of Thoughts",
                        "success": 0.6,
                        "gap": 0.07,
                        "llm_calls": 1200,
                        "tokens": 23000000,
                        "wall_clock_seconds": 300,
                        "status": "complete",
                        "source_policy": "native_tot_bfs",
                        "tot_generate_events": 1,
                        "tot_evaluate_events": 1,
                        "tot_select_events": 1,
                        "tot_value_sample_count": 1,
                    }
                ),
                encoding="utf-8",
            )
            (tot / "scheduling_tot_q2_trace.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps({"event": "generate", "rendering_id": "r0"}),
                        json.dumps(
                            {
                                "event": "evaluate_select",
                                "rendering_id": "r0",
                                "candidate_ids": ["c0"],
                                "selected_ids": ["c0"],
                                "averaged_scores": [1.0],
                                "value_samples": [{"sample_index": 0, "scores": [1.0]}],
                            }
                        ),
                        json.dumps({"event": "finish", "rendering_id": "r0"}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            (tot / "scheduling_tot_q2_manifest.json").write_text(
                json.dumps(
                    {
                        "splits": {"test": 1},
                        "native_tot_protocol": {
                            "official_entrypoints": ["method_generate", "method_evaluate", "method_select"],
                            "oracle_future_labels_exposed": False,
                            "search": "breadth-first frontier search",
                            "method_select": "sample one candidate",
                        },
                        "method_alignment": q2_method_contracts(["Tree of Thoughts"]),
                    }
                ),
                encoding="utf-8",
            )
            out = root / "out"
            audit_table5_q2_readiness(out, solver_prior_dir=root / "missing", gpt_frontier_rank_dir=tot, expected_test_count=1)
            audit_rows = {row["Method"]: row for row in _read_csv(out / "table5_q2_readiness.csv")}
            self.assertEqual(audit_rows["Tree of Thoughts"]["result_ready"], "False")
            self.assertIn("method_select must declare greedy top-b selection", audit_rows["Tree of Thoughts"]["result_blockers"])
            self.assertIn("positive n_select_sample", audit_rows["Tree of Thoughts"]["result_blockers"])
            self.assertIn("positive n_evaluate_sample", audit_rows["Tree of Thoughts"]["result_blockers"])

    def test_readiness_audit_reports_malformed_tot_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tot = root / "tot"
            tot.mkdir()
            (tot / "table5_native_row.json").write_text(
                json.dumps(
                    {
                        "Method": "Tree of Thoughts",
                        "success": 0.6,
                        "gap": 0.07,
                        "llm_calls": 1200,
                        "tokens": 23000000,
                        "wall_clock_seconds": 300,
                        "status": "complete",
                        "source_policy": "native_tot_bfs",
                    }
                ),
                encoding="utf-8",
            )
            (tot / "scheduling_tot_q2_trace.jsonl").write_text(
                json.dumps({"event": "finish", "rendering_id": "r0"}) + "\n",
                encoding="utf-8",
            )
            (tot / "scheduling_tot_q2_manifest.json").write_text("{bad json\n", encoding="utf-8")
            out = root / "out"
            audit_table5_q2_readiness(out, solver_prior_dir=root / "missing", gpt_frontier_rank_dir=tot, expected_test_count=1)
            audit_rows = {row["Method"]: row for row in _read_csv(out / "table5_q2_readiness.csv")}
            self.assertEqual(audit_rows["Tree of Thoughts"]["result_ready"], "False")
            self.assertIn("scheduling_tot_q2_manifest.json malformed json", audit_rows["Tree of Thoughts"]["result_blockers"])

    def test_readiness_audit_flags_pending_and_shared_cache_fallback_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            solver = root / "solver"
            solver.mkdir()
            _write_csv(
                solver / "scheduling_llm_pilot_table.csv",
                [
                    {
                        "Policy": "Solver-trained MLP",
                        "success": "0.89",
                        "gap": "0.02",
                        "nodes_mean": "80",
                        "proposal_calls_mean": "80",
                        "value_calls": "0",
                        "proposal_tokens": "1000",
                        "value_tokens": "0",
                        "total_tokens": "1000",
                    }
                ],
            )
            _write_csv(
                solver / "scheduling_llm_pilot_per_instance.csv",
                [{"Policy": "Solver-trained MLP", "task_id": f"t{i}"} for i in range(100)],
            )
            (solver / "scheduling_llm_pilot_ranker.json").write_text("{}", encoding="utf-8")
            _write_learned_metrics(solver, ["Solver-trained priority"], ["Solver-trained MLP"])
            _write_solver_audit_artifacts(solver, train_ids=["train-0"], validation_ids=["validation-0"], test_ids=[f"t{i}" for i in range(100)])
            out = root / "out"
            manifest = audit_table5_q2_readiness(out, solver_prior_dir=solver, expected_test_count=100)
            readiness = manifest["readiness"]
            self.assertFalse(readiness["all_rows_result_ready"])
            self.assertFalse(readiness["all_rows_compute_ready"])
            self.assertIn("Direct / CoT", readiness["pending_or_partial_rows"])
            self.assertIn("Solver-trained priority", readiness["compute_qualified_rows"])
            audit_rows = {row["Method"]: row for row in _read_csv(out / "table5_q2_readiness.csv")}
            self.assertEqual(audit_rows["Solver-trained priority"]["result_ready"], "True")
            self.assertEqual(audit_rows["Solver-trained priority"]["compute_ready"], "False")
            self.assertEqual(audit_rows["Solver-trained priority"]["Claim scope"], "ours")
            self.assertEqual(audit_rows["Solver-trained priority"]["Reference"], "CGR solver-to-LLM zero-shot controller")
            self.assertIn("zero-shot transfer", audit_rows["Solver-trained priority"]["Required wording"])
            self.assertIn("shared-cache fallback accounting", audit_rows["Solver-trained priority"]["compute_blockers"])
            self.assertTrue((out / "table5_q2_readiness.json").exists())
            readiness = json.loads((out / "table5_q2_readiness.json").read_text(encoding="utf-8"))
            self.assertIn("source_evidence", readiness)
            self.assertIn("Target-PRM priority", readiness["source_evidence"])
            self.assertTrue(readiness["source_ref_verification"]["all_static_refs_ready"])

    def test_readiness_audit_requires_native_trace_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            native = root / "native"
            native.mkdir()
            (native / "table5_native_row.json").write_text(
                json.dumps(
                    {
                        "Method": "Direct / CoT",
                        "success": 0.5,
                        "gap": 0.1,
                        "llm_calls": 100,
                        "tokens": 2000,
                        "wall_clock_seconds": 12,
                        "status": "complete",
                    }
                ),
                encoding="utf-8",
            )
            out = root / "out"
            audit_table5_q2_readiness(out, solver_prior_dir=root / "missing", direct_dir=native)
            audit_rows = {row["Method"]: row for row in _read_csv(out / "table5_q2_readiness.csv")}
            self.assertEqual(audit_rows["Direct / CoT"]["result_ready"], "False")
            self.assertIn("scheduling_native_q2_samples.jsonl", audit_rows["Direct / CoT"]["result_blockers"])

    def test_readiness_audit_rejects_complete_external_row_without_source_code_refs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            native = root / "native"
            native.mkdir()
            _write_csv(
                native / "scheduling_native_q2_table.csv",
                [
                    {
                        "Method": "Direct / CoT",
                        "success": "1.0",
                        "gap": "0.0",
                        "llm_calls": "2",
                        "tokens": "20",
                        "wall_clock_seconds": "1",
                        "status": "complete",
                        "source_policy": "native",
                        "native_runtime_sample_count": "2",
                        "native_runtime_selection_events": "2",
                    }
                ],
            )
            _write_csv(
                native / "scheduling_native_q2_per_instance.csv",
                [{"Method": "Direct / CoT", "task_id": f"t{i}"} for i in range(2)],
            )
            (native / "scheduling_native_q2_samples.jsonl").write_text(
                "".join(json.dumps({"Method": "Direct / CoT", "task_id": f"t{i}", "sample_index": 0}) + "\n" for i in range(2)),
                encoding="utf-8",
            )
            _write_native_manifest(native, ["Direct / CoT"], test_count=2)
            weakened_evidence = {method: dict(evidence) for method, evidence in Q2_SOURCE_EVIDENCE.items()}
            weakened_evidence["Direct / CoT"]["official_code_refs"] = []
            out = root / "out"
            with patch("cgr_exp.table5_q2.Q2_SOURCE_EVIDENCE", weakened_evidence):
                audit_table5_q2_readiness(out, solver_prior_dir=root / "missing", expected_test_count=2, direct_dir=native)
            audit_rows = {row["Method"]: row for row in _read_csv(out / "table5_q2_readiness.csv")}
            self.assertEqual(audit_rows["Direct / CoT"]["result_ready"], "False")
            self.assertIn("source evidence: missing official_code_refs", audit_rows["Direct / CoT"]["result_blockers"])

    def test_readiness_audit_rejects_paper_external_row_without_arxiv_version(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            native = root / "native"
            native.mkdir()
            _write_csv(
                native / "scheduling_native_q2_table.csv",
                [
                    {
                        "Method": "Direct / CoT",
                        "success": "1.0",
                        "gap": "0.0",
                        "llm_calls": "1",
                        "tokens": "10",
                        "wall_clock_seconds": "1",
                        "status": "complete",
                        "source_policy": "native",
                        "native_runtime_sample_count": "1",
                        "native_runtime_selection_events": "1",
                    }
                ],
            )
            _write_csv(native / "scheduling_native_q2_per_instance.csv", [{"Method": "Direct / CoT", "task_id": "t0"}])
            (native / "scheduling_native_q2_samples.jsonl").write_text(
                json.dumps({"Method": "Direct / CoT", "task_id": "t0", "sample_index": 0}) + "\n",
                encoding="utf-8",
            )
            _write_native_manifest(native, ["Direct / CoT"], test_count=1)
            weakened_evidence = {method: dict(evidence) for method, evidence in Q2_SOURCE_EVIDENCE.items()}
            weakened_evidence["Direct / CoT"]["source_revision"] = "arXiv:2201.11903"
            weakened_evidence["Direct / CoT"]["source_revision_url"] = "https://arxiv.org/abs/2201.11903"
            weakened_evidence["Direct / CoT"]["official_code_refs"] = [
                "paper@arXiv:2201.11903: one sampled reasoning chain before final answer"
            ]
            out = root / "out"
            with patch("cgr_exp.table5_q2.Q2_SOURCE_EVIDENCE", weakened_evidence):
                audit_table5_q2_readiness(out, solver_prior_dir=root / "missing", expected_test_count=1, direct_dir=native)
            audit_rows = {row["Method"]: row for row in _read_csv(out / "table5_q2_readiness.csv")}
            self.assertEqual(audit_rows["Direct / CoT"]["result_ready"], "False")
            self.assertIn("source evidence: arxiv source_revision must include explicit version", audit_rows["Direct / CoT"]["result_blockers"])

    def test_readiness_audit_rejects_external_artifact_without_embedded_source_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            native = root / "native"
            native.mkdir()
            _write_csv(
                native / "scheduling_native_q2_table.csv",
                [
                    {
                        "Method": "Direct / CoT",
                        "success": "1.0",
                        "gap": "0.0",
                        "llm_calls": "1",
                        "tokens": "10",
                        "wall_clock_seconds": "1",
                        "status": "complete",
                        "source_policy": "native",
                        "native_runtime_sample_count": "1",
                        "native_runtime_selection_events": "1",
                    }
                ],
            )
            _write_csv(native / "scheduling_native_q2_per_instance.csv", [{"Method": "Direct / CoT", "task_id": "t0"}])
            (native / "scheduling_native_q2_samples.jsonl").write_text(
                json.dumps({"Method": "Direct / CoT", "task_id": "t0", "sample_index": 0}) + "\n",
                encoding="utf-8",
            )
            (native / "scheduling_native_q2_manifest.json").write_text(
                json.dumps(
                    {
                        "splits": {"test": 1},
                        "native_protocol": {
                            "methods": {
                                "Direct / CoT": {
                                    "sample_count": 1,
                                    "selection_rule": "first_valid",
                                    "verifier_role": "validate complete schedules for metrics",
                                    "oracle_future_labels_exposed": False,
                                }
                            }
                        },
                        "method_alignment": q2_method_contracts(["Direct / CoT"]),
                    }
                ),
                encoding="utf-8",
            )
            out = root / "out"
            audit_table5_q2_readiness(out, solver_prior_dir=root / "missing", expected_test_count=1, direct_dir=native)
            audit_rows = {row["Method"]: row for row in _read_csv(out / "table5_q2_readiness.csv")}
            self.assertEqual(audit_rows["Direct / CoT"]["result_ready"], "False")
            self.assertIn(
                "scheduling_native_q2_manifest.json missing source_evidence for Direct / CoT",
                audit_rows["Direct / CoT"]["result_blockers"],
            )

    def test_readiness_audit_rejects_github_external_row_without_pinned_revision(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tot = root / "tot"
            tot.mkdir()
            _write_csv(
                tot / "scheduling_tot_q2_table.csv",
                [
                    {
                        "Method": "Tree of Thoughts",
                        "success": "1.0",
                        "gap": "0.0",
                        "llm_calls": "4",
                        "tokens": "40",
                        "wall_clock_seconds": "1",
                        "status": "complete",
                        "source_policy": "native_tot_bfs",
                        "tot_generate_events": "1",
                        "tot_evaluate_events": "1",
                        "tot_select_events": "1",
                        "tot_value_sample_count": "1",
                    }
                ],
            )
            _write_csv(tot / "scheduling_tot_q2_per_instance.csv", [{"task_id": "t0", "rendering_id": "r0"}])
            _write_tot_manifest(tot, test_count=1)
            (tot / "scheduling_tot_q2_trace.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps({"event": "generate", "rendering_id": "r0"}),
                        json.dumps(
                            {
                                "event": "evaluate_select",
                                "rendering_id": "r0",
                                "candidate_ids": ["c0"],
                                "selected_ids": ["c0"],
                                "averaged_scores": [1.0],
                                "value_samples": [{"scores": [1.0]}],
                            }
                        ),
                        json.dumps({"event": "finish", "rendering_id": "r0"}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            weakened_evidence = {method: dict(evidence) for method, evidence in Q2_SOURCE_EVIDENCE.items()}
            weakened_evidence["Tree of Thoughts"]["source_revision"] = ""
            out = root / "out"
            with patch("cgr_exp.table5_q2.Q2_SOURCE_EVIDENCE", weakened_evidence):
                audit_table5_q2_readiness(out, solver_prior_dir=root / "missing", expected_test_count=1, gpt_frontier_rank_dir=tot)
            audit_rows = {row["Method"]: row for row in _read_csv(out / "table5_q2_readiness.csv")}
            self.assertEqual(audit_rows["Tree of Thoughts"]["result_ready"], "False")
            self.assertIn("source evidence: missing source_revision", audit_rows["Tree of Thoughts"]["result_blockers"])

    def test_readiness_audit_requires_target_prm_training_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prm = root / "prm"
            prm.mkdir()
            _write_csv(
                prm / "scheduling_llm_pilot_table.csv",
                [
                    {
                        "Policy": "Target-PRM",
                        "success": "0.84",
                        "gap": "0.09",
                        "nodes_mean": "80",
                        "proposal_calls_mean": "80",
                        "value_calls": "0",
                        "proposal_tokens": "1000",
                        "value_tokens": "0",
                        "total_tokens": "1000",
                        "on_demand_total_tokens": "1000",
                    }
                ],
            )
            _write_csv(
                prm / "scheduling_llm_pilot_per_instance.csv",
                [{"Policy": "Target-PRM", "task_id": f"t{i}"} for i in range(100)],
            )
            out = root / "out"
            audit_table5_q2_readiness(out, solver_prior_dir=root / "missing", target_prm_dir=prm)
            audit_rows = {row["Method"]: row for row in _read_csv(out / "table5_q2_readiness.csv")}
            self.assertEqual(audit_rows["Target-PRM priority"]["result_ready"], "False")
            self.assertIn("prm/scheduling_llm_pilot_table.csv", audit_rows["Target-PRM priority"]["source_artifact"])
            self.assertEqual(audit_rows["Target-PRM priority"]["source_policy"], "Target-PRM")
            self.assertIn("target_prm_ranker.json", audit_rows["Target-PRM priority"]["result_blockers"])

    def test_readiness_audit_rejects_target_prm_training_eval_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prm = root / "prm"
            prm.mkdir()
            _write_csv(
                prm / "scheduling_llm_pilot_table.csv",
                [
                    {
                        "Policy": "Target-PRM",
                        "success": "0.84",
                        "gap": "0.09",
                        "nodes_mean": "80",
                        "proposal_calls_mean": "80",
                        "value_calls": "0",
                        "proposal_tokens": "1000",
                        "value_tokens": "0",
                        "total_tokens": "1000",
                        "on_demand_total_tokens": "1000",
                    }
                ],
            )
            _write_csv(
                prm / "scheduling_llm_pilot_per_instance.csv",
                [{"Policy": "Target-PRM", "task_id": "heldout-0"}],
            )
            (prm / "target_prm_ranker.json").write_text("{}", encoding="utf-8")
            (prm / "target_prm_trace_records.jsonl").write_text(
                json.dumps({"task_id": "heldout-0", "split": "target_train"}) + "\n",
                encoding="utf-8",
            )
            (prm / "target_prm_train_pairs.jsonl").write_text(json.dumps({"left": 0, "right": 1}) + "\n", encoding="utf-8")
            _write_learned_metrics(prm, ["Target-PRM priority"], ["Target-PRM"])
            out = root / "out"
            audit_table5_q2_readiness(out, solver_prior_dir=root / "missing", target_prm_dir=prm, expected_test_count=1)
            audit_rows = {row["Method"]: row for row in _read_csv(out / "table5_q2_readiness.csv")}
            self.assertEqual(audit_rows["Target-PRM priority"]["result_ready"], "False")
            self.assertIn("overlaps held-out evaluation tasks", audit_rows["Target-PRM priority"]["result_blockers"])

    def test_readiness_audit_rejects_target_prm_claiming_zero_shot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prm = root / "prm"
            prm.mkdir()
            _write_csv(
                prm / "scheduling_llm_pilot_table.csv",
                [
                    {
                        "Policy": "Target-PRM",
                        "success": "0.84",
                        "gap": "0.09",
                        "nodes_mean": "80",
                        "proposal_calls_mean": "80",
                        "value_calls": "0",
                        "proposal_tokens": "1000",
                        "value_tokens": "0",
                        "total_tokens": "1000",
                        "on_demand_total_tokens": "1000",
                    }
                ],
            )
            _write_csv(prm / "scheduling_llm_pilot_per_instance.csv", [{"Policy": "Target-PRM", "task_id": "heldout-0"}])
            (prm / "target_prm_ranker.json").write_text("{}", encoding="utf-8")
            (prm / "target_prm_trace_records.jsonl").write_text(json.dumps({"task_id": "train-0"}) + "\n", encoding="utf-8")
            (prm / "target_prm_train_pairs.jsonl").write_text(json.dumps({"left": 0, "right": 1}) + "\n", encoding="utf-8")
            _write_target_prm_label_manifest(prm)
            (prm / "scheduling_llm_pilot_metrics.json").write_text(
                json.dumps(
                    {
                        "q2_method_alignment": {"Target-PRM priority": {}},
                        "learned_search_protocol": {
                            "policies": {
                                "Target-PRM": {
                                    "training_distribution": "target LLM-induced Scheduling trajectories from disjoint training instances",
                                    "zero_shot_transfer": True,
                                }
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            out = root / "out"
            audit_table5_q2_readiness(out, solver_prior_dir=root / "missing", target_prm_dir=prm, expected_test_count=1)
            audit_rows = {row["Method"]: row for row in _read_csv(out / "table5_q2_readiness.csv")}
            self.assertEqual(audit_rows["Target-PRM priority"]["result_ready"], "False")
            self.assertIn("Target-PRM protocol must not claim zero-shot transfer", audit_rows["Target-PRM priority"]["result_blockers"])

    def test_readiness_audit_rejects_target_prm_outcome_only_labels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prm = root / "prm"
            prm.mkdir()
            _write_csv(
                prm / "scheduling_llm_pilot_table.csv",
                [
                    {
                        "Policy": "Target-PRM",
                        "success": "0.8",
                        "gap": "0.1",
                        "nodes_mean": "80",
                        "proposal_calls_mean": "80",
                        "value_calls": "0",
                        "proposal_tokens": "1000",
                        "value_tokens": "0",
                        "total_tokens": "1000",
                        "on_demand_total_tokens": "1000",
                    }
                ],
            )
            _write_csv(prm / "scheduling_llm_pilot_per_instance.csv", [{"Policy": "Target-PRM", "task_id": "heldout-0"}])
            (prm / "target_prm_ranker.json").write_text("{}", encoding="utf-8")
            (prm / "target_prm_trace_records.jsonl").write_text(json.dumps({"task_id": "train-0"}) + "\n", encoding="utf-8")
            (prm / "target_prm_train_pairs.jsonl").write_text(json.dumps({"left": 0, "right": 1}) + "\n", encoding="utf-8")
            _write_learned_metrics(prm, ["Target-PRM priority"], ["Target-PRM"])
            (prm / "target_prm_label_manifest.json").write_text(
                json.dumps(
                    {
                        "supervision": "outcome_only",
                        "label_source": "terminal_success",
                        "terminal_outcome_only": True,
                        "uses_intermediate_states": False,
                        "oracle_labels_in_inference_features": False,
                        "target_pair_count": 1,
                        "process_label_record_count": 1,
                        "finite_best_descendant_labels": 1,
                        "max_record_depth": 0,
                        "frontier_parent_group_count": 0,
                    }
                ),
                encoding="utf-8",
            )

            out = root / "out"
            audit_table5_q2_readiness(out, solver_prior_dir=root / "missing", target_prm_dir=prm, expected_test_count=1)
            audit_rows = {row["Method"]: row for row in _read_csv(out / "table5_q2_readiness.csv")}
            self.assertEqual(audit_rows["Target-PRM priority"]["result_ready"], "False")
            self.assertIn("must declare target_process_frontier_ranking supervision", audit_rows["Target-PRM priority"]["result_blockers"])
            self.assertIn("must not be terminal outcome only", audit_rows["Target-PRM priority"]["result_blockers"])
            self.assertIn("must declare intermediate state supervision", audit_rows["Target-PRM priority"]["result_blockers"])
            self.assertIn("must include non-root intermediate states", audit_rows["Target-PRM priority"]["result_blockers"])
            self.assertIn("must record frontier parent groups", audit_rows["Target-PRM priority"]["result_blockers"])

    def test_readiness_audit_rejects_rest_mcts_manifest_claiming_full_self_training(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rest = root / "rest"
            rest.mkdir()
            _write_csv(
                rest / "scheduling_llm_pilot_table.csv",
                [
                    {
                        "Policy": "PRM-guided MCTS (ReST-MCTS-style)",
                        "success": "0.84",
                        "gap": "0.09",
                        "nodes_mean": "80",
                        "proposal_calls_mean": "80",
                        "value_calls": "0",
                        "proposal_tokens": "1000",
                        "value_tokens": "0",
                        "total_tokens": "1000",
                        "on_demand_total_tokens": "1000",
                    }
                ],
            )
            _write_csv(
                rest / "scheduling_llm_pilot_per_instance.csv",
                [{"Policy": "PRM-guided MCTS (ReST-MCTS-style)", "task_id": "t0"}],
            )
            for name in ("target_prm_ranker.json", "target_prm_trace_records.jsonl", "target_prm_train_pairs.jsonl"):
                (rest / name).write_text("{}\n", encoding="utf-8")
            (rest / "scheduling_llm_pilot_metrics.json").write_text(
                json.dumps(
                    {
                        "q2_method_alignment": {"PRM-guided MCTS (ReST-MCTS-style)": {}},
                        "learned_search_protocol": {
                            "policies": {
                                "PRM-guided MCTS (ReST-MCTS-style)": {
                                    "full_iterative_self_training": True,
                                }
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            out = root / "out"
            audit_table5_q2_readiness(out, solver_prior_dir=root / "missing", rest_mcts_dir=rest, expected_test_count=1)
            audit_rows = {row["Method"]: row for row in _read_csv(out / "table5_q2_readiness.csv")}
            self.assertEqual(audit_rows["PRM-guided MCTS (ReST-MCTS-style)"]["result_ready"], "False")
            self.assertIn("must not claim full iterative self-training", audit_rows["PRM-guided MCTS (ReST-MCTS-style)"]["result_blockers"])

    def test_readiness_audit_rejects_method_alignment_claiming_wrong_reproduction_scope(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rest = root / "rest"
            rest.mkdir()
            policy = "PRM-guided MCTS (ReST-MCTS-style)"
            method = "PRM-guided MCTS (ReST-MCTS-style)"
            _write_solver_table_artifacts(rest, policy, ["heldout-0"])
            (rest / "target_prm_ranker.json").write_text("{}", encoding="utf-8")
            (rest / "target_prm_trace_records.jsonl").write_text(json.dumps({"task_id": "train-0"}) + "\n", encoding="utf-8")
            (rest / "target_prm_train_pairs.jsonl").write_text(json.dumps({"left": 0, "right": 1}) + "\n", encoding="utf-8")
            _write_target_prm_label_manifest(rest)
            wrong_alignment = q2_method_contracts([method])
            wrong_alignment[method]["claim_scope"] = "official_reproduction"
            wrong_alignment[method]["official_reproduction"] = True
            (rest / "scheduling_llm_pilot_metrics.json").write_text(
                json.dumps(
                    {
                        "q2_method_alignment": wrong_alignment,
                        "semantic_state_schema": {"oracle_labels_in_inference_features": False},
                        "learned_search_protocol": {
                            "policies": {
                                policy: {
                                    "training_distribution": "same target LLM-induced PRM as Target-PRM",
                                    "mcts_components": ["selection", "expansion", "simulation/evaluation", "backpropagation"],
                                    "process_reward_guidance": "target PRM prior and leaf value estimates",
                                    "selection_score": "Q(node) + prior_weight * PRM_prior(node) + exploration * PRM_prior(node) * sqrt(parent_visits + 1) / (1 + node_visits)",
                                    "full_iterative_self_training": False,
                                    "oracle_future_labels_exposed": False,
                                }
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            out = root / "out"
            audit_table5_q2_readiness(out, solver_prior_dir=root / "missing", rest_mcts_dir=rest, expected_test_count=1)
            audit_rows = {row["Method"]: row for row in _read_csv(out / "table5_q2_readiness.csv")}
            self.assertEqual(audit_rows[method]["result_ready"], "False")
            self.assertIn("method alignment claim_scope mismatch", audit_rows[method]["result_blockers"])
            self.assertIn("method alignment official_reproduction mismatch", audit_rows[method]["result_blockers"])

    def test_readiness_audit_rejects_rest_mcts_without_mcts_components(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rest = root / "rest"
            rest.mkdir()
            _write_csv(
                rest / "scheduling_llm_pilot_table.csv",
                [
                    {
                        "Policy": "PRM-guided MCTS (ReST-MCTS-style)",
                        "success": "0.84",
                        "gap": "0.09",
                        "nodes_mean": "80",
                        "proposal_calls_mean": "80",
                        "value_calls": "0",
                        "proposal_tokens": "1000",
                        "value_tokens": "0",
                        "total_tokens": "1000",
                        "on_demand_total_tokens": "1000",
                    }
                ],
            )
            _write_csv(
                rest / "scheduling_llm_pilot_per_instance.csv",
                [{"Policy": "PRM-guided MCTS (ReST-MCTS-style)", "task_id": "t0"}],
            )
            for name in ("target_prm_ranker.json", "target_prm_trace_records.jsonl", "target_prm_train_pairs.jsonl"):
                (rest / name).write_text("{}\n", encoding="utf-8")
            (rest / "scheduling_llm_pilot_metrics.json").write_text(
                json.dumps(
                    {
                        "q2_method_alignment": q2_method_contracts(["PRM-guided MCTS (ReST-MCTS-style)"]),
                        "learned_search_protocol": {
                            "policies": {
                                "PRM-guided MCTS (ReST-MCTS-style)": {
                                    "training_distribution": "same target LLM-induced PRM as Target-PRM",
                                    "full_iterative_self_training": False,
                                    "oracle_future_labels_exposed": False,
                                }
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            out = root / "out"
            audit_table5_q2_readiness(out, solver_prior_dir=root / "missing", rest_mcts_dir=rest, expected_test_count=1)
            audit_rows = {row["Method"]: row for row in _read_csv(out / "table5_q2_readiness.csv")}
            self.assertEqual(audit_rows["PRM-guided MCTS (ReST-MCTS-style)"]["result_ready"], "False")
            self.assertIn("must declare MCTS selection, expansion, simulation/evaluation, and backpropagation", audit_rows["PRM-guided MCTS (ReST-MCTS-style)"]["result_blockers"])

    def test_readiness_audit_rejects_rest_mcts_without_runtime_mcts_counts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rest = root / "rest"
            rest.mkdir()
            _write_csv(
                rest / "scheduling_llm_pilot_table.csv",
                [
                    {
                        "Policy": "PRM-guided MCTS (ReST-MCTS-style)",
                        "success": "0.84",
                        "gap": "0.09",
                        "nodes_mean": "80",
                        "proposal_calls_mean": "80",
                        "value_calls": "0",
                        "proposal_tokens": "1000",
                        "value_tokens": "0",
                        "total_tokens": "1000",
                        "on_demand_total_tokens": "1000",
                    }
                ],
            )
            _write_csv(
                rest / "scheduling_llm_pilot_per_instance.csv",
                [{"Policy": "PRM-guided MCTS (ReST-MCTS-style)", "task_id": "t0"}],
            )
            for name in ("target_prm_ranker.json", "target_prm_trace_records.jsonl", "target_prm_train_pairs.jsonl"):
                (rest / name).write_text("{}\n", encoding="utf-8")
            _write_learned_metrics(rest, ["PRM-guided MCTS (ReST-MCTS-style)"], ["PRM-guided MCTS (ReST-MCTS-style)"])

            out = root / "out"
            audit_table5_q2_readiness(out, solver_prior_dir=root / "missing", rest_mcts_dir=rest, expected_test_count=1)
            audit_rows = {row["Method"]: row for row in _read_csv(out / "table5_q2_readiness.csv")}
            self.assertEqual(audit_rows["PRM-guided MCTS (ReST-MCTS-style)"]["result_ready"], "False")
            self.assertIn("must record positive MCTS selection calls", audit_rows["PRM-guided MCTS (ReST-MCTS-style)"]["result_blockers"])
            self.assertIn("must record positive MCTS backpropagation updates", audit_rows["PRM-guided MCTS (ReST-MCTS-style)"]["result_blockers"])

    def test_readiness_audit_reports_malformed_learned_metrics_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            solver = root / "solver"
            solver.mkdir()
            _write_csv(
                solver / "scheduling_llm_pilot_table.csv",
                [
                    {
                        "Policy": "Solver-trained MLP",
                        "success": "0.89",
                        "gap": "0.02",
                        "nodes_mean": "80",
                        "proposal_calls_mean": "80",
                        "value_calls": "0",
                        "proposal_tokens": "1000",
                        "value_tokens": "0",
                        "total_tokens": "1000",
                        "on_demand_total_tokens": "1000",
                    }
                ],
            )
            _write_csv(
                solver / "scheduling_llm_pilot_per_instance.csv",
                [{"Policy": "Solver-trained MLP", "task_id": "t0"}],
            )
            (solver / "scheduling_llm_pilot_ranker.json").write_text("{}", encoding="utf-8")
            (solver / "scheduling_llm_pilot_metrics.json").write_text("{bad json\n", encoding="utf-8")
            out = root / "out"
            audit_table5_q2_readiness(out, solver_prior_dir=solver, expected_test_count=1)
            audit_rows = {row["Method"]: row for row in _read_csv(out / "table5_q2_readiness.csv")}
            self.assertEqual(audit_rows["Solver-trained priority"]["result_ready"], "False")
            self.assertIn("scheduling_llm_pilot_metrics.json malformed json", audit_rows["Solver-trained priority"]["result_blockers"])

    def test_readiness_audit_rejects_solver_prior_without_zero_shot_protocol(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            solver = root / "solver"
            solver.mkdir()
            _write_csv(
                solver / "scheduling_llm_pilot_table.csv",
                [
                    {
                        "Policy": "Solver-trained MLP",
                        "success": "0.89",
                        "gap": "0.02",
                        "nodes_mean": "80",
                        "proposal_calls_mean": "80",
                        "value_calls": "0",
                        "proposal_tokens": "1000",
                        "value_tokens": "0",
                        "total_tokens": "1000",
                        "on_demand_total_tokens": "1000",
                    }
                ],
            )
            _write_csv(solver / "scheduling_llm_pilot_per_instance.csv", [{"Policy": "Solver-trained MLP", "task_id": "heldout-0"}])
            (solver / "scheduling_llm_pilot_ranker.json").write_text("{}", encoding="utf-8")
            (solver / "scheduling_llm_pilot_metrics.json").write_text(
                json.dumps(
                    {
                        "q2_method_alignment": {"Solver-trained priority": {}},
                        "semantic_state_schema": {"oracle_labels_in_inference_features": False},
                        "learned_search_protocol": {
                            "policies": {
                                "Solver-trained MLP": {
                                    "training_distribution": "target LLM-induced Scheduling trajectories",
                                    "zero_shot_transfer": False,
                                }
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            out = root / "out"
            audit_table5_q2_readiness(out, solver_prior_dir=solver, expected_test_count=1)
            audit_rows = {row["Method"]: row for row in _read_csv(out / "table5_q2_readiness.csv")}
            self.assertEqual(audit_rows["Solver-trained priority"]["result_ready"], "False")
            self.assertIn("Solver-trained MLP protocol must declare zero-shot transfer", audit_rows["Solver-trained priority"]["result_blockers"])
            self.assertIn("source-only solver training", audit_rows["Solver-trained priority"]["result_blockers"])

    def test_readiness_audit_rejects_solver_prior_train_test_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            solver = root / "solver"
            solver.mkdir()
            _write_solver_table_artifacts(solver, "Solver-trained MLP", ["heldout-0"])
            _write_learned_metrics(solver, ["Solver-trained priority"], ["Solver-trained MLP"])
            _write_solver_audit_artifacts(solver, train_ids=["heldout-0"], validation_ids=[], test_ids=["heldout-0"])
            out = root / "out"
            audit_table5_q2_readiness(out, solver_prior_dir=solver, expected_test_count=1)
            audit_rows = {row["Method"]: row for row in _read_csv(out / "table5_q2_readiness.csv")}
            self.assertEqual(audit_rows["Solver-trained priority"]["result_ready"], "False")
            self.assertIn("train/test overlap", audit_rows["Solver-trained priority"]["result_blockers"])
            self.assertIn("training_instance_ids overlap held-out test", audit_rows["Solver-trained priority"]["result_blockers"])

    def test_readiness_audit_rejects_solver_prior_feature_schema_leakage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            solver = root / "solver"
            solver.mkdir()
            _write_solver_table_artifacts(solver, "Solver-trained MLP", ["heldout-0"])
            _write_learned_metrics(solver, ["Solver-trained priority"], ["Solver-trained MLP"])
            _write_solver_audit_artifacts(solver, train_ids=["train-0"], validation_ids=[], test_ids=["heldout-0"])
            (solver / "feature_schema.json").write_text(
                json.dumps({"oracle_labels_in_inference_features": True}),
                encoding="utf-8",
            )
            out = root / "out"
            audit_table5_q2_readiness(out, solver_prior_dir=solver, expected_test_count=1)
            audit_rows = {row["Method"]: row for row in _read_csv(out / "table5_q2_readiness.csv")}
            self.assertEqual(audit_rows["Solver-trained priority"]["result_ready"], "False")
            self.assertIn("feature_schema.json must exclude oracle labels", audit_rows["Solver-trained priority"]["result_blockers"])


def _write_csv(path: Path, rows) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_native_manifest(path: Path, methods: list[str], test_count: int) -> None:
    method_protocols = {}
    for method in methods:
        sample_count = 1 if method == "Direct / CoT" else 8
        selection_rule = {
            "Direct / CoT": "first_valid",
            "Self-Consistency": "majority_job_order_then_first_valid",
            "Best-of-N": "best_verified_objective",
        }[method]
        method_protocols[method] = {
            "sample_count": sample_count,
            "selection_rule": selection_rule,
            "verifier_role": (
                "filter invalid complete schedules and select minimum verified objective"
                if method == "Best-of-N"
                else "validate complete schedules for metrics"
            ),
            "oracle_future_labels_exposed": False,
            "intermediate_reasoning_requested": True,
            "reasoning_location": "JSON field: reasoning",
        }
    (path / "scheduling_native_q2_manifest.json").write_text(
        json.dumps(
            {
                "splits": {"test": test_count},
                "native_protocol": {"methods": method_protocols},
                "method_alignment": q2_method_contracts(methods),
                "source_evidence": q2_source_evidence_for_methods(methods),
            }
        ),
        encoding="utf-8",
    )


def _write_tot_manifest(path: Path, test_count: int) -> None:
    (path / "scheduling_tot_q2_manifest.json").write_text(
        json.dumps(
            {
                "splits": {"test": test_count},
                "native_tot_protocol": {
                    "official_entrypoints": ["method_generate", "method_evaluate", "method_select"],
                    "oracle_future_labels_exposed": False,
                    "search": "breadth-first frontier search",
                    "method_select": "greedy top-b by averaged value samples",
                    "method_evaluate": "LLM frontier value/vote evaluation",
                    "n_select_sample": 2,
                    "n_evaluate_sample": 2,
                },
                "method_alignment": q2_method_contracts(["Tree of Thoughts"]),
                "source_evidence": q2_source_evidence_for_methods(["Tree of Thoughts"]),
            }
        ),
        encoding="utf-8",
    )


def _write_structured_metrics(path: Path, methods: list[str], test_count: int) -> None:
    table_rows = _read_csv(path / "scheduling_structured_q2_table.csv")
    patched_rows = []
    for row in table_rows:
        if row.get("Method") == "CP-SAT":
            row = {
                **row,
                "structured_runtime_solver": "ortools_cp_sat",
                "structured_runtime_integer_variables": row.get("structured_runtime_integer_variables", "10"),
                "structured_runtime_boolean_variables": row.get("structured_runtime_boolean_variables", "4"),
                "structured_runtime_interval_variables": row.get("structured_runtime_interval_variables", "4"),
                "structured_runtime_exactly_one_constraints": row.get("structured_runtime_exactly_one_constraints", "2"),
                "structured_runtime_no_overlap_constraints": row.get("structured_runtime_no_overlap_constraints", "2"),
                "structured_runtime_precedence_constraints": row.get("structured_runtime_precedence_constraints", "1"),
                "structured_runtime_release_constraints": row.get("structured_runtime_release_constraints", "2"),
                "structured_runtime_objective_terms": row.get("structured_runtime_objective_terms", "3"),
            }
        if row.get("Method") == "EST / list scheduling heuristic":
            row = {
                **row,
                "structured_runtime_heuristic": "earliest_start_list_scheduling",
                "structured_runtime_dispatch_steps": row.get("structured_runtime_dispatch_steps", "2"),
                "structured_runtime_ready_list_selections": row.get("structured_runtime_ready_list_selections", "2"),
                "structured_runtime_candidate_evaluations": row.get("structured_runtime_candidate_evaluations", "4"),
                "structured_runtime_tie_break_rule": row.get("structured_runtime_tie_break_rule", "min(start, finish, tardiness, due_date, job_id, machine_id)"),
            }
        patched_rows.append(row)
    if patched_rows:
        _write_csv(path / "scheduling_structured_q2_table.csv", patched_rows)
    (path / "scheduling_structured_q2_metrics.json").write_text(
        json.dumps(
            {
                "splits": {"test": test_count},
                "structured_input": True,
                "llm_calls": 0,
                "tokens": 0,
                "method_alignment": q2_method_contracts(methods),
                "source_evidence": q2_source_evidence_for_methods(methods),
            }
        ),
        encoding="utf-8",
    )


def _write_structured_metrics_without_table_patch(path: Path, methods: list[str], test_count: int) -> None:
    (path / "scheduling_structured_q2_metrics.json").write_text(
        json.dumps(
            {
                "splits": {"test": test_count},
                "structured_input": True,
                "llm_calls": 0,
                "tokens": 0,
                "method_alignment": q2_method_contracts(methods),
                "source_evidence": q2_source_evidence_for_methods(methods),
            }
        ),
        encoding="utf-8",
    )


def _write_learned_metrics(path: Path, methods: list[str], policies: list[str]) -> None:
    policy_protocols = {}
    for policy in policies:
        if policy == "Target-PRM":
            policy_protocols[policy] = {
                "training_distribution": "target LLM-induced Scheduling trajectories from disjoint training instances",
                "zero_shot_transfer": False,
                "oracle_future_labels_exposed": False,
            }
        elif policy == "PRM-guided MCTS (ReST-MCTS-style)":
            policy_protocols[policy] = {
                "training_distribution": "same target LLM-induced PRM as Target-PRM",
                "mcts_components": ["selection", "expansion", "simulation/evaluation", "backpropagation"],
                "process_reward_guidance": "target PRM prior and leaf value estimates",
                "selection_score": "Q(node) + prior_weight * PRM_prior(node) + exploration * PRM_prior(node) * sqrt(parent_visits + 1) / (1 + node_visits)",
                "full_iterative_self_training": False,
                "oracle_future_labels_exposed": False,
            }
        elif policy == "Solver-trained MLP":
            policy_protocols[policy] = {
                "training_distribution": "solver-induced Scheduling source trajectories only",
                "zero_shot_transfer": True,
                "oracle_future_labels_exposed": False,
            }
        elif policy == "Solver-trained MLP + certified pruning":
            policy_protocols[policy] = {
                "training_distribution": "same solver-induced source controller as Solver-trained MLP",
                "zero_shot_transfer": True,
                "learned_score_authorizes_pruning": False,
                "pruning_authority": "verified lower-bound/feasibility logic only",
                "oracle_future_labels_exposed": False,
            }
        else:
            policy_protocols[policy] = {}
    (path / "scheduling_llm_pilot_metrics.json").write_text(
        json.dumps(
            {
                "q2_method_alignment": q2_method_contracts(methods),
                "source_evidence": q2_source_evidence_for_methods(methods),
                "semantic_state_schema": {"oracle_labels_in_inference_features": False},
                "learned_search_protocol": {"policies": policy_protocols},
            }
        ),
        encoding="utf-8",
    )


def _write_solver_table_artifacts(path: Path, policy: str, task_ids: list[str]) -> None:
    (path / "scheduling_llm_pilot_ranker.json").write_text("{}", encoding="utf-8")
    row = {
        "Policy": policy,
        "success": "0.89",
        "gap": "0.02",
        "nodes_mean": "80",
        "proposal_calls_mean": "80",
        "value_calls": "0",
        "proposal_tokens": "1000",
        "value_tokens": "0",
        "total_tokens": "1000",
        "on_demand_total_tokens": "1000",
    }
    if policy == "PRM-guided MCTS (ReST-MCTS-style)":
        row.update(
            {
                "mcts_selection_calls": "10",
                "mcts_expansion_observations": "10",
                "mcts_prior_evaluations": "10",
                "mcts_leaf_value_evaluations": "10",
                "mcts_backup_updates": "10",
                "policy_diagnostics": json.dumps(
                    {
                        "selection_score": "Q(node) + prior_weight * PRM_prior(node) + exploration * PRM_prior(node) * sqrt(parent_visits + 1) / (1 + node_visits)"
                    }
                ),
            }
        )
    _write_csv(
        path / "scheduling_llm_pilot_table.csv",
        [row],
    )
    _write_csv(path / "scheduling_llm_pilot_per_instance.csv", [{"Policy": policy, "task_id": task_id} for task_id in task_ids])


def _write_solver_audit_artifacts(path: Path, train_ids: list[str], validation_ids: list[str], test_ids: list[str]) -> None:
    feature_schema = {
        "schema_version": "test",
        "oracle_labels_in_inference_features": False,
    }
    (path / "split_manifest.json").write_text(
        json.dumps({"train": train_ids, "validation": validation_ids, "test": test_ids}),
        encoding="utf-8",
    )
    (path / "checkpoint_metadata.json").write_text(
        json.dumps(
            {
                "source_training_split": "train",
                "training_instance_ids": train_ids,
                "validation_instance_ids": validation_ids,
                "feature_schema_hash_sha256": "test",
            }
        ),
        encoding="utf-8",
    )
    (path / "feature_schema.json").write_text(json.dumps(feature_schema), encoding="utf-8")


def _write_target_prm_label_manifest(path: Path) -> None:
    (path / "target_prm_label_manifest.json").write_text(
        json.dumps(
            {
                "supervision": "target_process_frontier_ranking",
                "label_source": "exact_residual_scheduling_oracle_on_target_llm_states",
                "state_distribution": "target LLM-induced Scheduling trajectories from disjoint training instances",
                "terminal_outcome_only": False,
                "uses_intermediate_states": True,
                "process_label_record_count": 1,
                "target_pair_count": 1,
                "finite_best_descendant_labels": 1,
                "infeasible_best_descendant_labels": 0,
                "max_record_depth": 1,
                "frontier_parent_group_count": 1,
                "oracle_labels_in_inference_features": False,
            }
        ),
        encoding="utf-8",
    )
