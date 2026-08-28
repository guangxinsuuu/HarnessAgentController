from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from cgr_exp.table5_q2 import audit_table5_q2_readiness, build_table5_q2_results


class Table5Q2Test(unittest.TestCase):
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
                    }
                ),
                encoding="utf-8",
            )
            (native / "scheduling_native_q2_samples.jsonl").write_text(
                json.dumps({"Method": "Direct / CoT", "task_id": "t0"}) + "\n",
                encoding="utf-8",
            )
            out = root / "out"
            audit_table5_q2_readiness(out, solver_prior_dir=root / "missing", direct_dir=native, expected_test_count=1)
            audit_rows = {row["Method"]: row for row in _read_csv(out / "table5_q2_readiness.csv")}
            self.assertEqual(audit_rows["Direct / CoT"]["result_ready"], "True")
            self.assertNotIn("scheduling_native_q2_samples.jsonl", audit_rows["Direct / CoT"]["result_blockers"])

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
            self.assertIn("shared-cache fallback accounting", audit_rows["Solver-trained priority"]["compute_blockers"])
            self.assertTrue((out / "table5_q2_readiness.json").exists())

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
            self.assertIn("target_prm_ranker.json", audit_rows["Target-PRM priority"]["result_blockers"])


def _write_csv(path: Path, rows) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))
