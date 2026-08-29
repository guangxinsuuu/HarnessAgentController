from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from cgr_exp.q4_tau_source_evidence import q4_tau_source_evidence
from cgr_exp.table_q4_tau import audit_q4_tau_readiness, build_q4_tau_results


class TableQ4TauTest(unittest.TestCase):
    def test_q4_tau_builder_keeps_rows_pending_without_native_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = build_q4_tau_results(root, expected_test_count=5)
            rows = _read_csv(root / "q4_tau_results.csv")
            by_method = {row["Method"]: row for row in rows}
            self.assertEqual(by_method["Native agent rerun"]["Status"], "pending")
            self.assertEqual(by_method["Best-of-N"]["pass^1"], "--")
            self.assertEqual(by_method["Target-PRM"]["source_url"], "https://github.com/sierra-research/tau-bench")
            self.assertEqual(manifest["strict_pending_policy"], True)
            self.assertEqual(manifest["preferred_current_source_url"], "https://github.com/sierra-research/tau2-bench")
            self.assertTrue(manifest["source_evidence"]["all_sources_ready"])
            self.assertTrue(manifest["source_ref_verification"]["all_static_refs_ready"])
            self.assertTrue((root / "q4_tau_results.tex").exists())

    def test_q4_tau_audit_accepts_complete_native_protocol_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            native = root / "native"
            native.mkdir()
            _write_tau_artifacts(native, "Native agent rerun", ["t0", "t1"])
            out = root / "out"
            manifest = audit_q4_tau_readiness(out, expected_test_count=2, native_dir=native)
            readiness = manifest["readiness"]
            rows = {row["Method"]: row for row in _read_csv(out / "q4_tau_readiness.csv")}
            self.assertFalse(readiness["all_rows_result_ready"])
            self.assertEqual(rows["Native agent rerun"]["result_ready"], "True")
            self.assertIn("native/q4_tau_row.json", rows["Native agent rerun"]["source_artifact"])
            self.assertEqual(rows["Native agent rerun"]["source_revision"], q4_tau_source_evidence("tau-bench")["source_revision"])
            self.assertEqual(rows["Best-of-N"]["result_ready"], "False")

    def test_q4_tau_audit_rejects_non_native_protocol_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            native = root / "native"
            native.mkdir()
            _write_tau_artifacts(native, "Native agent rerun", ["t0"])
            (native / "q4_tau_manifest.json").write_text(
                json.dumps(
                    {
                        "benchmark": "synthetic-level3",
                        "native_protocol_preserved": False,
                        "source_url": "local",
                        "methods": ["Native agent rerun"],
                    }
                ),
                encoding="utf-8",
            )
            out = root / "out"
            audit_q4_tau_readiness(out, expected_test_count=1, native_dir=native)
            rows = {row["Method"]: row for row in _read_csv(out / "q4_tau_readiness.csv")}
            self.assertEqual(rows["Native agent rerun"]["result_ready"], "False")
            self.assertIn("must preserve native protocol", rows["Native agent rerun"]["result_blockers"])
            self.assertIn("source_url is not a tau source", rows["Native agent rerun"]["result_blockers"])

    def test_q4_tau_audit_rejects_manifest_without_detailed_native_protocol(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            native = root / "native"
            native.mkdir()
            _write_tau_artifacts(native, "Native agent rerun", ["t0"])
            (native / "q4_tau_manifest.json").write_text(
                json.dumps(
                    {
                        "benchmark": "tau-bench",
                        "native_protocol_preserved": True,
                        "source_url": "https://github.com/sierra-research/tau-bench",
                        "methods": ["Native agent rerun"],
                    }
                ),
                encoding="utf-8",
            )
            out = root / "out"
            audit_q4_tau_readiness(out, expected_test_count=1, native_dir=native)
            rows = {row["Method"]: row for row in _read_csv(out / "q4_tau_readiness.csv")}
            self.assertEqual(rows["Native agent rerun"]["result_ready"], "False")
            self.assertIn("missing native_protocol", rows["Native agent rerun"]["result_blockers"])
            self.assertIn("missing method_protocols", rows["Native agent rerun"]["result_blockers"])

    def test_q4_tau_audit_rejects_artifact_without_pinned_source_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            native = root / "native"
            native.mkdir()
            _write_tau_artifacts(native, "Native agent rerun", ["t0"])
            manifest = json.loads((native / "q4_tau_manifest.json").read_text(encoding="utf-8"))
            manifest.pop("source_evidence")
            (native / "q4_tau_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            out = root / "out"
            audit_q4_tau_readiness(out, expected_test_count=1, native_dir=native)
            rows = {row["Method"]: row for row in _read_csv(out / "q4_tau_readiness.csv")}
            self.assertEqual(rows["Native agent rerun"]["result_ready"], "False")
            self.assertIn("missing source_evidence", rows["Native agent rerun"]["result_blockers"])

    def test_q4_tau_audit_rejects_solver_controller_without_zero_shot_protocol(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            solver = root / "solver"
            solver.mkdir()
            _write_tau_artifacts(solver, "Solver-derived controller", ["t0"])
            manifest = json.loads((solver / "q4_tau_manifest.json").read_text(encoding="utf-8"))
            manifest["method_protocols"]["Solver-derived controller"]["zero_shot_transfer"] = False
            (solver / "q4_tau_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            out = root / "out"
            audit_q4_tau_readiness(out, expected_test_count=1, solver_controller_dir=solver)
            rows = {row["Method"]: row for row in _read_csv(out / "q4_tau_readiness.csv")}
            self.assertEqual(rows["Solver-derived controller"]["result_ready"], "False")
            self.assertIn("must declare zero-shot transfer", rows["Solver-derived controller"]["result_blockers"])

    def test_q4_tau_audit_rejects_pass4_without_four_rollouts_per_task(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            native = root / "native"
            native.mkdir()
            _write_tau_artifacts(native, "Native agent rerun", ["t0", "t1"], rollout_count=3)
            out = root / "out"
            audit_q4_tau_readiness(out, expected_test_count=2, native_dir=native)
            rows = {row["Method"]: row for row in _read_csv(out / "q4_tau_readiness.csv")}
            self.assertEqual(rows["Native agent rerun"]["result_ready"], "False")
            self.assertIn("pass^4 requires at least 4 rollouts per task", rows["Native agent rerun"]["result_blockers"])


def _write_tau_artifacts(path: Path, method: str, task_ids: list[str], rollout_count: int = 4) -> None:
    (path / "q4_tau_row.json").write_text(
        json.dumps(
            {
                "Method": method,
                "pass^1": 0.5,
                "pass^4": 0.75,
                "violations": 0.1,
                "tool_calls": 12,
                "tokens": 3456,
                "wall_clock_seconds": 7.89,
                "status": "complete",
                "source_url": "https://github.com/sierra-research/tau-bench",
            }
        ),
        encoding="utf-8",
    )
    _write_csv(
        path / "q4_tau_per_instance.csv",
        [{"Method": method, "task_id": task_id, "rollout_count": rollout_count} for task_id in task_ids],
    )
    (path / "q4_tau_manifest.json").write_text(
        json.dumps(
            {
                "benchmark": "tau-bench",
                "native_protocol_preserved": True,
                "source_url": "https://github.com/sierra-research/tau-bench",
                "source_revision": q4_tau_source_evidence("tau-bench")["source_revision"],
                "source_evidence": q4_tau_source_evidence("tau-bench"),
                "methods": [method],
                "native_protocol": {
                    "preserved_components": [
                        "stateful_tool_environment",
                        "user_simulator",
                        "official_task_verifier",
                        "pass_k_aggregation",
                        "tool_call_accounting",
                        "token_accounting",
                    ],
                    "uses_official_user_simulator": True,
                    "uses_official_task_verifier": True,
                    "preserves_tool_state": True,
                    "reported_metrics": ["pass^1", "pass^4", "violations", "tool_calls", "tokens"],
                },
                "method_protocols": {method: _tau_method_protocol(method)},
            }
        ),
        encoding="utf-8",
    )


def _tau_method_protocol(method: str) -> dict:
    primitives = {
        "Native agent rerun": ["single_native_agent_rollout"],
        "Best-of-N": ["multiple_independent_native_rollouts", "benchmark_compatible_selection"],
        "Value-guided search": ["online_value_guidance", "native_tool_state_preserved"],
        "Target-PRM": ["target_tool_agent_trajectory_training", "disjoint_train_test_tasks", "process_or_step_supervision"],
        "Solver-derived controller": ["source_solver_training", "zero_shot_target_tool_transfer", "native_tool_state_preserved"],
    }[method]
    return {
        "method_primitives": primitives,
        "training_eval_overlap": False,
        "zero_shot_transfer": True if method == "Solver-derived controller" else False if method == "Target-PRM" else None,
    }


def _write_csv(path: Path, rows) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


if __name__ == "__main__":
    unittest.main()
