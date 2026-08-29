from __future__ import annotations

import argparse
import json
from pathlib import Path
from dataclasses import replace

from cgr_exp.analysis.statistics import write_seed_aggregate
from cgr_exp.certification_table_readiness import audit_certification_tables
from cgr_exp.experiment_status_report import build_current_experiment_status
from cgr_exp.experiments import PRESETS, run_experiment
from cgr_exp.game24_experiments import run_game24_offline_experiment
from cgr_exp.game24_llm_pilot import ScriptedGame24Client, run_game24_llm_pilot, run_game24_llm_pilot_from_env
from cgr_exp.run_experiment import main as run_experiment_main
from cgr_exp.scheduling_certification import run_scheduling_certification_experiment
from cgr_exp.scheduling_diagnosis import run_scheduling_transfer_diagnosis
from cgr_exp.scheduling_micro import run_scheduling_micro_experiment
from cgr_exp.scheduling_native_q2 import run_scheduling_native_q2, run_scheduling_native_q2_from_env
from cgr_exp.scheduling_outcome_prm import run_scheduling_outcome_prm
from cgr_exp.scheduling_policy_diagnosis import run_scheduling_policy_diagnosis
from cgr_exp.scheduling_representation import run_scheduling_representation_ablation
from cgr_exp.scheduling_structured_q2 import run_scheduling_structured_q2
from cgr_exp.scheduling_tot_q2 import run_scheduling_tot_q2, run_scheduling_tot_q2_from_env
from cgr_exp.scheduling_transfer import (
    ScriptedSchedulingClient,
    preflight_scheduling_llm_pilot,
    run_scheduling_fewshot_curve,
    run_scheduling_llm_pilot,
    run_scheduling_llm_pilot_from_env,
    run_scheduling_zero_shot_transfer,
)
from cgr_exp.secret_safety import audit_secret_safety
from cgr_exp.q2_implementation_matrix import build_q2_implementation_matrix
from cgr_exp.q2_local_primitive_audit import audit_q2_local_primitives
from cgr_exp.q2_remaining_runs import build_q2_remaining_runs
from cgr_exp.q2_run_preflight import build_q2_run_preflight
from cgr_exp.q2_source_evidence import audit_q2_source_evidence
from cgr_exp.q2_source_ref_verifier import verify_q2_source_refs
from cgr_exp.q4_tau_source_evidence import audit_q4_tau_source_evidence
from cgr_exp.q4_tau_source_ref_verifier import verify_q4_tau_source_refs
from cgr_exp.q4_tau_adapter_contract import build_q4_tau_adapter_contract
from cgr_exp.table5_q2 import audit_table5_q2_readiness, build_table5_q2_results
from cgr_exp.table_q4_tau import audit_q4_tau_readiness, build_q4_tau_results
from cgr_exp.table6_authority import run_table6_authority_experiment

COMMANDS = [
    "run-game24-offline",
    "run-game24-llm-pilot",
    "run-scheduling-micro",
    "run-scheduling-representation",
    "run-scheduling-transfer",
    "run-scheduling-fewshot",
    "run-scheduling-certification",
    "preflight-scheduling-llm-pilot",
    "run-scheduling-llm-pilot",
    "run-scheduling-native-q2",
    "run-scheduling-structured-q2",
    "run-scheduling-tot-q2",
    "run-scheduling-outcome-prm",
    "diagnose-scheduling-policy",
    "run-scheduling-diagnosis",
    "generate-level1",
    "generate-level2",
    "train-ranker",
    "adapt-ranker",
    "evaluate-search",
    "run-certification-corruption",
    "run-transfer-matrix",
    "analyze-domain-shift",
    "build-tables",
    "build-table4-q2",
    "build-table5-q2",
    "audit-table4-q2",
    "audit-table5-q2",
    "build-q2-implementation-matrix",
    "audit-q2-local-primitives",
    "build-q2-remaining-runs",
    "build-q2-run-preflight",
    "audit-q2-source-evidence",
    "verify-q2-source-refs",
    "audit-q4-tau-source-evidence",
    "verify-q4-tau-source-refs",
    "build-q4-tau-adapter-contract",
    "build-table-q4-tau",
    "audit-table-q4-tau",
    "run-table6-authority",
    "audit-certification-tables",
    "audit-secret-safety",
    "build-current-experiment-status",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="CGR research command dispatcher.")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("run-smoke")
    for name in COMMANDS:
        p = sub.add_parser(name)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--seeds", default=None)
        p.add_argument("--preset", choices=sorted(PRESETS), default="smoke")
        p.add_argument("--output-dir", type=Path, required=True)
        p.add_argument("--overwrite", action="store_true", default=False)
        if name == "run-game24-offline":
            p.add_argument("--limit", type=int, default=100)
        if name == "run-game24-llm-pilot":
            p.add_argument("--model", default=None)
            p.add_argument("--start", type=int, default=900)
            p.add_argument("--count", type=int, default=20)
            p.add_argument("--proposal-k", type=int, default=4)
            p.add_argument("--node-budget", type=int, default=80)
            p.add_argument("--timeout", type=float, default=120.0)
            p.add_argument("--batch-size", type=int, default=1)
            p.add_argument("--mock", action="store_true", default=False)
        if name == "verify-q2-source-refs":
            p.add_argument("--online", action="store_true", default=False)
            p.add_argument("--timeout", type=float, default=10.0)
        if name == "verify-q4-tau-source-refs":
            p.add_argument("--online", action="store_true", default=False)
            p.add_argument("--timeout", type=float, default=10.0)
        if name == "run-scheduling-micro":
            p.add_argument("--train-count", type=int, default=200)
            p.add_argument("--validation-count", type=int, default=50)
            p.add_argument("--test-count", type=int, default=100)
            p.add_argument("--job-count", type=int, default=6)
            p.add_argument("--machine-count", type=int, default=3)
            p.add_argument("--node-budget", type=int, default=120)
            p.add_argument("--action-budget", type=int, default=8)
            p.add_argument("--records-per-instance", type=int, default=20)
            p.add_argument("--shard-size", type=int, default=50)
            p.add_argument("--rank-epochs", type=int, default=10)
        if name == "run-scheduling-representation":
            p.add_argument("--train-count", type=int, default=200)
            p.add_argument("--validation-count", type=int, default=50)
            p.add_argument("--test-count", type=int, default=100)
            p.add_argument("--job-count", type=int, default=6)
            p.add_argument("--machine-count", type=int, default=3)
            p.add_argument("--node-budget", type=int, default=120)
            p.add_argument("--action-budget", type=int, default=8)
            p.add_argument("--records-per-instance", type=int, default=60)
            p.add_argument("--rank-epochs", type=int, default=25)
        if name == "run-scheduling-transfer":
            p.add_argument("--train-count", type=int, default=200)
            p.add_argument("--validation-count", type=int, default=50)
            p.add_argument("--test-count", type=int, default=100)
            p.add_argument("--job-count", type=int, default=6)
            p.add_argument("--machine-count", type=int, default=3)
            p.add_argument("--node-budget", type=int, default=120)
            p.add_argument("--action-budget", type=int, default=8)
            p.add_argument("--records-per-instance", type=int, default=60)
            p.add_argument("--rank-epochs", type=int, default=25)
            p.add_argument("--useful-action-recall", type=float, default=0.88)
            p.add_argument("--batch-size", type=int, default=1)
            p.add_argument("--parallel-episodes", type=int, default=1)
            p.add_argument("--resume", action="store_true", default=False)
            p.add_argument("--perturbation", default="mixed")
        if name == "run-scheduling-fewshot":
            p.add_argument("--solver-train-count", type=int, default=200)
            p.add_argument("--target-train-count", type=int, default=500)
            p.add_argument("--validation-count", type=int, default=50)
            p.add_argument("--target-test-count", type=int, default=100)
            p.add_argument("--job-count", type=int, default=6)
            p.add_argument("--machine-count", type=int, default=3)
            p.add_argument("--node-budget", type=int, default=120)
            p.add_argument("--action-budget", type=int, default=8)
            p.add_argument("--records-per-instance", type=int, default=60)
            p.add_argument("--rank-epochs", type=int, default=25)
            p.add_argument("--fewshot-counts", default="10,50,100,500")
            p.add_argument("--useful-action-recall", type=float, default=0.88)
            p.add_argument("--perturbation", default="mixed")
        if name == "run-scheduling-certification":
            p.add_argument("--train-count", type=int, default=200)
            p.add_argument("--validation-count", type=int, default=50)
            p.add_argument("--test-count", type=int, default=100)
            p.add_argument("--job-count", type=int, default=6)
            p.add_argument("--machine-count", type=int, default=3)
            p.add_argument("--node-budget", type=int, default=120)
            p.add_argument("--action-budget", type=int, default=8)
            p.add_argument("--records-per-instance", type=int, default=60)
            p.add_argument("--rank-epochs", type=int, default=25)
            p.add_argument("--useful-action-recall", type=float, default=0.88)
            p.add_argument("--perturbation", default="mixed")
        if name in ("run-scheduling-llm-pilot", "preflight-scheduling-llm-pilot"):
            p.add_argument("--model", default=None)
            p.add_argument("--train-count", type=int, default=200)
            p.add_argument("--validation-count", type=int, default=50)
            p.add_argument("--test-count", type=int, default=20)
            p.add_argument("--job-count", type=int, default=6)
            p.add_argument("--machine-count", type=int, default=3)
            p.add_argument("--node-budget", type=int, default=80)
            p.add_argument("--action-budget", type=int, default=6)
            p.add_argument("--records-per-instance", type=int, default=60)
            p.add_argument("--rank-epochs", type=int, default=25)
            p.add_argument("--timeout", type=float, default=120.0)
            p.add_argument("--batch-size", type=int, default=1)
            p.add_argument("--parallel-episodes", type=int, default=1)
            p.add_argument("--resume", action="store_true", default=False)
            p.add_argument("--perturbation", default="mixed")
            p.add_argument("--mock", action="store_true", default=False)
            p.add_argument("--include-gpt-value", action="store_true", default=False)
            p.add_argument("--include-gpt-frontier-ranking", action="store_true", default=False)
            p.add_argument("--include-target-prm", action="store_true", default=False)
            p.add_argument("--include-rest-mcts", action="store_true", default=False)
            p.add_argument("--include-solver-certification", action="store_true", default=False)
            p.add_argument("--target-prm-train-count", type=int, default=100)
        if name == "run-scheduling-native-q2":
            p.add_argument("--model", default=None)
            p.add_argument("--train-count", type=int, default=200)
            p.add_argument("--validation-count", type=int, default=50)
            p.add_argument("--test-count", type=int, default=100)
            p.add_argument("--job-count", type=int, default=6)
            p.add_argument("--machine-count", type=int, default=3)
            p.add_argument("--timeout", type=float, default=120.0)
            p.add_argument("--methods", default="direct,self_consistency,best_of_n")
            p.add_argument("--sample-count", type=int, default=8)
            p.add_argument("--parallel-episodes", type=int, default=1)
            p.add_argument("--resume", action="store_true", default=False)
            p.add_argument("--mock", action="store_true", default=False)
        if name == "run-scheduling-structured-q2":
            p.add_argument("--train-count", type=int, default=200)
            p.add_argument("--validation-count", type=int, default=50)
            p.add_argument("--test-count", type=int, default=100)
            p.add_argument("--job-count", type=int, default=6)
            p.add_argument("--machine-count", type=int, default=3)
            p.add_argument("--cpsat-time-limit-seconds", type=float, default=5.0)
        if name == "run-scheduling-tot-q2":
            p.add_argument("--model", default=None)
            p.add_argument("--train-count", type=int, default=200)
            p.add_argument("--validation-count", type=int, default=50)
            p.add_argument("--test-count", type=int, default=100)
            p.add_argument("--job-count", type=int, default=6)
            p.add_argument("--machine-count", type=int, default=3)
            p.add_argument("--node-budget", type=int, default=80)
            p.add_argument("--action-budget", type=int, default=6)
            p.add_argument("--breadth", type=int, default=5)
            p.add_argument("--value-samples", type=int, default=3)
            p.add_argument("--timeout", type=float, default=120.0)
            p.add_argument("--parallel-episodes", type=int, default=1)
            p.add_argument("--resume", action="store_true", default=False)
            p.add_argument("--perturbation", default="mixed")
        if name == "build-q2-remaining-runs":
            p.add_argument("--q2-readiness", type=Path, default=Path("runs/q2_table_current_seed0_protocol_v2_audit/table5_q2_readiness.csv"))
            p.add_argument("--certification-readiness", type=Path, default=Path("runs/certification_tables_readiness/certification_tables_readiness.csv"))
        if name == "build-q2-run-preflight":
            p.add_argument("--model", default=None)
            p.add_argument("--train-count", type=int, default=200)
            p.add_argument("--validation-count", type=int, default=50)
            p.add_argument("--test-count", type=int, default=100)
            p.add_argument("--job-count", type=int, default=6)
            p.add_argument("--machine-count", type=int, default=3)
            p.add_argument("--node-budget", type=int, default=80)
            p.add_argument("--action-budget", type=int, default=6)
            p.add_argument("--records-per-instance", type=int, default=60)
            p.add_argument("--rank-epochs", type=int, default=25)
            p.add_argument("--perturbation", default="mixed")
            p.add_argument("--native-dir", type=Path, default=Path("runs/q2_native_llm_baselines_100_seed0_sc_v2"))
            p.add_argument("--learned-dir", type=Path, default=Path("runs/q2_learned_search_100_seed0_rest_gate_v2"))
            p.add_argument("--tot-dir", type=Path, default=Path("runs/q2_tot_bfs_100_seed0_tot_gate_v2"))
            p.add_argument("--sample-count", type=int, default=8)
            p.add_argument("--native-parallel-episodes", type=int, default=8)
            p.add_argument("--search-parallel-episodes", type=int, default=6)
            p.add_argument("--tot-parallel-episodes", type=int, default=6)
            p.add_argument("--batch-size", type=int, default=4)
            p.add_argument("--target-prm-train-count", type=int, default=100)
            p.add_argument("--tot-breadth", type=int, default=5)
            p.add_argument("--tot-value-samples", type=int, default=3)
        if name == "run-scheduling-outcome-prm":
            p.add_argument("--source-run-dir", type=Path, required=True)
            p.add_argument("--train-count", type=int, default=200)
            p.add_argument("--validation-count", type=int, default=50)
            p.add_argument("--test-count", type=int, default=100)
            p.add_argument("--job-count", type=int, default=6)
            p.add_argument("--machine-count", type=int, default=3)
            p.add_argument("--node-budget", type=int, default=80)
            p.add_argument("--action-budget", type=int, default=6)
            p.add_argument("--rank-epochs", type=int, default=25)
            p.add_argument("--max-pairs-per-task", type=int, default=80)
        if name == "diagnose-scheduling-policy":
            p.add_argument("--run-dir", type=Path, required=True)
            p.add_argument("--train-count", type=int, default=200)
            p.add_argument("--validation-count", type=int, default=50)
            p.add_argument("--test-count", type=int, default=100)
            p.add_argument("--job-count", type=int, default=6)
            p.add_argument("--machine-count", type=int, default=3)
            p.add_argument("--regret-threshold", type=float, default=5.0)
        if name == "run-scheduling-diagnosis":
            p.add_argument("--source-count", type=int, default=200)
            p.add_argument("--source-validation-count", type=int, default=50)
            p.add_argument("--target-count", type=int, default=100)
            p.add_argument("--job-count", type=int, default=6)
            p.add_argument("--machine-count", type=int, default=3)
            p.add_argument("--node-budget", type=int, default=120)
            p.add_argument("--action-budget", type=int, default=8)
            p.add_argument("--records-per-instance", type=int, default=60)
            p.add_argument("--rank-epochs", type=int, default=25)
            p.add_argument("--useful-action-recall", type=float, default=0.88)
            p.add_argument("--perturbation", default="mixed")
        if name in ("build-table4-q2", "build-table5-q2", "audit-table4-q2", "audit-table5-q2"):
            p.add_argument("--solver-prior-dir", type=Path, default=Path("runs/scheduling_llm_pilot_real_100_seed0"))
            p.add_argument("--expected-test-count", type=int, default=100)
            p.add_argument("--tot-dir", "--gpt-frontier-rank-dir", dest="gpt_frontier_rank_dir", type=Path, default=None)
            p.add_argument("--target-prm-dir", type=Path, default=None)
            p.add_argument("--direct-dir", type=Path, default=None)
            p.add_argument("--self-consistency-dir", type=Path, default=None)
            p.add_argument("--best-of-n-dir", type=Path, default=None)
            p.add_argument("--rest-mcts-dir", type=Path, default=None)
            p.add_argument("--solver-certification-dir", type=Path, default=None)
            p.add_argument("--structured-dir", type=Path, default=None)
        if name in ("build-table-q4-tau", "audit-table-q4-tau"):
            p.add_argument("--expected-test-count", type=int, default=100)
            p.add_argument("--native-dir", type=Path, default=None)
            p.add_argument("--best-of-n-dir", type=Path, default=None)
            p.add_argument("--value-guided-dir", type=Path, default=None)
            p.add_argument("--target-prm-dir", type=Path, default=None)
            p.add_argument("--solver-controller-dir", type=Path, default=None)
        if name == "build-q4-tau-adapter-contract":
            p.add_argument("--expected-test-count", type=int, default=100)
        if name == "run-table6-authority":
            p.add_argument("--train-count", type=int, default=200)
            p.add_argument("--validation-count", type=int, default=50)
            p.add_argument("--test-count", type=int, default=100)
            p.add_argument("--job-count", type=int, default=6)
            p.add_argument("--machine-count", type=int, default=3)
            p.add_argument("--node-budget", type=int, default=120)
            p.add_argument("--action-budget", type=int, default=8)
            p.add_argument("--records-per-instance", type=int, default=60)
            p.add_argument("--rank-epochs", type=int, default=25)
            p.add_argument("--corruption-seeds", default="0,1,2")
        if name == "audit-certification-tables":
            p.add_argument("--certification-dir", type=Path, default=None)
            p.add_argument("--authority-dir", type=Path, default=None)
            p.add_argument("--expected-test-count", type=int, default=100)
        if name == "audit-secret-safety":
            p.add_argument("--root", type=Path, default=Path("."))
        if name == "build-current-experiment-status":
            p.add_argument("--q2-readiness", type=Path, default=Path("runs/q2_table_current_seed0_protocol_v2_audit/table5_q2_readiness.csv"))
            p.add_argument("--q4-readiness", type=Path, default=Path("runs/q4_tau_table_current_audit/q4_tau_readiness.csv"))
            p.add_argument("--certification-readiness", type=Path, default=Path("runs/certification_tables_readiness/certification_tables_readiness.csv"))
            p.add_argument("--q2-implementation-matrix", type=Path, default=Path("runs/q2_implementation_matrix/q2_implementation_matrix.csv"))
            p.add_argument("--q2-local-primitives", type=Path, default=Path("runs/q2_local_primitive_audit/q2_local_primitive_audit.csv"))
            p.add_argument("--audit-command", default="PYTHON_BIN=.venv/bin/python scripts/run_paper_audits.sh all")
    args, rest = parser.parse_known_args()
    if args.command in (None, "run-smoke"):
        import sys

        sys.argv = [sys.argv[0], "--preset", "smoke", *rest]
        run_experiment_main()
        return
    if args.command in COMMANDS:
        if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite and not getattr(args, "resume", False):
            raise SystemExit(f"output directory exists and overwrite is false: {args.output_dir}")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        if args.command == "run-game24-offline":
            seeds = [int(part) for part in args.seeds.split(",") if part.strip()] if args.seeds else [args.seed]
            if len(seeds) > 1:
                summary_rows = []
                for seed in seeds:
                    seed_dir = args.output_dir / f"seed_{seed}"
                    metrics = run_game24_offline_experiment(seed_dir, limit=args.limit, seed=seed)
                    for policy, row in metrics["policies"].items():
                        summary_rows.append(
                            {
                                "seed": seed,
                                "Policy": policy,
                                "Success": row["success"],
                                "Nodes": row["nodes_mean"],
                                "Proposal calls": row["proposal_calls_mean"],
                                "frontier_top1_accuracy": row["frontier_top1_accuracy"],
                                "selection_regret_mean": row["selection_regret_mean"],
                            }
                        )
                _write_cli_csv(args.output_dir / "game24_multi_seed_summary.csv", summary_rows)
                manifest = {
                    "command": args.command,
                    "seeds": seeds,
                    "seed_count": len(seeds),
                    "limit": args.limit,
                    "output_dir": str(args.output_dir),
                    "overwrite": args.overwrite,
                    "status": "complete",
                    "llm_required": False,
                    "produced_artifacts": sorted(path.name for path in args.output_dir.iterdir()),
                }
                (args.output_dir / "command_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
                print(f"wrote {args.output_dir / 'game24_multi_seed_summary.csv'}")
                return
            metrics = run_game24_offline_experiment(args.output_dir, limit=args.limit, seed=seeds[0])
            manifest = {
                "command": args.command,
                "seed": seeds[0],
                "limit": args.limit,
                "output_dir": str(args.output_dir),
                "overwrite": args.overwrite,
                "status": "complete",
                "llm_required": metrics["llm_required"],
                "produced_artifacts": sorted(path.name for path in args.output_dir.iterdir() if path.is_file()),
            }
            (args.output_dir / "command_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
            print(f"wrote {args.output_dir / 'game24_offline_table.csv'}")
            return
        if args.command == "run-game24-llm-pilot":
            if args.mock:
                metrics = run_game24_llm_pilot(
                    args.output_dir,
                    client=ScriptedGame24Client(),
                    model=args.model or "mock-game24",
                    seed=args.seed,
                    start=args.start,
                    count=args.count,
                    proposal_k=args.proposal_k,
                    node_budget=args.node_budget,
                    batch_size=args.batch_size,
                )
            else:
                metrics = run_game24_llm_pilot_from_env(
                    args.output_dir,
                    model=args.model,
                    seed=args.seed,
                    start=args.start,
                    count=args.count,
                    proposal_k=args.proposal_k,
                    node_budget=args.node_budget,
                    timeout=args.timeout,
                    batch_size=args.batch_size,
                )
            manifest = {
                "command": args.command,
                "seed": args.seed,
                "model": metrics["model"],
                "start": args.start,
                "count": args.count,
                "proposal_k": args.proposal_k,
                "node_budget": args.node_budget,
                "batch_size": args.batch_size,
                "mock": args.mock,
                "output_dir": str(args.output_dir),
                "overwrite": args.overwrite,
                "status": "complete",
                "live_network_required": metrics["live_network_required"],
                "produced_artifacts": sorted(path.name for path in args.output_dir.iterdir() if path.is_file()),
            }
            (args.output_dir / "command_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
            print(f"wrote {args.output_dir / 'game24_llm_pilot_table.csv'}")
            return
        if args.command == "run-scheduling-micro":
            metrics = run_scheduling_micro_experiment(
                args.output_dir,
                seed=args.seed,
                train_count=args.train_count,
                validation_count=args.validation_count,
                test_count=args.test_count,
                job_count=args.job_count,
                machine_count=args.machine_count,
                node_budget=args.node_budget,
                action_budget=args.action_budget,
                records_per_instance=args.records_per_instance,
                shard_size=args.shard_size,
                rank_epochs=args.rank_epochs,
            )
            manifest = {
                "command": args.command,
                "seed": args.seed,
                "output_dir": str(args.output_dir),
                "overwrite": args.overwrite,
                "status": "complete",
                "splits": metrics["splits"],
                "trace_record_count": metrics["trace_record_count"],
                "train_pair_count": metrics["train_pair_count"],
                "validation_pair_count": metrics["validation_pair_count"],
                "produced_artifacts": sorted(path.name for path in args.output_dir.iterdir() if path.is_file()),
            }
            (args.output_dir / "command_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
            print(f"wrote {args.output_dir / 'scheduling_micro_table.csv'}")
            return
        if args.command == "run-scheduling-representation":
            metrics = run_scheduling_representation_ablation(
                args.output_dir,
                seed=args.seed,
                train_count=args.train_count,
                validation_count=args.validation_count,
                test_count=args.test_count,
                job_count=args.job_count,
                machine_count=args.machine_count,
                node_budget=args.node_budget,
                action_budget=args.action_budget,
                records_per_instance=args.records_per_instance,
                rank_epochs=args.rank_epochs,
            )
            manifest = {
                "command": args.command,
                "seed": args.seed,
                "output_dir": str(args.output_dir),
                "overwrite": args.overwrite,
                "status": "complete",
                "splits": metrics["splits"],
                "trace_record_count": metrics["trace_record_count"],
                "pair_counts": metrics["pair_counts"],
                "validation_ranking_errors": metrics["validation_ranking_errors"],
                "produced_artifacts": sorted(path.name for path in args.output_dir.iterdir() if path.is_file()),
            }
            (args.output_dir / "command_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
            print(f"wrote {args.output_dir / 'scheduling_representation_table.csv'}")
            return
        if args.command == "run-scheduling-transfer":
            metrics = run_scheduling_zero_shot_transfer(
                args.output_dir,
                seed=args.seed,
                train_count=args.train_count,
                validation_count=args.validation_count,
                test_count=args.test_count,
                job_count=args.job_count,
                machine_count=args.machine_count,
                node_budget=args.node_budget,
                action_budget=args.action_budget,
                records_per_instance=args.records_per_instance,
                rank_epochs=args.rank_epochs,
                useful_action_recall=args.useful_action_recall,
                perturbation=args.perturbation,
                batch_size=args.batch_size,
                parallel_episodes=args.parallel_episodes,
                resume=args.resume,
            )
            manifest = {
                "command": args.command,
                "seed": args.seed,
                "output_dir": str(args.output_dir),
                "overwrite": args.overwrite,
                "status": "complete",
                "batch_size": args.batch_size,
                "parallel_episodes": args.parallel_episodes,
                "resume": args.resume,
                "training_source": metrics["training_source"],
                "target_training_trace_count": metrics["target_training_trace_count"],
                "splits": metrics["splits"],
                "train_pair_count": metrics["train_pair_count"],
                "validation_pair_count": metrics["validation_pair_count"],
                "produced_artifacts": sorted(path.name for path in args.output_dir.iterdir() if path.is_file()),
            }
            (args.output_dir / "command_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
            print(f"wrote {args.output_dir / 'scheduling_transfer_table.csv'}")
            return
        if args.command == "run-scheduling-fewshot":
            fewshot_counts = tuple(int(part) for part in args.fewshot_counts.split(",") if part.strip())
            metrics = run_scheduling_fewshot_curve(
                args.output_dir,
                seed=args.seed,
                solver_train_count=args.solver_train_count,
                target_train_count=args.target_train_count,
                validation_count=args.validation_count,
                target_test_count=args.target_test_count,
                job_count=args.job_count,
                machine_count=args.machine_count,
                node_budget=args.node_budget,
                action_budget=args.action_budget,
                records_per_instance=args.records_per_instance,
                rank_epochs=args.rank_epochs,
                fewshot_counts=fewshot_counts,
                useful_action_recall=args.useful_action_recall,
                perturbation=args.perturbation,
            )
            manifest = {
                "command": args.command,
                "seed": args.seed,
                "output_dir": str(args.output_dir),
                "overwrite": args.overwrite,
                "status": "complete",
                "solver_train_count": metrics["solver_train_count"],
                "target_train_count": metrics["target_train_count"],
                "target_test_count": metrics["target_test_count"],
                "fewshot_counts_requested": metrics["fewshot_counts_requested"],
                "target_train_eval_overlap": metrics["target_train_eval_overlap"],
                "solver_pair_count": metrics["solver_pair_count"],
                "target_trace_record_count": metrics["target_trace_record_count"],
                "produced_artifacts": sorted(path.name for path in args.output_dir.iterdir() if path.is_file()),
            }
            (args.output_dir / "command_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
            print(f"wrote {args.output_dir / 'scheduling_fewshot_curve.csv'}")
            return
        if args.command == "run-scheduling-certification":
            metrics = run_scheduling_certification_experiment(
                args.output_dir,
                seed=args.seed,
                train_count=args.train_count,
                validation_count=args.validation_count,
                test_count=args.test_count,
                job_count=args.job_count,
                machine_count=args.machine_count,
                node_budget=args.node_budget,
                action_budget=args.action_budget,
                records_per_instance=args.records_per_instance,
                rank_epochs=args.rank_epochs,
                useful_action_recall=args.useful_action_recall,
                perturbation=args.perturbation,
            )
            manifest = {
                "command": args.command,
                "seed": args.seed,
                "output_dir": str(args.output_dir),
                "overwrite": args.overwrite,
                "status": "complete",
                "training_source": metrics["training_source"],
                "splits": metrics["splits"],
                "train_pair_count": metrics["train_pair_count"],
                "validation_pair_count": metrics["validation_pair_count"],
                "certification_rule": metrics["certification_rule"],
                "produced_artifacts": sorted(path.name for path in args.output_dir.iterdir() if path.is_file()),
            }
            (args.output_dir / "command_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
            print(f"wrote {args.output_dir / 'scheduling_certification_table.csv'}")
            return
        if args.command == "run-scheduling-llm-pilot":
            if args.mock:
                metrics = run_scheduling_llm_pilot(
                    args.output_dir,
                    client=ScriptedSchedulingClient(model=args.model or "mock-scheduling"),
                    model=args.model or "mock-scheduling",
                    seed=args.seed,
                    train_count=args.train_count,
                    validation_count=args.validation_count,
                    test_count=args.test_count,
                    job_count=args.job_count,
                    machine_count=args.machine_count,
                    node_budget=args.node_budget,
                    action_budget=args.action_budget,
                    records_per_instance=args.records_per_instance,
                    rank_epochs=args.rank_epochs,
                    perturbation=args.perturbation,
                    batch_size=args.batch_size,
                    parallel_episodes=args.parallel_episodes,
                    resume=args.resume,
                    include_gpt_value=args.include_gpt_value,
                    include_gpt_frontier_ranking=args.include_gpt_frontier_ranking,
                    include_target_prm=args.include_target_prm,
                    include_rest_mcts=args.include_rest_mcts,
                    include_solver_certification=args.include_solver_certification,
                    target_prm_train_count=args.target_prm_train_count,
                )
            else:
                metrics = run_scheduling_llm_pilot_from_env(
                    args.output_dir,
                    model=args.model,
                    seed=args.seed,
                    train_count=args.train_count,
                    validation_count=args.validation_count,
                    test_count=args.test_count,
                    job_count=args.job_count,
                    machine_count=args.machine_count,
                    node_budget=args.node_budget,
                    action_budget=args.action_budget,
                    records_per_instance=args.records_per_instance,
                    rank_epochs=args.rank_epochs,
                    timeout=args.timeout,
                    perturbation=args.perturbation,
                    batch_size=args.batch_size,
                    parallel_episodes=args.parallel_episodes,
                    resume=args.resume,
                    include_gpt_value=args.include_gpt_value,
                    include_gpt_frontier_ranking=args.include_gpt_frontier_ranking,
                    include_target_prm=args.include_target_prm,
                    include_rest_mcts=args.include_rest_mcts,
                    include_solver_certification=args.include_solver_certification,
                    target_prm_train_count=args.target_prm_train_count,
                )
            manifest = {
                "command": args.command,
                "seed": args.seed,
                "model": metrics["model"],
                "output_dir": str(args.output_dir),
                "overwrite": args.overwrite,
                "status": "complete",
                "batch_size": args.batch_size,
                "parallel_episodes": args.parallel_episodes,
                "resume": args.resume,
                "mock": args.mock,
                "include_gpt_value": args.include_gpt_value,
                "include_gpt_frontier_ranking": args.include_gpt_frontier_ranking,
                "include_target_prm": args.include_target_prm,
                "include_rest_mcts": args.include_rest_mcts,
                "include_solver_certification": args.include_solver_certification,
                "target_prm_train_count": metrics.get("target_prm_train_count", 0),
                "target_prm_pair_count": metrics.get("target_prm_pair_count", 0),
                "live_network_required": metrics["live_network_required"],
                "training_source": metrics["training_source"],
                "target_training_trace_count": metrics["target_training_trace_count"],
                "splits": metrics["splits"],
                "shared_proposal_replay": metrics["shared_proposal_replay"],
                "produced_artifacts": sorted(path.name for path in args.output_dir.iterdir() if path.is_file()),
            }
            (args.output_dir / "command_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
            print(f"wrote {args.output_dir / 'scheduling_llm_pilot_table.csv'}")
            return
        if args.command == "run-scheduling-native-q2":
            methods = tuple(part.strip() for part in args.methods.split(",") if part.strip())
            if args.mock:
                metrics = run_scheduling_native_q2(
                    args.output_dir,
                    client=ScriptedSchedulingClient(model=args.model or "mock-scheduling"),
                    model=args.model or "mock-scheduling",
                    seed=args.seed,
                    train_count=args.train_count,
                    validation_count=args.validation_count,
                    test_count=args.test_count,
                    job_count=args.job_count,
                    machine_count=args.machine_count,
                    methods=methods,
                    sample_count=args.sample_count,
                    parallel_episodes=args.parallel_episodes,
                    resume=args.resume,
                )
            else:
                metrics = run_scheduling_native_q2_from_env(
                    args.output_dir,
                    model=args.model,
                    seed=args.seed,
                    train_count=args.train_count,
                    validation_count=args.validation_count,
                    test_count=args.test_count,
                    job_count=args.job_count,
                    machine_count=args.machine_count,
                    timeout=args.timeout,
                    methods=methods,
                    sample_count=args.sample_count,
                    parallel_episodes=args.parallel_episodes,
                    resume=args.resume,
                )
            manifest = {
                "command": args.command,
                "seed": args.seed,
                "model": metrics["model"],
                "output_dir": str(args.output_dir),
                "overwrite": args.overwrite,
                "status": "complete",
                "methods": list(methods),
                "sample_count": args.sample_count,
                "parallel_episodes": args.parallel_episodes,
                "resume": args.resume,
                "mock": args.mock,
                "live_network_required": metrics["live_network_required"],
                "oracle_labels_in_inference_features": False,
                "produced_artifacts": sorted(path.name for path in args.output_dir.iterdir()),
            }
            (args.output_dir / "command_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
            print(f"wrote {args.output_dir / 'scheduling_native_q2_table.csv'}")
            return
        if args.command == "run-scheduling-structured-q2":
            metrics = run_scheduling_structured_q2(
                args.output_dir,
                seed=args.seed,
                train_count=args.train_count,
                validation_count=args.validation_count,
                test_count=args.test_count,
                job_count=args.job_count,
                machine_count=args.machine_count,
                cpsat_time_limit_seconds=args.cpsat_time_limit_seconds,
            )
            manifest = {
                "command": args.command,
                "seed": args.seed,
                "output_dir": str(args.output_dir),
                "overwrite": args.overwrite,
                "status": "complete",
                "benchmark": metrics["benchmark"],
                "structured_input": True,
                "llm_calls": 0,
                "tokens": 0,
                "produced_artifacts": sorted(path.name for path in args.output_dir.iterdir() if path.is_file()),
            }
            (args.output_dir / "command_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
            print(f"wrote {args.output_dir / 'scheduling_structured_q2_table.csv'}")
            return
        if args.command == "run-scheduling-tot-q2":
            if args.mock:
                metrics = run_scheduling_tot_q2(
                    args.output_dir,
                    client=ScriptedSchedulingClient(model=args.model or "mock-scheduling"),
                    model=args.model or "mock-scheduling",
                    seed=args.seed,
                    train_count=args.train_count,
                    validation_count=args.validation_count,
                    test_count=args.test_count,
                    job_count=args.job_count,
                    machine_count=args.machine_count,
                    node_budget=args.node_budget,
                    action_budget=args.action_budget,
                    breadth=args.breadth,
                    value_samples=args.value_samples,
                    perturbation=args.perturbation,
                    parallel_episodes=args.parallel_episodes,
                    resume=args.resume,
                )
            else:
                metrics = run_scheduling_tot_q2_from_env(
                    args.output_dir,
                    model=args.model,
                    seed=args.seed,
                    train_count=args.train_count,
                    validation_count=args.validation_count,
                    test_count=args.test_count,
                    job_count=args.job_count,
                    machine_count=args.machine_count,
                    node_budget=args.node_budget,
                    action_budget=args.action_budget,
                    breadth=args.breadth,
                    value_samples=args.value_samples,
                    timeout=args.timeout,
                    perturbation=args.perturbation,
                    parallel_episodes=args.parallel_episodes,
                    resume=args.resume,
                )
            manifest = {
                "command": args.command,
                "seed": args.seed,
                "model": metrics["model"],
                "output_dir": str(args.output_dir),
                "overwrite": args.overwrite,
                "status": "complete",
                "breadth": args.breadth,
                "value_samples": args.value_samples,
                "parallel_episodes": args.parallel_episodes,
                "resume": args.resume,
                "mock": args.mock,
                "live_network_required": metrics["live_network_required"],
                "native_tot_protocol": metrics["native_tot_protocol"],
                "produced_artifacts": sorted(path.name for path in args.output_dir.iterdir() if path.is_file()),
            }
            (args.output_dir / "command_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
            print(f"wrote {args.output_dir / 'scheduling_tot_q2_table.csv'}")
            return
        if args.command == "run-scheduling-outcome-prm":
            metrics = run_scheduling_outcome_prm(
                source_run_dir=args.source_run_dir,
                out_dir=args.output_dir,
                seed=args.seed,
                train_count=args.train_count,
                validation_count=args.validation_count,
                test_count=args.test_count,
                job_count=args.job_count,
                machine_count=args.machine_count,
                node_budget=args.node_budget,
                action_budget=args.action_budget,
                rank_epochs=args.rank_epochs,
                max_pairs_per_task=args.max_pairs_per_task,
            )
            manifest = {
                "command": args.command,
                "seed": args.seed,
                "source_run_dir": str(args.source_run_dir),
                "output_dir": str(args.output_dir),
                "overwrite": args.overwrite,
                "status": "complete",
                "benchmark": metrics["benchmark"],
                "outcome_pair_count": metrics["outcome_pair_count"],
                "shared_cached_proposal_replay": True,
                "produced_artifacts": sorted(path.name for path in args.output_dir.iterdir() if path.is_file()),
            }
            (args.output_dir / "command_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
            print(f"wrote {args.output_dir / 'outcome_prm_table.csv'}")
            return
        if args.command == "diagnose-scheduling-policy":
            metrics = run_scheduling_policy_diagnosis(
                run_dir=args.run_dir,
                out_dir=args.output_dir,
                seed=args.seed,
                train_count=args.train_count,
                validation_count=args.validation_count,
                test_count=args.test_count,
                job_count=args.job_count,
                machine_count=args.machine_count,
                regret_threshold=args.regret_threshold,
            )
            manifest = {
                "command": args.command,
                "seed": args.seed,
                "run_dir": str(args.run_dir),
                "output_dir": str(args.output_dir),
                "overwrite": args.overwrite,
                "status": "complete",
                "benchmark": metrics["benchmark"],
                "decision_count": metrics["decision_count"],
                "produced_artifacts": sorted(path.name for path in args.output_dir.iterdir() if path.is_file()),
            }
            (args.output_dir / "command_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
            print(f"wrote {args.output_dir / 'policy_diagnosis_summary.csv'}")
            return
        if args.command == "preflight-scheduling-llm-pilot":
            preflight = preflight_scheduling_llm_pilot(
                args.output_dir,
                seed=args.seed,
                train_count=args.train_count,
                validation_count=args.validation_count,
                test_count=args.test_count,
                job_count=args.job_count,
                machine_count=args.machine_count,
                node_budget=args.node_budget,
                action_budget=args.action_budget,
                model=args.model,
                batch_size=args.batch_size,
                parallel_episodes=args.parallel_episodes,
                perturbation=args.perturbation,
                include_gpt_value=args.include_gpt_value,
                include_gpt_frontier_ranking=args.include_gpt_frontier_ranking,
                include_target_prm=args.include_target_prm,
                include_rest_mcts=args.include_rest_mcts,
                include_solver_certification=args.include_solver_certification,
                target_prm_train_count=args.target_prm_train_count,
            )
            print(
                "preflight "
                f"api_key_present={preflight['api_key_present']} "
                f"remaining={preflight['remaining_episode_count']}/{preflight['test_episode_count']} "
                f"worst_case_requests={preflight['worst_case_request_count_upper_bound']} "
                f"wrote {args.output_dir / 'scheduling_llm_pilot_preflight.json'}"
            )
            return
        if args.command == "run-scheduling-diagnosis":
            metrics = run_scheduling_transfer_diagnosis(
                args.output_dir,
                seed=args.seed,
                source_count=args.source_count,
                source_validation_count=args.source_validation_count,
                target_count=args.target_count,
                job_count=args.job_count,
                machine_count=args.machine_count,
                node_budget=args.node_budget,
                action_budget=args.action_budget,
                records_per_instance=args.records_per_instance,
                rank_epochs=args.rank_epochs,
                useful_action_recall=args.useful_action_recall,
                perturbation=args.perturbation,
            )
            manifest = {
                "command": args.command,
                "seed": args.seed,
                "output_dir": str(args.output_dir),
                "overwrite": args.overwrite,
                "status": "complete",
                "source_record_count": metrics["source_record_count"],
                "target_record_count": metrics["target_record_count"],
                "domain_classifier": metrics["domain_classifier"],
                "ranking_transfer": metrics["ranking_transfer"],
                "produced_artifacts": sorted(path.name for path in args.output_dir.iterdir() if path.is_file()),
            }
            (args.output_dir / "command_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
            print(f"wrote {args.output_dir / 'scheduling_transfer_diagnosis_metrics.json'}")
            return
        if args.command in ("build-table4-q2", "build-table5-q2"):
            manifest = build_table5_q2_results(
                args.output_dir,
                solver_prior_dir=args.solver_prior_dir,
                expected_test_count=args.expected_test_count,
                gpt_frontier_rank_dir=args.gpt_frontier_rank_dir,
                target_prm_dir=args.target_prm_dir,
                direct_dir=args.direct_dir,
                self_consistency_dir=args.self_consistency_dir,
                best_of_n_dir=args.best_of_n_dir,
                rest_mcts_dir=args.rest_mcts_dir,
                solver_certification_dir=args.solver_certification_dir,
                structured_dir=args.structured_dir,
            )
            command_manifest = {
                "command": args.command,
                "output_dir": str(args.output_dir),
                "overwrite": args.overwrite,
                "status": "complete",
                "expected_test_count": args.expected_test_count,
                "input_dirs": {
                    "structured_dir": None if args.structured_dir is None else str(args.structured_dir),
                    "direct_dir": None if args.direct_dir is None else str(args.direct_dir),
                    "self_consistency_dir": None if args.self_consistency_dir is None else str(args.self_consistency_dir),
                    "best_of_n_dir": None if args.best_of_n_dir is None else str(args.best_of_n_dir),
                    "tot_dir": None if args.gpt_frontier_rank_dir is None else str(args.gpt_frontier_rank_dir),
                    "target_prm_dir": None if args.target_prm_dir is None else str(args.target_prm_dir),
                    "rest_mcts_dir": None if args.rest_mcts_dir is None else str(args.rest_mcts_dir),
                    "solver_prior_dir": str(args.solver_prior_dir),
                    "solver_certification_dir": None if args.solver_certification_dir is None else str(args.solver_certification_dir),
                },
                "produced_artifacts": manifest["produced_artifacts"],
            }
            (args.output_dir / "command_manifest.json").write_text(json.dumps(command_manifest, indent=2, sort_keys=True), encoding="utf-8")
            print(f"wrote {args.output_dir / 'table5_q2_results.csv'}")
            return
        if args.command in ("audit-table4-q2", "audit-table5-q2"):
            manifest = audit_table5_q2_readiness(
                args.output_dir,
                solver_prior_dir=args.solver_prior_dir,
                expected_test_count=args.expected_test_count,
                gpt_frontier_rank_dir=args.gpt_frontier_rank_dir,
                target_prm_dir=args.target_prm_dir,
                direct_dir=args.direct_dir,
                self_consistency_dir=args.self_consistency_dir,
                best_of_n_dir=args.best_of_n_dir,
                rest_mcts_dir=args.rest_mcts_dir,
                solver_certification_dir=args.solver_certification_dir,
                structured_dir=args.structured_dir,
            )
            command_manifest = {
                "command": args.command,
                "output_dir": str(args.output_dir),
                "overwrite": args.overwrite,
                "status": "complete",
                "expected_test_count": args.expected_test_count,
                "input_dirs": {
                    "structured_dir": None if args.structured_dir is None else str(args.structured_dir),
                    "direct_dir": None if args.direct_dir is None else str(args.direct_dir),
                    "self_consistency_dir": None if args.self_consistency_dir is None else str(args.self_consistency_dir),
                    "best_of_n_dir": None if args.best_of_n_dir is None else str(args.best_of_n_dir),
                    "tot_dir": None if args.gpt_frontier_rank_dir is None else str(args.gpt_frontier_rank_dir),
                    "target_prm_dir": None if args.target_prm_dir is None else str(args.target_prm_dir),
                    "rest_mcts_dir": None if args.rest_mcts_dir is None else str(args.rest_mcts_dir),
                    "solver_prior_dir": str(args.solver_prior_dir),
                    "solver_certification_dir": None if args.solver_certification_dir is None else str(args.solver_certification_dir),
                },
                "all_rows_result_ready": manifest["readiness"]["all_rows_result_ready"],
                "all_rows_compute_ready": manifest["readiness"]["all_rows_compute_ready"],
                "produced_artifacts": manifest["readiness"]["produced_artifacts"],
            }
            (args.output_dir / "command_manifest.json").write_text(json.dumps(command_manifest, indent=2, sort_keys=True), encoding="utf-8")
            print(f"wrote {args.output_dir / 'table5_q2_readiness.csv'}")
            return
        if args.command == "build-table-q4-tau":
            manifest = build_q4_tau_results(
                args.output_dir,
                expected_test_count=args.expected_test_count,
                native_dir=args.native_dir,
                best_of_n_dir=args.best_of_n_dir,
                value_guided_dir=args.value_guided_dir,
                target_prm_dir=args.target_prm_dir,
                solver_controller_dir=args.solver_controller_dir,
            )
            command_manifest = {
                "command": args.command,
                "output_dir": str(args.output_dir),
                "overwrite": args.overwrite,
                "status": "complete",
                "expected_test_count": args.expected_test_count,
                "produced_artifacts": manifest["produced_artifacts"],
            }
            (args.output_dir / "command_manifest.json").write_text(json.dumps(command_manifest, indent=2, sort_keys=True), encoding="utf-8")
            print(f"wrote {args.output_dir / 'q4_tau_results.csv'}")
            return
        if args.command == "audit-table-q4-tau":
            manifest = audit_q4_tau_readiness(
                args.output_dir,
                expected_test_count=args.expected_test_count,
                native_dir=args.native_dir,
                best_of_n_dir=args.best_of_n_dir,
                value_guided_dir=args.value_guided_dir,
                target_prm_dir=args.target_prm_dir,
                solver_controller_dir=args.solver_controller_dir,
            )
            command_manifest = {
                "command": args.command,
                "output_dir": str(args.output_dir),
                "overwrite": args.overwrite,
                "status": "complete",
                "expected_test_count": args.expected_test_count,
                "all_rows_result_ready": manifest["readiness"]["all_rows_result_ready"],
                "produced_artifacts": manifest["readiness"]["produced_artifacts"],
            }
            (args.output_dir / "command_manifest.json").write_text(json.dumps(command_manifest, indent=2, sort_keys=True), encoding="utf-8")
            print(f"wrote {args.output_dir / 'q4_tau_readiness.csv'}")
            return
        if args.command == "audit-q2-source-evidence":
            manifest = audit_q2_source_evidence(args.output_dir)
            command_manifest = {
                "command": args.command,
                "output_dir": str(args.output_dir),
                "overwrite": args.overwrite,
                "status": "complete",
                "all_external_sources_ready": manifest["all_external_sources_ready"],
                "produced_artifacts": manifest["produced_artifacts"],
            }
            (args.output_dir / "command_manifest.json").write_text(json.dumps(command_manifest, indent=2, sort_keys=True), encoding="utf-8")
            print(f"wrote {args.output_dir / 'q2_source_evidence_audit.csv'}")
            return
        if args.command == "verify-q2-source-refs":
            manifest = verify_q2_source_refs(args.output_dir, online=args.online, timeout=args.timeout)
            command_manifest = {
                "command": args.command,
                "output_dir": str(args.output_dir),
                "overwrite": args.overwrite,
                "status": "complete",
                "online_checked": manifest["online_checked"],
                "all_static_refs_ready": manifest["all_static_refs_ready"],
                "all_online_refs_ready": manifest["all_online_refs_ready"],
                "produced_artifacts": manifest["produced_artifacts"],
            }
            (args.output_dir / "command_manifest.json").write_text(json.dumps(command_manifest, indent=2, sort_keys=True), encoding="utf-8")
            print(f"wrote {args.output_dir / 'q2_source_ref_verification.csv'}")
            return
        if args.command == "audit-q4-tau-source-evidence":
            manifest = audit_q4_tau_source_evidence(args.output_dir)
            command_manifest = {
                "command": args.command,
                "output_dir": str(args.output_dir),
                "overwrite": args.overwrite,
                "status": "complete",
                "all_sources_ready": manifest["all_sources_ready"],
                "produced_artifacts": manifest["produced_artifacts"],
            }
            (args.output_dir / "command_manifest.json").write_text(json.dumps(command_manifest, indent=2, sort_keys=True), encoding="utf-8")
            print(f"wrote {args.output_dir / 'q4_tau_source_evidence_audit.csv'}")
            return
        if args.command == "verify-q4-tau-source-refs":
            manifest = verify_q4_tau_source_refs(args.output_dir, online=args.online, timeout=args.timeout)
            command_manifest = {
                "command": args.command,
                "output_dir": str(args.output_dir),
                "overwrite": args.overwrite,
                "status": "complete",
                "online_checked": manifest["online_checked"],
                "all_static_refs_ready": manifest["all_static_refs_ready"],
                "all_online_refs_ready": manifest["all_online_refs_ready"],
                "produced_artifacts": manifest["produced_artifacts"],
            }
            (args.output_dir / "command_manifest.json").write_text(json.dumps(command_manifest, indent=2, sort_keys=True), encoding="utf-8")
            print(f"wrote {args.output_dir / 'q4_tau_source_ref_verification.csv'}")
            return
        if args.command == "build-q4-tau-adapter-contract":
            manifest = build_q4_tau_adapter_contract(args.output_dir, expected_test_count=args.expected_test_count)
            command_manifest = {
                "command": args.command,
                "output_dir": str(args.output_dir),
                "overwrite": args.overwrite,
                "status": "complete",
                "expected_test_count": args.expected_test_count,
                "produced_artifacts": manifest["produced_artifacts"],
            }
            (args.output_dir / "command_manifest.json").write_text(json.dumps(command_manifest, indent=2, sort_keys=True), encoding="utf-8")
            print(f"wrote {args.output_dir / 'q4_tau_adapter_contract.csv'}")
            return
        if args.command == "build-q2-implementation-matrix":
            manifest = build_q2_implementation_matrix(args.output_dir)
            command_manifest = {
                "command": args.command,
                "output_dir": str(args.output_dir),
                "overwrite": args.overwrite,
                "status": "complete",
                "all_table_rows_covered": manifest["all_table_rows_covered"],
                "produced_artifacts": manifest["produced_artifacts"],
            }
            (args.output_dir / "command_manifest.json").write_text(json.dumps(command_manifest, indent=2, sort_keys=True), encoding="utf-8")
            print(f"wrote {args.output_dir / 'q2_implementation_matrix.csv'}")
            return
        if args.command == "audit-q2-local-primitives":
            manifest = audit_q2_local_primitives(args.output_dir)
            command_manifest = {
                "command": args.command,
                "output_dir": str(args.output_dir),
                "overwrite": args.overwrite,
                "status": "complete",
                "all_local_primitives_ready": manifest["all_local_primitives_ready"],
                "produced_artifacts": manifest["produced_artifacts"],
            }
            (args.output_dir / "command_manifest.json").write_text(json.dumps(command_manifest, indent=2, sort_keys=True), encoding="utf-8")
            print(f"wrote {args.output_dir / 'q2_local_primitive_audit.csv'}")
            return
        if args.command == "build-q2-remaining-runs":
            manifest = build_q2_remaining_runs(
                args.output_dir,
                q2_readiness=args.q2_readiness,
                certification_readiness=args.certification_readiness,
            )
            command_manifest = {
                "command": args.command,
                "output_dir": str(args.output_dir),
                "overwrite": args.overwrite,
                "status": "complete",
                "q2_readiness": str(args.q2_readiness),
                "certification_readiness": str(args.certification_readiness),
                "remaining_group_count": manifest["remaining_group_count"],
                "remaining_method_count": manifest["remaining_method_count"],
                "produced_artifacts": manifest["produced_artifacts"],
            }
            (args.output_dir / "command_manifest.json").write_text(json.dumps(command_manifest, indent=2, sort_keys=True), encoding="utf-8")
            print(f"wrote {args.output_dir / 'q2_remaining_runs.csv'}")
            return
        if args.command == "build-q2-run-preflight":
            manifest = build_q2_run_preflight(
                args.output_dir,
                seed=args.seed,
                train_count=args.train_count,
                validation_count=args.validation_count,
                test_count=args.test_count,
                job_count=args.job_count,
                machine_count=args.machine_count,
                node_budget=args.node_budget,
                action_budget=args.action_budget,
                records_per_instance=args.records_per_instance,
                rank_epochs=args.rank_epochs,
                model=args.model,
                perturbation=args.perturbation,
                native_dir=args.native_dir,
                learned_dir=args.learned_dir,
                tot_dir=args.tot_dir,
                sample_count=args.sample_count,
                native_parallel_episodes=args.native_parallel_episodes,
                search_parallel_episodes=args.search_parallel_episodes,
                tot_parallel_episodes=args.tot_parallel_episodes,
                batch_size=args.batch_size,
                target_prm_train_count=args.target_prm_train_count,
                tot_breadth=args.tot_breadth,
                tot_value_samples=args.tot_value_samples,
            )
            command_manifest = {
                "command": args.command,
                "output_dir": str(args.output_dir),
                "overwrite": args.overwrite,
                "status": "complete",
                "test_count": args.test_count,
                "total_remaining_units": manifest["total_remaining_units"],
                "total_worst_case_requests": manifest["total_worst_case_requests"],
                "api_key_present": manifest["api_key_present"],
                "produced_artifacts": manifest["produced_artifacts"],
            }
            (args.output_dir / "command_manifest.json").write_text(json.dumps(command_manifest, indent=2, sort_keys=True), encoding="utf-8")
            print(f"wrote {args.output_dir / 'q2_run_preflight.csv'}")
            return
        if args.command == "run-table6-authority":
            corruption_seeds = tuple(int(part) for part in args.corruption_seeds.split(",") if part.strip())
            metrics = run_table6_authority_experiment(
                args.output_dir,
                seed=args.seed,
                train_count=args.train_count,
                validation_count=args.validation_count,
                test_count=args.test_count,
                job_count=args.job_count,
                machine_count=args.machine_count,
                node_budget=args.node_budget,
                action_budget=args.action_budget,
                records_per_instance=args.records_per_instance,
                rank_epochs=args.rank_epochs,
                corruption_seeds=corruption_seeds,
            )
            manifest = {
                "command": args.command,
                "seed": args.seed,
                "output_dir": str(args.output_dir),
                "overwrite": args.overwrite,
                "status": "complete",
                "corruption_seeds": list(corruption_seeds),
                "benchmark": metrics["benchmark"],
                "oracle_labels_in_inference_features": False,
                "produced_artifacts": sorted(path.name for path in args.output_dir.iterdir() if path.is_file()),
            }
            (args.output_dir / "command_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
            print(f"wrote {args.output_dir / 'table6_authority.csv'}")
            return
        if args.command == "audit-certification-tables":
            manifest = audit_certification_tables(
                args.output_dir,
                certification_dir=args.certification_dir,
                authority_dir=args.authority_dir,
                expected_test_count=args.expected_test_count,
            )
            command_manifest = {
                "command": args.command,
                "output_dir": str(args.output_dir),
                "overwrite": args.overwrite,
                "expected_test_count": args.expected_test_count,
                "certification_dir": None if args.certification_dir is None else str(args.certification_dir),
                "authority_dir": None if args.authority_dir is None else str(args.authority_dir),
                "all_rows_ready": manifest["all_rows_ready"],
                "produced_artifacts": manifest["produced_artifacts"],
            }
            (args.output_dir / "command_manifest.json").write_text(json.dumps(command_manifest, indent=2, sort_keys=True), encoding="utf-8")
            print(f"wrote {args.output_dir / 'certification_tables_readiness.csv'}")
            return
        if args.command == "audit-secret-safety":
            manifest = audit_secret_safety(args.output_dir, root=args.root)
            command_manifest = {
                "command": args.command,
                "output_dir": str(args.output_dir),
                "overwrite": args.overwrite,
                "status": "complete",
                "root": str(args.root),
                "secrets_ready": manifest["secrets_ready"],
                "finding_count": manifest["finding_count"],
                "produced_artifacts": manifest["produced_artifacts"],
            }
            (args.output_dir / "command_manifest.json").write_text(json.dumps(command_manifest, indent=2, sort_keys=True), encoding="utf-8")
            print(f"wrote {args.output_dir / 'secret_safety_audit.csv'}")
            if not manifest["secrets_ready"]:
                raise SystemExit(f"secret safety audit failed: {manifest['finding_count']} finding(s)")
            return
        if args.command == "build-current-experiment-status":
            manifest = build_current_experiment_status(
                args.output_dir,
                q2_readiness=args.q2_readiness,
                q4_readiness=args.q4_readiness,
                certification_readiness=args.certification_readiness,
                q2_implementation_matrix=args.q2_implementation_matrix,
                q2_local_primitives=args.q2_local_primitives,
                audit_command=args.audit_command,
            )
            command_manifest = {
                "command": args.command,
                "output_dir": str(args.output_dir),
                "overwrite": args.overwrite,
                "status": "complete",
                "produced_artifacts": manifest["produced_artifacts"],
                "input_artifacts": manifest["input_artifacts"],
            }
            (args.output_dir / "command_manifest.json").write_text(json.dumps(command_manifest, indent=2, sort_keys=True), encoding="utf-8")
            print(f"wrote {args.output_dir / 'current_experiment_status.md'}")
            return
        if args.command == "run-transfer-matrix":
            seeds = [int(part) for part in (args.seeds or str(args.seed)).split(",") if part.strip()]
            aggregate_rows = []
            for seed in seeds:
                cfg = PRESETS[args.preset]
                cfg = replace(cfg, seed=seed)
                seed_dir = args.output_dir / f"seed_{seed}"
                metrics = run_experiment(cfg, seed_dir)
                for method, row in metrics["methods"].items():
                    aggregate_rows.append(
                        {
                            "seed": seed,
                            "method": method,
                            "success": row["success"],
                            "gap": row["gap"],
                            "nodes": row["nodes_mean"],
                            "proposal_calls": row["llm_calls_mean"],
                        }
                    )
            write_seed_aggregate(args.output_dir, aggregate_rows, "method", ["success", "gap", "nodes", "proposal_calls"])
            manifest = {
                "command": args.command,
                "preset": args.preset,
                "seeds": seeds,
                "seed_count": len(seeds),
                "output_dir": str(args.output_dir),
                "overwrite": args.overwrite,
                "status": "complete",
            }
            (args.output_dir / "command_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
            print(f"wrote {args.output_dir / 'multi_seed_summary.csv'}")
            return
        cfg = replace(PRESETS[args.preset], seed=args.seed)
        run_experiment(cfg, args.output_dir)
        produced = sorted(path.name for path in args.output_dir.iterdir() if path.is_file() and path.name != "command_manifest.json")
        manifest = {
            "command": args.command,
            "seed": args.seed,
            "preset": args.preset,
            "output_dir": str(args.output_dir),
            "overwrite": args.overwrite,
            "status": "complete",
            "execution_mode": "smoke_backed_experiment",
            "produced_artifacts": produced,
        }
        (args.output_dir / "command_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        print(f"wrote {args.output_dir / 'command_manifest.json'}")
        return
    raise SystemExit(f"unknown command: {args.command}")


def _write_cli_csv(path: Path, rows) -> None:
    import csv

    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
