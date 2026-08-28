from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping

from cgr_exp.q2_method_contracts import q2_method_contract


Q2_SOURCE_EVIDENCE: Dict[str, Dict[str, object]] = {
    "CP-SAT": {
        "source_url": "https://github.com/google/or-tools",
        "source_files": [
            "ortools/sat/docs/scheduling.md",
            "ortools/sat/python/cp_model.py",
        ],
        "official_source_anchors": [
            "scheduling.md: optional intervals are controlled by presence literals and ignored by NoOverlap when inactive",
            "cp_model API: NewOptionalIntervalVar / AddNoOverlap / Minimize",
        ],
        "official_primitives": [
            "integer variables for scheduling decisions",
            "interval variables",
            "NoOverlap machine-capacity constraints",
            "precedence constraints",
            "objective minimization",
        ],
        "local_promotion_gates": [
            "structured_input true",
            "llm_calls equals 0",
            "tokens equals 0",
        ],
    },
    "Direct / CoT": {
        "source_url": "https://arxiv.org/abs/2201.11903",
        "source_files": ["paper"],
        "official_source_anchors": [
            "paper method: sample one chain of intermediate reasoning steps before the final answer",
        ],
        "official_primitives": [
            "single chain-of-thought reasoning path",
            "intermediate natural-language rationale",
            "final answer extraction",
        ],
        "local_promotion_gates": [
            "selection_rule first_valid",
            "exactly one sample per held-out task",
            "oracle future labels excluded",
        ],
    },
    "Self-Consistency": {
        "source_url": "https://arxiv.org/abs/2203.11171",
        "source_files": ["paper"],
        "official_source_anchors": [
            "paper method: sample diverse reasoning paths and marginalize/aggregate final answers",
        ],
        "official_primitives": [
            "multiple independent reasoning paths",
            "final-answer aggregation",
            "consistency-based selection",
        ],
        "local_promotion_gates": [
            "selection_rule majority_job_order_then_objective",
            "multiple samples per held-out task",
            "not Best-of-N objective selection",
        ],
    },
    "Best-of-N": {
        "source_url": "https://arxiv.org/abs/2408.03314",
        "source_files": ["paper"],
        "official_source_anchors": [
            "test-time scaling primitive: draw N candidate outputs and select by reward/verifier score",
        ],
        "official_primitives": [
            "multiple complete answer samples",
            "verifier or reward scoring",
            "best sample selection",
        ],
        "local_promotion_gates": [
            "selection_rule best_verified_objective",
            "verifier objective selection disclosed",
            "declared sample_count coverage per held-out task",
        ],
    },
    "Tree of Thoughts": {
        "source_url": "https://github.com/princeton-nlp/tree-of-thought-llm",
        "source_files": ["src/tot/methods/bfs.py"],
        "official_source_anchors": [
            "README/run.py: ToT+BFS selected by method_generate/method_evaluate/method_select",
            "src/tot/methods/bfs.py: generation, evaluation, and greedy/sample selection loop",
        ],
        "official_primitives": [
            "method_generate",
            "method_evaluate",
            "method_select",
            "breadth-first search",
            "greedy top-b selection",
        ],
        "local_promotion_gates": [
            "native_tot_protocol declares official BFS entrypoints",
            "trace contains generate events",
            "trace contains evaluation and selection events",
            "trace contains value sample records",
            "one finish event per held-out rendering",
        ],
    },
    "Target-PRM priority": {
        "source_url": "https://github.com/openai/prm800k",
        "source_files": ["README.md", "data/*.jsonl"],
        "official_source_anchors": [
            "README: PRM800K is a process-supervision dataset with step-level correctness labels",
            "data schema: per-step completions receive ratings rather than only terminal labels",
        ],
        "official_primitives": [
            "intermediate-step labels",
            "process supervision",
            "reward/value model over partial solutions",
        ],
        "local_promotion_gates": [
            "target LLM-induced training trajectories",
            "target PRM ranker artifact",
            "target PRM trace and pair artifacts",
            "no training/evaluation task overlap",
            "not zero-shot",
        ],
    },
    "PRM-guided MCTS (ReST-MCTS-style)": {
        "source_url": "https://github.com/THUDM/ReST-MCTS",
        "source_files": ["MCTS/mcts.py", "MCTS/base.py", "README.md"],
        "official_source_anchors": [
            "README: MCTS mode uses propose_method, value_method, iteration_limit, and branch",
            "MCTS implementation: selection, expansion, simulation/evaluation, and backpropagation",
        ],
        "official_primitives": [
            "MCTS selection",
            "MCTS expansion",
            "MCTS simulation/evaluation",
            "MCTS backpropagation",
            "process reward guidance",
        ],
        "local_promotion_gates": [
            "target PRM training distribution",
            "mcts_components include selection",
            "mcts_components include expansion",
            "mcts_components include simulation/evaluation",
            "mcts_components include backpropagation",
            "full_iterative_self_training false unless fully implemented",
        ],
    },
}

Q2_EXTERNAL_SOURCE_METHODS = [
    "CP-SAT",
    "Direct / CoT",
    "Self-Consistency",
    "Best-of-N",
    "Tree of Thoughts",
    "Target-PRM priority",
    "PRM-guided MCTS (ReST-MCTS-style)",
]


def q2_source_evidence(method: str) -> Dict[str, object]:
    return dict(Q2_SOURCE_EVIDENCE[method])


def q2_source_evidence_methods() -> List[str]:
    return sorted(Q2_SOURCE_EVIDENCE)


def audit_q2_source_evidence(out_dir: Path) -> Dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [_audit_source_evidence_row(method) for method in Q2_EXTERNAL_SOURCE_METHODS]
    csv_path = out_dir / "q2_source_evidence_audit.csv"
    json_path = out_dir / "q2_source_evidence_audit.json"
    _write_csv(csv_path, rows)
    manifest = {
        "audit": "q2_source_evidence",
        "all_external_sources_ready": all(row["source_ready"] for row in rows),
        "external_methods": Q2_EXTERNAL_SOURCE_METHODS,
        "rows": rows,
        "produced_artifacts": [csv_path.name, json_path.name],
    }
    json_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def _audit_source_evidence_row(method: str) -> Dict[str, object]:
    blockers = []
    evidence = Q2_SOURCE_EVIDENCE.get(method)
    if not isinstance(evidence, Mapping):
        blockers.append("missing source evidence")
        evidence = {}
    contract = q2_method_contract(method)
    source_url = str(evidence.get("source_url", ""))
    contract_url = str(contract.get("source_url", ""))
    if not source_url:
        blockers.append("missing source_url")
    if source_url != contract_url:
        blockers.append("source_url mismatch with q2_method_contracts")
    source_files = [str(item) for item in evidence.get("source_files", [])] if isinstance(evidence.get("source_files"), list) else []
    if not source_files:
        blockers.append("missing source_files")
    official_primitives = [str(item) for item in evidence.get("official_primitives", [])] if isinstance(evidence.get("official_primitives"), list) else []
    if not official_primitives:
        blockers.append("missing official_primitives")
    source_anchors = [str(item) for item in evidence.get("official_source_anchors", [])] if isinstance(evidence.get("official_source_anchors"), list) else []
    if not source_anchors:
        blockers.append("missing official_source_anchors")
    contract_primitives = {str(item) for item in contract.get("source_primitives", [])}
    if official_primitives and contract_primitives and not (set(official_primitives) & contract_primitives):
        blockers.append("official_primitives do not intersect contract source_primitives")
    promotion_gates = [str(item) for item in evidence.get("local_promotion_gates", [])] if isinstance(evidence.get("local_promotion_gates"), list) else []
    if not promotion_gates:
        blockers.append("missing local_promotion_gates")
    return {
        "Method": method,
        "source_url": source_url,
        "contract_source_url": contract_url,
        "source_files": "; ".join(source_files),
        "official_source_anchors": "; ".join(source_anchors),
        "official_primitives": "; ".join(official_primitives),
        "local_promotion_gates": "; ".join(promotion_gates),
        "source_ready": not blockers,
        "blockers": "; ".join(blockers),
    }


def _write_csv(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
