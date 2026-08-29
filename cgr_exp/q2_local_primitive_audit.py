from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping


Q2_LOCAL_PRIMITIVE_CHECKS: Dict[str, Dict[str, object]] = {
    "CP-SAT": {
        "paths": ["cgr_exp/scheduling_structured_q2.py"],
        "anchors": ["NewOptionalIntervalVar", "AddNoOverlap", "Minimize", "llm_calls"],
    },
    "EST / list scheduling heuristic": {
        "paths": ["cgr_exp/scheduling_structured_q2.py"],
        "anchors": ["earliest_start_list_scheduling", "_earliest_start_list_schedule", "llm_calls"],
    },
    "Direct / CoT": {
        "paths": ["cgr_exp/scheduling_native_q2.py"],
        "anchors": ["first_valid", "generate_scheduling_solution", "reasoning"],
    },
    "Self-Consistency": {
        "paths": ["cgr_exp/scheduling_native_q2.py"],
        "anchors": ["majority_job_order_then_first_valid", "_select_sample", "sample_count"],
    },
    "Best-of-N": {
        "paths": ["cgr_exp/scheduling_native_q2.py"],
        "anchors": ["best_verified_objective", "_select_sample", "objective"],
    },
    "Tree of Thoughts": {
        "paths": ["cgr_exp/scheduling_tot_q2.py"],
        "anchors": [
            "method_generate",
            "method_evaluate",
            "method_select",
            "breadth-first frontier search",
            "tot_value_sample_count",
        ],
    },
    "Target-PRM priority": {
        "paths": ["cgr_exp/scheduling_transfer.py"],
        "anchors": [
            "target_prm_trace_records.jsonl",
            "target_prm_train_pairs.jsonl",
            "target_prm_label_manifest.json",
            "exact_residual_scheduling_oracle_on_target_llm_states",
        ],
    },
    "PRM-guided MCTS (ReST-MCTS-style)": {
        "paths": ["cgr_exp/scheduling_transfer.py"],
        "anchors": [
            "PRMGuidedMCTSStylePolicy",
            "mcts_components_observed",
            "selection",
            "expansion",
            "simulation/evaluation",
            "backpropagation",
        ],
    },
    "Solver-trained priority": {
        "paths": ["cgr_exp/scheduling_transfer.py"],
        "anchors": [
            "level1_solver_traces_only",
            "scheduling_llm_pilot_ranker.json",
            "oracle_labels_in_inference_features",
            "canonical semantic scheduling state",
        ],
    },
    "Solver-trained priority + certified pruning": {
        "paths": ["cgr_exp/scheduling_transfer.py", "cgr_exp/certification/prune.py"],
        "anchors": [
            "Solver-trained MLP + certified pruning",
            "certified_pruning",
            "should_prune",
            "learned_score_authorizes_pruning",
        ],
    },
}


def audit_q2_local_primitives(out_dir: Path) -> Dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [_audit_method(method, spec) for method, spec in Q2_LOCAL_PRIMITIVE_CHECKS.items()]
    csv_path = out_dir / "q2_local_primitive_audit.csv"
    json_path = out_dir / "q2_local_primitive_audit.json"
    _write_csv(csv_path, rows)
    manifest = {
        "artifact": "q2_local_primitive_audit",
        "all_local_primitives_ready": all(bool(row["local_primitives_ready"]) for row in rows),
        "rows": rows,
        "produced_artifacts": [csv_path.name, json_path.name],
    }
    json_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def _audit_method(method: str, spec: Mapping[str, object]) -> Dict[str, object]:
    paths = [Path(str(path)) for path in spec.get("paths", [])]
    anchors = [str(anchor) for anchor in spec.get("anchors", [])]
    missing_paths = [str(path) for path in paths if not path.exists()]
    texts = {str(path): path.read_text(encoding="utf-8") for path in paths if path.exists()}
    missing_anchors = []
    for anchor in anchors:
        if not any(anchor in text for text in texts.values()):
            missing_anchors.append(anchor)
    return {
        "Method": method,
        "local_paths": "; ".join(str(path) for path in paths),
        "required_anchors": "; ".join(anchors),
        "missing_paths": "; ".join(missing_paths),
        "missing_anchors": "; ".join(missing_anchors),
        "local_primitives_ready": not missing_paths and not missing_anchors,
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
