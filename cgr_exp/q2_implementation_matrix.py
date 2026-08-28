from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping

from cgr_exp.q2_method_contracts import q2_method_contract
from cgr_exp.q2_source_evidence import Q2_SOURCE_EVIDENCE
from cgr_exp.table5_q2 import TABLE5_METHODS


Q2_LOCAL_IMPLEMENTATION_PATHS: Dict[str, List[str]] = {
    "CP-SAT": ["cgr_exp/scheduling_structured_q2.py"],
    "EST / list scheduling heuristic": ["cgr_exp/scheduling_structured_q2.py"],
    "Direct / CoT": ["cgr_exp/scheduling_native_q2.py"],
    "Self-Consistency": ["cgr_exp/scheduling_native_q2.py"],
    "Best-of-N": ["cgr_exp/scheduling_native_q2.py"],
    "Tree of Thoughts": ["cgr_exp/scheduling_tot_q2.py"],
    "Target-PRM priority": ["cgr_exp/scheduling_transfer.py"],
    "PRM-guided MCTS (ReST-MCTS-style)": ["cgr_exp/scheduling_transfer.py"],
    "Solver-trained priority": ["cgr_exp/scheduling_transfer.py"],
    "Solver-trained priority + certified pruning": [
        "cgr_exp/scheduling_transfer.py",
        "cgr_exp/certification/prune.py",
    ],
}


Q2_REQUIRED_ARTIFACTS: Dict[str, List[str]] = {
    "CP-SAT": ["scheduling_structured_q2_metrics.json"],
    "EST / list scheduling heuristic": ["scheduling_structured_q2_metrics.json"],
    "Direct / CoT": ["scheduling_native_q2_samples.jsonl", "scheduling_native_q2_manifest.json"],
    "Self-Consistency": ["scheduling_native_q2_samples.jsonl", "scheduling_native_q2_manifest.json"],
    "Best-of-N": ["scheduling_native_q2_samples.jsonl", "scheduling_native_q2_manifest.json"],
    "Tree of Thoughts": ["scheduling_tot_q2_trace.jsonl", "scheduling_tot_q2_manifest.json"],
    "Target-PRM priority": [
        "scheduling_llm_pilot_metrics.json",
        "target_prm_ranker.json",
        "target_prm_trace_records.jsonl",
        "target_prm_train_pairs.jsonl",
        "target_prm_label_manifest.json",
    ],
    "PRM-guided MCTS (ReST-MCTS-style)": [
        "scheduling_llm_pilot_metrics.json",
        "target_prm_ranker.json",
        "target_prm_trace_records.jsonl",
        "target_prm_train_pairs.jsonl",
        "target_prm_label_manifest.json",
    ],
    "Solver-trained priority": [
        "scheduling_llm_pilot_metrics.json",
        "scheduling_llm_pilot_ranker.json",
        "split_manifest.json",
        "checkpoint_metadata.json",
        "feature_schema.json",
    ],
    "Solver-trained priority + certified pruning": [
        "scheduling_llm_pilot_metrics.json",
        "scheduling_llm_pilot_ranker.json",
        "split_manifest.json",
        "checkpoint_metadata.json",
        "feature_schema.json",
    ],
}


def build_q2_implementation_matrix(out_dir: Path) -> Dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [_matrix_row(method, setting) for method, setting, _source_policy, _kind in TABLE5_METHODS]
    table_methods = {method for method, _setting, _source_policy, _kind in TABLE5_METHODS}
    csv_path = out_dir / "q2_implementation_matrix.csv"
    json_path = out_dir / "q2_implementation_matrix.json"
    _write_csv(csv_path, rows)
    manifest = {
        "artifact": "q2_implementation_matrix",
        "row_count": len(rows),
        "all_table_rows_covered": table_methods <= set(Q2_LOCAL_IMPLEMENTATION_PATHS) and table_methods <= set(Q2_REQUIRED_ARTIFACTS),
        "rows": rows,
        "produced_artifacts": [csv_path.name, json_path.name],
    }
    json_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def _matrix_row(method: str, transfer_setting: str) -> Dict[str, object]:
    contract = q2_method_contract(method)
    evidence = Q2_SOURCE_EVIDENCE.get(method, {})
    local_paths = Q2_LOCAL_IMPLEMENTATION_PATHS.get(method, [])
    missing_local_paths = [path for path in local_paths if not Path(path).exists()]
    return {
        "Method": method,
        "Transfer setting": transfer_setting,
        "Claim scope": str(contract.get("claim_scope", "")),
        "Official full reproduction": bool(contract.get("official_reproduction", False)),
        "Source URL": str(contract.get("source_url", "")),
        "Official source files": _join(evidence.get("source_files", [])),
        "Official code refs": _join(evidence.get("official_code_refs", [])),
        "Official source anchors": _join(evidence.get("official_source_anchors", [])),
        "Official primitives": _join(evidence.get("official_primitives", contract.get("source_primitives", []))),
        "Local code paths": _join(local_paths),
        "Local code present": not missing_local_paths,
        "Missing local code paths": _join(missing_local_paths),
        "Implemented scope": _join(contract.get("implemented_scope", [])),
        "Required result artifacts": _join(Q2_REQUIRED_ARTIFACTS.get(method, [])),
        "Promotion gates": _join(evidence.get("local_promotion_gates", [])),
        "Required table wording": str(contract.get("required_table_wording", "")),
        "Non-claims": _join(contract.get("non_claims", [])),
    }


def _join(value: object) -> str:
    if isinstance(value, list):
        return "; ".join(str(item) for item in value)
    return str(value)


def _write_csv(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
