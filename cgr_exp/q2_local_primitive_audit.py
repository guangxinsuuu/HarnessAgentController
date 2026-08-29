from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List, Mapping

from cgr_exp.q2_method_contracts import q2_method_contract


MAX_CSV_LOCATIONS_PER_ANCHOR = 5


Q2_LOCAL_PRIMITIVE_CHECKS: Dict[str, Dict[str, object]] = {
    "CP-SAT": {
        "paths": ["cgr_exp/scheduling_structured_q2.py"],
        "anchors": ["NewOptionalIntervalVar", "AddNoOverlap", "Minimize", "llm_calls"],
        "primitive_anchors": {
            "integer scheduling variables": ["NewIntVar"],
            "optional interval variables": ["NewOptionalIntervalVar"],
            "NoOverlap machine constraints": ["AddNoOverlap"],
            "precedence/release constraints": ["release_times", "predecessors"],
            "objective minimization": ["Minimize"],
        },
    },
    "EST / list scheduling heuristic": {
        "paths": ["cgr_exp/scheduling_structured_q2.py"],
        "anchors": ["earliest_start_list_scheduling", "_earliest_start_list_schedule", "llm_calls"],
        "primitive_anchors": {
            "ready-list dispatch": ["ready_jobs", "dispatch_steps"],
            "earliest feasible start selection": ["_earliest_start_list_schedule", "min(start, finish"],
            "deterministic tie breaking": ["structured_runtime_tie_break_rule"],
        },
    },
    "Direct / CoT": {
        "paths": ["cgr_exp/scheduling_native_q2.py"],
        "anchors": ["first_valid", "generate_scheduling_solution", "reasoning"],
        "primitive_anchors": {
            "single reasoning path": ["first_valid", "sample_count"],
            "intermediate natural-language rationale": ["reasoning"],
            "final answer extraction": ["generate_scheduling_solution"],
        },
    },
    "Self-Consistency": {
        "paths": ["cgr_exp/scheduling_native_q2.py"],
        "anchors": ["majority_job_order_then_first_valid", "_select_sample", "sample_count"],
        "primitive_anchors": {
            "multiple independent reasoning paths": ["sample_count"],
            "final-answer aggregation": ["majority_job_order_then_first_valid"],
            "majority/consistency selection": ["_select_sample", "job_order"],
        },
    },
    "Best-of-N": {
        "paths": ["cgr_exp/scheduling_native_q2.py"],
        "anchors": ["best_verified_objective", "_select_sample", "objective"],
        "primitive_anchors": {
            "multiple complete answer samples": ["sample_count"],
            "verifier or reward scoring": ["objective", "best_verified_objective"],
            "best sample selection": ["_select_sample", "best_verified_objective"],
        },
    },
    "Tree of Thoughts": {
        "paths": ["third_party/tree-of-thought-llm/src/tot/methods/bfs.py", "cgr_exp/scheduling_tot_q2.py"],
        "source_checkouts": {"third_party/tree-of-thought-llm": "8050e67d0e3a0fddc424d7fa5801538722a4c4cc"},
        "anchors": [
            "def solve",
            "method_generate",
            "method_evaluate",
            "method_select",
            "get_values",
            "get_votes",
            "breadth-first frontier search",
            "tot_value_sample_count",
        ],
        "primitive_anchors": {
            "method_generate": ["method_generate"],
            "method_evaluate": ["method_evaluate", "tot_value_sample_count"],
            "method_select": ["method_select"],
            "breadth-first search": ["breadth-first frontier search"],
            "value or vote evaluation over generated candidates": ["method_evaluate", "value"],
            "value cache / duplicate candidate handling": ["value_cache"],
            "greedy top-b selection": ["greedy top-b", "n_select_sample"],
        },
    },
    "Target-PRM priority": {
        "paths": ["cgr_exp/scheduling_transfer.py"],
        "anchors": [
            "target_prm_trace_records.jsonl",
            "target_prm_train_pairs.jsonl",
            "target_prm_label_manifest.json",
            "exact_residual_scheduling_oracle_on_target_llm_states",
        ],
        "primitive_anchors": {
            "intermediate-step labels": ["uses_intermediate_states", "max_record_depth"],
            "per-step candidate completions": ["frontier_parent_group_count"],
            "step rating / process supervision": ["target_process_frontier_ranking"],
            "process supervision": ["process_label_record_count"],
            "reward/value model over partial solutions": ["TinyPairwiseMLPRanker", "target_prm_ranker"],
        },
    },
    "Solver-PRM priority": {
        "paths": ["cgr_exp/scheduling_transfer.py"],
        "anchors": [
            "solver_prm_trace_records.jsonl",
            "solver_prm_train_pairs.jsonl",
            "solver_prm_label_manifest.json",
            "exact_residual_scheduling_oracle_on_solver_states",
        ],
        "primitive_anchors": {
            "intermediate-step labels": ["uses_intermediate_states", "max_record_depth"],
            "per-step candidate completions": ["frontier_parent_group_count"],
            "step rating / process supervision": ["solver_process_frontier_ranking"],
            "process supervision": ["process_label_record_count"],
            "reward/value model over partial solutions": ["TinyPairwiseMLPRanker", "solver_prm_ranker"],
        },
    },
    "PRM-guided MCTS (ReST-MCTS-style)": {
        "paths": ["third_party/ReST-MCTS/MCTS/mcts.py", "third_party/ReST-MCTS/MCTS/base.py", "cgr_exp/scheduling_transfer.py"],
        "source_checkouts": {"third_party/ReST-MCTS": "2d5f488c3d6e24f99d50a9860e818383b1bb5883"},
        "anchors": [
            "def MCTS_search",
            "def executeRound",
            "def selectNode",
            "def expand",
            "def back_propagate",
            "def getBestChild",
            "class treeNode",
            "PRMGuidedMCTSStylePolicy",
            "mcts_components_observed",
            "selection",
            "expansion",
            "simulation/evaluation",
            "backpropagation",
        ],
        "primitive_anchors": {
            "MCTS selection": ["mcts_selection_calls", "selection"],
            "MCTS expansion": ["mcts_expansion_observations", "expansion"],
            "MCTS simulation/evaluation": ["mcts_leaf_value_evaluations", "simulation/evaluation"],
            "MCTS backpropagation": ["mcts_backup_updates", "backpropagation"],
            "visit-count exploration bonus": ["visits", "exploration"],
            "step-value/process-reward guided child scoring": ["PRM_prior", "process_reward_guidance"],
            "process reward guidance": ["target_prm_ranker", "process_reward_guidance"],
        },
    },
    "Solver-trained priority": {
        "paths": ["cgr_exp/scheduling_transfer.py"],
        "anchors": [
            "level1_solver_traces_only",
            "scheduling_llm_pilot_ranker.json",
            "oracle_labels_in_inference_features",
            "canonical semantic scheduling state",
        ],
        "primitive_anchors": {
            "solver-induced source trajectories": ["level1_solver_traces_only"],
            "frontier priority learning": ["TinyPairwiseMLPRanker", "scheduling_llm_pilot_ranker.json"],
            "frozen zero-shot transfer": ["zero_shot_transfer", "target_training_trace_count"],
        },
    },
    "Solver-trained priority + certified pruning": {
        "paths": ["cgr_exp/scheduling_transfer.py", "cgr_exp/certification/prune.py"],
        "anchors": [
            "Solver-trained MLP + certified pruning",
            "certified_pruning",
            "should_prune",
            "learned_score_authorizes_pruning",
        ],
        "primitive_anchors": {
            "solver-trained frontier priority": ["Solver-trained MLP + certified pruning"],
            "verified pruning authority": ["should_prune", "verified lower-bound"],
            "learned score separated from pruning decision": ["learned_score_authorizes_pruning"],
        },
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
    primitive_anchors = _primitive_anchor_spec(spec.get("primitive_anchors", {}))
    checkout_audit = _source_checkout_audit(spec.get("source_checkouts", {}))
    missing_paths = [str(path) for path in paths if not path.exists()]
    lines_by_path = {
        str(path): path.read_text(encoding="utf-8").splitlines()
        for path in paths
        if path.exists()
    }
    locations_by_anchor = {
        anchor: _anchor_locations(anchor, lines_by_path)
        for anchor in anchors
    }
    missing_anchors = []
    for anchor, locations in locations_by_anchor.items():
        if not locations:
            missing_anchors.append(anchor)
    primitive_alignment = _primitive_alignment(primitive_anchors, lines_by_path)
    contract_primitives = [str(item) for item in q2_method_contract(method).get("source_primitives", [])]
    missing_primitive_mappings = [
        primitive
        for primitive in contract_primitives
        if primitive not in primitive_alignment
    ]
    missing_primitive_anchors = [
        f"{primitive}: {anchor}"
        for primitive, anchors_by_name in primitive_alignment.items()
        for anchor, locations in anchors_by_name.items()
        if not locations
    ]
    return {
        "Method": method,
        "local_paths": "; ".join(str(path) for path in paths),
        "required_anchors": "; ".join(anchors),
        "anchor_locations": "; ".join(
            f"{anchor}: {', '.join(_trim_locations(locations))}"
            for anchor, locations in locations_by_anchor.items()
            if locations
        ),
        "missing_paths": "; ".join(missing_paths),
        "missing_anchors": "; ".join(missing_anchors),
        "primitive_alignment": _format_primitive_alignment(primitive_alignment),
        "primitive_alignment_json": json.dumps(primitive_alignment, sort_keys=True),
        "missing_primitive_mappings": "; ".join(missing_primitive_mappings),
        "missing_primitive_anchors": "; ".join(missing_primitive_anchors),
        "source_checkouts": json.dumps(checkout_audit, sort_keys=True),
        "missing_source_checkouts": "; ".join(item["path"] for item in checkout_audit if not item["ready"]),
        "anchor_locations_json": json.dumps(locations_by_anchor, sort_keys=True),
        "local_primitives_ready": not missing_paths
        and not missing_anchors
        and not missing_primitive_mappings
        and not missing_primitive_anchors
        and all(item["ready"] for item in checkout_audit),
    }


def _primitive_anchor_spec(value: object) -> Dict[str, List[str]]:
    if not isinstance(value, Mapping):
        return {}
    return {
        str(primitive): [str(anchor) for anchor in anchors]
        for primitive, anchors in value.items()
        if isinstance(anchors, list)
    }


def _source_checkout_audit(value: object) -> List[Dict[str, object]]:
    if not isinstance(value, Mapping):
        return []
    rows = []
    for path_text, expected_revision in value.items():
        path = Path(str(path_text))
        expected = str(expected_revision)
        observed = ""
        error = ""
        if not path.exists():
            error = "missing checkout"
        else:
            try:
                result = subprocess.run(
                    ["git", "-C", str(path), "rev-parse", "HEAD"],
                    check=False,
                    text=True,
                    capture_output=True,
                )
                if result.returncode == 0:
                    observed = result.stdout.strip()
                else:
                    error = result.stderr.strip() or f"git rev-parse exited {result.returncode}"
            except OSError as exc:
                error = exc.__class__.__name__
        rows.append(
            {
                "path": str(path),
                "expected_revision": expected,
                "observed_revision": observed,
                "ready": bool(observed and observed == expected and not error),
                "error": error,
            }
        )
    return rows


def _primitive_alignment(
    primitive_anchors: Mapping[str, List[str]],
    lines_by_path: Mapping[str, List[str]],
) -> Dict[str, Dict[str, List[str]]]:
    return {
        primitive: {
            anchor: _anchor_locations(anchor, lines_by_path)
            for anchor in anchors
        }
        for primitive, anchors in primitive_anchors.items()
    }


def _format_primitive_alignment(primitive_alignment: Mapping[str, Mapping[str, List[str]]]) -> str:
    parts = []
    for primitive, anchors_by_name in primitive_alignment.items():
        anchor_parts = []
        for anchor, locations in anchors_by_name.items():
            anchor_parts.append(f"{anchor} -> {', '.join(_trim_locations(locations))}")
        parts.append(f"{primitive}: " + " | ".join(anchor_parts))
    return "; ".join(parts)


def _anchor_locations(anchor: str, lines_by_path: Mapping[str, List[str]]) -> List[str]:
    locations = []
    for path, lines in lines_by_path.items():
        for idx, line in enumerate(lines, start=1):
            if anchor in line:
                locations.append(f"{path}:{idx}")
    return locations


def _trim_locations(locations: List[str]) -> List[str]:
    if len(locations) <= MAX_CSV_LOCATIONS_PER_ANCHOR:
        return locations
    return [
        *locations[:MAX_CSV_LOCATIONS_PER_ANCHOR],
        f"...(+{len(locations) - MAX_CSV_LOCATIONS_PER_ANCHOR} more)",
    ]


def _write_csv(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
