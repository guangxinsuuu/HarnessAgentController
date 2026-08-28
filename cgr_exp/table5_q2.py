from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional


TABLE5_METHODS = [
    ("Direct / CoT", "native direct reasoning", None),
    ("Self-Consistency", "native parallel sampling", None),
    ("Best-of-N", "native matched-budget sampling", None),
    ("Tree of Thoughts", "online LLM frontier guidance", "GPT-frontier-rank / ToT-style"),
    ("Target-PRM", "target LLM states", "Target-PRM"),
    ("ReST-MCTS*", "public PRM-guided tree search protocol", None),
    ("Solver prior", "solver states only", "Solver-trained MLP"),
    ("Solver prior + certification", "solver states only + certified pruning", None),
]


def build_table5_q2_results(
    out_dir: Path,
    solver_prior_dir: Path = Path("runs/scheduling_llm_pilot_real_100_seed0"),
    expected_test_count: int = 100,
    gpt_frontier_rank_dir: Optional[Path] = None,
    target_prm_dir: Optional[Path] = None,
    direct_dir: Optional[Path] = None,
    self_consistency_dir: Optional[Path] = None,
    best_of_n_dir: Optional[Path] = None,
    rest_mcts_dir: Optional[Path] = None,
    solver_certification_dir: Optional[Path] = None,
) -> Dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    method_dirs: Dict[str, Optional[Path]] = {
        "Direct / CoT": direct_dir,
        "Self-Consistency": self_consistency_dir,
        "Best-of-N": best_of_n_dir,
        "Tree of Thoughts": gpt_frontier_rank_dir,
        "Target-PRM": target_prm_dir,
        "ReST-MCTS*": rest_mcts_dir,
        "Solver prior": solver_prior_dir,
        "Solver prior + certification": solver_certification_dir,
    }
    rows: List[Dict[str, object]] = []
    for method, training_states, source_policy in TABLE5_METHODS:
        source_dir = method_dirs[method]
        row = _pending_row(method, training_states, source_dir)
        if source_dir is not None and source_policy is not None:
            loaded = _row_from_scheduling_artifacts(method, training_states, source_policy, source_dir, expected_test_count)
            if loaded is not None:
                row = loaded
        elif source_dir is not None:
            loaded = _row_from_native_artifacts(method, training_states, source_dir)
            if loaded is not None:
                row = loaded
        rows.append(row)
    csv_path = out_dir / "table5_q2_results.csv"
    tex_path = out_dir / "table5_q2_results.tex"
    manifest_path = out_dir / "table5_q2_manifest.json"
    _write_csv(csv_path, rows)
    tex_path.write_text(_latex_table(rows), encoding="utf-8")
    manifest = {
        "table": "Table 5",
        "question": "Q2 end-to-end quality-compute comparison",
        "strict_pending_policy": True,
        "solver_prior_dir": str(solver_prior_dir),
        "expected_test_count": expected_test_count,
        "gpt_frontier_rank_dir": None if gpt_frontier_rank_dir is None else str(gpt_frontier_rank_dir),
        "target_prm_dir": None if target_prm_dir is None else str(target_prm_dir),
        "rows": rows,
        "produced_artifacts": [csv_path.name, tex_path.name, manifest_path.name],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def _pending_row(method: str, training_states: str, source_dir: Optional[Path]) -> Dict[str, object]:
    return {
        "Method": method,
        "Training states": training_states,
        "Success": "--",
        "Gap": "--",
        "LLM calls": "--",
        "Tokens": "--",
        "Wall-clock": "--",
        "Status": "pending",
        "source_artifact": "" if source_dir is None else str(source_dir),
        "source_policy": "",
        "notes": "artifact missing or native runner pending",
    }


def _row_from_scheduling_artifacts(
    method: str,
    training_states: str,
    source_policy: str,
    source_dir: Path,
    expected_test_count: int,
) -> Optional[Dict[str, object]]:
    table_path = source_dir / "scheduling_llm_pilot_table.csv"
    per_instance_path = source_dir / "scheduling_llm_pilot_per_instance.csv"
    if not table_path.exists():
        return None
    table_rows = _read_csv(table_path)
    source_row = next((row for row in table_rows if row.get("Policy") == source_policy), None)
    if source_row is None:
        return None
    per_instance_rows = [row for row in _read_csv(per_instance_path) if row.get("Policy") == source_policy]
    instance_count = len(per_instance_rows)
    complete = instance_count >= expected_test_count
    proposal_calls = _as_float(source_row.get("proposal_calls_mean")) * instance_count
    value_calls = _as_float(source_row.get("value_calls"))
    total_calls = proposal_calls + value_calls
    wall_clock = _wall_clock_seconds(source_dir)
    return {
        "Method": method,
        "Training states": training_states,
        "Success": _fmt_float(_as_float(source_row.get("success")), 4),
        "Gap": _fmt_float(_as_float(source_row.get("gap")), 5),
        "LLM calls": _fmt_count(total_calls) if instance_count else "--",
        "Tokens": _fmt_tokens(_as_float(source_row.get("total_tokens"))),
        "Wall-clock": _fmt_seconds(wall_clock),
        "Status": "complete" if complete else f"partial: {instance_count}/{expected_test_count}",
        "source_artifact": str(table_path),
        "source_policy": source_policy,
        "notes": (
            "controller-row extracted from scheduling_llm_pilot artifacts"
            if complete
            else "partial checkpoint; do not promote to main result"
        ),
    }


def _row_from_native_artifacts(method: str, training_states: str, source_dir: Path) -> Optional[Dict[str, object]]:
    table_path = source_dir / "table5_native_row.json"
    if not table_path.exists():
        return None
    data = json.loads(table_path.read_text(encoding="utf-8"))
    return {
        "Method": method,
        "Training states": training_states,
        "Success": _fmt_float(_as_float(data.get("success")), 4),
        "Gap": _fmt_float(_as_float(data.get("gap")), 5),
        "LLM calls": _fmt_count(_as_float(data.get("llm_calls"))),
        "Tokens": _fmt_tokens(_as_float(data.get("tokens"))),
        "Wall-clock": _fmt_seconds(_as_float(data.get("wall_clock_seconds"))),
        "Status": str(data.get("status", "complete")),
        "source_artifact": str(table_path),
        "source_policy": str(data.get("source_policy", "native")),
        "notes": str(data.get("notes", "native protocol row")),
    }


def _wall_clock_seconds(source_dir: Path) -> Optional[float]:
    rows = _read_csv(source_dir / "timing_metrics.csv")
    if not rows:
        return None
    return sum(_as_float(row.get("wall_clock_seconds")) for row in rows)


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _latex_table(rows: List[Dict[str, object]]) -> str:
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Q2 end-to-end quality--compute comparison on real-LLM Scheduling.}",
        "\\label{tab:q2_results}",
        "\\small",
        "\\begin{tabular}{lrrrrrl}",
        "\\toprule",
        "Method & Success $\\uparrow$ & Gap $\\downarrow$ & LLM calls $\\downarrow$ & Tokens $\\downarrow$ & Wall-clock $\\downarrow$ & Status \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(
            " & ".join(
                [
                    _latex_escape(str(row["Method"])),
                    str(row["Success"]),
                    str(row["Gap"]),
                    str(row["LLM calls"]),
                    str(row["Tokens"]),
                    str(row["Wall-clock"]),
                    _latex_escape(str(row["Status"])),
                ]
            )
            + " \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""])
    return "\n".join(lines)


def _as_float(value: object) -> float:
    try:
        if value in (None, ""):
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _fmt_float(value: float, digits: int) -> str:
    return f"{value:.{digits}f}"


def _fmt_count(value: float) -> str:
    return str(int(round(value)))


def _fmt_tokens(value: float) -> str:
    if value <= 0:
        return "--"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if value >= 1_000:
        return f"{value / 1_000:.1f}K"
    return str(int(round(value)))


def _fmt_seconds(value: Optional[float]) -> str:
    if value is None or value <= 0:
        return "--"
    if value >= 3600:
        return f"{value / 3600:.2f}h"
    if value >= 60:
        return f"{value / 60:.1f}m"
    return f"{value:.1f}s"


def _latex_escape(text: str) -> str:
    return (
        text.replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("$", "\\$")
        .replace("#", "\\#")
        .replace("_", "\\_")
    )
