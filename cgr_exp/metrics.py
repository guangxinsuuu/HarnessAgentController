from __future__ import annotations

from dataclasses import asdict
import csv
import json
from pathlib import Path
from statistics import mean, median
from typing import Dict, Iterable, List, Mapping, Sequence

from .search.legacy import SearchResult


def summarize_results(results: Sequence[SearchResult]) -> Dict[str, float]:
    if not results:
        return {}
    return {
        "success": mean(1.0 if r.success else 0.0 for r in results),
        "gap": mean(r.gap for r in results),
        "false_prune": sum(r.false_prunes for r in results),
        "nodes_mean": mean(r.nodes_expanded for r in results),
        "nodes_median": median(r.nodes_expanded for r in results),
        "llm_calls_mean": mean(r.proposer_calls for r in results),
        "pruned_mean": mean(r.pruned for r in results),
        "proposal_recall": mean([e.proposal_recall for r in results for e in r.trace] or [0.0]),
        "schema_valid": mean([e.schema_valid for r in results for e in r.trace] or [0.0]),
        "duplicate_rate": mean([e.duplicate_rate for r in results for e in r.trace] or [0.0]),
    }


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = []
    for row in rows:
        for key in row.keys():
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def result_to_dict(result: SearchResult) -> Dict[str, object]:
    obj = asdict(result)
    obj["trace"] = [asdict(e) for e in result.trace]
    return obj
