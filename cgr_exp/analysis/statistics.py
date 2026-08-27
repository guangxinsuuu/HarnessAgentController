from __future__ import annotations

import csv
import json
import math
import random
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Mapping, Sequence


def _std(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    mu = mean(values)
    return math.sqrt(sum((x - mu) ** 2 for x in values) / (len(values) - 1))


def _p95(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = math.ceil(0.95 * len(ordered)) - 1
    return ordered[max(0, min(idx, len(ordered) - 1))]


def _bootstrap_ci(values: Sequence[float], seed: int = 0, samples: int = 500) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return float(values[0]), float(values[0])
    rng = random.Random(seed)
    means = []
    for _ in range(samples):
        draw = [values[rng.randrange(len(values))] for _ in values]
        means.append(mean(draw))
    means.sort()
    low = means[int(0.025 * (len(means) - 1))]
    high = means[int(0.975 * (len(means) - 1))]
    return float(low), float(high)


def aggregate_seed_metrics(rows: Sequence[Mapping[str, object]], group_key: str, metric_keys: Sequence[str]) -> List[Dict[str, object]]:
    groups: Dict[object, List[Mapping[str, object]]] = {}
    for row in rows:
        groups.setdefault(row[group_key], []).append(row)
    out: List[Dict[str, object]] = []
    for group, group_rows in sorted(groups.items(), key=lambda item: str(item[0])):
        agg: Dict[str, object] = {group_key: group, "seed_count": len({r.get("seed") for r in group_rows})}
        for metric in metric_keys:
            values = [float(r[metric]) for r in group_rows if metric in r and r[metric] != "--"]
            agg[f"{metric}_mean"] = mean(values) if values else 0.0
            agg[f"{metric}_std"] = _std(values)
            agg[f"{metric}_p95"] = _p95(values)
            ci_low, ci_high = _bootstrap_ci(values, seed=17 + len(metric))
            agg[f"{metric}_bootstrap_ci95_low"] = ci_low
            agg[f"{metric}_bootstrap_ci95_high"] = ci_high
        out.append(agg)
    return out


def write_seed_aggregate(out_dir: Path, rows: Sequence[Mapping[str, object]], group_key: str, metric_keys: Sequence[str]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    aggregate = aggregate_seed_metrics(rows, group_key, metric_keys)
    csv_path = out_dir / "multi_seed_summary.csv"
    if aggregate:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(aggregate[0]))
            writer.writeheader()
            writer.writerows(aggregate)
    else:
        csv_path.write_text("", encoding="utf-8")
    (out_dir / "multi_seed_manifest.json").write_text(
        json.dumps(
            {
                "group_key": group_key,
                "metric_keys": list(metric_keys),
                "seed_count": len({r.get("seed") for r in rows}),
                "row_count": len(rows),
                "statistics": ["mean", "sample_std", "nearest_rank_p95", "bootstrap_ci95"],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
