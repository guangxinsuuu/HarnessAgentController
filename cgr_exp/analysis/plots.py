from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple


def _scale(value: float, lo: float, hi: float, start: float, end: float) -> float:
    if hi == lo:
        return (start + end) / 2.0
    return start + (value - lo) * (end - start) / (hi - lo)


def _scatter_svg(title: str, x_label: str, y_label: str, points: Sequence[Tuple[float, float, str]]) -> str:
    width, height = 720, 420
    left, right, top, bottom = 70, 30, 40, 60
    xs = [p[0] for p in points] or [0.0]
    ys = [p[1] for p in points] or [0.0]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    if x0 == x1:
        x0 -= 1.0
        x1 += 1.0
    if y0 == y1:
        y0 -= 1.0
        y1 += 1.0
    body = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="24" text-anchor="middle" font-family="Arial" font-size="18">{html.escape(title)}</text>',
        f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="black"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="black"/>',
        f'<text x="{width/2}" y="{height-16}" text-anchor="middle" font-family="Arial" font-size="13">{html.escape(x_label)}</text>',
        f'<text x="16" y="{height/2}" text-anchor="middle" transform="rotate(-90 16 {height/2})" font-family="Arial" font-size="13">{html.escape(y_label)}</text>',
    ]
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf"]
    for idx, (x, y, label) in enumerate(points):
        px = _scale(x, x0, x1, left, width - right)
        py = _scale(y, y0, y1, height - bottom, top)
        color = colors[idx % len(colors)]
        body.append(f'<circle cx="{px:.2f}" cy="{py:.2f}" r="5" fill="{color}"><title>{html.escape(label)}</title></circle>')
        body.append(f'<text x="{px + 7:.2f}" y="{py - 7:.2f}" font-family="Arial" font-size="10">{html.escape(label[:22])}</text>')
    body.append("</svg>")
    return "\n".join(body)


def _write(path: Path, svg: str) -> Path:
    path.write_text(svg, encoding="utf-8")
    return path


def generate_paper_plots(metrics: Mapping[str, object], out_dir: Path) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    methods: Mapping[str, Mapping[str, float]] = metrics.get("methods", {})  # type: ignore[assignment]
    paths: List[Path] = []
    quality_points = [
        (float(row.get("llm_calls_mean", 0.0)), float(row.get("success", 0.0)), name)
        for name, row in methods.items()
    ]
    paths.append(_write(out_dir / "quality_compute_pareto.svg", _scatter_svg("Quality Compute Pareto", "LLM calls", "Success", quality_points)))
    sweep = metrics.get("quality_compute_sweep", [])
    if isinstance(sweep, list) and sweep:
        sweep_points = [
            (float(row.get("normalized_expensive_budget", 0.0)), float(row.get("success", 0.0)), str(row.get("method", "")))
            for row in sweep
        ]
        paths.append(_write(out_dir / "quality_compute_sweep.svg", _scatter_svg("Quality Compute Sweep", "Normalized expensive budget", "Success", sweep_points)))

    transfer = metrics.get("transfer", {})  # type: ignore[assignment]
    transfer_rows = []
    if isinstance(transfer, Mapping):
        for key in ("zero_shot", "llm_only"):
            if key in transfer:
                row = transfer[key]
                transfer_rows.append((float(row.get("target_training_pair_count", 0.0)), 1.0 - float(row.get("target_rank_error", 1.0)), key))  # type: ignore[union-attr]
        for row in transfer.get("few_shot", []):  # type: ignore[union-attr]
            transfer_rows.append((float(row.get("target_training_pair_count", 0.0)), 1.0 - float(row.get("target_rank_error", 1.0)), "few-shot"))
    paths.append(_write(out_dir / "transfer_sample_efficiency.svg", _scatter_svg("Transfer Sample Efficiency", "Target training pairs", "Ranking accuracy", transfer_rows)))

    proposal_points = [
        (float(row.get("proposal_recall", 0.0)), float(row.get("success", 0.0)), name)
        for name, row in methods.items()
    ]
    paths.append(_write(out_dir / "proposal_recall_vs_success.svg", _scatter_svg("Proposal Bottleneck", "ProposalRecall@K", "Success", proposal_points)))

    domain_points = [
        (
            float(metrics.get("domain_discrepancy_proxy", 0.0)),
            float(metrics.get("target_rank_error_zero_shot", 0.0)),
            "zero-shot",
        )
    ]
    paths.append(_write(out_dir / "distribution_shift_vs_ranking.svg", _scatter_svg("Distribution Shift vs Ranking", "Proxy discrepancy", "Zero-shot rank error", domain_points)))

    (out_dir / "plot_manifest.json").write_text(
        json.dumps({"plots": [p.name for p in paths], "format": "svg", "dependency_free": True}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return paths
