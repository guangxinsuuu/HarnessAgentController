from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import time

from .config import flatten_config, load_and_validate_config
from .experiments import PRESETS, run_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CGR pilot experiments.")
    parser.add_argument("--preset", choices=sorted(PRESETS), default="smoke")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--train-count", type=int, default=None)
    parser.add_argument("--test-count", type=int, default=None)
    parser.add_argument("--node-budget", type=int, default=None)
    parser.add_argument("--llm-recall", type=float, default=None)
    parser.add_argument("--false-negative-rate", type=float, default=None)
    parser.add_argument("--false-positive-rate", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = PRESETS[args.preset]
    if args.config is not None:
        cfg = load_and_validate_config(args.config, cfg)
    overrides = {}
    for arg_name, field_name in [
        ("seed", "seed"),
        ("train_count", "train_count"),
        ("test_count", "test_count"),
        ("node_budget", "node_budget"),
        ("llm_recall", "llm_recall"),
        ("false_negative_rate", "false_negative_rate"),
        ("false_positive_rate", "false_positive_rate"),
    ]:
        value = getattr(args, arg_name)
        if value is not None:
            overrides[field_name] = value
    if overrides:
        cfg = replace(cfg, **overrides)

    run_id = f"{args.preset}-seed{cfg.seed}-{time.strftime('%Y%m%d-%H%M%S')}"
    out_dir = args.out or args.output_dir or Path("runs") / run_id
    metrics = run_experiment(cfg, out_dir)
    print(f"wrote {out_dir}")
    print(f"solver ranking pairs: {metrics['solver_pair_count']}")
    print(f"llm ranking pairs: {metrics['llm_pair_count']}")
    print(f"zero-shot target rank error: {metrics['target_rank_error_zero_shot']:.3f}")
    print(f"domain discrepancy proxy: {metrics['domain_discrepancy_proxy']:.3f}")
    for name, row in metrics["methods"].items():
        print(f"{name:22s} success={row['success']:.3f} gap={row['gap']:.3f} nodes={row['nodes_mean']:.1f} calls={row['llm_calls_mean']:.1f}")


def _flatten_config(raw):
    return flatten_config(raw)


if __name__ == "__main__":
    main()
