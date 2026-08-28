# Certified Graph-Guided Reasoning Experiments

This repository is a runnable, dependency-free pilot implementation of the PDF's CGR experiment protocol.

It implements:

- Level-I solver-induced traces from exact assignment, scheduling, and set-cover search.
- Level-II LLM-induced traces using a typed proposer simulator with controllable recall, invalid actions, duplicates, and paraphrase noise.
- Conservative verification: only verified constraints can prune. Uncertain or semantic features can affect priority but never certification.
- Three transfer regimes: solver-only zero-shot, solver pretraining plus few-shot target adaptation, and LLM-only training.
- Baselines: DFS, random, certified best-bound, solver oracle ordering, and learned priority.
- Metrics: success, optimality gap, false-prune rate, nodes, proposer calls, proposal recall, schema validity, ranking error, and domain-classifier discrepancy proxy.

The default smoke pipeline uses only the Python standard library. Real-LLM and
structured-reference experiments have optional dependencies:

- OpenAI-compatible API access for real LLM pilots.
- `ortools` for the Q2 CP-SAT structured scheduling reference.

Install the CP-SAT extra when you need the CP-SAT row:

```bash
python3 -m pip install -e ".[q2-structured]"
```

## Quick Start

```bash
python3 -m cgr_exp.run_experiment --preset smoke
```

Outputs are written to `runs/<run_id>/`:

- `metrics.json`: all aggregate metrics.
- `primary_table.csv`: matched-budget comparison table.
- `transfer_table.csv`: zero-shot, few-shot, and LLM-only transfer regimes.
- `config.json`: exact experiment configuration.
- `trace_sample.jsonl`: example search traces.

For a larger pilot:

```bash
python3 -m cgr_exp.run_experiment --preset pilot --seed 7
```

## Pre-LLM Game24 Sanity Pipeline

Before spending real LLM calls, run the exact Game24 pipeline:

```bash
python3 -m cgr_exp.cli run-game24-offline --output-dir runs/game24-offline --seed 0 --limit 1000
```

This writes:

- `game24_split_manifest.json`: `0-699` train, `700-799` validation, `800-899` development, `900-999` final test when `--limit 1000`.
- `game24_trajectory_sample.jsonl`: exact states with oracle supervision stored as labels only.
- `game24_offline_table.csv`: Random, DFS, greedy heuristic, Oracle, MLP, Tree-GNN, and Graph-GNN under fixed replay proposals.
- `game24_offline_metrics.json`: pair count, policy metrics, leakage guard metadata, and `llm_required=false`.

## Important Scope

The PDF is a method/protocol document, not an attached source repository. This implementation is therefore a faithful runnable pilot rather than recovered original code. The default Level-II proposer is a deterministic simulator so experiments can run without paid LLM calls or network access. To use a real LLM, provide a concrete `LLMClient` for `cgr_exp/proposers/llm_adapter.py` and keep the same typed action contract.

## Reproducibility

All random sources use the configured seed. The problem generator, search budget, corruption rates, train/test splits, and transfer settings are saved in each run directory.
