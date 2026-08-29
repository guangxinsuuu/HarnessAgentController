# Q2 Remaining Runs

This checklist is generated from the current promotion blockers. It lists the
minimum remaining Q2 Scheduling artifacts that still need real-LLM execution
before Table 4 can be promoted as an end-to-end quality-compute result.

Current audit:

```bash
CERTIFICATION_DIR=runs/q2_structured_cert_reference_100_seed0_budget200_protocol_v2 \
AUTHORITY_DIR=runs/table7_grounding_robustness_100_seed0_budget200_protocol_v2 \
PYTHON_BIN=.venv/bin/python scripts/run_paper_audits.sh all
```

## Already Ready

- CP-SAT structured reference:
  `runs/q2_structured_100_seed0_runtime_v2`
- EST / list scheduling structured reference:
  `runs/q2_structured_100_seed0_runtime_v2`
- Certification Table 6:
  `runs/q2_structured_cert_reference_100_seed0_budget200_protocol_v2`
- Certification Table 7:
  `runs/table7_grounding_robustness_100_seed0_budget200_protocol_v2`

## Must Rerun For Q2 Promotion

The old learned-search artifact
`runs/scheduling_table5_prm_rest_cert_100_seed0` has useful historical numbers
but is not promotion-ready. It was generated before the current
`q2_method_alignment`, pinned `source_evidence`, `learned_search_protocol`, and
on-demand per-policy accounting gates. Do not patch these fields into old
metrics by hand; rerun the command below.

```bash
python3 -m cgr_exp.cli run-scheduling-llm-pilot \
  --output-dir runs/q2_learned_search_100_seed0_rest_gate_v2 \
  --seed 0 \
  --train-count 200 \
  --validation-count 50 \
  --test-count 100 \
  --job-count 6 \
  --machine-count 3 \
  --node-budget 80 \
  --action-budget 6 \
  --records-per-instance 60 \
  --rank-epochs 25 \
  --model gpt-4o-mini \
  --timeout 300 \
  --perturbation mixed \
  --batch-size 4 \
  --parallel-episodes 6 \
  --include-target-prm \
  --include-rest-mcts \
  --include-solver-certification \
  --target-prm-train-count 100 \
  --resume
```

The native frozen baselines still need complete native artifacts:

```bash
python3 -m cgr_exp.cli run-scheduling-native-q2 \
  --output-dir runs/q2_native_llm_baselines_100_seed0_sc_v2 \
  --seed 0 \
  --train-count 200 \
  --validation-count 50 \
  --test-count 100 \
  --job-count 6 \
  --machine-count 3 \
  --model gpt-4o-mini \
  --timeout 300 \
  --methods direct,self_consistency,best_of_n \
  --sample-count 8 \
  --parallel-episodes 8 \
  --resume
```

The strict ToT row must come from the native BFS runner, not the older
controller-isolation GPT-frontier-rank directory:

```bash
python3 -m cgr_exp.cli run-scheduling-tot-q2 \
  --output-dir runs/q2_tot_bfs_100_seed0_tot_gate_v2 \
  --seed 0 \
  --train-count 200 \
  --validation-count 50 \
  --test-count 100 \
  --job-count 6 \
  --machine-count 3 \
  --node-budget 80 \
  --action-budget 6 \
  --breadth 5 \
  --value-samples 3 \
  --model gpt-4o-mini \
  --timeout 300 \
  --parallel-episodes 6 \
  --perturbation mixed \
  --resume
```

## Fast Path

To run all remaining real-LLM Q2 rows concurrently, set the API key in your
shell and run:

```bash
export CGR_LLM_API_KEY="..."
export CGR_LLM_MODEL="gpt-4o-mini"
PYTHON_BIN=.venv/bin/python scripts/run_q2_table4.sh llm
```

That command starts native, learned-search, and strict ToT as separate jobs.
Each job also uses its own `--parallel-episodes` setting internally.

## Promotion Check

After any run finishes:

```bash
CERTIFICATION_DIR=runs/q2_structured_cert_reference_100_seed0_budget200_protocol_v2 \
AUTHORITY_DIR=runs/table7_grounding_robustness_100_seed0_budget200_protocol_v2 \
PYTHON_BIN=.venv/bin/python scripts/run_paper_audits.sh all
```

Only rows with both `result_ready=True` and `compute_ready=True` in
`runs/q2_table_current_seed0_protocol_v2_audit/table5_q2_readiness.csv` should
be copied into the paper table.
