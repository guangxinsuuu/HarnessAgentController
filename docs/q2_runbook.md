# Q2 Scheduling Runbook

This runbook lists the code path, method contract, and command needed for each
row in the Q2 end-to-end Scheduling table. It is intentionally strict: rows
with partial artifacts, diagnostic-only artifacts, or shared-cache fallback
token accounting should remain marked as pending or qualified.

## Global Setup

Real-LLM rows require an OpenAI-compatible chat-completions endpoint:

```bash
export CGR_LLM_API_KEY="..."
export CGR_LLM_MODEL="gpt-4o-mini"
```

Do not commit API keys or run outputs. `runs/*`, `.env*`, `.venv/`, and
editable-install metadata are ignored.

`--resume` is policy-set aware for Scheduling transfer runs: an episode is
skipped only when the checkpoint already contains every policy requested by the
current command. If a previous partial run is missing one requested policy row,
that rendering is rerun instead of being silently promoted.
When new policies are added incrementally, existing `proposal_cache.jsonl`
state rows are preloaded so the new policy reuses the frozen candidate actions
where available; new LLM proposal calls occur only for frontier states that were
not already in the cache.
In `scheduling_llm_pilot_proposal_cache.csv`, `total_tokens` is the frozen-cache
equivalent proposal cost for that rendering, while `new_proposal_tokens` and
`new_proposal_calls` record the additional API work caused by the current
resume command.

Native Direct/Self-Consistency/Best-of-N resume is sample-audit aware: a
method-task row is skipped only when its expected sample rows are present in
`scheduling_native_q2_samples.jsonl`. Native ToT resume is trace aware: a
rendering is skipped only when `scheduling_tot_q2_trace.jsonl` contains its
`finish` event. Missing audit artifacts cause reruns rather than incomplete
promotion.

## Row Contracts and Commands

### CP-SAT

Code path: `cgr_exp/scheduling_structured_q2.py`

Reference: Google OR-Tools CP-SAT scheduling models
(`https://github.com/google/or-tools`).

Command:

```bash
.venv/bin/python -m cgr_exp.cli run-scheduling-structured-q2 \
  --output-dir runs/q2_structured_100_seed0 \
  --seed 0 \
  --train-count 200 \
  --validation-count 50 \
  --test-count 100 \
  --job-count 6 \
  --machine-count 3 \
  --cpsat-time-limit-seconds 5 \
  --overwrite
```

Completion evidence:

- `scheduling_structured_q2_table.csv`
- status `complete`
- `llm_calls = 0`, `tokens = 0`

Current local result:

- success `1.0000`
- gap `0.00000`
- tokens `0`

### EST / List Scheduling Heuristic

Code path: `cgr_exp/scheduling_structured_q2.py`

Reference: standard earliest-start list scheduling heuristic.

Command: same as CP-SAT structured command above.

Current local result:

- success `0.4200`
- gap `0.16829`
- tokens `0`

### Direct / CoT

Code path: `cgr_exp/scheduling_native_q2.py`

Reference: Chain-of-Thought prompting (`https://arxiv.org/abs/2201.11903`).

Command:

```bash
python3 -m cgr_exp.cli run-scheduling-native-q2 \
  --output-dir runs/q2_native_llm_baselines_100_seed0 \
  --seed 0 \
  --train-count 200 \
  --validation-count 50 \
  --test-count 100 \
  --job-count 6 \
  --machine-count 3 \
  --model gpt-4o-mini \
  --timeout 300 \
  --methods direct \
  --parallel-episodes 8 \
  --resume
```

Completion evidence:

- `scheduling_native_q2_table.csv`
- `scheduling_native_q2_per_instance.csv`
- `scheduling_native_q2_samples.jsonl`
- Direct row status `complete`
- `valid_sample_rate`, `parse_failures`, `valid_samples`, and `total_samples`
  are present for audit.

### Self-Consistency

Code path: `cgr_exp/scheduling_native_q2.py`

Reference: self-consistency decoding
(`https://arxiv.org/abs/2203.11171`).

Command:

```bash
python3 -m cgr_exp.cli run-scheduling-native-q2 \
  --output-dir runs/q2_native_llm_baselines_100_seed0 \
  --seed 0 \
  --train-count 200 \
  --validation-count 50 \
  --test-count 100 \
  --job-count 6 \
  --machine-count 3 \
  --model gpt-4o-mini \
  --timeout 300 \
  --methods self_consistency \
  --sample-count 8 \
  --parallel-episodes 8 \
  --resume
```

Completion evidence is the same native audit artifact set. The selection rule
is majority canonical final job order, with objective used only as a tie-break.

### Best-of-N

Code path: `cgr_exp/scheduling_native_q2.py`

Reference: test-time answer sampling with verifier/reward selection.

Command:

```bash
python3 -m cgr_exp.cli run-scheduling-native-q2 \
  --output-dir runs/q2_native_llm_baselines_100_seed0 \
  --seed 0 \
  --train-count 200 \
  --validation-count 50 \
  --test-count 100 \
  --job-count 6 \
  --machine-count 3 \
  --model gpt-4o-mini \
  --timeout 300 \
  --methods best_of_n \
  --sample-count 8 \
  --parallel-episodes 8 \
  --resume
```

This is a verifier-selected Best-of-N variant: valid samples are filtered by
the Scheduling verifier and the lowest verified objective is selected. Table
notes should disclose this stronger selection rule.

### Tree of Thoughts

Code path: `cgr_exp/scheduling_tot_q2.py`

Reference: Princeton NLP Tree-of-Thoughts BFS
(`https://github.com/princeton-nlp/tree-of-thought-llm`).

Command:

```bash
python3 -m cgr_exp.cli run-scheduling-tot-q2 \
  --output-dir runs/q2_tot_bfs_100_seed0 \
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

Completion evidence:

- `scheduling_tot_q2_table.csv`
- `scheduling_tot_q2_per_instance.csv`
- `scheduling_tot_q2_trace.jsonl`
- `table5_native_row.json`
- manifest contains `native_tot_protocol` mapping the official ToT
  `method_generate`, `method_evaluate`, and `method_select` primitives to the
  Scheduling adaptation.

Older `GPT-frontier-rank / ToT-style` controller-isolation artifacts should not
be used as this strict native ToT row. `build-table5-q2` only promotes the ToT
row from native `run-scheduling-tot-q2` artifacts.

### Target-PRM Priority

Code path: `cgr_exp/scheduling_transfer.py`

Reference: process reward modeling / PRM800K-style process supervision
(`https://github.com/openai/prm800k`).

Command:

```bash
python3 -m cgr_exp.cli run-scheduling-llm-pilot \
  --output-dir runs/q2_learned_search_100_seed0 \
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
  --target-prm-train-count 100 \
  --resume
```

Completion evidence:

- Target-PRM row status `complete`
- `target_prm_trace_records.jsonl`
- `target_prm_train_pairs.jsonl`
- `target_prm_ranker.json`

Do not substitute `Outcome-PRM`; it is an ablation, not a PRM-style process
reward row.

### PRM-Guided MCTS (ReST-MCTS-Style)

Code path: `cgr_exp/scheduling_transfer.py::PRMGuidedMCTSStylePolicy`

Reference: THUDM ReST-MCTS MCTS search component
(`https://github.com/THUDM/ReST-MCTS`).

Command:

```bash
python3 -m cgr_exp.cli run-scheduling-llm-pilot \
  --output-dir runs/q2_learned_search_100_seed0 \
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
  --target-prm-train-count 100 \
  --resume
```

This row implements the ReST-MCTS-style search component: PRM priors,
PUCT-style selection, expansion observation, and backup. It does not implement
the full ReST-MCTS iterative self-training loop, so the table name must keep
`style`.

### Solver Prior

Code path: `cgr_exp/scheduling_transfer.py`

Command:

```bash
python3 -m cgr_exp.cli run-scheduling-llm-pilot \
  --output-dir runs/q2_learned_search_100_seed0 \
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
  --resume
```

The controller is trained only on solver-induced source trajectories and then
frozen before real LLM-induced target search.

### Solver Prior + Certification

Code path: `cgr_exp/scheduling_transfer.py` plus
`cgr_exp/certification/prune.py`

Command:

```bash
python3 -m cgr_exp.cli run-scheduling-llm-pilot \
  --output-dir runs/q2_learned_search_100_seed0 \
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
  --include-solver-certification \
  --resume
```

For the final Q2 table, prefer an artifact with `on_demand_total_tokens` so
certification receives compute credit only for proposal calls it actually
avoids.

## Fast Combined Runs

Native frozen baselines can run together:

```bash
python3 -m cgr_exp.cli run-scheduling-native-q2 \
  --output-dir runs/q2_native_llm_baselines_100_seed0 \
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

Learned-search rows can run together:

```bash
python3 -m cgr_exp.cli run-scheduling-llm-pilot \
  --output-dir runs/q2_learned_search_100_seed0 \
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

Run strict native ToT separately because it has a different native
generate/evaluate/select protocol:

```bash
python3 -m cgr_exp.cli run-scheduling-tot-q2 \
  --output-dir runs/q2_tot_bfs_100_seed0 \
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

## Build the Current Q2 Table

```bash
python3 -m cgr_exp.cli build-table5-q2 \
  --output-dir runs/q2_table_current_seed0 \
  --expected-test-count 100 \
  --structured-dir runs/q2_structured_100_seed0 \
  --direct-dir runs/q2_native_llm_baselines_100_seed0 \
  --self-consistency-dir runs/q2_native_llm_baselines_100_seed0 \
  --best-of-n-dir runs/q2_native_llm_baselines_100_seed0 \
  --tot-dir runs/q2_tot_bfs_100_seed0 \
  --target-prm-dir runs/q2_learned_search_100_seed0 \
  --rest-mcts-dir runs/q2_learned_search_100_seed0 \
  --solver-prior-dir runs/q2_learned_search_100_seed0 \
  --solver-certification-dir runs/q2_learned_search_100_seed0 \
  --overwrite
```

## Audit Table Readiness

After building or updating any run directory, run the readiness audit with the
same inputs:

```bash
python3 -m cgr_exp.cli audit-table5-q2 \
  --output-dir runs/q2_table_current_seed0_audit \
  --expected-test-count 100 \
  --structured-dir runs/q2_structured_100_seed0 \
  --direct-dir runs/q2_native_llm_baselines_100_seed0 \
  --self-consistency-dir runs/q2_native_llm_baselines_100_seed0 \
  --best-of-n-dir runs/q2_native_llm_baselines_100_seed0 \
  --tot-dir runs/q2_tot_bfs_100_seed0 \
  --target-prm-dir runs/q2_learned_search_100_seed0 \
  --rest-mcts-dir runs/q2_learned_search_100_seed0 \
  --solver-prior-dir runs/q2_learned_search_100_seed0 \
  --solver-certification-dir runs/q2_learned_search_100_seed0 \
  --overwrite
```

The audit writes:

- `table5_q2_results.csv`
- `table5_q2_results.tex`
- `table5_q2_manifest.json`
- `table5_q2_readiness.csv`
- `table5_q2_readiness.json`

Readiness is artifact-level, not just summary-table-level. Native rows require
their sample audit JSONL, native ToT requires its generate/evaluate/select
trace JSONL, and Target-PRM/ReST-MCTS-style rows require the target PRM ranker,
trace records, and train-pair artifacts. This prevents a hand-edited or
partial summary CSV from being promoted as a source-aligned Q2 result.

Use `table5_q2_readiness.csv` to check row-level blockers before promoting a
number into the paper.

`--gpt-frontier-rank-dir` remains a backwards-compatible alias for `--tot-dir`,
but new commands should use `--tot-dir` because this row requires native ToT
BFS artifacts, not older controller-isolation GPT-frontier-rank diagnostics.

Final acceptance checks:

- No row with `partial:*` is promoted to the main result.
- No row with `artifact missing` is promoted.
- Learned-search Q2 compute rows should use `on-demand per-policy accounting`;
  rows using `shared-cache fallback accounting` need a qualification.
- `method_alignment` is present in each runner manifest and in
  `table5_q2_manifest.json`.
