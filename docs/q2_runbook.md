# Q2 Scheduling Runbook

This runbook lists the code path, method contract, and command needed for each
row in the Q2 end-to-end Scheduling table. It is intentionally strict: rows
with partial artifacts, diagnostic-only artifacts, or shared-cache fallback
token accounting should remain marked as pending or qualified.

For the row-level algorithm requirements and source-aligned claim boundaries,
see `docs/q2_table4_required_algorithms.md`.

## Global Setup

Real-LLM rows require an OpenAI-compatible chat-completions endpoint:

```bash
export CGR_LLM_API_KEY="..."
export CGR_LLM_MODEL="gpt-4o-mini"
```

Do not commit API keys or run outputs. `runs/*`, `.env*`, `.venv/`, and
editable-install metadata are ignored.

Structured CP-SAT rows require OR-Tools in the Python interpreter used by the
runner. On this machine, use:

```bash
export PYTHON_BIN=.venv/bin/python
```

The Q2 run script refuses to run `structured` or `all` if
`${PYTHON_BIN}` cannot import `ortools.sat.python.cp_model`, because that would
otherwise produce a partial CP-SAT artifact.

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

The Q2 readiness audit also checks coverage against `--expected-test-count`.
Native rows need sample records for the expected number of distinct test tasks,
and the native ToT row needs finish events for the expected number of distinct
renderings. A summary row marked `complete` is not enough by itself.

Before promoting any Q2 table, run the local preflight:

```bash
PYTHON_BIN=.venv/bin/python scripts/run_q2_table4.sh preflight
```

This writes the implementation matrix, source-evidence audit, source-ref
verification, and table readiness audit without using the LLM API. The
source-ref verifier checks pinned GitHub full SHAs, pinned raw GitHub URLs,
symbol/schema anchors, and explicit arXiv versions. For a network reachability
check of the same refs, run:

```bash
python3 -m cgr_exp.cli verify-q2-source-refs \
  --output-dir runs/q2_source_ref_verification_online \
  --online \
  --timeout 10 \
  --overwrite
```

Structured, native, ToT, and learned-search artifacts also embed
`source_evidence` for the external methods they instantiate. If an older run
directory lacks this field, the Q2 readiness audit keeps that row pending until
the row is rerun with the current code.

## Row Contracts and Commands

### CP-SAT

Code path: `cgr_exp/scheduling_structured_q2.py`

Reference: Google OR-Tools CP-SAT scheduling models
(`https://github.com/google/or-tools`).

Command:

```bash
.venv/bin/python -m cgr_exp.cli run-scheduling-structured-q2 \
  --output-dir runs/q2_structured_100_seed0_runtime_v2 \
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
  --output-dir runs/q2_native_llm_baselines_100_seed0_sc_v2 \
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
  --output-dir runs/q2_native_llm_baselines_100_seed0_sc_v2 \
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
is majority canonical final job order, with the earliest valid sample used to
break ties so the row does not become verifier-selected Best-of-N.

### Best-of-N

Code path: `cgr_exp/scheduling_native_q2.py`

Reference: test-time answer sampling with verifier/reward selection.

Command:

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

Completion evidence:

- `scheduling_tot_q2_table.csv`
- `scheduling_tot_q2_per_instance.csv`
- `scheduling_tot_q2_trace.jsonl`
- `table5_native_row.json`
- manifest contains `native_tot_protocol` mapping the official ToT
  `method_generate`, `method_evaluate`, and `method_select` primitives to the
  Scheduling adaptation.

Older `GPT-frontier-rank / ToT-style` controller-isolation artifacts should not
be used as this strict native ToT row. `build-table4-q2` only promotes the ToT
row from native `run-scheduling-tot-q2` artifacts.

### Target-PRM Priority

Code path: `cgr_exp/scheduling_transfer.py`

Reference: process reward modeling / PRM800K-style process supervision
(`https://github.com/openai/prm800k`).

Command:

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
  --include-solver-certification \
  --resume
```

For the final Q2 table, prefer an artifact with `on_demand_total_tokens` so
certification receives compute credit only for proposal calls it actually
avoids.

### Clean Certification / Grounding Robustness

Clean certification rows for the paper's certification table come from
`run-scheduling-certification`. The runner writes both the historical
`scheduling_certification_table.csv` and the table-specific alias
`table6_clean_certification.csv`. Its manifest records the authority contract:
learned scores may rank frontier states, but only verified lower-bound or
feasibility logic may authorize irreversible pruning.

Grounding/pruning robustness rows come from `run-table6-authority`; despite the
legacy command name, it now also writes `table7_grounding_robustness.csv`. This
artifact is the corruption sweep over verified false negatives, verified false
positives, uncertain-only conditions, and the learned-score-pruning stress test.
It audits each prune/rejection against the uncorrupted descendant subproblem and
records that uncertain constraints have no pruning authority.

## Fast Combined Runs

The checked-in helper script mirrors the commands below:

```bash
scripts/run_q2_table4.sh all
```

It never sets or prints API keys. Set `CGR_LLM_API_KEY` and optionally
`CGR_LLM_MODEL` in your shell first. `scripts/run_q2_table4.sh llm` runs the
native frozen baselines, learned-search rows, and strict ToT BFS row as
separate background jobs, while each runner still uses its own
`--parallel-episodes` setting internally.

Before spending API tokens, run:

```bash
PYTHON_BIN=.venv/bin/python scripts/run_q2_table4.sh preflight
```

This checks the local CLI, reports whether an LLM key is present without
printing it, reports OR-Tools CP-SAT availability, and writes the current
`audit-table4-q2` readiness report.

Native frozen baselines can run together:

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

Learned-search rows can run together:

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

Run strict native ToT separately because it has a different native
generate/evaluate/select protocol:

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

## Build the Current Q2 Table

```bash
python3 -m cgr_exp.cli build-table4-q2 \
  --output-dir runs/q2_table_current_seed0_protocol_v2 \
  --expected-test-count 100 \
  --structured-dir runs/q2_structured_100_seed0_runtime_v2 \
  --direct-dir runs/q2_native_llm_baselines_100_seed0_sc_v2 \
  --self-consistency-dir runs/q2_native_llm_baselines_100_seed0_sc_v2 \
  --best-of-n-dir runs/q2_native_llm_baselines_100_seed0_sc_v2 \
  --tot-dir runs/q2_tot_bfs_100_seed0_tot_gate_v2 \
  --target-prm-dir runs/q2_learned_search_100_seed0_rest_gate_v2 \
  --rest-mcts-dir runs/q2_learned_search_100_seed0_rest_gate_v2 \
  --solver-prior-dir runs/q2_learned_search_100_seed0_rest_gate_v2 \
  --solver-certification-dir runs/q2_learned_search_100_seed0_rest_gate_v2 \
  --overwrite
```

## Audit Table Readiness

After building or updating any run directory, run the readiness audit with the
same inputs:

```bash
python3 -m cgr_exp.cli audit-table4-q2 \
  --output-dir runs/q2_table_current_seed0_protocol_v2_audit \
  --expected-test-count 100 \
  --structured-dir runs/q2_structured_100_seed0_runtime_v2 \
  --direct-dir runs/q2_native_llm_baselines_100_seed0_sc_v2 \
  --self-consistency-dir runs/q2_native_llm_baselines_100_seed0_sc_v2 \
  --best-of-n-dir runs/q2_native_llm_baselines_100_seed0_sc_v2 \
  --tot-dir runs/q2_tot_bfs_100_seed0_tot_gate_v2 \
  --target-prm-dir runs/q2_learned_search_100_seed0_rest_gate_v2 \
  --rest-mcts-dir runs/q2_learned_search_100_seed0_rest_gate_v2 \
  --solver-prior-dir runs/q2_learned_search_100_seed0_rest_gate_v2 \
  --solver-certification-dir runs/q2_learned_search_100_seed0_rest_gate_v2 \
  --overwrite
```

The audit writes the legacy-prefixed artifact files:

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
Completed rows must also carry `method_alignment` or `q2_method_alignment`
entries with the expected `source_primitives` from
`cgr_exp/q2_method_contracts.py`. Old artifacts generated before this field was
added are intentionally blocked by the audit and should be rerun rather than
patched by hand.

Use `table5_q2_readiness.csv` to check row-level blockers before promoting a
number into the paper.

`build-table5-q2` and `audit-table5-q2` remain backwards-compatible aliases for
the same Q2 Table 4 builder. New commands should use `build-table4-q2` and
`audit-table4-q2`. `--gpt-frontier-rank-dir` remains a backwards-compatible
alias for `--tot-dir`, but new commands should use `--tot-dir` because this row
requires native ToT BFS artifacts, not older controller-isolation
GPT-frontier-rank diagnostics.

Final acceptance checks:

- No row with `partial:*` is promoted to the main result.
- No row with `artifact missing` is promoted.
- Learned-search Q2 compute rows should use `on-demand per-policy accounting`;
  rows using `shared-cache fallback accounting` need a qualification.
- `method_alignment` is present in each runner manifest and in
  `table5_q2_manifest.json`.
