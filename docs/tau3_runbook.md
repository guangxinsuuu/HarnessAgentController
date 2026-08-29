# Tau3 Canonical Harness Runbook

This runbook is for Table10 tau3/tau2 maintained-protocol evaluation.
It keeps native end-to-end rows, Best-of-N reuse, legacy solver-controller
rows, canonical controller rows, and certified-controller rows under one
auditable protocol.

## Protocol

- Use tau2/tau3 official text half-duplex runtime, user simulator, task
  verifier, and pass-k aggregation.
- Use either `OPENAI_API_KEY` or `CGR_LLM_API_KEY` for API-backed rows. The
  runner maps `CGR_LLM_API_KEY` into `OPENAI_API_KEY` before calling
  tau2/litellm, so setting either one is enough when using
  `scripts/run_table10_tau3.sh`.
- If `PYTHON_BIN` is not set, the runner selects `/opt/anaconda3/bin/python`
  when it exists, then falls back to `python3`. `TAU_PYTHON_BIN` defaults to
  the same interpreter unless explicitly overridden.
- Build the source controller before canonical tau3 runs. This is local and
  does not use the API.
- Promote rows only when `table10_tau3_readiness.csv` marks them ready.
- Do not use tau3 train trajectories, tau3 held-out outcomes, or oracle labels
  as controller inference features.

## Preflight

```bash
cd /Users/neilsu/Desktop/iclr27
PYTHON_BIN=/opt/anaconda3/bin/python \
TAU_PYTHON_BIN=/opt/anaconda3/bin/python \
BASE_DIR=runs/table10_tau3 \
NUM_TASKS=100 \
NUM_TRIALS=4 \
SEED=0 \
scripts/run_table10_tau3.sh preflight-runs
```

Read:

```bash
cat runs/table10_tau3/run_preflight/table10_tau3_run_preflight.md
```

If the source controller is missing, the recommended next command should be:

```bash
BASE_DIR=runs/table10_tau3 SEED=0 scripts/run_table10_tau3.sh source-controller
```

Once the source controller is ready, preflight should recommend
`solver-canonical-smoke` before any full row. This keeps the canonical adapter,
ranker checkpoint, trace schema, and smoke gate checked before the expensive
100-task API-backed runs.

## Source Controller

Fast local smoke, no API:

```bash
PYTHON_BIN=/opt/anaconda3/bin/python \
BASE_DIR=runs/table10_tau3_local_smoke \
SEED=0 \
scripts/run_table10_tau3.sh source-controller-smoke
```

Full source-controller build, no API:

```bash
PYTHON_BIN=/opt/anaconda3/bin/python \
BASE_DIR=runs/table10_tau3 \
SEED=0 \
SOURCE_TRAIN_COUNT=80 \
SOURCE_VALIDATION_COUNT=30 \
SOURCE_EPOCHS=20 \
SOURCE_MAX_STATES=200 \
scripts/run_table10_tau3.sh source-controller
```

Both source-controller targets refresh `progress/` and `run_preflight/` after
the checkpoint is written. The next recommended command should then advance to
`solver-canonical-smoke`.

Required source artifacts:

- `source_controller_seed0/generic_harness_ranker.json`
- `source_controller_seed0/canonical_harness_source_validation.json`
- `source_controller_seed0/canonical_harness_tool_identity_stress.csv`
- `source_controller_seed0/canonical_harness_contract_coverage_stress.csv`

## Smoke Runs

Set either API key before smoke or full API-backed runs:

```bash
export CGR_LLM_API_KEY="..."
# or:
export OPENAI_API_KEY="..."
```

Canonical controller smoke:

```bash
PYTHON_BIN=/opt/anaconda3/bin/python \
TAU_PYTHON_BIN=/opt/anaconda3/bin/python \
AGENT_LLM=openai/gpt-4o-mini \
USER_LLM=openai/gpt-4o-mini \
BASE_DIR=runs/table10_tau3 \
NUM_TRIALS=4 \
MAX_CONCURRENCY=4 \
CANDIDATE_WORKERS=4 \
CANDIDATE_RETRIES=2 \
SEED=0 \
scripts/run_table10_tau3.sh solver-canonical-smoke
```

Certified canonical controller smoke:

```bash
PYTHON_BIN=/opt/anaconda3/bin/python \
TAU_PYTHON_BIN=/opt/anaconda3/bin/python \
AGENT_LLM=openai/gpt-4o-mini \
USER_LLM=openai/gpt-4o-mini \
BASE_DIR=runs/table10_tau3 \
NUM_TRIALS=4 \
MAX_CONCURRENCY=4 \
CANDIDATE_WORKERS=4 \
CANDIDATE_RETRIES=2 \
SEED=0 \
scripts/run_table10_tau3.sh solver-canonical-certified-smoke
```

Smoke targets automatically write trace audits next to the smoke rows:

- `solver_canonical_smoke_retail_trace_audit/tau3_controller_trace_audit.csv`
- `solver_canonical_smoke_airline_trace_audit/tau3_controller_trace_audit.csv`
- `solver_canonical_certified_smoke_retail_trace_audit/tau3_controller_trace_audit.csv`
- `solver_canonical_certified_smoke_airline_trace_audit/tau3_controller_trace_audit.csv`

For canonical smoke to be valid, the trace audit should show
`canonical_feature_schema_match=True`, `canonical_feature_count_match=True`,
`ranker_checkpoint_loaded_events>0`, and all oracle visibility/use flags
`False`.

They also write a promotion gate:

- `solver_canonical_smoke_gate/table10_tau3_smoke_gate.csv`
- `solver_canonical_certified_smoke_gate/table10_tau3_smoke_gate.csv`

Only move from smoke to full when the corresponding gate has
`all_smoke_rows_ready=True` in `table10_tau3_smoke_gate.json`. The smoke
targets call the gate with `--fail-on-not-ready`, so a failed gate exits
nonzero instead of silently producing an unusable smoke row.
Successful smoke and full targets refresh `progress/` and `run_preflight/`
automatically, so rerun `preflight-runs` manually only when inspecting a
partially failed command.

## Canonical Controller Suite

To run only the canonical controller rows for our method, including smoke
gates and the certified variant, use:

```bash
PYTHON_BIN=/opt/anaconda3/bin/python \
TAU_PYTHON_BIN=/opt/anaconda3/bin/python \
AGENT_LLM=openai/gpt-4o-mini \
USER_LLM=openai/gpt-4o-mini \
BASE_DIR=runs/table10_tau3 \
NUM_TASKS=100 \
NUM_TRIALS=4 \
MAX_CONCURRENCY=6 \
CANDIDATE_WORKERS=4 \
CANDIDATE_RETRIES=2 \
PARALLEL_DOMAINS=1 \
SEED=0 \
scripts/run_table10_tau3.sh solver-canonical-suite
```

This target runs canonical smoke, certified canonical smoke, canonical full,
certified canonical full, trace audits, smoke gates, table build, progress,
and run preflight. It skips native, Best-of-N, and legacy solver-controller
rows.

## Full Rows

Run all Table10 rows:

```bash
PYTHON_BIN=/opt/anaconda3/bin/python \
TAU_PYTHON_BIN=/opt/anaconda3/bin/python \
AGENT_LLM=openai/gpt-4o-mini \
USER_LLM=openai/gpt-4o-mini \
BASE_DIR=runs/table10_tau3 \
NUM_TASKS=100 \
NUM_TRIALS=4 \
MAX_CONCURRENCY=6 \
CANDIDATE_WORKERS=4 \
CANDIDATE_RETRIES=2 \
PARALLEL_DOMAINS=1 \
PARALLEL_METHOD_GROUPS=0 \
SEED=0 \
scripts/run_table10_tau3.sh full
```

For faster but heavier API usage:

```bash
PARALLEL_METHOD_GROUPS=1 scripts/run_table10_tau3.sh full
```

Preflight reports two concurrency numbers. `Raw effective max concurrency`
is the configured request capacity after domain, method-group, and
candidate-worker multipliers. `Effective max concurrency` caps that raw value
by the number of runnable tau2 simulations in the selected run group. For
example, a smoke group with only 16 simulations can report raw capacity 48 but
effective concurrency 16.
With `PARALLEL_DOMAINS=1`, raw native API concurrency is roughly
`2 * MAX_CONCURRENCY`. With `PARALLEL_METHOD_GROUPS=1`, the `full` target can
reach roughly `8 * MAX_CONCURRENCY` before the runnable-work cap.
For solver-controller rows, `CANDIDATE_WORKERS` also parallelizes the internal
candidate samples at each decision point; set `CANDIDATE_WORKERS=1` for a
strictly serial controller run.
`CANDIDATE_RETRIES` controls per-candidate transient API retry attempts; trace
audits report `candidate_sample_failures` after retries.

## Progress

Check progress without making API calls:

```bash
BASE_DIR=runs/table10_tau3 \
NUM_TASKS=100 \
BEST_OF_N=4 \
scripts/run_table10_tau3.sh progress
```

Read:

```bash
cat runs/table10_tau3/progress/table10_tau3_progress.md
```

This scans existing `q4_tau_row.json` and partial
`official_tau2_results/results.json` artifacts. It reports per-row task
coverage, simulation coverage, `pass^1`, `pass^4`, calls, tokens, and
wall-clock estimates when available.

## Multi-Seed

Run seeds into separate base directories or keep the same base directory with
seed-specific source-controller subdirectories:

```bash
for seed in 0 1 2; do
  BASE_DIR="runs/table10_tau3_seed${seed}" \
  SEED="${seed}" \
  NUM_TASKS=100 \
  NUM_TRIALS=4 \
  MAX_CONCURRENCY=6 \
  scripts/run_table10_tau3.sh full
done
```

The preflight is seed-aware and should point to
`source_controller_seed${seed}/generic_harness_ranker.json`.

## Resume

API-backed tau3 runs use the official tau2 `--auto-resume` path. If a run
stops after partial simulations, rerun the same command with the same
`BASE_DIR`, `SEED`, `NUM_TASKS`, and `NUM_TRIALS`.

Generated commands use `--auto-resume --overwrite`. Here `--overwrite` only
allows writing command manifests and aggregate files into an existing output
directory. It does not force a completed tau2 result to rerun. Use `--force`
only when intentionally discarding/recomputing an existing row.

Do not change the source-controller checkpoint, model, or task count when
resuming a row intended for the same table.
API-backed canonical runs reuse an existing `SOLVER_RANKER_CHECKPOINT` by
default. Set `FORCE_SOURCE_CONTROLLER=1` only when intentionally rebuilding the
source controller as a new experiment version.

## Promotion Gate

```bash
BASE_DIR=runs/table10_tau3 \
NUM_TASKS=100 \
BEST_OF_N=4 \
scripts/run_table10_tau3.sh build
```

For the paper-wide readiness audit, run:

```bash
TAU3_TABLE_BASE_DIR=runs/table10_tau3 \
TEST_COUNT=100 \
scripts/run_paper_audits.sh tau3
```

This audits the 10 maintained-protocol Table10 rows:

- native retail and airline
- Best-of-N retail and airline
- legacy solver-controller retail and airline
- canonical solver-controller retail and airline
- certified canonical solver-controller retail and airline

Promote only rows whose readiness audit has `result_ready=True`.
Canonical rows must show evidence for:

- source generator frozen before target test
- no tau3 trajectory training
- no oracle label inference leakage
- anonymous source tools
- tool-identity stress passed
- contract-coverage stress passed
- canonical ranker trace events
- complete tool call, LLM call, token, and wall-clock accounting

Routine status generation should write to `runs/current_experiment_status`,
which is the default used by `scripts/run_paper_audits.sh`. Only write status
artifacts directly under `docs/` when intentionally refreshing a checked-in
paper snapshot.
