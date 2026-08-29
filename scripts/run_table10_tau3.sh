#!/usr/bin/env zsh
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
TAU_PYTHON_BIN="${TAU_PYTHON_BIN:-python3}"
AGENT_LLM="${AGENT_LLM:-openai/gpt-4o-mini}"
USER_LLM="${USER_LLM:-openai/gpt-4o-mini}"
NUM_TASKS="${NUM_TASKS:-100}"
NUM_TRIALS="${NUM_TRIALS:-4}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-6}"
WORKERS="${WORKERS:-0}"
TIMEOUT="${TIMEOUT:-300}"
PARALLEL_DOMAINS="${PARALLEL_DOMAINS:-1}"
PARALLEL_METHOD_GROUPS="${PARALLEL_METHOD_GROUPS:-0}"
SEED="${SEED:-0}"
BASE_DIR="${BASE_DIR:-runs/table10_tau3}"
SOURCE_CONTROLLER_DIR="${SOURCE_CONTROLLER_DIR:-${BASE_DIR}/source_controller_seed${SEED}}"
SOLVER_RANKER_CHECKPOINT="${SOLVER_RANKER_CHECKPOINT:-${SOURCE_CONTROLLER_DIR}/generic_harness_ranker.json}"
SOURCE_TRAIN_COUNT="${SOURCE_TRAIN_COUNT:-80}"
SOURCE_VALIDATION_COUNT="${SOURCE_VALIDATION_COUNT:-30}"
SOURCE_EPOCHS="${SOURCE_EPOCHS:-20}"
SOURCE_MAX_STATES="${SOURCE_MAX_STATES:-200}"
BEST_OF_N="${BEST_OF_N:-4}"

usage() {
  cat <<'EOF'
scripts/run_table10_tau3.sh [preflight|preflight-runs|source-controller|native-smoke|native-full|best-of-n|solver-smoke|solver-full|solver-canonical-smoke|solver-canonical-full|solver-canonical-certified-smoke|solver-canonical-certified-full|full|audit-solver-canonical|audit-solver-canonical-certified|audit-table|build]

Environment overrides:
  PYTHON_BIN=python3
  TAU_PYTHON_BIN=/opt/anaconda3/bin/python
  AGENT_LLM=openai/gpt-4o-mini
  USER_LLM=openai/gpt-4o-mini
  NUM_TASKS=100
  NUM_TRIALS=4
  MAX_CONCURRENCY=6
  PARALLEL_DOMAINS=1
  PARALLEL_METHOD_GROUPS=0
  WORKERS=0
  TIMEOUT=300
  BASE_DIR=runs/table10_tau3
  SOURCE_CONTROLLER_DIR=runs/table10_tau3/source_controller_seed0
  SOLVER_RANKER_CHECKPOINT=runs/table10_tau3/source_controller_seed0/generic_harness_ranker.json
  BEST_OF_N=4

For OpenAI via tau2/litellm, set OPENAI_API_KEY. If you only have
CGR_LLM_API_KEY, export OPENAI_API_KEY="$CGR_LLM_API_KEY" first.
EOF
}

run_preflight() {
  "${PYTHON_BIN}" -m cgr_exp.cli preflight-tau3-native \
    --output-dir "${BASE_DIR}/preflight" \
    --tau-python-bin "${TAU_PYTHON_BIN}" \
    --overwrite
}

build_run_preflight() {
  "${PYTHON_BIN}" -m cgr_exp.cli build-table10-tau3-run-preflight \
    --output-dir "${BASE_DIR}/run_preflight" \
    --base-dir "${BASE_DIR}" \
    --expected-test-count "${NUM_TASKS}" \
    --best-of-n "${BEST_OF_N}" \
    --num-trials "${NUM_TRIALS}" \
    --max-concurrency "${MAX_CONCURRENCY}" \
    --agent-llm "${AGENT_LLM}" \
    --user-llm "${USER_LLM}" \
    --tau-python-bin "${TAU_PYTHON_BIN}" \
    --overwrite
}

run_native_domain() {
  local domain="$1"
  local count="$2"
  local suffix="$3"
  local method="${4:-native}"
  local checkpoint="${5:-}"
  local certified="${6:-0}"
  local extra_args=()
  if [[ -n "${checkpoint}" ]]; then
    extra_args+=(--solver-ranker-checkpoint "${checkpoint}")
  fi
  if [[ "${certified}" == "1" ]]; then
    extra_args+=(--enable-certified-filter)
  fi
  "${PYTHON_BIN}" -m cgr_exp.cli run-tau3-native \
    --output-dir "${BASE_DIR}/${suffix}_${domain}" \
    --domain "${domain}" \
    --method "${method}" \
    --task-split-name base \
    --num-tasks "${count}" \
    --num-trials "${NUM_TRIALS}" \
    --agent-llm "${AGENT_LLM}" \
    --user-llm "${USER_LLM}" \
    --agent-llm-args '{"temperature": 0}' \
    --user-llm-args '{"temperature": 0}' \
    --max-steps 200 \
    --max-errors 10 \
    --timeout "${TIMEOUT}" \
    --max-concurrency "${MAX_CONCURRENCY}" \
    --workers "${WORKERS}" \
    --seed "${SEED}" \
    --tau-python-bin "${TAU_PYTHON_BIN}" \
    --auto-resume \
    --overwrite \
    "${extra_args[@]}"
}

run_source_controller() {
  "${PYTHON_BIN}" -m cgr_exp.cli run-workflow-source-validation \
    --output-dir "${SOURCE_CONTROLLER_DIR}" \
    --seed "${SEED}" \
    --train-count "${SOURCE_TRAIN_COUNT}" \
    --validation-count "${SOURCE_VALIDATION_COUNT}" \
    --epochs "${SOURCE_EPOCHS}" \
    --max-states-per-task "${SOURCE_MAX_STATES}" \
    --overwrite
}

run_domain_pair() {
  local count="$1"
  local suffix="$2"
  local method="${3:-native}"
  local checkpoint="${4:-}"
  local certified="${5:-0}"
  if [[ "${PARALLEL_DOMAINS}" == "1" ]]; then
    run_native_domain retail "${count}" "${suffix}" "${method}" "${checkpoint}" "${certified}" &
    local retail_pid=$!
    run_native_domain airline "${count}" "${suffix}" "${method}" "${checkpoint}" "${certified}" &
    local airline_pid=$!
    local status=0
    wait "${retail_pid}" || status=$?
    wait "${airline_pid}" || status=$?
    return "${status}"
  fi
  run_native_domain retail "${count}" "${suffix}" "${method}" "${checkpoint}" "${certified}"
  run_native_domain airline "${count}" "${suffix}" "${method}" "${checkpoint}" "${certified}"
}

run_table10_full() {
  run_preflight
  run_source_controller
  if [[ "${PARALLEL_METHOD_GROUPS}" == "1" ]]; then
    run_domain_pair "${NUM_TASKS}" native &
    local native_pid=$!
    run_domain_pair "${NUM_TASKS}" solver solver &
    local solver_pid=$!
    run_domain_pair "${NUM_TASKS}" solver_canonical solver "${SOLVER_RANKER_CHECKPOINT}" &
    local canonical_pid=$!
    run_domain_pair "${NUM_TASKS}" solver_canonical_certified solver "${SOLVER_RANKER_CHECKPOINT}" 1 &
    local certified_pid=$!
    local status=0
    wait "${native_pid}" || status=$?
    wait "${solver_pid}" || status=$?
    wait "${canonical_pid}" || status=$?
    wait "${certified_pid}" || status=$?
    if [[ "${status}" != "0" ]]; then
      return "${status}"
    fi
  else
    run_domain_pair "${NUM_TASKS}" native
    run_domain_pair "${NUM_TASKS}" solver solver
    run_domain_pair "${NUM_TASKS}" solver_canonical solver "${SOLVER_RANKER_CHECKPOINT}"
    run_domain_pair "${NUM_TASKS}" solver_canonical_certified solver "${SOLVER_RANKER_CHECKPOINT}" 1
  fi
  build_best_of_n
  audit_solver_canonical_traces
  audit_solver_canonical_certified_traces
  build_table
}

build_best_of_n() {
  "${PYTHON_BIN}" -m cgr_exp.cli build-tau3-best-of-n \
    --output-dir "${BASE_DIR}/best_of_${BEST_OF_N}_retail" \
    --native-dir "${BASE_DIR}/native_retail" \
    --n "${BEST_OF_N}" \
    --overwrite
  "${PYTHON_BIN}" -m cgr_exp.cli build-tau3-best-of-n \
    --output-dir "${BASE_DIR}/best_of_${BEST_OF_N}_airline" \
    --native-dir "${BASE_DIR}/native_airline" \
    --n "${BEST_OF_N}" \
    --overwrite
}

build_table() {
  "${PYTHON_BIN}" -m cgr_exp.cli build-table10-tau3 \
    --output-dir "${BASE_DIR}/table10_current" \
    --expected-test-count "${NUM_TASKS}" \
    --row-dirs \
      "${BASE_DIR}/native_retail" \
      "${BASE_DIR}/native_airline" \
      "${BASE_DIR}/best_of_${BEST_OF_N}_retail" \
      "${BASE_DIR}/best_of_${BEST_OF_N}_airline" \
      "${BASE_DIR}/solver_retail" \
      "${BASE_DIR}/solver_airline" \
      "${BASE_DIR}/solver_canonical_retail" \
      "${BASE_DIR}/solver_canonical_airline" \
      "${BASE_DIR}/solver_canonical_certified_retail" \
      "${BASE_DIR}/solver_canonical_certified_airline" \
    --overwrite
}

audit_table() {
  "${PYTHON_BIN}" -m cgr_exp.cli audit-table10-tau3 \
    --output-dir "${BASE_DIR}/table10_readiness" \
    --expected-test-count "${NUM_TASKS}" \
    --row-dirs \
      "${BASE_DIR}/native_retail" \
      "${BASE_DIR}/native_airline" \
      "${BASE_DIR}/best_of_${BEST_OF_N}_retail" \
      "${BASE_DIR}/best_of_${BEST_OF_N}_airline" \
      "${BASE_DIR}/solver_retail" \
      "${BASE_DIR}/solver_airline" \
      "${BASE_DIR}/solver_canonical_retail" \
      "${BASE_DIR}/solver_canonical_airline" \
      "${BASE_DIR}/solver_canonical_certified_retail" \
      "${BASE_DIR}/solver_canonical_certified_airline" \
    --overwrite
}

audit_solver_canonical_traces() {
  "${PYTHON_BIN}" -m cgr_exp.cli audit-tau3-controller-trace \
    --output-dir "${BASE_DIR}/solver_canonical_retail_trace_audit" \
    --trace-path "${BASE_DIR}/solver_canonical_retail/tau3_controller_trace.jsonl" \
    --overwrite
  "${PYTHON_BIN}" -m cgr_exp.cli audit-tau3-controller-trace \
    --output-dir "${BASE_DIR}/solver_canonical_airline_trace_audit" \
    --trace-path "${BASE_DIR}/solver_canonical_airline/tau3_controller_trace.jsonl" \
    --overwrite
}

audit_solver_canonical_certified_traces() {
  "${PYTHON_BIN}" -m cgr_exp.cli audit-tau3-controller-trace \
    --output-dir "${BASE_DIR}/solver_canonical_certified_retail_trace_audit" \
    --trace-path "${BASE_DIR}/solver_canonical_certified_retail/tau3_controller_trace.jsonl" \
    --overwrite
  "${PYTHON_BIN}" -m cgr_exp.cli audit-tau3-controller-trace \
    --output-dir "${BASE_DIR}/solver_canonical_certified_airline_trace_audit" \
    --trace-path "${BASE_DIR}/solver_canonical_certified_airline/tau3_controller_trace.jsonl" \
    --overwrite
}

case "${1:-}" in
  preflight)
    run_preflight
    ;;
  preflight-runs)
    build_run_preflight
    ;;
  source-controller)
    run_source_controller
    ;;
  native-smoke)
    run_preflight
    run_domain_pair 2 native_smoke
    ;;
  native-full)
    run_preflight
    run_domain_pair "${NUM_TASKS}" native
    build_best_of_n
    build_table
    ;;
  best-of-n)
    build_best_of_n
    ;;
  solver-smoke)
    run_preflight
    run_domain_pair 2 solver_smoke solver
    ;;
  solver-full)
    run_preflight
    run_domain_pair "${NUM_TASKS}" solver solver
    ;;
  solver-canonical-smoke)
    run_preflight
    run_source_controller
    run_domain_pair 2 solver_canonical_smoke solver "${SOLVER_RANKER_CHECKPOINT}"
    ;;
  solver-canonical-full)
    run_preflight
    run_source_controller
    run_domain_pair "${NUM_TASKS}" solver_canonical solver "${SOLVER_RANKER_CHECKPOINT}"
    audit_solver_canonical_traces
    ;;
  solver-canonical-certified-smoke)
    run_preflight
    run_source_controller
    run_domain_pair 2 solver_canonical_certified_smoke solver "${SOLVER_RANKER_CHECKPOINT}" 1
    ;;
  solver-canonical-certified-full)
    run_preflight
    run_source_controller
    run_domain_pair "${NUM_TASKS}" solver_canonical_certified solver "${SOLVER_RANKER_CHECKPOINT}" 1
    audit_solver_canonical_certified_traces
    ;;
  full)
    run_table10_full
    ;;
  audit-solver-canonical)
    audit_solver_canonical_traces
    ;;
  audit-solver-canonical-certified)
    audit_solver_canonical_certified_traces
    ;;
  audit-table)
    audit_table
    ;;
  build)
    build_table
    audit_table
    ;;
  *)
    usage
    exit 2
    ;;
esac
