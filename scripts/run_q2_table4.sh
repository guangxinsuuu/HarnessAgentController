#!/usr/bin/env zsh
set -euo pipefail

MODE="${1:-all}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL="${CGR_LLM_MODEL:-gpt-4o-mini}"
SEED="${SEED:-0}"
TRAIN_COUNT="${TRAIN_COUNT:-200}"
VALIDATION_COUNT="${VALIDATION_COUNT:-50}"
TEST_COUNT="${TEST_COUNT:-100}"
JOB_COUNT="${JOB_COUNT:-6}"
MACHINE_COUNT="${MACHINE_COUNT:-3}"
NODE_BUDGET="${NODE_BUDGET:-80}"
ACTION_BUDGET="${ACTION_BUDGET:-6}"
RECORDS_PER_INSTANCE="${RECORDS_PER_INSTANCE:-60}"
RANK_EPOCHS="${RANK_EPOCHS:-25}"
TIMEOUT="${TIMEOUT:-300}"
NATIVE_PARALLEL_EPISODES="${NATIVE_PARALLEL_EPISODES:-8}"
SEARCH_PARALLEL_EPISODES="${SEARCH_PARALLEL_EPISODES:-6}"
TOT_PARALLEL_EPISODES="${TOT_PARALLEL_EPISODES:-6}"
BATCH_SIZE="${BATCH_SIZE:-4}"
SAMPLE_COUNT="${SAMPLE_COUNT:-8}"
TARGET_PRM_TRAIN_COUNT="${TARGET_PRM_TRAIN_COUNT:-100}"
TOT_BREADTH="${TOT_BREADTH:-5}"
TOT_VALUE_SAMPLES="${TOT_VALUE_SAMPLES:-3}"
PERTURBATION="${PERTURBATION:-mixed}"

prefer_existing_dir() {
  local primary="$1"
  shift
  if [[ -d "${primary}" ]]; then
    print "${primary}"
    return
  fi
  local candidate
  for candidate in "$@"; do
    if [[ -d "${candidate}" ]]; then
      print "${candidate}"
      return
    fi
  done
  print "${primary}"
}

if [[ -z "${STRUCTURED_DIR:-}" ]]; then
  STRUCTURED_DIR="$(prefer_existing_dir runs/q2_structured_100_seed0_runtime_v2 runs/q2_structured_100_seed0)"
fi
if [[ -z "${NATIVE_DIR:-}" ]]; then
  NATIVE_DIR="$(prefer_existing_dir runs/q2_native_llm_baselines_100_seed0_sc_v2 runs/q2_native_llm_baselines_100_seed0)"
fi
if [[ -z "${LEARNED_DIR:-}" ]]; then
  LEARNED_DIR="$(prefer_existing_dir runs/q2_learned_search_100_seed0_rest_gate_v2 runs/scheduling_table5_prm_rest_cert_100_seed0 runs/scheduling_llm_pilot_real_100_seed0)"
fi
if [[ -z "${TOT_DIR:-}" ]]; then
  TOT_DIR="$(prefer_existing_dir runs/q2_tot_bfs_100_seed0_tot_gate_v2 runs/scheduling_missing_gpt_frontier_rank_100_seed0)"
fi
TABLE_DIR="${TABLE_DIR:-runs/q2_table_current_seed0_protocol_v2}"
AUDIT_DIR="${AUDIT_DIR:-runs/q2_table_current_seed0_protocol_v2_audit}"
SOURCE_AUDIT_DIR="${SOURCE_AUDIT_DIR:-runs/q2_source_evidence_audit}"
SOURCE_REF_DIR="${SOURCE_REF_DIR:-runs/q2_source_ref_verification}"
IMPLEMENTATION_MATRIX_DIR="${IMPLEMENTATION_MATRIX_DIR:-runs/q2_implementation_matrix}"
RUN_PREFLIGHT_DIR="${RUN_PREFLIGHT_DIR:-runs/q2_run_preflight}"

usage() {
  cat <<'EOF'
Usage:
  scripts/run_q2_table4.sh [all|preflight|implementation-matrix|source-audit|source-ref-verify|structured|llm|native|learned|tot|build|audit]

Environment:
  CGR_LLM_API_KEY must be set for llm/native/learned/tot/all.
  CGR_LLM_MODEL defaults to gpt-4o-mini.

The script never sets or prints API keys.
EOF
}

require_llm_key() {
  if [[ -z "${CGR_LLM_API_KEY:-}" ]]; then
    print -u2 "missing CGR_LLM_API_KEY"
    exit 2
  fi
}

require_cpsat() {
  if ! "${PYTHON_BIN}" -c "from ortools.sat.python import cp_model" >/dev/null 2>&1; then
    print -u2 "missing OR-Tools CP-SAT in ${PYTHON_BIN}; set PYTHON_BIN=.venv/bin/python or install ortools before running structured/Q2-all"
    exit 2
  fi
}

run_structured() {
  require_cpsat
  "${PYTHON_BIN}" -m cgr_exp.cli run-scheduling-structured-q2 \
    --output-dir "${STRUCTURED_DIR}" \
    --seed "${SEED}" \
    --train-count "${TRAIN_COUNT}" \
    --validation-count "${VALIDATION_COUNT}" \
    --test-count "${TEST_COUNT}" \
    --job-count "${JOB_COUNT}" \
    --machine-count "${MACHINE_COUNT}" \
    --cpsat-time-limit-seconds 5 \
    --overwrite
}

run_native() {
  require_llm_key
  "${PYTHON_BIN}" -m cgr_exp.cli run-scheduling-native-q2 \
    --output-dir "${NATIVE_DIR}" \
    --seed "${SEED}" \
    --train-count "${TRAIN_COUNT}" \
    --validation-count "${VALIDATION_COUNT}" \
    --test-count "${TEST_COUNT}" \
    --job-count "${JOB_COUNT}" \
    --machine-count "${MACHINE_COUNT}" \
    --model "${MODEL}" \
    --timeout "${TIMEOUT}" \
    --methods direct,self_consistency,best_of_n \
    --sample-count "${SAMPLE_COUNT}" \
    --parallel-episodes "${NATIVE_PARALLEL_EPISODES}" \
    --resume
}

run_learned() {
  require_llm_key
  "${PYTHON_BIN}" -m cgr_exp.cli run-scheduling-llm-pilot \
    --output-dir "${LEARNED_DIR}" \
    --seed "${SEED}" \
    --train-count "${TRAIN_COUNT}" \
    --validation-count "${VALIDATION_COUNT}" \
    --test-count "${TEST_COUNT}" \
    --job-count "${JOB_COUNT}" \
    --machine-count "${MACHINE_COUNT}" \
    --node-budget "${NODE_BUDGET}" \
    --action-budget "${ACTION_BUDGET}" \
    --records-per-instance "${RECORDS_PER_INSTANCE}" \
    --rank-epochs "${RANK_EPOCHS}" \
    --model "${MODEL}" \
    --timeout "${TIMEOUT}" \
    --perturbation "${PERTURBATION}" \
    --batch-size "${BATCH_SIZE}" \
    --parallel-episodes "${SEARCH_PARALLEL_EPISODES}" \
    --include-target-prm \
    --include-rest-mcts \
    --include-solver-certification \
    --target-prm-train-count "${TARGET_PRM_TRAIN_COUNT}" \
    --resume
}

run_tot() {
  require_llm_key
  "${PYTHON_BIN}" -m cgr_exp.cli run-scheduling-tot-q2 \
    --output-dir "${TOT_DIR}" \
    --seed "${SEED}" \
    --train-count "${TRAIN_COUNT}" \
    --validation-count "${VALIDATION_COUNT}" \
    --test-count "${TEST_COUNT}" \
    --job-count "${JOB_COUNT}" \
    --machine-count "${MACHINE_COUNT}" \
    --node-budget "${NODE_BUDGET}" \
    --action-budget "${ACTION_BUDGET}" \
    --breadth "${TOT_BREADTH}" \
    --value-samples "${TOT_VALUE_SAMPLES}" \
    --model "${MODEL}" \
    --timeout "${TIMEOUT}" \
    --parallel-episodes "${TOT_PARALLEL_EPISODES}" \
    --perturbation "${PERTURBATION}" \
    --resume
}

build_table() {
  "${PYTHON_BIN}" -m cgr_exp.cli build-table4-q2 \
    --output-dir "${TABLE_DIR}" \
    --expected-test-count "${TEST_COUNT}" \
    --structured-dir "${STRUCTURED_DIR}" \
    --direct-dir "${NATIVE_DIR}" \
    --self-consistency-dir "${NATIVE_DIR}" \
    --best-of-n-dir "${NATIVE_DIR}" \
    --tot-dir "${TOT_DIR}" \
    --target-prm-dir "${LEARNED_DIR}" \
    --rest-mcts-dir "${LEARNED_DIR}" \
    --solver-prior-dir "${LEARNED_DIR}" \
    --solver-certification-dir "${LEARNED_DIR}" \
    --overwrite
}

audit_table() {
  "${PYTHON_BIN}" -m cgr_exp.cli audit-table4-q2 \
    --output-dir "${AUDIT_DIR}" \
    --expected-test-count "${TEST_COUNT}" \
    --structured-dir "${STRUCTURED_DIR}" \
    --direct-dir "${NATIVE_DIR}" \
    --self-consistency-dir "${NATIVE_DIR}" \
    --best-of-n-dir "${NATIVE_DIR}" \
    --tot-dir "${TOT_DIR}" \
    --target-prm-dir "${LEARNED_DIR}" \
    --rest-mcts-dir "${LEARNED_DIR}" \
    --solver-prior-dir "${LEARNED_DIR}" \
    --solver-certification-dir "${LEARNED_DIR}" \
    --overwrite
}

source_audit() {
  "${PYTHON_BIN}" -m cgr_exp.cli audit-q2-source-evidence \
    --output-dir "${SOURCE_AUDIT_DIR}" \
    --overwrite
}

source_ref_verify() {
  "${PYTHON_BIN}" -m cgr_exp.cli verify-q2-source-refs \
    --output-dir "${SOURCE_REF_DIR}" \
    --overwrite
}

implementation_matrix() {
  "${PYTHON_BIN}" -m cgr_exp.cli build-q2-implementation-matrix \
    --output-dir "${IMPLEMENTATION_MATRIX_DIR}" \
    --overwrite
}

preflight() {
  "${PYTHON_BIN}" -m cgr_exp.cli audit-table4-q2 --help >/dev/null
  "${PYTHON_BIN}" -m cgr_exp.cli audit-q2-source-evidence --help >/dev/null
  "${PYTHON_BIN}" -m cgr_exp.cli verify-q2-source-refs --help >/dev/null
  "${PYTHON_BIN}" -m cgr_exp.cli build-q2-implementation-matrix --help >/dev/null
  if "${PYTHON_BIN}" -c "from ortools.sat.python import cp_model" >/dev/null 2>&1; then
    cpsat_status="present"
  else
    cpsat_status="missing"
  fi
  print "python: ${PYTHON_BIN}"
  print "ortools_cp_sat: ${cpsat_status}"
  print "model: ${MODEL}"
  print "seed/test_count: ${SEED}/${TEST_COUNT}"
  print "structured_dir: ${STRUCTURED_DIR}"
  print "native_dir: ${NATIVE_DIR}"
  print "learned_dir: ${LEARNED_DIR}"
  print "tot_dir: ${TOT_DIR}"
  print "source_audit_dir: ${SOURCE_AUDIT_DIR}"
  print "source_ref_dir: ${SOURCE_REF_DIR}"
  print "implementation_matrix_dir: ${IMPLEMENTATION_MATRIX_DIR}"
  print "run_preflight_dir: ${RUN_PREFLIGHT_DIR}"
  if [[ -z "${CGR_LLM_API_KEY:-}" ]]; then
    print "llm_key: missing"
  else
    print "llm_key: present"
  fi
  implementation_matrix
  source_audit
  source_ref_verify
  "${PYTHON_BIN}" -m cgr_exp.cli build-q2-run-preflight \
    --output-dir "${RUN_PREFLIGHT_DIR}" \
    --seed "${SEED}" \
    --train-count "${TRAIN_COUNT}" \
    --validation-count "${VALIDATION_COUNT}" \
    --test-count "${TEST_COUNT}" \
    --job-count "${JOB_COUNT}" \
    --machine-count "${MACHINE_COUNT}" \
    --node-budget "${NODE_BUDGET}" \
    --action-budget "${ACTION_BUDGET}" \
    --records-per-instance "${RECORDS_PER_INSTANCE}" \
    --rank-epochs "${RANK_EPOCHS}" \
    --model "${MODEL}" \
    --perturbation "${PERTURBATION}" \
    --native-dir "${NATIVE_DIR}" \
    --learned-dir "${LEARNED_DIR}" \
    --tot-dir "${TOT_DIR}" \
    --sample-count "${SAMPLE_COUNT}" \
    --native-parallel-episodes "${NATIVE_PARALLEL_EPISODES}" \
    --search-parallel-episodes "${SEARCH_PARALLEL_EPISODES}" \
    --tot-parallel-episodes "${TOT_PARALLEL_EPISODES}" \
    --batch-size "${BATCH_SIZE}" \
    --target-prm-train-count "${TARGET_PRM_TRAIN_COUNT}" \
    --tot-breadth "${TOT_BREADTH}" \
    --tot-value-samples "${TOT_VALUE_SAMPLES}" \
    --overwrite
  audit_table
}

run_llm_parallel() {
  require_llm_key
  run_native &
  native_pid=$!
  run_learned &
  learned_pid=$!
  run_tot &
  tot_pid=$!
  failed=0
  wait "${native_pid}" || failed=1
  wait "${learned_pid}" || failed=1
  wait "${tot_pid}" || failed=1
  if [[ "${failed}" -ne 0 ]]; then
    print -u2 "one or more Q2 LLM jobs failed"
    exit 1
  fi
}

case "${MODE}" in
  all)
    implementation_matrix
    source_audit
    source_ref_verify
    run_structured
    run_llm_parallel
    build_table
    audit_table
    ;;
  preflight) preflight ;;
  implementation-matrix) implementation_matrix ;;
  source-audit) source_audit ;;
  source-ref-verify) source_ref_verify ;;
  structured) run_structured ;;
  llm) run_llm_parallel ;;
  native) run_native ;;
  learned) run_learned ;;
  tot) run_tot ;;
  build) build_table ;;
  audit) audit_table ;;
  -h|--help|help) usage ;;
  *)
    usage
    exit 2
    ;;
esac
