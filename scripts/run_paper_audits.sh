#!/usr/bin/env zsh
set -euo pipefail

MODE="${1:-all}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
TEST_COUNT="${TEST_COUNT:-100}"

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

prefer_existing_artifact_dir() {
  local artifact="$1"
  shift
  local candidate
  for candidate in "$@"; do
    if [[ -f "${candidate}/${artifact}" ]]; then
      print "${candidate}"
      return
    fi
  done
  print "$1"
}

if [[ -z "${Q2_STRUCTURED_DIR:-}" ]]; then
  Q2_STRUCTURED_DIR="$(prefer_existing_dir runs/q2_structured_100_seed0_runtime_v2 runs/q2_structured_100_seed0)"
fi
if [[ -z "${Q2_NATIVE_DIR:-}" ]]; then
  Q2_NATIVE_DIR="$(prefer_existing_dir runs/q2_native_llm_baselines_100_seed0_sc_v2 runs/q2_native_llm_baselines_100_seed0)"
fi
if [[ -z "${Q2_LEARNED_DIR:-}" ]]; then
  Q2_LEARNED_DIR="$(prefer_existing_artifact_dir scheduling_llm_pilot_table.csv runs/q2_learned_search_100_seed0_rest_gate_v2 runs/scheduling_table5_prm_rest_cert_100_seed0 runs/scheduling_llm_pilot_real_100_seed0)"
fi
if [[ -z "${Q2_TOT_DIR:-}" ]]; then
  Q2_TOT_DIR="$(prefer_existing_dir runs/q2_tot_bfs_100_seed0_tot_gate_v2 runs/scheduling_missing_gpt_frontier_rank_100_seed0)"
fi
Q2_TABLE_AUDIT_DIR="${Q2_TABLE_AUDIT_DIR:-runs/q2_table_current_seed0_protocol_v2_audit}"
Q2_SOURCE_AUDIT_DIR="${Q2_SOURCE_AUDIT_DIR:-runs/q2_source_evidence_audit}"
Q2_SOURCE_REF_DIR="${Q2_SOURCE_REF_DIR:-runs/q2_source_ref_verification}"
Q2_IMPLEMENTATION_MATRIX_DIR="${Q2_IMPLEMENTATION_MATRIX_DIR:-runs/q2_implementation_matrix}"
Q2_LOCAL_PRIMITIVE_AUDIT_DIR="${Q2_LOCAL_PRIMITIVE_AUDIT_DIR:-runs/q2_local_primitive_audit}"
Q2_REMAINING_RUNS_DIR="${Q2_REMAINING_RUNS_DIR:-runs/q2_remaining_runs}"
Q2_RUN_PREFLIGHT_DIR="${Q2_RUN_PREFLIGHT_DIR:-runs/q2_run_preflight}"

Q4_TABLE_AUDIT_DIR="${Q4_TABLE_AUDIT_DIR:-runs/q4_tau_table_current_audit}"
Q4_SOURCE_AUDIT_DIR="${Q4_SOURCE_AUDIT_DIR:-runs/q4_tau_source_evidence_audit}"
Q4_SOURCE_REF_DIR="${Q4_SOURCE_REF_DIR:-runs/q4_tau_source_ref_verification}"
Q4_ADAPTER_CONTRACT_DIR="${Q4_ADAPTER_CONTRACT_DIR:-runs/q4_tau_adapter_contract}"
Q4_NATIVE_DIR="${Q4_NATIVE_DIR:-runs/q4_tau_native}"
Q4_BEST_OF_N_DIR="${Q4_BEST_OF_N_DIR:-runs/q4_tau_best_of_n}"
Q4_VALUE_GUIDED_DIR="${Q4_VALUE_GUIDED_DIR:-runs/q4_tau_value_guided}"
Q4_TARGET_PRM_DIR="${Q4_TARGET_PRM_DIR:-runs/q4_tau_target_prm}"
Q4_SOLVER_CONTROLLER_DIR="${Q4_SOLVER_CONTROLLER_DIR:-runs/q4_tau_solver_controller}"

TAU3_TABLE_BASE_DIR="${TAU3_TABLE_BASE_DIR:-runs/table10_tau3_canonical}"
TAU3_TABLE_AUDIT_DIR="${TAU3_TABLE_AUDIT_DIR:-${TAU3_TABLE_BASE_DIR}/table10_readiness}"
TAU3_RUN_PREFLIGHT_DIR="${TAU3_RUN_PREFLIGHT_DIR:-${TAU3_TABLE_BASE_DIR}/run_preflight}"
TAU3_NATIVE_RETAIL_DIR="${TAU3_NATIVE_RETAIL_DIR:-${TAU3_TABLE_BASE_DIR}/native_retail}"
TAU3_NATIVE_AIRLINE_DIR="${TAU3_NATIVE_AIRLINE_DIR:-${TAU3_TABLE_BASE_DIR}/native_airline}"
TAU3_BEST_OF_N_RETAIL_DIR="${TAU3_BEST_OF_N_RETAIL_DIR:-${TAU3_TABLE_BASE_DIR}/best_of_n_retail}"
TAU3_BEST_OF_N_AIRLINE_DIR="${TAU3_BEST_OF_N_AIRLINE_DIR:-${TAU3_TABLE_BASE_DIR}/best_of_n_airline}"
TAU3_SOLVER_RETAIL_DIR="${TAU3_SOLVER_RETAIL_DIR:-${TAU3_TABLE_BASE_DIR}/solver_canonical_retail}"
TAU3_SOLVER_AIRLINE_DIR="${TAU3_SOLVER_AIRLINE_DIR:-${TAU3_TABLE_BASE_DIR}/solver_canonical_airline}"
TAU3_SOLVER_CERT_RETAIL_DIR="${TAU3_SOLVER_CERT_RETAIL_DIR:-${TAU3_TABLE_BASE_DIR}/solver_canonical_certified_retail}"
TAU3_SOLVER_CERT_AIRLINE_DIR="${TAU3_SOLVER_CERT_AIRLINE_DIR:-${TAU3_TABLE_BASE_DIR}/solver_canonical_certified_airline}"
TAU3_BEST_OF_N="${TAU3_BEST_OF_N:-4}"
TAU_PYTHON_BIN="${TAU_PYTHON_BIN:-${PYTHON_BIN}}"
AGENT_LLM="${AGENT_LLM:-openai/gpt-4o-mini}"
USER_LLM="${USER_LLM:-openai/gpt-4o-mini}"
NUM_TRIALS="${NUM_TRIALS:-4}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-6}"
TIMEOUT="${TIMEOUT:-300}"

CERTIFICATION_AUDIT_DIR="${CERTIFICATION_AUDIT_DIR:-runs/certification_tables_readiness}"
CERTIFICATION_AUTHORITY_SOURCE_AUDIT_DIR="${CERTIFICATION_AUTHORITY_SOURCE_AUDIT_DIR:-runs/certification_authority_source_audit}"
CERTIFICATION_DIR="${CERTIFICATION_DIR:-runs/q2_structured_cert_reference_100_seed0}"
AUTHORITY_DIR="${AUTHORITY_DIR:-runs/table7_grounding_robustness_100_seed0}"
STATUS_REPORT_DIR="${STATUS_REPORT_DIR:-runs/current_experiment_status}"
SECRET_AUDIT_DIR="${SECRET_AUDIT_DIR:-runs/secret_safety_audit}"

usage() {
  cat <<'EOF'
Usage:
  scripts/run_paper_audits.sh [all|q2|q4|tau3|certification|status]

This script runs paper-level promotion audits only. It does not call the LLM
API, set API keys, or generate new experiment trajectories.
EOF
}

audit_q2() {
  "${PYTHON_BIN}" -m cgr_exp.cli build-q2-implementation-matrix \
    --output-dir "${Q2_IMPLEMENTATION_MATRIX_DIR}" \
    --overwrite
  "${PYTHON_BIN}" -m cgr_exp.cli audit-q2-source-evidence \
    --output-dir "${Q2_SOURCE_AUDIT_DIR}" \
    --overwrite
  "${PYTHON_BIN}" -m cgr_exp.cli verify-q2-source-refs \
    --output-dir "${Q2_SOURCE_REF_DIR}" \
    --overwrite
  "${PYTHON_BIN}" -m cgr_exp.cli audit-q2-local-primitives \
    --output-dir "${Q2_LOCAL_PRIMITIVE_AUDIT_DIR}" \
    --overwrite
  if [[ -f "${Q2_LEARNED_DIR}/scheduling_llm_pilot_metrics.json" ]]; then
    "${PYTHON_BIN}" -m cgr_exp.cli backfill-scheduling-llm-pilot-audit-artifacts \
      --output-dir "${Q2_LEARNED_DIR}" \
      --overwrite
  fi
  "${PYTHON_BIN}" -m cgr_exp.cli audit-table4-q2 \
    --output-dir "${Q2_TABLE_AUDIT_DIR}" \
    --expected-test-count "${TEST_COUNT}" \
    --structured-dir "${Q2_STRUCTURED_DIR}" \
    --direct-dir "${Q2_NATIVE_DIR}" \
    --self-consistency-dir "${Q2_NATIVE_DIR}" \
    --best-of-n-dir "${Q2_NATIVE_DIR}" \
    --tot-dir "${Q2_TOT_DIR}" \
    --target-prm-dir "${Q2_LEARNED_DIR}" \
    --rest-mcts-dir "${Q2_LEARNED_DIR}" \
    --solver-prior-dir "${Q2_LEARNED_DIR}" \
    --solver-certification-dir "${Q2_LEARNED_DIR}" \
    --q2-local-primitives "${Q2_LOCAL_PRIMITIVE_AUDIT_DIR}/q2_local_primitive_audit.csv" \
    --overwrite
}

audit_q4() {
  "${PYTHON_BIN}" -m cgr_exp.cli audit-q4-tau-source-evidence \
    --output-dir "${Q4_SOURCE_AUDIT_DIR}" \
    --overwrite
  "${PYTHON_BIN}" -m cgr_exp.cli verify-q4-tau-source-refs \
    --output-dir "${Q4_SOURCE_REF_DIR}" \
    --overwrite
  "${PYTHON_BIN}" -m cgr_exp.cli build-q4-tau-adapter-contract \
    --output-dir "${Q4_ADAPTER_CONTRACT_DIR}" \
    --expected-test-count "${TEST_COUNT}" \
    --overwrite
  "${PYTHON_BIN}" -m cgr_exp.cli audit-table-q4-tau \
    --output-dir "${Q4_TABLE_AUDIT_DIR}" \
    --expected-test-count "${TEST_COUNT}" \
    --native-dir "${Q4_NATIVE_DIR}" \
    --best-of-n-dir "${Q4_BEST_OF_N_DIR}" \
    --value-guided-dir "${Q4_VALUE_GUIDED_DIR}" \
    --target-prm-dir "${Q4_TARGET_PRM_DIR}" \
    --solver-controller-dir "${Q4_SOLVER_CONTROLLER_DIR}" \
    --overwrite
}

audit_certification() {
  "${PYTHON_BIN}" -m cgr_exp.cli audit-certification-authority-source \
    --output-dir "${CERTIFICATION_AUTHORITY_SOURCE_AUDIT_DIR}" \
    --overwrite
  "${PYTHON_BIN}" -m cgr_exp.cli audit-certification-tables \
    --output-dir "${CERTIFICATION_AUDIT_DIR}" \
    --certification-dir "${CERTIFICATION_DIR}" \
    --authority-dir "${AUTHORITY_DIR}" \
    --expected-test-count "${TEST_COUNT}" \
    --authority-source-audit "${CERTIFICATION_AUTHORITY_SOURCE_AUDIT_DIR}/certification_authority_source_audit.csv" \
    --overwrite
}

print_summary() {
  print "python: ${PYTHON_BIN}"
  print "test_count: ${TEST_COUNT}"
  print "q2_audit_dir: ${Q2_TABLE_AUDIT_DIR}"
  print "q4_audit_dir: ${Q4_TABLE_AUDIT_DIR}"
  print "certification_audit_dir: ${CERTIFICATION_AUDIT_DIR}"
  print "certification_authority_source_audit_dir: ${CERTIFICATION_AUTHORITY_SOURCE_AUDIT_DIR}"
  print "status_report_dir: ${STATUS_REPORT_DIR}"
  print "q2_local_primitive_audit_dir: ${Q2_LOCAL_PRIMITIVE_AUDIT_DIR}"
  print "q2_remaining_runs_dir: ${Q2_REMAINING_RUNS_DIR}"
  print "q2_run_preflight_dir: ${Q2_RUN_PREFLIGHT_DIR}"
  print "q4_adapter_contract_dir: ${Q4_ADAPTER_CONTRACT_DIR}"
  print "tau3_table_base_dir: ${TAU3_TABLE_BASE_DIR}"
  print "tau3_table_audit_dir: ${TAU3_TABLE_AUDIT_DIR}"
  print "tau3_run_preflight_dir: ${TAU3_RUN_PREFLIGHT_DIR}"
  print "secret_audit_dir: ${SECRET_AUDIT_DIR}"
  if [[ -z "${CGR_LLM_API_KEY:-}" ]]; then
    print "llm_key: not required"
  else
    print "llm_key: present but unused"
  fi
}

build_status_report() {
  "${PYTHON_BIN}" -m cgr_exp.cli build-current-experiment-status \
    --output-dir "${STATUS_REPORT_DIR}" \
    --q2-readiness "${Q2_TABLE_AUDIT_DIR}/table5_q2_readiness.csv" \
    --q4-readiness "${Q4_TABLE_AUDIT_DIR}/q4_tau_readiness.csv" \
    --tau3-readiness "${TAU3_TABLE_AUDIT_DIR}/table10_tau3_readiness.csv" \
    --certification-readiness "${CERTIFICATION_AUDIT_DIR}/certification_tables_readiness.csv" \
    --q2-implementation-matrix "${Q2_IMPLEMENTATION_MATRIX_DIR}/q2_implementation_matrix.csv" \
    --q2-local-primitives "${Q2_LOCAL_PRIMITIVE_AUDIT_DIR}/q2_local_primitive_audit.csv" \
    --certification-authority-source "${CERTIFICATION_AUTHORITY_SOURCE_AUDIT_DIR}/certification_authority_source_audit.csv" \
    --audit-command "PYTHON_BIN=${PYTHON_BIN} scripts/run_paper_audits.sh all" \
    --overwrite
}

build_tau3_run_preflight() {
  "${PYTHON_BIN}" -m cgr_exp.cli build-table10-tau3-run-preflight \
    --output-dir "${TAU3_RUN_PREFLIGHT_DIR}" \
    --base-dir "${TAU3_TABLE_BASE_DIR}" \
    --expected-test-count "${TEST_COUNT}" \
    --best-of-n "${TAU3_BEST_OF_N}" \
    --tau-python-bin "${TAU_PYTHON_BIN}" \
    --agent-llm "${AGENT_LLM}" \
    --user-llm "${USER_LLM}" \
    --num-trials "${NUM_TRIALS}" \
    --max-concurrency "${MAX_CONCURRENCY}" \
    --overwrite
}

audit_tau3_table10() {
  "${PYTHON_BIN}" -m cgr_exp.cli audit-table10-tau3 \
    --output-dir "${TAU3_TABLE_AUDIT_DIR}" \
    --expected-test-count "${TEST_COUNT}" \
    --row-dirs \
      "${TAU3_NATIVE_RETAIL_DIR}" \
      "${TAU3_NATIVE_AIRLINE_DIR}" \
      "${TAU3_BEST_OF_N_RETAIL_DIR}" \
      "${TAU3_BEST_OF_N_AIRLINE_DIR}" \
      "${TAU3_SOLVER_RETAIL_DIR}" \
      "${TAU3_SOLVER_AIRLINE_DIR}" \
      "${TAU3_SOLVER_CERT_RETAIL_DIR}" \
      "${TAU3_SOLVER_CERT_AIRLINE_DIR}" \
    --overwrite
}

audit_secret_safety() {
  "${PYTHON_BIN}" -m cgr_exp.cli audit-secret-safety \
    --output-dir "${SECRET_AUDIT_DIR}" \
    --root . \
    --overwrite
}

build_q2_remaining_runs() {
  "${PYTHON_BIN}" -m cgr_exp.cli build-q2-remaining-runs \
    --output-dir "${Q2_REMAINING_RUNS_DIR}" \
    --q2-readiness "${Q2_TABLE_AUDIT_DIR}/table5_q2_readiness.csv" \
    --certification-readiness "${CERTIFICATION_AUDIT_DIR}/certification_tables_readiness.csv" \
    --overwrite
}

build_q2_run_preflight() {
  "${PYTHON_BIN}" -m cgr_exp.cli build-q2-run-preflight \
    --output-dir "${Q2_RUN_PREFLIGHT_DIR}" \
    --q2-readiness "${Q2_TABLE_AUDIT_DIR}/table5_q2_readiness.csv" \
    --test-count "${TEST_COUNT}" \
    --native-dir "${Q2_NATIVE_DIR}" \
    --learned-dir "${Q2_LEARNED_DIR}" \
    --tot-dir "${Q2_TOT_DIR}" \
    --overwrite
}

case "${MODE}" in
  all)
    print_summary
    audit_q2
    audit_q4
    audit_certification
    audit_secret_safety
    build_q2_remaining_runs
    build_q2_run_preflight
    build_tau3_run_preflight
    audit_tau3_table10
    build_status_report
    ;;
  q2)
    print_summary
    audit_q2
    ;;
  q4)
    print_summary
    audit_q4
    ;;
  tau3)
    print_summary
    build_tau3_run_preflight
    audit_tau3_table10
    ;;
  certification)
    print_summary
    audit_certification
    ;;
  status)
    print_summary
    audit_secret_safety
    build_q2_remaining_runs
    build_q2_run_preflight
    build_tau3_run_preflight
    audit_tau3_table10
    build_status_report
    ;;
  -h|--help|help) usage ;;
  *)
    usage
    exit 2
    ;;
esac
