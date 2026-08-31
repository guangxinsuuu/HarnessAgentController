#!/usr/bin/env zsh
set -euo pipefail

MODE="${1:-smoke}"
PYTHON_BIN="${PYTHON_BIN:-/opt/anaconda3/envs/cgr-alfworld/bin/python}"
PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
SEED="${SEED:-0}"
OUT_ROOT="${OUT_ROOT:-runs/alfworld_final_seed${SEED}}"

case "${MODE}" in
  smoke|symbolic-smoke|llm-mock|all-mock)
    TRAIN_EPISODES="${TRAIN_EPISODES:-1}"
    AGGREGATION_ROUNDS="${AGGREGATION_ROUNDS:-0}"
    AGGREGATION_EPISODES="${AGGREGATION_EPISODES:-0}"
    EVAL_EPISODES="${EVAL_EPISODES:-1}"
    LLM_EPISODES="${LLM_EPISODES:-1}"
    MAX_STEPS="${MAX_STEPS:-8}"
    ;;
  *)
    TRAIN_EPISODES="${TRAIN_EPISODES:-20}"
    AGGREGATION_ROUNDS="${AGGREGATION_ROUNDS:-1}"
    AGGREGATION_EPISODES="${AGGREGATION_EPISODES:-20}"
    EVAL_EPISODES="${EVAL_EPISODES:-10}"
    LLM_EPISODES="${LLM_EPISODES:-20}"
    MAX_STEPS="${MAX_STEPS:-40}"
    ;;
esac

EPOCHS="${EPOCHS:-50}"
RANKER_MODEL="${RANKER_MODEL:-mlp}"
RANK_HIDDEN_DIM="${RANK_HIDDEN_DIM:-32}"
RANK_LR="${RANK_LR:-0.01}"
MAX_FRONTIER_ACTIONS="${MAX_FRONTIER_ACTIONS:-6}"
MAX_PAIRS_PER_STATE="${MAX_PAIRS_PER_STATE:-16}"
FRONTIER_STRATEGY="${FRONTIER_STRATEGY:-score-diverse}"
PLANNER_TIMEOUT_SECONDS="${PLANNER_TIMEOUT_SECONDS:-3}"
MAX_EPISODE_LABEL_SECONDS="${MAX_EPISODE_LABEL_SECONDS:-60}"
COUNTERFACTUAL_ROLLOUT_STEPS="${COUNTERFACTUAL_ROLLOUT_STEPS:-30}"
LABEL_SOURCE="${LABEL_SOURCE:-planner}"
EVAL_SPLIT="${EVAL_SPLIT:-eval_in_distribution}"

DENSE_TRAIN_DIR="${DENSE_TRAIN_DIR:-${OUT_ROOT}_dense_train}"
DENSE_AUDIT_DIR="${DENSE_AUDIT_DIR:-${OUT_ROOT}_dense_audit}"
DENSE_EVAL_DIR="${DENSE_EVAL_DIR:-${OUT_ROOT}_dense_eval}"
RANKER_CHECKPOINT="${RANKER_CHECKPOINT:-${DENSE_TRAIN_DIR}/alfworld_dense_ranker.json}"
FINAL_SUMMARY_DIR="${FINAL_SUMMARY_DIR:-${OUT_ROOT}_summary}"

SHORTLIST_K="${SHORTLIST_K:-4}"
LLM_EPISODE_START="${LLM_EPISODE_START:-0}"
MODEL="${MODEL:-${CGR_LLM_MODEL:-gpt-4o-mini}}"
TIMEOUT="${TIMEOUT:-120}"
LLM_OUT_DIR="${LLM_OUT_DIR:-${OUT_ROOT}_llm_shortlist}"
LLM_SUMMARY_DIR="${LLM_SUMMARY_DIR:-${LLM_OUT_DIR}_diagnosis}"
SHORTLIST_CACHE_PATH="${SHORTLIST_CACHE_PATH:-${LLM_OUT_DIR}/alfworld_llm_shortlist_cache.jsonl}"
LLM_POLICIES="${LLM_POLICIES:-Frozen LLM top-1,Frozen LLM shortlist + Random,Frozen LLM shortlist + Dense-ranker,Frozen LLM shortlist + Planner diagnostic}"

export PYTHON_BIN
export PROJECT_ROOT
export SEED
export TRAIN_EPISODES
export AGGREGATION_ROUNDS
export AGGREGATION_EPISODES
export EVAL_EPISODES
export MAX_STEPS
export EPOCHS
export RANKER_MODEL
export RANK_HIDDEN_DIM
export RANK_LR
export MAX_FRONTIER_ACTIONS
export MAX_PAIRS_PER_STATE
export FRONTIER_STRATEGY
export PLANNER_TIMEOUT_SECONDS
export MAX_EPISODE_LABEL_SECONDS
export COUNTERFACTUAL_ROLLOUT_STEPS
export LABEL_SOURCE
export EVAL_SPLIT
export TRAIN_DIR="${DENSE_TRAIN_DIR}"
export AUDIT_DIR="${DENSE_AUDIT_DIR}"
export EVAL_DIR="${DENSE_EVAL_DIR}"
export RANKER_CHECKPOINT

usage() {
  cat <<'EOF'
Usage:
  scripts/run_alfworld_final_pipeline.sh [smoke|symbolic|llm-mock|llm-real|all-real]

Modes:
  smoke       Symbolic dense smoke, then mock LLM-shortlist smoke. No API.
  symbolic    Dense planner-label train, audit, and symbolic eval. No API.
  llm-mock    Mock LLM-shortlist with an existing dense ranker checkpoint. No API.
  llm-real    Real LLM-shortlist with an existing dense ranker checkpoint. Requires CGR_LLM_API_KEY.
  all-real    Symbolic dense pipeline, then real LLM-shortlist. Requires CGR_LLM_API_KEY.

Default non-smoke settings reproduce the current ALFWorld pilot shape:
  TRAIN_EPISODES=20 AGGREGATION_ROUNDS=1 AGGREGATION_EPISODES=20
  EVAL_EPISODES=10 LLM_EPISODES=20 RANKER_MODEL=mlp FRONTIER_STRATEGY=score-diverse
EOF
}

run_symbolic() {
  scripts/run_alfworld_planner_dense.sh all
}

run_llm_shortlist() {
  if [[ ! -f "${RANKER_CHECKPOINT}" ]]; then
    print -u2 "missing ranker checkpoint: ${RANKER_CHECKPOINT}"
    print -u2 "run symbolic first or set RANKER_CHECKPOINT=/path/to/alfworld_dense_ranker.json"
    exit 2
  fi
  local mock_flag="${1}"
  MOCK="${mock_flag}" \
  EPISODES="${LLM_EPISODES}" \
  OUT_DIR="${LLM_OUT_DIR}" \
  SUMMARY_DIR="${LLM_SUMMARY_DIR}" \
  SHORTLIST_CACHE_PATH="${SHORTLIST_CACHE_PATH}" \
  SHORTLIST_K="${SHORTLIST_K}" \
  EPISODE_START="${LLM_EPISODE_START}" \
  MODEL="${MODEL}" \
  TIMEOUT="${TIMEOUT}" \
  POLICIES="${LLM_POLICIES}" \
  RANKER_CHECKPOINT="${RANKER_CHECKPOINT}" \
  RESUME="${RESUME:-1}" \
  OVERWRITE="${OVERWRITE:-0}" \
  SUMMARIZE=1 \
  scripts/run_alfworld_llm_shortlist.sh
}

run_final_summary() {
  local include_llm="${1}"
  local llm_args=()
  if [[ "${include_llm}" == "1" ]]; then
    llm_args=("--llm-shortlist-dir" "${LLM_OUT_DIR}" "--llm-diagnosis-dir" "${LLM_SUMMARY_DIR}")
  fi
  "${PYTHON_BIN}" -m cgr_exp.cli summarize-alfworld-final-pipeline \
    --output-dir "${FINAL_SUMMARY_DIR}" \
    --dense-train-dir "${DENSE_TRAIN_DIR}" \
    --dense-audit-dir "${DENSE_AUDIT_DIR}" \
    --dense-eval-dir "${DENSE_EVAL_DIR}" \
    "${llm_args[@]}" \
    --overwrite
}

case "${MODE}" in
  smoke)
    run_symbolic
    run_llm_shortlist 1
    run_final_summary 1
    ;;
  symbolic|symbolic-smoke)
    run_symbolic
    run_final_summary 0
    ;;
  llm-mock|all-mock)
    run_llm_shortlist 1
    run_final_summary 1
    ;;
  llm-real)
    run_llm_shortlist 0
    run_final_summary 1
    ;;
  all-real)
    run_symbolic
    run_llm_shortlist 0
    run_final_summary 1
    ;;
  help|-h|--help)
    usage
    ;;
  *)
    usage
    exit 2
    ;;
esac
