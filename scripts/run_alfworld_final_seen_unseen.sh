#!/usr/bin/env zsh
set -euo pipefail

MODE="${1:-smoke}"
PYTHON_BIN="${PYTHON_BIN:-/opt/anaconda3/envs/cgr-alfworld/bin/python}"
PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
ALFWORLD_DATA="${ALFWORLD_DATA:-${PROJECT_ROOT}/data/alfworld}"
SEED="${SEED:-0}"
OUT_ROOT="${OUT_ROOT:-runs/alfworld_final_seen_unseen_seed${SEED}}"

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
    EVAL_EPISODES="${EVAL_EPISODES:-20}"
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

DENSE_TRAIN_DIR="${DENSE_TRAIN_DIR:-${OUT_ROOT}_dense_train}"
DENSE_AUDIT_DIR="${DENSE_AUDIT_DIR:-${OUT_ROOT}_dense_audit}"
RANKER_CHECKPOINT="${RANKER_CHECKPOINT:-${DENSE_TRAIN_DIR}/alfworld_dense_ranker.json}"
FINAL_SPLIT_SUMMARY_DIR="${FINAL_SPLIT_SUMMARY_DIR:-${OUT_ROOT}_summary}"
PREFLIGHT_DIR="${PREFLIGHT_DIR:-${OUT_ROOT}_preflight}"

SHORTLIST_K="${SHORTLIST_K:-4}"
LLM_SHARDS="${LLM_SHARDS:-1}"
LLM_SHARD_SIZE="${LLM_SHARD_SIZE:-}"
MODEL="${MODEL:-${CGR_LLM_MODEL:-gpt-4o-mini}}"
TIMEOUT="${TIMEOUT:-120}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-1.0}"
LLM_POLICIES="${LLM_POLICIES:-Frozen LLM top-1,Frozen LLM shortlist + Random,Frozen LLM shortlist + Dense-ranker,Frozen LLM shortlist + Planner diagnostic}"

export PYTHON_BIN
export PROJECT_ROOT
export ALFWORLD_DATA
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
export TRAIN_DIR="${DENSE_TRAIN_DIR}"
export AUDIT_DIR="${DENSE_AUDIT_DIR}"
export RANKER_CHECKPOINT

usage() {
  cat <<'EOF'
Usage:
  scripts/run_alfworld_final_seen_unseen.sh [smoke|symbolic|llm-mock|llm-real|all-real]

Runs the ALFWorld final protocol on both public validation splits:
  valid_seen   -> eval_in_distribution
  valid_unseen -> eval_out_of_distribution

The dense planner-supervised controller is trained once on ALFWorld train
symbolic worlds, then frozen for both split evaluations. Set LLM_SHARDS>1 to
parallelize the LLM-shortlist stage inside each split.
EOF
}

run_train_once() {
  scripts/run_alfworld_planner_dense.sh train
  scripts/run_alfworld_planner_dense.sh audit
}

run_split() {
  local label="${1}"
  local eval_split="${2}"
  local mock_flag="${3}"
  local include_llm="${4}"
  local split_root="${OUT_ROOT}_${label}"
  local dense_eval_dir="${split_root}_dense_eval"
  local final_summary_dir="${split_root}_summary"
  local llm_out_dir="${split_root}_llm_shortlist"
  local llm_summary_dir="${split_root}_llm_shortlist_diagnosis"

  EVAL_SPLIT="${eval_split}" \
  EVAL_DIR="${dense_eval_dir}" \
  RANKER_CHECKPOINT="${RANKER_CHECKPOINT}" \
  scripts/run_alfworld_planner_dense.sh eval

  if [[ "${include_llm}" != "1" ]]; then
    "${PYTHON_BIN}" -m cgr_exp.cli summarize-alfworld-final-pipeline \
      --output-dir "${final_summary_dir}" \
      --dense-train-dir "${DENSE_TRAIN_DIR}" \
      --dense-audit-dir "${DENSE_AUDIT_DIR}" \
      --dense-eval-dir "${dense_eval_dir}" \
      --overwrite
    return
  fi

  local llm_mode="llm-real"
  if [[ "${mock_flag}" == "1" ]]; then
    llm_mode="llm-mock"
  fi
  EVAL_SPLIT="${eval_split}" \
  DENSE_TRAIN_DIR="${DENSE_TRAIN_DIR}" \
  DENSE_AUDIT_DIR="${DENSE_AUDIT_DIR}" \
  DENSE_EVAL_DIR="${dense_eval_dir}" \
  RANKER_CHECKPOINT="${RANKER_CHECKPOINT}" \
  FINAL_SUMMARY_DIR="${final_summary_dir}" \
  LLM_OUT_DIR="${llm_out_dir}" \
  LLM_SUMMARY_DIR="${llm_summary_dir}" \
  LLM_EPISODES="${LLM_EPISODES}" \
  LLM_SHARDS="${LLM_SHARDS}" \
  LLM_SHARD_SIZE="${LLM_SHARD_SIZE}" \
  SHORTLIST_K="${SHORTLIST_K}" \
  MODEL="${MODEL}" \
  TIMEOUT="${TIMEOUT}" \
  TEMPERATURE="${TEMPERATURE}" \
  TOP_P="${TOP_P}" \
  LLM_POLICIES="${LLM_POLICIES}" \
  scripts/run_alfworld_final_pipeline.sh "${llm_mode}"
}

merge_splits() {
  "${PYTHON_BIN}" -m cgr_exp.cli summarize-alfworld-final-splits \
    --output-dir "${FINAL_SPLIT_SUMMARY_DIR}" \
    --split-labels valid_seen valid_unseen \
    --split-summary-dirs "${OUT_ROOT}_valid_seen_summary" "${OUT_ROOT}_valid_unseen_summary" \
    --overwrite
}

run_preflight() {
  "${PYTHON_BIN}" -m cgr_exp.cli preflight-alfworld-final \
    --output-dir "${PREFLIGHT_DIR}" \
    --data-root "${ALFWORLD_DATA}" \
    --ranker-checkpoint "${RANKER_CHECKPOINT}" \
    --mode "${MODE}" \
    --train-episodes "${TRAIN_EPISODES}" \
    --eval-episodes "${EVAL_EPISODES}" \
    --llm-episodes "${LLM_EPISODES}" \
    --max-steps "${MAX_STEPS}" \
    --shortlist-k "${SHORTLIST_K}" \
    --llm-shards "${LLM_SHARDS}" \
    --model "${MODEL}" \
    --overwrite
}

case "${MODE}" in
  smoke)
    run_preflight
    run_train_once
    run_split valid_seen eval_in_distribution 1 1
    run_split valid_unseen eval_out_of_distribution 1 1
    merge_splits
    ;;
  symbolic|symbolic-smoke)
    run_preflight
    run_train_once
    run_split valid_seen eval_in_distribution 1 0
    run_split valid_unseen eval_out_of_distribution 1 0
    merge_splits
    ;;
  llm-mock|all-mock)
    run_preflight
    run_split valid_seen eval_in_distribution 1 1
    run_split valid_unseen eval_out_of_distribution 1 1
    merge_splits
    ;;
  llm-real)
    run_preflight
    run_split valid_seen eval_in_distribution 0 1
    run_split valid_unseen eval_out_of_distribution 0 1
    merge_splits
    ;;
  all-real)
    run_preflight
    run_train_once
    run_split valid_seen eval_in_distribution 0 1
    run_split valid_unseen eval_out_of_distribution 0 1
    merge_splits
    ;;
  help|-h|--help)
    usage
    ;;
  *)
    usage
    exit 2
    ;;
esac
